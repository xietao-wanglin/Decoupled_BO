"""Convergence study: how many L-BFGS iterations does each acquisition function
need before the value plateaus?

Tests CoupledCKG (FastConstrainedKG) and AllSourcesDcKG (decoupled all-sources)
at 10 and 100 training points, sweeping maxiter from 10 to 200.
"""

import time

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ConstrainedMCObjective
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import SumMarginalLogLikelihood

from bo.acquisition_functions.refactored_acquisition_functions import (
    CoupledCKG, AllSourcesDcKG,
)
from bo.bo_loops.bo_loop import OptimizationLoop
from bo.constrained_functions.synthetic_problems import ConstrainedBranin
from bo.model.Model import ConstrainedPosteriorMean

dtype = torch.double
torch.set_default_dtype(dtype)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obj_callable(Z, X=None):
    return Z[..., 0]


def _build_model(n_points, seed=42):
    torch.manual_seed(seed)
    d = 2
    X = torch.rand(n_points, d, dtype=dtype, device=DEVICE)
    NOISE = torch.tensor(1e-6, dtype=dtype, device=DEVICE)
    func = ConstrainedBranin()
    Y_obj = func.evaluate_true(X.cpu()).unsqueeze(-1).to(DEVICE)
    Y_con = func.evaluate_slack_true(X.cpu()).to(DEVICE)
    m_obj = SingleTaskGP(X, Y_obj, train_Yvar=NOISE.expand_as(Y_obj))
    m_con = SingleTaskGP(X, Y_con, train_Yvar=NOISE.expand_as(Y_con))
    model = ModelListGP(m_obj, m_con)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _best_location(model):
    d = 2
    bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=dtype, device=DEVICE)
    loc, _ = optimize_acqf(
        ConstrainedPosteriorMean(model, penalty_value=torch.tensor([0.0], device=DEVICE)),
        bounds=bounds, q=1, num_restarts=10, raw_samples=128,
    )
    return loc


def _build_ckg_ics(acqf, bounds, num_restarts=5, raw_samples=32):
    """Build initial conditions for CoupledCKG (same as _optimize_fast_ckg)."""
    from botorch.optim.initializers import gen_batch_initial_conditions
    from botorch.utils.transforms import standardize

    Q = 1 + 16
    eta = 2.0

    fantasy_cands, fantasy_vals = optimize_acqf(
        acq_function=ConstrainedPosteriorMean(
            model=acqf.model, penalty_value=acqf.penalty_value,
        ),
        bounds=bounds, q=1, num_restarts=20, raw_samples=1024,
        return_best_only=False,
    )
    std = fantasy_vals.std()
    weights = torch.exp(eta * standardize(fantasy_vals)) if std > 0 else torch.ones_like(fantasy_vals)

    ics = gen_batch_initial_conditions(
        acq_function=acqf, bounds=bounds, q=Q,
        num_restarts=num_restarts, raw_samples=raw_samples,
        options={"seed": 0, "eta": eta},
    )
    ics[:, 1:2, :] = acqf.x_best
    n_value = int(0.9 * 15)
    if n_value > 0:
        idx = torch.multinomial(weights, num_restarts * n_value, replacement=True)
        ics[..., -n_value:, :] = fantasy_cands[idx, 0].view(num_restarts, n_value, -1)

    xbest_ic = ics[-1:].clone()
    xbest_ic[:, 0:1, :] = acqf.x_best
    return torch.cat([ics, xbest_ic], dim=0), Q


def _build_all_sources_ics(acqf, bounds, num_restarts=5, raw_samples=32):
    """Build initial conditions for AllSourcesDcKG (per-source init)."""
    from botorch.optim.initializers import gen_batch_initial_conditions
    from botorch.utils.transforms import standardize

    S = acqf.n_sources
    Q = acqf.q_per_source
    q_total = S * Q
    n_disc = acqf.n_disc
    eta = 2.0

    fantasy_cands, fantasy_vals = optimize_acqf(
        acq_function=ConstrainedPosteriorMean(
            model=acqf.model, penalty_value=acqf.penalty_value,
        ),
        bounds=bounds, q=1, num_restarts=20, raw_samples=1024,
        return_best_only=False,
    )
    std = fantasy_vals.std()
    weights = torch.exp(eta * standardize(fantasy_vals)) if std > 0 else torch.ones_like(fantasy_vals)
    n_value = int(0.9 * (n_disc - 1))

    per_source_ics = []
    for s, source_acqf in enumerate(acqf.sources):
        ics_s = gen_batch_initial_conditions(
            acq_function=source_acqf, bounds=bounds, q=Q,
            num_restarts=num_restarts, raw_samples=raw_samples,
            options={"seed": s, "eta": eta},
        )
        ics_s[:, 1:2, :] = acqf.x_best
        if n_value > 0:
            idx = torch.multinomial(weights, num_restarts * n_value, replacement=True)
            ics_s[..., -n_value:, :] = fantasy_cands[idx, 0].view(num_restarts, n_value, -1)
        per_source_ics.append(ics_s)

    ics = torch.cat(per_source_ics, dim=1)
    with torch.no_grad():
        joint_vals = acqf(ics)
    k = min(num_restarts, len(joint_vals))
    _, top = torch.topk(joint_vals, k)
    ics = ics[top]

    xbest_ic = ics[-1:].clone()
    for s in range(S):
        xbest_ic[0, s * Q, :] = acqf.x_best.squeeze(0)
    ics = torch.cat([ics, xbest_ic], dim=0)

    return ics, q_total


def _print_results(label, results):
    best_val = max(r[1] for r in results)
    print(f"\n  {label}:")
    print(f"  {'maxiter':>8s}  {'value':>12s}  {'% of best':>10s}  {'time':>8s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*8}")
    for maxiter, val, t in results:
        pct = val / best_val * 100 if best_val != 0 else 0
        print(f"  {maxiter:>8d}  {val:>12.4f}  {pct:>9.1f}%  {t:>7.2f}s")


def run():
    objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dtype, device=DEVICE)
    maxiter_values = [10, 25, 50, 75, 100, 150, 200]

    print("\n" + "#" * 70)
    print("#  CONVERGENCE STUDY: acqf value vs maxiter")
    print(f"#  Device: {DEVICE}")
    print("#" * 70)

    for n_points in [10, 100]:
        print(f"\n{'=' * 70}")
        print(f"  Training points: {n_points}")
        print(f"{'=' * 70}")

        model = _build_model(n_points)
        x_best = _best_location(model)
        penalty = torch.tensor([2.0], device=DEVICE)

        # --- CoupledCKG (FastConstrainedKG) ---
        ckg = CoupledCKG(
            model=model, penalty_value=penalty, x_best_location=x_best,
            objective=objective, n_fantasies=7, n_constraint_samples=5, seed=0,
        )
        ics_ckg, q_ckg = _build_ckg_ics(ckg, bounds)

        results_ckg = []
        for maxiter in maxiter_values:
            t0 = time.perf_counter()
            _, val = optimize_acqf(
                acq_function=ckg, bounds=bounds, q=q_ckg,
                num_restarts=ics_ckg.shape[0],
                batch_initial_conditions=ics_ckg.clone(),
                options={"maxiter": maxiter},
            )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter() - t0
            results_ckg.append((maxiter, val.item(), t))

        _print_results("CoupledCKG (FastConstrainedKG)", results_ckg)

        # --- AllSourcesDcKG ---
        all_kg = AllSourcesDcKG(
            model=model, penalty_value=penalty, x_best_location=x_best,
            objective=objective, n_fantasies=7, n_constraint_samples=5,
            n_disc=16, seed=0,
        )
        ics_all, q_all = _build_all_sources_ics(all_kg, bounds)

        results_all = []
        for maxiter in maxiter_values:
            t0 = time.perf_counter()
            _, val = optimize_acqf(
                acq_function=all_kg, bounds=bounds, q=q_all,
                num_restarts=ics_all.shape[0],
                batch_initial_conditions=ics_all.clone(),
                options={"maxiter": maxiter},
            )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter() - t0
            results_all.append((maxiter, val.item(), t))

        _print_results("AllSourcesDcKG (decoupled all-sources)", results_all)

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
