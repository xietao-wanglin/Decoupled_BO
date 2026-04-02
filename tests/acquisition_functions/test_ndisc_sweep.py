"""Sweep n_disc to find diminishing returns for CoupledCKG and AllSourcesDcKG.

Uses ConstrainedBranin (d=2, K=1) at 10 and 50 training points with
multiple random seeds to reduce variance.
"""

import gc
import time

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ConstrainedMCObjective
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.utils.transforms import standardize
from gpytorch.mlls import SumMarginalLogLikelihood

from bo.acquisition_functions.refactored_acquisition_functions import (
    CoupledCKG, AllSourcesDcKG,
)
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
    X = torch.rand(n_points, d, device=DEVICE, dtype=dtype)
    NOISE = torch.tensor(1e-6, device=DEVICE, dtype=dtype)
    func = ConstrainedBranin()
    Y_obj = func.evaluate_true(X.cpu()).unsqueeze(-1).to(DEVICE)
    Y_con = func.evaluate_slack_true(X.cpu()).to(DEVICE)
    m_obj = SingleTaskGP(X, Y_obj, train_Yvar=NOISE.expand_as(Y_obj))
    m_con = SingleTaskGP(X, Y_con, train_Yvar=NOISE.expand_as(Y_con))
    model = ModelListGP(m_obj, m_con)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=DEVICE, dtype=dtype)
    return model, bounds


def _best_location(model, bounds):
    loc, _ = optimize_acqf(
        ConstrainedPosteriorMean(model, penalty_value=torch.tensor([2.0], device=DEVICE)),
        bounds=bounds, q=1, num_restarts=10, raw_samples=128,
    )
    return loc


def _get_fantasy_cands(model, bounds, eta=2.0):
    fantasy_cands, fantasy_vals = optimize_acqf(
        acq_function=ConstrainedPosteriorMean(
            model=model, penalty_value=torch.tensor([2.0], device=DEVICE),
        ),
        bounds=bounds, q=1, num_restarts=20, raw_samples=512,
        return_best_only=False,
    )
    std = fantasy_vals.std()
    weights = torch.exp(eta * standardize(fantasy_vals)) if std > 0 else torch.ones_like(fantasy_vals)
    return fantasy_cands, weights


def _reset_gpu():
    if DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _gpu_mb():
    return torch.cuda.max_memory_allocated() / 1024**2 if DEVICE.type == "cuda" else 0.0


def _sweep_ckg(model, bounds, x_best, objective, fantasy_cands, weights,
               ndisc_values, num_restarts=10, raw_samples=64, n_seeds=3):
    results = []
    for n_disc in ndisc_values:
        Q = 1 + n_disc
        vals_across_seeds = []
        total_time = 0
        peak_mem = 0

        for seed in range(n_seeds):
            ckg = CoupledCKG(
                model=model, penalty_value=torch.tensor([2.0], device=DEVICE),
                x_best_location=x_best, objective=objective,
                n_fantasies=7, n_constraint_samples=5, seed=seed,
            )

            ics = gen_batch_initial_conditions(
                acq_function=ckg, bounds=bounds, q=Q,
                num_restarts=num_restarts, raw_samples=raw_samples,
                options={"seed": seed, "eta": 2.0},
            )
            ics[:, 1:2, :] = ckg.x_best
            n_value = int(0.9 * (n_disc - 1)) if n_disc > 1 else 0
            if n_value > 0:
                idx = torch.multinomial(weights, num_restarts * n_value, replacement=True)
                ics[..., -n_value:, :] = fantasy_cands[idx, 0].view(num_restarts, n_value, -1)
            xbest_ic = ics[-1:].clone()
            xbest_ic[:, 0:1, :] = ckg.x_best
            ics = torch.cat([ics, xbest_ic], dim=0)

            _reset_gpu()
            t0 = time.perf_counter()
            _, val = optimize_acqf(
                acq_function=ckg, bounds=bounds, q=Q,
                num_restarts=ics.shape[0],
                batch_initial_conditions=ics,
                options={"maxiter": 100},
            )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            peak_mem = max(peak_mem, _gpu_mb())
            vals_across_seeds.append(val.item())
            del ckg

        mean_val = sum(vals_across_seeds) / len(vals_across_seeds)
        mean_time = total_time / n_seeds
        results.append((n_disc, mean_val, mean_time, peak_mem))
    return results


def _sweep_all_sources(model, bounds, x_best, objective, fantasy_cands, weights,
                        ndisc_values, num_restarts=10, raw_samples=64, n_seeds=3):
    results = []
    for n_disc in ndisc_values:
        vals_across_seeds = []
        total_time = 0
        peak_mem = 0

        for seed in range(n_seeds):
            all_kg = AllSourcesDcKG(
                model=model, penalty_value=torch.tensor([2.0], device=DEVICE),
                x_best_location=x_best, objective=objective,
                n_fantasies=7, n_constraint_samples=5, n_disc=n_disc, seed=seed,
            )
            S = all_kg.n_sources
            Q = all_kg.q_per_source
            q_total = S * Q
            n_value = int(0.9 * (n_disc - 1)) if n_disc > 1 else 0

            per_source_ics = []
            for s, source_acqf in enumerate(all_kg.sources):
                ics_s = gen_batch_initial_conditions(
                    acq_function=source_acqf, bounds=bounds, q=Q,
                    num_restarts=num_restarts, raw_samples=raw_samples,
                    options={"seed": seed * 100 + s, "eta": 2.0},
                )
                ics_s[:, 1:2, :] = all_kg.x_best
                if n_value > 0:
                    idx = torch.multinomial(weights, num_restarts * n_value, replacement=True)
                    ics_s[..., -n_value:, :] = fantasy_cands[idx, 0].view(num_restarts, n_value, -1)
                per_source_ics.append(ics_s)

            ics = torch.cat(per_source_ics, dim=1)
            with torch.no_grad():
                joint_vals = all_kg(ics)
            k = min(num_restarts, len(joint_vals))
            _, top = torch.topk(joint_vals, k)
            ics = ics[top]
            xbest_ic = ics[-1:].clone()
            for s in range(S):
                xbest_ic[0, s * Q, :] = all_kg.x_best.squeeze(0)
            ics = torch.cat([ics, xbest_ic], dim=0)

            _reset_gpu()
            t0 = time.perf_counter()
            _, val = optimize_acqf(
                acq_function=all_kg, bounds=bounds, q=q_total,
                num_restarts=ics.shape[0],
                batch_initial_conditions=ics,
                options={"maxiter": 100},
            )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            peak_mem = max(peak_mem, _gpu_mb())
            vals_across_seeds.append(val.item())
            del all_kg

        mean_val = sum(vals_across_seeds) / len(vals_across_seeds)
        mean_time = total_time / n_seeds
        results.append((n_disc, mean_val, mean_time, peak_mem))
    return results


def _print_results(label, results, n_sources=None):
    best_val = max(r[1] for r in results)
    print(f"\n  {label}:")
    q_label = "q_tot" if n_sources else "q"
    print(f"  {'n_disc':>6s}  {q_label:>5s}  {'mean value':>12s}  {'% of best':>10s}  "
          f"{'avg time':>8s}  {'GPU MB':>7s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*7}")
    for n_disc, val, t, mem in results:
        pct = val / best_val * 100 if best_val != 0 else 0
        q_str = str(n_sources * (1 + n_disc)) if n_sources else str(1 + n_disc)
        print(f"  {n_disc:>6d}  {q_str:>5s}  {val:>12.4f}  {pct:>9.1f}%  {t:>7.2f}s  {mem:>6.0f}")


def run():
    objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
    ndisc_values = [4, 8, 16, 32, 64, 128]

    print("\n" + "#" * 75)
    print("#  n_disc SWEEP: mean acqf value vs discretisation size")
    print(f"#  Device: {DEVICE}")
    print(f"#  Problem: ConstrainedBranin (d=2, K=1)")
    print(f"#  Averaged over 3 random seeds")
    print("#" * 75)

    for n_points in [10, 50]:
        print(f"\n{'=' * 75}")
        print(f"  Training points: {n_points}")
        print(f"{'=' * 75}")

        model, bounds = _build_model(n_points)
        x_best = _best_location(model, bounds)
        fantasy_cands, weights = _get_fantasy_cands(model, bounds)

        results_ckg = _sweep_ckg(model, bounds, x_best, objective,
                                  fantasy_cands, weights, ndisc_values)
        _print_results("CoupledCKG (FastConstrainedKG)", results_ckg)

        results_all = _sweep_all_sources(model, bounds, x_best, objective,
                                          fantasy_cands, weights, ndisc_values)
        _print_results("AllSourcesDcKG (3 sources)", results_all, n_sources=3)

    print(f"\n{'=' * 75}")
    print("  DONE")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    run()
