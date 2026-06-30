"""Memory and time benchmark for CoupledCKG and AllSourcesDcKG on the
CNN CIFAR-10 problem (d=5, K=10, 11 outputs).

Simulates conditions at budget=160 by building GP models with various
numbers of training points and measuring forward pass time, full
optimize_acqf time, and peak GPU memory.
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
from bo.model.Model import ConstrainedPosteriorMean
from bo.synthetic_test_functions.cnn_takena22_benchmark import const_cnn_cifar10

dtype = torch.double
torch.set_default_dtype(dtype)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obj_callable(Z, X=None):
    return Z[..., 0]


def _build_cnn_model(func, n_points, seed=42):
    """Build a fitted ModelListGP for the CNN problem with n_points."""
    torch.manual_seed(seed)
    d = 5
    bounds = func._bounds.to(DEVICE)

    # Sample random points in the input space
    X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_points, d, device=DEVICE)

    NOISE = torch.tensor(1e-6, device=DEVICE, dtype=dtype)

    # Evaluate all outputs
    all_Y = func.evaluate_slack_true(X.cpu())  # (n_points, 11)

    models = []
    for m in range(func.M):
        Y_m = all_Y[:, m:m+1].to(DEVICE)
        gp = SingleTaskGP(X, Y_m, train_Yvar=NOISE.expand_as(Y_m))
        models.append(gp)

    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model, bounds


def _best_location(model, bounds):
    penalty = torch.tensor([3.0], device=DEVICE)
    loc, _ = optimize_acqf(
        ConstrainedPosteriorMean(model, penalty_value=penalty),
        bounds=bounds, q=1, num_restarts=10, raw_samples=128,
    )
    return loc


def _reset_gpu_memory():
    if DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _get_gpu_memory_mb():
    if DEVICE.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def _time_forward(acqf, test_X, warmup=1, repeats=3):
    for _ in range(warmup):
        acqf.forward(test_X.clone())
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        _reset_gpu_memory()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = acqf.forward(test_X.clone())
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2], _get_gpu_memory_mb()


def _time_optimize(acqf, ics, bounds, q, maxiter=100):
    _reset_gpu_memory()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    candidates, val = optimize_acqf(
        acq_function=acqf, bounds=bounds, q=q,
        num_restarts=ics.shape[0],
        batch_initial_conditions=ics,
        options={"maxiter": maxiter},
    )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter() - t0
    return t, val.item(), _get_gpu_memory_mb()


def _build_ckg_ics(acqf, bounds, n_disc=16, num_restarts=5, raw_samples=32):
    Q = 1 + n_disc
    eta = 2.0
    fantasy_cands, fantasy_vals = optimize_acqf(
        acq_function=ConstrainedPosteriorMean(
            model=acqf.model, penalty_value=acqf.penalty_value,
        ),
        bounds=bounds, q=1, num_restarts=20, raw_samples=512,
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
    n_value = int(0.9 * (n_disc - 1))
    if n_value > 0:
        idx = torch.multinomial(weights, num_restarts * n_value, replacement=True)
        ics[..., -n_value:, :] = fantasy_cands[idx, 0].view(num_restarts, n_value, -1)

    xbest_ic = ics[-1:].clone()
    xbest_ic[:, 0:1, :] = acqf.x_best
    return torch.cat([ics, xbest_ic], dim=0), Q


def _build_all_sources_ics(acqf, bounds, num_restarts=5, raw_samples=32):
    S = acqf.n_sources
    Q = acqf.q_per_source
    q_total = S * Q
    n_disc = acqf.n_disc
    eta = 2.0

    fantasy_cands, fantasy_vals = optimize_acqf(
        acq_function=ConstrainedPosteriorMean(
            model=acqf.model, penalty_value=acqf.penalty_value,
        ),
        bounds=bounds, q=1, num_restarts=20, raw_samples=512,
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


def run():
    print("\n" + "#" * 75)
    print("#  MEMORY & TIME BENCHMARK: CNN CIFAR-10 (d=5, K=10, 11 outputs)")
    print(f"#  Device: {DEVICE}")
    print("#" * 75)

    func = const_cnn_cifar10(negate=False)
    objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
    penalty = torch.tensor([3.0], device=DEVICE)
    n_disc = 16
    n_cands = 5  # batch size for forward pass timing

    for n_points in [30, 80, 160]:
        print(f"\n{'=' * 75}")
        print(f"  Training points: {n_points}")
        print(f"{'=' * 75}")

        model, bounds = _build_cnn_model(func, n_points)
        x_best = _best_location(model, bounds)

        # ---- CoupledCKG ----
        print(f"\n  --- CoupledCKG (FastConstrainedKG) ---")
        print(f"  d=5, K=10, n_disc={n_disc}, q={1+n_disc}")

        _reset_gpu_memory()
        t0 = time.perf_counter()
        ckg = CoupledCKG(
            model=model, penalty_value=penalty, x_best_location=x_best,
            objective=objective, n_fantasies=7, n_constraint_samples=5, seed=0,
        )
        t_construct = time.perf_counter() - t0
        mem_construct = _get_gpu_memory_mb()
        print(f"  Construct:  {t_construct:.2f}s, peak GPU: {mem_construct:.0f} MB")

        # Forward pass
        test_X = torch.rand(n_cands, 1 + n_disc, 5, device=DEVICE)
        t_fwd, mem_fwd = _time_forward(ckg, test_X)
        print(f"  Forward ({n_cands} pts): {t_fwd:.3f}s, peak GPU: {mem_fwd:.0f} MB")

        # Full optimize
        try:
            ics_ckg, q_ckg = _build_ckg_ics(ckg, bounds, n_disc=n_disc,
                                             num_restarts=5, raw_samples=32)
            t_opt, val_opt, mem_opt = _time_optimize(ckg, ics_ckg, bounds, q_ckg)
            print(f"  optimize_acqf (5 restarts, maxiter=100): {t_opt:.2f}s, "
                  f"value={val_opt:.4f}, peak GPU: {mem_opt:.0f} MB")
        except Exception as e:
            print(f"  optimize_acqf FAILED: {e}")

        del ckg
        _reset_gpu_memory()

        # ---- AllSourcesDcKG ----
        print(f"\n  --- AllSourcesDcKG (12 sources) ---")
        print(f"  d=5, K=10, n_disc={n_disc}, q_per_source={1+n_disc}, "
              f"total q={12*(1+n_disc)}")

        _reset_gpu_memory()
        t0 = time.perf_counter()
        all_kg = AllSourcesDcKG(
            model=model, penalty_value=penalty, x_best_location=x_best,
            objective=objective, n_fantasies=7, n_constraint_samples=5,
            n_disc=n_disc, seed=0,
        )
        t_construct = time.perf_counter() - t0
        mem_construct = _get_gpu_memory_mb()
        print(f"  Construct:  {t_construct:.2f}s, peak GPU: {mem_construct:.0f} MB")

        # Forward pass
        q_total = all_kg.n_sources * all_kg.q_per_source
        test_X_all = torch.rand(n_cands, q_total, 5, device=DEVICE)
        t_fwd, mem_fwd = _time_forward(all_kg, test_X_all)
        print(f"  Forward ({n_cands} pts): {t_fwd:.3f}s, peak GPU: {mem_fwd:.0f} MB")

        # Full optimize
        try:
            ics_all, q_all = _build_all_sources_ics(all_kg, bounds,
                                                     num_restarts=5,
                                                     raw_samples=32)
            t_opt, val_opt, mem_opt = _time_optimize(all_kg, ics_all, bounds, q_all)
            print(f"  optimize_acqf (5 restarts, maxiter=100): {t_opt:.2f}s, "
                  f"value={val_opt:.4f}, peak GPU: {mem_opt:.0f} MB")
        except Exception as e:
            print(f"  optimize_acqf FAILED: {e}")

        del all_kg
        _reset_gpu_memory()

    print(f"\n{'=' * 75}")
    print("  DONE")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    run()
