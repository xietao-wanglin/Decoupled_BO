"""Timing comparison: Old vs Fast cKG and dcKG.

Compares forward pass time at 10 and 100 training points for:
  1. Coupled cKG: old (DecopledHybridConstrainedKnowledgeGradient) vs FastConstrainedKG
  2. Decoupled dcKG (source 0): old vs FastConstrainedKG
  3. Decoupled dcKG (source 1): old vs FastConstrainedKG
"""

import time

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ConstrainedMCObjective
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import SumMarginalLogLikelihood

from bo.acquisition_functions.acquisition_functions import (
    AcquisitionFunctionType,
    acquisition_function_factory,
)
from bo.constrained_functions.synthetic_problems import ConstrainedBranin
from bo.model.Model import ConstrainedPosteriorMean

dtype = torch.double
torch.set_default_dtype(dtype)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obj_callable(Z, X=None):
    return Z[..., 0]


def _build_model(d, n_points, seed=0):
    """Build a fitted ModelListGP (1 objective + 1 constraint) on GPU."""
    torch.manual_seed(seed)
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


def _best_location(model, d):
    bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=dtype, device=DEVICE)
    loc, _ = optimize_acqf(
        ConstrainedPosteriorMean(model, penalty_value=torch.tensor([0.0], device=DEVICE)),
        bounds=bounds, q=1, num_restarts=10, raw_samples=128,
    )
    return loc


def _time_forward(acqf, test_X, warmup=1, repeats=3):
    """Time a forward pass with warmup. Returns median time."""
    for _ in range(warmup):
        acqf.forward(test_X.clone())
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = acqf.forward(test_X.clone())
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def _make_old_ckg(model, objective, loc, penalty):
    return acquisition_function_factory(
        type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
        model=model, objective=objective, best_value=None, idx=None,
        number_of_outputs=2, penalty_value=penalty, iteration=0,
        initial_condition_internal_optimizer=loc,
    )


def _make_fast_ckg(model, objective, loc, penalty):
    return acquisition_function_factory(
        type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2,
        model=model, objective=objective, best_value=None, idx=None,
        number_of_outputs=2, penalty_value=penalty, iteration=0,
        initial_condition_internal_optimizer=loc,
    )


def _make_old_dckg(model, objective, loc, penalty, source_idx):
    return acquisition_function_factory(
        type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
        model=model, objective=objective, best_value=None, idx=source_idx,
        number_of_outputs=2, penalty_value=penalty, iteration=0,
        initial_condition_internal_optimizer=loc,
    )


def _make_fast_dckg(model, objective, loc, penalty, source_idx):
    return acquisition_function_factory(
        type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2,
        model=model, objective=objective, best_value=None, idx=source_idx,
        number_of_outputs=2, penalty_value=penalty, iteration=0,
        initial_condition_internal_optimizer=loc,
    )


def _print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_row(label, t_old, t_new):
    speedup = t_old / max(t_new, 1e-9)
    print(f"  {label:45s}  {t_old:8.3f}s  {t_new:8.3f}s  {speedup:6.1f}x")


def run_benchmark():
    d = 2
    n_candidates = 5
    penalty = torch.tensor([2.0], dtype=dtype, device=DEVICE)
    objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])

    print("\n" + "#" * 70)
    print("#  TIMING COMPARISON: Old vs Fast cKG / dcKG")
    print("#  Problem: ConstrainedBranin (2D, 1 constraint)")
    print(f"#  Device: {DEVICE}")
    print(f"#  Forward pass on {n_candidates} candidate points")
    print("#" * 70)

    for n_points in [10, 100]:
        _print_header(f"Training points: {n_points}")
        print(f"  {'Method':45s}  {'Old':>8s}  {'Fast':>8s}  {'Speedup':>6s}")
        print(f"  {'-' * 45}  {'-' * 8}  {'-' * 8}  {'-' * 6}")

        model = _build_model(d, n_points, seed=42)
        loc = _best_location(model, d)

        # --- Coupled cKG ---
        test_X_ckg = torch.rand(n_candidates, 1, d, device=DEVICE)

        old_ckg = _make_old_ckg(model, objective, loc, penalty)
        t_old_ckg = _time_forward(old_ckg, test_X_ckg)

        fast_ckg = _make_fast_ckg(model, objective, loc, penalty)
        t_fast_ckg = _time_forward(fast_ckg, test_X_ckg)

        _print_row("Coupled cKG forward", t_old_ckg, t_fast_ckg)

        # --- dcKG source 0 (objective) ---
        test_X_dckg = torch.rand(n_candidates, 1, d, device=DEVICE)

        old_dckg0 = _make_old_dckg(model, objective, loc, penalty, 0)
        t_old_dckg0 = _time_forward(old_dckg0, test_X_dckg)

        fast_dckg0 = _make_fast_dckg(model, objective, loc, penalty, 0)
        t_fast_dckg0 = _time_forward(fast_dckg0, test_X_dckg)

        _print_row("dcKG source 0 (objective) forward", t_old_dckg0, t_fast_dckg0)

        # --- dcKG source 1 (constraint) ---
        old_dckg1 = _make_old_dckg(model, objective, loc, penalty, 1)
        t_old_dckg1 = _time_forward(old_dckg1, test_X_dckg)

        fast_dckg1 = _make_fast_dckg(model, objective, loc, penalty, 1)
        t_fast_dckg1 = _time_forward(fast_dckg1, test_X_dckg)

        _print_row("dcKG source 1 (constraint) forward", t_old_dckg1, t_fast_dckg1)

        # --- Full iteration: construct + forward + optimize ---
        print()
        print(f"  {'Full single iteration (construct + optimize_acqf):':50s}")
        print(f"  {'-' * 45}  {'-' * 8}  {'-' * 8}  {'-' * 6}")

        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=dtype, device=DEVICE)

        # Old cKG full iteration
        torch.manual_seed(0)
        t0 = time.perf_counter()
        old_ckg_iter = _make_old_ckg(model, objective, loc, penalty)
        optimize_acqf(old_ckg_iter, bounds, q=1, num_restarts=5, raw_samples=30,
                      options={"maxiter": 50})
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_old_ckg_iter = time.perf_counter() - t0

        torch.manual_seed(0)
        t0 = time.perf_counter()
        fast_ckg_iter = _make_fast_ckg(model, objective, loc, penalty)
        optimize_acqf(fast_ckg_iter, bounds, q=1, num_restarts=5, raw_samples=30,
                      options={"maxiter": 50})
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_fast_ckg_iter = time.perf_counter() - t0

        _print_row("Coupled cKG (optimize_acqf)", t_old_ckg_iter, t_fast_ckg_iter)

        # Old dcKG full iteration (both sources)
        torch.manual_seed(0)
        t0 = time.perf_counter()
        for src in [0, 1]:
            old_d = _make_old_dckg(model, objective, loc, penalty, src)
            optimize_acqf(old_d, bounds, q=1, num_restarts=5, raw_samples=30,
                          options={"maxiter": 50})
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_old_dckg_iter = time.perf_counter() - t0

        torch.manual_seed(0)
        t0 = time.perf_counter()
        for src in [0, 1]:
            fast_d = _make_fast_dckg(model, objective, loc, penalty, src)
            optimize_acqf(fast_d, bounds, q=1, num_restarts=5, raw_samples=30,
                          options={"maxiter": 50})
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_fast_dckg_iter = time.perf_counter() - t0

        _print_row("dcKG all sources (optimize_acqf)", t_old_dckg_iter, t_fast_dckg_iter)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
