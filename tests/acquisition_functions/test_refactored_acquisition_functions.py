"""Tests for FastConstrainedKG (reparametrization-based cKG/dcKG).

Tests:
  1. KGCB epigraph helper functions
  2. Shape correctness (1D/2D, single-source, all-sources)
  3. Numerical equivalence with original on small problems
  4. Timing comparison: old vs new
"""

import time
from typing import Optional

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ConstrainedMCObjective
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling import ListSampler
from botorch.utils.testing import BotorchTestCase
from gpytorch import settings
from gpytorch.mlls import SumMarginalLogLikelihood

from bo.acquisition_functions.acquisition_functions import (
    AcquisitionFunctionType,
    DecopledHybridConstrainedKnowledgeGradient,
    acquisition_function_factory,
    filter_a_b,
)
from bo.acquisition_functions.refactored_acquisition_functions import (
    FastConstrainedKG,
    _kgcb,
    _kgcb_batched,
    _filter_a_b,
    _cpu_epigraph_sweep,
    _batch_epigraph_sweep,
)
from bo.constrained_functions.synthetic_problems import ConstrainedBranin
from bo.model.Model import ConstrainedPosteriorMean
from bo.samplers.samplers import quantileSampler

dtype = torch.double
torch.set_default_dtype(dtype)
settings.min_fixed_noise._global_double_value = 1e-6


def obj_callable(Z, X=None):
    return Z[..., 0]


def _build_model(d, n_points, seed=0):
    """Build a fitted ModelListGP (1 objective + 1 constraint)."""
    torch.manual_seed(seed)
    X = torch.rand(n_points, d, dtype=dtype)
    NOISE = torch.tensor(1e-6, dtype=dtype)
    if d == 2:
        func = ConstrainedBranin()
        Y_obj = func.evaluate_true(X).unsqueeze(-1)
        Y_con = func.evaluate_slack_true(X)
    else:
        Y_obj = torch.rand(n_points, 1, dtype=dtype)
        Y_con = torch.rand(n_points, 1, dtype=dtype)
    m_obj = SingleTaskGP(X, Y_obj, train_Yvar=NOISE.expand_as(Y_obj))
    m_con = SingleTaskGP(X, Y_con, train_Yvar=NOISE.expand_as(Y_con))
    model = ModelListGP(m_obj, m_con)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model, d


def _best_location(model, d):
    bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=dtype)
    loc, _ = optimize_acqf(
        ConstrainedPosteriorMean(model, penalty_value=torch.tensor([0.0])),
        bounds=bounds, q=1, num_restarts=10, raw_samples=128,
    )
    return loc


# ─── KGCB helpers ──────────────────────────────────────────────────

class TestKGCB(BotorchTestCase):
    def test_basic(self):
        a = torch.tensor([1.0, 2.0, 0.5])
        b = torch.tensor([0.1, 0.2, 0.3])
        self.assertTrue(torch.isfinite(_kgcb(a, b, torch.tensor(2.0))))

    def test_filter_a_b(self):
        a = torch.tensor([1, 0, 3, 9, 4, 7, 2], dtype=dtype)
        b = torch.tensor([2, 2, 2, 2, 3, 3, 1], dtype=dtype)
        fa, fb = _filter_a_b(a, b)
        self.assertAllClose(fa, torch.tensor([2, 9, 7], dtype=dtype))
        self.assertAllClose(fb, torch.tensor([1, 2, 3], dtype=dtype))

    def test_filter_same_b(self):
        a = torch.tensor([1, 0, 3, 9, 4, 7, 2], dtype=dtype)
        b = torch.tensor([2, 2, 2, 2, 2, 2, 2], dtype=dtype)
        fa, fb = _filter_a_b(a, b)
        self.assertAllClose(fa, torch.tensor([9], dtype=dtype))
        self.assertAllClose(fb, torch.tensor([2], dtype=dtype))


# ─── Shape tests ───────────────────────────────────────────────────

class TestFastDcKGShapes1D(BotorchTestCase):
    def test_single_source(self):
        model, d = _build_model(1, 10)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        loc = _best_location(model, d)
        penalty = torch.tensor([2.0], dtype=dtype)

        for n in [1, 2, 5]:
            for idx in [0, 1]:
                acqf = FastConstrainedKG(
                    model, penalty_value=penalty, x_best_location=loc,
                    evaluate_all_sources=False, source_index=idx,
                    n_fantasies=7, number_of_raw_points=50,
                    number_of_restarts=5, seed=0, objective=objective,
                    x_evaluation_mask=torch.zeros(1, 2, dtype=torch.bool),
                )
                kgs = acqf.forward(torch.rand(n, 1, d))
                self.assertEqual(kgs.shape, torch.Size([n]))


class TestFastDcKGShapes2D(BotorchTestCase):
    def test_single_source(self):
        model, d = _build_model(2, 8)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        loc = _best_location(model, d)
        penalty = torch.tensor([2.0], dtype=dtype)

        for n in [1, 2, 5]:
            for idx in [0, 1]:
                acqf = FastConstrainedKG(
                    model, penalty_value=penalty, x_best_location=loc,
                    evaluate_all_sources=False, source_index=idx,
                    n_fantasies=7, number_of_raw_points=50,
                    number_of_restarts=5, seed=0, objective=objective,
                    x_evaluation_mask=torch.zeros(1, 2, dtype=torch.bool),
                )
                kgs = acqf.forward(torch.rand(n, 1, d))
                self.assertEqual(kgs.shape, torch.Size([n]))


class TestFastCKGShapes(BotorchTestCase):
    def test_1d(self):
        model, d = _build_model(1, 10)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        loc = _best_location(model, d)

        for n in [1, 2, 5]:
            acqf = acquisition_function_factory(
                type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2,
                model=model, objective=objective, best_value=None, idx=None,
                number_of_outputs=2, penalty_value=torch.tensor([0.0]),
                iteration=0, initial_condition_internal_optimizer=loc,
            )
            kgs = acqf.forward(torch.rand(n, 1, d))
            self.assertEqual(kgs.shape, torch.Size([n]))

    def test_2d(self):
        model, d = _build_model(2, 8)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        loc = _best_location(model, d)

        for n in [1, 2, 5]:
            acqf = acquisition_function_factory(
                type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2,
                model=model, objective=objective, best_value=None, idx=None,
                number_of_outputs=2, penalty_value=torch.tensor([0.0]),
                iteration=0, initial_condition_internal_optimizer=loc,
            )
            kgs = acqf.forward(torch.rand(n, 1, d))
            self.assertEqual(kgs.shape, torch.Size([n]))


# ─── Non-negativity and sanity ─────────────────────────────────────

class TestFastKGSanity(BotorchTestCase):
    """KG values should be non-negative (or very close to zero)."""

    def test_objective_kg_nonneg(self):
        model, d = _build_model(1, 10)
        loc = _best_location(model, d)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        acqf = FastConstrainedKG(
            model, penalty_value=torch.tensor([2.0]), x_best_location=loc,
            evaluate_all_sources=False, source_index=0, n_fantasies=7,
            number_of_raw_points=50, number_of_restarts=5, seed=0,
            objective=objective,
            x_evaluation_mask=torch.zeros(1, 2, dtype=torch.bool),
        )
        kgs = acqf.forward(torch.rand(10, 1, d))
        self.assertTrue((kgs >= -1e-6).all(), f"Negative KG: {kgs}")

    def test_constraint_kg_nonneg(self):
        model, d = _build_model(1, 10)
        loc = _best_location(model, d)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        acqf = FastConstrainedKG(
            model, penalty_value=torch.tensor([2.0]), x_best_location=loc,
            evaluate_all_sources=False, source_index=1, n_fantasies=7,
            number_of_raw_points=50, number_of_restarts=5, seed=0,
            objective=objective,
            x_evaluation_mask=torch.zeros(1, 2, dtype=torch.bool),
        )
        kgs = acqf.forward(torch.rand(10, 1, d))
        self.assertTrue((kgs >= -1e-6).all(), f"Negative KG: {kgs}")

    def test_coupled_kg_nonneg(self):
        model, d = _build_model(1, 10)
        loc = _best_location(model, d)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        acqf = FastConstrainedKG(
            model, penalty_value=torch.tensor([2.0]), x_best_location=loc,
            evaluate_all_sources=True, n_fantasies=7, n_constraint_samples=5,
            number_of_raw_points=50, number_of_restarts=5, seed=0,
            objective=objective,
            x_evaluation_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        kgs = acqf.forward(torch.rand(10, 1, d))
        # coupled cKG can be slightly negative due to MC approx, allow small tolerance
        self.assertTrue((kgs >= -0.1).all(), f"Large negative KG: {kgs}")


# ─── Timing comparison ─────────────────────────────────────────────

class TestTiming(BotorchTestCase):
    """Benchmark old vs new. Tests always pass — results are printed."""

    def _time_forward(self, acqf, test_X, warmup=1):
        for _ in range(warmup):
            acqf.forward(test_X.clone())
        t0 = time.perf_counter()
        _ = acqf.forward(test_X.clone())
        return time.perf_counter() - t0

    def test_dckg_timing(self):
        model, d = _build_model(2, 10)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        loc = _best_location(model, d)
        penalty = torch.tensor([2.0], dtype=dtype)
        test_X = torch.rand(5, 1, d)

        results = {}
        for label, make_acqf in [
            ("OLD dcKG", lambda idx: acquisition_function_factory(
                type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                model=model, objective=objective, best_value=None, idx=idx,
                number_of_outputs=2, penalty_value=penalty, iteration=0,
                initial_condition_internal_optimizer=loc)),
            ("NEW dcKG (Fast)", lambda idx: FastConstrainedKG(
                model, penalty_value=penalty, x_best_location=loc,
                evaluate_all_sources=False, source_index=idx, n_fantasies=7,
                number_of_raw_points=50, number_of_restarts=5, seed=0,
                objective=objective,
                x_evaluation_mask=torch.zeros(1, 2, dtype=torch.bool))),
        ]:
            t_total = 0
            for idx in [0, 1]:
                acqf = make_acqf(idx)
                t_total += self._time_forward(acqf, test_X, warmup=1)
            results[label] = t_total / 2

        self._print_results("dcKG forward (2D, 5 pts)", results)

    def test_ckg_timing(self):
        model, d = _build_model(2, 10)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        loc = _best_location(model, d)
        test_X = torch.rand(3, 1, d)

        results = {}
        for label, acqf_type in [
            ("OLD cKG", AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT),
            ("NEW cKG (Fast)", AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2),
        ]:
            torch.manual_seed(0)
            acqf = acquisition_function_factory(
                type=acqf_type, model=model, objective=objective,
                best_value=None, idx=None, number_of_outputs=2,
                penalty_value=torch.tensor([2.0]), iteration=0,
                initial_condition_internal_optimizer=loc,
            )
            results[label] = self._time_forward(acqf, test_X, warmup=1)

        self._print_results("coupled cKG forward (2D, 3 pts)", results)

    def test_optimize_acqf_timing(self):
        model, d = _build_model(2, 10)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=dtype)
        loc = _best_location(model, d)
        penalty = torch.tensor([2.0], dtype=dtype)

        results = {}
        for label, make_acqf in [
            ("OLD dcKG optimize_acqf", lambda: acquisition_function_factory(
                type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                model=model, objective=objective, best_value=None, idx=0,
                number_of_outputs=2, penalty_value=penalty, iteration=0,
                initial_condition_internal_optimizer=loc)),
            ("NEW dcKG optimize_acqf", lambda: FastConstrainedKG(
                model, penalty_value=penalty, x_best_location=loc,
                evaluate_all_sources=False, source_index=0, n_fantasies=7,
                number_of_raw_points=50, number_of_restarts=5, seed=0,
                objective=objective,
                x_evaluation_mask=torch.zeros(1, 2, dtype=torch.bool))),
        ]:
            acqf = make_acqf()
            t0 = time.perf_counter()
            optimize_acqf(acqf, bounds, q=1, num_restarts=5, raw_samples=30,
                          options={"maxiter": 50})
            results[label] = time.perf_counter() - t0

        self._print_results("optimize_acqf dcKG objective (2D)", results)

    @staticmethod
    def _print_results(title, results):
        print(f"\n{'=' * 60}")
        print(f"TIMING: {title}")
        print("=" * 60)
        for label, t in results.items():
            print(f"  {label:40s}: {t:.4f}s")
        if len(results) == 2:
            vals = list(results.values())
            speedup = vals[0] / max(vals[1], 1e-9)
            print(f"  Speedup: {speedup:.1f}x")
        print("=" * 60)


# ─── Numba vs original kgcb equivalence ──────────────────────────

import numpy as np

def _original_kgcb(a_row, b_row, current_best):
    """Call the original kgcb static method as the ground-truth reference."""
    return DecopledHybridConstrainedKnowledgeGradient.kgcb(a_row, b_row, current_best)


class TestEpigraphEquivalence(BotorchTestCase):
    """Verify the numba batch epigraph sweep produces the same envelope
    structure as the original kgcb implementation."""

    def setUp(self):
        super().setUp()
        if _batch_epigraph_sweep is None:
            self.skipTest("numba not available")

    def _reference_sweep(self, a_row, b_row):
        """Run the original filter_a_b + epigraph sweep on a single row.

        Uses filter_a_b from acquisition_functions.py (threshold=1e-9)
        to match the original kgcb exactly.
        Returns (envelope_original_indices, breakpoints).
        """
        a_0, b_0 = filter_a_b(a_row, b_row, threshold=1e-16)

        dev = a_row.device
        idz = [0]
        i_last = 0
        x = [torch.tensor(-torch.inf, device=dev)]
        n_lines = len(a_0)
        while i_last < n_lines - 1:
            i_mask = torch.arange(i_last + 1, n_lines, device=dev)
            x_mask = -(a_0[i_last] - a_0[i_mask]) / (b_0[i_last] - b_0[i_mask])
            best_pos = torch.argmin(x_mask)
            idz.append(i_mask[best_pos].item())
            x.append(x_mask[best_pos])
            i_last = idz[-1]
        x.append(torch.tensor(torch.inf, device=dev))

        x = torch.stack(x)
        idz = torch.tensor(idz, dtype=torch.long, device=dev)
        return a_0[idz], b_0[idz], x

    def _assert_sweep_equal(self, a_batch, b_batch):
        """Compare numba batch sweep against the original reference for each row."""
        a_np = a_batch.numpy().astype(np.float64)
        b_np = b_batch.numpy().astype(np.float64)
        epi_idx_np, epi_breaks_np, epi_counts_np = _batch_epigraph_sweep(a_np, b_np)

        N = a_batch.shape[0]
        for i in range(N):
            ref_a, ref_b, ref_breaks = self._reference_sweep(a_batch[i], b_batch[i])
            count = int(epi_counts_np[i])

            # Numba envelope a, b values via gather from filtered a, b
            numba_idx = epi_idx_np[i, :count]
            numba_a = a_batch[i][numba_idx]
            numba_b = b_batch[i][numba_idx]

            # Compare envelope a and b values
            self.assertEqual(len(numba_a), len(ref_a),
                             f"Row {i}: envelope size differs")
            self.assertAllClose(numba_a, ref_a, atol=1e-12, rtol=1e-12)
            self.assertAllClose(numba_b, ref_b, atol=1e-12, rtol=1e-12)

            # Compare breakpoints
            numba_breaks = torch.from_numpy(epi_breaks_np[i, :count + 1].copy())
            self.assertEqual(len(numba_breaks), len(ref_breaks),
                             f"Row {i}: breakpoint count differs")
            self.assertAllClose(numba_breaks, ref_breaks.double(),
                                atol=1e-12, rtol=1e-12)

    def test_simple_case(self):
        a = torch.tensor([[1.0, 2.0, 0.5],
                          [3.0, 1.0, 2.0]], dtype=dtype)
        b = torch.tensor([[0.1, 0.2, 0.3],
                          [0.5, 0.1, 0.3]], dtype=dtype)
        self._assert_sweep_equal(a, b)

    def test_single_point(self):
        a = torch.tensor([[5.0]], dtype=dtype)
        b = torch.tensor([[0.3]], dtype=dtype)
        self._assert_sweep_equal(a, b)

    def test_two_points(self):
        a = torch.tensor([[1.0, 3.0]], dtype=dtype)
        b = torch.tensor([[0.5, 0.1]], dtype=dtype)
        self._assert_sweep_equal(a, b)

    def test_duplicate_b_values(self):
        """When all b values are identical, only the highest-a point survives."""
        a = torch.tensor([[1.0, 5.0, 3.0, 2.0]], dtype=dtype)
        b = torch.tensor([[0.2, 0.2, 0.2, 0.2]], dtype=dtype)
        self._assert_sweep_equal(a, b)

    def test_random_small(self):
        torch.manual_seed(42)
        for _ in range(20):
            M = torch.randint(2, 10, (1,)).item()
            N = torch.randint(1, 5, (1,)).item()
            a = torch.randn(N, M, dtype=dtype)
            b = torch.randn(N, M, dtype=dtype).abs() + 0.01
            self._assert_sweep_equal(a, b)

    def test_random_larger(self):
        torch.manual_seed(123)
        N, M = 50, 30
        a = torch.randn(N, M, dtype=dtype)
        b = torch.randn(N, M, dtype=dtype).abs() + 0.01
        self._assert_sweep_equal(a, b)

    def test_negative_b_values(self):
        torch.manual_seed(7)
        N, M = 10, 8
        a = torch.randn(N, M, dtype=dtype)
        b = torch.randn(N, M, dtype=dtype)
        self._assert_sweep_equal(a, b)


class TestKGCBBatchedEquivalence(BotorchTestCase):
    """Verify _kgcb_batched produces the same KG values as the original
    ConstrainedKnowledgeGradient.kgcb called sequentially on each row."""

    def setUp(self):
        super().setUp()
        if _batch_epigraph_sweep is None:
            self.skipTest("numba not available")

    def _assert_batched_equals_original(self, a, b, current_best, atol=1e-10):
        # Batched (numba)
        kg_batched = _kgcb_batched(a, b, current_best)

        # Original kgcb, row by row
        N = a.shape[0]
        kg_orig = torch.stack([_original_kgcb(a[i], b[i], current_best[i]) for i in range(N)])

        self.assertAllClose(kg_batched, kg_orig, atol=atol, rtol=1e-10)

    def test_simple(self):
        a = torch.tensor([[1.0, 2.0, 0.5],
                          [3.0, 1.0, 2.0]], dtype=dtype)
        b = torch.tensor([[0.1, 0.2, 0.3],
                          [0.5, 0.1, 0.3]], dtype=dtype)
        current = torch.tensor([2.0, 3.0], dtype=dtype)
        self._assert_batched_equals_original(a, b, current)

    def test_single_row(self):
        a = torch.tensor([[1.0, 3.0, 2.0, 4.0]], dtype=dtype)
        b = torch.tensor([[0.1, 0.3, 0.2, 0.5]], dtype=dtype)
        current = torch.tensor([3.0], dtype=dtype)
        self._assert_batched_equals_original(a, b, current)

    def test_identical_b(self):
        a = torch.tensor([[5.0, 1.0, 3.0]], dtype=dtype)
        b = torch.tensor([[0.2, 0.2, 0.2]], dtype=dtype)
        current = torch.tensor([5.0], dtype=dtype)
        self._assert_batched_equals_original(a, b, current)

    def test_random_small(self):
        torch.manual_seed(99)
        for _ in range(20):
            M = torch.randint(2, 10, (1,)).item()
            N = torch.randint(1, 8, (1,)).item()
            a = torch.randn(N, M, dtype=dtype)
            b = torch.randn(N, M, dtype=dtype).abs() + 0.01
            current = a.max(dim=1).values
            self._assert_batched_equals_original(a, b, current)

    def test_random_larger(self):
        torch.manual_seed(456)
        N, M = 100, 20
        a = torch.randn(N, M, dtype=dtype)
        b = torch.randn(N, M, dtype=dtype).abs() + 0.01
        current = a.max(dim=1).values
        self._assert_batched_equals_original(a, b, current)

    def test_mixed_sign_b(self):
        torch.manual_seed(77)
        N, M = 15, 12
        a = torch.randn(N, M, dtype=dtype)
        b = torch.randn(N, M, dtype=dtype)
        current = a.max(dim=1).values
        self._assert_batched_equals_original(a, b, current)

    def test_wide_value_range(self):
        """Test with a and b values spanning several orders of magnitude."""
        torch.manual_seed(33)
        N, M = 10, 15
        a = torch.randn(N, M, dtype=dtype) * 100
        b = torch.randn(N, M, dtype=dtype).abs() * 50 + 0.1
        current = a.max(dim=1).values
        self._assert_batched_equals_original(a, b, current, atol=1e-8)
