"""Tests for FastConstrainedKG (reparametrization-based cKG/dcKG).

Tests:
  1. KGCB epigraph helper functions
  2. Shape correctness (1D/2D, single-source, all-sources)
  3. Non-negativity and sanity of the KG values
  4. Equivalence of the batched and sequential KGCB implementations
"""

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ConstrainedMCObjective
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.utils.testing import BotorchTestCase
from gpytorch import settings
from gpytorch.mlls import SumMarginalLogLikelihood

from bo.acquisition_functions.acquisition_functions import (
    AcquisitionFunctionType,
    acquisition_function_factory,
)
from bo.acquisition_functions.refactored_acquisition_functions import (
    FastConstrainedKG,
    _kgcb,
    _kgcb_batched,
    _filter_a_b,
    _batch_epigraph_sweep,
)
from bo.constrained_functions.synthetic_problems import ConstrainedBranin
from bo.model.Model import ConstrainedPosteriorMean

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



class TestKGCBBatchedEquivalence(BotorchTestCase):
    """Verify the numba-batched _kgcb_batched matches the sequential _kgcb
    reference implementation called row by row."""

    def setUp(self):
        super().setUp()
        if _batch_epigraph_sweep is None:
            self.skipTest("numba not available")

    def _assert_batched_equals_original(self, a, b, current_best, atol=1e-10):
        # Batched (numba)
        kg_batched = _kgcb_batched(a, b, current_best)

        # Original kgcb, row by row
        N = a.shape[0]
        kg_orig = torch.stack([_kgcb(a[i], b[i], current_best[i]) for i in range(N)])

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
