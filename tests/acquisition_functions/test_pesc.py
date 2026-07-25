import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls import SumMarginalLogLikelihood

from bo.acquisition_functions.pesc import (
    sample_constrained_optima, PESCObjective, PESCConstraint,
)
from bo.model.Model import ConstrainedPosteriorMean

dtype = torch.double
torch.set_default_dtype(dtype)

NOISE = torch.tensor(1e-9, dtype=dtype)


def _fit_model(train_X, outputs):
    """outputs: list of (n,1) tensors, index 0 objective, 1.. constraints."""
    models = [SingleTaskGP(train_X, y, train_Yvar=NOISE.expand_as(y)) for y in outputs]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def _default_problem(n=8, d=2, seed=0):
    torch.manual_seed(seed)
    train_X = torch.rand(n, d, dtype=dtype)
    # negated objective (maximise), one active constraint x0 - 0.5 <= 0
    f = -(train_X.pow(2).sum(-1, keepdim=True))
    c = train_X[:, :1] - 0.5
    model = _fit_model(train_X, [f, c])
    bounds = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])
    return model, train_X, bounds


class TestPESCSampleOptima(BotorchTestCase):
    def test_shape_and_bounds(self):
        model, _, bounds = _default_problem()
        x_star = sample_constrained_optima(model, bounds, num_samples=12, n_grid=256, seed=1)
        self.assertEqual(x_star.shape, torch.Size([12, 2]))
        self.assertTrue(bool((x_star >= 0).all() and (x_star <= 1).all()))

    def test_sampled_optima_are_feasible(self):
        # Feasible region (x0 <= 0.5) is non-empty; sampled optima should have
        # high posterior feasibility on the active constraint.
        model, _, bounds = _default_problem()
        x_star = sample_constrained_optima(model, bounds, num_samples=16, n_grid=256, seed=2)
        cpm = ConstrainedPosteriorMean(model=model)
        pf = cpm.evaluate_feasibility_by_index(x_star.unsqueeze(1), 1).detach()
        # Most sampled optima should be feasible under the posterior.
        self.assertGreater(float((pf > 0.5).float().mean()), 0.5)


class TestPESCNoiselessRepeat(BotorchTestCase):
    """Sampling an already-observed source/location gives ~no information.

    Uses a sparse model so that unexplored regions carry real information: the
    property is that a *re-observed* location's information gain is negligible
    both in absolute terms and relative to an unexplored location.
    """

    def _grid(self):
        import itertools
        return torch.tensor(list(itertools.product([0.0, 0.5, 1.0], repeat=2)), dtype=dtype)

    def test_objective_zero_at_observed_location(self):
        model, train_X, bounds = _default_problem(n=5, seed=3)
        x_star = sample_constrained_optima(model, bounds, num_samples=16, n_grid=256, seed=3)
        acqf = PESCObjective(model, x_star=x_star)
        obs = torch.tensor([float(acqf(train_X[i:i + 1].unsqueeze(1))) for i in range(train_X.shape[0])])
        unexplored = torch.tensor([float(acqf(g.reshape(1, 1, 2))) for g in self._grid()])
        self.assertLess(float(obs.max()), 1e-4)
        self.assertLess(float(obs.max()), 0.01 * float(unexplored.max()))

    def test_constraint_zero_at_observed_location(self):
        model, train_X, bounds = _default_problem(n=5, seed=4)
        x_star = sample_constrained_optima(model, bounds, num_samples=16, n_grid=256, seed=4)
        acqf = PESCConstraint(model, constraint_index=0, x_star=x_star)
        obs = torch.tensor([float(acqf(train_X[i:i + 1].unsqueeze(1))) for i in range(train_X.shape[0])])
        unexplored = torch.tensor([float(acqf(g.reshape(1, 1, 2))) for g in self._grid()])
        self.assertLess(float(obs.max()), 1e-4)
        self.assertLess(float(obs.max()), 0.01 * float(unexplored.max()))


class TestPESCNumericalSanity(BotorchTestCase):
    def test_non_negativity(self):
        model, _, bounds = _default_problem()
        x_star = sample_constrained_optima(model, bounds, num_samples=12, n_grid=256, seed=5)
        obj = PESCObjective(model, x_star=x_star)
        con = PESCConstraint(model, constraint_index=0, x_star=x_star)
        X = torch.rand(20, 1, 2, dtype=dtype)
        self.assertTrue(bool((obj(X) >= -1e-6).all()))
        self.assertTrue(bool((con(X) >= -1e-6).all()))

    def test_positive_away_from_data(self):
        # A location far from every constraint observation retains uncertainty,
        # so its constraint information gain is strictly positive.
        model, train_X, bounds = _default_problem(n=5)
        x_star = sample_constrained_optima(model, bounds, num_samples=16, n_grid=256, seed=6)
        con = PESCConstraint(model, constraint_index=0, x_star=x_star)
        # farthest corner from the (few) training points
        candidates = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=dtype)
        dists = torch.cdist(candidates, train_X).min(dim=1).values
        x_far = candidates[dists.argmax()].reshape(1, 1, 2)
        self.assertGreater(float(con(x_far)), 0.0)

    def test_objective_and_constraint_same_scale(self):
        # Both source terms use the same moment-matching mechanism, so their
        # values are of comparable magnitude (unlike qPES vs a feasibility term).
        model, _, bounds = _default_problem(n=5)
        x_star = sample_constrained_optima(model, bounds, num_samples=16, n_grid=256, seed=7)
        obj = PESCObjective(model, x_star=x_star)
        con = PESCConstraint(model, constraint_index=0, x_star=x_star)
        X = torch.rand(64, 1, 2, dtype=dtype)
        omax = float(obj(X).max())
        cmax = float(con(X).max())
        self.assertGreater(cmax, 0.0)
        # within ~2 orders of magnitude, not the ~50x qPES imbalance
        self.assertLess(omax / max(cmax, 1e-9), 30.0)

    def test_forward_shape(self):
        model, _, bounds = _default_problem()
        x_star = sample_constrained_optima(model, bounds, num_samples=8, n_grid=256, seed=8)
        obj = PESCObjective(model, x_star=x_star)
        con = PESCConstraint(model, constraint_index=0, x_star=x_star)
        X = torch.rand(7, 1, 2, dtype=dtype)
        self.assertEqual(obj(X).shape, torch.Size([7]))
        self.assertEqual(con(X).shape, torch.Size([7]))


class TestPESCDummyConstraint(BotorchTestCase):
    def test_trivially_feasible_constraint_gives_no_info(self):
        # A constraint that is strongly satisfied everywhere (c ~ -1) makes x*
        # certainly feasible, so observing it yields ~zero information and it is
        # never selected.
        torch.manual_seed(0)
        d = 2
        train_X = torch.rand(8, d, dtype=dtype)
        f = -(train_X.pow(2).sum(-1, keepdim=True))
        c_active = train_X[:, :1] - 0.5
        c_dummy = torch.full((8, 1), -1.0, dtype=dtype)
        model = _fit_model(train_X, [f, c_active, c_dummy])
        bounds = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])
        x_star = sample_constrained_optima(model, bounds, num_samples=16, n_grid=256, seed=9)
        dummy = PESCConstraint(model, constraint_index=1, x_star=x_star)
        X = torch.rand(20, 1, d, dtype=dtype)
        self.assertLess(float(dummy(X).max()), 1e-3)


class TestPESCSourceSelection(BotorchTestCase):
    def test_cost_normalized_argmax(self):
        info = torch.tensor([0.10, 0.30, 0.05], dtype=dtype)
        costs = torch.tensor([1.0, 5.0, 1.0], dtype=dtype)
        # value/cost = [0.10, 0.06, 0.05] -> objective (index 0) wins
        self.assertEqual(int(torch.argmax(info / costs)), 0)
        costs2 = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
        # equal costs -> largest raw info (index 1) wins
        self.assertEqual(int(torch.argmax(info / costs2)), 1)
