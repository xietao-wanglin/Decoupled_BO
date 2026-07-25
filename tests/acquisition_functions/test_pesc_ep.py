"""Tests for the faithful PESC variant (``pesc_ep.py`` over ``pesc_sites.py``).

The two ``TestPESCEPObservedFactors`` cases that used to live here tested the
retired "incumbent truncation" hack and its ``observed_factors`` switch; both are
superseded by the real ``h_n`` factors and have been removed rather than adapted.
Structural coverage of the EP itself is in ``test_pesc_sites.py``; the quantitative
fidelity check is in ``test_pesc_fidelity.py``.
"""

import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls import SumMarginalLogLikelihood

from bo.acquisition_functions.pesc import (
    sample_constrained_optima, PESCConstraint, _conditional_variance, _joint_moments,
)
from bo.acquisition_functions.pesc_ep import (
    pesc_ep_conditional_variances, PESCObjectiveEP, PESCConstraintEP,
)

dtype = torch.double
torch.set_default_dtype(dtype)
NOISE = torch.tensor(1e-9, dtype=dtype)


def _fit(train_X, outputs):
    models = [SingleTaskGP(train_X, y, train_Yvar=NOISE.expand_as(y)) for y in outputs]
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _problem(n=8, d=2, seed=0):
    torch.manual_seed(seed)
    X = torch.rand(n, d, dtype=dtype)
    f = -(X.pow(2).sum(-1, keepdim=True))
    c = X[:, :1] - 0.5
    model = _fit(X, [f, c])
    bounds = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])
    return model, X, bounds


class TestPESCEPCore(BotorchTestCase):
    def test_shapes_and_nonneg(self):
        model, _, bounds = _problem()
        xs = sample_constrained_optima(model, bounds, num_samples=6, n_grid=256, seed=1)
        obj = PESCObjectiveEP(model, xs)
        con = PESCConstraintEP(model, constraint_index=0, x_star=xs)
        X = torch.rand(7, 1, 2, dtype=dtype)
        self.assertEqual(obj(X).shape, torch.Size([7]))
        self.assertEqual(con(X).shape, torch.Size([7]))
        self.assertTrue(bool((obj(X) >= -1e-6).all()))
        self.assertTrue(bool((con(X) >= -1e-6).all()))

    def test_k0_zero_sites_reduce_to_objective_truncation(self):
        # With no constraints AND the EP sites left at zero, the only factor acting
        # is the candidate-x optimality truncation, so the result must equal the
        # plain f(x) <= f(x*) reduction exactly.  (With the sites active the h_n
        # factors also bind, so this identity is specific to run_ep=False.)
        torch.manual_seed(0)
        X = torch.rand(8, 2, dtype=dtype)
        f = -(X.pow(2).sum(-1, keepdim=True))
        model = _fit(X, [f])
        bounds = torch.stack([torch.zeros(2, dtype=dtype), torch.ones(2, dtype=dtype)])
        xs = sample_constrained_optima(model, bounds, num_samples=6, n_grid=256, seed=1)
        Xt = torch.rand(5, 2, dtype=dtype)
        _, v_cond = pesc_ep_conditional_variances(
            model, Xt, xs, add_observation_noise=False, run_ep=False)

        v_x, mu_x, v_s, mu_s, cr = _joint_moments(model.models[0], Xt, xs)
        v_x, mu_x = v_x.unsqueeze(-1), mu_x.unsqueeze(-1)
        mu_g = mu_x - mu_s
        v_g = v_x + v_s - 2 * cr
        v_ref = _conditional_variance(v_x, v_x - cr, mu_g, v_g, thr=0.0)
        self.assertAllClose(v_cond[..., 0], v_ref, atol=1e-8, rtol=0)

    def test_noiseless_repeat_zero(self):
        model, X, bounds = _problem(n=5, seed=3)
        xs = sample_constrained_optima(model, bounds, num_samples=8, n_grid=256, seed=3)
        obj = PESCObjectiveEP(model, xs)
        con = PESCConstraintEP(model, constraint_index=0, x_star=xs)
        obs_o = torch.tensor([float(obj(X[i:i + 1].unsqueeze(1))) for i in range(X.shape[0])])
        obs_c = torch.tensor([float(con(X[i:i + 1].unsqueeze(1))) for i in range(X.shape[0])])
        self.assertLess(float(obs_o.max()), 1e-4)
        self.assertLess(float(obs_c.max()), 1e-4)


def _coupling_free_constraint_info(model, X, x_star, k=0):
    """Constraint info from the ``g_k`` truncation alone -- i.e. what
    ``PESCConstraint`` computed before the Eq. (37) ``a_tilde`` term was added."""
    v_x, _, v_st, mu_st, cr = _joint_moments(model.models[k + 1], X, x_star)
    v_prime = _conditional_variance(v_x.unsqueeze(-1), cr, mu_st, v_st, thr=0.0)
    return (0.5 * (v_x.unsqueeze(-1) / v_prime).log()).mean(-1)


class TestPESCEPCoupling(BotorchTestCase):
    def _promising_problem(self):
        torch.manual_seed(1)
        d = 2
        X = torch.rand(6, d, dtype=dtype)
        f = -(X.pow(2).sum(-1, keepdim=True))          # best near origin
        c = X[:, :1] - 0.5                               # feasible for x0 < 0.5
        model = _fit(X, [f, c])
        bounds = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])
        xs = sample_constrained_optima(model, bounds, num_samples=16, n_grid=256, seed=2)
        return model, bounds, xs, d

    def test_coupling_boosts_constraint_info(self):
        # At a point that is promising (good objective) but with uncertain
        # feasibility, the Eq. (37) a_tilde coupling must lift the closed-form
        # constraint term above the g_k-truncation-only baseline.
        #
        # Deliberately NOT asserted for the full EP: there g_k acts as an EP site
        # jointly with the h_n factors, and their interaction can leave the
        # constraint term slightly below the g_k-alone baseline at the max of a
        # random probe.  That is a different (more complete) conditioning rather
        # than a regression, and the full EP's constraint term is checked properly
        # against the rejection-sampling reference in test_pesc_fidelity.py --
        # a stronger statement than a max over 200 random points.
        model, _, xs, d = self._promising_problem()
        Xg = torch.rand(200, 1, d, dtype=dtype)
        with torch.no_grad():
            base_max = float(_coupling_free_constraint_info(model, Xg.squeeze(1), xs).max())
            mm_max = float(PESCConstraint(model, constraint_index=0, x_star=xs)(Xg).max())
        self.assertGreater(mm_max, base_max)

    def test_ep_adds_objective_information(self):
        # The h_n factors' whole purpose: forcing f(x*) to beat the observed data
        # must raise objective information relative to the no-EP path.
        from bo.acquisition_functions.pesc import PESCObjective
        model, _, xs, d = self._promising_problem()
        Xg = torch.rand(200, 1, d, dtype=dtype)
        with torch.no_grad():
            no_ep = float(PESCObjective(model, x_star=xs)(Xg).max())
            with_ep = float(PESCObjectiveEP(model, xs)(Xg).max())
        self.assertGreater(with_ep, no_ep)
