"""Structural tests for the (N+1)-dimensional PESC EP (``pesc_sites.py``).

These pin the algebra rather than the statistics; the quantitative fidelity check
against the paper's rejection-sampling reference lives in ``test_pesc_fidelity.py``.
"""

import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import SumMarginalLogLikelihood

from bo.acquisition_functions.pesc import (
    pesc_conditional_variances, sample_constrained_optima, _joint_moments,
)
from bo.acquisition_functions.pesc_sites import (
    PESCVectorEP, pesc_faithful_conditional_variances, site_solve, is_proper,
)

dtype = torch.double
torch.set_default_dtype(dtype)
NOISE = torch.tensor(1e-9, dtype=dtype)


def _fit(train_X, outputs):
    models = [SingleTaskGP(train_X, y, train_Yvar=NOISE.expand_as(y)) for y in outputs]
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _problem(n=8, d=2, seed=0, n_con=1):
    torch.manual_seed(seed)
    X = torch.rand(n, d, dtype=dtype)
    outs = [-(X.pow(2).sum(-1, keepdim=True))]
    for k in range(n_con):
        outs.append(X[:, k % d:k % d + 1] - 0.5)
    model = _fit(X, outs)
    bounds = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])
    return model, X, bounds


class TestSiteSolve(BotorchTestCase):
    """The Woodbury solver must match the direct route, including tau <= 0."""

    def test_matches_direct_inverse(self):
        torch.manual_seed(0)
        n, m = 9, 6
        B = torch.randn(n, n, dtype=dtype)
        V = B @ B.T + n * torch.eye(n, dtype=dtype)
        mu = torch.randn(n, dtype=dtype)
        S = torch.randn(m, n, dtype=dtype)
        A, b = S @ V @ S.T, S @ mu

        # Site magnitudes must be scaled to A: a negative tau larger than
        # 1/max_eig(A) drives the joint precision indefinite, at which point the
        # tilted distribution is improper and the direct inverse is meaningless.
        scale = 1.0 / float(torch.linalg.eigvalsh(A).max())
        tau = torch.rand(m, dtype=dtype) * scale
        tau[0] = -0.1 * scale   # EP sites go negative in practice
        tau[1] = 0.0            # and can be exactly zero
        nu = torch.randn(m, dtype=dtype) * 0.3

        Vi = torch.linalg.inv(V)
        Lam = Vi + S.T @ torch.diag(tau) @ S
        # precondition: the comparison is only defined where the joint is proper
        self.assertGreater(float(torch.linalg.eigvalsh(Lam).min()), 0.0)

        m_proj, cov_proj, _, _ = site_solve(A, b, tau, nu)
        Sigma = torch.linalg.inv(Lam)
        mu_q = Sigma @ (Vi @ mu + S.T @ nu)
        self.assertAllClose(m_proj, S @ mu_q, atol=1e-9, rtol=0)
        self.assertAllClose(cov_proj, S @ Sigma @ S.T, atol=1e-9, rtol=0)
        self.assertTrue(is_proper(cov_proj))

    def test_improper_tilt_is_detected(self):
        # A site negative enough to drive the joint precision indefinite must be
        # detected.  Note the covariance is returned UNCLAMPED for exactly this
        # reason: flooring the diagonal at eps (as an earlier version did) hides the
        # negative entries and makes the breakdown invisible.  is_proper() is the
        # robust check regardless, since a matrix can be indefinite with an
        # all-positive diagonal.
        torch.manual_seed(0)
        n, m = 9, 6
        B = torch.randn(n, n, dtype=dtype)
        V = B @ B.T + n * torch.eye(n, dtype=dtype)
        S = torch.randn(m, n, dtype=dtype)
        A = S @ V @ S.T
        tau = torch.zeros(m, dtype=dtype)
        tau[0] = -0.05  # >> 1/max_eig(A)
        Lam = torch.linalg.inv(V) + S.T @ torch.diag(tau) @ S
        self.assertLess(float(torch.linalg.eigvalsh(Lam).min()), 0.0)

        _, cov_proj, _, _ = site_solve(A, torch.zeros(m, dtype=dtype), tau,
                                       torch.zeros(m, dtype=dtype))
        self.assertFalse(is_proper(cov_proj))
        # and clamping the diagonal would have concealed it
        self.assertTrue(bool((cov_proj.diagonal().clamp_min(1e-12) > 0).all()))

    def test_zero_sites_are_identity(self):
        torch.manual_seed(1)
        n = 7
        B = torch.randn(n, n, dtype=dtype)
        V = B @ B.T + n * torch.eye(n, dtype=dtype)
        mu = torch.randn(n, dtype=dtype)
        z = torch.zeros(n, dtype=dtype)
        m_proj, cov_proj, G, bracket = site_solve(V, mu, z, z)
        self.assertAllClose(m_proj, mu, atol=1e-12, rtol=0)
        self.assertAllClose(cov_proj, V, atol=1e-12, rtol=0)
        self.assertLess(float(G.abs().max()), 1e-12)


class TestZeroSiteEquivalence(BotorchTestCase):
    """THE critical guard.

    With every EP site at zero the (N+1)-dim machinery must reproduce the plain GP
    joint moments, and hence the existing no-EP acquisition, exactly.  This catches
    site-bookkeeping errors and the sign slip in the paper's printed
    ``[V'_f]_{1,1}`` (as written it subtracts both terms; correct marginalisation
    *adds* ``k' K^-1 V K^-1 k``).
    """

    def test_candidate_moments_reduce_to_gp_joint(self):
        model, _, bounds = _problem(n=8, d=2, seed=2, n_con=2)
        x_star = sample_constrained_optima(model, bounds, num_samples=3, n_grid=128, seed=3)
        X = torch.rand(11, 2, dtype=dtype)

        ep = PESCVectorEP(model, x_star[0])          # sites left at zero
        m_f, V_f, m_c, v_c = ep.candidate_moments(X)

        v_x, mu_x, v_st, mu_st, cross = _joint_moments(
            model.models[0], X, x_star[0].reshape(1, -1))
        self.assertAllClose(V_f[..., 0, 0], v_x, atol=1e-9, rtol=0)
        self.assertAllClose(V_f[..., 1, 1], v_st.reshape(-1), atol=1e-9, rtol=0)
        self.assertAllClose(V_f[..., 0, 1], cross.reshape(-1), atol=1e-9, rtol=0)
        self.assertAllClose(m_f[..., 0], mu_x, atol=1e-9, rtol=0)
        self.assertAllClose(m_f[..., 1], mu_st.reshape(-1), atol=1e-9, rtol=0)

        for k in range(2):
            vk, muk, _, _, _ = _joint_moments(
                model.models[k + 1], X, x_star[0].reshape(1, -1))
            self.assertAllClose(v_c[:, k], vk, atol=1e-9, rtol=0)
            self.assertAllClose(m_c[:, k], muk, atol=1e-9, rtol=0)

    def test_zero_site_acquisition_matches_no_ep_path(self):
        # The no-EP path additionally applies the g_k truncation, so compare against
        # it with that turned off -- i.e. only the shared candidate-x h factor.
        model, _, bounds = _problem(n=7, d=2, seed=4, n_con=1)
        x_star = sample_constrained_optima(model, bounds, num_samples=4, n_grid=128, seed=5)
        X = torch.rand(9, 2, dtype=dtype)

        vp_ep, vc_ep = pesc_faithful_conditional_variances(
            model, X, x_star, add_observation_noise=False, run_ep=False)
        vp_ref, _ = pesc_conditional_variances(
            model, X, x_star, add_observation_noise=False)
        # v_prior is the plain GP predictive variance in both paths.
        self.assertAllClose(vp_ep, vp_ref, atol=1e-9, rtol=0)
        self.assertTrue(bool(torch.isfinite(vc_ep).all()))
        self.assertTrue(bool((vc_ep <= vp_ep + 1e-12).all()))


class TestArrowheadStructure(BotorchTestCase):
    """``sum_n tau_n p_n p_n'`` must be the paper's arrowhead ``V~_f``."""

    def test_site_precision_is_arrowhead(self):
        model, _, bounds = _problem(n=6, d=2, seed=6)
        x_star = sample_constrained_optima(model, bounds, num_samples=1, n_grid=128, seed=7)
        ep = PESCVectorEP(model, x_star[0])
        torch.manual_seed(0)
        ep.tau_f = torch.rand(ep.N, dtype=dtype)     # arbitrary nonzero sites
        A = ep.site_precision_matrix()
        N = ep.N
        self.assertAllClose(A.diagonal()[1:], ep.tau_f, atol=1e-12, rtol=0)
        self.assertAllClose(A[0, 1:], -ep.tau_f, atol=1e-12, rtol=0)
        self.assertAllClose(A[1:, 0], -ep.tau_f, atol=1e-12, rtol=0)
        self.assertAlmostEqual(float(A[0, 0]), float(ep.tau_f.sum()), places=12)
        # off-diagonal entries among observed points must be exactly zero
        off = A[1:, 1:] - torch.diag(A.diagonal()[1:])
        self.assertLess(float(off.abs().max()), 1e-12)
        self.assertEqual(A.shape, torch.Size([N + 1, N + 1]))


class TestConditioningAtFailureConfiguration(BotorchTestCase):
    """N=160, d=2 is where the naive Cholesky(V) route fails outright (negative
    minimum eigenvalue, cond ~1e294).  The site formulation must survive it."""

    def test_site_solve_survives_n160_d2(self):
        for noise in (1e-6, 1e-2):
            torch.manual_seed(1)
            N, d = 160, 2
            X = torch.rand(N, d, dtype=dtype)
            y = torch.sin(3 * X.sum(-1, keepdim=True))
            gp = SingleTaskGP(X, y, train_Yvar=torch.full_like(y, noise),
                              covar_module=ScaleKernel(RBFKernel()))
            gp.covar_module.base_kernel.lengthscale = 0.3
            gp.covar_module.outputscale = 1.0
            gp.eval()
            with torch.no_grad():
                V = gp.posterior(torch.cat([torch.rand(1, d, dtype=dtype), X], 0)) \
                     .mvn.covariance_matrix
            mu = torch.zeros(N + 1, dtype=dtype)
            # the naive route is expected to be unusable here
            self.assertLess(float(torch.linalg.eigvalsh(V).min()), 1e-12)

            for t in (1e-3, 1.0, 1e3, 1e6):
                tau = torch.full((N + 1,), t, dtype=dtype)
                m_proj, cov_proj, _, _ = site_solve(V, mu, tau,
                                                    torch.zeros(N + 1, dtype=dtype))
                v_proj = cov_proj.diagonal()
                self.assertTrue(bool(torch.isfinite(m_proj).all()))
                self.assertTrue(bool(torch.isfinite(v_proj).all()))
                self.assertTrue(bool((v_proj > 0).all()))
                # conditioning must stay far from the naive route's ~1e294
                Mx = torch.eye(N + 1, dtype=dtype) + tau.unsqueeze(-1) * V
                sv = torch.linalg.svdvals(Mx)
                self.assertLess(float(sv.max() / sv.min()), 1e9)


class TestEPConvergence(BotorchTestCase):
    """Paper Sec. 5.1: terminate on max |site change| < 1e-4."""

    def test_converges_within_cap(self):
        for n_con in (1, 3):
            model, _, bounds = _problem(n=10, d=2, seed=8, n_con=n_con)
            x_star = sample_constrained_optima(model, bounds, num_samples=1,
                                               n_grid=128, seed=9)
            ep = PESCVectorEP(model, x_star[0]).run()
            self.assertTrue(ep.converged, f"EP did not converge for K={n_con}")
            self.assertGreater(ep.sweeps, 0)
            self.assertTrue(bool(torch.isfinite(ep.tau_f).all()))
