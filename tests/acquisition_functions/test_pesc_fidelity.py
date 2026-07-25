"""Quantitative fidelity tests for PESC against the paper's own ground truth.

The property tests in ``test_pesc.py`` (shapes, non-negativity, vanishing at
observed points) all pass for a *wrong* formula -- which is how the missing
Eq. (37) ``a_tilde`` coupling term went unnoticed.  These tests instead compare
the acquisition against the rejection-sampling (RS) reference of Sec. 4.1 of
Hernandez-Lobato et al. (2015), reproducing the paper's setup: d=1, f and c1
drawn from a zero-mean GP with SE kernel of unit amplitude and l=0.1,
sigma_f^2 = sigma_1^2 = 0.01, 5 evaluations, the *true* GP hyperparameters, and
50 x* samples.

Deterministic (both the RS reference and x* sampling are seeded), 5 seeds, M=50:

    source      no-EP (``pesc.py``)      faithful (``pesc_ep.py``)
    objective   0.71 mean, 0.31 min      **0.95 mean, 0.87 min**
    constraint  0.95 mean, 0.85 min      0.95 mean, 0.86 min

The no-EP path implements only the closed-form per-candidate step (Eqs. 36-37) and
omits the N observed-point factors ``h_n`` of supplementary Eq. (4); nothing then
forces f(x*) to beat the observed data, which is the dominant source of objective
information.  Adding those factors (``pesc_sites.py``) is what moves the objective
row.  The constraint row is unchanged, as expected -- it was already faithful once
the Eq. (37) ``a_tilde`` term was in place.

**Known residual, deliberately not asserted away.** The faithful objective *peak* is
still 0.3-0.7x of RS, and on seeds 2 and 3 the argmax lands far from RS's at the
paper's N=5.  Investigated:

* not x* sampling variance -- M=50/200/800 lifts the correlation (0.87 -> 0.96 on
  seed 2) but leaves the argmax gap at ~0.84;
* not x* grid resolution -- n_grid 512 vs 4096 changes nothing;
* not an RS artifact -- empirical variances at RS's argmax are 0.4-0.9 against a
  1e-2 noise floor, with healthy bin counts (~2200);
* *partly* the paper's own approximation: supp. Eqs. (11)-(12) restrict the
  domain-wide optimality product to the N observed points, so at N=5 the condition
  "x* beats everything" is only enforced at 5 places while RS enforces it over the
  whole grid.  Raising N confirms this -- seed 2 reaches corr 0.993 and argmax gap
  0.005 at N=40.  Seed 3 does not recover at N=40 (0.53 for both paths), so the
  residual is not fully explained.

Consequently the objective assertions below are a floor plus a *paired* comparison
against the no-EP path, not an argmax test: the paired form is what ties the
assertion to the ``h_n`` mechanism.
"""

import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import RBFKernel, ScaleKernel

from bo.acquisition_functions.pesc import (
    pesc_conditional_variances, sample_constrained_optima,
    _conditional_variance, _joint_moments,
)
from bo.acquisition_functions.pesc_ep import pesc_ep_conditional_variances

dtype = torch.double
torch.set_default_dtype(dtype)

LS = 0.1        # true lengthscale
SIG2 = 0.01     # true observation-noise variance
G = 200         # grid resolution
N_RS = 200_000  # RS samples; correlations are stable from ~150k
MIN_COUNT = 300  # minimum samples per x* group for a usable empirical variance


def _prior_samples(seed, n):
    """``n`` independent zero-mean GP prior draws on the grid (the ground truth)."""
    torch.manual_seed(seed)
    grid = torch.linspace(0, 1, G, dtype=dtype).unsqueeze(-1)
    K = torch.exp(-0.5 * (grid - grid.T).pow(2) / LS ** 2)
    L = torch.linalg.cholesky(K + 1e-8 * torch.eye(G, dtype=dtype))
    return grid, [L @ torch.randn(G, dtype=dtype) for _ in range(n)]


def _fixed_hyper_gp(train_X, train_Y):
    """GP pinned to the true hyperparameters, as the paper's Sec. 4.1 test uses."""
    gp = SingleTaskGP(
        train_X, train_Y,
        train_Yvar=torch.full_like(train_Y, SIG2),
        covar_module=ScaleKernel(RBFKernel()),
    )
    gp.covar_module.base_kernel.lengthscale = LS
    gp.covar_module.outputscale = 1.0
    gp.mean_module.constant.data.fill_(0.0)
    gp.eval()
    return gp


def _build(seed, n_init=5):
    grid, (f_true, c_true) = _prior_samples(seed, 2)
    torch.manual_seed(seed + 1000)
    idx = torch.randperm(G)[:n_init]
    sd = SIG2 ** 0.5
    ys = [(t[idx] + sd * torch.randn(n_init, dtype=dtype)).unsqueeze(-1)
          for t in (f_true, c_true)]
    model = ModelListGP(*[_fixed_hyper_gp(grid[idx], y) for y in ys])
    bounds = torch.stack([torch.zeros(1, dtype=dtype), torch.ones(1, dtype=dtype)])
    return grid, model, bounds


def _alpha_rs(model, grid, n_samples=N_RS, chunk=20_000, rs_seed=12345):
    """Per-source alpha by rejection sampling, following Sec. 4.1.

    Samples f and c jointly on the grid, takes the feasible best cell as x*, then
    groups samples by their x* and uses each group's empirical marginal variances
    (plus noise) as the conditioned entropy -- i.e. rejection sampling with the
    grid cells as the x* bins.
    """
    # Seeded: with ~120 x* bins the per-bin empirical variances carry real Monte
    # Carlo error, so an unseeded reference makes the measured correlation wobble by
    # several hundredths between runs -- enough to flip a threshold.
    torch.manual_seed(rs_seed)
    cnt = torch.zeros(G, dtype=dtype)
    s1 = [torch.zeros(G, G, dtype=dtype) for _ in range(2)]
    s2 = [torch.zeros(G, G, dtype=dtype) for _ in range(2)]
    neg_inf = torch.tensor(-float("inf"), dtype=dtype)

    with torch.no_grad():
        post = [model.models[i].posterior(grid) for i in range(2)]
        drawn = 0
        while drawn < n_samples:
            n = min(chunk, n_samples - drawn)
            drawn += n
            fs = post[0].rsample(torch.Size([n])).squeeze(-1)
            cs = post[1].rsample(torch.Size([n])).squeeze(-1)
            feas = cs <= 0
            keep = feas.any(1)
            if not bool(keep.any()):
                continue
            fs, cs, feas = fs[keep], cs[keep], feas[keep]
            star = torch.where(feas, fs, neg_inf).argmax(1)
            cnt.index_add_(0, star, torch.ones(star.shape[0], dtype=dtype))
            for j, samp in enumerate((fs, cs)):
                s1[j].index_add_(0, star, samp)
                s2[j].index_add_(0, star, samp.pow(2))

    ok = cnt >= MIN_COUNT
    assert int(ok.sum()) >= 10, "too few x* groups for a stable RS reference"
    n_g = cnt[ok].unsqueeze(-1)
    w = (cnt[ok] / cnt[ok].sum()).unsqueeze(-1)
    out = []
    for j in range(2):
        var_emp = (s2[j][ok] / n_g - (s1[j][ok] / n_g).pow(2)).clamp_min(1e-12)
        h_cpd = (w * 0.5 * (var_emp + SIG2).log()).sum(0)
        v_pd = post[j].variance.reshape(-1) + SIG2
        out.append(0.5 * v_pd.log() - h_cpd)
    return torch.stack(out, dim=-1)  # (G, 2)


def _alpha_pesc(model, grid, bounds, M=50, seed=0, faithful=False, n_grid=512):
    x_star = sample_constrained_optima(model, bounds, num_samples=M, n_grid=n_grid, seed=seed)
    fn = pesc_ep_conditional_variances if faithful else pesc_conditional_variances
    v_prior, v_cond = fn(model, grid, x_star)
    return (0.5 * (v_prior / v_cond).log()).mean(1)  # (G, 2)


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


class TestPESCAgainstRejectionSampling(BotorchTestCase):
    """Sec. 4.1 accuracy check, per source."""

    def test_constraint_term_matches_rs(self):
        # The Eq. (37) a_tilde coupling term makes this faithful; without it the
        # correlation collapses and the peak is ~20x too small.
        for seed in (0, 1):
            grid, model, bounds = _build(seed)
            rs = _alpha_rs(model, grid)[:, 1]
            pe = _alpha_pesc(model, grid, bounds, seed=seed)[:, 1]
            self.assertGreater(_corr(pe, rs), 0.90)
            gap = abs(int(pe.argmax()) - int(rs.argmax())) / G
            self.assertLess(gap, 0.10)
            # magnitude within a factor of 2 either way
            self.assertLess(float(pe.max()) / float(rs.max()), 2.0)
            self.assertGreater(float(pe.max()) / float(rs.max()), 0.5)

    def test_objective_term_needs_the_hn_factors(self):
        # The no-EP path is structurally incapable of matching RS on the objective.
        # Kept as a weak guard AND as the baseline the faithful path must beat.
        for seed in (0, 1):
            grid, model, bounds = _build(seed)
            rs = _alpha_rs(model, grid)[:, 0]
            pe = _alpha_pesc(model, grid, bounds, seed=seed)[:, 0]
            self.assertGreater(_corr(pe, rs), 0.30)

    def test_faithful_objective_matches_rs(self):
        # Acceptance criterion for the h_n factors.  Measured min over 5 seeds is
        # 0.868 (seed 2), so 0.85 is a floor with margin; the paired comparison is
        # what ties the assertion to the mechanism rather than to a magic number.
        # No argmax assertion here -- see the module docstring for why.
        for seed in (1, 2, 3):  # 2 and 3 are the hard ones; 3 is the no-EP worst case
            grid, model, bounds = _build(seed)
            rs = _alpha_rs(model, grid)[:, 0]
            no_ep = _alpha_pesc(model, grid, bounds, seed=seed)[:, 0]
            faithful = _alpha_pesc(model, grid, bounds, seed=seed, faithful=True)[:, 0]
            r_ep = _corr(faithful, rs)
            self.assertGreater(r_ep, 0.85)
            self.assertGreater(r_ep, _corr(no_ep, rs))

    def test_faithful_objective_recovers_fully_at_larger_n(self):
        # The N=5 residual is substantially the paper's own N-point restriction of
        # the domain-wide optimality product (supp. Eqs. 11-12).  With more observed
        # points the condition is enforced in more places and agreement becomes
        # near-exact -- which is the evidence that the implementation is right and
        # the approximation is what binds.
        grid, model, bounds = _build(2, n_init=40)
        rs = _alpha_rs(model, grid)[:, 0]
        faithful = _alpha_pesc(model, grid, bounds, M=200, seed=2, faithful=True)[:, 0]
        self.assertGreater(_corr(faithful, rs), 0.95)
        self.assertLess(abs(int(faithful.argmax()) - int(rs.argmax())) / G, 0.05)

    def test_faithful_constraint_does_not_regress(self):
        for seed in (0, 1):
            grid, model, bounds = _build(seed)
            rs = _alpha_rs(model, grid)[:, 1]
            no_ep = _alpha_pesc(model, grid, bounds, seed=seed)[:, 1]
            faithful = _alpha_pesc(model, grid, bounds, seed=seed, faithful=True)[:, 1]
            self.assertGreater(_corr(faithful, rs), 0.90)
            # adding h_n must not cost the constraint term anything material
            self.assertGreater(_corr(faithful, rs), _corr(no_ep, rs) - 0.05)


class TestEq37ObjectiveReducesToTruncation(BotorchTestCase):
    """With no constraints, PF == 1 exactly and Eq. (37) must collapse to the
    plain one-sided truncation of ``g = f(x) - f(x*)``.

    This pins ``beta = PF * phi(alpha) / Z`` to the Mills ratio in the ``PF = 1``
    limit, which is the algebraic identity the new weighting rests on.
    """

    def test_matches_conditional_variance_at_pf_one(self):
        torch.manual_seed(0)
        train_X = torch.rand(6, 2, dtype=dtype)
        y = -(train_X.pow(2).sum(-1, keepdim=True))
        gp = SingleTaskGP(train_X, y, train_Yvar=torch.full_like(y, 1e-6))
        gp.eval()
        model = ModelListGP(gp)  # K = 0  ->  PF = 1
        bounds = torch.stack([torch.zeros(2, dtype=dtype), torch.ones(2, dtype=dtype)])
        x_star = sample_constrained_optima(model, bounds, num_samples=8, n_grid=256, seed=0)

        X = torch.rand(15, 2, dtype=dtype)
        _, v_cond = pesc_conditional_variances(model, X, x_star, add_observation_noise=False)

        v_x, mu_x, v_star, mu_star, cross = _joint_moments(model.models[0], X, x_star)
        v_x, mu_x = v_x.unsqueeze(-1), mu_x.unsqueeze(-1)
        mu_g = mu_x - mu_star
        v_g = (v_x + v_star - 2.0 * cross).clamp_min(1e-12)
        ref = _conditional_variance(v_x, v_x - cross, mu_g, v_g, thr=0.0)

        self.assertAllClose(v_cond[..., 0], ref, atol=1e-10, rtol=0)
