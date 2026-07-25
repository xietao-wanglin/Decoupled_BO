"""Predictive Entropy Search with Constraints (PESC).

BoTorch-native, decoupled implementation of PESC (Hernandez-Lobato et al.,
2015, arXiv:1502.05312).  PESC's information gain factorises additively across
the objective and each constraint, so it yields a per-source acquisition value

    alpha_s(x) = H[p(y_s(x)|D)] - E_{x*}[ H[p(y_s(x)|D, x*)] ]

where x* is the (sampled) constrained global optimum.  For Gaussian marginals
this reduces to  alpha_s(x) = 0.5 * E_{x*}[ log( v_s(x) / v_s(x|x*) ) ].

This module implements the paper's *closed-form* per-candidate step, i.e. the
final Gaussian approximation to the noise-free CPD given by Eqs. (36)-(37) of
the supplementary material.  Two non-Gaussian factors act on each x:

* ``g_k``: feasibility of the optimum, ``c_k(x*) <= 0`` (repo convention: c <= 0
  is feasible).  A one-sided truncation of the ``c_k(x*)`` marginal, whose mean
  and variance shift propagate to ``c_k(x)`` through the cross-covariance.  This
  yields the paper's ``(m'_k, v'_k)``.
* ``Psi(x)`` / ``h``: "x does not dominate x*", i.e. NOT (x feasible AND x better
  than x*).  Its normaliser ``Z = (1 - PF) + PF * Phi(alpha)`` couples the
  objective and every constraint at the candidate, with ``alpha`` the optimality
  z-score and ``PF = prod_k Phi(-m'_k / sqrt(v'_k))``.

Both source terms then share the single reduction factor ``beta * (beta + alpha)``
of Eq. (37) -- scaled by ``Delta^2 / s`` for the objective (whose truncation acts
on the difference ``g = f(x) - f(x*)``, not on ``f(x)`` directly) and by ``v'_k``
for constraint k (whose de-weighting acts on ``c_k(x)`` directly).

The paper's remaining factors -- the N observed-point factors ``h_n`` of
supplementary Eq. (4), which require EP over an (N+1)-dimensional vector -- are
*not* included here; see ``pesc_ep.py``.  Without them the objective's incoming
moments are the plain GP posterior, which is the correct Tier-1 behaviour but
means ``x*`` is not yet constrained to beat the observed data.

Observation noise is added to the marginals at ``x`` before the entropies are
formed, as the paper requires (Sec. 3.2).  ``x*`` is a latent value and stays
noise-free, so all truncation algebra runs on latent quantities and the noise is
added only at the end.  This is also what makes the log-ratio go cleanly to 0 at
an already-observed location (``sigma^2 / sigma^2``) instead of ``0/0``.
"""

import math

import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import draw_sobol_samples
from botorch.utils.gp_sampling import get_gp_samples
from botorch.utils.probability.utils import log_ndtr as log_Phi
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_EPS = 1e-12

# Z-scores are clamped to +/-_ZMAX before any log-pdf / log-cdf.  Beyond ~8 the
# normal cdf is 0 or 1 to double precision, so this changes no value that matters,
# but it removes a genuine NaN: both -0.5*z^2 and log_Phi(z) diverge to -inf in the
# tail, and their difference in _log_mills becomes inf - inf = NaN.  That NaN
# propagates into the gradient and makes optimize_acqf fail outright.
_ZMAX = 8.0

# Diagnostics for the non-negativity clamp.  The paper's Gaussian approximation to
# the CPD can legitimately produce v_cond > v_prior (negative information gain) --
# the h factor *de-weights* rather than truncates, so conditioning can widen a
# marginal.  Clamping is load-bearing for stability, but a high rate means the
# approximation is misbehaving, so it is counted rather than hidden.
CLAMP_STATS = {"total": 0, "clamped": 0}


def reset_clamp_stats() -> None:
    CLAMP_STATS["total"] = 0
    CLAMP_STATS["clamped"] = 0


def clamped_fraction() -> float:
    total = CLAMP_STATS["total"]
    return 0.0 if total == 0 else CLAMP_STATS["clamped"] / total


def _zclamp(z: Tensor) -> Tensor:
    """Clamp a z-score into the range where log-pdf and log-cdf are both finite."""
    return z.clamp(min=-_ZMAX, max=_ZMAX)


def _std_normal_log_pdf(z: Tensor) -> Tensor:
    return -0.5 * z.pow(2) - _LOG_SQRT_2PI


def _log_mills(alpha: Tensor) -> Tensor:
    """``log(phi(alpha) / Phi(alpha))`` -- the log Mills ratio, stable in the tails."""
    return _std_normal_log_pdf(alpha) - log_Phi(alpha)


def _truncated_unit_variance(alpha: Tensor) -> Tensor:
    """Variance of a standard normal truncated to (-inf, alpha]."""
    alpha = _zclamp(alpha)
    delta = _log_mills(alpha).exp()  # phi/Phi (Mills ratio)
    var = 1.0 - alpha * delta - delta.pow(2)
    return var.clamp(min=_EPS, max=1.0)


def _conditional_variance(v_target, cov_tu, mu_u, v_u, thr=0.0):
    """Variance of a Gaussian ``T`` after conditioning on ``U <= thr``.

    ``(T, U)`` is bivariate Gaussian; ``U ~ N(mu_u, v_u)`` and ``cov_tu = Cov(T, U)``.
    Uses moment matching of the one-sided truncation ``U <= thr``.  The result is
    always ``<= v_target`` so the induced information gain is non-negative.

    Retained as the reference for the ``PF = 1`` limit of Eq. (37); the production
    path goes through :func:`pesc_conditional_variances`.
    """
    v_u = v_u.clamp_min(_EPS)
    alpha = _zclamp((thr - mu_u) / v_u.sqrt())
    var_z = _truncated_unit_variance(alpha)  # in [0, 1]
    reduction = (cov_tu.pow(2) / v_u) * (1.0 - var_z)
    return (v_target - reduction).clamp_min(_EPS)


def _truncated_moments(mu: Tensor, v: Tensor, thr: float = 0.0):
    """Mean and variance of ``N(mu, v)`` truncated to ``(-inf, thr]``."""
    s = v.clamp_min(_EPS).sqrt()
    alpha = _zclamp((thr - mu) / s)
    delta = _log_mills(alpha).exp()
    return mu - s * delta, v * _truncated_unit_variance(alpha)


def _joint_moments(gp, X: Tensor, x_star: Tensor):
    """Joint posterior moments of one GP over ``[x, x*_1, ..., x*_M]``.

    ``X`` is ``(b, d)``.  Returns ``(v_x, mu_x)`` shaped ``(b,)`` and
    ``(v_star, mu_star, cross)`` shaped ``(b, M)``, where ``cross`` is the
    covariance between the test point and each ``x*``.
    """
    b = X.shape[0]
    M = x_star.shape[0]
    star = x_star.to(X).unsqueeze(0).expand(b, M, -1)
    post = gp.posterior(torch.cat([X.unsqueeze(1), star], dim=1))
    cov = post.mvn.covariance_matrix  # (b, 1+M, 1+M)
    mean = post.mvn.mean  # (b, 1+M)
    return (
        cov[:, 0, 0].clamp_min(_EPS),  # (b,)
        mean[:, 0],  # (b,)
        cov[:, 1:, 1:].diagonal(dim1=-2, dim2=-1).clamp_min(_EPS),  # (b, M)
        mean[:, 1:],  # (b, M)
        cov[:, 0, 1:],  # (b, M)
    )


def _observation_noise(gp, X: Tensor) -> Tensor:
    """Observation-noise variance ``sigma_s^2`` of one GP at ``X``, shape ``(b,)``.

    Recovered as the gap between the noisy and latent predictive variances so the
    same code path works for fixed-noise, homoskedastic and heteroskedastic GPs.
    Treated as a constant w.r.t. ``X`` (it is, for every model used in this repo).
    """
    with torch.no_grad():
        try:
            noisy = gp.posterior(X, observation_noise=True).variance
        except (NotImplementedError, AttributeError, RuntimeError):
            return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
        latent = gp.posterior(X).variance
    return (noisy - latent).reshape(-1).clamp_min(0.0)


def _polish_optima(
    obj_path, con_paths, x0: Tensor, bounds: Tensor, n_steps: int = 60,
) -> Tensor:
    """Continuously optimise each sampled path from its best grid point.

    The paper solves a constrained optimisation on each sampled function; this does
    it with a penalised objective and a ramped penalty, which keeps every sample's
    optimum off the grid.  ``x0`` is ``(S, d)``; sample ``s`` is optimised against
    path ``s`` only, and the samples are independent so one joint Adam run suffices.

    Feeding ``(S, 1, d)`` makes the sample axis broadcast against the point axis, so
    the deterministic-model call returns path ``s`` evaluated at point ``s`` directly
    (verified equal to the diagonal of the all-paths-by-all-points evaluation).
    """
    lo, hi = bounds[0], bounds[1]
    S = x0.shape[0]
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.02)
    best_x, best_val = x0.clone(), torch.full((S,), -float("inf"), dtype=x0.dtype,
                                              device=x0.device)
    for step in range(n_steps):
        opt.zero_grad()
        f = obj_path.posterior(x.unsqueeze(1)).mean.reshape(S)
        pen = torch.zeros_like(f)
        for cp in con_paths:
            c = cp.posterior(x.unsqueeze(1)).mean.reshape(S)
            pen = pen + c.clamp_min(0.0).pow(2)
        lam = 10.0 * (1.0 + step)  # ramp so feasibility dominates by the end
        (-f + lam * pen).sum().backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(min=lo, max=hi)
            # Track the best *feasible-penalised* iterate rather than trusting the
            # last one: the ramping penalty makes the trajectory non-monotone.
            score = torch.where(pen <= 1e-10, f, torch.full_like(f, -float("inf")))
            better = score > best_val
            best_val = torch.where(better, score, best_val)
            best_x[better] = x.detach()[better]
    # Samples that never reached feasibility keep their grid start.
    never = torch.isinf(best_val)
    best_x[never] = x0[never]
    return best_x.detach()


def sample_constrained_optima(
    model: Model,
    bounds: Tensor,
    num_samples: int,
    n_grid: int = 1000,   # Spearmint pes_x*_grid_size
    seed: int = 0,
    mode: str = "spectral",
) -> Tensor:
    """Sample constrained global optima x* from the joint GP posterior.

    Draws ``num_samples`` joint posterior samples of the objective and every
    constraint over a Sobol grid, and returns, per sample, the feasible
    grid point with the largest sampled objective (the objective GP models the
    negated problem, so "largest" = best).  If a sample has no feasible grid
    point, the point closest to feasibility (smallest worst-constraint value) is
    used instead, mirroring the infeasible handling in ``_thompson_disc``.

    Two modes:

    * ``"spectral"`` (default, the paper's approach): draw a finite random-Fourier
      parameterisation of each posterior, then optimise each sampled path
      continuously under its own sampled constraints.  The Sobol grid is retained
      only to pick each sample's starting point, so x* is no longer grid-locked.
    * ``"grid"``: draw the exact joint posterior on the grid and take the feasible
      argmax.  Avoids the spectral approximation error but confines x* to grid
      points, which is coarse for larger ``d``.  Kept for A/B comparison.

    ``seed`` covers the Sobol grid *and* the sampled paths.  It previously covered
    only the grid, leaving the draws on the global RNG -- so runs were not
    reproducible for a given seed, and the spread dominated the acquisition: on the
    paper's Sec. 4.1 problem the measured objective correlation moved by ~0.11
    between draws.

    Returns:
        ``(num_samples, d)`` tensor of sampled constrained optima.
    """
    dev = bounds.device
    K = len(model.models) - 1
    X_grid = draw_sobol_samples(bounds=bounds, n=n_grid, q=1, seed=seed).squeeze(1).to(dev)

    # fork_rng keeps the stream deterministic without disturbing the caller.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)

        if mode == "spectral":
            obj_path = get_gp_samples(model=model.models[0], num_outputs=1,
                                      n_samples=num_samples, num_rff_features=1000)
            con_paths = [
                get_gp_samples(model=model.models[k + 1], num_outputs=1,
                               n_samples=num_samples, num_rff_features=1000)
                for k in range(K)
            ]
            with torch.no_grad():
                f_samp = obj_path.posterior(X_grid).mean.reshape(num_samples, n_grid)
                c_samp = [cp.posterior(X_grid).mean.reshape(num_samples, n_grid)
                          for cp in con_paths]
        else:
            with torch.no_grad():
                obj_post = model.models[0].posterior(X_grid)
                f_samp = obj_post.rsample(torch.Size([num_samples])).squeeze(-1)
                c_samp = [
                    model.models[k + 1].posterior(X_grid)
                    .rsample(torch.Size([num_samples])).squeeze(-1)
                    for k in range(K)
                ]
            obj_path = con_paths = None

    worst_c = torch.full((num_samples, n_grid), -float("inf"), device=dev, dtype=X_grid.dtype)
    feasible = torch.ones(num_samples, n_grid, dtype=torch.bool, device=dev)
    for c in c_samp:
        feasible &= c <= 0
        worst_c = torch.maximum(worst_c, c)

    neg_inf = torch.tensor(-float("inf"), device=dev, dtype=f_samp.dtype)
    masked_f = torch.where(feasible, f_samp, neg_inf)
    has_feasible = feasible.any(dim=1)
    # Where a sample has no feasible grid point, start from the point closest to
    # feasibility (smallest worst-constraint value) instead.
    best_idx = torch.where(has_feasible, masked_f.argmax(dim=1), worst_c.argmin(dim=1))
    x0 = X_grid[best_idx]

    if mode != "spectral":
        return x0
    return _polish_optima(obj_path, con_paths, x0, bounds)


def eq37_conditional_variances(v_x, mu_x, v_ss, mu_ss, cross, m_c, v_c):
    """Eqs. (36)-(37): conditioned variances of ``f(x)`` and each ``c_k(x)``.

    Takes the *already conditioned* moments as arguments so that both the no-EP
    path (raw GP moments plus the ``g_k`` truncation) and the faithful path (moments
    from the (N+1)-dim EP in ``pesc_sites.py``) share one implementation.  Leading
    batch dimensions are arbitrary; ``m_c``/``v_c`` carry a trailing ``K`` axis.

    Args:
        v_x, mu_x: moments of ``f(x)``.
        v_ss, mu_ss: moments of ``f(x*)``.
        cross: ``Cov(f(x), f(x*))``.
        m_c, v_c: moments of each ``c_k(x)`` -- the paper's ``(m'_k, v'_k)``.

    Returns:
        ``(v_cond_f, v_cond_c)`` shaped ``(...)`` and ``(..., K)``.
    """
    # g = f(x) - f(x*); x* is the best feasible point, so the event is g <= 0.
    s = (v_x + v_ss - 2.0 * cross).clamp_min(_EPS)  # V[g], the paper's s
    alpha = _zclamp((mu_ss - mu_x) / s.sqrt())  # Phi(alpha) = P(g <= 0)
    delta_f = v_x - cross  # [V'_f]_11 - [V'_f]_12
    log_Bd = log_Phi(alpha)  # log P(g <= 0)

    K = m_c.shape[-1]
    alpha_k = _zclamp(-m_c / v_c.clamp_min(_EPS).sqrt())  # Phi(alpha_k) = P(c_k(x) <= 0)
    log_C = log_Phi(alpha_k)
    # Empty product over K == 0 constraints is 1, i.e. log PF == 0.
    log_PF = log_C.sum(-1) if K > 0 else torch.zeros_like(alpha)

    # Z = (1 - PF) + PF * Phi(alpha), built in log space: when PF -> 1 the first
    # term underflows to -inf and logaddexp still recovers logZ -> log Phi(alpha),
    # which is what keeps beta finite where naive PF*phi/Z would go 0/0.
    PF = log_PF.exp().clamp(max=1.0)
    log_Z = torch.logaddexp(torch.log1p(-PF), log_PF + log_Bd)

    # ---- objective: Eq. (37) vf^NFCPD ----
    beta = (log_PF + _std_normal_log_pdf(alpha) - log_Z).exp()
    v_cond_f = v_x - (beta / s) * (beta + alpha) * delta_f.pow(2)

    # ---- constraints: Eq. (37) vk^NFCPD ----
    # a_tilde = -D / (1 + D v')  with  D = d2logZ/dm'^2 = -beta_k (alpha_k + beta_k) / v',
    # so  v_cond = v' / (1 + v' a_tilde) = v' * (1 + D v') = v' * (1 - beta_k (alpha_k + beta_k)).
    # Written this way there is no division by D, which vanishes for an inactive
    # constraint (beta_k -> 0) and would otherwise blow up.
    if K > 0:
        # (Z - 1)/Z = -PF * (1 - Phi(alpha)) / Z, so beta_k <= 0.
        log_frac = log_PF + log_Phi(-alpha) - log_Z  # log(PF * P(g > 0) / Z)
        beta_k = -(_log_mills(alpha_k) + log_frac.unsqueeze(-1)).exp()
        v_cond_c = v_c * (1.0 - beta_k * (alpha_k + beta_k))
    else:
        v_cond_c = v_c
    return v_cond_f, v_cond_c


def _finalise(v_prior, v_cond, model, X, K, add_observation_noise):
    """Add observation noise (Sec. 3.2), record clamp diagnostics, floor v_cond."""
    if add_observation_noise:
        sig2 = [_observation_noise(model.models[s_i], X) for s_i in range(K + 1)]
        sig2 = torch.stack(sig2, dim=-1).reshape(X.shape[0], 1, K + 1)  # (b, 1, K+1)
        v_prior = v_prior + sig2
        v_cond = v_cond + sig2

    CLAMP_STATS["total"] += v_cond.numel()
    CLAMP_STATS["clamped"] += int((v_cond > v_prior).sum())
    return v_prior, torch.minimum(v_cond.clamp_min(_EPS), v_prior)


def pesc_conditional_variances(
    model: Model,
    X: Tensor,
    x_star: Tensor,
    add_observation_noise: bool = True,
):
    """Eq. (36)-(37) without the observed-point EP (the ``h_n`` factors).

    Conditions only on ``g_k`` (feasibility of x*) and the candidate-x ``h`` factor.
    Retained as the cheap, differentiable variant and as the A/B baseline against
    :func:`pesc_sites.pesc_faithful_conditional_variances`.

    Args:
        X: ``(b, d)`` test points.
        x_star: ``(M, d)`` sampled constrained optima.
        add_observation_noise: add ``sigma_s^2`` to both marginals at ``x``, as
            Sec. 3.2 requires before the entropies are formed.

    Returns:
        ``(v_prior, v_cond)``, each ``(b, M, K+1)`` -- the predictive and
        conditioned variances of ``value_s(x)`` (index 0 = objective,
        ``1..K`` = constraints).
    """
    K = len(model.models) - 1

    v_f, mu_f, v_fstar, mu_fstar, cross_f = _joint_moments(model.models[0], X, x_star)
    v_f = v_f.unsqueeze(-1)  # (b, 1) -> broadcasts over M
    mu_f = mu_f.unsqueeze(-1)

    # ---- constraint blocks: apply g_k, keeping BOTH conditioned moments ----
    # The mean shift matters: alpha_k in the core is a function of m'_k, so ignoring
    # it (as a variance-only treatment would) mis-scales the whole coupling.
    v_prior_c, m_c, v_c = [], [], []
    for k in range(K):
        v_x, mu_x, v_st, mu_st, cr = _joint_moments(model.models[k + 1], X, x_star)
        m_t, v_t = _truncated_moments(mu_st, v_st, thr=0.0)  # c_k(x*) <= 0
        reg = cr / v_st  # regression coefficient onto c_k(x*)
        v_prior_c.append(v_x.unsqueeze(-1).expand_as(cr))
        m_c.append(mu_x.unsqueeze(-1) + reg * (m_t - mu_st))
        v_c.append((v_x.unsqueeze(-1) + reg.pow(2) * (v_t - v_st)).clamp_min(_EPS))

    zeros = torch.zeros(*cross_f.shape, 0, dtype=X.dtype, device=X.device)
    m_ck = torch.stack(m_c, dim=-1) if K else zeros
    v_ck = torch.stack(v_c, dim=-1) if K else zeros

    v_cond_f, v_cond_c = eq37_conditional_variances(
        v_x=v_f, mu_x=mu_f, v_ss=v_fstar, mu_ss=mu_fstar, cross=cross_f,
        m_c=m_ck, v_c=v_ck,
    )

    if K > 0:
        v_prior = torch.cat([v_f.expand_as(v_cond_f).unsqueeze(-1),
                             torch.stack(v_prior_c, dim=-1)], dim=-1)
        v_cond = torch.cat([v_cond_f.unsqueeze(-1), v_cond_c], dim=-1)
    else:
        v_prior = v_f.expand_as(v_cond_f).unsqueeze(-1)
        v_cond = v_cond_f.unsqueeze(-1)

    return _finalise(v_prior, v_cond, model, X, K, add_observation_noise)


class _PESCSource(AnalyticAcquisitionFunction):
    """Base for a single PESC source term.

    ``source_model_index`` selects which output's information gain to return
    (0 = objective, ``k+1`` = constraint k).  Subclasses need only supply that
    index; the conditioning itself is shared, because the objective and every
    constraint term depend on the *same* normaliser ``Z`` and so cannot be
    computed independently without recomputing every posterior.
    """

    def __init__(self, model: Model, source_model_index: int, x_star: Tensor) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.posterior_transform = None
        self.source_model_index = source_model_index
        self.register_buffer("x_star", x_star.detach())

    def _joint(self, X: Tensor):
        """Posterior of the source GP over ``[x, x*_1, ..., x*_M]``."""
        return _joint_moments(self.model.models[self.source_model_index],
                              X.squeeze(1), self.x_star)

    def _conditional_variances(self, X: Tensor):
        return pesc_conditional_variances(self.model, X.squeeze(1), self.x_star)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        v_prior, v_cond = self._conditional_variances(X)
        s = self.source_model_index
        info = 0.5 * (v_prior[..., s] / v_cond[..., s]).log()  # (b, M)
        info = torch.nan_to_num(info, nan=0.0, posinf=0.0, neginf=0.0)
        return info.mean(dim=-1).clamp_min(0.0)


class PESCObjective(_PESCSource):
    """PESC objective source term (source 0), Eq. (37) ``vf^NFCPD``."""

    def __init__(self, model: Model, x_star: Tensor, maximize: bool = True) -> None:
        super().__init__(model, source_model_index=0, x_star=x_star)
        if not maximize:
            raise NotImplementedError(
                "PESC assumes the objective GP models the negated (maximised) problem"
            )
        self.maximize = maximize
        self.K = len(model.models) - 1


class PESCConstraint(_PESCSource):
    """PESC term for constraint ``constraint_index`` (0-based), Eq. (37) ``vk^NFCPD``."""

    def __init__(self, model: Model, constraint_index: int, x_star: Tensor) -> None:
        super().__init__(model, source_model_index=constraint_index + 1, x_star=x_star)
        self.constraint_index = constraint_index
