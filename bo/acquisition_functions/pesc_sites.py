"""Expectation propagation over the (N+1)-dimensional PESC vector.

This is the stage the paper runs **once per sampled optimum x\\***, whose result is
then reused for every candidate x (Hernandez-Lobato et al. 2015, supplementary
Sections 2-4).  It supplies the conditioned moments ``(m'_f, V'_f, m'_k, v'_k)``
that the closed-form Eqs. (36)-(37) in ``pesc.py`` consume.

Two families of non-Gaussian factors act on the vector
``v_s = [value_s(x*), value_s(u_1), ..., value_s(u_N)]``:

* ``g_k`` (supp. Eq. 5): feasibility of the optimum, ``c_k(x*) <= 0``.  A one-sided
  truncation on coordinate 0 of constraint k's vector.
* ``h_n`` (supp. Eq. 4): "``u_n`` does not dominate x*", i.e. NOT (``u_n`` feasible
  AND ``f(u_n)`` better than ``f(x*)``).  One factor per evaluated location.

**Every site is one-dimensional.** ``h_n`` depends on ``(f_n, f_0)`` only through
``d_n = f(u_n) - f(x*)``, so the paper's 2x2 ``A_hn`` is rank-1 along ``[1, -1]``.
Summing ``sum_n tau_n p_n p_n^T`` with ``p_n = e_n - e_0`` reproduces exactly the
arrowhead ``V~_f`` the supplement describes (diagonal ``tau_n``, first row/column
``-tau_n``, corner ``sum_n tau_n``) -- see ``test_pesc_sites.py``.

**Numerics.** The paper's parameterisation needs ``V_pred^-1``, and its Eq. (34)
needs the prior ``K^-1``.  Both are unusable here: the GP posterior covariance over
``[x*, u_1..u_N]`` has a *negative* minimum eigenvalue and fails Cholesky at N=160,
d=2 (cond ~1e294), which is the normal operating point for this repo (budget 160,
Mystery/TestFunc3 are d=2).  Everything below therefore goes through
:func:`site_solve`, which inverts only ``(I + T A)`` -- measured cond <= 1.2e2 at
that same configuration.  ``V`` appears exclusively inside products, so its
float-level indefiniteness is harmless.

**Decoupled generalisation.** In the decoupled setting each source has its own
design.  ``h_n`` is a statement about a *location* ("u_n does not dominate x*"), so
the index set is the **union** of all evaluated locations; where a source was never
evaluated at ``u_n`` its value there is simply a latent GP variable.  This is the
faithful reading of supp. Eq. (4) when the datasets differ per output.
"""

from typing import List, Optional

import torch
from botorch.models.model import Model
from botorch.utils.probability.utils import log_ndtr as log_Phi
from torch import Tensor

from bo.acquisition_functions.pesc import (
    _EPS,
    _log_mills,
    _std_normal_log_pdf,
    _truncated_moments,
    _zclamp,
)

# Paper Sec. 5.1-5.2: sites start at zero, damping starts at 1 and decays by 0.99
# per sweep, convergence is max |change in site parameters| < 1e-4.
_DAMP_DECAY = 0.99
_TOL = 1e-4
_MAX_SWEEPS = 100

# Convergence diagnostics from the most recent call to
# pesc_faithful_conditional_variances; read by tests and by the loop's logging.
EP_STATS = {"sweeps": [], "converged": []}


def _phi(z: Tensor) -> Tensor:
    return _std_normal_log_pdf(z).exp()


def _Phi(z: Tensor) -> Tensor:
    return log_Phi(z).exp()


def site_solve(A: Tensor, b: Tensor, tau: Tensor, nu: Tensor):
    """Projected marginals of ``N(mu, V)`` times ``exp(-.5 (Sv)'T(Sv) + (Sv)'nu)``.

    Works purely in the projected space: ``A = S V S'`` and ``b = S mu``, both of
    which are constant across EP sweeps and so are computed once.

    Uses ``(T^-1 + A)^-1 = (I + T A)^-1 T``, so there is no ``V^-1``, no ``T^-1``
    and no ``sqrt(T)``.  Zero and negative site precisions are therefore both fine,
    which matters because EP produces them routinely.

    Returns:
        ``(m_proj, cov_proj, G, bracket)`` -- the projected mean and *full* projected
        covariance ``S Sigma S' = A - A G A``, plus ``G = (I + T A)^-1 T``
        (symmetric) and ``bracket = nu - G (b + A nu)``.  The latter two are what
        the per-candidate step needs, and neither depends on the candidate, which is
        what makes the reuse across x possible.

    The full covariance is returned rather than just its diagonal because a
    positive diagonal does *not* imply the tilted joint is proper: a site negative
    enough to make ``A^-1 + T`` indefinite can still leave every projected variance
    positive and below the prior.  Only an eigenvalue check catches that, and the
    matrix is already formed here, so returning it is free.
    """
    m = A.shape[0]
    eye = torch.eye(m, dtype=A.dtype, device=A.device)
    G = torch.linalg.solve(eye + tau.unsqueeze(-1) * A, torch.diag(tau))
    G = 0.5 * (G + G.transpose(-1, -2))  # symmetric in exact arithmetic
    bracket = nu - G @ (b + A @ nu)
    m_proj = b + A @ bracket
    cov_proj = A - A @ G @ A
    cov_proj = 0.5 * (cov_proj + cov_proj.transpose(-1, -2))
    return m_proj, cov_proj, G, bracket


def is_proper(cov_proj: Tensor) -> bool:
    """Whether the tilted joint is proper, i.e. ``S Sigma S' = (A^-1 + T)^-1`` is PD.

    This is the check the paper describes in Sec. 5.2 ("some covariance matrices may
    become non positive definite due to an excessively large step size"), and the
    trigger for halving the damping and repeating the sweep.
    """
    try:
        return bool(torch.linalg.eigvalsh(cov_proj).min() > 0)
    except (torch._C._LinAlgError, RuntimeError):
        return False


def _cavity(m: Tensor, v: Tensor, tau: Tensor, nu: Tensor):
    """Remove a 1-D site from its own projected marginal."""
    prec = (1.0 / v.clamp_min(_EPS)) - tau
    prec = torch.where(prec.abs() < _EPS, torch.full_like(prec, _EPS), prec)
    v_cav = 1.0 / prec
    m_cav = v_cav * (m / v.clamp_min(_EPS) - nu)
    return m_cav, v_cav


def _site_from_moments(m_t: Tensor, v_t: Tensor, m_cav: Tensor, v_cav: Tensor):
    """Site natural parameters matching tilted moments against a cavity."""
    v_t = v_t.clamp_min(_EPS)
    tau = 1.0 / v_t - 1.0 / v_cav
    nu = m_t / v_t - m_cav / v_cav
    return tau, nu


def h_factor_tilted_moments(m_d, v_d, m_c, v_c):
    """Tilted moments of the ``h`` factor ``1 - PF * 1[d > 0]``.

    The same factor acts at the candidate x (where ``pesc.py`` handles it in closed
    form via Eq. 37) and at every evaluated location ``u_n`` (where it is an EP
    site).  Shapes are arbitrary leading batch ``(...)``; ``m_c``/``v_c`` carry a
    trailing constraint axis ``(..., K)``.  ``K = 0`` is allowed (``PF == 1``).

    Args:
        m_d, v_d: moments of ``d = value(p) - value(x*)`` for the objective.
        m_c, v_c: moments of each ``c_k(p)``.

    Returns:
        ``(m_d_t, v_d_t, m_c_t, v_c_t)`` -- tilted moments, same shapes in.
    """
    s_d = v_d.clamp_min(_EPS).sqrt()
    alpha = _zclamp(-m_d / s_d)  # Phi(alpha) = P(d <= 0)
    log_Bd = log_Phi(alpha)

    if m_c is not None and m_c.shape[-1] > 0:
        s_c = v_c.clamp_min(_EPS).sqrt()
        gamma = _zclamp(-m_c / s_c)  # Phi(gamma) = P(c_k <= 0)
        log_C = log_Phi(gamma)
        log_PF = log_C.sum(-1)
    else:
        log_C = None
        log_PF = torch.zeros_like(m_d)

    PF = log_PF.exp().clamp(max=1.0)
    # Z = (1 - PF) + PF * Phi(alpha), in log space so that PF -> 1 stays finite.
    log_Z = torch.logaddexp(torch.log1p(-PF), log_PF + log_Bd)
    Z = log_Z.exp().clamp_min(_EPS)

    # --- objective d-site: tilted ~ N(d) * (1 - PF * 1[d > 0]) ---
    beta = _zclamp(m_d / s_d)
    Phi_up, phi_up = _Phi(beta), _phi(beta)
    E1 = m_d * Phi_up + s_d * phi_up  # E[d 1(d>0)]
    E2 = (m_d.pow(2) + v_d) * Phi_up + s_d * m_d * phi_up  # E[d^2 1(d>0)]
    m_d_t = (m_d - PF * E1) / Z
    v_d_t = ((m_d.pow(2) + v_d - PF * E2) / Z - m_d_t.pow(2)).clamp_min(_EPS)

    if log_C is None:
        return m_d_t, v_d_t, None, None

    # --- constraint c_k-sites: tilted ~ N(c) * (1 - w_k * 1[c <= 0]) ---
    # w_k = (PF / C_k) * P(d > 0); the normaliser integrates back to the shared Z.
    log_w = (log_PF.unsqueeze(-1) - log_C) + log_Phi(-alpha).unsqueeze(-1)
    w = log_w.exp()
    C = log_C.exp()
    phi_c = _phi(gamma)
    Ec1 = m_c * C - s_c * phi_c  # E[c 1(c<=0)]
    Ec2 = (m_c.pow(2) + v_c) * C - s_c * m_c * phi_c
    Zc = Z.unsqueeze(-1)
    m_c_t = (m_c - w * Ec1) / Zc
    v_c_t = ((m_c.pow(2) + v_c - w * Ec2) / Zc - m_c_t.pow(2)).clamp_min(_EPS)
    return m_d_t, v_d_t, m_c_t, v_c_t


def _union_locations(model: Model) -> Tensor:
    """Union of every evaluated location across sources (see module docstring)."""
    parts = [m.train_inputs[0] for m in model.models]
    allX = torch.cat(parts, dim=0)
    # Shared initial designs are bitwise identical across sources, so exact unique
    # collapses them; genuinely distinct points are kept.
    return torch.unique(allX, dim=0)


class PESCVectorEP:
    """EP over the PESC vector for a single sampled optimum ``x*``.

    Usage::

        ep = PESCVectorEP(model, x_star_row)
        ep.run()
        m_f, V_f, m_c, v_c = ep.candidate_moments(X)
    """

    def __init__(self, model: Model, x_star: Tensor, locations: Optional[Tensor] = None):
        self.model = model
        self.K = len(model.models) - 1
        self.x_star = x_star.reshape(1, -1)
        U = _union_locations(model) if locations is None else locations
        self.U = U
        self.N = U.shape[0]
        vec = torch.cat([self.x_star.to(U), U], dim=0)  # [x*, u_1..u_N]
        self.vec = vec

        # Projections. Objective sites act on d_n = f(u_n) - f(x*)  ->  p_n = e_n - e_0.
        # Constraint sites act on coordinates directly (index 0 = g_k, 1..N = h_n).
        n1 = self.N + 1
        P = torch.zeros(self.N, n1, dtype=U.dtype, device=U.device)
        P[:, 0] = -1.0
        P[torch.arange(self.N), torch.arange(1, n1)] = 1.0
        self.P = P

        with torch.no_grad():
            post_f = model.models[0].posterior(vec)
            self.V_f = post_f.mvn.covariance_matrix
            self.mu_f = post_f.mvn.mean.reshape(-1)
            self.V_c, self.mu_c = [], []
            for k in range(self.K):
                p = model.models[k + 1].posterior(vec)
                self.V_c.append(p.mvn.covariance_matrix)
                self.mu_c.append(p.mvn.mean.reshape(-1))

        # Constant across sweeps, so built once (this is the bulk of the algebra).
        self.A_f = P @ self.V_f @ P.T
        self.b_f = P @ self.mu_f
        self.A_c = list(self.V_c)
        self.b_c = list(self.mu_c)

        z = lambda n: torch.zeros(n, dtype=U.dtype, device=U.device)
        self.tau_f, self.nu_f = z(self.N), z(self.N)          # objective d_n sites
        self.tau_c = [z(n1) for _ in range(self.K)]           # [g_k, h_1..h_N]
        self.nu_c = [z(n1) for _ in range(self.K)]
        self.sweeps = 0
        self.converged = False
        self._cache = None

    # ------------------------------------------------------------------ EP driver

    def _marginals(self):
        m_f, cov_f, G_f, br_f = site_solve(self.A_f, self.b_f, self.tau_f, self.nu_f)
        mc, cc, Gc, brc = [], [], [], []
        for k in range(self.K):
            a, c, g, r = site_solve(self.A_c[k], self.b_c[k], self.tau_c[k], self.nu_c[k])
            mc.append(a); cc.append(c); Gc.append(g); brc.append(r)
        return (m_f, cov_f, G_f, br_f), (mc, cc, Gc, brc)

    def _all_proper(self, cov_f, cov_c) -> bool:
        return is_proper(cov_f) and all(is_proper(c) for c in cov_c)

    def run(self, max_sweeps: int = _MAX_SWEEPS, tol: float = _TOL):
        """Run damped parallel EP to convergence (paper Sec. 5.1-5.2)."""
        damp = 1.0
        prev = None
        for sweep in range(max_sweeps):
            state = (self.tau_f.clone(), self.nu_f.clone(),
                     [t.clone() for t in self.tau_c], [n.clone() for n in self.nu_c])
            try:
                delta = self._sweep(damp)
            except (torch._C._LinAlgError, RuntimeError):
                delta = None
            if delta is None or not torch.isfinite(torch.as_tensor(delta)):
                # Non-PD / breakdown: restore, halve the step, retry (paper Sec. 5.2).
                self.tau_f, self.nu_f, self.tau_c, self.nu_c = state
                damp *= 0.5
                if damp < 1e-6:
                    break
                continue
            self.sweeps = sweep + 1
            if delta < tol:
                self.converged = True
                break
            # The paper's schedule (eps=1 decaying by 0.99) is too aggressive once
            # several constraint blocks are coupled: measured 2/10 x* samples
            # converging within 100 sweeps on TestFunc3 (K=3), versus 10/10 on
            # Mystery (K=1).  Halving on a non-decreasing residual is the same
            # remedy Sec. 5.2 applies to non-PD covariances, extended to oscillation.
            if prev is not None and delta > prev:
                damp *= 0.5
            else:
                damp *= _DAMP_DECAY
            prev = delta
        self._cache = self._marginals()
        return self

    def _sweep(self, damp: float) -> float:
        (m_d, cov_d, _, _), (mc, cov_c, _, _) = self._marginals()
        v_d = cov_d.diagonal().clamp_min(_EPS)
        vc = [c.diagonal().clamp_min(_EPS) for c in cov_c]

        # --- h_n factors: cavities on d_n and on each c_k(u_n) ---
        md_cav, vd_cav = _cavity(m_d, v_d, self.tau_f, self.nu_f)
        mc_cav, vc_cav = [], []
        for k in range(self.K):
            a, b = _cavity(mc[k][1:], vc[k][1:], self.tau_c[k][1:], self.nu_c[k][1:])
            mc_cav.append(a); vc_cav.append(b)
        if self.K:
            m_c_st = torch.stack(mc_cav, dim=-1)  # (N, K)
            v_c_st = torch.stack(vc_cav, dim=-1)
        else:
            m_c_st = v_c_st = None

        md_t, vd_t, mc_t, vc_t = h_factor_tilted_moments(md_cav, vd_cav, m_c_st, v_c_st)
        tau_f_new, nu_f_new = _site_from_moments(md_t, vd_t, md_cav, vd_cav)

        # --- g_k factor: truncate c_k(x*) <= 0 on coordinate 0 ---
        tau_c_new, nu_c_new = [], []
        for k in range(self.K):
            m0_cav, v0_cav = _cavity(mc[k][:1], vc[k][:1],
                                     self.tau_c[k][:1], self.nu_c[k][:1])
            m0_t, v0_t = _truncated_moments(m0_cav, v0_cav, thr=0.0)
            t0, n0 = _site_from_moments(m0_t, v0_t, m0_cav, v0_cav)
            tk, nk = _site_from_moments(mc_t[:, k], vc_t[:, k], mc_cav[k], vc_cav[k])
            tau_c_new.append(torch.cat([t0, tk]))
            nu_c_new.append(torch.cat([n0, nk]))

        # --- damped parallel application (paper Sec. 5.2) ---
        deltas = []
        for cur, new in ((self.tau_f, tau_f_new), (self.nu_f, nu_f_new)):
            deltas.append(float((damp * (new - cur)).abs().max()))
        self.tau_f = (1 - damp) * self.tau_f + damp * tau_f_new
        self.nu_f = (1 - damp) * self.nu_f + damp * nu_f_new
        for k in range(self.K):
            deltas.append(float((damp * (tau_c_new[k] - self.tau_c[k])).abs().max()))
            deltas.append(float((damp * (nu_c_new[k] - self.nu_c[k])).abs().max()))
            self.tau_c[k] = (1 - damp) * self.tau_c[k] + damp * tau_c_new[k]
            self.nu_c[k] = (1 - damp) * self.nu_c[k] + damp * nu_c_new[k]

        # Reject the step if it left any tilted joint improper (paper Sec. 5.2).
        (_, cov_f_chk, _, _), (_, cov_c_chk, _, _) = self._marginals()
        if not self._all_proper(cov_f_chk, cov_c_chk):
            return None
        return max(deltas)

    # --------------------------------------------------------- per-candidate step

    def candidate_moments(self, X: Tensor):
        """Conditioned moments at candidates ``X`` ``(b, d)``.

        Replaces the paper's Eq. (34) with an equivalent form that avoids the prior
        ``K^-1``.  ``G`` and ``bracket`` come from the converged EP and do not
        depend on ``X``, so each candidate costs ``O(N^2)`` -- the reuse property
        that makes gradient-based optimisation of the acquisition viable.

        Returns:
            ``m_f`` ``(b, 2)`` and ``V_f`` ``(b, 2, 2)`` for ``(f(x), f(x*))``, plus
            ``m_c`` and ``v_c`` each ``(b, K)`` for ``c_k(x)``.
        """
        if self._cache is None:
            # Not run() -- with sites still at zero this must reproduce the plain GP
            # joint moments exactly, which is the zero-site equivalence guard.
            self._cache = self._marginals()
        (_, _, G_f, br_f), (_, _, Gc, brc) = self._cache
        b = X.shape[0]
        n1 = self.N + 1
        e0 = torch.zeros(n1, dtype=X.dtype, device=X.device)
        e0[0] = 1.0

        # --- objective: need Sigma_xx, Sigma_x,v P', and the x* row ---
        # NOT under no_grad: the converged sites are constants w.r.t. X, but the
        # candidate posterior is not, and the acquisition has to stay differentiable
        # for optimize_acqf to work (that is the point of the reuse property).
        #
        # Dense joint, evaluated in bounded chunks by the caller.  Lazy slicing was
        # tried and is a pessimisation here: `lazy[:b, b:].to_dense()` does not push
        # the slice into the factors of the MatmulLinearOperator that the Standardize
        # outcome transform produces, so it expands instead -- measured 272 s of pure
        # torch.matmul across one 8-iteration run.  `lazy[:b,:b].diagonal()` is worse
        # still, routing through `_expand_batch`/`repeat` and requesting ~1 TiB at
        # b=4096.  The dense block is O(b^2) so the batch must stay chunked.
        Z = torch.cat([X, self.vec.to(X)], dim=0)
        post = self.model.models[0].posterior(Z)
        cov = post.mvn.covariance_matrix
        mean = post.mvn.mean.reshape(-1)
        s_xx = cov[:b, :b].diagonal()
        s_xv = cov[:b, b:]  # (b, N+1)
        mu_x = mean[:b]
        u = s_xv @ self.P.T  # (b, N) -- Cov(f(x), d_n)
        Vp_e0 = self.P @ (self.V_f @ e0)  # (N,) -- Cov(d_n, f(x*))
        v_xx = s_xx - ((u @ G_f) * u).sum(-1)
        v_x_star = cov[:b, b] - (u @ G_f) @ Vp_e0  # column b of Z is x*
        v_ss = self.V_f[0, 0] - Vp_e0 @ G_f @ Vp_e0
        m_x = mu_x + u @ br_f
        m_ss = self.mu_f[0] + Vp_e0 @ br_f

        m_f = torch.stack([m_x, m_ss.expand_as(m_x)], dim=-1)  # (b, 2)
        V_f = torch.stack([
            torch.stack([v_xx, v_x_star], dim=-1),
            torch.stack([v_x_star, v_ss.expand_as(v_xx)], dim=-1),
        ], dim=-2)  # (b, 2, 2)

        # --- constraints: sites act on coordinates, so S = I and u = Sigma_x,v ---
        m_c, v_c = [], []
        for k in range(self.K):
            pk = self.model.models[k + 1].posterior(Z)
            ck = pk.mvn.covariance_matrix
            mk = pk.mvn.mean.reshape(-1)
            uk = ck[:b, b:]  # (b, N+1)
            v_c.append((ck[:b, :b].diagonal() - ((uk @ Gc[k]) * uk).sum(-1)).clamp_min(_EPS))
            m_c.append(mk[:b] + uk @ brc[k])
        m_c = torch.stack(m_c, dim=-1) if self.K else torch.zeros(b, 0, dtype=X.dtype)
        v_c = torch.stack(v_c, dim=-1) if self.K else torch.zeros(b, 0, dtype=X.dtype)
        return m_f, V_f, m_c, v_c

    # ------------------------------------------------------------------ diagnostics

    def site_precision_matrix(self) -> Tensor:
        """``V~_f = sum_n tau_n p_n p_n'`` -- the paper's arrowhead, for testing."""
        return self.P.T @ torch.diag(self.tau_f) @ self.P


class PESCFaithfulConditioner:
    """Holds one converged EP per sampled optimum and reuses it across candidates.

    The whole point of the paper's structure is that EP runs once per x* and its
    result is reused for every x.  Rebuilding it inside each acquisition call throws
    that away and makes ``optimize_acqf`` cost one EP per line-search evaluation, so
    the acquisition functions hold an instance of this instead.
    """

    def __init__(self, model: Model, x_star: Tensor, run_ep: bool = True,
                 max_sweeps: int = _MAX_SWEEPS, locations: Optional[Tensor] = None):
        self.model = model
        self.K = len(model.models) - 1
        self.eps = []
        for m in range(x_star.shape[0]):
            ep = PESCVectorEP(model, x_star[m], locations=locations)
            if run_ep:
                ep.run(max_sweeps=max_sweeps)
            self.eps.append(ep)
        EP_STATS["sweeps"] = [e.sweeps for e in self.eps]
        EP_STATS["converged"] = [bool(e.converged) for e in self.eps]

    def conditional_variances(self, X: Tensor, add_observation_noise: bool = True):
        from bo.acquisition_functions.pesc import (
            _finalise, eq37_conditional_variances,
        )
        v_pd = torch.stack(
            [self.model.models[i].posterior(X).variance.reshape(-1)
             for i in range(self.K + 1)], dim=-1,
        )  # (b, K+1) -- v^PD, the unconditioned predictive variance (paper Eq. 8)

        cond_f, cond_c = [], []
        for ep in self.eps:
            m_f, V_f, m_c, v_c = ep.candidate_moments(X)
            vf, vc = eq37_conditional_variances(
                v_x=V_f[..., 0, 0], mu_x=m_f[..., 0],
                v_ss=V_f[..., 1, 1], mu_ss=m_f[..., 1],
                cross=V_f[..., 0, 1], m_c=m_c, v_c=v_c,
            )
            cond_f.append(vf)
            cond_c.append(vc)

        v_cond = torch.stack(cond_f, dim=1).unsqueeze(-1)
        if self.K:
            v_cond = torch.cat([v_cond, torch.stack(cond_c, dim=1)], dim=-1)
        v_prior = v_pd.unsqueeze(1).expand_as(v_cond)
        return _finalise(v_prior, v_cond, self.model, X, self.K, add_observation_noise)


def pesc_faithful_conditional_variances(
    model: Model,
    X: Tensor,
    x_star: Tensor,
    add_observation_noise: bool = True,
    run_ep: bool = True,
    max_sweeps: int = _MAX_SWEEPS,
    locations: Optional[Tensor] = None,
):
    """PESC with the full factor set: ``g_k``, ``h_n`` and the candidate-x ``h``.

    Runs the (N+1)-dim EP once per sampled optimum, then feeds the conditioned
    moments into the closed-form Eqs. (36)-(37).  Signature matches
    :func:`pesc.pesc_conditional_variances` so the two are drop-in swappable.

    Args:
        run_ep: if False the sites stay at zero, which makes this reduce to the
            plain GP conditioning -- used by the zero-site equivalence test.

    Returns:
        ``(v_prior, v_cond)`` each ``(b, M, K+1)``.
    """
    conditioner = PESCFaithfulConditioner(
        model, x_star, run_ep=run_ep, max_sweeps=max_sweeps, locations=locations,
    )
    return conditioner.conditional_variances(
        X, add_observation_noise=add_observation_noise
    )
