"""
Fast cKG and dcKG via the reparametrisation trick (Section 3.5, Ungredda & Branke 2024).

Key equations implemented:
  - dcKG^0 (objective source): KG_d via KGCB — PF unchanged, linear in Z_y
  - dcKG^k (constraint source k, Eq. 8):
        dcKG^k(x) = 1/(b_k n_c) Σ_i [ G*^{m+1}(Z_k^i) − G^{m+1}(x_r; Z_k^i) ]
    Each Z_k realisation produces an updated G^{m+1} landscape which is
    OPTIMISED (via gen_candidates_torch) to find G*^{m+1}.
  - cKG (coupled, Eq. 7): outer MC over Z_c, inner KG_d over Z_y via KGCB.

Reparametrisation (Section 3.5):
  μ^{m+1}(x') = μ^m(x') + σ̃(x', x_{m+1}) Z
  σ^{m+1}(x') = σ^m(x')² − σ̃(x', x_{m+1})²
  σ̃(x', x)   = κ^m(x', x) / √(σ^m(x, x) + σ²_noise)
"""

from typing import Optional

import gpytorch
import numpy as np
import torch

try:
    import numba

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
from botorch.acquisition import (
    MCAcquisitionFunction,
    MCAcquisitionObjective,
    DecoupledAcquisitionFunction,
)
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.sampling import MCSampler
from botorch.utils import draw_sobol_normal_samples
from torch import Tensor

_NORMAL = torch.distributions.Normal(0, 1)


def _log_normal_cdf(x):
    """Numerically stable log Φ(x).

    Uses log(erfc(-x/√2)/2) for x < -5 (avoids underflow in the left tail),
    and log(Φ(x)) with clamping elsewhere.
    """
    safe = x > -5
    out = torch.empty_like(x)
    # Standard path where Φ(x) is not dangerously small
    out[safe] = _NORMAL.cdf(x[safe]).clamp_min(1e-30).log()
    # Left tail: log erfc(-x/sqrt(2)) - log(2)
    out[~safe] = torch.erfc(-x[~safe] / 1.4142135623730951).clamp_min(1e-300).log() - 0.6931471805599453
    return out


# ──────────────────────────────────────────────────────────────────────
# KGCB epigraph (Scott et al. 2011)
# ──────────────────────────────────────────────────────────────────────

def _filter_a_b(a, b, threshold=1e-16):
    _, a_sort = torch.sort(-a)
    sorted_idx = torch.argsort(b[a_sort])
    final_idx = a_sort[sorted_idx]
    sa, sb = a[final_idx], b[final_idx]
    if len(sa) <= 1:
        return sa, sb
    keep = torch.cat([torch.tensor([True], device=a.device),
                      (sb[1:] - sb[:-1]).abs() > threshold])
    return sa[keep], sb[keep]


def _cpu_epigraph_sweep(a_cpu, b_cpu):
    """Epigraph sweep on CPU. Returns (indices, breakpoints)."""
    n = len(a_cpu)
    if n == 0:
        return torch.zeros(0, dtype=torch.long), torch.tensor([-torch.inf, torch.inf])
    idz = torch.zeros(n, dtype=torch.long)
    x_vals = torch.full((n + 1,), -torch.inf)
    idz[0] = 0
    count = 1
    i_last = 0
    while i_last < n - 1:
        rest = torch.arange(i_last + 1, n)
        x_cross = -(a_cpu[i_last] - a_cpu[rest]) / (b_cpu[i_last] - b_cpu[rest])
        best_pos = torch.argmin(x_cross)
        i_next = rest[best_pos].item()
        idz[count] = i_next
        x_vals[count] = x_cross[best_pos]
        count += 1
        i_last = i_next
    idz = idz[:count]
    x_vals = torch.cat([x_vals[:count], torch.tensor([torch.inf])])
    return idz, x_vals


def _kgcb(a: Tensor, b: Tensor, current_best: Tensor) -> Tensor:
    """Discrete KG: E[max_i(a_i + b_i Z)] − current_best, Z~N(0,1).

    Sweep on detached CPU copies for speed; final expectation uses
    original tensors so autograd flows through a, b.
    """
    a_flat = a.squeeze()
    b_flat = b.squeeze()
    a_cpu = a_flat.detach().cpu()
    b_cpu = b_flat.detach().cpu()
    a0_cpu, b0_cpu = _filter_a_b(a_cpu, b_cpu)
    epigraph_idx, x_breaks_cpu = _cpu_epigraph_sweep(a0_cpu, b0_cpu)

    # Replicate filter on original (grad-enabled) tensors
    _, a_sort = torch.sort(-a_flat)
    sorted_idx = torch.argsort(b_flat[a_sort])
    final_idx = a_sort[sorted_idx]
    sa, sb = a_flat[final_idx], b_flat[final_idx]
    if len(sa) > 1:
        keep = torch.cat([torch.tensor([True], device=a.device),
                          (sb[1:] - sb[:-1]).detach().abs() > 1e-16])
        sa, sb = sa[keep], sb[keep]

    epigraph_idx_dev = epigraph_idx.to(a.device)
    ae, be = sa[epigraph_idx_dev], sb[epigraph_idx_dev]
    x = x_breaks_cpu.to(a.device)

    pdf = torch.exp(_NORMAL.log_prob(x))
    cdf = _NORMAL.cdf(x)
    kg = torch.sum(ae * (cdf[1:] - cdf[:-1]) + be * (pdf[:-1] - pdf[1:]))
    return kg - current_best


# ──────────────────────────────────────────────────────────────────────
# Batched KGCB (numba-accelerated sweep + vectorised expectation)
# ──────────────────────────────────────────────────────────────────────

def _make_batch_epigraph_sweep():
    """Build the numba-jitted batch sweep; returns None if numba unavailable."""
    if not _HAS_NUMBA:
        return None

    @numba.jit(nopython=True, cache=True, parallel=True)
    def _batch_sweep(a_batch, b_batch):
        """Compute epigraph structures for N instances in parallel.

        Parameters
        ----------
        a_batch, b_batch : (N, M) float64 numpy arrays

        Returns
        -------
        epi_idx    : (N, M)   int64   — original-space indices on the envelope
        epi_breaks : (N, M+1) float64 — breakpoints (padded with +inf)
        epi_counts : (N,)     int64   — number of envelope segments per instance
        """
        N, M = a_batch.shape
        epi_idx = np.zeros((N, M), dtype=np.int64)
        epi_breaks = np.full((N, M + 1), np.inf)
        epi_counts = np.zeros(N, dtype=np.int64)

        for n in numba.prange(N):
            a = a_batch[n]
            b = b_batch[n]

            # Sort by –a, then by b (replicates _filter_a_b ordering)
            order_neg_a = np.argsort(-a)
            sub_order = np.argsort(b[order_neg_a])
            final = order_neg_a[sub_order]
            sa = a[final]
            sb = b[final]

            # Filter near-duplicate b values
            n_keep = 0
            kept_orig = np.empty(M, dtype=np.int64)  # maps to original index
            fa = np.empty(M, dtype=np.float64)
            fb = np.empty(M, dtype=np.float64)
            for i in range(M):
                if i == 0 or np.abs(sb[i] - sb[i - 1]) > 1e-16:
                    kept_orig[n_keep] = final[i]
                    fa[n_keep] = sa[i]
                    fb[n_keep] = sb[i]
                    n_keep += 1

            if n_keep == 0:
                epi_counts[n] = 0
                continue

            # Epigraph sweep
            sweep_local = np.zeros(n_keep, dtype=np.int64)
            sweep_x = np.full(n_keep + 1, -np.inf)
            sweep_local[0] = 0
            count = 1
            i_last = 0

            while i_last < n_keep - 1:
                best_cross = np.inf
                best_j = i_last + 1
                for j in range(i_last + 1, n_keep):
                    denom = fb[i_last] - fb[j]
                    if np.abs(denom) < 1e-30:
                        continue
                    cross = -(fa[i_last] - fa[j]) / denom
                    if cross < best_cross:
                        best_cross = cross
                        best_j = j
                sweep_local[count] = best_j
                sweep_x[count] = best_cross
                count += 1
                i_last = best_j

            sweep_x[count] = np.inf

            epi_counts[n] = count
            for i in range(count):
                epi_idx[n, i] = kept_orig[sweep_local[i]]
            for i in range(count + 1):
                epi_breaks[n, i] = sweep_x[i]

        return epi_idx, epi_breaks, epi_counts

    return _batch_sweep


_batch_epigraph_sweep = _make_batch_epigraph_sweep()


def _kgcb_batched(a: Tensor, b: Tensor, current_best: Tensor) -> Tensor:
    """Batched KGCB with autograd support.

    a, b         : (N, M) grad-enabled tensors
    current_best : (N,)   detached
    Returns      : (N,)   KG values (gradients flow through a and b)

    Phase 1 — numba: epigraph sweep for all N instances in parallel.
    Phase 2 — PyTorch: gather envelope (a, b) via indices, compute
              E[max] with vectorised Φ/φ.  Autograd is preserved because
              torch.gather back-propagates through the gathered values.
    """
    N, M = a.shape
    dev = a.device

    if _batch_epigraph_sweep is None:
        # Fallback: sequential loop
        kg = torch.zeros(N, device=dev, dtype=a.dtype)
        for i in range(N):
            kg[i] = _kgcb(a[i], b[i], current_best[i])
        return kg

    # Phase 1: epigraph structures on CPU via numba
    a_det = a.detach()
    if a_det.is_cuda:
        a_det = a_det.cpu()
    a_np = a_det.double().numpy()
    b_det = b.detach()
    if b_det.is_cuda:
        b_det = b_det.cpu()
    b_np = b_det.double().numpy()
    epi_idx_np, epi_breaks_np, epi_counts_np = _batch_epigraph_sweep(a_np, b_np)

    max_env = int(epi_counts_np.max())
    if max_env == 0:
        return torch.zeros(N, device=dev, dtype=a.dtype)

    # Phase 2: batched expectation with autograd
    epi_idx = torch.from_numpy(epi_idx_np[:, :max_env]).to(dev)  # (N, max_env)
    epi_breaks = torch.from_numpy(epi_breaks_np[:, :max_env + 1]).to(  # (N, max_env+1)
        dtype=a.dtype, device=dev)

    ae = torch.gather(a, 1, epi_idx)  # (N, max_env) — autograd flows
    be = torch.gather(b, 1, epi_idx)  # (N, max_env)

    # Φ and φ at breakpoints; padding is ±inf so Φ(−∞)=0, Φ(+∞)=1, φ(±∞)=0
    pdf = torch.exp(_NORMAL.log_prob(epi_breaks))
    cdf = _NORMAL.cdf(epi_breaks)

    kg = torch.sum(ae * (cdf[:, 1:] - cdf[:, :-1]) + be * (pdf[:, :-1] - pdf[:, 1:]), dim=1)
    # return kg
    return kg - current_best


# ──────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────

class BaseConstrainedKG(DecoupledAcquisitionFunction, MCAcquisitionFunction):
    """Base class for fast constrained KG acquisition functions.

    Provides shared state (model, penalty, x_best, z_y) and posterior
    helpers used by all source-specific subclasses.
    """

    def __init__(
            self,
            model: Model,
            *,
            penalty_value: Tensor,
            x_best_location: Tensor,
            n_fantasies: int = 7,
            seed: int = 0,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            x_evaluation_mask: Optional[Tensor] = None,
            **kwargs,
    ) -> None:
        assert _HAS_NUMBA, (
            "numba is required but is not installed. "
            "Install it with: conda install numba"
        )
        assert torch.cuda.is_available(), (
            "A CUDA GPU is required but none was detected."
        )
        super().__init__(
            model=model, sampler=sampler, objective=objective,
            posterior_transform=posterior_transform,
            X_evaluation_mask=x_evaluation_mask,
        )
        try:
            _dev = next(model.parameters()).device
        except StopIteration:
            _dev = torch.device("cpu")
        self.penalty_value = penalty_value.to(_dev)
        self.x_best = x_best_location.detach().reshape(1, -1).to(_dev)
        self.n_zy = n_fantasies
        self.seed = seed
        self.dim = self.x_best.shape[-1]
        self.K = len(model.models) - 1

        # Deterministic Z_y quantiles
        q = (torch.arange(self.n_zy, device=_dev, dtype=torch.double) + 0.5) / self.n_zy
        self.z_y = _NORMAL.icdf(q)

    # ---- posterior helpers ----

    @staticmethod
    def _sigma_tilde(gp_model, X_full):
        """sigma_tilde(X_full, x_cand) where x_cand is index 0 in X_full."""
        posterior = gp_model.posterior(X_full)
        mu = posterior.mean.squeeze(-1)
        var = posterior.variance.squeeze(-1)
        cov = posterior.mvn.covariance_matrix
        noise = gp_model.likelihood.noise_covar.noise.view(-1)[0]
        cov_col = cov[:, :, 0]
        st = cov_col / (cov[:, 0, 0] + noise).sqrt().unsqueeze(-1)
        return st, mu, var

    def _all_constraint_posteriors(self, X):
        """Compute all constraint posteriors in one pass.

        Returns: (mu_list, sig_list) where each is a list of K tensors of
        shape matching X's batch dims (B, M).
        """
        mu_list, sig_list = [], []
        for ck in range(self.K):
            mn, std = self._idx_constraint_posterior(X, ck)
            mu_list.append(mn)
            sig_list.append(std)
        return mu_list, sig_list

    def _idx_constraint_posterior(self, X, idx):
        cp = self.model.models[idx + 1].posterior(X)
        return cp.mean.squeeze(-1), cp.variance.squeeze(-1).clamp_min(1e-12).sqrt()

    def _compute_pf(self, X):
        """Probability of feasibility. X: (B, M, d). Returns: (B, M).

        Uses log-space accumulation for numerical stability with many
        constraints (avoids underflow when multiplying many CDF values).
        """
        mu_list, sig_list = self._all_constraint_posteriors(X)
        log_pf = torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
        for mu, sig in zip(mu_list, sig_list):
            log_pf = log_pf + _log_normal_cdf(-mu / sig)
        return log_pf.exp()

    def _compute_pf_except_k(self, X, k):
        """PF excluding constraint k. X: (B, M, d). Returns: (B, M).

        Uses log-space accumulation for numerical stability.
        """
        log_pf = torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
        for j in range(self.K):
            if j == k:
                continue
            mu, sig = self._idx_constraint_posterior(X, j)
            log_pf = log_pf + _log_normal_cdf(-mu / sig)
        return log_pf.exp()

    # ---- forward shell ----

    def forward(self, X: Tensor) -> Tensor:
        """X: (batch, 1 + n_disc, d) — slot 0 is candidate, rest is discretisation."""
        if X.dim() == 2:
            X = X.unsqueeze(0)
        B = X.shape[0]
        x = X[:, 0, :]
        X_disc = X[:, 1:, :]
        with gpytorch.settings.fast_pred_var():
            return self._evaluate(x, X_disc, B)

    def _evaluate(self, x, X_disc, B):
        raise NotImplementedError


class ObjectiveDcKG(BaseConstrainedKG):
    """dcKG^0: PF unchanged, mu_f updates linearly -> KGCB."""

    def _evaluate(self, x, X_disc, B):
        X_full = torch.cat([x.unsqueeze(1),
                            self.x_best.unsqueeze(0).expand(B, -1, -1),
                            X_disc,
                            ], dim=1)
        st_y, mu_y, _ = self._sigma_tilde(self.model.models[0], X_full)
        # Compute all constraint posteriors once, pass to _compute_pf
        pf = self._compute_pf(X_full)
        a = mu_y * pf - self.penalty_value * (1 - pf)
        b = st_y * pf
        return _kgcb_batched(a, b, a.max(dim=1).values.detach())


class ConstraintDcKG(BaseConstrainedKG):
    """dcKG^k (k >= 1): objective fixed, PF_k updates nonlinearly (Eq. 8)."""

    def __init__(self, model, *, constraint_index: int, **kwargs):
        super().__init__(model=model, **kwargs)
        self.constraint_index = constraint_index  # 0-based
        self.source_model_index = constraint_index + 1  # index into model.models

    def _evaluate(self, x, X_disc, B):
        k = self.constraint_index
        X_grid = torch.cat([
            self.x_best.unsqueeze(0).expand(B, -1, -1),
            X_disc,
        ], dim=1)
        G_size = X_grid.shape[1]

        X_joint = torch.cat([X_grid, x.unsqueeze(1)], dim=1)

        con_model = self.model.models[self.source_model_index]
        con_p = con_model.posterior(X_joint)
        mu_ck_all = con_p.mean.squeeze(-1)
        var_ck_all = con_p.variance.squeeze(-1)
        cov_ck_all = con_p.mvn.covariance_matrix

        noise_k = con_model.likelihood.noise_covar.noise.view(-1)[0]
        st_ck = cov_ck_all[:, :G_size, -1] / (cov_ck_all[:, -1, -1] + noise_k).sqrt().unsqueeze(-1)
        mu_ck = mu_ck_all[:, :G_size]
        var_ck = var_ck_all[:, :G_size]

        # Compute objective + all constraint posteriors on X_grid once
        mu_f = self.model.models[0].posterior(X_grid).mean.squeeze(-1)
        pf_others = self._compute_pf_except_k(X_grid, k)

        z_exp = self.z_y.view(-1, 1, 1)
        mu_ck_new = mu_ck.unsqueeze(0) + st_ck.unsqueeze(0) * z_exp
        var_ck_new = (var_ck.unsqueeze(0) - st_ck.unsqueeze(0) ** 2).clamp_min(1e-12)
        pf_k_new = _NORMAL.cdf(-mu_ck_new / var_ck_new.sqrt())

        pf_new = pf_k_new * pf_others.unsqueeze(0)
        G_new = mu_f.unsqueeze(0) * pf_new - self.penalty_value * (1 - pf_new)

        G_star = G_new.max(dim=-1)[0]
        G_xr = G_new[:, :, 0]  # x_best is at index 0

        return (G_star - G_xr).clamp_min(0).mean(dim=0)
        # return (G_star).mean(dim=0)


class CoupledCKG(BaseConstrainedKG):
    """cKG (Eq. 7): all sources updated jointly, outer MC over Z_c."""

    def __init__(self, model, *, n_constraint_samples: int = 5, **kwargs):
        super().__init__(model=model, **kwargs)
        dev = self.x_best.device
        if self.K > 0:
            self.z_c = draw_sobol_normal_samples(
                d=self.K, n=n_constraint_samples, device=dev,
                dtype=torch.double, seed=self.seed,
            )
        else:
            self.z_c = torch.zeros(n_constraint_samples, 1, device=dev, dtype=torch.double)
        self.n_zc = n_constraint_samples

    def _evaluate(self, x, X_disc, B):
        X_full = torch.cat([
            x.unsqueeze(1),
            self.x_best.unsqueeze(0).expand(B, -1, -1),
            X_disc,
        ], dim=1)
        M = X_full.shape[1]

        # Compute all K+1 posteriors: objective + constraints
        # Objective: need st and mu for KGCB
        st_y, mu_y, _ = self._sigma_tilde(self.model.models[0], X_full)

        # All constraints in one pass: need st, mu, var for reparametrisation
        st_ck_list, mu_ck_list, var_ck_list = [], [], []
        for ck in range(self.K):
            p = self.model.models[ck + 1].posterior(X_full)
            mu = p.mean.squeeze(-1)
            var = p.variance.squeeze(-1)
            cov = p.mvn.covariance_matrix
            noise = self.model.models[ck + 1].likelihood.noise_covar.noise.view(-1)[0]
            st = cov[:, :, 0] / (cov[:, 0, 0] + noise).sqrt().unsqueeze(-1)
            st_ck_list.append(st)
            mu_ck_list.append(mu)
            var_ck_list.append(var)

        st_ck = torch.stack(st_ck_list)  # (K, B, M)
        mu_ck = torch.stack(mu_ck_list)  # (K, B, M)
        var_ck = torch.stack(var_ck_list)  # (K, B, M)

        z_exp = self.z_c.unsqueeze(-1).unsqueeze(-1)
        mu_new = mu_ck.unsqueeze(0) + st_ck.unsqueeze(0) * z_exp
        var_new = (var_ck.unsqueeze(0) - st_ck.unsqueeze(0) ** 2).clamp_min(1e-12)
        log_pf_k = _log_normal_cdf(-mu_new / var_new.sqrt())
        pf_new = log_pf_k.sum(dim=1).exp()

        a_all = mu_y.unsqueeze(0) * pf_new - self.penalty_value * (1 - pf_new)
        b_all = st_y.unsqueeze(0) * pf_new

        a_flat = a_all.reshape(-1, M)
        b_flat = b_all.reshape(-1, M)
        current_flat = a_flat[:, 1].detach()  # x_best is at index 1
        kg_flat = _kgcb_batched(a_flat, b_flat, current_flat)
        return kg_flat.reshape(self.n_zc, B).mean(dim=0)


# Backwards-compatible alias
FastConstrainedKG = CoupledCKG


# ──────────────────────────────────────────────────────────────────────
# All-sources dcKG: evaluates every source in a single forward pass
# ──────────────────────────────────────────────────────────────────────

class AllSourcesDcKG(torch.nn.Module):
    """Jointly optimises candidate + discretisation for all K+2 sources.

    Composes K+2 independent FastConstrainedKG instances:
      0      — dcKG^0 (objective): FastConstrainedKG(source_index=0)
      1..K   — dcKG^k (constraint k): FastConstrainedKG(source_index=k)
      K+1    — cKG (coupled): FastConstrainedKG(evaluate_all_sources=True)

    Each source gets its own candidate and discretisation, all jointly
    optimised by L-BFGS in a single ``optimize_acqf`` call.

    Input X shape: ``(R, n_sources * q_per_source, d)`` where
      ``n_sources = K + 2`` and ``q_per_source = 1 + n_disc``.
    Internally reshaped to ``(R, n_sources, q_per_source, d)``; for each
    source, slot 0 is the candidate and slots 1.. are discretisation.

    Returns shape ``(R,)`` — sum across sources.

    Usage::

        acqf = AllSourcesDcKG(model, ...)
        q = acqf.n_sources * acqf.q_per_source
        candidates, values = optimize_acqf(acqf, bounds, q=q, ...)
        per_source = acqf.evaluate_per_source(candidates)  # (K+2,)
        best_source = (per_source / costs).argmax()
        best_x = candidates.view(acqf.n_sources, acqf.q_per_source, -1)[best_source, 0]
    """

    def __init__(
            self,
            model,
            *,
            penalty_value,
            x_best_location,
            objective=None,
            n_fantasies=7,
            n_constraint_samples=5,
            n_disc=16,
            seed=0,
    ):
        assert _HAS_NUMBA, (
            "numba is required for AllSourcesDcKG but is not installed. "
            "Install it with: conda install numba"
        )
        assert torch.cuda.is_available(), (
            "AllSourcesDcKG requires a CUDA GPU but none was detected."
        )
        super().__init__()
        self.model = model
        try:
            _dev = next(model.parameters()).device
        except StopIteration:
            _dev = torch.device("cpu")
        self.penalty_value = penalty_value.to(_dev)
        self.x_best = x_best_location.detach().reshape(1, -1).to(_dev)
        self.n_disc = n_disc
        self.K = len(model.models) - 1
        self.n_sources = self.K + 2  # K+1 decoupled + 1 coupled
        self.q_per_source = 1 + n_disc
        n_outputs = len(model.models)

        # Shared kwargs for all sources
        common = dict(
            model=model, penalty_value=penalty_value,
            x_best_location=x_best_location, objective=objective,
            n_fantasies=n_fantasies, seed=seed,
        )
        self.sources = torch.nn.ModuleList()

        # Source 0: objective dcKG
        self.sources.append(ObjectiveDcKG(**common))

        # Sources 1..K: constraint dcKG
        for k in range(self.K):
            self.sources.append(ConstraintDcKG(
                constraint_index=k, **common,
            ))

        # Source K+1: coupled cKG
        self.sources.append(CoupledCKG(
            n_constraint_samples=n_constraint_samples, **common,
        ))

    # ---- forward ----

    def forward(self, X):
        """Evaluate all K+2 sources, each with its own candidate + discretisation.

        X: (R, n_sources * q_per_source, d) from optimize_acqf.
        Returns: (R,) — sum across sources for joint L-BFGS.
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)
        return self._evaluate_all(X).sum(dim=-1)

    def evaluate_per_source(self, X):
        """Per-source dcKG values for cost-normalised source selection.

        X: (1, n_sources * q_per_source, d) or (n_sources * q_per_source, d).
        Returns: (K+2,)
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)
        return self._evaluate_all(X).squeeze(0)

    def _evaluate_all(self, X):
        """Evaluate each source's FastConstrainedKG on its chunk of X.

        X: (R, n_sources * q_per_source, d).
        Returns: (R, K+2)
        """
        R = X.shape[0]
        Q = self.q_per_source
        result = torch.zeros(R, self.n_sources, device=X.device, dtype=X.dtype)

        for s, source_acqf in enumerate(self.sources):
            X_s = X[:, s * Q:(s + 1) * Q, :]  # (R, q_per_source, d)
            result[:, s] = source_acqf.forward(X_s)

        return result
