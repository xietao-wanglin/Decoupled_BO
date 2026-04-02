"""Visualise discretisation + epigraph side by side, before and after opt.

Generates 4 separate files:
  - disc_ObjDcKG_before.png
  - disc_ObjDcKG_after.png
  - disc_CoupCKG_before.png
  - disc_CoupCKG_after.png

Each file has two panels:
  Left:  Landscape with disc points, candidate, x_best, epigraph highlights
  Right: KGCB epigraph (ObjectiveDcKG) or per-Z_c bar chart (CoupledCKG)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ConstrainedMCObjective
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from gpytorch.mlls import SumMarginalLogLikelihood

from bo.acquisition_functions.refactored_acquisition_functions import (
    ObjectiveDcKG, CoupledCKG, _batch_epigraph_sweep, _kgcb_batched, _NORMAL,
)
from bo.bo_loops.bo_loop import OptimizationLoop
from bo.model.Model import ConstrainedPosteriorMean
from bo.synthetic_test_functions.synthetic_test_functions import ConstrainedBraninNew

dtype = torch.double
torch.set_default_dtype(dtype)
DEVICE = torch.device("cpu")


def obj_callable(Z, X=None):
    return Z[..., 0]


def _build_model(n_points, seed=42):
    torch.manual_seed(seed)
    func = ConstrainedBraninNew(noise_std=1e-6, negate=True)
    X = torch.rand(n_points, 2, device=DEVICE, dtype=dtype)
    NOISE = torch.tensor(1e-6, device=DEVICE, dtype=dtype)
    Y_obj = -func.evaluate_true(X.cpu()).unsqueeze(-1).to(DEVICE)
    Y_con = func.evaluate_slack_true(X.cpu()).to(DEVICE)
    if Y_con.dim() == 1:
        Y_con = Y_con.unsqueeze(-1)
    m_obj = SingleTaskGP(X, Y_obj, train_Yvar=NOISE.expand_as(Y_obj),
                         outcome_transform=Standardize(m=1))
    m_con = SingleTaskGP(X, Y_con, train_Yvar=NOISE.expand_as(Y_con),
                         outcome_transform=Standardize(m=1))
    model = ModelListGP(m_obj, m_con)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _compute_landscape(model, penalty, grid_size=100):
    x1 = torch.linspace(0, 1, grid_size, device=DEVICE, dtype=dtype)
    x2 = torch.linspace(0, 1, grid_size, device=DEVICE, dtype=dtype)
    X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
    X_grid = torch.stack([X1.ravel(), X2.ravel()], dim=-1).unsqueeze(1)
    with torch.no_grad():
        mu_f = model.models[0].posterior(X_grid).mean.squeeze()
        cp = model.models[1].posterior(X_grid)
        mu_c = cp.mean.squeeze()
        sig_c = cp.variance.squeeze().clamp_min(1e-12).sqrt()
        pf = _NORMAL.cdf(-mu_c / sig_c)
        landscape = mu_f * pf - penalty * (1 - pf)
    return (X1.cpu().numpy(), X2.cpu().numpy(),
            landscape.reshape(grid_size, grid_size).cpu().numpy())


def _build_ics(source_acqf, bounds, x_best, n_disc=32,
               num_restarts=5, raw_samples=64, eta=2.0, seed=0,
               use_sobol_z=False):
    Q = 1 + n_disc
    ics = gen_batch_initial_conditions(
        acq_function=source_acqf, bounds=bounds.to(x_best.device), q=Q,
        num_restarts=num_restarts, raw_samples=raw_samples,
        options={"seed": seed, "eta": eta, "init_batch_limit": 16},
    )
    n_thompson = source_acqf.n_zc if use_sobol_z else source_acqf.n_zy
    thompson_pts = OptimizationLoop._thompson_disc(
        model=source_acqf.model,
        penalty_value=source_acqf.penalty_value,
        bounds=bounds.to(x_best.device), n_thompson=n_thompson,
        seed=seed, use_sobol_z=use_sobol_z,
    )
    ics[:, 1:1 + n_thompson, :] = thompson_pts.unsqueeze(0)
    xbest_ic = ics[-1:].clone()
    xbest_ic[:, 0:1, :] = x_best
    ics = torch.cat([ics, xbest_ic], dim=0)
    return ics, n_thompson


def _get_ab(acqf, X_full):
    with torch.no_grad():
        st_y, mu_y, _ = acqf._sigma_tilde(acqf.model.models[0], X_full)
        pf = acqf._compute_pf(X_full)
    a = (mu_y * pf - acqf.penalty_value * (1 - pf))[0]
    b = (st_y * pf)[0]
    return a.cpu().numpy(), b.cpu().numpy()


def _epigraph_info(a, b):
    a_np = a.astype(np.float64).reshape(1, -1)
    b_np = b.astype(np.float64).reshape(1, -1)
    idx, brk, cnt = _batch_epigraph_sweep(a_np, b_np)
    c = int(cnt[0])
    return idx[0, :c], brk[0, :c + 1]


def _ckg_per_zc(acqf, X_full):
    with torch.no_grad():
        st_y, mu_y, _ = acqf._sigma_tilde(acqf.model.models[0], X_full)
        st_list, mu_list, var_list = [], [], []
        for ck in range(acqf.K):
            p = acqf.model.models[ck + 1].posterior(X_full)
            mu = p.mean.squeeze(-1)
            var = p.variance.squeeze(-1)
            cov = p.mvn.covariance_matrix
            noise = acqf.model.models[ck + 1].likelihood.noise_covar.noise.view(-1)[0]
            st = cov[:, :, 0] / (cov[:, 0, 0] + noise).sqrt().unsqueeze(-1)
            st_list.append(st); mu_list.append(mu); var_list.append(var)
        M = X_full.shape[1]
        st_ck = torch.stack(st_list); mu_ck = torch.stack(mu_list)
        var_ck = torch.stack(var_list)
        z_exp = acqf.z_c.unsqueeze(-1).unsqueeze(-1)
        mu_new = mu_ck.unsqueeze(0) + st_ck.unsqueeze(0) * z_exp
        var_new = (var_ck.unsqueeze(0) - st_ck.unsqueeze(0)**2).clamp_min(1e-12)
        pf_k = _NORMAL.cdf(-mu_new / var_new.sqrt())
        pf_new = pf_k.prod(dim=1)
        a_all = mu_y.unsqueeze(0) * pf_new - acqf.penalty_value * (1 - pf_new)
        b_all = st_y.unsqueeze(0) * pf_new
        vals = []
        for i in range(acqf.n_zc):
            kg = _kgcb_batched(a_all[i], b_all[i], a_all[i, :, 1:2].max(dim=-1).values.detach())
            vals.append(kg.item())
    return vals


# ── Plotting ────────────────────────────────────────────────────────

def _plot_landscape_panel(ax, X1, X2, Z, disc, cand, x_best,
                          n_thompson, is_before, epi_disc_idx=None):
    ax.contourf(X1, X2, Z, levels=30, cmap="RdYlGn", alpha=0.7)
    ax.contour(X1, X2, Z, levels=30, colors="k", linewidths=0.2, alpha=0.3)
    ax.contour(X1, X2, Z, levels=[0], colors="black", linewidths=2,
               linestyles="--")

    color_t = "dodgerblue" if is_before else "darkorange"
    color_gb = "white" if is_before else "yellow"

    # Thompson disc
    ax.scatter(disc[:n_thompson, 0], disc[:n_thompson, 1],
               c=color_t, s=30, marker="o", edgecolors="k",
               linewidths=0.5, zorder=3, alpha=0.9, label="Thompson")
    # gen_batch disc
    if disc.shape[0] > n_thompson:
        ax.scatter(disc[n_thompson:, 0], disc[n_thompson:, 1],
                   c=color_gb, s=18, marker="s", edgecolors="gray",
                   linewidths=0.4, zorder=3, alpha=0.7, label="gen_batch")
    # Epigraph points
    if epi_disc_idx is not None and len(epi_disc_idx) > 0:
        valid = epi_disc_idx[epi_disc_idx < disc.shape[0]]
        if len(valid) > 0:
            ax.scatter(disc[valid, 0], disc[valid, 1], facecolors="none",
                       s=110, marker="o", edgecolors="magenta", linewidths=2.5,
                       zorder=4, label=f"Epigraph ({len(valid)})")

    ax.scatter(x_best[0], x_best[1], c="lime", s=60, marker="D",
               edgecolors="darkgreen", linewidths=1.5, zorder=5)
    ax.annotate("x*", (x_best[0], x_best[1]), fontsize=7, fontweight="bold",
                color="darkgreen", xytext=(4, 4), textcoords="offset points")
    ax.scatter(cand[0, 0], cand[0, 1], c="red", s=100, marker="*",
               edgecolors="darkred", linewidths=1, zorder=6)
    ax.annotate("cand", (cand[0, 0], cand[0, 1]), fontsize=7,
                fontweight="bold", color="darkred",
                xytext=(4, -8), textcoords="offset points")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=5.5, framealpha=0.9)


def _plot_epigraph_panel(ax, a, b, epi_idx, epi_breaks, current_best, kg_val):
    z_range = np.linspace(-3, 3, 300)
    for i in range(len(a)):
        ax.plot(z_range, a[i] + b[i] * z_range,
                color="lightgray", linewidth=0.5, alpha=0.5)
    for i in epi_idx:
        ax.plot(z_range, a[i] + b[i] * z_range,
                color="magenta", linewidth=1.8, alpha=0.9)
    ax.axhline(current_best, color="green", linewidth=1.2, linestyle="--",
               label=f"max(a) = {current_best:.2f}")
    for brk in epi_breaks:
        if np.isfinite(brk):
            ax.axvline(brk, color="gray", linewidth=0.4, linestyle=":")
    ax.set_xlim(-3, 3)
    ax.set_xlabel("Z", fontsize=9)
    ax.set_ylabel("a + b·Z", fontsize=9)
    ax.set_title(f"KGCB epigraph — {len(epi_idx)} envelope / {len(a)} lines\n"
                 f"KG = {kg_val:.4f}", fontsize=9)
    ax.legend(fontsize=7)


def _plot_ckg_bars_panel(ax, per_zc, kg_val):
    n = len(per_zc)
    x = np.arange(n)
    ax.bar(x, per_zc, color="darkorange", edgecolor="k", linewidth=0.5,
           alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Z_c[{i}]" for i in range(n)], fontsize=7)
    ax.set_ylabel("KG per Z_c sample", fontsize=9)
    ax.set_title(f"Per-Z_c KG values\n"
                 f"cKG = mean = {kg_val:.4f}", fontsize=9)
    ax.axhline(np.mean(per_zc), color="red", linewidth=1, linestyle="--",
               label=f"mean = {np.mean(per_zc):.4f}")
    ax.legend(fontsize=7)


def _make_figure(ax_land, ax_right, X1, X2, Z, disc, cand, x_best,
                 n_thompson, is_before, epi_disc_idx, name, stage, kg_val,
                 # For ObjectiveDcKG:
                 a=None, b=None, epi_idx=None, epi_breaks=None,
                 current_best=None,
                 # For CoupledCKG:
                 per_zc=None):

    _plot_landscape_panel(ax_land, X1, X2, Z, disc, cand, x_best,
                          n_thompson, is_before, epi_disc_idx)
    ax_land.set_title(f"{name} — {stage}\nKG = {kg_val:.4f}", fontsize=10)

    if a is not None:
        _plot_epigraph_panel(ax_right, a, b, epi_idx, epi_breaks,
                             current_best, kg_val)
    elif per_zc is not None:
        _plot_ckg_bars_panel(ax_right, per_zc, kg_val)


def main():
    func = ConstrainedBraninNew(noise_std=1e-6, negate=True)
    bounds = torch.tensor([[0., 0.], [1., 1.]], device=DEVICE, dtype=dtype)
    penalty = torch.tensor([func.get_penalty()], device=DEVICE)
    objective = ConstrainedMCObjective(objective=obj_callable,
                                       constraints=[obj_callable])
    n_disc = 32

    model = _build_model(5, seed=42)
    loc, _ = optimize_acqf(
        ConstrainedPosteriorMean(model, penalty_value=penalty),
        bounds=bounds, q=1, num_restarts=10, raw_samples=128,
    )
    x_best_np = loc.detach().cpu().numpy().squeeze()
    X1, X2, Z = _compute_landscape(model, penalty.item())

    common = dict(model=model, penalty_value=penalty, x_best_location=loc,
                  objective=objective, n_fantasies=7, seed=0)

    for name, make_acqf, use_sobol, short in [
        ("ObjectiveDcKG", lambda: ObjectiveDcKG(**common), False, "ObjDcKG"),
        ("CoupledCKG", lambda: CoupledCKG(n_constraint_samples=5, **common),
         True, "CoupCKG"),
    ]:
        acqf = make_acqf()
        Q = 1 + n_disc

        ics, n_thompson = _build_ics(
            acqf, bounds, loc, n_disc=n_disc, num_restarts=5,
            raw_samples=64, seed=0, use_sobol_z=use_sobol,
        )

        # --- BEFORE ---
        with torch.no_grad():
            init_vals = acqf(ics)
        best_ic = ics[init_vals.argmax()]
        cand_before = best_ic[0:1].cpu().numpy()
        disc_before = best_ic[1:].cpu().numpy()
        kg_before = init_vals.max().item()

        X_full_b = torch.cat([
            best_ic[0:1].unsqueeze(0), acqf.x_best.unsqueeze(0),
            best_ic[1:].unsqueeze(0),
        ], dim=1)

        epi_disc_b = None
        extra_b = {}
        if isinstance(acqf, ObjectiveDcKG):
            a_b, b_b = _get_ab(acqf, X_full_b)
            ei_b, eb_b = _epigraph_info(a_b, b_b)
            epi_disc_b = ei_b[ei_b >= 2] - 2
            extra_b = dict(a=a_b, b=b_b, epi_idx=ei_b, epi_breaks=eb_b,
                           current_best=a_b.max())
        else:
            pzc_b = _ckg_per_zc(acqf, X_full_b)
            extra_b = dict(per_zc=pzc_b)

        fig_b, (ax1_b, ax2_b) = plt.subplots(1, 2, figsize=(14, 6))
        _make_figure(ax1_b, ax2_b, X1, X2, Z, disc_before, cand_before,
                     x_best_np, n_thompson, True, epi_disc_b,
                     name, "BEFORE opt", kg_before, **extra_b)
        fig_b.suptitle(f"{name} — BEFORE optimisation (5 pts)",
                       fontsize=12, fontweight="bold")
        fig_b.tight_layout(rect=[0, 0, 1, 0.95])
        fname_b = f"plots/disc_{short}_before.png"
        fig_b.savefig(fname_b, dpi=150, bbox_inches="tight")
        plt.close(fig_b)
        print(f"Saved {fname_b}")

        # --- AFTER ---
        candidates, val = optimize_acqf(
            acq_function=acqf, bounds=bounds, q=Q,
            num_restarts=ics.shape[0], batch_initial_conditions=ics,
            options={"maxiter": 100},
        )
        cand_after = candidates[0:1, :].detach().cpu().numpy()
        disc_after = candidates[1:, :].detach().cpu().numpy()
        kg_after = val.item()

        X_full_a = torch.cat([
            candidates[0:1, :].unsqueeze(0), acqf.x_best.unsqueeze(0),
            candidates[1:, :].unsqueeze(0),
        ], dim=1)

        epi_disc_a = None
        extra_a = {}
        if isinstance(acqf, ObjectiveDcKG):
            a_a, b_a = _get_ab(acqf, X_full_a)
            ei_a, eb_a = _epigraph_info(a_a, b_a)
            epi_disc_a = ei_a[ei_a >= 2] - 2
            extra_a = dict(a=a_a, b=b_a, epi_idx=ei_a, epi_breaks=eb_a,
                           current_best=a_a.max())
        else:
            pzc_a = _ckg_per_zc(acqf, X_full_a)
            extra_a = dict(per_zc=pzc_a)

        fig_a, (ax1_a, ax2_a) = plt.subplots(1, 2, figsize=(14, 6))
        _make_figure(ax1_a, ax2_a, X1, X2, Z, disc_after, cand_after,
                     x_best_np, n_thompson, False, epi_disc_a,
                     name, "AFTER opt", kg_after, **extra_a)
        fig_a.suptitle(f"{name} — AFTER optimisation (5 pts)",
                       fontsize=12, fontweight="bold")
        fig_a.tight_layout(rect=[0, 0, 1, 0.95])
        fname_a = f"plots/disc_{short}_after.png"
        fig_a.savefig(fname_a, dpi=150, bbox_inches="tight")
        plt.close(fig_a)
        print(f"Saved {fname_a}")


if __name__ == "__main__":
    main()
