"""Old vs new dcKG acquisition functions (observation-noise units fix).

The Old* classes below are verbatim copies of the pre-fix methods, which used the raw
likelihood noise (standardized units) next to posterior covariances in original units.
Each source acquisition function is evaluated with the new code and with the old code
(same object, same fantasy samples), on a 6-source GP fitted to TestFunc3Redundant.

Run with output:  pytest -s tests/acquisition_functions/test_noise_fix_old_vs_new.py
"""
import time

import torch
from botorch.acquisition import ConstrainedMCObjective

from bo.acquisition_functions.refactored_acquisition_functions import (
    _NORMAL, _kgcb_batched, _log_normal_cdf, AllSourcesDcKG, ConstraintDcKG, CoupledCKG, ObjectiveDcKG)
from bo.model.Model import ConstrainedDeoupledGPModelWrapper, constraint_callable_wrapper, obj_callable
from bo.synthetic_test_functions.synthetic_test_functions import ConstrainedFunc3Redundant

DT = torch.double
PENALTY = 4.0


# ───────────── OLD versions (copied from before the fix) ─────────────

def _old_sigma_tilde(gp_model, X_full):
    posterior = gp_model.posterior(X_full)
    mu = posterior.mean.squeeze(-1)
    var = posterior.variance.squeeze(-1)
    cov = posterior.mvn.covariance_matrix
    noise = gp_model.likelihood.noise_covar.noise.view(-1)[0]
    cov_col = cov[:, :, 0]
    st = cov_col / (cov[:, 0, 0] + noise).sqrt().unsqueeze(-1)
    return st, mu, var


class OldObjectiveDcKG(ObjectiveDcKG):
    _sigma_tilde = staticmethod(_old_sigma_tilde)


class OldConstraintDcKG(ConstraintDcKG):
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

        mu_f = self.model.models[0].posterior(X_grid).mean.squeeze(-1)
        pf_others = self._compute_pf_except_k(X_grid, k)

        z_exp = self.z_y.view(-1, 1, 1)
        mu_ck_new = mu_ck.unsqueeze(0) + st_ck.unsqueeze(0) * z_exp
        var_ck_new = (var_ck.unsqueeze(0) - st_ck.unsqueeze(0) ** 2).clamp_min(1e-12)
        pf_k_new = _NORMAL.cdf(-mu_ck_new / var_ck_new.sqrt())

        pf_new = pf_k_new * pf_others.unsqueeze(0)
        G_new = mu_f.unsqueeze(0) * pf_new - self.penalty_value * (1 - pf_new)

        G_star = G_new.max(dim=-1)[0]
        G_xr = G_new[:, :, 0]
        return (G_star - G_xr).clamp_min(0).mean(dim=0)


class OldCoupledCKG(CoupledCKG):
    _sigma_tilde = staticmethod(_old_sigma_tilde)

    def _evaluate(self, x, X_disc, B):
        X_full = torch.cat([
            x.unsqueeze(1),
            self.x_best.unsqueeze(0).expand(B, -1, -1),
            X_disc,
        ], dim=1)
        M = X_full.shape[1]

        st_y, mu_y, _ = self._sigma_tilde(self.model.models[0], X_full)

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

        st_ck = torch.stack(st_ck_list)
        mu_ck = torch.stack(mu_ck_list)
        var_ck = torch.stack(var_ck_list)
        if self.update_mask is not None:
            st_ck = st_ck * self.update_mask.view(-1, 1, 1)

        z_exp = self.z_c.unsqueeze(-1).unsqueeze(-1)
        mu_new = mu_ck.unsqueeze(0) + st_ck.unsqueeze(0) * z_exp
        var_new = (var_ck.unsqueeze(0) - st_ck.unsqueeze(0) ** 2).clamp_min(1e-12)
        log_pf_k = _log_normal_cdf(-mu_new / var_new.sqrt())
        pf_new = log_pf_k.sum(dim=1).exp()

        a_all = mu_y.unsqueeze(0) * pf_new - self.penalty_value * (1 - pf_new)
        b_all = st_y.unsqueeze(0) * pf_new

        a_flat = a_all.reshape(-1, M)
        b_flat = b_all.reshape(-1, M)
        current_flat = a_flat[:, 1].detach()
        kg_flat = _kgcb_batched(a_flat, b_flat, current_flat)
        return kg_flat.reshape(self.n_zc, B).mean(dim=0)


OLD_CLASS = {ObjectiveDcKG: OldObjectiveDcKG, ConstraintDcKG: OldConstraintDcKG, CoupledCKG: OldCoupledCKG}


# ───────────── comparison ─────────────

def _setup():
    torch.manual_seed(0)
    f = ConstrainedFunc3Redundant(noise_std=0.0, negate=True, noisy_active_constraint=2)
    X0 = torch.quasirandom.SobolEngine(2, scramble=True, seed=0).draw(30).to(DT)
    Y0 = f.evaluate_black_box(X0)
    w = ConstrainedDeoupledGPModelWrapper(num_constraints=5, is_noisy=True)
    w.fit([X0] * 6, [Y0[:, s] for s in range(6)])
    model = w.optimize()

    # Incumbent = argmax of the penalised posterior mean on a grid; the discretisation is the
    # 64 best grid points (incumbent first), candidates are spread over the domain.
    grid = torch.quasirandom.SobolEngine(2, scramble=True, seed=1).draw(4096).to(DT)
    with torch.no_grad():
        mu_f = model.models[0].posterior(grid).mean.squeeze(-1)
        pf = torch.ones_like(mu_f)
        for m in model.models[1:]:
            p = m.posterior(grid)
            pf *= _NORMAL.cdf(-p.mean.squeeze(-1) / p.variance.squeeze(-1).sqrt())
        top = (mu_f * pf - PENALTY * (1 - pf)).argsort(descending=True)
    x_best = grid[top[0]].reshape(1, 2)
    cand = torch.quasirandom.SobolEngine(2, scramble=True, seed=2).draw(32).to(DT)
    X = torch.cat([cand.unsqueeze(1), grid[top[:64]].expand(32, 64, 2)], dim=1)  # (32, 65, 2)

    objective = ConstrainedMCObjective(objective=obj_callable,
                                       constraints=[constraint_callable_wrapper(i) for i in range(1, 6)])
    acqf = AllSourcesDcKG(model, penalty_value=torch.tensor([PENALTY]), x_best_location=x_best,
                          objective=objective, n_fantasies=7, n_constraint_samples=5, n_disc=64, seed=1)
    return model, acqf, X


def _run(src, cls, X, reps=3):
    src.__class__ = cls
    with torch.no_grad():
        vals = src(X)
        t0 = time.perf_counter()
        for _ in range(reps):
            src(X)
    return vals, (time.perf_counter() - t0) / reps


def test_old_vs_new_acquisition():
    model, acqf, X = _setup()
    names = ["obj", "c1", "c2", "c3", "c4", "c5", "coupled"]

    print(f"\n{'src':>4} {'y_std':>7} {'raw noise':>10} {'noise x y_std^2':>16}")
    for n, m in zip(names, model.models):
        s = m.outcome_transform.stdvs.item()
        raw = m.likelihood.noise_covar.noise.view(-1)[0].item()
        print(f"{n:>4} {s:7.3f} {raw:10.3g} {raw * s * s:16.3g}")

    print(f"\n{'source':>8} {'max KG old':>11} {'max KG new':>11} {'argmax old':>10} {'argmax new':>10} "
          f"{'t old ms':>9} {'t new ms':>9}")
    best_old, best_new, t_old_tot, t_new_tot = [], [], 0.0, 0.0
    for n, src in zip(names, acqf.sources):
        new_cls = type(src)
        v_new, t_new = _run(src, new_cls, X)
        v_old, t_old = _run(src, OLD_CLASS[new_cls], X)
        src.__class__ = new_cls
        assert torch.isfinite(v_new).all()
        best_old.append(v_old.max().item()); best_new.append(v_new.max().item())
        t_old_tot += t_old; t_new_tot += t_new
        print(f"{n:>8} {v_old.max().item():11.4g} {v_new.max().item():11.4g} {v_old.argmax().item():10d} "
              f"{v_new.argmax().item():10d} {t_old * 1e3:9.1f} {t_new * 1e3:9.1f}")

    print(f"\nselected source  old: {names[int(torch.tensor(best_old).argmax())]}"
          f"   new: {names[int(torch.tensor(best_new).argmax())]}")
    print(f"total forward time  old: {t_old_tot * 1e3:.1f} ms   new: {t_new_tot * 1e3:.1f} ms")
    # The fix is a scalar multiplication per source: forward time must be unchanged.
    assert t_new_tot < 1.25 * t_old_tot
