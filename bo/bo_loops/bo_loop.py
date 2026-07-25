import math
import time
import warnings
from typing import Optional

import torch
from botorch import gen_candidates_scipy, gen_candidates_torch
from botorch.acquisition import AcquisitionFunction, MCAcquisitionObjective
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.utils import draw_sobol_samples
from botorch.utils.transforms import standardize
from scipy.stats import qmc
from torch import Tensor

from bo.acquisition_functions.acquisition_functions import (
    acquisition_function_factory, AcquisitionFunctionType,
    DecopledHybridConstrainedKnowledgeGradient,
)
from bo.acquisition_functions.refactored_acquisition_functions import (
    FastConstrainedKG, AllSourcesDcKG, ObjectiveDcKG, ConstraintDcKG, CoupledCKG,
)
from bo.acquisition_functions.pesc import sample_constrained_optima
from bo.device_utils import DTYPE as dtype
from bo.model.Model import (
    ConstrainedPosteriorMean, ConstrainedDeoupledGPModelWrapper,
    gaussian_copula_transform,
)
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import SingleObjectiveProblem

warnings.filterwarnings("ignore")  # Comment out if there are issues


class _FixedCandidateV2DcKG(AcquisitionFunction):
    """Thin wrapper so optimize_acqf optimises only the discretisation.

    V2 source acqfs (ObjectiveDcKG / ConstraintDcKG) consume X of shape
    (B, 1 + n_disc, d) with slot 0 as the candidate. This wrapper takes Y of
    shape (B, n_disc, d) and prepends a fixed candidate before forwarding.
    """

    def __init__(self, base_acqf, fixed_x: Tensor):
        super().__init__(model=base_acqf.model)
        self.base = base_acqf
        self.register_buffer("fixed_x", fixed_x.detach().reshape(1, 1, -1))

    def forward(self, X: Tensor) -> Tensor:
        if X.dim() == 2:
            X = X.unsqueeze(0)
        B = X.shape[0]
        x_exp = self.fixed_x.to(X).expand(B, 1, -1)
        return self.base.forward(torch.cat([x_exp, X], dim=1))


class OptimizationLoop:

    def __init__(self, black_box_func: SingleObjectiveProblem, model: ConstrainedDeoupledGPModelWrapper,
                 objective: Optional[MCAcquisitionObjective], ei_type: AcquisitionFunctionType, seed: int, budget: int,
                 performance_type: str, bounds: Tensor, results: Results,
                 penalty_value: Optional[Tensor] = torch.tensor([0.0]), number_initial_designs: Optional[int] = 6,
                 costs: Optional[Tensor] = None,
                 initial_train_x=None, initial_train_y=None, initial_budget_consumed: float = 0.0,
                 ):

        print("Starting Loop: OptimizationLoop")
        if costs is None:
            print("Using default costs")
            costs = torch.ones(model.getNumberOfOutputs())
        self.results = results
        self.objective = objective
        self.bounds = bounds
        self.black_box_func = black_box_func
        self.dim_x = self.black_box_func.dim
        self.seed = seed
        self.model_wrapper = model
        self.budget = budget
        self.performance_type = performance_type
        self.acquisition_function_type = ei_type
        self.number_of_outputs = self.model_wrapper.getNumberOfOutputs()
        self.penalty_value = penalty_value
        self.number_initial_designs = number_initial_designs
        self.costs = costs
        self.initial_train_x = initial_train_x
        self.initial_train_y = initial_train_y
        self.initial_budget_consumed = initial_budget_consumed

    def _initialize_state(self):
        """Either start fresh or resume from loaded data."""
        if self.initial_train_x is not None:
            train_x = self.initial_train_x
            train_y = self.initial_train_y
            budget_consumed = float(self.initial_budget_consumed)
            print(f"[resume] continuing from budget_consumed={budget_consumed} / target={self.budget}")
        else:
            train_x, train_y = self.generate_initial_data(n=self.number_initial_designs)
            budget_consumed = 0.0
        model = self.update_model(train_x, train_y)
        return train_x, train_y, model, budget_consumed

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y, model, budget_consumed = self._initialize_state()
        start_time = time.time()
        iteration = 0
        while budget_consumed < self.budget:
            best_observed_location, best_observed_value = self.best_observed(
                best_value_computation_type=self.performance_type,
                train_x=train_x,
                train_y=train_y,
                model=model,
                bounds=self.bounds)
            best_observed_all_sampled.append(best_observed_value)

            kg_values_list = torch.zeros(self.number_of_outputs, dtype=dtype)
            new_x_list = []
            for task_idx in range(self.number_of_outputs):
                # print("Running Task:", task_idx)
                acquisition_function = acquisition_function_factory(model=model,
                                                                    type=self.acquisition_function_type,
                                                                    objective=self.objective,
                                                                    best_value=best_observed_value,
                                                                    idx=task_idx,
                                                                    number_of_outputs=self.number_of_outputs,
                                                                    penalty_value=self.penalty_value,
                                                                    iteration=iteration,
                                                                    initial_condition_internal_optimizer=best_observed_location)

                new_x, kgvalue = self.compute_next_sample(acquisition_function=acquisition_function,
                                                          smart_initial_locations=best_observed_location)
                kg_values_list[task_idx] = kgvalue
                new_x_list.append(new_x)
            if (kg_values_list == 0).all():
                # All KG values are zero — fall back to coupled evaluation
                # at the best posterior mean location
                new_x = best_observed_location.detach().reshape(1, -1)
                for src_idx in range(self.number_of_outputs):
                    new_y = self.evaluate_black_box_func(new_x, src_idx)
                    train_x[src_idx] = torch.cat([train_x[src_idx].cpu(), new_x.cpu()])
                    train_y[src_idx] = torch.cat([train_y[src_idx].cpu(), new_y.cpu()])
                index = torch.tensor(0)
                budget_consumed += self.costs.sum()
            else:
                index = torch.argmax(torch.tensor(kg_values_list) / self.costs)
                new_y = self.evaluate_black_box_func(new_x_list[index], index)
                train_x[index] = torch.cat([train_x[index], new_x_list[index].cpu()])
                train_y[index] = torch.cat([train_y[index], new_y.cpu()])
                budget_consumed += self.costs[index]
            model = self.update_model(X=train_x, y=train_y)
            kg_str = ", ".join(f"src{i}={v:.5f}" for i, v in enumerate(kg_values_list))
            print(
                f"\nBatch{iteration:>2} finished: best value = "
                f"({best_observed_value:>4.5f}), KG values: [{kg_str}], "
                f"selected task {index}, "
                f"best location " + str(
                    best_observed_location.detach().cpu().numpy()) + " current sample decision x: " + str(
                    new_x_list[index].detach().cpu().numpy()) + "\n",
                end="",
            )
            self.save_parameters(train_x=train_x,
                                 train_y=train_y,
                                 model_length_scales=self.model_wrapper.get_model_length_scales(),
                                 best_predicted_location=best_observed_location,
                                 best_predicted_location_value=self.evaluate_location_true_quality(
                                     best_observed_location),
                                 acqf_recommended_location=new_x_list[index],
                                 acqf_recommended_location_true_value=None if self.black_box_func.is_expensive() else self.evaluate_location_true_quality(
                                     new_x_list[index]),
                                 acqf_recommended_output_index=index, acqf_values=kg_values_list,
                                 budget_consumed=budget_consumed)
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end = time.time() - start_time
        print(f'Total time: {end} seconds')

    def save_parameters(self, train_x, train_y, best_predicted_location, best_predicted_location_value,
                        acqf_recommended_output_index, acqf_recommended_location, acqf_recommended_location_true_value,
                        model_length_scales, budget_consumed, acqf_values=None, cost_configuration=None):
        self.results.random_seed(self.seed)
        self.results.save_budget(self.budget)
        self.results.save_model_length_scales(model_length_scales)
        self.results.save_input_data(train_x)
        self.results.save_output_data(train_y)
        self.results.save_acqf_values(acqf_values)
        self.results.save_number_initial_points(self.number_initial_designs)
        self.results.save_performance_type(self.performance_type)
        self.results.save_best_predicted_location(best_predicted_location)
        self.results.save_best_predicted_location_true_value(best_predicted_location_value)
        self.results.save_acqf_recommended_output_index(acqf_recommended_output_index)
        self.results.save_acqf_recommended_location(acqf_recommended_location)
        self.results.save_acqf_recommended_location_true_value(acqf_recommended_location_true_value)
        self.results.save_budget_consumed(budget_consumed)
        self.results.save_cost_configurations(cost_configuration)
        self.results.generate_pkl_file()

    def evaluate_location_true_quality(self, X):
        tasks_values = self.black_box_func.evaluate_black_box(X.cpu(), True)
        f_value = tasks_values[:, 0]
        if self.is_design_feasible(tasks_values):
            return f_value
        return -self.penalty_value

    def is_design_feasible(self, task_values):
        for idx in range(1, self.model_wrapper.getNumberOfOutputs()):
            c_val = task_values[:, idx]
            if c_val > 0:
                return False
        return True

    def evaluate_black_box_func(self, X, task_idx):
        return self.black_box_func.evaluate_task(X.cpu(), task_idx)

    def generate_initial_data(self, n: int):
        # generate training data
        train_x_list = []
        train_y_list = []
        sampler = qmc.LatinHypercube(d=self.dim_x, seed=self.seed)

        if self.black_box_func.is_expensive():
            train_x = torch.Tensor(sampler.random(n=n))
            output_tensor = self.black_box_func.evaluate_black_box(train_x, False)
            for i in range(self.model_wrapper.getNumberOfOutputs()):
                train_x_list += [train_x]
                train_y_list += [output_tensor[:, i]]
        else:
            for i in range(self.model_wrapper.getNumberOfOutputs()):
                train_x = torch.Tensor(sampler.random(n=n))
                train_x_list += [train_x]
                train_y_list += [self.evaluate_black_box_func(train_x, i)]

        return train_x_list, train_y_list

    def update_model(self, X, y):

        # Apply Gaussian copula to the objective if the function requests it
        if self.black_box_func.get_objective_transform() is not None:
            y = list(y)  # don't mutate the caller's list
            y[0] = gaussian_copula_transform(y[0])
        self.model_wrapper.fit(X, y)
        optimized_model = self.model_wrapper.optimize()
        return optimized_model

    def compute_important_idxs(self, model, new_x):
        """Find sources to evaluate for coupled cKG (Algorithm 1, line 6).

        Returns indices of the objective (0) plus constraints that are not
        trivially feasible (PF_k(x) < 1 - delta).  These are the sources
        worth evaluating at the coupled location.
        """
        INFEASIBILITY_THRESHOLD = 1e-7
        posterior_mean_feasibility = ConstrainedPosteriorMean(
            model=model, penalty_value=self.penalty_value,
        )
        ignore_list = []
        for i in range(1, self.number_of_outputs):
            pf_i = posterior_mean_feasibility.evaluate_feasibility_by_index(
                new_x, i,
            ).detach().item()
            if 1 - INFEASIBILITY_THRESHOLD <= pf_i:
                ignore_list.append(i)
        return list(set(range(self.number_of_outputs)) - set(ignore_list))

    def best_observed(self, best_value_computation_type, train_x, train_y, model, bounds):
        if best_value_computation_type == "sampled":
            return self.compute_best_sampled_value(train_x, train_y)
        elif best_value_computation_type == "model":
            return self.compute_best_posterior_mean(model, bounds)

    @staticmethod
    def compute_best_sampled_value(train_x, train_y):
        return train_x[torch.argmax(train_y)], torch.max(train_y)

    def compute_best_posterior_mean(self, model, bounds):
        argmax_mean, _ = optimize_acqf(
            acq_function=ConstrainedPosteriorMean(model, maximize=True, penalty_value=self.penalty_value),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=2048,
        )
        # Evaluate the true black-box at the predicted best location
        true_value = self.evaluate_location_true_quality(argmax_mean).item()
        return argmax_mean, true_value

    @staticmethod
    def _thompson_disc(source_acqf, bounds, n_grid=2048, seed=0):
        """Generate discretisation via optimised Thompson samples of the
        penalised constrained posterior G(x) = f(x)·PF(x) - M·(1-PF(x)).

        Uses the **same Z values** stored on source_acqf (z_y for decoupled,
        z_c for coupled), so the Thompson samples match the acquisition
        function's internal fantasies exactly.

        - ObjectiveDcKG:  Z_f = acqf.z_y, Z_c = 0 (PF at posterior mean)
        - ConstraintDcKG: Z_f = 0, Z_ck = acqf.z_y, Z_cj=0 for j!=k
        - CoupledCKG:     Z_f = acqf.z_y, Z_c = acqf.z_c (all vary)
        """

        model = source_acqf.model
        penalty_value = source_acqf.penalty_value
        dev = bounds.device
        K = source_acqf.K

        # Build Z_f and Z_c from the acqf's own stored samples
        if isinstance(source_acqf, CoupledCKG):
            Z_f = source_acqf.z_y                      # (n_zy,)
            Z_c = source_acqf.z_c                      # (n_zc, K)
            # Use the smaller of n_zy and n_zc as n_thompson
            # (each Thompson sample needs one Z_f and one Z_c row)
            n_thompson = min(len(Z_f), Z_c.shape[0])
            Z_f = Z_f[:n_thompson]
            Z_c = Z_c[:n_thompson]
        elif isinstance(source_acqf, ObjectiveDcKG):
            n_thompson = source_acqf.n_zy
            Z_f = source_acqf.z_y                      # (n_zy,)
            Z_c = torch.zeros(n_thompson, max(K, 1), device=dev, dtype=torch.double)
        elif isinstance(source_acqf, ConstraintDcKG):
            n_thompson = source_acqf.n_zy
            k = source_acqf.constraint_index
            Z_f = torch.zeros(n_thompson, device=dev, dtype=torch.double)
            Z_c = torch.zeros(n_thompson, max(K, 1), device=dev, dtype=torch.double)
            Z_c[:, k] = source_acqf.z_y
        else:
            raise ValueError(f"Unknown source type: {type(source_acqf)}")

        # Warm-start: compute posteriors once on grid, vectorised over all Z samples
        X_grid = draw_sobol_samples(
            bounds=bounds, n=n_grid, q=1, seed=seed + 7,
        ).to(dev)  # (n_grid, d)

        with torch.no_grad():
            X_g = X_grid # (n_grid, 1, d)
            obj_p = model.models[0].posterior(X_g)
            mu_f = obj_p.mean.reshape(n_grid, 1) # (n_grid, 1)
            sig_f = obj_p.variance.reshape(n_grid, 1).clamp_min(1e-12).sqrt() # (n_grid, 1)

            f_all = mu_f + sig_f * Z_f.unsqueeze(0) # (n_grid, n_z)
            log_pf_all = torch.zeros(n_grid, n_thompson, device=dev, dtype=torch.double)
            for ck in range(K):
                cp = model.models[ck + 1].posterior(X_g)
                mu_c = cp.mean.reshape(n_grid, 1)
                sig_c = cp.variance.reshape(n_grid, 1).clamp_min(1e-12).sqrt()
                c_all = mu_c + sig_c * Z_c[:, ck].unsqueeze(0)
                log_pf_all = log_pf_all + torch.nn.functional.logsigmoid(-c_all * 10)
            pf_all = log_pf_all.exp()

            G_all = f_all * pf_all - penalty_value * (1 - pf_all)
            best_G_idx = G_all.argmax(dim=0)  # (n_thompson,)
            best_G_val = G_all.max(dim=0).values  # (n_thompson,)
            best_pf_idx = log_pf_all.argmax(dim=0)  # (n_thompson,)
            # If best G equals -penalty (all infeasible), use max-PF point
            infeasible = (best_G_val <= -penalty_value + 1e-6)
            best_idx = torch.where(infeasible, best_pf_idx, best_G_idx)
            x0 = X_grid[best_idx].unsqueeze(1)  # (n_thompson, 1, d)
            # (n_thompson, 1, d)

        # Batched Thompson objective for L-BFGS (reuses same Z values)
        class _BatchThompson(torch.nn.Module):
            def __init__(self, mdl, pen, zf, zc, k):
                super().__init__()
                self.mdl, self.pen, self.zf, self.zc, self.k = mdl, pen, zf, zc, k
            def forward(self, X):
                op = self.mdl.models[0].posterior(X)
                mf = op.mean.squeeze(-1).squeeze(-1)
                sf = op.variance.squeeze(-1).squeeze(-1).clamp_min(1e-12).sqrt()
                fs = mf + sf * self.zf
                log_pf = torch.zeros_like(fs)
                for j in range(self.k):
                    cp = self.mdl.models[j + 1].posterior(X)
                    mc = cp.mean.squeeze(-1).squeeze(-1)
                    sc = cp.variance.squeeze(-1).squeeze(-1).clamp_min(1e-12).sqrt()
                    log_pf = log_pf + torch.nn.functional.logsigmoid(-(mc + sc * self.zc[:, j]) * 10)
                pf = log_pf.exp()
                return fs * pf - self.pen * (1 - pf)

        batch_obj = _BatchThompson(model, penalty_value, Z_f, Z_c, K)
        with torch.enable_grad():
            x_opt, _ = gen_candidates_torch(
                initial_conditions=x0,
                acquisition_function=batch_obj,
                lower_bounds=bounds[0], upper_bounds=bounds[1],
                options={"maxiter": 100},
            )
        return x_opt[:, 0, :].detach()

    def _optimize_fast_ckg(self, acqf, num_restarts=8, raw_samples=512, eta=2.0,
                           disc_init="thompson"):
        """Optimize FastConstrainedKG (CoupledCKG).

        Args:
            disc_init: "sobol" for gen_batch_initial_conditions only (original),
                       "thompson" for n_zc Thompson points injected into the
                       first disc slots, rest from gen_batch_initial_conditions.
        """


        n_disc = 64
        q_aug = 1 + n_disc
        dev = acqf.x_best.device
        bounds = self.bounds.to(dev)

        ics = gen_batch_initial_conditions(
            acq_function=acqf, bounds=bounds, q=q_aug,
            num_restarts=num_restarts, raw_samples=raw_samples,
            options={"seed": self.seed, "eta": eta},
        )

        if disc_init == "thompson":
            thompson_pts = self._thompson_disc(
                acqf, bounds, seed=self.seed,
            )
            n_thompson = thompson_pts.shape[0]
            ics[:, 1:1 + n_thompson, :] = thompson_pts.squeeze(1)

        xbest_restart = ics[-1:].clone()
        xbest_restart[:, 0:1, :] = acqf.x_best
        ics = torch.cat([ics, xbest_restart], dim=0)

        candidates, acqf_value = optimize_acqf(
            acq_function=acqf, bounds=bounds, q=q_aug,
            num_restarts=ics.shape[0], batch_initial_conditions=ics,
            options={"maxiter": 100},
        )
        x_optimised = candidates[0:1, :].detach()
        x_optimised_val = acqf_value.detach()
        return torch.atleast_2d(x_optimised), x_optimised_val

    @staticmethod
    def _optimize_all_sources_dckg(acqf, bounds, num_restarts=15, raw_samples=72,
                                   frac_random=0.1, eta=2.0,
                                   num_inner_restarts=20, raw_inner_samples=1024,
                                   seed=0):
        """Optimize AllSourcesDcKG with per-source initialised discretisations.

        Each source in acqf.sources is an independent acquisition function.
        We initialise each one separately (like _optimize_fast_ckg), then
        assemble the per-source initial conditions into the joint tensor,
        evaluate with the joint forward, select top restarts, and run L-BFGS.

        Steps:
        1. Find value function maximisers (shared across all sources).
        2. For each source independently: use gen_batch_initial_conditions
           to get good (candidate + disc) tensors, then overwrite disc slots
           with x_best + softmax-sampled maximisers.
        3. Assemble per-source ICs into joint tensors, evaluate with the
           joint forward, select top num_restarts.
        4. Run L-BFGS on the joint tensor.
        """



        S = acqf.n_sources
        Q = acqf.q_per_source
        q_total = S * Q
        n_disc = acqf.n_disc
        dev = acqf.x_best.device
        bounds = bounds.to(dev)

        # --- Step 1: shared value function maximisers ---
        fantasy_cands, fantasy_vals = optimize_acqf(
            acq_function=ConstrainedPosteriorMean(
                model=acqf.model, penalty_value=acqf.penalty_value,
            ),
            bounds=bounds, q=1,
            num_restarts=num_inner_restarts,
            raw_samples=raw_inner_samples,
            return_best_only=False,
        )  # (num_inner_restarts, 1, d)
        fantasy_cands = fantasy_cands.detach()
        fantasy_vals = fantasy_vals.detach()

        std = fantasy_vals.std()
        if std > 0:
            weights = torch.exp(eta * standardize(fantasy_vals))
        else:
            weights = torch.ones_like(fantasy_vals)

        n_fantasy_slots = n_disc - 1  # disc slots after x_best
        n_value = int((1 - frac_random) * n_fantasy_slots)

        # --- Step 2: per-source initialisation ---
        # gen_batch_initial_conditions uses torch.no_grad() internally
        per_source_ics = []  # each: (num_restarts, Q, d)
        for s, source_acqf in enumerate(acqf.sources):
            ics_s = gen_batch_initial_conditions(
                acq_function=source_acqf,
                bounds=bounds,
                q=Q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"seed": seed + s, "eta": eta},
            )  # (num_restarts, Q, d)

            # Overwrite disc slots with value fn maximisers
            # (x_best is appended inside each acqf's _evaluate())
            if n_value > 0 and fantasy_vals.numel() > 0:
                idx = torch.multinomial(weights, num_restarts * n_value,
                                        replacement=True)
                ics_s[..., -n_value:, :] = fantasy_cands[idx, 0].view(
                    num_restarts, n_value, -1,
                )

            per_source_ics.append(ics_s)

        # --- Step 3: assemble joint tensor and evaluate ---
        # Stack per-source ICs: for restart r, source s → per_source_ics[s][r]
        # Joint tensor: (num_restarts, S*Q, d)
        ics = torch.cat(per_source_ics, dim=1)  # (num_restarts, q_total, d)

        # Re-evaluate with the joint forward to find the best restarts
        with torch.no_grad():
            joint_vals = acqf(ics)
        k = min(num_restarts, len(joint_vals))
        _, top = torch.topk(joint_vals, k)
        ics = ics[top]

        # Add one extra restart with x_best as candidate for every source
        xbest_ic = ics[-1:].clone()
        for s in range(S):
            xbest_ic[0, s * Q, :] = acqf.x_best.squeeze(0)
        ics = torch.cat([ics, xbest_ic], dim=0)

        # --- Step 4: joint L-BFGS ---
        candidates, acqf_value = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q_total,
            num_restarts=ics.shape[0],
            batch_initial_conditions=ics,
            options={"maxiter": 100},
        )

        return candidates.detach(), acqf_value.detach()


def compute_next_sample(self, acquisition_function, smart_initial_locations=None):

    if isinstance(acquisition_function, FastConstrainedKG):
        return self._optimize_fast_ckg(
            acquisition_function,
            smart_initial_locations=smart_initial_locations,
        )
    # Legacy path
    opt_kwargs = dict(
        acq_function=acquisition_function,
        bounds=self.bounds,
        q=1,
        num_restarts=15,
        raw_samples=72,
        options={"maxiter": 100},
        gen_candidates=gen_candidates_scipy,
    )
    candidates, acqf_value = optimize_acqf(**opt_kwargs)
    # observe new values
    x_optimised = candidates.detach()
    x_optimised_val = acqf_value.detach()
    if smart_initial_locations is not None:
        if isinstance(acquisition_function, DecopledHybridConstrainedKnowledgeGradient):
            acquisition_function.set_scipy_as_internal_optimizer()
        candidates, kgvalue = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.bounds,
            num_restarts=smart_initial_locations.shape[0],
            batch_initial_conditions=smart_initial_locations,
            q=1,
            options={"maxiter": 100}
        )
        x_smart_optimised = candidates.detach()
        x_smart_optimised_val = kgvalue.detach()
        if x_smart_optimised_val >= x_optimised_val:
            return torch.atleast_2d(x_smart_optimised), x_smart_optimised_val
    return torch.atleast_2d(x_optimised), acqf_value


class CoupledAndDecoupledOptimizationLoop(OptimizationLoop):

    def __init__(self, black_box_func: SingleObjectiveProblem, model: ConstrainedDeoupledGPModelWrapper,
                 objective: Optional[MCAcquisitionObjective], ei_type: AcquisitionFunctionType, seed: int, budget: int,
                 performance_type: str, bounds: Tensor, results: Results,
                 penalty_value: Optional[Tensor] = torch.tensor([0.0]), number_initial_designs: Optional[int] = 6,
                 costs: Optional[Tensor] = None, **kwargs):

        super().__init__(black_box_func, model, objective, ei_type, seed, budget, performance_type, bounds, results,
                         penalty_value, number_initial_designs, costs, **kwargs)

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y, model, budget_consumed = self._initialize_state()

        start_time = time.time()
        iteration = 0
        while budget_consumed <= self.budget:
            iteration += 1
            best_observed_location, best_observed_value = self.best_observed(
                best_value_computation_type=self.performance_type,
                train_x=train_x,
                train_y=train_y,
                model=model,
                bounds=self.bounds)
            best_observed_all_sampled.append(best_observed_value)

            kg_values_list = torch.zeros(self.number_of_outputs + 1, dtype=dtype)
            new_x_list = []
            for task_idx in range(self.number_of_outputs):
                acquisition_function = acquisition_function_factory(model=model,
                                                                    type=self.acquisition_function_type,
                                                                    objective=self.objective,
                                                                    best_value=best_observed_value,
                                                                    idx=task_idx,
                                                                    number_of_outputs=self.number_of_outputs,
                                                                    penalty_value=self.penalty_value,
                                                                    iteration=iteration,
                                                                    initial_condition_internal_optimizer=best_observed_location)

                new_x, kgvalue = self.compute_next_sample(acquisition_function=acquisition_function,
                                                          smart_initial_locations=best_observed_location)
                kg_values_list[task_idx] = kgvalue
                new_x_list.append(new_x)
            new_x_ckg, acqf_value_ckg = self.get_best_coupled_kg_value(best_observed_location,
                                                                       best_observed_value,
                                                                       iteration, model)
            idx_to_eval = self.compute_important_idxs(model, new_x_ckg)
            total_cost_filtered = torch.sum(self.costs[idx_to_eval])
            best_ckG_value_per_cost = acqf_value_ckg / total_cost_filtered
            best_dckg_value_per_cost = torch.max(torch.tensor(kg_values_list[:-1]) / self.costs)
            kg_values_list[-1] = best_ckG_value_per_cost

            all_zero = (kg_values_list == 0).all()

            if all_zero:
                # All KG values are zero — fall back to coupled evaluation
                # at the best posterior mean location
                location_to_sample = best_observed_location.detach().reshape(1, -1)
                new_output = self.black_box_func.evaluate_black_box(location_to_sample.cpu(), False)
                for task_idx in range(self.number_of_outputs):
                    train_x[task_idx] = torch.cat([train_x[task_idx], location_to_sample.cpu()])
                    train_y[task_idx] = torch.cat([train_y[task_idx], new_output[:, task_idx].cpu()])
                index = list(range(self.number_of_outputs))
                budget_consumed += self.costs.sum()
            elif best_ckG_value_per_cost > best_dckg_value_per_cost:  # Run coupled cKG
                new_output = self.black_box_func.evaluate_black_box(new_x_ckg.cpu(), False)
                for task_idx in idx_to_eval:
                    train_x[task_idx] = torch.cat([train_x[task_idx], new_x_ckg.cpu()])
                    train_y[task_idx] = torch.cat([train_y[task_idx], new_output[:, task_idx].cpu()])
                index = idx_to_eval  # Will have to change for non-ones costs
                location_to_sample = new_x_ckg
                budget_consumed += torch.sum(total_cost_filtered)
            else:  # Run dcKG
                index = torch.argmax(torch.tensor(kg_values_list[:-1]) / self.costs)
                new_y = self.evaluate_black_box_func(new_x_list[index], index)
                train_x[index] = torch.cat([train_x[index], new_x_list[index].cpu()])
                train_y[index] = torch.cat([train_y[index], new_y.cpu()])
                location_to_sample = new_x_list[index]
                index = [index.item()]
                budget_consumed += torch.sum(self.costs[index])
            model = self.update_model(X=train_x, y=train_y)
            kg_str = ", ".join(f"src{i}={v:.5f}" for i, v in enumerate(kg_values_list))
            print(
                f"\nBatch{iteration:>2} finished: best value = "
                f"({best_observed_value:>4.5f}), KG values: [{kg_str}], "
                f"selected tasks {index}, "
                f"best location " + str(
                    best_observed_location.detach().cpu().numpy()) + " current sample decision x: " + str(
                    location_to_sample.detach().cpu().numpy()) + "\n",
                end="", )

            self.save_parameters(train_x=train_x,
                                 train_y=train_y,
                                 model_length_scales=self.model_wrapper.get_model_length_scales(),
                                 best_predicted_location=best_observed_location,
                                 best_predicted_location_value=self.evaluate_location_true_quality(
                                     best_observed_location),
                                 acqf_recommended_location=location_to_sample,
                                 acqf_recommended_location_true_value=None if self.black_box_func.is_expensive() else self.evaluate_location_true_quality(
                                     location_to_sample),
                                 acqf_recommended_output_index=index,
                                 acqf_values=kg_values_list,
                                 budget_consumed=budget_consumed,
                                 cost_configuration=self.costs)

            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end = time.time() - start_time
        print(f'Total time: {end} seconds')

    def compute_important_idxs(self, model, new_x_ckg):
        INFEASIBILITY_THRESHOLD = 1e-7
        posterior_mean_feasibility = ConstrainedPosteriorMean(model=model, penalty_value=self.penalty_value)
        ignore_list = []
        for i in range(1, self.number_of_outputs):
            feasibility_value = posterior_mean_feasibility.evaluate_feasibility_by_index(new_x_ckg, i).detach().item()
            if 1 - INFEASIBILITY_THRESHOLD <= feasibility_value:
                ignore_list.append(i)
        idx_to_eval = list(set(range(self.number_of_outputs)) - set(ignore_list))
        return idx_to_eval

    def get_best_coupled_kg_value(self, best_observed_location, best_observed_value, iteration, model):
        acquisition_function = acquisition_function_factory(model=model,
                                                            type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                                            objective=self.objective,
                                                            best_value=best_observed_value,
                                                            idx=None,
                                                            number_of_outputs=self.number_of_outputs,
                                                            penalty_value=self.penalty_value,
                                                            iteration=iteration,
                                                            initial_condition_internal_optimizer=best_observed_location)
        new_x, kg_value = self.compute_next_sample(acquisition_function=acquisition_function,
                                                   smart_initial_locations=best_observed_location)
        return new_x, kg_value


class EI_Decoupled_OptimizationLoop(OptimizationLoop):

    def __init__(self, black_box_func: SingleObjectiveProblem, model: ConstrainedDeoupledGPModelWrapper,
                 objective: Optional[MCAcquisitionObjective], ei_type: AcquisitionFunctionType, seed: int, budget: int,
                 performance_type: str, bounds: Tensor, results: Results,
                 penalty_value: Optional[Tensor] = torch.tensor([0.0]), number_initial_designs: Optional[int] = 6,
                 costs: Optional[Tensor] = None, **kwargs):

        super().__init__(black_box_func, model, objective, ei_type, seed, budget, performance_type, bounds, results,
                         penalty_value, number_initial_designs, costs, **kwargs)

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y, model, consumed_budget = self._initialize_state()

        start_time = time.time()
        iteration = 0
        while consumed_budget < self.budget:
            iteration += 1
            best_observed_location, best_observed_value = self.best_observed(
                best_value_computation_type=self.performance_type,
                train_x=train_x,
                train_y=train_y,
                model=model,
                bounds=self.bounds)
            best_observed_all_sampled.append(best_observed_value)

            acquisition_function = acquisition_function_factory(model=model,
                                                                type=self.acquisition_function_type,
                                                                objective=self.objective,
                                                                best_value=best_observed_value,
                                                                idx=1,
                                                                number_of_outputs=self.number_of_outputs,
                                                                penalty_value=self.penalty_value,
                                                                iteration=iteration,
                                                                initial_condition_internal_optimizer=best_observed_location)
            new_x, kg_val = self.compute_next_sample(acquisition_function=acquisition_function,
                                                     smart_initial_locations=best_observed_location)
            posterior = model.posterior(new_x)
            mu = posterior.mean
            std = posterior.variance.sqrt().clamp_min(1e-9)
            z = -mu / std
            probability_infeasibility = []
            size = self.model_wrapper.getNumberOfOutputs()
            for i in range(0, size):
                probability_infeasibility = probability_infeasibility + [
                    1 - torch.distributions.Normal(0, 1).cdf(z)[0][i].detach().item()]
            evaluation_order = sorted(range(len(probability_infeasibility)), key=probability_infeasibility.__getitem__)[
                               ::-1]
            evaluated_idx = []
            i = 0  # index for evaluation order
            failing_constraint = None
            while i < size:
                new_y = self.evaluate_black_box_func(new_x, evaluation_order[i])
                consumed_budget += self.costs[evaluation_order[i]]
                train_x[evaluation_order[i]] = torch.cat([train_x[evaluation_order[i]], new_x.cpu()])
                train_y[evaluation_order[i]] = torch.cat([train_y[evaluation_order[i]], new_y.cpu()])
                model = self.update_model(X=train_x, y=train_y)
                evaluated_idx.append(evaluation_order[i])
                if new_y < 0:
                    i = i + 1
                else:
                    failing_constraint = i
                    i = size

            print(
                f"\nBatch{iteration:>2} finished: best value = "
                f"({best_observed_value:>4.5f}), acqf value = {kg_val:.5f}, "
                f"best location " + str(
                    best_observed_location.detach().cpu().numpy()) + " current sample decision x: " + str(
                    new_x.detach().cpu().numpy()), end="\n"
            )

            print(f'Evaluated functions: {evaluated_idx}')
            self.save_parameters(train_x=train_x,
                                 train_y=train_y,
                                 model_length_scales=self.model_wrapper.get_model_length_scales(),
                                 best_predicted_location=best_observed_location,
                                 best_predicted_location_value=self.evaluate_location_true_quality(
                                     best_observed_location),
                                 acqf_recommended_location=new_x,
                                 acqf_recommended_location_true_value=None if self.black_box_func.is_expensive() else self.evaluate_location_true_quality(
                                     new_x),
                                 failing_constraint=failing_constraint,
                                 func_evals=evaluated_idx,
                                 consumed_budget=consumed_budget)  # last one gives index of failing constraint
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end = time.time() - start_time
        print(f'Total time: {end} seconds')

    def save_parameters(self, train_x, train_y, best_predicted_location, best_predicted_location_value,
                        acqf_recommended_location, acqf_recommended_location_true_value, failing_constraint, func_evals,
                        consumed_budget,
                        **kwargs):

        self.results.random_seed(self.seed)
        self.results.save_budget(self.budget)
        self.results.save_model_length_scales(kwargs["model_length_scales"])
        self.results.save_input_data(train_x)
        self.results.save_output_data(train_y)
        self.results.save_number_initial_points(self.number_initial_designs)
        self.results.save_performance_type(self.performance_type)
        self.results.save_best_predicted_location(best_predicted_location)
        self.results.save_best_predicted_location_true_value(best_predicted_location_value)
        self.results.save_acqf_recommended_location(acqf_recommended_location)
        self.results.save_acqf_recommended_location_true_value(acqf_recommended_location_true_value)
        self.results.save_failing_constraint(failing_constraint)
        self.results.save_evaluated_functions(func_evals)
        self.results.save_budget_consumed(consumed_budget)
        self.results.generate_pkl_file()

    def compute_next_sample(self, acquisition_function, smart_initial_locations=None):
    
        if isinstance(acquisition_function, FastConstrainedKG):
            return self._optimize_fast_ckg(
                acquisition_function,
            )
        candidates, kgvalue = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.bounds,
            q=1,
            num_restarts=15,  # can make smaller if too slow, not too small though
            raw_samples=72,  # used for intialization heuristic
            options={"maxiter": 100},
        )
        # observe new values
        x_optimised = candidates.detach()
        x_optimised_val = kgvalue.detach()
        if smart_initial_locations is not None:
            candidates, kgvalue = optimize_acqf(
                acq_function=acquisition_function,
                bounds=self.bounds,
                num_restarts=smart_initial_locations.shape[0],
                batch_initial_conditions=smart_initial_locations,
                q=1,
                options={"maxiter": 100}
            )
            x_smart_optimised = candidates.detach()
            x_smart_optimised_val = kgvalue.detach()
            if x_smart_optimised_val >= x_optimised_val:
                return x_smart_optimised[None, :], x_smart_optimised_val

        return x_optimised, x_optimised_val


class EI_OptimizationLoop(OptimizationLoop):

    def __init__(self, black_box_func: SingleObjectiveProblem, model: ConstrainedDeoupledGPModelWrapper,
                 objective: Optional[MCAcquisitionObjective], ei_type: AcquisitionFunctionType, seed: int, budget: int,
                 performance_type: str, bounds: Tensor, results: Results,
                 penalty_value: Optional[Tensor] = torch.tensor([0.0]), number_initial_designs: Optional[int] = 6,
                 costs: Optional[Tensor] = None, **kwargs):

        super().__init__(black_box_func, model, objective, ei_type, seed, budget, performance_type, bounds, results,
                         penalty_value, number_initial_designs, costs, **kwargs)

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y, model, _ = self._initialize_state()
        start_iter = (train_x[0].shape[0] - self.number_initial_designs) if self.initial_train_x is not None else 0

        start_time = time.time()
        for iteration in range(start_iter, self.budget):
            best_observed_location, best_observed_value = self.best_observed(
                best_value_computation_type=self.performance_type,
                train_x=train_x,
                train_y=train_y,
                model=model,
                bounds=self.bounds)
            best_observed_all_sampled.append(best_observed_value)

            acquisition_function = acquisition_function_factory(model=model,
                                                                type=self.acquisition_function_type,
                                                                objective=self.objective,
                                                                best_value=best_observed_value,
                                                                idx=1,
                                                                number_of_outputs=self.number_of_outputs,
                                                                penalty_value=self.penalty_value,
                                                                iteration=iteration,
                                                                initial_condition_internal_optimizer=best_observed_location)
            initialization = self.get_smart_initialization(acquisition_function, model, best_observed_location)
            new_x, kg_val = self.compute_next_sample(acquisition_function=acquisition_function,
                                                     smart_initial_locations=initialization)

            new_y = self.black_box_func.evaluate_black_box(new_x.cpu(), False)
            for i in range(self.model_wrapper.getNumberOfOutputs()):
                train_x[i] = torch.cat([train_x[i].cpu(), new_x.cpu()])
                train_y[i] = torch.cat([train_y[i].cpu(), new_y[:, i].cpu()])
            model = self.update_model(X=train_x, y=train_y)

            best_observed_location_value = self.evaluate_location_true_quality(
                best_observed_location).detach().cpu().numpy()
            print(f"\nBatch{iteration:>2} finished: best value = {best_observed_location_value}, "
                  f"acqf value = {kg_val:.5f}, best location " + str(
                best_observed_location.detach().cpu().numpy()) + " current sample decision x: " + str(
                new_x.detach().cpu().numpy()), end="\n")

            self.save_parameters(train_x=train_x,
                                 train_y=train_y,
                                 best_predicted_location=best_observed_location,
                                 best_predicted_location_value=best_observed_location_value,
                                 acqf_recommended_location=new_x,
                                 acqf_recommended_location_true_value=None if self.black_box_func.is_expensive() else self.evaluate_location_true_quality(
                                     new_x),
                                 acqf_values=[kg_val])
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end = time.time() - start_time
        print(f'Total time: {end} seconds')

    def get_smart_initialization(self, acquisition_function, model, best_observed_location):
    
        if isinstance(acquisition_function, (DecopledHybridConstrainedKnowledgeGradient, FastConstrainedKG)):
            # Detect model device so LHS points go to the right place
            try:
                _dev = next(model.parameters()).device
            except StopIteration:
                _dev = torch.device("cpu")
            sampler = qmc.LatinHypercube(d=self.dim_x, seed=self.seed)
            test_x = torch.tensor(sampler.random(n=1000), dtype=torch.double, device=_dev)
            constrained_posterior_mean = ConstrainedPosteriorMean(model, maximize=True,
                                                                  penalty_value=self.penalty_value)
            feasibility = constrained_posterior_mean._compute_feasibility(test_x)
            feasible_x_locations = test_x[feasibility > 0.1, :]
            if feasible_x_locations.shape[0] == 0:
                objective = constrained_posterior_mean._evaluate_objective(test_x)
                best_x_location = test_x[torch.argmax(objective), :]
            else:
                objective = constrained_posterior_mean._evaluate_objective(feasible_x_locations)
                best_x_location = feasible_x_locations[torch.argmax(objective), :]
            return torch.vstack([best_x_location[None, :], best_observed_location])
        else:
            return best_observed_location

    def save_parameters(self, train_x, train_y, best_predicted_location, best_predicted_location_value,
                        acqf_recommended_location, acqf_recommended_location_true_value, acqf_values, **kwargs):

        self.results.random_seed(self.seed)
        self.results.save_budget(self.budget)
        # self.results.save_model_length_scales(kwargs["model_length_scales"])
        self.results.save_input_data(train_x)
        self.results.save_output_data(train_y)
        self.results.save_number_initial_points(self.number_initial_designs)
        self.results.save_performance_type(self.performance_type)
        self.results.save_best_predicted_location(best_predicted_location)
        self.results.save_best_predicted_location_true_value(best_predicted_location_value)
        self.results.save_acqf_recommended_location(acqf_recommended_location)
        self.results.save_acqf_recommended_location_true_value(acqf_recommended_location_true_value)
        self.results.save_acqf_values(acqf_values)
        self.results.generate_pkl_file()

    def compute_next_sample(self, acquisition_function, smart_initial_locations=None):
        if isinstance(acquisition_function, FastConstrainedKG):
            return self._optimize_fast_ckg(
                acquisition_function,
            )
        candidates, kgvalue = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.bounds,
            q=1,
            num_restarts=15,
            raw_samples=72,
            options={"maxiter": 100}
        )
        x_optimised = candidates.detach()
        x_optimised_val = kgvalue.detach()
        if smart_initial_locations is not None:
            if isinstance(acquisition_function, DecopledHybridConstrainedKnowledgeGradient):
                acquisition_function.set_scipy_as_internal_optimizer()
            x_smart_optimised, x_smart_optimised_val = optimize_acqf(
                acq_function=acquisition_function,
                bounds=self.bounds,
                num_restarts=1,
                batch_initial_conditions=smart_initial_locations,
                q=1,
                options={"maxiter": 100})

            if x_smart_optimised_val >= x_optimised_val:
                return torch.atleast_2d(x_smart_optimised), x_smart_optimised_val
        return torch.atleast_2d(x_optimised), x_optimised_val


class Decoupled_EIKG_OptimizationLoop(OptimizationLoop):

    def __init__(self, black_box_func: SingleObjectiveProblem, model: ConstrainedDeoupledGPModelWrapper,
                 objective: Optional[MCAcquisitionObjective], ei_type: AcquisitionFunctionType, seed: int, budget: int,
                 performance_type: str, bounds: Tensor, results: Results,
                 penalty_value: Optional[Tensor] = torch.tensor([0.0]), number_initial_designs: Optional[int] = 6,
                 costs: Optional[Tensor] = None, **kwargs):

        super().__init__(black_box_func, model, objective, ei_type, seed, budget, performance_type, bounds, results,
                         penalty_value, number_initial_designs, costs, **kwargs)

    def _evaluate_per_source_kg_v2(self, model, new_x, best_observed_location, iteration,
                                   n_fantasies=7, n_disc=64, num_restarts=15, raw_samples=72):
        """Per-source V2 dcKG values with candidate fixed, discretisation optimised.

        Returns a tensor of length self.number_of_outputs (objective + K constraints).
        """
        K = self.number_of_outputs - 1
        common = dict(
            model=model,
            penalty_value=self.penalty_value,
            x_best_location=best_observed_location,
            objective=self.objective,
            n_fantasies=n_fantasies,
            seed=iteration,
        )
        sources = [ObjectiveDcKG(**common)]
        for k in range(K):
            sources.append(ConstraintDcKG(constraint_index=k, **common))

        kg_values = torch.zeros(len(sources), dtype=dtype)
        for s, base_acqf in enumerate(sources):
            wrapped = _FixedCandidateV2DcKG(base_acqf, new_x)
            ics = gen_batch_initial_conditions(
                acq_function=wrapped, bounds=self.bounds, q=n_disc,
                num_restarts=num_restarts, raw_samples=raw_samples,
                options={"seed": iteration + s * 100},
            )
            _, vals = optimize_acqf(
                acq_function=wrapped, bounds=self.bounds, q=n_disc,
                num_restarts=ics.shape[0], batch_initial_conditions=ics,
                return_best_only=False, options={"maxiter": 1000},
            )
            kg_values[s] = vals.max().detach()
            print("kgvals: " , kg_values)
        return kg_values

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y, model, budget_consumed = self._initialize_state()

        start_time = time.time()
        iteration = 0
        while budget_consumed < self.budget:
            iteration += 1
            best_observed_location, best_observed_value = self.best_observed(
                best_value_computation_type=self.performance_type,
                train_x=train_x,
                train_y=train_y,
                model=model,
                bounds=self.bounds)
            best_observed_all_sampled.append(best_observed_value)

            acquisition_function = acquisition_function_factory(model=model,
                                                                type=self.acquisition_function_type,
                                                                objective=self.objective,
                                                                best_value=best_observed_value,
                                                                idx=None,
                                                                number_of_outputs=self.number_of_outputs,
                                                                penalty_value=self.penalty_value,
                                                                iteration=iteration,
                                                                initial_condition_internal_optimizer=best_observed_location)

            new_x, _ = self.compute_next_sample(acquisition_function=acquisition_function,
                                                smart_initial_locations=best_observed_location)
            kg_values_list = self._evaluate_per_source_kg_v2(
                model=model,
                new_x=new_x,
                best_observed_location=best_observed_location,
                iteration=iteration,
            )

            index = torch.argmax(torch.tensor(kg_values_list) / self.costs)
            new_y = self.evaluate_black_box_func(new_x, index)
            train_x[index] = torch.cat([train_x[index], new_x.cpu()])
            train_y[index] = torch.cat([train_y[index], new_y.cpu()])
            model = self.update_model(X=train_x, y=train_y)
            budget_consumed += self.costs[index]
            kg_str = ", ".join(f"src{i}={v:.5f}" for i, v in enumerate(kg_values_list))
            print(
                f"\nBatch{iteration:>2} finished: best value = "
                f"({best_observed_value:>4.5f}), KG values: [{kg_str}], "
                f"selected task {index}, "
                f"best location " + str(
                    best_observed_location.detach().cpu().numpy()) + " current sample decision x: " + str(
                    new_x.detach().cpu().numpy()), end="\n"
            )

            self.save_parameters(train_x=train_x,
                                 train_y=train_y,
                                 best_predicted_location=best_observed_location,
                                 model_length_scales=self.model_wrapper.get_model_length_scales(),
                                 best_predicted_location_value=self.evaluate_location_true_quality(
                                     best_observed_location),
                                 acqf_recommended_output_index=index,
                                 acqf_recommended_location=new_x,
                                 acqf_recommended_location_true_value=None if self.black_box_func.is_expensive() else self.evaluate_location_true_quality(
                                     new_x),
                                 failing_constraint="None",
                                 acqf_values=kg_values_list,
                                 budget_consumed=budget_consumed)
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end = time.time() - start_time
        print(f'Total time: {end} seconds')

    def save_parameters(self, train_x, train_y, best_predicted_location, best_predicted_location_value,
                        acqf_recommended_output_index, acqf_recommended_location, acqf_recommended_location_true_value,
                        failing_constraint, acqf_values, budget_consumed, **kwargs):

        self.results.random_seed(self.seed)
        self.results.save_budget(self.budget)
        self.results.save_input_data(train_x)
        self.results.save_output_data(train_y)
        self.results.save_model_length_scales(kwargs["model_length_scales"])
        self.results.save_number_initial_points(self.number_initial_designs)
        self.results.save_performance_type(self.performance_type)
        self.results.save_best_predicted_location(best_predicted_location)
        self.results.save_best_predicted_location_true_value(best_predicted_location_value)
        self.results.save_acqf_recommended_output_index(acqf_recommended_output_index)
        self.results.save_acqf_recommended_location(acqf_recommended_location)
        self.results.save_acqf_values(acqf_values)
        self.results.save_acqf_recommended_location_true_value(acqf_recommended_location_true_value)
        self.results.save_failing_constraint(failing_constraint)
        self.results.save_budget_consumed(budget_consumed)

        self.results.generate_pkl_file()

    def compute_next_sample(self, acquisition_function, smart_initial_locations=None):
        candidates, kgvalue = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.bounds,
            q=1,
            num_restarts=15,  # can make smaller if too slow, not too small though
            raw_samples=72,  # used for intialization heuristic
            options={"maxiter": 100},
        )
        # observe new values
        x_optimised = candidates.detach()
        x_optimised_val = kgvalue.detach()
        if smart_initial_locations is not None:
            candidates, kgvalue = optimize_acqf(
                acq_function=acquisition_function,
                bounds=self.bounds,
                num_restarts=smart_initial_locations.shape[0],
                batch_initial_conditions=smart_initial_locations,
                q=1,
                options={"maxiter": 100}
            )
            x_smart_optimised = candidates.detach()
            x_smart_optimised_val = kgvalue.detach()
            if x_smart_optimised_val >= x_optimised_val:
                return x_smart_optimised[None, :], x_smart_optimised_val

        return x_optimised, x_optimised_val


class OPT_UCB_OptimizationLoop(OptimizationLoop):

    def __init__(self, black_box_func: SingleObjectiveProblem, model: ConstrainedDeoupledGPModelWrapper,
                 objective: Optional[MCAcquisitionObjective], ei_type: AcquisitionFunctionType, seed: int, budget: int,
                 performance_type: str, bounds: Tensor, results: Results,
                 penalty_value: Optional[Tensor] = torch.tensor([0.0]), number_initial_designs: Optional[int] = 6,
                 costs: Optional[Tensor] = None):
        super().__init__(black_box_func, model, objective, ei_type, seed, budget, performance_type, bounds, results,
                         penalty_value, number_initial_designs, costs)

        self.number_of_input_dimensions = self.black_box_func.dim

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y, model, budget_consumed = self._initialize_state()

        start_time = time.time()
        iteration = 0
        while budget_consumed < self.budget:
            iteration += 1
            best_observed_location, best_observed_value = self.best_observed(
                best_value_computation_type=self.performance_type,
                train_x=train_x,
                train_y=train_y,
                model=model,
                bounds=self.bounds)
            best_observed_all_sampled.append(best_observed_value)

            acquisition_function = acquisition_function_factory(model=model,
                                                                type=self.acquisition_function_type,
                                                                objective=self.objective,
                                                                best_value=best_observed_value,
                                                                idx=None,
                                                                number_of_outputs=self.number_of_outputs,
                                                                penalty_value=-1 * self.penalty_value,
                                                                iteration=iteration,
                                                                initial_condition_internal_optimizer=best_observed_location)

            new_x, new_fval = self.compute_next_sample(acquisition_function=acquisition_function,
                                                       smart_initial_locations=best_observed_location)

            beta = acquisition_function.compute_beta(self.number_of_input_dimensions)
            vt = self.compute_vt(model, new_x, beta)
            index = self.get_task_to_evaluate(beta, model, new_x, vt)
            new_y = self.evaluate_black_box_func(new_x, index)
            train_x[index] = torch.cat([train_x[index], new_x.cpu()])
            train_y[index] = torch.cat([train_y[index], new_y.cpu()])
            model = self.update_model(X=train_x, y=train_y)
            budget_consumed += self.costs[index]

            print(f"\nBatch{iteration:>2} finished: best value (EI) = "
                  f"({best_observed_value:>4.5f}), best location " + str(
                best_observed_location.detach().cpu().numpy()) + " current sample decision x: " + str(
                new_x.detach().cpu().numpy()) + f" on task {index}", end="\n"
                  )

            self.save_parameters(train_x=train_x,
                                 train_y=train_y,
                                 best_predicted_location=best_observed_location,
                                 model_length_scales=self.model_wrapper.get_model_length_scales(),
                                 best_predicted_location_value=self.evaluate_location_true_quality(
                                     best_observed_location),
                                 acqf_recommended_output_index=index,
                                 acqf_recommended_location=new_x,
                                 acqf_recommended_location_true_value=None if self.black_box_func.is_expensive() else self.evaluate_location_true_quality(
                                     new_x),
                                 acqf_values=new_fval,
                                 budget_consumed=budget_consumed)
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end = time.time() - start_time
        print(f'Total time: {end} seconds')

    def save_parameters(self, train_x, train_y, best_predicted_location, best_predicted_location_value,
                        acqf_recommended_output_index, acqf_recommended_location, acqf_recommended_location_true_value,
                        model_length_scales, budget_consumed, acqf_values=None, cost_configuration=None):
        self.results.random_seed(self.seed)
        self.results.save_budget(self.budget)
        self.results.save_model_length_scales(model_length_scales)
        self.results.save_input_data(train_x)
        self.results.save_output_data(train_y)
        self.results.save_acqf_values(acqf_values)
        self.results.save_number_initial_points(self.number_initial_designs)
        self.results.save_performance_type(self.performance_type)
        self.results.save_best_predicted_location(best_predicted_location)
        self.results.save_best_predicted_location_true_value(best_predicted_location_value)
        self.results.save_acqf_recommended_output_index(acqf_recommended_output_index)
        self.results.save_acqf_recommended_location(acqf_recommended_location)
        self.results.save_acqf_recommended_location_true_value(acqf_recommended_location_true_value)
        self.results.save_budget_consumed(budget_consumed)
        self.results.save_cost_configurations(cost_configuration)
        self.results.generate_pkl_file()

    def get_task_to_evaluate(self, beta, model, new_x, vt):
        posterior = model.posterior(new_x, observation_noise=False)
        constraints_mean = posterior.mean[..., 1:]
        constraints_variance = posterior.variance[..., 1:]
        constraints_ucb = (constraints_mean + torch.sqrt(beta * constraints_variance)).reshape(-1) / self.costs[1:]
        constraint_index = torch.argmax(constraints_ucb)
        if constraints_ucb[constraint_index] > vt / self.costs[0]:
            return constraint_index + 1
        return 0

    def compute_vt(self, model, new_x, beta):
        objective_variance = model.posterior(new_x).variance[..., 0]
        return 2 * torch.sqrt(beta * objective_variance).reshape(1)


class AllSourcesOptimizationLoop(OptimizationLoop):
    """BO loop using AllSourcesDcKG: all K+1 sources optimised jointly.

    Uses q=K+1 in optimize_acqf so each source gets its own x*_k
    via the batch dimension, all in a single L-BFGS call.
    """

    def __init__(self, black_box_func, model, objective, ei_type, seed, budget,
                 performance_type, bounds, results, penalty_value=torch.tensor([0.0]),
                 number_initial_designs=6, costs=None, **kwargs):
        super().__init__(black_box_func, model, objective, ei_type, seed, budget,
                         performance_type, bounds, results, penalty_value,
                         number_initial_designs, costs, **kwargs)

    def run(self):


        train_x, train_y, model, budget_consumed = self._initialize_state()

        start_time = time.time()
        iteration = 0

        while budget_consumed < self.budget:
            iteration += 1
            best_observed_location, best_observed_value = self.best_observed(
                best_value_computation_type=self.performance_type,
                train_x=train_x, train_y=train_y, model=model, bounds=self.bounds)

            # Build shared object ONCE per iteration
            all_dckg = AllSourcesDcKG(
                model,
                penalty_value=self.penalty_value,
                x_best_location=best_observed_location,
                objective=self.objective,
                n_fantasies=7,
                n_constraint_samples=5,
                n_disc=128,
                seed=iteration,
            )

            # Optimise with per-source initialised discretisations
            candidates, _ = self._optimize_all_sources_dckg(
                acqf=all_dckg,
                bounds=self.bounds,
                num_restarts=15,
                raw_samples=72,
                seed=iteration,
            )
            # candidates: (q_total, d) → reshape to (n_sources, q_per_source, d)
            candidates = candidates.detach()
            cand_per_source = candidates.view(
                all_dckg.n_sources, all_dckg.q_per_source, -1
            )  # (K+2, 1+n_disc, d)

            # Evaluate per-source dcKG at the optimised locations
            with torch.no_grad():
                kg_values_list = all_dckg.evaluate_per_source(
                    candidates.unsqueeze(0)
                ).cpu()  # (K+2,)

            # For the coupled source, compute which constraints are not
            # trivially feasible at its candidate (Algorithm 1, line 6)
            x_ckg = cand_per_source[all_dckg.n_sources - 1, 0:1, :]
            idx_to_eval = self.compute_important_idxs(model, x_ckg)
            coupled_cost = torch.sum(self.costs[idx_to_eval])

            # Cost vector: dcKG sources use individual costs,
            # cKG uses filtered cost (obj + non-trivially-feasible constraints)
            costs_with_ckg = torch.cat([
                self.costs, coupled_cost.unsqueeze(0)
            ])

            if (kg_values_list == 0).all():
                # All KG values are zero — fall back to coupled evaluation
                # at the best posterior mean.
                new_x = best_observed_location.detach().reshape(1, -1)
                idx_to_eval_fallback = self.compute_important_idxs(model, new_x)
                for src_idx in idx_to_eval_fallback:
                    new_y = self.evaluate_black_box_func(new_x, src_idx)
                    train_x[src_idx] = torch.cat([train_x[src_idx].cpu(), new_x.cpu()])
                    train_y[src_idx] = torch.cat([train_y[src_idx].cpu(), new_y.cpu()])
                index = all_dckg.n_sources - 1
                budget_consumed += torch.sum(self.costs[idx_to_eval_fallback])
            else:
                index = torch.argmax(kg_values_list / costs_with_ckg)
                new_x = cand_per_source[index, 0:1, :]

                if index == all_dckg.n_sources - 1:
                    # Coupled source: evaluate only non-trivially-feasible outputs
                    for src_idx in idx_to_eval:
                        new_y = self.evaluate_black_box_func(new_x, src_idx)
                        train_x[src_idx] = torch.cat([train_x[src_idx].cpu(), new_x.cpu()])
                        train_y[src_idx] = torch.cat([train_y[src_idx].cpu(), new_y.cpu()])
                    budget_consumed += coupled_cost
                else:
                    # Decoupled source: evaluate only the selected output
                    new_y = self.evaluate_black_box_func(new_x, index)
                    train_x[index] = torch.cat([train_x[index].cpu(), new_x.cpu()])
                    train_y[index] = torch.cat([train_y[index].cpu(), new_y.cpu()])
                    budget_consumed += self.costs[index]
            model = self.update_model(X=train_x, y=train_y)

            kg_str = ", ".join(f"src{i}={v:.5f}" for i, v in enumerate(kg_values_list))
            print(
                f"\nBatch{iteration:>2} finished: best value = "
                f"({best_observed_value:>4.5f}), KG values: [{kg_str}], "
                f"selected task {index}, "
                f"best location " + str(
                    best_observed_location.detach().cpu().numpy()) + " current sample decision x: " + str(
                    new_x.detach().cpu().numpy()) + "\n",
                end="",
            )
            self.save_parameters(
                train_x=train_x, train_y=train_y,
                model_length_scales=self.model_wrapper.get_model_length_scales(),
                best_predicted_location=best_observed_location,
                best_predicted_location_value=self.evaluate_location_true_quality(best_observed_location),
                acqf_recommended_location=new_x,
                acqf_recommended_location_true_value=(
                    None if self.black_box_func.is_expensive()
                    else self.evaluate_location_true_quality(new_x)
                ),
                acqf_recommended_output_index=index,
                acqf_values=kg_values_list,
                budget_consumed=budget_consumed,
            )
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end = time.time() - start_time
        print(f'Total time: {end} seconds')


class IndependentSourcesOptimizationLoop(OptimizationLoop):
    """BO loop that optimises each source independently with its own discretisation.

    Unlike AllSourcesOptimizationLoop (one joint optimize_acqf for all sources),
    this runs a separate optimize_acqf per source, each with q = 1 + n_disc.
    After all sources are optimised, the best source is selected via
    cost-normalised KG values.
    """

    def __init__(self, black_box_func, model, objective, ei_type, seed, budget,
                 performance_type, bounds, results, penalty_value=torch.tensor([0.0]),
                 number_initial_designs=6, costs=None, **kwargs):
        super().__init__(black_box_func, model, objective, ei_type, seed, budget,
                         performance_type, bounds, results, penalty_value,
                         number_initial_designs, costs, **kwargs)

    @staticmethod
    def _optimize_single_source(source_acqf, bounds, x_best, num_restarts=8,
                                raw_samples=512, eta=2.0, seed=0,
                                disc_init="thompson", warm_start_ic=None):
        """Optimize a single source acqf with candidate + discretisation.

        Args:
            disc_init: "sobol" for gen_batch_initial_conditions only (original),
                       "thompson" for n_zy (or n_zc) Thompson points injected
                       into the first disc slots, rest from
                       gen_batch_initial_conditions.
            warm_start_ic: Optional (1, Q, d) tensor from a previous iteration's
                       optimised result (candidate + discretisation), injected
                       as an extra restart.
        """


        n_disc = 64
        Q = 1 + n_disc
        dev = x_best.device
        bounds = bounds.to(dev)

        ics = gen_batch_initial_conditions(
            acq_function=source_acqf, bounds=bounds, q=Q,
            num_restarts=num_restarts, raw_samples=raw_samples,
            options={"seed": seed, "eta": eta},
        )

        if disc_init == "thompson":
            thompson_pts = OptimizationLoop._thompson_disc(
                source_acqf, bounds, seed=seed,
            )
            n_thompson = thompson_pts.shape[0]
            ics[:, 1:1 + n_thompson, :] = thompson_pts.squeeze(1)

        xbest_ic = ics[-1:].clone()
        xbest_ic[:, 0:1, :] = x_best
        ics = torch.cat([ics, xbest_ic], dim=0)

        if warm_start_ic is not None:
            ics = torch.cat([ics, warm_start_ic.to(dev)], dim=0)

        candidates, acqf_values = optimize_acqf(
            acq_function=source_acqf, bounds=bounds, q=Q,
            num_restarts=ics.shape[0], batch_initial_conditions=ics,
            return_best_only=False,
            options={"maxiter": 100},
        )
        # candidates: (num_restarts, Q, d), acqf_values: (num_restarts,)
        best = acqf_values.argmax()
        return candidates[best:best+1, :].detach(), acqf_values[best].detach()

    def run(self):
        train_x, train_y, model, budget_consumed = self._initialize_state()

        start_time = time.time()
        iteration = 0

        warm_cache = {}  # source index -> (1, Q, d) from previous iteration

        while budget_consumed < self.budget:
            iteration += 1
            best_observed_location, best_observed_value = self.best_observed(
                best_value_computation_type=self.performance_type,
                train_x=train_x, train_y=train_y, model=model, bounds=self.bounds)

            all_dckg = AllSourcesDcKG(
                model,
                penalty_value=self.penalty_value,
                x_best_location=best_observed_location,
                objective=self.objective,
                n_fantasies=7,
                n_constraint_samples=5,
                n_disc=128,
                seed=iteration,
            )

            # Optimize each source INDEPENDENTLY
            kg_values = torch.zeros(all_dckg.n_sources)
            best_xs = []
            for s, source_acqf in enumerate(all_dckg.sources):
                start = time.time()
                best_x_s, val_s = self._optimize_single_source(
                    source_acqf, self.bounds,
                    x_best=all_dckg.x_best,
                    num_restarts=15, raw_samples=72,
                    seed=iteration + s * 100,
                    warm_start_ic=warm_cache.get(s),
                )
                kg_values[s] = val_s
                best_xs.append(best_x_s)
                stop = time.time()
                print("source: ", s, " finished: ", stop - start)
            # For the coupled source, compute which constraints are not
            # trivially feasible at its candidate
            # best_xs[s] is (1, Q, d); slot 0 is the candidate point
            x_ckg = best_xs[all_dckg.n_sources - 1][:, 0:1, :].reshape(1, -1)
            idx_to_eval = self.compute_important_idxs(model, x_ckg)
            coupled_cost = torch.sum(self.costs[idx_to_eval])
            costs_with_ckg = torch.cat([self.costs, coupled_cost.unsqueeze(0)])
            if (kg_values == 0).all():
                new_x = best_observed_location.detach().reshape(1, -1)
                idx_to_eval_fallback = self.compute_important_idxs(model, new_x)
                for src_idx in idx_to_eval_fallback:
                    new_y = self.evaluate_black_box_func(new_x, src_idx)
                    train_x[src_idx] = torch.cat([train_x[src_idx].cpu(), new_x.cpu()])
                    train_y[src_idx] = torch.cat([train_y[src_idx].cpu(), new_y.cpu()])
                index = all_dckg.n_sources - 1
                saved_output_index = idx_to_eval_fallback
                budget_consumed += torch.sum(self.costs[idx_to_eval_fallback])
            else:
                index = torch.argmax(kg_values / costs_with_ckg)
                new_x = best_xs[index][:, 0:1, :].reshape(1, -1)

                if index == all_dckg.n_sources - 1:
                    # Coupled: evaluate only non-trivially-feasible outputs
                    for src_idx in idx_to_eval:
                        new_y = self.evaluate_black_box_func(new_x, src_idx)
                        train_x[src_idx] = torch.cat([train_x[src_idx].cpu(), new_x.cpu()])
                        train_y[src_idx] = torch.cat([train_y[src_idx].cpu(), new_y.cpu()])
                    saved_output_index = idx_to_eval
                    budget_consumed += coupled_cost
                else:
                    new_y = self.evaluate_black_box_func(new_x, index)
                    train_x[index] = torch.cat([train_x[index].cpu(), new_x.cpu()])
                    train_y[index] = torch.cat([train_y[index].cpu(), new_y.cpu()])
                    saved_output_index = [index.item()]
                    budget_consumed += self.costs[index]
            model = self.update_model(X=train_x, y=train_y)

            # Cache optimised candidates from non-elected sources with KG > 0
            warm_cache = {}
            for s in range(all_dckg.n_sources):
                if s != index and kg_values[s] > 0:
                    warm_cache[s] = best_xs[s].detach().cpu()

            kg_str = ", ".join(f"src{i}={v:.5f}" for i, v in enumerate(kg_values))
            print(
                f"\nBatch{iteration:>2} finished: best value = "
                f"({best_observed_value:>4.5f}), KG values: [{kg_str}], "
                f"selected task {index}, "
                f"best location " + str(
                    best_observed_location.detach().cpu().numpy()) + " current sample decision x: " + str(
                    new_x.detach().cpu().numpy()) + "\n",
                end="",
            )
            self.save_parameters(
                train_x=train_x, train_y=train_y,
                model_length_scales=self.model_wrapper.get_model_length_scales(),
                best_predicted_location=best_observed_location,
                best_predicted_location_value=self.evaluate_location_true_quality(best_observed_location),
                acqf_recommended_location=new_x,
                acqf_recommended_location_true_value=(
                    None if self.black_box_func.is_expensive()
                    else self.evaluate_location_true_quality(new_x)
                ),
                acqf_recommended_output_index=saved_output_index,
                acqf_values=kg_values,
                budget_consumed=budget_consumed,
            )
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end_time = time.time() - start_time
        print(f'Total time: {end_time} seconds')


class PESC_OptimizationLoop(OptimizationLoop):
    """Strictly decoupled PESC loop.

    Each iteration samples the constrained optima x* once, optimises the PESC
    acquisition of each source (objective + every constraint) independently, and
    evaluates the single source maximising the cost-normalised information gain.
    There is no coupled-evaluation fallback: even when every source value is ~0
    the argmax source is still selected.
    """

    def __init__(self, black_box_func, model, objective, ei_type, seed, budget,
                 performance_type, bounds, results, penalty_value=torch.tensor([0.0]),
                 number_initial_designs=6, costs=None, num_optima_samples=10, **kwargs):
        super().__init__(black_box_func, model, objective, ei_type, seed, budget,
                         performance_type, bounds, results, penalty_value,
                         number_initial_designs, costs, **kwargs)
        self.num_optima_samples = num_optima_samples

    def _grid_search(self, source_acqf, seed, grid_size=10000, chunk=256, n_spray=10,
                     spray_std=1e-4, x_center=None, maxfev=100):
        """Spearmint's acquisition strategy: a vectorised sweep, no gradients.

        The reference PESC implementation sets ``has_gradients = False`` and does a
        grid sweep plus "spray" points around the incumbent rather than gradient
        optimisation.  Profiling here found ``optimize_acqf``'s ~120 gradient
        forward/backward passes to be 91-97% of per-iteration cost.

        Chunking is not optional.  The joint posterior over ``[X, vec]`` is dense and
        O(b^2), so cost per candidate *rises* with batch size (156 us at b=256,
        672 us at b=1024, 1.9 ms at b=3072).  Each call also carries ~24 ms of fixed
        overhead, so chunks want to be big enough to amortise that and no bigger:
        ~256 sits near the minimum.  A derivative-free polish is deliberately not
        used -- at 24 ms per single-point call it would cost ~3 s, which is why the
        local refinement is folded in as spray points instead.

        Grid settings match the reference implementation's defaults:
        ``acq_grid_size=10000``, ``num_spray=10``, ``spray_std=1e-4``.  For scale, on
        d=2 the measured quality against L-BFGS as reference maximiser was a median
        0.84x of the optimum at 1024 and 0.98x at 4096, so 10000 is comfortably past
        the knee.  (Some cases exceed 1.0 -- the grid finds peaks L-BFGS misses -- so
        this is not a one-sided sacrifice.)  A fixed grid still thins out as ``d``
        grows; the reference has the same weakness, but on the higher-dimensional
        problems here (SpeedReducer d=7) it is worth re-checking.

        ``chunk`` is ours, not Spearmint's: they can evaluate 10000 candidates in one
        call because each costs a couple of triangular solves against a cached
        Cholesky, whereas our joint posterior is dense and O(b^2).
        """
        n_spray = 0 if x_center is None else min(n_spray, grid_size // 4)
        X = draw_sobol_samples(bounds=self.bounds, n=grid_size - n_spray, q=1,
                               seed=seed).to(self.bounds)
        if n_spray:
            c = x_center.reshape(1, 1, -1).to(self.bounds)
            cloud = c + spray_std * torch.randn(n_spray - 1, 1, c.shape[-1]).to(self.bounds)
            cloud = torch.max(torch.min(cloud, self.bounds[1]), self.bounds[0])
            X = torch.cat([X, cloud, c], dim=0)

        vals = []
        with torch.no_grad():
            for i in range(0, X.shape[0], chunk):
                vals.append(source_acqf(X[i:i + chunk]))
        vals = torch.nan_to_num(torch.cat(vals), nan=0.0, posinf=0.0, neginf=0.0)
        best = int(vals.argmax())
        x0, v0 = X[best].detach().reshape(1, -1), vals[best].detach()
        if maxfev <= 0:
            return x0, v0

        # Reference behaviour is optimize_acq=True: a derivative-free polish from the
        # single best grid point, nlopt LN_BOBYQA at opt_acq_tol=1e-4.  nlopt is not
        # installed here, so scipy's Powell stands in (also derivative-free, handles
        # box bounds).  Their maxeval is acq_grid_size=10000; at our ~24 ms per
        # single-point call that would be four minutes per source, against
        # microseconds for their cached-Cholesky path, so maxfev is capped instead.
        from scipy.optimize import minimize
        lo = self.bounds[0].detach().cpu().numpy()
        hi = self.bounds[1].detach().cpu().numpy()

        def neg(x_np):
            xt = torch.as_tensor(x_np, dtype=self.bounds.dtype, device=self.bounds.device)
            xt = torch.max(torch.min(xt, self.bounds[1]), self.bounds[0]).reshape(1, 1, -1)
            with torch.no_grad():
                v = float(source_acqf(xt).reshape(-1)[0])
            return -v if math.isfinite(v) else 0.0

        try:
            res = minimize(neg, x0.reshape(-1).detach().cpu().numpy(), method="Powell",
                           bounds=list(zip(lo, hi)),
                           options={"maxfev": maxfev, "xtol": 1e-4, "ftol": 1e-4})
            v_p = -float(res.fun)
            if math.isfinite(v_p) and v_p > float(v0):  # never return a worse point
                x_p = torch.as_tensor(res.x, dtype=self.bounds.dtype,
                                      device=self.bounds.device)
                x_p = torch.max(torch.min(x_p, self.bounds[1]), self.bounds[0])
                return x_p.reshape(1, -1), torch.as_tensor(v_p, dtype=v0.dtype,
                                                           device=v0.device)
        except (RuntimeError, ValueError):
            pass
        return x0, v0

    def _optimize_source(self, source_acqf, seed, x_center=None):
        # The EP variant uses the grid sweep (see _grid_search); the cheap
        # closed-form variant keeps gradient-based optimisation, where it is
        # affordable and slightly sharper.
        is_ep = self.acquisition_function_type == AcquisitionFunctionType.PESC_EP
        if is_ep:
            return self._grid_search(source_acqf, seed, x_center=x_center)
        try:
            candidates, value = optimize_acqf(
                acq_function=source_acqf, bounds=self.bounds, q=1,
                num_restarts=10, raw_samples=512,
                options={"maxiter": 100, "seed": seed},
            )
            if torch.isfinite(value).all():
                return candidates.detach(), value.detach()
        except RuntimeError:
            # Entropy-search acqfs can still produce NaN gradients; fall back to a
            # gradient-free sweep.
            pass
        return self._grid_search(source_acqf, seed, x_center=x_center)

    def run(self):
        train_x, train_y, model, budget_consumed = self._initialize_state()
        start_time = time.time()
        iteration = 0

        while budget_consumed < self.budget:
            iteration += 1
            best_observed_location, best_observed_value = self.best_observed(
                best_value_computation_type=self.performance_type,
                train_x=train_x, train_y=train_y, model=model, bounds=self.bounds)

            x_star = sample_constrained_optima(
                model, self.bounds, num_samples=self.num_optima_samples,
                seed=self.seed + iteration,
            )

            # One converged EP shared by every source, as Spearmint does (its
            # predictEP returns all task variances from a single EP solution).
            conditioner = None
            if self.acquisition_function_type == AcquisitionFunctionType.PESC_EP:
                from bo.acquisition_functions.pesc_sites import PESCFaithfulConditioner
                conditioner = PESCFaithfulConditioner(model, x_star)

            info_values = torch.zeros(self.number_of_outputs, dtype=dtype)
            new_x_list = []
            for task_idx in range(self.number_of_outputs):
                source_acqf = acquisition_function_factory(
                    model=model, type=self.acquisition_function_type,
                    objective=self.objective, best_value=best_observed_value,
                    idx=task_idx, number_of_outputs=self.number_of_outputs,
                    penalty_value=self.penalty_value, iteration=iteration,
                    initial_condition_internal_optimizer=best_observed_location,
                    x_star=x_star, conditioner=conditioner,
                )
                new_x, value = self._optimize_source(
                    source_acqf, seed=iteration + task_idx * 100,
                    x_center=best_observed_location)
                info_values[task_idx] = value
                new_x_list.append(new_x)

            index = torch.argmax(info_values / self.costs)
            new_x = new_x_list[index]
            new_y = self.evaluate_black_box_func(new_x, index)
            train_x[index] = torch.cat([train_x[index].cpu(), new_x.cpu()])
            train_y[index] = torch.cat([train_y[index].cpu(), new_y.cpu()])
            budget_consumed += self.costs[index]
            model = self.update_model(X=train_x, y=train_y)

            info_str = ", ".join(f"src{i}={v:.5f}" for i, v in enumerate(info_values))
            print(
                f"\nBatch{iteration:>2} finished: best value = "
                f"({best_observed_value:>4.5f}), PESC values: [{info_str}], "
                f"selected task {index}, "
                f"best location " + str(best_observed_location.detach().cpu().numpy())
                + " current sample decision x: " + str(new_x.detach().cpu().numpy()) + "\n",
                end="",
            )
            self.save_parameters(
                train_x=train_x, train_y=train_y,
                model_length_scales=self.model_wrapper.get_model_length_scales(),
                best_predicted_location=best_observed_location,
                best_predicted_location_value=self.evaluate_location_true_quality(best_observed_location),
                acqf_recommended_location=new_x,
                acqf_recommended_location_true_value=(
                    None if self.black_box_func.is_expensive()
                    else self.evaluate_location_true_quality(new_x)
                ),
                acqf_recommended_output_index=[index.item()],
                acqf_values=info_values,
                budget_consumed=budget_consumed,
            )
            print(f'took {time.time() - start_time} seconds')

        print(f'Total time: {time.time() - start_time} seconds')
