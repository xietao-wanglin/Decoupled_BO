import time
import warnings
from typing import Optional

import torch
from botorch import gen_candidates_torch
from botorch.acquisition import MCAcquisitionObjective
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.utils import draw_sobol_samples
from scipy.stats import qmc
from torch import Tensor

from bo.acquisition_functions.acquisition_functions import (
    acquisition_function_factory, AcquisitionFunctionType,
)
from bo.acquisition_functions.refactored_acquisition_functions import (
    FastConstrainedKG, AllSourcesDcKG, ObjectiveDcKG, ConstraintDcKG, CoupledCKG,
)
from bo.model.Model import (
    ConstrainedPosteriorMean, ConstrainedDeoupledGPModelWrapper,
    gaussian_copula_transform,
)
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import SingleObjectiveProblem

warnings.filterwarnings("ignore")  # Comment out if there are issues


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
    
        if isinstance(acquisition_function, FastConstrainedKG):
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


class IndependentSourcesOptimizationLoop(OptimizationLoop):
    """BO loop that optimises each source independently with its own discretisation.

    A separate optimize_acqf is run per source, each with q = 1 + n_disc. After all
    sources are optimised, the best source is selected via cost-normalised KG values.
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


class AblationIndependentSourcesOptimizationLoop(IndependentSourcesOptimizationLoop):
    """IndependentSourcesOptimizationLoop with the coupled cKG candidate removed.

    ablation_mode:
      "no_coupled_candidate" — only the K+1 decoupled sources compete on KG/cost;
          the all-zero fallback still evaluates every non-trivially-feasible output.
      "fully_decoupled" — as above, but the all-zero fallback evaluates a single
          source (round-robin), so no iteration ever queries more than one source.
    """

    def __init__(self, *args, ablation_mode="no_coupled_candidate", **kwargs):
        assert ablation_mode in ("no_coupled_candidate", "fully_decoupled")
        self.ablation_mode = ablation_mode
        super().__init__(*args, **kwargs)

    def run(self):
        train_x, train_y, model, budget_consumed = self._initialize_state()

        start_time = time.time()
        iteration = 0

        warm_cache = {}  # source index -> (1, Q, d) from previous iteration
        fallback_rr = 0  # round-robin cursor for the fully-decoupled fallback

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
            # Drop the trailing CoupledCKG source; it is built but never optimised.
            n_decoupled = all_dckg.n_sources - 1

            kg_values = torch.zeros(n_decoupled)
            best_xs = []
            for s in range(n_decoupled):
                start = time.time()
                best_x_s, val_s = self._optimize_single_source(
                    all_dckg.sources[s], self.bounds,
                    x_best=all_dckg.x_best,
                    num_restarts=15, raw_samples=72,
                    seed=iteration + s * 100,
                    warm_start_ic=warm_cache.get(s),
                )
                kg_values[s] = val_s
                best_xs.append(best_x_s)
                stop = time.time()
                print("source: ", s, " finished: ", stop - start)

            if (kg_values == 0).all():
                new_x = best_observed_location.detach().reshape(1, -1)
                if self.ablation_mode == "fully_decoupled":
                    idx_to_eval_fallback = [fallback_rr % n_decoupled]
                    fallback_rr += 1
                else:
                    idx_to_eval_fallback = self.compute_important_idxs(model, new_x)
                for src_idx in idx_to_eval_fallback:
                    new_y = self.evaluate_black_box_func(new_x, src_idx)
                    train_x[src_idx] = torch.cat([train_x[src_idx].cpu(), new_x.cpu()])
                    train_y[src_idx] = torch.cat([train_y[src_idx].cpu(), new_y.cpu()])
                index = n_decoupled  # sentinel: fallback taken, no source elected
                saved_output_index = idx_to_eval_fallback
                budget_consumed += torch.sum(self.costs[idx_to_eval_fallback])
            else:
                index = torch.argmax(kg_values / self.costs)
                new_x = best_xs[index][:, 0:1, :].reshape(1, -1)
                new_y = self.evaluate_black_box_func(new_x, index)
                train_x[index] = torch.cat([train_x[index].cpu(), new_x.cpu()])
                train_y[index] = torch.cat([train_y[index].cpu(), new_y.cpu()])
                saved_output_index = [index.item()]
                budget_consumed += self.costs[index]
            model = self.update_model(X=train_x, y=train_y)

            # Cache optimised candidates from non-elected sources with KG > 0
            warm_cache = {}
            for s in range(n_decoupled):
                if s != index and kg_values[s] > 0:
                    warm_cache[s] = best_xs[s].detach().cpu()

            kg_str = ", ".join(f"src{i}={v:.5f}" for i, v in enumerate(kg_values))
            print(
                f"\nBatch{iteration:>2} finished [{self.ablation_mode}]: best value = "
                f"({best_observed_value:>4.5f}), KG values: [{kg_str}], "
                f"selected task {index}, evaluated {saved_output_index}, "
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
