import time
import warnings
from typing import Optional

import torch
from botorch import gen_candidates_scipy
from botorch.acquisition import MCAcquisitionObjective
from botorch.optim import optimize_acqf
from scipy.stats import qmc
from torch import Tensor

from bo.acquisition_functions.acquisition_functions import acquisition_function_factory, AcquisitionFunctionType, \
    DecopledHybridConstrainedKnowledgeGradient
from bo.model.Model import ConstrainedPosteriorMean, ConstrainedDeoupledGPModelWrapper
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import SingleObjectiveProblem

# constants
device = torch.device("cpu")
dtype = torch.float64
warnings.filterwarnings("ignore")  # Comment out if there are issues


class OptimizationLoop:

    def __init__(self, black_box_func: SingleObjectiveProblem, model: ConstrainedDeoupledGPModelWrapper,
                 objective: Optional[MCAcquisitionObjective], ei_type: AcquisitionFunctionType, seed: int, budget: int,
                 performance_type: str, bounds: Tensor, results: Results,
                 penalty_value: Optional[Tensor] = torch.tensor([0.0]), number_initial_designs: Optional[int] = 6,
                 costs: Optional[Tensor] = None
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

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y = self.generate_initial_data(n=self.number_initial_designs)
        model = self.update_model(train_x, train_y)
        start_time = time.time()
        iteration = 0
        budget_consumed = 0
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
            index = torch.argmax(torch.tensor(kg_values_list) / self.costs)
            new_y = self.evaluate_black_box_func(new_x_list[index], index)

            train_x[index] = torch.cat([train_x[index], new_x_list[index]])
            train_y[index] = torch.cat([train_y[index], new_y])
            model = self.update_model(X=train_x, y=train_y)
            budget_consumed += self.costs[index]
            print(
                f"\nBatch{iteration:>2} finished: best value (EI) = "
                f"({best_observed_value:>4.5f}), best location " + str(
                    best_observed_location.numpy()) + " current sample decision x: " + str(
                    new_x_list[index].numpy()) + f" on task {index}\n",
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
        tasks_values = self.black_box_func.evaluate_black_box(X, True)
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
        return self.black_box_func.evaluate_task(X, task_idx)

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
        self.model_wrapper.fit(X, y)
        optimized_model = self.model_wrapper.optimize()
        return optimized_model

    def best_observed(self, best_value_computation_type, train_x, train_y, model, bounds):
        if best_value_computation_type == "sampled":
            return self.compute_best_sampled_value(train_x, train_y)
        elif best_value_computation_type == "model":
            return self.compute_best_posterior_mean(model, bounds)

    @staticmethod
    def compute_best_sampled_value(train_x, train_y):
        return train_x[torch.argmax(train_y)], torch.max(train_y)

    def compute_best_posterior_mean(self, model, bounds):
        argmax_mean, max_mean = optimize_acqf(
            acq_function=ConstrainedPosteriorMean(model, maximize=True, penalty_value=self.penalty_value),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=2048,
        )
        return argmax_mean, max_mean

    def compute_next_sample(self, acquisition_function, smart_initial_locations=None):
        candidates, acqf_value = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.bounds,
            gen_candidates=gen_candidates_scipy,
            q=1,
            num_restarts=15,  # can make smaller if too slow, not too small though
            raw_samples=72,  # used for intialization heuristic
            options={"maxiter": 100},
        )
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
                 costs: Optional[Tensor] = None):

        super().__init__(black_box_func, model, objective, ei_type, seed, budget, performance_type, bounds, results,
                         penalty_value, number_initial_designs, costs)

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y = self.generate_initial_data(n=self.number_initial_designs)
        model = self.update_model(train_x, train_y)

        start_time = time.time()
        iteration = 0
        budget_consumed = 0
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
            if best_ckG_value_per_cost > best_dckg_value_per_cost:  # Run coupled cKG
                new_output = self.black_box_func.evaluate_black_box(new_x_ckg, False)
                for task_idx in idx_to_eval:
                    train_x[task_idx] = torch.cat([train_x[task_idx], new_x_ckg])
                    train_y[task_idx] = torch.cat([train_y[task_idx], new_output[:, task_idx]])
                index = idx_to_eval  # Will have to change for non-ones costs
                location_to_sample = new_x_ckg
                budget_consumed += torch.sum(total_cost_filtered)
            else:  # Run dcKG
                index = torch.argmax(torch.tensor(kg_values_list[:-1]) / self.costs)
                new_y = self.evaluate_black_box_func(new_x_list[index], index)
                train_x[index] = torch.cat([train_x[index], new_x_list[index]])
                train_y[index] = torch.cat([train_y[index], new_y])
                location_to_sample = new_x_list[index]
                index = [index.item()]
                budget_consumed += torch.sum(self.costs[index])
            model = self.update_model(X=train_x, y=train_y)
            print(
                f"\nBatch{iteration:>2} finished: best value (EI) = "
                f"({best_observed_value:>4.5f}), best location " + str(
                    best_observed_location.numpy()) + " current sample decision x: " + str(
                    location_to_sample.numpy()) + f" on tasks " + str(index) + "\n",
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
                 costs: Optional[Tensor] = None):

        super().__init__(black_box_func, model, objective, ei_type, seed, budget, performance_type, bounds, results,
                         penalty_value, number_initial_designs, costs)

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y = self.generate_initial_data(n=self.number_initial_designs)
        model = self.update_model(train_x, train_y)

        start_time = time.time()
        iteration = 0
        consumed_budget = 0
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
            new_x, _ = self.compute_next_sample(acquisition_function=acquisition_function,
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
                train_x[evaluation_order[i]] = torch.cat([train_x[evaluation_order[i]], new_x])
                train_y[evaluation_order[i]] = torch.cat([train_y[evaluation_order[i]], new_y])
                model = self.update_model(X=train_x, y=train_y)
                evaluated_idx.append(evaluation_order[i])
                if new_y < 0:
                    i = i + 1
                else:
                    failing_constraint = i
                    i = size

            print(
                f"\nBatch{iteration:>2} finished: best value (EI) = "
                f"({best_observed_value:>4.5f}), best location " + str(
                    best_observed_location.numpy()) + " current sample decision x: " + str(new_x.numpy()), end="\n"
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
                 costs: Optional[Tensor] = None):

        super().__init__(black_box_func, model, objective, ei_type, seed, budget, performance_type, bounds, results,
                         penalty_value, number_initial_designs, costs)

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y = self.generate_initial_data(n=self.number_initial_designs)
        model = self.update_model(train_x, train_y)

        start_time = time.time()
        for iteration in range(self.budget):
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

            new_y = self.black_box_func.evaluate_black_box(new_x, False)
            for i in range(self.model_wrapper.getNumberOfOutputs()):
                train_x[i] = torch.cat([train_x[i], new_x])
                train_y[i] = torch.cat([train_y[i], new_y[:, i]])
            model = self.update_model(X=train_x, y=train_y)

            best_observed_location_value = self.evaluate_location_true_quality(best_observed_location).numpy()
            print(f"\nBatch{iteration:>2} finished: best value (EI) =" + str(
                best_observed_location_value) + ", best location " + str(
                best_observed_location.numpy()) + " current sample decision x: " + str(new_x.numpy()), end="\n")

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
        if isinstance(acquisition_function, DecopledHybridConstrainedKnowledgeGradient):
            sampler = qmc.LatinHypercube(d=self.dim_x, seed=self.seed)
            test_x = torch.Tensor(sampler.random(n=1000))
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
        candidates, kgvalue = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.bounds,
            q=1,
            num_restarts=15,  # can make smaller if too slow, not too small though
            raw_samples=72,  # used for intialization heuristic
            options={"maxiter": 100}
        )
        # observe new values
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
                 costs: Optional[Tensor] = None):

        super().__init__(black_box_func, model, objective, ei_type, seed, budget, performance_type, bounds, results,
                         penalty_value, number_initial_designs, costs)

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y = self.generate_initial_data(n=self.number_initial_designs)
        model = self.update_model(train_x, train_y)

        start_time = time.time()
        budget_consumed = 0
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
            kg_values_list = torch.zeros(self.number_of_outputs, dtype=dtype)
            for task_idx in range(self.number_of_outputs):
                # print("Running Task:", task_idx)
                acquisition_function = acquisition_function_factory(model=model,
                                                                    type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                                                    objective=self.objective,
                                                                    best_value=best_observed_value,
                                                                    idx=task_idx,
                                                                    number_of_outputs=self.number_of_outputs,
                                                                    penalty_value=self.penalty_value,
                                                                    iteration=iteration,
                                                                    initial_condition_internal_optimizer=best_observed_location)

                kg_values_list[task_idx] = acquisition_function(new_x)

            index = torch.argmax(torch.tensor(kg_values_list) / self.costs)
            new_y = self.evaluate_black_box_func(new_x, index)
            train_x[index] = torch.cat([train_x[index], new_x])
            train_y[index] = torch.cat([train_y[index], new_y])
            model = self.update_model(X=train_x, y=train_y)
            budget_consumed += self.costs[index]
            print(
                f"\nBatch{iteration:>2} finished: best value (EI) = "
                f"({best_observed_value:>4.5f}), best location " + str(
                    best_observed_location.numpy()) + " current sample decision x: " + str(
                    new_x.numpy()) + f" on task {index}", end="\n"
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
        train_x, train_y = self.generate_initial_data(n=self.number_initial_designs)
        model = self.update_model(train_x, train_y)

        start_time = time.time()
        budget_consumed = 0
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
            train_x[index] = torch.cat([train_x[index], new_x])
            train_y[index] = torch.cat([train_y[index], new_y])
            model = self.update_model(X=train_x, y=train_y)
            budget_consumed += self.costs[index]

            print(f"\nBatch{iteration:>2} finished: best value (EI) = "
                  f"({best_observed_value:>4.5f}), best location " + str(
                best_observed_location.numpy()) + " current sample decision x: " + str(
                new_x.numpy()) + f" on task {index}", end="\n"
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
