import time
import warnings
from typing import Optional

import torch
from botorch import gen_candidates_scipy
from botorch.acquisition import MCAcquisitionObjective
from botorch.optim import optimize_acqf
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor

from bo.acquisition_functions.acquisition_functions import acquisition_function_factory, AcquisitionFunctionType
from bo.model.Model import ConstrainedPosteriorMean, ConstrainedDeoupledGPModelWrapper
from bo.result_utils.result_container import Results

# constants
device = torch.device("cpu")
dtype = torch.float64
warnings.filterwarnings("ignore")  # Comment out if there are issues


class OptimizationLoop:

    def __init__(self, black_box_func: BaseTestProblem, model: ConstrainedDeoupledGPModelWrapper,
                 objective: Optional[MCAcquisitionObjective], ei_type: AcquisitionFunctionType, seed: int, budget: int,
                 performance_type: str, bounds: Tensor, results: Results,
                 penalty_value: Optional[Tensor] = torch.tensor([0.0]), number_initial_designs: Optional[int] = 6,
                 costs: Optional[Tensor] = None
                 ):

        print("Starting Loop: OptimizationLoop")
        if costs is None:
            print("Using default costs")
            costs = torch.ones(model.getNumberOfOutputs())
        torch.random.manual_seed(seed)
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
        for iteration in range(self.budget):
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

            print(
                f"\nBatch{iteration:>2} finished: best value (EI) = "
                f"({best_observed_value:>4.5f}), best location " + str(
                    best_observed_location.numpy()) + " current sample decision x: " + str(
                    new_x_list[index].numpy()) + f" on task {index}\n",
                end="",
            )
            self.save_parameters(train_x=train_x,
                                 train_y=train_y,
                                 best_predicted_location=best_observed_location,
                                 best_predicted_location_value=self.evaluate_location_true_quality(
                                     best_observed_location),
                                 acqf_recommended_location=new_x_list[index],
                                 acqf_recommended_location_true_value=self.evaluate_location_true_quality(
                                     new_x_list[index]),
                                 acqf_recommended_output_index=index, acqf_values=kg_values_list)
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end = time.time() - start_time
        print(f'Total time: {end} seconds')

    def save_parameters(self, train_x, train_y, best_predicted_location,
                        best_predicted_location_value, acqf_recommended_output_index, acqf_recommended_location,
                        acqf_recommended_location_true_value, acqf_values=None):

        self.results.random_seed(self.seed)
        self.results.save_budget(self.budget)
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
        self.results.generate_pkl_file()

    def evaluate_location_true_quality(self, X):
        f_value = self.evaluate_black_box_func(X, 0)
        if self.is_design_feasible(X):
            return f_value
        return -self.penalty_value

    def is_design_feasible(self, X):
        for idx in range(1, self.model_wrapper.getNumberOfOutputs()):
            c_val = self.evaluate_black_box_func(X, idx)
            if c_val > 0:
                return False
        return True

    def evaluate_black_box_func(self, X, task_idx):
        return self.black_box_func.evaluate_task(X, task_idx)

    def generate_initial_data(self, n: int):
        # generate training data
        train_x_list = []
        train_y_list = []
        for i in range(self.model_wrapper.getNumberOfOutputs()):
            train_x = torch.rand(n, self.dim_x, device=device, dtype=dtype)
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

    def compute_best_sampled_value(self, train_x, train_y):
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
        candidates, _ = optimize_acqf(
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
        x_optimised_val = acquisition_function.evaluate_kg_value(x_optimised,
                                                                 number_of_restarts=20,
                                                                 number_of_raw_points=128).detach()
        if smart_initial_locations is not None:
            candidates, _ = optimize_acqf(
                acq_function=acquisition_function,
                bounds=self.bounds,
                num_restarts=smart_initial_locations.shape[0],
                batch_initial_conditions=smart_initial_locations,
                q=1,
                options={"maxiter": 100}
            )
            x_smart_optimised = candidates.detach()
            x_smart_optimised_val = acquisition_function.evaluate_kg_value(x_smart_optimised[None, :],
                                                                           number_of_restarts=20,
                                                                           number_of_raw_points=128).detach()
            if x_smart_optimised_val >= x_optimised_val:
                return x_smart_optimised[None, :], x_smart_optimised_val

        return x_optimised, x_optimised_val


class EI_Decoupled_OptimizationLoop(OptimizationLoop):

    def __init__(self, black_box_func: BaseTestProblem, model: ConstrainedDeoupledGPModelWrapper,
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
            new_x, _ = self.compute_next_sample(acquisition_function=acquisition_function,
                                                smart_initial_locations=best_observed_location)
            posterior = model.posterior(new_x)
            mu = posterior.mean
            std = posterior.variance.sqrt().clamp_min(1e-9)
            z = -mu / std
            probability_infeasibility = []
            size = self.model_wrapper.getNumberOfOutputs()
            for i in range(1, size):
                probability_infeasibility = probability_infeasibility + [
                    1 - torch.distributions.Normal(0, 1).cdf(z)[0][i].detach().item()]
            evaluation_order = sorted(range(len(probability_infeasibility)), key=probability_infeasibility.__getitem__)[
                               ::-1]
            evaluated_idx = []
            i = 0
            j = 0
            k = -1
            while i < size - 1:
                if probability_infeasibility[i] > 0.1:
                    # print(evaluation_order[i]+1)
                    new_y = self.evaluate_black_box_func(new_x, evaluation_order[i] + 1)
                    train_x[evaluation_order[i] + 1] = torch.cat([train_x[evaluation_order[i] + 1], new_x])
                    train_y[evaluation_order[i] + 1] = torch.cat([train_y[evaluation_order[i] + 1], new_y])
                    model = self.update_model(X=train_x, y=train_y)
                    evaluated_idx.append(evaluation_order[i] + 1)
                    if new_y < 0:
                        i = i + 1
                    else:
                        k = i + 1
                        i = size
                elif probability_infeasibility[i] < 0.1 and j == 0:
                    # print(0)
                    new_y = self.evaluate_black_box_func(new_x, 0)
                    train_x[0] = torch.cat([train_x[0], new_x])
                    train_y[0] = torch.cat([train_y[0], new_y])
                    model = self.update_model(X=train_x, y=train_y)
                    evaluated_idx.append(0)
                    j = 1
                    if new_y < best_observed_value:
                        k = 0
                        i = size
                else:
                    # print(evaluation_order[i]+1)
                    new_y = self.evaluate_black_box_func(new_x, evaluation_order[i]+1)
                    train_x[evaluation_order[i]+1] = torch.cat([train_x[evaluation_order[i]+1], new_x])
                    train_y[evaluation_order[i]+1] = torch.cat([train_y[evaluation_order[i]+1], new_y])
                    model = self.update_model(X=train_x, y=train_y)
                    evaluated_idx.append(evaluation_order[i]+1)
                    if new_y < 0:
                        i=i+1
                    else:
                        k = i+1
                        i = size
            if i == size - 1 and j == 0:
                # print(0)
                new_y = self.evaluate_black_box_func(new_x, 0)
                train_x[0] = torch.cat([train_x[0], new_x])
                train_y[0] = torch.cat([train_y[0], new_y])
                model = self.update_model(X=train_x, y=train_y)
                evaluated_idx.append(0)

            print(
                f"\nBatch{iteration:>2} finished: best value (EI) = "
                f"({best_observed_value:>4.5f}), best location " + str(
                    best_observed_location.numpy()) + " current sample decision x: " + str(new_x.numpy()), end="\n"
            )

            print(f'Evaluated functions: {evaluated_idx}')
            self.save_parameters(train_x=train_x,
                                 train_y=train_y,
                                 best_predicted_location=best_observed_location,
                                 best_predicted_location_value=self.evaluate_location_true_quality(
                                     best_observed_location),
                                 acqf_recommended_location=new_x,
                                 acqf_recommended_location_true_value=self.evaluate_location_true_quality(new_x),
                                 failing_constraint=(k),
                                 func_evals=evaluated_idx)  # last one gives index of failing constraint
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end = time.time() - start_time
        print(f'Total time: {end} seconds')

    def save_parameters(self, train_x, train_y, best_predicted_location,
                        best_predicted_location_value, acqf_recommended_location,
                        acqf_recommended_location_true_value, failing_constraint, func_evals):

        self.results.random_seed(self.seed)
        self.results.save_budget(self.budget)
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
        self.results.generate_pkl_file()

    def compute_next_sample(self, acquisition_function, smart_initial_locations=None):
        candidates, kgvalue = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.bounds,
            q=1,
            num_restarts=15, # can make smaller if too slow, not too small though
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

        return x_optimised, kgvalue


class EI_OptimizationLoop(OptimizationLoop):

    def __init__(self, black_box_func: BaseTestProblem, model: ConstrainedDeoupledGPModelWrapper,
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
            new_x, _ = self.compute_next_sample(acquisition_function=acquisition_function,
                                                smart_initial_locations=best_observed_location)

            for i in range(self.model_wrapper.getNumberOfOutputs()):
                new_y = self.evaluate_black_box_func(new_x, i)
                train_x[i] = torch.cat([train_x[i], new_x])
                train_y[i] = torch.cat([train_y[i], new_y])
                model = self.update_model(X=train_x, y=train_y)

            print(
                f"\nBatch{iteration:>2} finished: best value (EI) = "
                f"({best_observed_value:>4.5f}), best location " + str(
                    best_observed_location.numpy()) + " current sample decision x: " + str(new_x.numpy()), end="\n"
            )

            self.save_parameters(train_x=train_x,
                                 train_y=train_y,
                                 best_predicted_location=best_observed_location,
                                 best_predicted_location_value=self.evaluate_location_true_quality(
                                     best_observed_location),
                                 acqf_recommended_location=new_x,
                                 acqf_recommended_location_true_value=self.evaluate_location_true_quality(new_x))
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end = time.time() - start_time
        print(f'Total time: {end} seconds')

    def save_parameters(self, train_x, train_y, best_predicted_location,
                        best_predicted_location_value, acqf_recommended_location,
                        acqf_recommended_location_true_value):

        self.results.random_seed(self.seed)
        self.results.save_budget(self.budget)
        self.results.save_input_data(train_x)
        self.results.save_output_data(train_y)
        self.results.save_number_initial_points(self.number_initial_designs)
        self.results.save_performance_type(self.performance_type)
        self.results.save_best_predicted_location(best_predicted_location)
        self.results.save_best_predicted_location_true_value(best_predicted_location_value)
        self.results.save_acqf_recommended_location(acqf_recommended_location)
        self.results.save_acqf_recommended_location_true_value(acqf_recommended_location_true_value)
        self.results.generate_pkl_file()

    def compute_next_sample(self, acquisition_function, smart_initial_locations=None):
        candidates, kgvalue = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.bounds,
            q=1,
            num_restarts=15, # can make smaller if too slow, not too small though
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

        return x_optimised, kgvalue

class Decoupled_EIKG_OptimizationLoop(OptimizationLoop):

    def __init__(self, black_box_func: BaseTestProblem, model: ConstrainedDeoupledGPModelWrapper,
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

            print(
                f"\nBatch{iteration:>2} finished: best value (EI) = "
                f"({best_observed_value:>4.5f}), best location " + str(
                    best_observed_location.numpy()) + " current sample decision x: " + str(
                    new_x.numpy()) + f" on task {index}", end="\n"
            )

            self.save_parameters(train_x=train_x,
                                 train_y=train_y,
                                 best_predicted_location=best_observed_location,
                                 best_predicted_location_value=self.evaluate_location_true_quality(
                                     best_observed_location),
                                 acqf_recommended_location=new_x,
                                 acqf_recommended_location_true_value=self.evaluate_location_true_quality(new_x),
                                 failing_constraint="None",  # last one gives index of failing constraint
                                 acqf_recommended_output_index=index)
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')

        end = time.time() - start_time
        print(f'Total time: {end} seconds')

    def save_parameters(self, train_x, train_y, best_predicted_location,
                        best_predicted_location_value, acqf_recommended_output_index, acqf_recommended_location,
                        acqf_recommended_location_true_value, failing_constraint):

        self.results.random_seed(self.seed)
        self.results.save_budget(self.budget)
        self.results.save_input_data(train_x)
        self.results.save_output_data(train_y)
        self.results.save_number_initial_points(self.number_initial_designs)
        self.results.save_performance_type(self.performance_type)
        self.results.save_best_predicted_location(best_predicted_location)
        self.results.save_best_predicted_location_true_value(best_predicted_location_value)
        self.results.save_acqf_recommended_output_index(acqf_recommended_output_index)
        self.results.save_acqf_recommended_location(acqf_recommended_location)
        self.results.save_acqf_recommended_location_true_value(acqf_recommended_location_true_value)
        self.results.save_failing_constraint(failing_constraint)

        self.results.generate_pkl_file()

    def compute_next_sample(self, acquisition_function, smart_initial_locations=None):
        candidates, kgvalue = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.bounds,
            q=1,
            num_restarts=15, # can make smaller if too slow, not too small though
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

        return x_optimised, kgvalue
