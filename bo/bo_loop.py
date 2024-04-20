from typing import Optional

import torch
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


class OptimizationLoop:

    def __init__(self, black_box_func: BaseTestProblem, model: ConstrainedDeoupledGPModelWrapper,
                 objective: Optional[MCAcquisitionObjective], ei_type: AcquisitionFunctionType, seed: int, budget: int,
                 performance_type: str, bounds: Tensor, results: Results,
                 penalty_value: Optional[Tensor] = torch.tensor([0.0]), number_initial_designs: Optional[int] = 6
                 ):

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

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y = self.generate_initial_data(n=self.number_initial_designs)
        model = self.update_model(train_x, train_y)

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
                print("task_idx", task_idx)
                acquisition_function = acquisition_function_factory(model=model,
                                                                    type=self.acquisition_function_type,
                                                                    objective=self.objective,
                                                                    best_value=best_observed_value,
                                                                    idx=task_idx,
                                                                    number_of_outputs=self.number_of_outputs,
                                                                    penalty_value=self.penalty_value,
                                                                    iteration=iteration)

                new_x, kgvalue = self.compute_next_sample(acquisition_function=acquisition_function)  # Coupled
                kg_values_list[task_idx] = kgvalue
                new_x_list.append(new_x)

            index = torch.argmax(kg_values_list)
            new_y = self.evaluate_black_box_func(new_x_list[index], index)

            train_x[index] = torch.cat([train_x[index], new_x_list[index]])
            train_y[index] = torch.cat([train_y[index], new_y])
            model = self.update_model(X=train_x, y=train_y)

            print(
                f"\nBatch {iteration:>2}: best_value (EI) = "
                f"({best_observed_value:>4.5f}), best location " + str(
                    best_observed_location) + " current sample decision x: " + str(new_x),
                end="",
            )
            self.save_parameters(train_x=train_x,
                                 train_y=train_y,
                                 best_predicted_location=best_observed_location,
                                 best_predicted_location_value=self.evaluate_location_true_quality(
                                     best_observed_location),
                                 acqf_recommended_location=new_x,
                                 acqf_recommended_location_true_value=self.evaluate_location_true_quality(new_x),
                                 acqf_recommended_output_index=index)

    def save_parameters(self, train_x, train_y, best_predicted_location,
                        best_predicted_location_value, acqf_recommended_output_index, acqf_recommended_location,
                        acqf_recommended_location_true_value):

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

    def compute_next_sample(self, acquisition_function):
        candidates, kgvalue = optimize_acqf(
            acq_function=acquisition_function,
            bounds=self.bounds,
            q=1,
            num_restarts=15, # can make smaller if too slow, not too small though
            raw_samples=128,  # used for intialization heuristic
            options={"maxiter": 60},
        )
        # observe new values
        new_x = candidates.detach()
        return new_x, kgvalue
