from botorch.acquisition import MCAcquisitionObjective
from botorch.test_functions.base import BaseTestProblem
import torch
import time
from torch import Tensor
from torch import tensor
from bo.acquisition_functions.Turbo import ScboState, create_trust_region
from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType, acquisition_function_factory
from bo.bo_loop import OptimizationLoop
from bo.model.Model import ConstrainedDeoupledGPModelWrapper
from bo.result_utils.result_container import Results
from bo.acquisition_functions.Turbo import update_state
from botorch.optim import optimize_acqf

dtype = torch.float

class turbo_boloop(OptimizationLoop):

    def __init__(self, black_box_func: BaseTestProblem, model: ConstrainedDeoupledGPModelWrapper, objective: MCAcquisitionObjective | None, ei_type: AcquisitionFunctionType, seed: int, budget: int, performance_type: str, bounds: Tensor, results: Results, penalty_value: Tensor | None = ..., number_initial_designs: int | None = 6, costs: Tensor | None = ...):
        super().__init__(black_box_func, model, objective, ei_type, seed, budget, performance_type, bounds, results, penalty_value, number_initial_designs, costs)

    def run(self):
        best_observed_all_sampled = []
        train_x, train_y = self.generate_initial_data(n=self.number_initial_designs)
        state = ScboState(dim = self.black_box_func.dim, batch_size = 1)
        model = self.update_model(train_x,train_y)
        start_time = time.time()
        for iteration in range(self.budget):
            best_observed_location, best_observed_value = self.best_observed(
                best_value_computation_type=self.performance_type,
                train_x=train_x,
                train_y=train_y,
                model=model,
                bounds=self.bounds)
            state = update_state(state=state, x_best_location = best_observed_location, model=model)
            tr_lb, tr_ub = create_trust_region(best_observed_location,state)
            best_observed_all_sampled.append(best_observed_value)
            #TODO restrict train_x/y to within trust region
            model = self.update_model(train_x, train_y)
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
                                                                    iteration=iteration)

                new_x, kgvalue = self.compute_next_sample(acquisition_function=acquisition_function, bounds=torch.stack([tr_lb,tr_ub]))  
                kg_values_list[task_idx] = kgvalue
                new_x_list.append(new_x)

            index = torch.argmax(torch.tensor(kg_values_list)/self.costs)
            new_y = self.evaluate_black_box_func(new_x_list[index], index)

            train_x[index] = torch.cat([train_x[index], new_x_list[index]])
            train_y[index] = torch.cat([train_y[index], new_y])
            model = self.update_model(X=train_x, y=train_y)


            print(
                f"\nBatch {iteration:>2} finished: best value (EI) = "
                f"({best_observed_value:>4.5f}), best location " + str(
                    best_observed_location) + " current sample decision x: " + str(new_x_list[index]) + f" on task {index}\n",
                end="",
            )
            self.save_parameters(train_x=train_x, train_y=train_y, best_predicted_location=best_observed_location,
                                 best_predicted_location_value=self.evaluate_location_true_quality(
                                     best_observed_location), acqf_recommended_output_index=index,
                                 acqf_recommended_location=new_x_list[index],
                                 acqf_recommended_location_true_value=self.evaluate_location_true_quality(
                                     new_x_list[index]))
            middle_time = time.time() - start_time
            print(f'took {middle_time} seconds')
        
        end = time.time() - start_time
        print(f'Total time: {end} seconds')

    def compute_next_sample(self, acquisition_function, bounds):
        candidates, kgvalue = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds,
            q=1,
            num_restarts=15, # can make smaller if too slow, not too small though
            raw_samples=128,  # used for intialization heuristic
            options={"maxiter": 60},
        )
        # observe new values
        new_x = candidates.detach()
        return new_x, kgvalue
