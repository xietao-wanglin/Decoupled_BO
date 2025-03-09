from gpytorch import settings

from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType
from bo.bo_loops.bayesian_optimization_loop_type import BayesianOptimizationLoopType
from bo.bo_loops.bo_loop import OptimizationLoop, EI_Decoupled_OptimizationLoop, EI_OptimizationLoop, \
    Decoupled_EIKG_OptimizationLoop, CoupledAndDecoupledOptimizationLoop
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import *

device = torch.device("cpu")
dtype = torch.double
torch.set_default_dtype(dtype)
settings.min_fixed_noise._global_double_value = 1e-6


class BayesianOptimizationLoopFactory:
    def __init__(self, black_box_function, constrained_obj, model, seed, budget, penalty_value, costs,
                 number_of_constraints, base_file_name):
        self.base_file_name = base_file_name
        self.number_of_constraints = number_of_constraints
        self.costs = costs
        self.penalty_value = penalty_value
        self.budget = budget
        self.seed = seed
        self.model = model
        self.constrained_obj = constrained_obj
        self.black_box_function = black_box_function

    def create(self, bayesian_optimization_loop_type: BayesianOptimizationLoopType):
        number_initial_designs = 6
        dim = self.black_box_function.dim
        bounds = torch.zeros(2, dim, device=device, dtype=dtype)
        bounds[0] = 0.0
        bounds[1] = 1.0
        performance_type = "model"
        if bayesian_optimization_loop_type == BayesianOptimizationLoopType.DCKG_CKG:
            print('\n Starting dcKG + cKG:')
            results = Results(filename=self.base_file_name + "_dckg_ckg" + str(self.seed) + ".pkl")
            bo_loop = CoupledAndDecoupledOptimizationLoop(black_box_func=self.black_box_function,
                                                          objective=self.constrained_obj,
                                                          ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                                          bounds=bounds,
                                                          performance_type=performance_type,
                                                          model=self.model,
                                                          seed=self.seed,
                                                          budget=self.budget,
                                                          number_initial_designs=number_initial_designs,
                                                          results=results,
                                                          penalty_value=torch.tensor([self.penalty_value]))

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.DCKG:
            print('\n Starting dcKG:')
            results = Results(filename=self.base_file_name + "_dckg" + str(self.seed) + ".pkl")
            bo_loop = OptimizationLoop(black_box_func=self.black_box_function,
                                       objective=self.constrained_obj,
                                       ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                       bounds=bounds,
                                       performance_type=performance_type,
                                       model=self.model,
                                       seed=self.seed,
                                       budget=self.budget,
                                       number_initial_designs=number_initial_designs,
                                       results=results,
                                       penalty_value=torch.tensor([self.penalty_value]))

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.EIKG:
            print('\n Starting EI+KG:')
            results = Results(filename=self.base_file_name + "_ei_kg" + str(self.seed) + ".pkl")
            bo_loop = Decoupled_EIKG_OptimizationLoop(black_box_func=self.black_box_function,
                                                      objective=self.constrained_obj,
                                                      ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                                      bounds=bounds,
                                                      performance_type=performance_type,
                                                      model=self.model,
                                                      seed=self.seed,
                                                      budget=self.budget,
                                                      number_initial_designs=number_initial_designs,
                                                      results=results,
                                                      penalty_value=torch.tensor([self.penalty_value]))

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.DEI:
            print('\n Starting dEI:')
            results = Results(filename=self.base_file_name + "_dei" + str(self.seed) + ".pkl")
            bo_loop = EI_Decoupled_OptimizationLoop(black_box_func=self.black_box_function,
                                                    objective=self.constrained_obj,
                                                    ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                                    bounds=bounds,
                                                    performance_type=performance_type,
                                                    model=self.model,
                                                    seed=self.seed,
                                                    budget=self.budget,
                                                    number_initial_designs=number_initial_designs,
                                                    results=results,
                                                    penalty_value=torch.tensor([self.penalty_value]))

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.CEI:
            print('\n Starting cEI:')
            results = Results(filename=self.base_file_name + "_cei" + str(self.seed) + ".pkl")
            bo_loop = EI_OptimizationLoop(black_box_func=self.black_box_function,
                                          objective=self.constrained_obj,
                                          ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                          bounds=bounds,
                                          performance_type=performance_type,
                                          model=self.model,
                                          seed=self.seed,
                                          budget=int(self.budget / (self.number_of_constraints + 1)),
                                          number_initial_designs=number_initial_designs,
                                          results=results,
                                          penalty_value=torch.tensor([self.penalty_value]))

        # Coupled cKG
        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.CKG:
            print('\n Starting cKG:')
            results = Results(filename=self.base_file_name + "_ckg" + str(self.seed) + ".pkl")
            loop = EI_OptimizationLoop(black_box_func=self.black_box_function, objective=self.constrained_obj,
                                       ei_type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                       bounds=bounds,
                                       performance_type=performance_type, model=self.model, seed=self.seed,
                                       budget=int(self.budget / (self.number_of_constraints + 1)),
                                       number_initial_designs=number_initial_designs, results=results,
                                       penalty_value=torch.tensor([self.penalty_value]))
            bo_loop = loop
        else:
            raise TypeError("Bayesian Optimization loop type not recognized")
        return bo_loop
