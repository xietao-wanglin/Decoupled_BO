import os

from gpytorch import settings

from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType  # includes V2 types
from bo.bo_loops.bayesian_optimization_loop_type import BayesianOptimizationLoopType
from bo.bo_loops.bo_loop import OptimizationLoop, EI_Decoupled_OptimizationLoop, EI_OptimizationLoop, \
    Decoupled_EIKG_OptimizationLoop, CoupledAndDecoupledOptimizationLoop, OPT_UCB_OptimizationLoop, \
    AllSourcesOptimizationLoop, IndependentSourcesOptimizationLoop
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import *

from bo.device_utils import DEVICE as device, DTYPE as dtype
torch.set_default_dtype(dtype)
settings.min_fixed_noise._global_double_value = 1e-6


class BayesianOptimizationLoopFactory:
    def __init__(self, black_box_function: SingleObjectiveProblem, constrained_obj, model, seed, budget, penalty_value,
                 costs,
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

    def _load_or_skip(self, filename, effective_budget):
        """Detect existing results file and decide whether to skip, resume, or run fresh.

        Returns (results_obj, resume_state_or_None, skip_flag).
        Skip rule: if any per-iteration history was already saved past `effective_budget`,
        the run is considered done. We use budget_consumed[-1] when available (decoupled
        loops record it) and fall back to len(acqf_recommended_location) for coupled
        loops that don't track budget_consumed.
        """
        filepath = os.path.join('results', filename)
        if not os.path.exists(filepath):
            return Results(filename=filename), None, False
        loaded = Results.load_from_file(filepath)
        if loaded.budget_consumed:
            last_consumed = float(loaded.budget_consumed[-1])
        else:
            last_consumed = float(len(loaded.acqf_recommended_location))
        if last_consumed >= effective_budget:
            print(f"[skip] {filename}: consumed={last_consumed} >= target={effective_budget} "
                  f"(stored budget={loaded.budget})")
            return loaded, None, True
        print(f"[resume] {filename}: consumed={last_consumed}, stored budget={loaded.budget}, "
              f"extending to {effective_budget}")
        return loaded, {
            'train_x': loaded.input_data,
            'train_y': loaded.output_data,
            'budget_consumed': last_consumed,
        }, False

    @staticmethod
    def _resume_kwargs(resume_state):
        if resume_state is None:
            return {}
        return {
            'initial_train_x': resume_state['train_x'],
            'initial_train_y': resume_state['train_y'],
            'initial_budget_consumed': resume_state['budget_consumed'],
        }

    def create(self, bayesian_optimization_loop_type: BayesianOptimizationLoopType, number_initial_designs):
        dim = self.black_box_function.dim
        bounds = torch.zeros(2, dim, device=device, dtype=dtype)
        bounds[0] = 0.0
        bounds[1] = 1.0
        performance_type = "model"
        if bayesian_optimization_loop_type == BayesianOptimizationLoopType.DCKG_CKG:
            print('\n Starting dcKG + cKG:')
            filename = self.base_file_name + "_dckg_ckg" + str(self.seed) + ".pkl"
            results, resume_state, skip = self._load_or_skip(filename, self.budget)
            if skip:
                return None
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
                                                          costs=self.costs,
                                                          penalty_value=torch.tensor([self.penalty_value]),
                                                          **self._resume_kwargs(resume_state))

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.DCKG:
            print('\n Starting dcKG:')
            filename = self.base_file_name + "_dckg" + str(self.seed) + ".pkl"
            results, resume_state, skip = self._load_or_skip(filename, self.budget)
            if skip:
                return None
            bo_loop = OptimizationLoop(black_box_func=self.black_box_function,
                                       objective=self.constrained_obj,
                                       ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                       bounds=bounds,
                                       performance_type=performance_type,
                                       model=self.model,
                                       seed=self.seed,
                                       budget=self.budget,
                                       number_initial_designs=number_initial_designs,
                                       costs=self.costs,
                                       results=results,
                                       penalty_value=torch.tensor([self.penalty_value]),
                                       **self._resume_kwargs(resume_state))

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.EIKG:
            print('\n Starting EI+KG:')
            filename = self.base_file_name + "_ei_kg" + str(self.seed) + ".pkl"
            results, resume_state, skip = self._load_or_skip(filename, self.budget)
            if skip:
                return None
            bo_loop = Decoupled_EIKG_OptimizationLoop(black_box_func=self.black_box_function,
                                                      objective=self.constrained_obj,
                                                      ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                                      bounds=bounds,
                                                      performance_type=performance_type,
                                                      model=self.model,
                                                      seed=self.seed,
                                                      budget=self.budget,
                                                      number_initial_designs=number_initial_designs,
                                                      costs=self.costs,
                                                      results=results,
                                                      penalty_value=torch.tensor([self.penalty_value]),
                                                      **self._resume_kwargs(resume_state))

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.DEI:
            print('\n Starting dEI:')
            filename = self.base_file_name + "_dei" + str(self.seed) + ".pkl"
            results, resume_state, skip = self._load_or_skip(filename, self.budget)
            if skip:
                return None
            bo_loop = EI_Decoupled_OptimizationLoop(black_box_func=self.black_box_function,
                                                    objective=self.constrained_obj,
                                                    ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                                    bounds=bounds,
                                                    performance_type=performance_type,
                                                    model=self.model,
                                                    seed=self.seed,
                                                    budget=self.budget,
                                                    number_initial_designs=number_initial_designs,  # 36
                                                    costs=self.costs,
                                                    results=results,
                                                    penalty_value=torch.tensor([self.penalty_value]),
                                                    **self._resume_kwargs(resume_state))

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.CEI:
            print('\n Starting cEI:')
            filename = self.base_file_name + "_cei" + str(self.seed) + ".pkl"
            effective_budget = int(self.budget / (self.number_of_constraints + 1))
            results, resume_state, skip = self._load_or_skip(filename, effective_budget)
            if skip:
                return None
            bo_loop = EI_OptimizationLoop(black_box_func=self.black_box_function,
                                          objective=self.constrained_obj,
                                          ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                          bounds=bounds,
                                          performance_type=performance_type,
                                          model=self.model,
                                          seed=self.seed,
                                          budget=effective_budget,
                                          number_initial_designs=number_initial_designs,
                                          results=results,
                                          costs=self.costs,
                                          penalty_value=torch.tensor([self.penalty_value]),
                                          **self._resume_kwargs(resume_state))

        # Coupled cKG
        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.CKG:
            print('\n Starting cKG:')
            filename = self.base_file_name + "_ckg" + str(self.seed) + ".pkl"
            effective_budget = int(self.budget / (self.number_of_constraints + 1))
            results, resume_state, skip = self._load_or_skip(filename, effective_budget)
            if skip:
                return None
            loop = EI_OptimizationLoop(black_box_func=self.black_box_function, objective=self.constrained_obj,
                                       ei_type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                       bounds=bounds,
                                       performance_type=performance_type, model=self.model, seed=self.seed,
                                       budget=effective_budget,
                                       number_initial_designs=number_initial_designs, results=results,
                                       penalty_value=torch.tensor([self.penalty_value]),
                                       **self._resume_kwargs(resume_state))
            bo_loop = loop
        # ---- Refactored GPU-aware variants ----
        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.DCKG_CKG_V2:
            print('\n Starting dcKG + cKG (V2 refactored):')
            filename = self.base_file_name + "_dckg_ckg_v2_" + str(self.seed) + ".pkl"
            results, resume_state, skip = self._load_or_skip(filename, self.budget)
            if skip:
                return None
            bo_loop = CoupledAndDecoupledOptimizationLoop(black_box_func=self.black_box_function,
                                                          objective=self.constrained_obj,
                                                          ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2,
                                                          bounds=bounds,
                                                          performance_type=performance_type,
                                                          model=self.model,
                                                          seed=self.seed,
                                                          budget=self.budget,
                                                          number_initial_designs=number_initial_designs,
                                                          results=results,
                                                          costs=self.costs,
                                                          penalty_value=torch.tensor([self.penalty_value]),
                                                          **self._resume_kwargs(resume_state))

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.DCKG_V2:
            print('\n Starting dcKG (V2 refactored):')
            filename = self.base_file_name + "_dckg_v2_" + str(self.seed) + ".pkl"
            results, resume_state, skip = self._load_or_skip(filename, self.budget)
            if skip:
                return None
            bo_loop = OptimizationLoop(black_box_func=self.black_box_function,
                                       objective=self.constrained_obj,
                                       ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2,
                                       bounds=bounds,
                                       performance_type=performance_type,
                                       model=self.model,
                                       seed=self.seed,
                                       budget=self.budget,
                                       number_initial_designs=number_initial_designs,
                                       costs=self.costs,
                                       results=results,
                                       penalty_value=torch.tensor([self.penalty_value]),
                                       **self._resume_kwargs(resume_state))

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.CKG_V2:
            print('\n Starting cKG (V2 refactored):')
            filename = self.base_file_name + "_ckg_v2_" + str(self.seed) + ".pkl"
            effective_budget = int(self.budget / (self.number_of_constraints + 1))
            results, resume_state, skip = self._load_or_skip(filename, effective_budget)
            if skip:
                return None
            loop = EI_OptimizationLoop(black_box_func=self.black_box_function, objective=self.constrained_obj,
                                       ei_type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2,
                                       bounds=bounds,
                                       performance_type=performance_type, model=self.model, seed=self.seed,
                                       budget=effective_budget,
                                       number_initial_designs=number_initial_designs, results=results,
                                       penalty_value=torch.tensor([self.penalty_value]),
                                       **self._resume_kwargs(resume_state))
            bo_loop = loop

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.DCKG_ALL_SOURCES:
            print('\n Starting dcKG all-sources (single optimize_acqf):')
            filename = self.base_file_name + "_dckg_allsrc_" + str(self.seed) + ".pkl"
            results, resume_state, skip = self._load_or_skip(filename, self.budget)
            if skip:
                return None
            bo_loop = AllSourcesOptimizationLoop(
                black_box_func=self.black_box_function,
                objective=self.constrained_obj,
                ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2,
                bounds=bounds,
                performance_type=performance_type,
                model=self.model,
                seed=self.seed,
                budget=self.budget,
                number_initial_designs=number_initial_designs,
                costs=self.costs,
                results=results,
                penalty_value=torch.tensor([self.penalty_value]),
                **self._resume_kwargs(resume_state),
            )

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.DCKG_INDEPENDENT:
            print('\n Starting dcKG independent sources:')
            filename = self.base_file_name + "_dckg_indep_" + str(self.seed) + ".pkl"
            results, resume_state, skip = self._load_or_skip(filename, self.budget)
            if skip:
                return None
            bo_loop = IndependentSourcesOptimizationLoop(
                black_box_func=self.black_box_function,
                objective=self.constrained_obj,
                ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2,
                bounds=bounds,
                performance_type=performance_type,
                model=self.model,
                seed=self.seed,
                budget=self.budget,
                number_initial_designs=number_initial_designs,
                costs=self.costs,
                results=results,
                penalty_value=torch.tensor([self.penalty_value]),
                **self._resume_kwargs(resume_state),
            )

        elif bayesian_optimization_loop_type == BayesianOptimizationLoopType.OPTIMISTIC_UCB:
            print('\n Starting optimisitic ucb:')
            filename = self.base_file_name + "_optimistic_ucb" + str(self.seed) + ".pkl"
            results, resume_state, skip = self._load_or_skip(filename, self.budget)
            if skip:
                return None
            loop = OPT_UCB_OptimizationLoop(black_box_func=self.black_box_function,
                                            objective=self.constrained_obj,
                                            ei_type=AcquisitionFunctionType.OPTIMISTIC_UCB,
                                            bounds=bounds,
                                            performance_type=performance_type, model=self.model, seed=self.seed,
                                            budget=self.budget,
                                            number_initial_designs=number_initial_designs, results=results,
                                            penalty_value=torch.tensor([1000]),
                                            **self._resume_kwargs(resume_state))
            bo_loop = loop
        else:
            raise TypeError("Bayesian Optimization loop type not recognized")
        return bo_loop
