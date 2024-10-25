from typing import Optional

from botorch.acquisition import ConstrainedMCObjective
from gpytorch import settings

from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType
from bo.bo_loop import OptimizationLoop, EI_Decoupled_OptimizationLoop, EI_OptimizationLoop, \
    Decoupled_EIKG_OptimizationLoop, CoupledAndDecoupledOptimizationLoop
from bo.model.Model import ConstrainedDeoupledGPModelWrapper
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import *

device = torch.device("cpu")
dtype = torch.double
torch.set_default_dtype(dtype)
settings.min_fixed_noise._global_double_value = 1e-6


def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]


def constraint_callable_wrapper(constraint_idx):
    def constraint_callable(Z):
        return Z[..., constraint_idx]

    return constraint_callable


if __name__ == "__main__":

    # Set problem and constrains here
    black_box_function = MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True)
    num_constraints = 9
    filename_pf = 'mystery_lots_final_'
    budget = 240
    penalty = 40.0

    DCKG_CKG = True
    DCKG = True
    EIKG = True
    DEI = False
    CEI = True
    CKG = True

    seed = int(sys.argv[1])
    print(f'Running seed {seed}')

    # Decoupled cKG
    if DCKG_CKG:
        print('\n Starting dcKG + cKG:')
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
        )
        results = Results(filename=filename_pf + "dckg_ckg" + str(seed) + ".pkl")
        loop_dckg_ckg = CoupledAndDecoupledOptimizationLoop(black_box_func=black_box_function,
                                                        objective=constrained_obj,
                                                        ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                                        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device,
                                                                            dtype=dtype),
                                                        performance_type="model",
                                                        model=model,
                                                        seed=seed,
                                                        budget=budget,
                                                        number_initial_designs=6,
                                                        results=results,
                                                        penalty_value=torch.tensor([penalty]))
        loop_dckg_ckg.run()
    
    # Decoupled KG
    if DCKG:
        print('\n Starting dcKG:')
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
        )
        results = Results(filename=filename_pf + "dckg" + str(seed) + ".pkl")
        loop_dckg = OptimizationLoop(black_box_func=black_box_function,
                                     objective=constrained_obj,
                                     ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                     bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype),
                                     performance_type="model",
                                     model=model,
                                     seed=seed,
                                     budget=budget,
                                     number_initial_designs=6,
                                     results=results,
                                     penalty_value=torch.tensor([penalty]))
        loop_dckg.run()

    # EI + KG
    if EIKG:
        print('\n Starting EI+KG:')
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
        )
        results = Results(filename=filename_pf + "eikg" + str(seed) + ".pkl")
        loop_eikg = Decoupled_EIKG_OptimizationLoop(black_box_func=black_box_function,
                                                    objective=constrained_obj,
                                                    ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                                    bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device,
                                                                        dtype=dtype),
                                                    performance_type="model",
                                                    model=model,
                                                    seed=seed,
                                                    budget=budget,
                                                    number_initial_designs=6,
                                                    results=results,
                                                    penalty_value=torch.tensor([penalty]))
        loop_eikg.run()

    # Decoupled EI
    if DEI:
        print('\n Starting dEI:')
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
        )
        results = Results(filename=filename_pf + "dei" + str(seed) + ".pkl")
        loop_dei = EI_Decoupled_OptimizationLoop(black_box_func=black_box_function,
                                                 objective=constrained_obj,
                                                 ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                                 bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device,
                                                                     dtype=dtype),
                                                 performance_type="model",
                                                 model=model,
                                                 seed=seed,
                                                 budget=budget,
                                                 number_initial_designs=6,
                                                 results=results,
                                                 penalty_value=torch.tensor([penalty]))
        loop_dei.run()

    # Coupled EI
    if CEI:
        print('\n Starting cEI:')
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
        )
        results = Results(filename=filename_pf + "cei" + str(seed) + ".pkl")
        loop_cei = EI_OptimizationLoop(black_box_func=black_box_function,
                                       objective=constrained_obj,
                                       ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                       bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype),
                                       performance_type="model",
                                       model=model,
                                       seed=seed,
                                       budget=int(budget / (num_constraints + 1)),
                                       number_initial_designs=6,
                                       results=results,
                                       penalty_value=torch.tensor([penalty]))
        loop_cei.run()

    # Coupled cKG
    if CKG:
        print('\n Starting cKG:')
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
        )
        results = Results(filename=filename_pf + "ckg" + str(seed) + ".pkl")
        loop_ckg = EI_OptimizationLoop(black_box_func=black_box_function,
                                       objective=constrained_obj,
                                       ei_type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                       bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype),
                                       performance_type="model",
                                       model=model,
                                       seed=seed,
                                       budget=int(budget / (num_constraints + 1)),
                                       number_initial_designs=6,
                                       results=results,
                                       penalty_value=torch.tensor([penalty]))
        loop_ckg.run()
