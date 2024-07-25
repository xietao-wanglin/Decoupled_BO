from typing import Optional

from botorch.acquisition import ConstrainedMCObjective

from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType
from bo.bo_loop import *
from bo.model.Model import ConstrainedDeoupledGPModelWrapper
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import *

device = torch.device("cpu")
dtype = torch.double


def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]


def constraint_callable_wrapper(constraint_idx):
    def constraint_callable(Z):
        return Z[..., constraint_idx]

    return constraint_callable


if __name__ == "__main__":

    # Note: the launcher assumes that all inequalities are less than and the limit of the constraint is zero.
    # Transform accordingly in the problem.
    seed_list = [0]
    for s in seed_list:
        black_box_function = ConstrainedFunc3(noise_std=1e-6, negate=True)
        num_constraints = 3
        seed = s
        print(s)
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
        )
        results = Results(filename="resultcheck2" + str(seed) + ".pkl")
        loop = EI_Decoupled_OptimizationLoop(black_box_func=black_box_function,
                                objective=constrained_obj,
                                ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device,
                                                    dtype=dtype),
                                performance_type="model",
                                model=model,
                                seed=seed,
                                budget=60,
                                number_initial_designs=6,
                                results=results,
                                penalty_value=torch.tensor([40.0]))
        loop.run()
