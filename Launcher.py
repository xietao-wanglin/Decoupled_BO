from typing import Optional

from botorch.acquisition import ConstrainedMCObjective
from gpytorch import settings

from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType
from bo.model.Model import ConstrainedDeoupledGPModelWrapper
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import *
from bo.turbo_loop import turbo_boloop

device = torch.device("cpu")
dtype = torch.double
torch.set_default_dtype(dtype)
settings.min_fixed_noise._global_double_value = 1e-09


def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]


def constraint_callable_wrapper(constraint_idx):
    def constraint_callable(Z):
        return Z[..., constraint_idx]

    return constraint_callable


if __name__ == "__main__":
    # TODO: save the information bayesian optimization information.

    black_box_function = ConstrainedFunc3(noise_std=1e-6, negate=True)
    num_constraints = 3
    seed = 0
    model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
    # define a feasibility-weighted objective for optimization
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
    )
    results = Results(filename="resultcheck_" + str(seed) + ".pkl")
    loop = turbo_boloop(black_box_func=black_box_function,
                        objective=constrained_obj,
                        ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype),
                        performance_type="model",
                        model=model,
                        seed=seed,
                        budget=20,
                        number_initial_designs=6,
                        results=results,
                        penalty_value=torch.tensor([100.0]))
    loop.run()
