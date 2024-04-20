from typing import Optional

import torch
from botorch.acquisition import ConstrainedMCObjective

from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType
from bo.bo_loop import OptimizationLoop
from bo.model.Model import ConstrainedDeoupledGPModelWrapper
from bo.result_utils.result_container import Results
from bo.constrained_functions.synthetic_problems import ConstrainedBranin

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
    # TODO: Launcher should be adapted to run different problems and random seeds....
    # NOTE: If the objective values are negative please change the penalty_value to something sensible.
    # NOTE: if code is too slow you may make it faster by chaging the number_of_raw_points in the acquisition function and reduce this number (you'll sacrifice accuracy).
    # NOTE: budget you could set it as in the ckg paper...
    # NOTE: number of initial designs could be set as in the paper.
    # Note: you may do 15-20 replications with different random seeds for each problem...
    # SUGGESTION!!!!! don't run EVERYTHING without checking...do a dummy run and make sure its optimizing....
    black_box_function = ConstrainedBranin(noise_std=1e-6, negate=True)
    num_constraints = 1
    seed = 0
    model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
    # define a feasibility-weighted objective for optimization
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
    )
    results = Results(filename="constrained_branin_" + str(seed) + ".pkl")
    loop = OptimizationLoop(black_box_func=black_box_function,
                            objective=constrained_obj,
                            ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype),
                            performance_type="model",
                            model=model,
                            seed=seed,
                            budget=5,
                            number_initial_designs=6,
                            results=results,
                            penalty_value=torch.tensor([0.0]))  # penalty value -M should be at least as low as the lowest value of the objective function
    # play with the penalty value if the objective function has negative values....
    loop.run()
