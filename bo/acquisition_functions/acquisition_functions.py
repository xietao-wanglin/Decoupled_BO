from enum import Enum, auto

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler


class AcquisitionFunctionType(Enum):
    BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT = auto()
    COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2 = auto()
    DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2 = auto()


def acquisition_function_factory(type, model, objective, best_value, idx, number_of_outputs, penalty_value, iteration,
                                 initial_condition_internal_optimizer):
    if type is AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT:
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([100]))
        return qExpectedImprovement(model=model, best_f=best_value, sampler=qmc_sampler, objective=objective)

    elif type is AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2:
        from bo.acquisition_functions.refactored_acquisition_functions import CoupledCKG
        return CoupledCKG(
            model,
            penalty_value=penalty_value,
            x_best_location=initial_condition_internal_optimizer,
            n_fantasies=7,
            n_constraint_samples=7,
            seed=iteration,
            objective=objective,
        )

    elif type is AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2:
        from bo.acquisition_functions.refactored_acquisition_functions import (
            ObjectiveDcKG, ConstraintDcKG,
        )
        common = dict(
            model=model, penalty_value=penalty_value,
            x_best_location=initial_condition_internal_optimizer,
            n_fantasies=7, seed=iteration, objective=objective,
        )
        if idx == 0:
            return ObjectiveDcKG(**common)
        else:
            return ConstraintDcKG(constraint_index=idx - 1, **common)

    else:
        raise TypeError("Acquisition function type not recognized")
