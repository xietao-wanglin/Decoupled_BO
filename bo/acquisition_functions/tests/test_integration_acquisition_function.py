from typing import Optional

import torch
from botorch.acquisition import ConstrainedMCObjective
from botorch.optim import optimize_acqf
from botorch.utils.testing import BotorchTestCase
from gpytorch import settings
from numpy.ma.testutils import assert_close

from Launcher import constraint_callable_wrapper
from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType, acquisition_function_factory
from bo.bo_loop import OptimizationLoop
from bo.model.Model import ConstrainedDeoupledGPModelWrapper, ConstrainedPosteriorMean
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import ConstrainedBraninNew

device = torch.device("cpu")
dtype = torch.double
torch.set_default_dtype(dtype)
settings.min_fixed_noise._global_double_value = 1e-16


def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]


class TestDecoupledKgIntegration(BotorchTestCase):
    def test_sampled_locations_have_cKG_value_zero_coupled(self):
        torch.manual_seed(0)
        dtype = torch.double
        torch.set_default_dtype(dtype)
        black_box_function = ConstrainedBraninNew(noise_std=1e-9, negate=True)
        num_constraints = 1
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)])
        results = Results(filename="remove_me.pkl")
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)
        loop = OptimizationLoop(black_box_func=black_box_function,
                                objective=constrained_obj,
                                ei_type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                bounds=bounds,
                                performance_type="model",
                                model=model,
                                seed=0,
                                budget=50,
                                number_initial_designs=6,
                                results=results,
                                penalty_value=torch.tensor([2.0]))

        x_train = torch.rand((20, 2))
        X = [x_train, x_train]
        Y = [black_box_function.evaluate_task(x, task) for task, x in enumerate(X)]
        updated_model = loop.update_model(X, Y)

        constrained_posterior_model = ConstrainedPosteriorMean(updated_model, maximize=True, penalty_value=torch.Tensor([0]))
        argmax_mean, _ = optimize_acqf(
            acq_function=constrained_posterior_model,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        kg = acquisition_function_factory(type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                          model=updated_model, objective=constrained_obj, best_value=None,
                                          idx=None, number_of_outputs=2, penalty_value=loop.penalty_value,
                                          iteration=0, initial_condition_internal_optimizer=argmax_mean)
        kg.use_scipy = True
        for x in x_train:
            acqf = kg.forward(x[None, :])
            history = kg.compute_discrete_kg_values_fast.history

            for i in range(len(history)):
                x_discretisation = history[i]["kwargs"]["discretisation_per_realisation"]
                objective_mean = history[i]["kwargs"]["objective_mean"].detach().numpy()
                objective_variance = history[i]["kwargs"]["objective_variance"].detach().numpy()
                objective_posterior_covariance = history[i]["kwargs"]["objective_posterior_covariance"].detach().numpy()
                probability_of_feasibility = history[i]["kwargs"]["probability_of_feasibility"].detach().numpy()
                acqf = history[i]["result"].detach().numpy()

                current_posterior = updated_model.posterior(x_discretisation)
                assert_close(probability_of_feasibility,
                             constrained_posterior_model.evaluate_feasibility_by_index(x_discretisation, 1).detach().numpy(), 5)
                assert_close(x_discretisation[0].detach().numpy(), x.detach().numpy())
                assert_close(objective_mean, current_posterior.mean[:, 0].detach().numpy(), 7)
                assert_close(objective_variance, current_posterior.variance[:, 0].detach().numpy(), 7)
                assert_close(objective_posterior_covariance.reshape(-1),
                             updated_model.models[0].posterior(
                                 x_discretisation).mvn.covariance_matrix.detach().numpy().reshape(-1), 7)
                assert_close(0.0, acqf, 5)
            kg.compute_discrete_kg_values_fast.history.clear()
