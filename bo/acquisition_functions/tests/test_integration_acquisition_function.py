from typing import Optional

import numpy as np
import torch
from botorch.acquisition import ConstrainedMCObjective
from botorch.optim import optimize_acqf
from botorch.sampling import ListSampler
from botorch.utils.testing import BotorchTestCase
from gpytorch import settings
from matplotlib import pyplot as plt
from numpy.ma.testutils import assert_close, assert_equal

from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType, \
    DecopledHybridConstrainedKnowledgeGradient, acquisition_function_factory
from bo.bo_loops.bo_loop import OptimizationLoop
from bo.model.Model import ConstrainedDeoupledGPModelWrapper, ConstrainedPosteriorMean, constraint_callable_wrapper
from bo.result_utils.result_container import Results
from bo.samplers.samplers import objectiveQuantileSampler, RepeatedInterleavedSobolQMCNormalSampler
from bo.synthetic_test_functions.synthetic_test_functions import MysteryFunctionSuperRedundant

device = torch.device("cpu")
dtype = torch.double
torch.set_default_dtype(dtype)
settings.min_fixed_noise._global_double_value = 1e-16


def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]


class TestDecoupledKgIntegration(BotorchTestCase):
    def setUp(self):
        super().setUp()
        TESTING_MODE = True
        torch.manual_seed(0)
        dtype = torch.double
        torch.set_default_dtype(dtype)
        self.black_box_function = MysteryFunctionSuperRedundant(noise_std=1e-9, negate=True)
        num_constraints = 1
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        self.constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)])
        results = Results(filename="remove_me.pkl")
        self.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)
        self.loop = OptimizationLoop(black_box_func=self.black_box_function,
                                     objective=self.constrained_obj,
                                     ei_type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                     bounds=self.bounds,
                                     performance_type="model",
                                     model=model,
                                     seed=0,
                                     budget=50,
                                     number_initial_designs=6,
                                     results=results,
                                     penalty_value=torch.tensor([40.0]))

        self.x_train = torch.rand((6, 2))
        X = [self.x_train, self.x_train]
        Y = [self.black_box_function.evaluate_task(x, task) for task, x in enumerate(X)]
        self.updated_model = self.loop.update_model(X, Y)
        self.constrained_posterior_model = ConstrainedPosteriorMean(self.updated_model, maximize=True,
                                                                    penalty_value=torch.Tensor([0]))

    def test_discretisation(self):
        argmax_mean, _ = optimize_acqf(
            acq_function=self.constrained_posterior_model,
            bounds=self.bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        number_of_fantasies_for_objective = torch.Size([7])
        number_of_fantasies_for_constraints = torch.Size([3])
        total_number_of_fantasies = number_of_fantasies_for_objective.numel() * number_of_fantasies_for_constraints.numel()
        samplers = []
        samplers.append(objectiveQuantileSampler(sample_shape=number_of_fantasies_for_objective,
                                                 number_of_fantasies_for_constraints=number_of_fantasies_for_constraints))
        for _ in range(1):
            samplers.append(RepeatedInterleavedSobolQMCNormalSampler(sample_shape=number_of_fantasies_for_constraints,
                                                                     number_of_fantasies_for_objective=number_of_fantasies_for_objective))
        sampler_list = ListSampler(*samplers)
        x_eval_mask = torch.ones(1, 2, dtype=torch.bool)
        cKG = DecopledHybridConstrainedKnowledgeGradient(self.updated_model,
                                                         sampler=sampler_list,
                                                         num_fantasies=total_number_of_fantasies,
                                                         objective=self.constrained_obj,
                                                         number_of_raw_points=500,
                                                         number_of_restarts=1,
                                                         x_evaluation_mask=x_eval_mask,
                                                         seed=0, penalty_value=self.loop.penalty_value,
                                                         x_best_location=argmax_mean,
                                                         evaluate_all_sources=True)

        # discretisation = cKG.compute_optimized_X_discretisation(self.x_train[0].unsqueeze(0))
        acqf_value = cKG.forward(self.x_train[0][None, :])
        print("acqf_value: ", acqf_value)
        print("ok")

    def test_sampled_locations_have_cKG_value_zero_coupled(self):
        argmax_mean, _ = optimize_acqf(
            acq_function=self.constrained_posterior_model,
            bounds=self.bounds,
            q=1,
            num_restarts=20,
            raw_samples=248)
        number_of_fantasies_for_objective = torch.Size([7])
        number_of_fantasies_for_constraints = torch.Size([5])
        total_number_of_fantasies = number_of_fantasies_for_objective.numel() * number_of_fantasies_for_constraints.numel()
        samplers = []
        samplers.append(objectiveQuantileSampler(sample_shape=number_of_fantasies_for_objective,
                                                 number_of_fantasies_for_constraints=number_of_fantasies_for_constraints))
        for _ in range(1):
            samplers.append(RepeatedInterleavedSobolQMCNormalSampler(sample_shape=number_of_fantasies_for_constraints,
                                                                     number_of_fantasies_for_objective=number_of_fantasies_for_objective))
        sampler_list = ListSampler(*samplers)
        x_eval_mask = torch.ones(1, 2, dtype=torch.bool)
        kg = DecopledHybridConstrainedKnowledgeGradient(self.updated_model,
                                                        sampler=sampler_list,
                                                        num_fantasies=total_number_of_fantasies,
                                                        objective=self.constrained_obj,
                                                        number_of_raw_points=500,
                                                        number_of_restarts=1,
                                                        x_evaluation_mask=x_eval_mask,
                                                        seed=0, penalty_value=self.loop.penalty_value,
                                                        x_best_location=argmax_mean,
                                                        evaluate_all_sources=True)
        kg.use_scipy = True
        for x in self.x_train:
            kg.forward(x[None, :])
            history = kg.compute_discrete_kg_values_fast.history
            assert_equal(len(history), number_of_fantasies_for_constraints.numel())
            for i in range(len(history)):
                x_discretisation = history[i]["kwargs"]["discretisation_per_realisation"]
                objective_mean = history[i]["kwargs"]["objective_mean"].detach().numpy()
                objective_variance = history[i]["kwargs"]["objective_variance"].detach().numpy()
                objective_posterior_covariance = history[i]["kwargs"]["objective_posterior_covariance"].detach().numpy()
                probability_of_feasibility = history[i]["kwargs"]["probability_of_feasibility"].detach().numpy()
                acqf = history[i]["result"].detach().numpy()
                current_posterior = self.updated_model.posterior(x_discretisation)
                assert_close(probability_of_feasibility,
                             self.constrained_posterior_model.evaluate_feasibility_by_index(x_discretisation,
                                                                                            1).detach().numpy(), 5)
                assert_close(x_discretisation[0].detach().numpy(), x.detach().numpy())
                assert_close(objective_mean, current_posterior.mean[:, 0].detach().numpy(), 7)
                assert_close(objective_variance, current_posterior.variance[:, 0].detach().numpy(), 7)
                assert_close(objective_posterior_covariance.reshape(-1),
                             self.updated_model.models[0].posterior(
                                 x_discretisation).mvn.covariance_matrix.detach().numpy().reshape(-1), 7)
                assert_close(0.0, acqf, 5)
            kg.compute_discrete_kg_values_fast.history.clear()

    def test_shapes_correct_deleteme_later_should_be_unitest(self):
        argmax_mean, _ = optimize_acqf(
            acq_function=self.constrained_posterior_model,
            bounds=self.bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        number_of_fantasies_for_objective = torch.Size([7])
        number_of_fantasies_for_constraints = torch.Size([5])
        total_number_of_fantasies = number_of_fantasies_for_objective.numel() * number_of_fantasies_for_constraints.numel()
        samplers = []
        samplers.append(objectiveQuantileSampler(sample_shape=number_of_fantasies_for_objective,
                                                 number_of_fantasies_for_constraints=number_of_fantasies_for_constraints))
        for _ in range(1):
            samplers.append(
                RepeatedInterleavedSobolQMCNormalSampler(sample_shape=number_of_fantasies_for_constraints,
                                                         number_of_fantasies_for_objective=number_of_fantasies_for_objective))
        sampler_list = ListSampler(*samplers)
        x_eval_mask = torch.ones(1, 2, dtype=torch.bool)
        cKG = DecopledHybridConstrainedKnowledgeGradient(self.updated_model,
                                                         sampler=sampler_list,
                                                         num_fantasies=total_number_of_fantasies,
                                                         objective=self.constrained_obj,
                                                         number_of_raw_points=500,
                                                         number_of_restarts=1,
                                                         x_evaluation_mask=x_eval_mask,
                                                         seed=0, penalty_value=self.loop.penalty_value,
                                                         x_best_location=argmax_mean,
                                                         evaluate_all_sources=True)

        acqf_value = cKG.forward(torch.rand((72, 1, 2)))
        print("acqf_value: ", acqf_value)
        print("ok")

    def test_optimized_discretisation(self):
        argmax_mean, _ = optimize_acqf(
            acq_function=self.constrained_posterior_model,
            bounds=self.bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        acqf = acquisition_function_factory(type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                            model=self.updated_model, objective=self.constrained_obj,
                                            best_value=None, idx=None, number_of_outputs=2,
                                            penalty_value=self.loop.penalty_value, iteration=0,
                                            initial_condition_internal_optimizer=argmax_mean)

        num_test_locations = 5
        X_test = torch.rand((num_test_locations, 1, 2))
        X_test[0, :, : ] = argmax_mean
        discretisation, fantasy_model = acqf.compute_optimized_X_discretisation(X_test, False)
        plot_X_locations = torch.cat([torch.cat([torch.rand((5000, 1, 2))[:, None, None, :, :]] * 5, dim=2)] * 35, dim=1)
        constrained_posterior_mean_model = ConstrainedPosteriorMean(model=fantasy_model,
                                                                    penalty_value=acqf.penalty_value)
        posterior_model_value = constrained_posterior_mean_model(plot_X_locations)
        # posterior_mean_objective = fantasy_model.posterior(plot_X_locations).mean[..., 0]
        # posterior_mean_constraint = fantasy_model.posterior(plot_X_locations).mean[..., 1]


        posterior_mean_test_locations_objective = fantasy_model.posterior(torch.ones(discretisation.shape) * argmax_mean ).mean[..., 0][:, :, 0, :].squeeze()
        posterior_mean_test_locations_constraint = fantasy_model.posterior(torch.ones(discretisation.shape) * argmax_mean ).mean[..., 1][:, :, 0, :].squeeze()

        plt.scatter(torch.tensor([1.0] * 35), posterior_mean_test_locations_objective.detach().numpy())
        plt.show()
        plt.scatter(torch.tensor([1.0] * 35), posterior_mean_test_locations_constraint.detach().numpy())
        plt.show()
        # raise
        for i in range(num_test_locations):
            x_test = X_test[i, ...].squeeze()
            # raise
            for fantasy_idx in range(35):
                fantasy_discretisation = discretisation[:, fantasy_idx, i, :].detach().numpy().squeeze()
                plot_x_locations = plot_X_locations[:, fantasy_idx, i, :].detach().numpy().squeeze()
                plot_x_values = posterior_model_value[:, fantasy_idx, i].detach().numpy().squeeze()
                best_plot_x_location = plot_x_locations[np.argmax(plot_x_values)]
                # fantasy_posterior_mean_objective = posterior_mean_objective[:, fantasy_idx, i].detach().numpy().squeeze()
                # fantasy_posterior_mean_constraint = posterior_mean_constraint[:, fantasy_idx, i].detach().numpy().squeeze()
            #
            #     # Create the figure and subplots
            #     # fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)  # 1 row, 3 columns
            #     # axes[0].scatter(plot_x_locations[:, 0], plot_x_locations[:, 1], c=fantasy_posterior_mean_objective)
            #     # axes[0].set_title("fanatsy mean")
            #     # axes[1].scatter(plot_x_locations[:, 0], plot_x_locations[:, 1], c=fantasy_posterior_mean_constraint)
            #     # axes[1].set_title("fanatsy constraint")
            #     # axes[2].scatter(plot_x_locations[:, 0], plot_x_locations[:, 1], c=plot_x_values)
            #     # axes[2].set_title("fanatsy constraint")
            #     # plt.tight_layout()
            #     # plt.show()

            plt_scatter = plt.scatter(plot_x_locations[:, 0], plot_x_locations[:, 1], c=plot_x_values)
            plt.scatter(x_test[0], x_test[1], color="red", label="test location")
            plt.scatter(best_plot_x_location[0], best_plot_x_location[1], color="black", label="best from plot")
            plt.scatter(fantasy_discretisation[0], fantasy_discretisation[1], color="blue", label="best fantasised")
            plt.title("test_location: " + str(i) + " virtual sample: " + str(fantasy_idx))
            plt.xlim((0,1))
            plt.ylim((0,1))
            plt.legend()
            plt.colorbar(plt_scatter)
            plt.show()
        print("ok")


    def test_optimized_discretisation_1d_example(self):

        self.x_train = torch.rand((3, 1))
        X = [self.x_train, self.x_train]
        Y = [torch.rand((3, 1)) , torch.rand((3, 1)) -1]
        self.updated_model = self.loop.update_model(X, Y)
        self.constrained_posterior_model = ConstrainedPosteriorMean(self.updated_model, maximize=True,
                                                                    penalty_value=torch.Tensor([0]))

        bounds = torch.tensor([[0.0], [1.0]], device=device, dtype=dtype)
        argmax_mean, _ = optimize_acqf(
            acq_function=self.constrained_posterior_model,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        acqf = acquisition_function_factory(type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                               model=self.updated_model, objective=self.constrained_obj,
                                               best_value=None, idx=None, number_of_outputs=2,
                                               penalty_value=torch.tensor([0.0]), iteration=0,
                                               initial_condition_internal_optimizer=argmax_mean)

        num_test_locations = 5
        X_test = torch.rand((num_test_locations, 1, 1))
        discretisation, fantasy_model = acqf.compute_optimized_X_discretisation(X_test, False)
        plot_X_locations = torch.cat([torch.cat([torch.rand((5000, 1, 1))[:, None, None, :, :]] * 5, dim=2)] * 25, dim=1)
        constrained_posterior_mean_model = ConstrainedPosteriorMean(model=fantasy_model,
                                                                    penalty_value=acqf.penalty_value)
        posterior_model_value = constrained_posterior_mean_model(plot_X_locations)
        posterior_mean_objective = fantasy_model.posterior(plot_X_locations).mean[..., 0]
        posterior_mean_constraint = fantasy_model.posterior(plot_X_locations).mean[..., 1]

        for i in range(1):
            x_test = X_test[i, ...].squeeze()
            for fantasy_idx in range(25):
                fantasy_discretisation = discretisation[:, fantasy_idx, i, :].detach().numpy().squeeze()
                plot_x_locations = plot_X_locations[:, fantasy_idx, i, :].detach().numpy().squeeze()
                plot_x_values = posterior_model_value[:, fantasy_idx, i].detach().numpy().squeeze()
                best_plot_x_location = plot_x_locations[np.argmax(plot_x_values)]
                fantasy_posterior_mean_objective = posterior_mean_objective[:, fantasy_idx, i].detach().numpy().squeeze()
                fantasy_posterior_mean_constraint = posterior_mean_constraint[:, fantasy_idx, i].detach().numpy().squeeze()

                # Create the figure and subplots
                # fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)  # 1 row, 3 columns
                # axes[0].scatter(plot_x_locations[:, 0], plot_x_locations[:, 1], c=fantasy_posterior_mean_objective)
                # axes[0].set_title("fanatsy mean")
                # axes[1].scatter(plot_x_locations[:, 0], plot_x_locations[:, 1], c=fantasy_posterior_mean_constraint)
                # axes[1].set_title("fanatsy constraint")
                # axes[2].scatter(plot_x_locations[:, 0], plot_x_locations[:, 1], c=plot_x_values)
                # axes[2].set_title("fanatsy constraint")
                # plt.tight_layout()
                # plt.show()

                plt_scatter = plt.scatter(plot_x_locations, plot_x_values)
                plt.vlines(x_test, -5, 5 , color="red", label="test location")
                plt.vlines(best_plot_x_location, -5, 5 , color="black", label="best from plot")
                plt.vlines(fantasy_discretisation, -5, 5, color="blue", label="best fantasised")
                plt.title("test_location: " + str(i) + " virtual sample: " + str(fantasy_idx))
                plt.xlim((0,1))
                # plt.ylim((0,1))
                plt.legend()
                plt.colorbar(plt_scatter)
                plt.show()
        print("ok")


