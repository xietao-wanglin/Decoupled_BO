from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ConstrainedMCObjective
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls import SumMarginalLogLikelihood
from torch.testing import assert_allclose

from Launcher import obj_callable, constraint_callable_wrapper
from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType, acquisition_function_factory
from bo.bo_loop import OptimizationLoop
from bo.constrained_functions.synthetic_problems import ConstrainedBranin
from bo.model.Model import ConstrainedGPModelWrapper, ConstrainedPosteriorMean, ConstrainedDeoupledGPModelWrapper, \
    BatchedConstrainedPosteriorMean
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import ConstrainedBraninNew, ConstrainedFunc3, MysteryFunction

device = torch.device("cpu")
dtype = torch.double

"""
TODO: Outdated test
class TestBranin(BotorchTestCase):

    def test_Branin(self):
        problem = ConstrainedBranin()

        def black_box_evaluation(X):
            y = problem(X).reshape(-1, 1)
            c1 = problem.evaluate_slack_true(X)
            return torch.concat([y, c1], dim=1)

        d = 2
        n_points = 40
        torch.manual_seed(0)
        dtype = torch.double
        train_X = torch.rand(n_points, d, device=self.device, dtype=dtype)
        train_X = torch.concat([train_X, torch.tensor([[0.0, 0.0]])])
        test_X = torch.rand(1000, d, device=self.device, dtype=dtype)
        evalu = black_box_evaluation(train_X)

        unfeas_x = train_X[evalu[:, 1] >= 0]
        plt.scatter(train_X[:, 0], train_X[:, 1], c=evalu[:, 0])
        plt.scatter(unfeas_x[:, 0], unfeas_x[:, 1], color="grey")
        plt.show()

        model = ConstrainedDeoupledGPModelWrapper(num_constraints=1)
        model.fit(train_X, evalu)
        optimized_model = model.optimize()

        posterior_distribution = optimized_model.posterior(test_X)
        mean, var = posterior_distribution.mean, posterior_distribution.variance

        unfeas_x = test_X[mean[:, 1] >= 0]

        # plt.scatter(black_box_evaluation(test_X)[:, 1], mean[:, 1].detach())
        # plt.show()
        # plt.scatter(black_box_evaluation(test_X)[:, 0], mean[:, 0].detach())
        # plt.show()
        plt.scatter(test_X[:, 0], test_X[:, 1], c=mean[:, 0].detach())
        plt.scatter(unfeas_x[:, 0], unfeas_x[:, 1], color="grey")
        plt.show()
"""


class TestMysteryFunction(BotorchTestCase):

    def test_shape(self):
        problem = MysteryFunction()

        d = 2
        n_points = 40000

        train_X = torch.rand(n_points, d, device=self.device, dtype=dtype)

        test_X = torch.rand(n_points, d, device=self.device, dtype=dtype)

        evalu = problem.evaluate_black_box(test_X)  # Objective
        plt.scatter(test_X[:, 0], test_X[:, 1], c=evalu[:, 0], alpha=0.2)
        evalu = problem.evaluate_black_box(train_X)  # Constraint
        unfeas_x = train_X[evalu[:, 1] >= 0]
        plt.scatter(unfeas_x[:, 0], unfeas_x[:, 1], color="grey")
        plt.colorbar()
        plt.show()


class TestPosteriorConstrainedMean(BotorchTestCase):

    def test_forward(self):
        problem = ConstrainedBranin()

        def black_box_evaluation(X):
            y = problem(X).reshape(-1, 1)
            c1 = problem.evaluate_slack_true(X)
            return torch.concat([y, c1], dim=1)

        d = 2
        n_points = 40
        torch.manual_seed(0)
        dtype = torch.double
        train_X = torch.rand(n_points, d, device=self.device, dtype=dtype)
        train_X = torch.concat([train_X, torch.tensor([[0.0, 0.0]])])
        test_X = torch.rand(1000, d, device=self.device, dtype=dtype)
        eval = black_box_evaluation(train_X)

        model = ConstrainedGPModelWrapper(num_constraints=1)
        model.fit(train_X, eval)
        optimized_model = model.optimize()

        constrained_posterior = ConstrainedPosteriorMean(model=optimized_model, maximize=True)
        penalised_posterior_values = constrained_posterior.forward(test_X[:, None, :])

        plt.scatter(test_X[:, 0], test_X[:, 1], c=penalised_posterior_values.detach())
        plt.show()


class TestBatchedConstrainedMean(BotorchTestCase):
    def test_forward(self):
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
        constrained_posterior_model = ConstrainedPosteriorMean(updated_model, maximize=True,
                                                               penalty_value=torch.Tensor([0]))

        batched_constrained_posterior_mean = BatchedConstrainedPosteriorMean(updated_model, maximize=True,
                                                                             penalty_value=torch.Tensor([0]),
                                                                             batch_size=5)

        random_values = torch.rand((248, 1, 2))
        posterior_model_values = constrained_posterior_model(random_values)
        batched_posterior_model_values = batched_constrained_posterior_mean(random_values)
        assert_allclose(posterior_model_values, batched_posterior_model_values)

    def test_forward_with_fantasies(self):
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

        x_train = torch.rand((100, 2))
        X = [x_train, x_train]
        Y = [black_box_function.evaluate_task(x, task) for task, x in enumerate(X)]
        updated_model = loop.update_model(X, Y)

        constrained_posterior_model = ConstrainedPosteriorMean(updated_model, maximize=True,
                                                               penalty_value=torch.Tensor([0]))

        argmax_mean, _ = optimize_acqf(
            acq_function=constrained_posterior_model,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )

        acqf = acquisition_function_factory(type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                            model=updated_model, objective=constrained_obj,
                                            best_value=None, idx=None, number_of_outputs=2,
                                            penalty_value=loop.penalty_value, iteration=0,
                                            initial_condition_internal_optimizer=argmax_mean)

        random_values = torch.rand((72, 1, 2))
        fantasize = updated_model.fantasize(X=random_values, sampler=acqf.sampler,
                                            evaluation_mask=acqf.construct_evaluation_mask(X))

        constrained_posterior_mean_with_fantasy = ConstrainedPosteriorMean(fantasize,
                                                                           maximize=True,
                                                                           penalty_value=torch.Tensor([0]))

        batched_constrained_posterior_mean_with_fantasy = BatchedConstrainedPosteriorMean(fantasize,
                                                                                          maximize=True,
                                                                                          penalty_value=torch.Tensor(
                                                                                              [0]),
                                                                                          batch_size=20)

        test_values = torch.rand((100, 25, 72, 1, 2))
        print("batched")
        batched_posterior_model_values = batched_constrained_posterior_mean_with_fantasy(test_values)
        print("standard")
        posterior_model_values = constrained_posterior_mean_with_fantasy(test_values)
        assert_allclose(posterior_model_values, batched_posterior_model_values)


class TestPosteriorConstrainedDecoupledMean(BotorchTestCase):

    def test_forward(self):
        problem = ConstrainedBranin()

        def black_box_evaluation(X):
            y = problem(X).reshape(-1, 1)
            c1 = problem.evaluate_slack_true(X)
            return torch.concat([y - 1000, c1], dim=1)

        d = 2
        n_points = 40
        torch.manual_seed(0)
        dtype = torch.double
        train_X = torch.rand(n_points, d, device=self.device, dtype=dtype)
        train_X = torch.concat([train_X, torch.tensor([[0.0, 0.0]])])
        test_X = torch.rand(1000, d, device=self.device, dtype=dtype)
        eval = black_box_evaluation(train_X)
        print("max", torch.max(eval[:, 0]), "min", torch.min(eval[:, 0]))
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=1)
        model.fit([train_X, train_X], [eval[:, 0], eval[:, 1]])
        optimized_model = model.optimize()

        constrained_posterior = ConstrainedPosteriorMean(model=optimized_model, maximize=True, penalty_value=00)
        penalised_posterior_values = constrained_posterior(test_X[:, None, :])

        plt.scatter(test_X[:, 0], test_X[:, 1], c=penalised_posterior_values.detach())
        plt.scatter(train_X[:, 0], train_X[:, 1], color="red")
        plt.show()


class TestConstrainedGPModelWrapper(TestCase):
    """
    TODO: outdated test
    """

    def test_fit(self):
        d = 1
        n_points_objective = 10
        n_points_constraints = 6
        torch.manual_seed(0)
        train_Xf = torch.rand(n_points_objective, d, device=device, dtype=dtype)
        train_Xc = torch.rand(n_points_constraints, d, device=device, dtype=dtype)
        problem = ConstrainedBranin()
        train_f_vals = problem.evaluate_true(train_Xf)
        train_c_vals = problem.evaluate_slack_true(train_Xc)

        train_var_noise = torch.tensor(1e-6, device=device, dtype=dtype)
        model_f = SingleTaskGP(train_X=train_Xf,
                               train_Y=train_f_vals.reshape(-1, 1),
                               train_Yvar=train_var_noise.expand_as(train_f_vals.reshape(-1, 1)),
                               outcome_transform=Standardize(m=1))
        model_c = SingleTaskGP(train_X=train_Xc,
                               train_Y=train_c_vals.reshape(-1, 1),
                               train_Yvar=train_var_noise.expand_as(train_c_vals.reshape(-1, 1)))

        model = ModelListGP(model_f, model_c)

        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # kg = DecoupledConstrainedKnowledgeGradient(model=model, num_fantasies=5, current_value=torch.Tensor([0.0]))


class TestConstrainedGPDecoupledModelWrapper(TestCase):
    def test_fit(self):
        d = 1
        n_points_objective = 10
        n_points_constraints = 6
        torch.manual_seed(0)
        train_Xf = torch.rand(n_points_objective, d, device=device, dtype=dtype)
        train_Xc = torch.rand(n_points_constraints, d, device=device, dtype=dtype)
        problem = ConstrainedBranin()
        train_f_vals = problem.evaluate_true(train_Xf)
        train_c_vals = problem.evaluate_slack_true(train_Xc)

        model = ConstrainedDeoupledGPModelWrapper(num_constraints=1)
        model.fit([train_Xf, train_Xc], [train_f_vals, train_c_vals])
        model.optimize()


class TestFunction3(BotorchTestCase):

    def test_shape(self):
        problem = ConstrainedFunc3()

        d = 2
        n_points = 4000

        train_X = torch.rand(n_points, d, device=self.device, dtype=dtype)

        test_X = torch.rand(n_points, d, device=self.device, dtype=dtype)

        evalu = problem.evaluate_black_box(test_X)  # f c1 c2 c3
        plt.scatter(test_X[:, 0], test_X[:, 1], c=evalu[:, 0], alpha=0.2)

        evalu = problem.evaluate_black_box(train_X)
        unfeas1_x = train_X[evalu[:, 1] > 0]
        plt.scatter(unfeas1_x[:, 0], unfeas1_x[:, 1], color="grey")

        unfeas2_x = train_X[evalu[:, 2] >= 0]
        plt.scatter(unfeas2_x[:, 0], unfeas2_x[:, 1], color="blue", alpha=0.2)

        # unfeas3_x = train_X[evalu[:, 3] >= 0]
        # plt.scatter(unfeas3_x[:, 0], unfeas3_x[:, 1], color="grey")

        plt.plot(0.2017, 0.8332)

        plt.colorbar()
        plt.show()


class TestBraninFunctionNew(BotorchTestCase):

    def test_shape(self):
        problem = ConstrainedBraninNew()
        d = 2
        n_points = 4000
        train_X = torch.rand(n_points, d, device=self.device, dtype=dtype)
        test_X = torch.rand(n_points, d, device=self.device, dtype=dtype)
        evalu = problem.evaluate_black_box(test_X)
        plt.scatter(test_X[:, 0], test_X[:, 1], c=evalu[:, 0], alpha=0.2)
        evalu = problem.evaluate_black_box(train_X)
        unfeas_x = train_X[evalu[:, 1] >= 0]
        plt.scatter(unfeas_x[:, 0], unfeas_x[:, 1], color="grey")
        plt.colorbar()
        plt.show()


class TestBraninFunctionNew(BotorchTestCase):

    def test_shape(self):
        problem = ConstrainedBraninNew()
        d = 2
        n_points = 4000
        train_X = torch.rand(n_points, d, device=self.device, dtype=dtype)
        test_X = torch.rand(n_points, d, device=self.device, dtype=dtype)
        evalu = problem.evaluate_black_box(test_X)
        plt.scatter(test_X[:, 0], test_X[:, 1], c=evalu[:, 0], alpha=0.2)
        evalu = problem.evaluate_black_box(train_X)
        unfeas_x = train_X[evalu[:, 1] >= 0]
        plt.scatter(unfeas_x[:, 0], unfeas_x[:, 1], color="grey")
        plt.colorbar()
        plt.show()
        bounds = [(-5.0, 10.0), (0.0, 15.0)]
        x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = torch.tensor(np.hstack((X1.reshape(100 * 100, 1), X2.reshape(100 * 100, 1))))
        Y = problem.forward(X)

        plt.figure()
        plt.contourf(X1, X2, Y.reshape((100, 100)), 100)
        if (len(self.min) > 1):
            plt.plot(np.array(self.min)[:, 0], np.array(self.min)[:, 1], 'w.', markersize=20, label=u'Observations')
        else:
            plt.plot(self.min[0][0], self.min[0][1], 'w.', markersize=20, label=u'Observations')
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
