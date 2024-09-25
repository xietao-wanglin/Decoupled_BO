import pickle
import time
from typing import Optional

import matplotlib.pyplot as plt
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ConstrainedMCObjective
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler, ListSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch.utils.transforms import normalize
from gpytorch import settings, ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from pip._internal.resolution.resolvelib import candidates

from bo.acquisition_functions.acquisition_functions import MathsysExpectedImprovement, \
    DecopledHybridConstrainedKnowledgeGradient
from bo.constrained_functions.synthetic_problems import testing_function, testing_function_dummy_constraint, \
    ConstrainedBranin
from bo.model.Model import ConstrainedPosteriorMean
from bo.samplers.samplers import quantileSampler, quantileSamplerCoupledSources
from bo.synthetic_test_functions.synthetic_test_functions import MOPTA08, MysteryFunction


class TestMathsysExpectedImprovement(BotorchTestCase):
    def test_forward_evaluation(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[-0.5]], device=self.device, dtype=dtype)
            variance = torch.ones(1, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            ei_expected = torch.tensor([0.1978], dtype=dtype)
            X = torch.empty(1, 1, device=self.device, dtype=dtype)  # dummy
            module = MathsysExpectedImprovement(model=mm, best_f=0.0)
            ei_actual = module(X)

            self.assertAllClose(ei_actual, ei_expected, atol=1e-4)
            self.assertEqual(ei_actual.shape, torch.Size([1]))

    def test_forward_shape(self):
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([[-0.5], [0.7]], device=self.device, dtype=dtype)
            variance = torch.ones(2, 1, device=self.device, dtype=dtype)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))

            X = torch.empty(2, 1, 1, device=self.device, dtype=dtype)  # dummy
            # module =? initialize your acquisition function
            # ei_actual = module(X)

            # self.assertTrue(ei_actual.shape == torch.Size([2]))


class TestMathsysMCExpectedImprovement(BotorchTestCase):
    def test_forward_shape(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            # the event shape is `b x q x t` = 2 x 1 x 1
            samples = torch.zeros(2, 1, 1, **tkwargs)
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 2 x 1. X is a dummy and unused b/c of mocking
            X = torch.zeros(2, 1, 1, **tkwargs)

            # basic test
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
            # acqf = ?
            # ei_actual = acqf(X)

            # self.assertTrue(ei_actual.shape == torch.Size([2]))


class NumericalTest(BotorchTestCase):
    def test_acquisition_functions_are_equivalent_single_run(self):
        d = 2
        torch.manual_seed(0)
        dtype = torch.double
        train_X = torch.rand(5, d, device=self.device, dtype=dtype)
        train_Y = torch.rand(5, 1, device=self.device, dtype=dtype)
        model = SingleTaskGP(train_X, train_Y)

        X = torch.rand(10, 1, d, device=self.device, dtype=dtype)
        ei = ExpectedImprovement(model=model, best_f=0)
        ei_val = ei(X)

        sampler = IIDNormalSampler(sample_shape=torch.Size([1000000]))
        mc_ei = qExpectedImprovement(model=model, best_f=0, sampler=sampler)
        mc_ei_val = mc_ei(X)

        self.assertAllClose(ei_val, mc_ei_val, atol=1e-3)


def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]


class TestDecoupledKG(BotorchTestCase):

    def test_all_sources(self):
        torch.manual_seed(0)
        dtype = torch.double
        func = testing_function()
        d = 1
        num_of_points = 5
        train_X = torch.rand(num_of_points, d, device=self.device, dtype=dtype)
        train_Y_objective = func.evaluate_true(train_X)
        train_Y_constraint = func.evaluate_slack_true(train_X)
        NOISE = torch.tensor(1e-9, device=self.device, dtype=dtype)
        model_objective = SingleTaskGP(train_X, train_Y_objective,
                                       train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)))
        model_constraint = SingleTaskGP(train_X, train_Y_constraint,
                                        train_Yvar=NOISE.expand_as(train_Y_constraint.reshape(-1, 1)))
        model = ModelListGP(*[model_objective, model_constraint])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        #
        model.posterior(torch.rand(5, 1, d), dtype=dtype)
        n_fantasised_samples = 10
        sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        sampler_constraint = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        sampler_list = ListSampler(*[sampler_objective, sampler_constraint])
        x_eval_mask = torch.ones(1, 2, dtype=torch.bool)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        print("started")
        start = time.time()
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        argmax_mean, max_mean = optimize_acqf(
            acq_function=ConstrainedPosteriorMean(model, maximize=True, penalty_value=torch.Tensor([0])),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )

        acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                          num_fantasies=n_fantasised_samples,
                                                          objective=objective, number_of_raw_points=100,
                                                          number_of_restarts=15, X_evaluation_mask=x_eval_mask,
                                                          seed=0, penalty_value=torch.Tensor([0]),
                                                          x_best_location=argmax_mean,
                                                          evaluate_all_sources=True)
        # x = X_test[:, None, None]
        # with torch.no_grad():
        #     kg_vals = acqf.forward(x)
        # #
        candidates, candidates_values = optimize_acqf(acqf, bounds, 1, 5, 60, options={'maxiter': 100})
        stop = time.time()
        print("total eval time " + str(stop - start))
        self.plot_source(acqf, argmax_mean, candidates, candidates_values, dtype, max_mean, model_constraint,
                         model_objective, train_X, train_Y_constraint, train_Y_objective)

    def test_all_sources_2d_problem(self):
        torch.manual_seed(0)
        dtype = torch.double
        func = ConstrainedBranin()
        d = 2
        num_of_points = 5
        train_X = torch.rand(num_of_points, d, device=self.device, dtype=dtype)
        train_Y_objective = func.evaluate_true(train_X).unsqueeze(dim=1)
        train_Y_constraint = func.evaluate_slack_true(train_X)
        NOISE = torch.tensor(1e-6, device=self.device, dtype=dtype)
        model_objective = SingleTaskGP(train_X, train_Y_objective,
                                       train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)))
        model_constraint = SingleTaskGP(train_X, train_Y_constraint,
                                        train_Yvar=NOISE.expand_as(train_Y_constraint.reshape(-1, 1)))
        model = ModelListGP(*[model_objective, model_constraint])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        #
        model.posterior(torch.rand(5, 1, d), dtype=dtype)
        n_fantasised_samples = 5
        sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        sampler_constraint = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        sampler_list = ListSampler(*[sampler_objective, sampler_constraint])
        x_eval_mask = torch.ones(1, 2, dtype=torch.bool)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        print("started")
        start = time.time()
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        argmax_mean, max_mean = optimize_acqf(
            acq_function=ConstrainedPosteriorMean(model, maximize=True, penalty_value=torch.Tensor([0])),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )

        acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                          num_fantasies=n_fantasised_samples,
                                                          objective=objective, number_of_raw_points=100,
                                                          number_of_restarts=15, X_evaluation_mask=x_eval_mask,
                                                          seed=0, penalty_value=torch.Tensor([0]),
                                                          x_best_location=argmax_mean,
                                                          evaluate_all_sources=True)

        acqf.forward(torch.rand((15, 1, 2), dtype=dtype))
        stop = time.time()
        print("total eval time " + str(stop - start))

    def test_single_source_2d_problem(self):
        torch.manual_seed(0)
        dtype = torch.double
        func = ConstrainedBranin()
        d = 2
        num_of_points = 5
        train_X = torch.rand(num_of_points, d, device=self.device, dtype=dtype)
        train_Y_objective = func.evaluate_true(train_X).unsqueeze(dim=1)
        train_Y_constraint = func.evaluate_slack_true(train_X)
        NOISE = torch.tensor(1e-6, device=self.device, dtype=dtype)
        model_objective = SingleTaskGP(train_X, train_Y_objective,
                                       train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)))
        model_constraint = SingleTaskGP(train_X, train_Y_constraint,
                                        train_Yvar=NOISE.expand_as(train_Y_constraint.reshape(-1, 1)))
        model = ModelListGP(*[model_objective, model_constraint])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        #
        model.posterior(torch.rand(5, 1, d), dtype=dtype)
        n_fantasised_samples = 5
        sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        sampler_constraint = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        sampler_list = ListSampler(*[sampler_objective, sampler_constraint])
        x_eval_mask = torch.ones(1, 2, dtype=torch.bool)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        print("started")
        start = time.time()
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        argmax_mean, max_mean = optimize_acqf(
            acq_function=ConstrainedPosteriorMean(model, maximize=True, penalty_value=torch.Tensor([0])),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        kg_values = torch.zeros(2, dtype=dtype)
        penalty_value = torch.tensor([2.0], dtype=dtype)
        for index in range(2):
            start = time.time()
            n_fantasised_samples = 5
            sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_constraint1 = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_constraint2 = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_list = ListSampler(*[sampler_objective, sampler_constraint1, sampler_constraint2])
            x_eval_mask = torch.zeros(1, 2, dtype=torch.bool)  # 2 outputs, 1 == True
            x_eval_mask[0, index] = 1
            acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                              num_fantasies=n_fantasised_samples,
                                                              source_index=index,
                                                              objective=objective, number_of_raw_points=100,
                                                              number_of_restarts=15, X_evaluation_mask=x_eval_mask,
                                                              seed=0, penalty_value=penalty_value,
                                                              x_best_location=argmax_mean,
                                                              evaluate_all_sources=False)
            # acqf(rd) # 5 is no of points, 1 is for q-batch, d is dimension of input space
            candidates, candidates_values = optimize_acqf(acqf, bounds, 1, 5, 60, options={"maxiter": 100})
            kg_values[index] = candidates_values
            stop = time.time()
            print("output: " + str(index) + " , kg: " + str(kg_values[index]), " ,time: " + str(stop - start))
        print("kgvals: " + str(kg_values))

    def test_sampling_decision_always_objective(self):
        dtype = torch.double
        func = testing_function()
        d = 1
        num_points_objective = 5
        num_points_constraint = 50
        expected_decision = 0  # Objective
        n_fantasised_samples = 20
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        sampler_constraint = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        for seed in range(50):
            torch.manual_seed(seed)
            train_X_objective = torch.rand(num_points_objective, d, device=self.device, dtype=dtype)
            train_X_constraint = torch.rand(num_points_constraint, d, device=self.device, dtype=dtype)
            train_Y_objective = func.evaluate_true(train_X_objective)
            train_Y_constraint = func.evaluate_slack_true(train_X_constraint)
            NOISE = torch.tensor(1e-9, device=self.device, dtype=dtype)
            model_objective = SingleTaskGP(train_X_objective, train_Y_objective,
                                           train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)))
            model_constraint = SingleTaskGP(train_X_constraint, train_Y_constraint,
                                            train_Yvar=NOISE.expand_as(train_Y_constraint.reshape(-1, 1)))

            model = ModelListGP(*[model_objective, model_constraint])
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.posterior(torch.rand(5, 1, d), dtype=dtype)
            sampler_list = ListSampler(*[sampler_objective, sampler_constraint])
            objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
            penalty_value = torch.Tensor([2])
            argmax_mean, max_mean = optimize_acqf(
                acq_function=ConstrainedPosteriorMean(model, maximize=True, penalty_value=penalty_value),
                bounds=bounds,
                q=1,
                num_restarts=15,
                raw_samples=100,
            )
            kg_values = torch.zeros(2, dtype=dtype)
            for index in range(2):
                x_eval_mask = torch.zeros(1, 2, dtype=torch.bool)  # 2 outputs, 1 == True
                x_eval_mask[0, index] = 1
                # index = 1
                # x_eval_mask[0, index] = 1
                torch.manual_seed(seed)
                acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                                  num_fantasies=n_fantasised_samples,
                                                                  source_index=index,
                                                                  objective=objective, number_of_raw_points=100,
                                                                  number_of_restarts=15, X_evaluation_mask=x_eval_mask,
                                                                  seed=seed, penalty_value=penalty_value,
                                                                  x_best_location=argmax_mean,
                                                                  evaluate_all_sources=False)
                # acqf(rd) # 5 is no of points, 1 is for q-batch, d is dimension of input space

                candidates, candidates_values = optimize_acqf(acqf, bounds, 1, 5, 60, options={'maxiter': 100})
                kg_values[index] = candidates_values
                print("output: " + str(index) + " , kg: " + str(kg_values[index]))
            print("seed: " + str(seed))
            print("kgvals: " + str(kg_values))

            self.assertEqual(expected_decision, torch.argmax(kg_values))

    def test_sampling_decision_always_constraint(self):
        dtype = torch.double
        func = testing_function()
        d = 1
        num_points_objective = 50
        num_points_constraint = 5
        expected_decision = 1  # Constraint
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        for seed in range(50):
            torch.manual_seed(seed)
            train_X_objective = torch.rand(num_points_objective, d, device=self.device, dtype=dtype)
            train_X_constraint = torch.rand(num_points_constraint, d, device=self.device, dtype=dtype)
            train_Y_objective = func.evaluate_true(train_X_objective)
            train_Y_constraint = func.evaluate_slack_true(train_X_constraint)
            NOISE = torch.tensor(1e-9, device=self.device, dtype=dtype)
            model_objective = SingleTaskGP(train_X_objective, train_Y_objective,
                                           train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)))
            model_constraint = SingleTaskGP(train_X_constraint, train_Y_constraint,
                                            train_Yvar=NOISE.expand_as(train_Y_constraint.reshape(-1, 1)))

            model = ModelListGP(*[model_objective, model_constraint])
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.posterior(torch.rand(5, 1, d), dtype=dtype)
            objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
            penalty_value = torch.Tensor([2])
            argmax_mean, max_mean = optimize_acqf(
                acq_function=ConstrainedPosteriorMean(model, maximize=True, penalty_value=penalty_value),
                bounds=bounds,
                q=1,
                num_restarts=15,
                raw_samples=100,
            )
            kg_values = torch.zeros(2, dtype=dtype)
            for index in range(2):
                n_fantasised_samples = 5
                if index != 0:
                    n_fantasised_samples = 10
                sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
                sampler_constraint = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
                sampler_list = ListSampler(*[sampler_objective, sampler_constraint])
                x_eval_mask = torch.zeros(1, 2, dtype=torch.bool)  # 2 outputs, 1 == True
                x_eval_mask[0, index] = 1
                torch.manual_seed(seed)
                acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                                  num_fantasies=n_fantasised_samples,
                                                                  source_index=index,
                                                                  objective=objective, number_of_raw_points=100,
                                                                  number_of_restarts=15, X_evaluation_mask=x_eval_mask,
                                                                  seed=seed, penalty_value=penalty_value,
                                                                  x_best_location=argmax_mean,
                                                                  evaluate_all_sources=False)
                # acqf(rd) # 5 is no of points, 1 is for q-batch, d is dimension of input space

                candidates, candidates_values = optimize_acqf(acqf, bounds, 1, 5, 60, options={'maxiter': 100})
                kg_values[index] = candidates_values
                print("output: " + str(index) + " , kg: " + str(kg_values[index]))
            print("seed: " + str(seed))
            print("kgvals: " + str(kg_values))

            self.assertEqual(expected_decision, torch.argmax(kg_values))

    def test_sampling_decision_always_important_constraints(self):
        dtype = torch.double
        torch.set_default_dtype(dtype)
        func = testing_function_dummy_constraint()
        d = 1
        num_points_objective = 60
        num_points_constraint1 = 60
        num_points_constraint2 = 60
        expected_decision_to_avoid = 2  # Constraint
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=dtype)
        for seed in range(10):
            torch.manual_seed(seed)
            train_X_objective = torch.rand(num_points_objective, d, device=self.device, dtype=dtype)
            train_X_constraint1 = torch.rand(num_points_constraint1, d, device=self.device, dtype=dtype)
            train_X_constraint2 = torch.rand(num_points_constraint2, d, device=self.device, dtype=dtype)
            train_Y_objective = func.evaluate_true(train_X_objective)
            train_Y_constraint1 = func.evaluate_slack_true1(train_X_constraint1)
            train_Y_constraint2 = func.evaluate_slack_true2(train_X_constraint2)
            NOISE = torch.tensor(1e-5, device=self.device, dtype=dtype)
            model_objective = SingleTaskGP(train_X_objective, train_Y_objective,
                                           train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)))
            model_constraint1 = SingleTaskGP(train_X_constraint1, train_Y_constraint1,
                                             train_Yvar=NOISE.expand_as(train_Y_constraint1.reshape(-1, 1)))
            model_constraint2 = SingleTaskGP(train_X_constraint2, train_Y_constraint2,
                                             train_Yvar=NOISE.expand_as(train_Y_constraint2.reshape(-1, 1)))
            model = ModelListGP(*[model_objective, model_constraint1, model_constraint2])
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.posterior(torch.rand(5, 1, d), dtype=dtype)
            objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
            penalty_value = torch.tensor([2.0], dtype=dtype)
            argmax_mean, max_mean = optimize_acqf(
                acq_function=ConstrainedPosteriorMean(model, maximize=True, penalty_value=penalty_value),
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=248,
            )
            kg_values = torch.zeros(3, dtype=dtype)
            for index in range(3):
                start = time.time()
                n_fantasised_samples = 5
                sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
                sampler_constraint1 = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
                sampler_constraint2 = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
                sampler_list = ListSampler(*[sampler_objective, sampler_constraint1, sampler_constraint2])
                x_eval_mask = torch.zeros(1, 3, dtype=torch.bool)  # 2 outputs, 1 == True
                x_eval_mask[0, index] = 1
                torch.manual_seed(seed)
                acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                                  num_fantasies=n_fantasised_samples,
                                                                  source_index=index,
                                                                  objective=objective, number_of_raw_points=100,
                                                                  number_of_restarts=15, X_evaluation_mask=x_eval_mask,
                                                                  seed=seed, penalty_value=penalty_value,
                                                                  x_best_location=argmax_mean,
                                                                  evaluate_all_sources=False)
                # acqf(rd) # 5 is no of points, 1 is for q-batch, d is dimension of input space
                candidates, candidates_values = optimize_acqf(acqf, bounds, 1, 5, 60, options={"maxiter": 100})
                kg_values[index] = candidates_values
                stop = time.time()
                print("output: " + str(index) + " , kg: " + str(kg_values[index]), " ,time: " + str(stop - start))
            print("seed: " + str(seed))
            print("kgvals: " + str(kg_values))
            self.assertNotEquals(expected_decision_to_avoid, torch.argmax(kg_values))

    def test_shapes(self):
        torch.manual_seed(0)
        dtype = torch.double
        torch.set_default_dtype(dtype)
        func = ConstrainedBranin()
        d = 2
        num_of_points = 5
        train_X = torch.rand(num_of_points, d, device=self.device, dtype=dtype)
        train_Y_objective = func.evaluate_true(train_X).unsqueeze(dim=1)
        train_Y_constraint = func.evaluate_slack_true(train_X)
        NOISE = torch.tensor(1e-6, device=self.device, dtype=dtype)
        model_objective = SingleTaskGP(train_X, train_Y_objective,
                                       train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)))
        model_constraint = SingleTaskGP(train_X, train_Y_constraint,
                                        train_Yvar=NOISE.expand_as(train_Y_constraint.reshape(-1, 1)))
        model = ModelListGP(*[model_objective, model_constraint])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        #
        model.posterior(torch.rand(5, 1, d), dtype=dtype)
        n_fantasised_samples = 5
        sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        sampler_constraint = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        sampler_list = ListSampler(*[sampler_objective, sampler_constraint])
        x_eval_mask = torch.ones(1, 2, dtype=torch.bool)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        print("started")
        start = time.time()
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        argmax_mean, max_mean = optimize_acqf(
            acq_function=ConstrainedPosteriorMean(model, maximize=True, penalty_value=torch.Tensor([0])),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        kg_values = torch.zeros(2, dtype=dtype)
        penalty_value = torch.tensor([2.0], dtype=dtype)
        for index in range(2):
            start = time.time()
            n_fantasised_samples = 5
            sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_constraint1 = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_constraint2 = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_list = ListSampler(*[sampler_objective, sampler_constraint1, sampler_constraint2])
            x_eval_mask = torch.zeros(1, 2, dtype=torch.bool)  # 2 outputs, 1 == True
            x_eval_mask[0, index] = 1
            acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                              num_fantasies=n_fantasised_samples,
                                                              source_index=index,
                                                              objective=objective, number_of_raw_points=100,
                                                              number_of_restarts=1, X_evaluation_mask=x_eval_mask,
                                                              seed=0, penalty_value=penalty_value,
                                                              x_best_location=argmax_mean,
                                                              evaluate_all_sources=True)
            kgs = acqf.forward(torch.ones(1, 1, 2))
            print("OK")

    def test_consistency(self):
        torch.manual_seed(0)
        dtype = torch.double
        torch.set_default_dtype(dtype)
        func = MysteryFunction(noise_std=1e-06, negate=True)
        settings.min_fixed_noise._global_double_value = 1e-06
        d = 2
        num_of_points = 22
        with open('/home/jungredda/CRYPT/GITHUB_REPOS/xietaorepo/Mathsys_RG_2024/results/mystery_lots_final_ckg0.pkl', 'rb') as file:
            data = pickle.load(file)
        train_X = data["input_data"][-1]
        # train_X = torch.rand((num_of_points, d))
        train_Y_objective = func(train_X).unsqueeze(dim=1)
        train_Y_constraint = func.evaluate_slack_true(train_X).unsqueeze(dim=1)
        NOISE = torch.tensor(1e-6, device=self.device, dtype=dtype)
        covar_modules = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=1,
                lengthscale_prior=GammaPrior(1.0, 3.0),
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        model_objective = SingleTaskGP(train_X, train_Y_objective,
                                       train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)),
                                       outcome_transform=Standardize(m=1),
                                       covar_module=covar_modules)
        model_constraint = SingleTaskGP(train_X, train_Y_constraint,
                                        train_Yvar=NOISE.expand_as(train_Y_constraint.reshape(-1, 1)),
                                        covar_module=None)
        model = ModelListGP(*[model_objective, model_constraint])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        #
        model.posterior(torch.rand(5, 1, d), dtype=dtype)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        print("started")
        start = time.time()
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        penalty_value = torch.tensor([30.0], dtype=dtype)
        mean = ConstrainedPosteriorMean(model, maximize=True, penalty_value=penalty_value)
        argmax_mean, max_mean = optimize_acqf(
            acq_function=mean,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        n_fantasised_samples = 5
        sampler_list = ListSampler(
            *[quantileSamplerCoupledSources(sample_shape=torch.Size([n_fantasised_samples])),
              quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))])
        x_eval_mask = torch.ones(1, 2, dtype=torch.bool)  # 2 outputs, 1 == True
        acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                          num_fantasies=n_fantasised_samples,
                                                          source_index=0,
                                                          objective=objective, number_of_raw_points=1024,
                                                          number_of_restarts=5, X_evaluation_mask=x_eval_mask,
                                                          seed=0, penalty_value=penalty_value,
                                                          x_best_location=argmax_mean,
                                                          evaluate_all_sources=True)

        x_optimal = torch.tensor([[0.5515455756386866, 0.4728209571786274]], dtype=dtype)
        # discretisation, fantasy_model = acqf.compute_optimized_X_discretisation(x_optimal)
        x_optimal_value = acqf(x_optimal)
        x_test = torch.rand((2000, 1, 2))
        posterior_mean = mean(x_test)

        print("amax ", x_optimal, " val opt", x_optimal_value)

        x_posterior_mean_value = acqf(argmax_mean)
        print("amax ", argmax_mean, " val posterior mean", x_posterior_mean_value)

        candidates, kgvalue = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            sequential=False,
            num_restarts=15,  # can make smaller if too slow, not too small though
            raw_samples=70,  # used for intialization heuristic
            options={"maxiter": 100}
        )
        # candidates = torch.tensor([[[0.5611, 0.0000]]])
        discretisation, fantasy_model = acqf.compute_optimized_X_discretisation(candidates)
        kgvalue = acqf.compute_coupled_kg(candidates[None, :], fantasy_model, discretisation)
        print("candidates ", candidates, "kgval", kgvalue)
        plt.scatter(x_test[:, 0, 0], x_test[:, 0, 1], c=posterior_mean.detach())
        plt.scatter(argmax_mean[:, 0], argmax_mean[:, 1], color="red")
        # plt.scatter(train_X[:, 0], train_X[:, 1], color="magenta", marker="x")
        plt.scatter(discretisation.reshape(-1, 2)[:, 0], discretisation.reshape(-1, 2)[:, 1], color="magenta",
                    marker="x")
        plt.scatter(train_X[:, 0], train_X[:, 1], c='red')
        plt.scatter(candidates.squeeze()[0], candidates.squeeze()[1], c="red", marker="x")
        plt.colorbar()
        plt.show()

        raise

    def test_check_kg_gradients(self):
        torch.manual_seed(0)
        dtype = torch.double
        torch.set_default_dtype(dtype)
        func = MysteryFunction(noise_std=1e-06, negate=True)
        settings.min_fixed_noise._global_double_value = 1e-06
        d = 2
        num_of_points = 22
        # with open('/home/jungredda/CRYPT/GITHUB_REPOS/xietaorepo/Mathsys_RG_2024/results/mystery_lots_final_ckg0.pkl', 'rb') as file:
        #     data = pickle.load(file)
        # train_X = data["input_data"][-1]
        train_X = torch.rand((num_of_points, d))
        train_Y_objective = func(train_X).unsqueeze(dim=1)
        train_Y_constraint = func.evaluate_slack_true(train_X).unsqueeze(dim=1)
        NOISE = torch.tensor(1e-6, device=self.device, dtype=dtype)
        covar_modules = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=1,
                lengthscale_prior=GammaPrior(1.0, 3.0),
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        model_objective = SingleTaskGP(train_X, train_Y_objective,
                                       train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)),
                                       outcome_transform=Standardize(m=1),
                                       covar_module=covar_modules)
        model_constraint = SingleTaskGP(train_X, train_Y_constraint,
                                        train_Yvar=NOISE.expand_as(train_Y_constraint.reshape(-1, 1)),
                                        covar_module=None)
        model = ModelListGP(*[model_objective, model_constraint])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        #
        model.posterior(torch.rand(5, 1, d), dtype=dtype)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        print("started")
        start = time.time()
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        penalty_value = torch.tensor([30.0], dtype=dtype)
        mean = ConstrainedPosteriorMean(model, maximize=True, penalty_value=penalty_value)
        argmax_mean, max_mean = optimize_acqf(
            acq_function=mean,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        n_fantasised_samples = 5
        sampler_list = ListSampler(
            *[quantileSamplerCoupledSources(sample_shape=torch.Size([n_fantasised_samples])),
              quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))])
        x_eval_mask = torch.ones(1, 2, dtype=torch.bool)  # 2 outputs, 1 == True
        acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                          num_fantasies=n_fantasised_samples,
                                                          source_index=0,
                                                          objective=objective, number_of_raw_points=1024,
                                                          number_of_restarts=5, X_evaluation_mask=x_eval_mask,
                                                          seed=0, penalty_value=penalty_value,
                                                          x_best_location=argmax_mean,
                                                          evaluate_all_sources=True)

        x_optimal = torch.tensor([[0.5515455756386866, 0.4728209571786274]], dtype=dtype)
        torch.manual_seed(14)
        x_test = torch.rand((5,1,2), dtype=dtype, requires_grad=True)
        discretisation, fantasy_model = acqf.compute_optimized_X_discretisation(x_test)
        def check_gradient_kg_posteriormean(X):
            constrained_posterior_mean_model = ConstrainedPosteriorMean(acqf.model, penalty_value=torch.tensor([0]))
            return constrained_posterior_mean_model(X).sum() #WORKS!

        def check_gradient_kg_discrete(X):
            _, fantasy_model = acqf.compute_optimized_X_discretisation(X)
            kg = acqf.compute_coupled_kg(X, fantasy_model, discretisation.detach())
            return kg.sum() # WORKS for fixed discretisation!


        # Perform the gradient check with retain_graph=True
        grad_test = torch.autograd.gradcheck(check_gradient_kg_discrete, x_test, eps=1e-6, atol=1e-4)
        print(f"Gradients are working: {grad_test}")

    def test_plot_posterior_mean(self):
        torch.manual_seed(0)
        dtype = torch.double
        torch.set_default_dtype(dtype)
        func = MysteryFunction(noise_std=1e-06, negate=True)
        d = 2
        num_of_points = 30
        train_X = torch.rand(num_of_points, d, device=self.device, dtype=dtype)
        train_Y_objective = func(train_X).unsqueeze(dim=1)
        train_Y_constraint = func.evaluate_slack_true(train_X).unsqueeze(dim=1)
        NOISE = torch.tensor(1e-6, device=self.device, dtype=dtype)
        covar_modules = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=1,
                lengthscale_prior=GammaPrior(1.0, 5.0),
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        model_objective = SingleTaskGP(train_X, train_Y_objective,
                                       train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)),
                                       outcome_transform=Standardize(m=1),
                                       covar_module=covar_modules)
        model_constraint = SingleTaskGP(train_X, train_Y_constraint,
                                        train_Yvar=NOISE.expand_as(train_Y_constraint.reshape(-1, 1)),
                                        covar_module=covar_modules)
        model = ModelListGP(*[model_objective, model_constraint])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        #
        model.posterior(torch.rand(5, 1, d), dtype=dtype)
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        print("started")
        start = time.time()
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        penalty_value = torch.tensor([30.0], dtype=dtype)
        posterior_mean = ConstrainedPosteriorMean(model, maximize=True, penalty_value=penalty_value)
        argmax_mean, max_mean = optimize_acqf(
            acq_function=posterior_mean,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )

        x_test = torch.rand((2000, 1, 2))
        posterior_mean = posterior_mean(x_test)
        plt.scatter(x_test[:, 0, 0], x_test[:, 0, 1], c=posterior_mean.detach())
        plt.colorbar()
        plt.show()

    def test_check_simple_grad(self):
        # Input tensor with requires_grad=True
        x = torch.tensor([3.0], dtype=torch.double, requires_grad=True)

        # Perform multiple operations
        y = x ** 2  # Intermediate tensor
        y.retain_grad()  # Retain the gradient for intermediate tensor

        z = torch.sin(y)  # Another intermediate tensor
        z.retain_grad()  # Retain the gradient for z

        w = z * 3
        output = w.sum()

        # Perform backward pass
        output.backward()

        # Inspect gradients of the original input and intermediate tensors
        print(f"Gradient of x: {x.grad}")  # Gradient w.r.t. x
        print(f"Gradient of y: {y.grad}")  # Gradient w.r.t. y
        print(f"Gradient of z: {z.grad}")  # Gradient w.r.t. z

def test_MOPTA08_function_optimal_values(self):
    expected_best_fval = 222.427088
    best_recommended_point = torch.tensor([257, 0, 0, 123, 10, 59, 0, 400, 0, 757, 0, 0, 0, 0, 0, 491, 0, 0, 0, 0,
                                           0, 21, 0, 0, 0, 47, 339, 0, 0, 689, 0, 310, 0, 192, 1000, 227, 856, 622,
                                           589, 278,
                                           476, 710, 116, 0, 0, 0, 800, 406, 281, 32, 0, 0, 159, 0, 0, 0, 290, 0,
                                           423, 266,
                                           72, 416, 937, 285, 0, 733, 663, 355, 0, 110, 217, 0, 0, 181, 23, 299, 0,
                                           331, 767, 0,
                                           814, 848, 393, 838, 0, 837, 263, 341, 422, 68, 452, 674, 32, 0, 0, 0,
                                           136, 0, 653, 846,
                                           0, 1000, 786, 15, 76, 465, 0, 164, 235, 149, 614, 1000, 1000, 718, 463,
                                           199, 387, 115, 1000, 735,
                                           0, 897, 0, 1000]) / 1000
    bounds = torch.tensor([0.0, 1.0] * 124)
    normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
    function = MOPTA08()

    actual_best_fval = function.evaluate_task(normalized_best_recommended_point, 0)
    actual_constraint_value = function.evaluate_task(normalized_best_recommended_point, 1)
    is_location_feasible = actual_constraint_value <= 0
    print(actual_constraint_value)
    self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, atol=1e-4)
    self.assertEqual(True, is_location_feasible)


class TestHybriddKG(BotorchTestCase):

    def test_constraints(self):
        # Prepare tests.
        # KG should be positive in places where there has not been a sample.
        # KG should not be negative. give best value so far.
        # if everything is clearly unfeasible, the KG value is zero.
        # If one of the sources is sampled A LOT the preference should be in a different source.
        # IF all sources are sampled equally and A LOT all x's should be closer to the optimum...not sure which source
        # should sample since all are active actually.
        # test single run and compare against expected improvement or paper.
        dtype = torch.double
        d = 1
        num_points_objective = 5
        num_points_constraint = 50
        expected_decision = 0  # Objective

        torch.manual_seed(0)
        train_X_objective = torch.rand(num_points_objective, d, device=self.device, dtype=dtype)
        train_X_constraint = torch.rand(num_points_constraint, d, device=self.device, dtype=dtype)
        func = testing_function()
        train_Y_objective = func.evaluate_true(train_X_objective)
        train_Y_constraint = func.evaluate_slack_true(train_X_constraint)
        NOISE = torch.tensor(1e-9, device=self.device, dtype=dtype)
        model_objective = SingleTaskGP(train_X_objective, train_Y_objective,
                                       train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)))
        model_constraint = SingleTaskGP(train_X_constraint, train_Y_constraint,
                                        train_Yvar=NOISE.expand_as(train_Y_constraint.reshape(-1, 1)))

        model = ModelListGP(*[model_objective, model_constraint])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        model.posterior(torch.rand(5, 1, d), dtype=dtype)

        sampler = quantileSampler(sample_shape=torch.Size([5]))
        sampler_list = ListSampler(*[sampler, sampler])

        kg_values = torch.zeros(2, dtype=dtype)
        for index in range(3):
            x_eval_mask = torch.zeros(1, 2, dtype=torch.bool)  # 2 outputs, 1 == True
            x_eval_mask[0, index] = 1

            torch.manual_seed(0)
            acqf = DecopledHybridConstrainedKnowledgeGradient(model=model,
                                                              sampler=sampler_list,
                                                              num_fantasies=5,
                                                              objective=ConstrainedMCObjective(objective=obj_callable,
                                                                                               constraints=[
                                                                                                   obj_callable]),
                                                              evaluate_all_sources=False,
                                                              source_index=index,
                                                              X_evaluation_mask=x_eval_mask,
                                                              penalty_value=torch.Tensor([0.0]),
                                                              x_best_location=torch.zeros((1, d)))
            # acqf(rd) # 5 is no of points, 1 is for q-batch, d is dimension of input space

            bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
            candidates, candidates_values = optimize_acqf(acqf, bounds, 1, 5, 17, options={'maxiter': 200})
            kg_values[index] = candidates_values
            print(kg_values)
            # print(candidates.shape, candidates_values.shape)

        self.assertEqual(expected_decision, torch.argmax(kg_values))
