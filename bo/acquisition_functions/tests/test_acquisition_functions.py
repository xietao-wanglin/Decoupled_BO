import pickle
import time
from typing import Optional

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ConstrainedMCObjective
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler, ListSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch.utils.transforms import normalize
from gpytorch import settings
from gpytorch.mlls import SumMarginalLogLikelihood
import matplotlib.pyplot as plt
from Launcher import constraint_callable_wrapper
from bo.acquisition_functions.acquisition_functions import MathsysExpectedImprovement, \
    DecopledHybridConstrainedKnowledgeGradient, filter_a_b, AcquisitionFunctionType, acquisition_function_factory
from bo.bo_loop import CoupledAndDecoupledOptimizationLoop, OptimizationLoop
from bo.constrained_functions.synthetic_problems import testing_function, testing_function_dummy_constraint, \
    ConstrainedBranin
from bo.model.Model import ConstrainedPosteriorMean, ConstrainedDeoupledGPModelWrapper, DecoupledConstraintPosteriorMean
from bo.result_utils.result_container import Results
from bo.samplers.samplers import quantileSampler, constantSampler
from bo.synthetic_test_functions.synthetic_test_functions import MOPTA08, MysteryFunctionSuperRedundant, \
    MysteryFunctionRedundant, ConstrainedBraninNew

device = torch.device("cpu")
dtype = torch.double
torch.set_default_dtype(dtype)


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

    def test_sampling_decision_always_objective(self):
        dtype = torch.double
        func = testing_function()
        d = 1
        num_points_objective = 5
        num_points_constraint = 50
        expected_decision = 0  # Objective
        n_fantasised_samples = 11
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        sampler_constraint = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
        for seed in range(3):
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
                torch.manual_seed(seed)
                acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                                  num_fantasies=n_fantasised_samples,
                                                                  source_index=index,
                                                                  objective=objective, number_of_raw_points=100,
                                                                  number_of_restarts=15, X_evaluation_mask=x_eval_mask,
                                                                  seed=seed, penalty_value=penalty_value,
                                                                  x_best_location=argmax_mean,
                                                                  evaluate_all_sources=False)
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
        for seed in range(3):
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
                n_fantasised_samples = 7
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

    def test_sampling_decision_dummy_constrained_never_queried(self):
        dtype = torch.double
        torch.set_default_dtype(dtype)
        settings.min_fixed_noise._global_double_value = 1e-09
        func = testing_function_dummy_constraint()
        d = 1
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=dtype)
        for num_data_points in [5, 20, 50, 80]:
            seed = 1
            num_points_objective = num_data_points
            num_points_constraint1 = num_data_points
            num_points_constraint2 = num_data_points
            torch.manual_seed(seed)
            train_X_objective = torch.rand(num_points_objective, d, device=self.device, dtype=dtype)
            train_X_constraint1 = torch.rand(num_points_constraint1, d, device=self.device, dtype=dtype)
            train_X_constraint2 = torch.rand(num_points_constraint2, d, device=self.device, dtype=dtype)
            train_Y_objective = func.evaluate_true(train_X_objective)
            train_Y_constraint1 = func.evaluate_slack_true1(train_X_constraint1)
            train_Y_constraint2 = func.evaluate_slack_true2(train_X_constraint2)
            NOISE = torch.tensor(1e-9, device=self.device, dtype=dtype)
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
                num_restarts=15,
                raw_samples=128,
            )
            n_outputs = 3
            kg_values = torch.zeros(n_outputs, dtype=dtype)
            for index in range(n_outputs):
                start = time.time()
                n_fantasised_samples = 7
                sampler_list = ListSampler(*[quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))] * 3)
                x_eval_mask = torch.zeros(1, n_outputs, dtype=torch.bool)  # 2 outputs, 1 == True
                x_eval_mask[0, index] = 1
                torch.manual_seed(seed)
                acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                                  num_fantasies=n_fantasised_samples,
                                                                  source_index=index,
                                                                  objective=objective, number_of_raw_points=200,
                                                                  number_of_restarts=20, X_evaluation_mask=x_eval_mask,
                                                                  seed=seed, penalty_value=penalty_value,
                                                                  x_best_location=argmax_mean,
                                                                  evaluate_all_sources=False)
                candidates, candidates_values = optimize_acqf(acqf, bounds, 1, 20, 60, options={"maxiter": 100})
                kg_values[index] = candidates_values
                stop = time.time()
                print("output: " + str(index) + " , kg: " + str(kg_values[index]), " ,time: " + str(stop - start))
            print("seed: " + str(seed))
            print("kgvals: " + str(kg_values))
            self.assertEqual(0.0, kg_values[2], "non important constaint should be zero")

    def test_sampling_decision_dummy_constrained_never_queried_from_data2(self):
        dtype = torch.double
        torch.set_default_dtype(dtype)
        settings.min_fixed_noise._global_double_value = 1e-06
        func = MysteryFunctionRedundant(noise_std=1e-6, negate=True)
        d = 2
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=dtype)
        seed = 1
        torch.manual_seed(seed)
        num_points_objective = 50
        train_X_objective = torch.rand(num_points_objective, d, device=self.device, dtype=dtype)
        train_Y_objective = func.evaluate_true(train_X_objective).unsqueeze(-1)
        train_Y_constraint1 = func.evaluate_task(train_X_objective, 1).unsqueeze(-1)
        train_Y_constraint2 = func.evaluate_task(train_X_objective, 2).unsqueeze(-1)
        NOISE = torch.tensor(1e-6, device=self.device, dtype=dtype)
        model_objective = SingleTaskGP(train_X_objective, train_Y_objective,
                                       train_Yvar=NOISE.expand_as(train_Y_objective.reshape(-1, 1)))
        model_constraint1 = SingleTaskGP(train_X_objective, train_Y_constraint1,
                                         train_Yvar=NOISE.expand_as(train_Y_constraint1.reshape(-1, 1)))
        model_constraint2 = SingleTaskGP(train_X_objective, train_Y_constraint2,
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
            num_restarts=15,
            raw_samples=128,
        )
        n_outputs = 3
        kg_values = torch.zeros(n_outputs, dtype=dtype)
        for index in range(n_outputs):
            index = 2
            start = time.time()
            n_fantasised_samples = 7
            sampler_list = ListSampler(*[quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))] * 3)
            x_eval_mask = torch.zeros(1, n_outputs, dtype=torch.bool)  # 2 outputs, 1 == True
            x_eval_mask[0, index] = 1
            torch.manual_seed(seed)
            acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                              num_fantasies=n_fantasised_samples,
                                                              source_index=index,
                                                              objective=objective, number_of_raw_points=200,
                                                              number_of_restarts=20, X_evaluation_mask=x_eval_mask,
                                                              seed=seed, penalty_value=penalty_value,
                                                              x_best_location=argmax_mean,
                                                              evaluate_all_sources=False)
            candidates, candidates_values = optimize_acqf(acqf, bounds, 1, 20, 60, options={"maxiter": 100})
            kg_values[index] = candidates_values
            stop = time.time()
            print("output: " + str(index) + " , kg: " + str(kg_values[index]), " ,time: " + str(stop - start))
        print("seed: " + str(seed))
        print("kgvals: " + str(kg_values))
        self.assertEqual(0.0, kg_values[2], "non important constaint should be zero")

    def test_ordering(self):
        a = torch.tensor([1, 0, 3, 9, 4, 7, 2])
        b = torch.tensor([2, 2, 2, 2, 3, 3, 1])

        # If tie in b values, ensure ai ≤ ai+1
        final_a, final_b = filter_a_b(a, b, 1e-9)

        self.assertAllClose(final_a, torch.tensor([2, 9, 7]))
        self.assertAllClose(final_b, torch.tensor([1, 2, 3]))

    def test_ordering_same_b(self):
        a = torch.tensor([1, 0, 3, 9, 4, 7, 2])
        b = torch.tensor([2, 2, 2, 2, 2, 2, 2])

        # If tie in b values, ensure ai ≤ ai+1
        final_a, final_b = filter_a_b(a, b, 1e-9)

        self.assertAllClose(final_a, torch.tensor([9]))
        self.assertAllClose(final_b, torch.tensor([2]))

    def test_fantasy_model_fantasises_correct_models(self):
        device = torch.device("cpu")
        dtype = torch.double
        torch.set_default_dtype(dtype)
        settings.min_fixed_noise._global_double_value = 1e-06
        black_box_function = MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True)
        num_constraints = 2
        budget = 240
        penalty = 40.0
        seed = int(0)
        print('\n Starting dcKG + cKG:')
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
        )
        results = Results(filename="dummy.pkl")
        tensor = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)
        torch_tensor = torch.tensor([penalty])
        initial_data = 6
        loop_dckg = CoupledAndDecoupledOptimizationLoop(black_box_func=black_box_function,
                                                        objective=constrained_obj,
                                                        ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                                        bounds=tensor,
                                                        performance_type="model",
                                                        model=model,
                                                        seed=seed,
                                                        budget=budget,
                                                        number_initial_designs=initial_data,
                                                        results=results,
                                                        penalty_value=torch_tensor)
        X, Y = loop_dckg.generate_initial_data(n=initial_data)
        model = loop_dckg.update_model(X, Y)
        test_X = torch.rand((1000, 1, 2))

        posterior1 = model.posterior(test_X)
        posterior_mean1 = posterior1.mean
        posterior_sqrt1 = posterior1.variance.squeeze().clamp_min(1e-12).sqrt()

        for i in torch.linspace(-5, 5, 5):
            print("constant: " + str(i))
            conditioning_x = torch.ones((1, 2)) * 0.5
            zero_sampler = constantSampler(sample_shape=torch.Size([1]), constant=i)
            sampler_list = ListSampler(*[zero_sampler, zero_sampler, zero_sampler])
            evaluation_mask = torch.ones((1, 3), dtype=torch.bool)
            evaluation_mask[:, 0] = False
            evaluation_mask[:, 1] = False
            evaluation_mask[:, 2] = True
            fantasized_model = model.fantasize(conditioning_x,
                                               sampler=sampler_list,
                                               evaluation_mask=evaluation_mask)

            constrained_posterior_model = ConstrainedPosteriorMean(model=model, maximize=True,
                                                                   penalty_value=torch.Tensor([0]))
            diff_model0 = torch.sum(
                torch.abs(constrained_posterior_model.evaluate_posterior(test_X)[0][..., 0].squeeze() - posterior_mean1[
                    ..., 0].squeeze()))
            diff_model1 = torch.sum(
                torch.abs(constrained_posterior_model.evaluate_posterior(test_X)[0][..., 1].squeeze() - posterior_mean1[
                    ..., 1].squeeze()))
            constrained_model_mean = constrained_posterior_model(test_X)
            fantasised_constrained_posterior_model = ConstrainedPosteriorMean(model=model,
                                                                              fantasised_models=fantasized_model,
                                                                              evaluation_mask=evaluation_mask,
                                                                              maximize=True,
                                                                              penalty_value=torch.Tensor([0]))

            diff_model0_var = torch.sum(torch.abs(
                fantasised_constrained_posterior_model.evaluate_posterior(test_X)[1][
                    ..., 0].squeeze() - posterior_sqrt1[:, 0].squeeze()))
            diff_model1_var = torch.sum(torch.abs(
                fantasised_constrained_posterior_model.evaluate_posterior(test_X)[1][
                    ..., 1].squeeze() - posterior_sqrt1[:, 1].squeeze()))

            constrained_fantasised_model = fantasised_constrained_posterior_model(test_X[:, None, :])
            diff_model3 = torch.sum(
                torch.abs(constrained_fantasised_model.squeeze() - constrained_model_mean.squeeze()))
            diff_index2 = torch.sum(
                torch.abs(fantasised_constrained_posterior_model._evaluate_feasibility_by_index(test_X, 2) -
                          constrained_posterior_model._evaluate_feasibility_by_index(test_X, 2)))
            self.assertEqual(0.0, diff_model0)
            self.assertEqual(0.0, diff_model1)
            self.assertEqual(0.0, diff_model0_var)
            self.assertEqual(0.0, diff_model1_var)
            self.assertEqual(0.0, diff_index2)
            self.assertEqual(0.0, diff_model3)

    def test_shapes_1d_single_source(self):
        torch.manual_seed(0)
        dtype = torch.double
        torch.set_default_dtype(dtype)
        d = 1
        num_of_points = 10
        train_X = torch.rand(num_of_points, d, device=self.device, dtype=dtype)
        train_Y_objective = torch.rand(num_of_points, 1, device=self.device, dtype=dtype)
        train_Y_constraint = torch.rand(num_of_points, 1, device=self.device, dtype=dtype)
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
        objective = ConstrainedMCObjective(objective=obj_callable, constraints=[obj_callable])
        bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
        argmax_mean, max_mean = optimize_acqf(
            acq_function=ConstrainedPosteriorMean(model, maximize=True, penalty_value=torch.Tensor([0])),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        penalty_value = torch.tensor([2.0], dtype=dtype)
        for n_designs in [1, 2, 10]:
            for index in [0, 1]:
                n_fantasised_samples = 7
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
                                                                  number_of_restarts=21, X_evaluation_mask=x_eval_mask,
                                                                  seed=0, penalty_value=penalty_value,
                                                                  x_best_location=argmax_mean,
                                                                  evaluate_all_sources=False)

                kgs = acqf.forward(torch.ones(n_designs, 1, 1))
                self.assertEqual(n_designs, len(kgs))

    def test_shapes_2d_single_source(self):
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
        for n_designs in [1, 2, 10]:
            for index in range(2):
                n_fantasised_samples = 7
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
                                                                  number_of_restarts=21, X_evaluation_mask=x_eval_mask,
                                                                  seed=0, penalty_value=penalty_value,
                                                                  x_best_location=argmax_mean,
                                                                  evaluate_all_sources=False)

                kgs = acqf.forward(torch.ones(n_designs, 1, 2))
                self.assertEqual(n_designs, len(kgs))

    def test_shapes_1d_all_source(self):
        torch.manual_seed(0)
        dtype = torch.double
        torch.set_default_dtype(dtype)
        func = ConstrainedBranin()
        d = 1
        num_of_points = 5
        train_X = torch.rand(num_of_points, d, device=self.device, dtype=dtype)
        train_Y_objective = torch.rand(num_of_points, 1, device=self.device, dtype=dtype)
        train_Y_constraint = torch.rand(num_of_points, 1, device=self.device, dtype=dtype)
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
        penalty_value = torch.tensor([2.0], dtype=dtype)
        for n_designs in [1, 2, 10]:
            n_fantasised_samples = 7
            sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_constraint1 = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_constraint2 = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_list = ListSampler(*[sampler_objective, sampler_constraint1, sampler_constraint2])
            x_eval_mask = torch.ones(1, 2, dtype=torch.bool)  # 2 outputs, 1 == True
            acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                              num_fantasies=n_fantasised_samples,
                                                              objective=objective, number_of_raw_points=100,
                                                              number_of_restarts=21, X_evaluation_mask=x_eval_mask,
                                                              seed=0, penalty_value=penalty_value,
                                                              x_best_location=argmax_mean,
                                                              evaluate_all_sources=True)

            kgs = acqf.forward(torch.ones(n_designs, 1, d))
            self.assertEqual(n_designs, len(kgs))

    def test_shapes_2d_all_source(self):
        torch.manual_seed(0)
        dtype = torch.double
        torch.set_default_dtype(dtype)
        d = 2
        num_of_points = 5
        train_X = torch.rand(num_of_points, d, device=self.device, dtype=dtype)
        train_Y_objective = torch.rand(num_of_points, 1, device=self.device, dtype=dtype)
        train_Y_constraint = torch.rand(num_of_points, 1, device=self.device, dtype=dtype)
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
        penalty_value = torch.tensor([2.0], dtype=dtype)
        for n_designs in [1, 2, 10]:
            n_fantasised_samples = 7
            sampler_objective = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_constraint1 = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_constraint2 = quantileSampler(sample_shape=torch.Size([n_fantasised_samples]))
            sampler_list = ListSampler(*[sampler_objective, sampler_constraint1, sampler_constraint2])
            x_eval_mask = torch.ones(1, 2, dtype=torch.bool)  # 2 outputs, 1 == True
            acqf = DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                              num_fantasies=n_fantasised_samples,
                                                              objective=objective, number_of_raw_points=100,
                                                              number_of_restarts=21, X_evaluation_mask=x_eval_mask,
                                                              seed=0, penalty_value=penalty_value,
                                                              x_best_location=argmax_mean,
                                                              evaluate_all_sources=True)

            kgs = acqf.forward(torch.ones(n_designs, 1, d))
            self.assertEqual(n_designs, len(kgs))

    def test_on_branin(self):
        torch.manual_seed(0)
        dtype = torch.double
        torch.set_default_dtype(dtype)

        black_box_function = ConstrainedBraninNew(noise_std=1e-6, negate=True)
        num_constraints = 1
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        # define a feasibility-weighted objective for optimization
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
        )
        results = Results(filename="remove_me.pkl")
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)
        loop = OptimizationLoop(black_box_func=black_box_function,
                                objective=constrained_obj,
                                ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                bounds= bounds,
                                performance_type="model",
                                model=model,
                                seed=0,
                                budget=50,
                                number_initial_designs=6,
                                results=results,
                                penalty_value=torch.tensor([2.0]))

        file_path = "/home/jungredda/CRYPT/GITHUB_REPOS/xietaorepo/Mathsys_RG_2024/results/constrained_branin_test_0.pkl"
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        X = [d for d in data["input_data"]]
        Y = [black_box_function.evaluate_task(d, task) for task, d in enumerate(data["input_data"])]
        # Y[0] = (Y[0] - 85)/170
        updated_model = loop.update_model(X, Y)

        x_locations_test = torch.rand((5000, 1, 2))
        # plot current model values
        constrained_posterior_mean = ConstrainedPosteriorMean(updated_model, maximize=True,
                                                              penalty_value=loop.penalty_value)
        mean = constrained_posterior_mean(x_locations_test)
        x_argmax = torch.argmax(mean)
        sn = plt.scatter(x_locations_test.squeeze()[:, 0], x_locations_test.squeeze()[:, 1], c=mean.detach())
        plt.scatter(X[0][:, 0],
                    X[0][:, 1], color="red")
        plt.scatter(x_locations_test.squeeze()[x_argmax, 0], x_locations_test.squeeze()[x_argmax, 1], color="red",
                    marker="x")
        plt.colorbar(sn)
        plt.show()
        # plot fantasised samples
        argmax_mean, _ = optimize_acqf(
            acq_function=ConstrainedPosteriorMean(updated_model, maximize=True, penalty_value=torch.Tensor([0])),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        x_star = torch.Tensor([[0.5522, 0.0186]])
        for z_constraint in [-5, -3, 0, 3, 5]:
            for z_objective in [-5, -3, 0, 3, 5]:
                quantile_sampler1 = constantSampler(sample_shape=torch.Size([3]), constant=z_objective)
                quantile_sampler2 = constantSampler(sample_shape=torch.Size([3]), constant=z_constraint)
                sampler_list = ListSampler(*[quantile_sampler1, quantile_sampler2])
                x_eval_mask = torch.ones(1, 2, dtype=torch.bool)
                constrained_obj = ConstrainedMCObjective(
                    objective=obj_callable,
                    constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
                )
                kg = DecopledHybridConstrainedKnowledgeGradient(updated_model,
                                                                sampler=sampler_list,
                                                                num_fantasies=3,
                                                                objective=constrained_obj,
                                                                number_of_raw_points=100,
                                                                evaluate_all_sources=True,
                                                                source_index=0,
                                                                number_of_restarts=17,
                                                                X_evaluation_mask=x_eval_mask,
                                                                seed=0,
                                                                penalty_value=loop.penalty_value,
                                                                x_best_location=argmax_mean)
                kg.use_scipy = False
                discretisation, fantasy_model = kg.compute_optimized_X_discretisation(
                    x_star.squeeze().unsqueeze(0).unsqueeze(0))
                reshaped_discretisation = discretisation[:, 0, 0, :].detach()
                constrained_posterior_mean = ConstrainedPosteriorMean(model=fantasy_model,
                                                                      penalty_value=loop.penalty_value)
                # constrained_posterior_mean = DecoupledConstraintPosteriorMean(model=fantasy_model,
                #                                                       penalty_value=loop.penalty_value)
                # fantasised_mean_model_value = constrained_posterior_mean._evaluate_objective(x_locations_test.unsqueeze(1))[:, 0]
                fantasised_mean_model_value = constrained_posterior_mean._evaluate_feasibility_by_index(x_locations_test.unsqueeze(1), 1)[:,0]



                x_argmax = torch.argmax(fantasised_mean_model_value)
                sn = plt.scatter(x_locations_test.squeeze()[:, 0],
                                 x_locations_test.squeeze()[:, 1],
                                 c=fantasised_mean_model_value.detach())
                plt.scatter(X[0][:, 0], X[0][:, 1], color="red")
                plt.scatter(reshaped_discretisation[:, 0], reshaped_discretisation[:, 1], color="magenta")
                plt.scatter(x_locations_test.squeeze()[x_argmax, 0],
                            x_locations_test.squeeze()[x_argmax, 1],
                            color="black",
                            marker="x")
                plt.scatter(x_star.squeeze()[0],
                            x_star.squeeze()[1],
                            color="red",
                            marker="s")
                plt.colorbar(sn)
                plt.title("Zo: " + str(z_objective) + " Zc: " + str(z_constraint))
                plt.show()

        print("ok")
    def test_on_branin_optimization(self):
        torch.manual_seed(0)
        dtype = torch.double
        torch.set_default_dtype(dtype)

        black_box_function = ConstrainedBraninNew(noise_std=1e-6, negate=True)
        num_constraints = 1
        model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
        # define a feasibility-weighted objective for optimization
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
        )
        results = Results(filename="remove_me.pkl")
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)
        loop = OptimizationLoop(black_box_func=black_box_function,
                                objective=constrained_obj,
                                ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                bounds= bounds,
                                performance_type="model",
                                model=model,
                                seed=0,
                                budget=50,
                                number_initial_designs=6,
                                results=results,
                                penalty_value=torch.tensor([2.0]))

        file_path = "/home/jungredda/CRYPT/GITHUB_REPOS/xietaorepo/Mathsys_RG_2024/results/constrained_branin_test_2_0.pkl"
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        X = [d for d in data["input_data"]]
        Y = [black_box_function.evaluate_task(d, task) for task, d in enumerate(data["input_data"])]
        updated_model = loop.update_model(X, Y)

        x_locations_test = torch.rand((5000, 1, 2)) * (torch.Tensor([0.65, 0.35]) - torch.Tensor([0.45, 0.0])) + torch.Tensor([0.45, 0.0])
        # plot current model values
        constrained_posterior_mean = ConstrainedPosteriorMean(updated_model, maximize=True,
                                                              penalty_value=loop.penalty_value)
        mean = constrained_posterior_mean(x_locations_test)
        x_argmax = torch.argmax(mean)
        x_eval_mask = torch.ones(1, 2, dtype=torch.bool)
        # plot fantasised samples
        argmax_mean, _ = optimize_acqf(
            acq_function=ConstrainedPosteriorMean(updated_model, maximize=True, penalty_value=torch.Tensor([2.0])),
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=248,
        )
        sn = plt.scatter(x_locations_test.squeeze()[:, 0], x_locations_test.squeeze()[:, 1], c=mean.detach())
        plt.scatter(X[0][:, 0],
                    X[0][:, 1], color="red")
        plt.scatter(x_locations_test.squeeze()[x_argmax, 0], x_locations_test.squeeze()[x_argmax, 1], color="red",
                    marker="x")
        plt.scatter(argmax_mean.squeeze()[0], argmax_mean.squeeze()[1], color="black")
        plt.colorbar(sn)
        plt.xlim(0.45, 0.65)
        plt.ylim(0, 0.35)
        plt.show()
        x_star = torch.Tensor([[0.5522, 0.0186]])
        kg = acquisition_function_factory(type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                          model=updated_model, objective=constrained_obj, best_value=None,
                                          idx=None, number_of_outputs=2, penalty_value=loop.penalty_value,
                                          iteration=0, initial_condition_internal_optimizer=argmax_mean)
        kg.use_scipy = True
        acqf = kg.forward(x_star)
        print("acqf: ", acqf)
        discretisation, fantasised_models = kg.compute_optimized_X_discretisation(x_star.squeeze().unsqueeze(0).unsqueeze(0))
        unconstrained_posterior_mean = ConstrainedPosteriorMean(model=fantasised_models,
                                                                penalty_value=loop.penalty_value)
        x_discretisation_posterior_mean = unconstrained_posterior_mean(discretisation)
        posterior_mean = unconstrained_posterior_mean._evaluate_feasibility_by_index(x_locations_test.unsqueeze(1), 1)
        for i in range(7):
            posterior_mean_idx = posterior_mean.detach()[:, i]
            best_location = self.best_discretisation_value(discretisation, i, x_discretisation_posterior_mean)
            sn = plt.scatter(x_locations_test.squeeze()[:, 0],
                        x_locations_test.squeeze()[:, 1],
                        c=posterior_mean_idx)
            plt.colorbar(sn)
            best_xs = discretisation[:, i, 0, :].detach().numpy()
            plt.scatter(best_xs[:, 0], best_xs[:, 1], color="magenta")
            plt.scatter(best_location[:, 0], best_location[:,1], c="black")
            plt.scatter(x_star.squeeze()[0], x_star.squeeze()[1], color="red")
            plt.xlim(0.45, 0.65)
            plt.ylim(0, 0.35)
            plt.show()
        print("ok")

    def best_discretisation_value(self, discretisation, i, x_discretisation_posterior_mean):
        argmax_idxs = torch.argmax(x_discretisation_posterior_mean, dim=0)
        best_location = discretisation[argmax_idxs[i], i].detach().numpy()
        return best_location


class Mopta(BotorchTestCase):
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