import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, DecoupledAcquisitionFunction
from botorch.models import SingleTaskGP, ModelListGP
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler, ListSampler
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.acquisition import ConstrainedMCObjective
from botorch.optim import optimize_acqf
from typing import Optional

from bo.acquisition_functions.acquisition_functions import MathsysExpectedImprovement, \
    DecoupledConstrainedKnowledgeGradient, DecopledHybridConstrainedKnowledgeGradient
from bo.constrained_functions.synthetic_problems import testing_function
from bo.synthetic_test_functions.synthetic_test_functions import MOPTA08
from botorch.utils.transforms import normalize
from bo.samplers.samplers import quantileSampler


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

    def test_constraints(self):
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
        for i in range(2):
            x_eval_mask = torch.zeros(1, 2, dtype=torch.bool)  # 2 outputs, 1 == True
            x_eval_mask[0, i] = 1

            torch.manual_seed(0)
            acqf = DecoupledConstrainedKnowledgeGradient(model, sampler=sampler_list, num_fantasies=5,
                                                         objective=ConstrainedMCObjective(objective=obj_callable,
                                                                                          constraints=[obj_callable]),
                                                         X_evaluation_mask=x_eval_mask, penalty_value=0.0)
            rd = torch.rand(6, 1, d, dtype=dtype)
            # acqf(rd) # 5 is no of points, 1 is for q-batch, d is dimension of input space

            bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.double)
            candidates, candidates_values = optimize_acqf(acqf, bounds, 1, 5, 15, options={'maxiter': 200})
            kg_values[i] = candidates_values
            print(kg_values)
            # print(candidates.shape, candidates_values.shape)

        self.assertEqual(expected_decision, torch.argmax(kg_values))

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
