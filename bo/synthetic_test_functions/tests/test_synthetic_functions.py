import torch
from botorch.utils.testing import BotorchTestCase
from botorch.utils.transforms import normalize

from bo.synthetic_test_functions.synthetic_test_functions import MysteryFunction, ConstrainedBraninNew, ConstrainedFunc3, PressureVessel, WeldedBeamSO


class TestDecoupledKG(BotorchTestCase):

    def test_mistery_function_optimal_values(self):
        expected_best_fval = 1.1743
        best_recommended_point = torch.tensor([2.7450, 2.3523])
        bounds = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
        normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
        function = MysteryFunction(negate=True)

        actual_best_fval = function.evaluate_task(normalized_best_recommended_point, 0)
        actual_constraint_value = function.evaluate_task(normalized_best_recommended_point, 1)
        is_location_feasible = actual_constraint_value <= 0

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, atol=1e-4)
        self.assertEqual(True, is_location_feasible)

    def test_new_branin_function_optimal_values(self):
        expected_best_fval = 268.781
        best_recommended_point = torch.tensor([3.273, 0.0489])
        bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])
        normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
        function = ConstrainedBraninNew(negate=True)

        actual_best_fval = function.evaluate_task(normalized_best_recommended_point, 0)
        actual_constraint_value = function.evaluate_task(normalized_best_recommended_point, 1)
        is_location_feasible = actual_constraint_value <= 0

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, atol=1e-2)
        self.assertEqual(True, is_location_feasible)

    def test_test_function_2_function_optimal_values(self):
        expected_best_fval = 0.7483
        best_recommended_point = torch.tensor([0.2018, 0.8332])
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
        function = ConstrainedFunc3(negate=True)

        actual_best_fval = function.evaluate_task(normalized_best_recommended_point, 0)
        actual_constraint1_value = function.evaluate_task(normalized_best_recommended_point, 1)
        actual_constraint2_value = function.evaluate_task(normalized_best_recommended_point, 2)
        actual_constraint3_value = function.evaluate_task(normalized_best_recommended_point, 3)
        is_location_feasible1 = actual_constraint1_value <= 0
        is_location_feasible2 = actual_constraint2_value <= 0
        is_location_feasible3 = actual_constraint3_value <= 0

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, atol=1e-3)
        self.assertEqual(True, is_location_feasible1)
        self.assertEqual(True, is_location_feasible2)
        self.assertEqual(True, is_location_feasible3)

    def test_pressure_vessel(self):
        expected_best_fval = 6059.946341 
        best_recommended_point = torch.tensor([0.812500 , 0.437500, 42.097398, 176.654047])
        bounds = torch.tensor([[0.0, 0.0, 10.0, 150.0], [10.0, 10.0, 50.0, 200.0]])
        normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
        function = PressureVessel(negate=True)

        actual_best_fval = -function.evaluate_task(normalized_best_recommended_point, 0)
        actual_constraint1_value = function.evaluate_task(normalized_best_recommended_point, 1)
        actual_constraint2_value = function.evaluate_task(normalized_best_recommended_point, 2)
        actual_constraint3_value = function.evaluate_task(normalized_best_recommended_point, 3)
        actual_constraint4_value = function.evaluate_task(normalized_best_recommended_point, 4)
        is_location_feasible1 = actual_constraint1_value <= 0
        is_location_feasible2 = actual_constraint2_value <= 0
        is_location_feasible3 = actual_constraint3_value <= 0
        is_location_feasible4 = actual_constraint4_value <= 0

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, rtol=1e-2)
        self.assertEqual(True, is_location_feasible1)
        self.assertEqual(True, is_location_feasible2)
        self.assertEqual(True, is_location_feasible3)
        self.assertEqual(True, is_location_feasible4)

    def test_welded_beam(self):
        expected_best_fval = 1.728226 
        best_recommended_point = torch.tensor([0.205986 , 3.471328, 9.020224 , 0.206480])
        bounds = torch.tensor([[0.125, 0.1, 0.1, 0.1], [10.0, 10.0, 10.0, 10.0]])
        normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
        function = WeldedBeamSO(negate=True)

        actual_best_fval = -function.evaluate_task(normalized_best_recommended_point, 0)
        actual_constraint1_value = function.evaluate_task(normalized_best_recommended_point, 1)
        actual_constraint2_value = function.evaluate_task(normalized_best_recommended_point, 2)
        actual_constraint3_value = function.evaluate_task(normalized_best_recommended_point, 3)
        actual_constraint4_value = function.evaluate_task(normalized_best_recommended_point, 4)
        actual_constraint5_value = function.evaluate_task(normalized_best_recommended_point, 5)
        actual_constraint6_value = function.evaluate_task(normalized_best_recommended_point, 6)
        is_location_feasible1 = actual_constraint1_value <= 0
        is_location_feasible2 = actual_constraint2_value <= 0
        is_location_feasible3 = actual_constraint3_value <= 0
        is_location_feasible4 = actual_constraint4_value <= 0
        is_location_feasible5 = actual_constraint5_value <= 0
        is_location_feasible6 = actual_constraint6_value <= 0

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, rtol=1e-4)
        self.assertEqual(True, is_location_feasible1)
        self.assertEqual(True, is_location_feasible2)
        self.assertEqual(True, is_location_feasible3)
        self.assertEqual(True, is_location_feasible4)
        self.assertEqual(True, is_location_feasible5)
        self.assertEqual(True, is_location_feasible6)
    