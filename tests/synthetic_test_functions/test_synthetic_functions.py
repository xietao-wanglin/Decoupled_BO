import random

import numpy as np
import torch
from botorch.utils.testing import BotorchTestCase
from botorch.utils.transforms import normalize

from bo.synthetic_test_functions.cnn_takena22_benchmark import const_cnn_cifar10
from bo.synthetic_test_functions.synthetic_test_functions import MysteryFunction, ConstrainedBraninNew, \
    ConstrainedFunc3, PressureVessel, WeldedBeamSO, TwoLayerCNN_train, TensionCompression, SpeedReducer

device = torch.device("cpu")
dtype = torch.double

def set_all_seeds(seed):
    """
    Sets the random seed for reproducibility across PyTorch, NumPy, and Python's random module.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Disable benchmark for deterministic behavior
    np.random.seed(seed)
    random.seed(seed)
    print(f"All random seeds set to {seed}")


class TestDecoupledKG(BotorchTestCase):

    def test_mistery_function_optimal_values(self):
        expected_best_fval = 1.1743
        best_recommended_point = torch.tensor([2.7450, 2.3523])
        bounds = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
        normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
        function = MysteryFunction(negate=True)

        actual_best_fval = function.evaluate_task(normalized_best_recommended_point, 0)
        actual_constraint_value = function.evaluate_task(normalized_best_recommended_point, 1)
        actual_full_vector = function.evaluate_black_box(normalized_best_recommended_point, False)
        is_location_feasible = actual_constraint_value <= 0

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, atol=1e-4)
        self.assertEqual(True, is_location_feasible)
        self.assertAllClose(expected_best_fval, actual_full_vector[:, 0].item(), atol=1e-4)
        self.assertAllClose(actual_constraint_value.item(), actual_full_vector[:, 1].item(), atol=1e-4)
        self.assertFalse(function.is_expensive())
        self.assertFalse(function.is_noisy())
        self.assertEqual(actual_full_vector.shape[1], 2)

    def test_new_branin_function_optimal_values(self):
        expected_best_fval = 268.781
        best_recommended_point = torch.tensor([3.273, 0.0489])
        bounds = torch.tensor([[-5.0, 0.0],
                               [10.0, 15.0]])
        normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
        function = ConstrainedBraninNew(negate=True)

        actual_best_fval = function.evaluate_task(normalized_best_recommended_point, 0)
        actual_constraint_value = function.evaluate_task(normalized_best_recommended_point, 1)
        actual_full_vector = function.evaluate_black_box(normalized_best_recommended_point, False)
        is_location_feasible = actual_constraint_value <= 0

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, atol=1e-2)
        self.assertEqual(True, is_location_feasible)
        self.assertAllClose(expected_best_fval, actual_full_vector[:, 0].item(), atol=1e-2)
        self.assertAllClose(actual_constraint_value.item(), actual_full_vector[:, 1].item(), atol=1e-4)
        self.assertFalse(function.is_expensive())
        self.assertFalse(function.is_noisy())
        self.assertEqual(actual_full_vector.shape[1], 2)

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
        actual_full_vector = function.evaluate_black_box(normalized_best_recommended_point, False)

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, atol=1e-3)
        self.assertEqual(True, is_location_feasible1)
        self.assertEqual(True, is_location_feasible2)
        self.assertEqual(True, is_location_feasible3)
        self.assertAllClose(expected_best_fval, actual_full_vector[:, 0].item(), atol=1e-3)
        self.assertAllClose(actual_constraint1_value.item(), actual_full_vector[:, 1].item(), atol=1e-4)
        self.assertAllClose(actual_constraint2_value.item(), actual_full_vector[:, 2].item(), atol=1e-4)
        self.assertAllClose(actual_constraint3_value.item(), actual_full_vector[:, 3].item(), atol=1e-4)
        self.assertFalse(function.is_expensive())
        self.assertFalse(function.is_noisy())
        self.assertEqual(actual_full_vector.shape[1], 4)

    def test_pressure_vessel(self):
        expected_best_fval = -6059.946341
        best_recommended_point = torch.tensor([0.812500, 0.437500, 42.097398, 176.654047])
        bounds = torch.tensor([[0.0, 0.0, 10.0, 150.0], [10.0, 10.0, 50.0, 200.0]])
        normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
        function = PressureVessel(negate=True)

        actual_best_fval = function.evaluate_task(normalized_best_recommended_point, 0)
        actual_constraint1_value = function.evaluate_task(normalized_best_recommended_point, 1)
        actual_constraint2_value = function.evaluate_task(normalized_best_recommended_point, 2)
        actual_constraint3_value = function.evaluate_task(normalized_best_recommended_point, 3)
        actual_constraint4_value = function.evaluate_task(normalized_best_recommended_point, 4)
        is_location_feasible1 = actual_constraint1_value <= 0
        is_location_feasible2 = actual_constraint2_value <= 0
        is_location_feasible3 = actual_constraint3_value <= 0
        is_location_feasible4 = actual_constraint4_value <= 0
        actual_full_vector = function.evaluate_black_box(normalized_best_recommended_point, False)

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, rtol=1e-2)
        self.assertEqual(True, is_location_feasible1)
        self.assertEqual(True, is_location_feasible2)
        self.assertEqual(True, is_location_feasible3)
        self.assertEqual(True, is_location_feasible4)
        self.assertAllClose(expected_best_fval, actual_full_vector[:, 0].item(), atol=1e-3)
        self.assertAllClose(actual_constraint1_value.item(), actual_full_vector[:, 1].item(), atol=1e-4)
        self.assertAllClose(actual_constraint2_value.item(), actual_full_vector[:, 2].item(), atol=1e-4)
        self.assertAllClose(actual_constraint3_value.item(), actual_full_vector[:, 3].item(), atol=1e-4)
        self.assertAllClose(actual_constraint4_value.item(), actual_full_vector[:, 4].item(), atol=1e-4)
        self.assertFalse(function.is_expensive())
        self.assertFalse(function.is_noisy())
        self.assertEqual(actual_full_vector.shape[1], 5)

    def test_tension_compression(self):
        expected_best_fval = -0.012666
        best_recommended_point = torch.tensor([0.05174250340926, 0.35800478345599, 11.21390736278739])
        bounds = torch.tensor([[0.01, 0.01, 0.01], [1.0, 1.0, 20.0]])
        normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
        function = TensionCompression(negate=True)

        actual_best_fval = function.evaluate_task(normalized_best_recommended_point, 0)
        actual_constraint1_value = function.evaluate_task(normalized_best_recommended_point, 1)
        actual_constraint2_value = function.evaluate_task(normalized_best_recommended_point, 2)
        actual_constraint3_value = function.evaluate_task(normalized_best_recommended_point, 3)
        actual_constraint4_value = function.evaluate_task(normalized_best_recommended_point, 4)
        is_location_feasible1 = actual_constraint1_value <= 0
        is_location_feasible2 = actual_constraint2_value <= 0
        is_location_feasible3 = actual_constraint3_value <= 0
        is_location_feasible4 = actual_constraint4_value <= 0
        actual_full_vector = function.evaluate_black_box(normalized_best_recommended_point, False)

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, rtol=1e-2)
        self.assertEqual(True, is_location_feasible1)
        self.assertEqual(True, is_location_feasible2)
        self.assertEqual(True, is_location_feasible3)
        self.assertEqual(True, is_location_feasible4)
        self.assertAllClose(expected_best_fval, actual_full_vector[:, 0].item(), atol=1e-3)
        self.assertAllClose(actual_constraint1_value.item(), actual_full_vector[:, 1].item(), atol=1e-4)
        self.assertAllClose(actual_constraint2_value.item(), actual_full_vector[:, 2].item(), atol=1e-4)
        self.assertAllClose(actual_constraint3_value.item(), actual_full_vector[:, 3].item(), atol=1e-4)
        self.assertAllClose(actual_constraint4_value.item(), actual_full_vector[:, 4].item(), atol=1e-4)
        self.assertFalse(function.is_expensive())
        self.assertFalse(function.is_noisy())
        self.assertEqual(actual_full_vector.shape[1], 5)

    def test_welded_beam(self):
        expected_best_fval = -1.728226
        best_recommended_point = torch.tensor([0.205986, 3.471328, 9.020224, 0.206480])
        bounds = torch.tensor([[0.125, 0.1, 0.1, 0.1], [10.0, 10.0, 10.0, 10.0]])
        normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
        function = WeldedBeamSO(negate=True)

        actual_best_fval = function.evaluate_task(normalized_best_recommended_point, 0)
        actual_constraint1_value = function.evaluate_task(normalized_best_recommended_point, 1)
        actual_constraint2_value = function.evaluate_task(normalized_best_recommended_point, 2)
        actual_constraint3_value = function.evaluate_task(normalized_best_recommended_point, 3)
        actual_constraint4_value = function.evaluate_task(normalized_best_recommended_point, 4)
        actual_constraint5_value = function.evaluate_task(normalized_best_recommended_point, 5)
        is_location_feasible1 = actual_constraint1_value <= 0
        is_location_feasible2 = actual_constraint2_value <= 0
        is_location_feasible3 = actual_constraint3_value <= 0
        is_location_feasible4 = actual_constraint4_value <= 0
        is_location_feasible5 = actual_constraint5_value <= 0
        actual_full_vector = function.evaluate_black_box(normalized_best_recommended_point, False)

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, rtol=1e-4)
        self.assertEqual(True, is_location_feasible1)
        self.assertEqual(True, is_location_feasible2)
        self.assertEqual(True, is_location_feasible3)
        self.assertEqual(True, is_location_feasible4)
        self.assertEqual(True, is_location_feasible5)
        self.assertAllClose(expected_best_fval, actual_full_vector[:, 0].item(), atol=1e-3)
        self.assertAllClose(actual_constraint1_value.item(), actual_full_vector[:, 1].item(), atol=1e-4)
        self.assertAllClose(actual_constraint2_value.item(), actual_full_vector[:, 2].item(), atol=1e-4)
        self.assertAllClose(actual_constraint3_value.item(), actual_full_vector[:, 3].item(), atol=1e-4)
        self.assertAllClose(actual_constraint4_value.item(), actual_full_vector[:, 4].item(), atol=1e-4)
        self.assertAllClose(actual_constraint5_value.item(), actual_full_vector[:, 5].item(), atol=1e-4)
        self.assertFalse(function.is_expensive())
        self.assertFalse(function.is_noisy())
        self.assertEqual(actual_full_vector.shape[1], 6)

    def test_speed_reducer(self):
        expected_best_fval = -2996.3482
        best_recommended_point = torch.tensor([3.50, 0.7, 17, 7.3, 7.8, 3.350215, 5.286683])
        bounds = torch.tensor([[2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0],
                               [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]])
        normalized_best_recommended_point = normalize(best_recommended_point, bounds=bounds)
        function = SpeedReducer(negate=True)

        actual_best_fval = function.evaluate_task(normalized_best_recommended_point, 0)
        actual_constraint1_value = function.evaluate_task(normalized_best_recommended_point, 1)
        actual_constraint2_value = function.evaluate_task(normalized_best_recommended_point, 2)
        actual_constraint3_value = function.evaluate_task(normalized_best_recommended_point, 3)
        actual_constraint4_value = function.evaluate_task(normalized_best_recommended_point, 4)
        actual_constraint5_value = function.evaluate_task(normalized_best_recommended_point, 5)
        actual_constraint6_value = function.evaluate_task(normalized_best_recommended_point, 6)
        actual_constraint7_value = function.evaluate_task(normalized_best_recommended_point, 7)
        actual_constraint8_value = function.evaluate_task(normalized_best_recommended_point, 8)
        actual_constraint9_value = function.evaluate_task(normalized_best_recommended_point, 9)
        actual_constraint10_value = function.evaluate_task(normalized_best_recommended_point, 10)
        actual_constraint11_value = function.evaluate_task(normalized_best_recommended_point, 11)
        is_location_feasible1 = actual_constraint1_value <= 0
        is_location_feasible2 = actual_constraint2_value <= 0
        is_location_feasible3 = actual_constraint3_value <= 0
        is_location_feasible4 = actual_constraint4_value <= 0
        is_location_feasible5 = actual_constraint5_value <= 0
        is_location_feasible6 = actual_constraint6_value <= 0
        is_location_feasible7 = actual_constraint7_value <= 0
        is_location_feasible8 = actual_constraint8_value <= 0
        is_location_feasible9 = actual_constraint9_value <= 0
        is_location_feasible10 = actual_constraint10_value <= 0
        is_location_feasible11 = actual_constraint11_value <= 0
        actual_full_vector = function.evaluate_black_box(normalized_best_recommended_point, False)

        self.assertAllClose(torch.tensor(expected_best_fval), actual_best_fval, rtol=1e-4)
        self.assertEqual(True, is_location_feasible1)
        self.assertEqual(True, is_location_feasible2)
        self.assertEqual(True, is_location_feasible3)
        self.assertEqual(True, is_location_feasible4)
        self.assertEqual(True, is_location_feasible5)
        # self.assertEqual(True, is_location_feasible6)
        self.assertEqual(True, is_location_feasible7)
        self.assertEqual(True, is_location_feasible8)
        self.assertEqual(True, is_location_feasible9)
        self.assertEqual(True, is_location_feasible10)
        self.assertEqual(True, is_location_feasible11)
        self.assertAllClose(expected_best_fval, actual_full_vector[:, 0].item(), atol=1e-3)
        self.assertAllClose(actual_constraint1_value.item(), actual_full_vector[:, 1].item(), atol=1e-4)
        self.assertAllClose(actual_constraint2_value.item(), actual_full_vector[:, 2].item(), atol=1e-4)
        self.assertAllClose(actual_constraint3_value.item(), actual_full_vector[:, 3].item(), atol=1e-4)
        self.assertAllClose(actual_constraint4_value.item(), actual_full_vector[:, 4].item(), atol=1e-4)
        self.assertAllClose(actual_constraint5_value.item(), actual_full_vector[:, 5].item(), atol=1e-4)
        self.assertAllClose(actual_constraint6_value.item(), actual_full_vector[:, 6].item(), atol=1e-4)
        self.assertAllClose(actual_constraint7_value.item(), actual_full_vector[:, 7].item(), atol=1e-4)
        self.assertAllClose(actual_constraint8_value.item(), actual_full_vector[:, 8].item(), atol=1e-4)
        self.assertAllClose(actual_constraint9_value.item(), actual_full_vector[:, 9].item(), atol=1e-4)
        self.assertAllClose(actual_constraint10_value.item(), actual_full_vector[:, 10].item(), atol=1e-4)
        self.assertAllClose(actual_constraint11_value.item(), actual_full_vector[:, 11].item(), atol=1e-4)
        self.assertFalse(function.is_expensive())
        self.assertFalse(function.is_noisy())
        self.assertEqual(actual_full_vector.shape[1], 12)

    def test_cnn_becnhmark(self):
        set_all_seeds(8)
        function = TwoLayerCNN_train(negate=False)
        configuration_tensor = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
        actual_full_vector = function.evaluate_black_box(configuration_tensor, False)

        task_idx = 0
        set_all_seeds(8)
        function = TwoLayerCNN_train(negate=False)
        task_value = function.evaluate_task(configuration_tensor, task_idx)
        self.assertAllClose(actual_full_vector[:, task_idx].item(), task_value.item(), 1e-6)
        self.assertTrue(task_value.dtype == torch.float32)

        self.assertTrue(function.is_expensive())
        self.assertTrue(function.is_noisy())
        self.assertEqual(actual_full_vector.shape[1], 11)

    def test_cnn_becnhmark_takena22_input_transform(self):
        set_all_seeds(8)
        function = const_cnn_cifar10()
        tensor = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0]], dtype=dtype)

        scaled_vector = function.transform_cube_to_hypers(tensor)
        lb_vector = function.transform_cube_to_hypers(tensor[0, :])
        ub_vector = function.transform_cube_to_hypers(tensor[1, :])

        self.assertEqual((2, 5), scaled_vector.shape)
        self.assertEqual((1, 5), lb_vector.shape)
        self.assertEqual((1, 5), ub_vector.shape)
        self.assertAllClose(function.bounds[0, :], scaled_vector[0, :])
        self.assertAllClose(function.bounds[1, :], scaled_vector[1, :])
        self.assertAllClose(scaled_vector[0, :], lb_vector.reshape(-1))
        self.assertAllClose(scaled_vector[1, :], ub_vector.reshape(-1))
        self.assertAllClose(function.bounds[0, :], lb_vector.reshape(-1))
        self.assertAllClose(function.bounds[1, :], ub_vector.reshape(-1))

    def test_cnn_becnhmark_takena22_discrete_approximation(self):
        set_all_seeds(8)
        function = const_cnn_cifar10()
        x = torch.tensor([[-2.2, 5.2, 3.1, 5.9, 0.36]])
        result = function.discretise_inputs(x.clone())

        expected = torch.tensor([[-2., 5., 3., 6., 0.4]])

        self.assertEqual((1, 5), result.shape)
        self.assertAllClose(result, expected)

    def test_cnn_becnhmark_takena22_discrete_approximation_2D(self):
        set_all_seeds(8)
        function = const_cnn_cifar10()
        x = torch.tensor([[-2.2, 5.2, 3.1, 5.9, 0.36],
                          [-0.3, 7.6, 4.3, 5.5, 2.0]])

        result = function.discretise_inputs(x.clone())

        expected = torch.tensor([[-2., 5., 3., 6., 0.4],
                                 [0., 8, 4, 5, 1.9]])

        self.assertEqual((2, 5), result.shape)
        self.assertAllClose(result, expected)

    def test_cnn_becnhmark_takena22_discrete_approximation_edge_case(self):
        set_all_seeds(8)
        function = const_cnn_cifar10()
        x = torch.tensor([[-1., 7., 6., 4., 1.2]])
        result = function.discretise_inputs(x.clone())

        expected = torch.tensor([[-1., 7., 6., 4., 1.2]])
        self.assertEqual((1, 5), result.shape)
        self.assertAllClose(result, expected)

    def test_cnn_becnhmark_takena22_evaluate_task_lb(self):
        set_all_seeds(8)
        function = const_cnn_cifar10()
        x = torch.tensor([[0., 0., 0., 0., 0.]])
        expected = torch.tensor([-2.1113349054557897])

        actual = function.evaluate_task(x.clone(), 0)

        self.assertAlmostEqual(expected, actual, delta=1e-6)

    def test_cnn_becnhmark_takena22_evaluate_task_ub(self):
        set_all_seeds(8)
        function = const_cnn_cifar10()
        x = torch.tensor([[1., 1., 1., 1., 1.]])
        expected = torch.tensor([0.612013624050499])

        actual = function.evaluate_task(x.clone(), 0)

        self.assertAlmostEqual(expected, actual, delta=1e-6)

    def test_cnn_becnhmark_takena22_evaluate_slack_true(self):
        set_all_seeds(8)
        function = const_cnn_cifar10()
        x = torch.tensor([[0., 0., 0., 0., 0.],
                          [1., 1., 1., 1., 1.]])
        expected = torch.tensor([[-2.11133490545578972, 11.51291546492022810, 11.51291546492022810,
                                  11.51291546492022810, 11.51291546492022810, 11.51291546492022810,
                                  -3.00946730194414247, 11.51291546492022810, 11.51291546492022810,
                                  11.51291546492022810, 1.92774846938101097],
                                 [0.61201362405049897, -0.99462257514406194, -1.39255612234386716,
                                  -0.19259310711578442, -0.26962157572526196, -0.47260441094579281,
                                  0.41380544002438319, -0.85206431364390345, -0.77221570403563855,
                                  -1.04077751362550841, -0.84729786038720345]], dtype=dtype)
        actual = function.evaluate_slack_true(x.clone())

        self.assertEqual(expected.shape, actual.shape)
        self.assertFalse(function.is_expensive())
        self.assertFalse(function.is_noisy())
        self.assertAllClose(expected, actual)
