import random

import numpy as np
import torch
from botorch.utils.testing import BotorchTestCase
from botorch.utils.transforms import normalize

from bo.model.Model import ConstrainedDeoupledGPModelWrapper
from bo.synthetic_test_functions.cnn_takena22_benchmark import const_cnn_cifar10
from bo.synthetic_test_functions.synthetic_test_functions import MysteryFunction, ConstrainedBraninNew, \
    ConstrainedFunc3, ConstrainedFunc3Redundant, PressureVessel, WeldedBeamSO, TwoLayerCNN_train, \
    TensionCompression, SpeedReducer

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
        expected = torch.tensor([[-2., 5., 3., 6., 0.4],
                                 [0., 8, 4, 5, 1.9]])
        x = torch.tensor([[-2.2, 5.2, 3.1, 5.9, 0.36],
                          [-0.3, 7.6, 4.3, 5.5, 2.0]])

        result = function.discretise_inputs(x.clone())

        self.assertEqual((2, 5), result.shape)
        self.assertAllClose(result, expected)

    def test_cnn_becnhmark_takena22_discrete_approximation_edge_case(self):
        set_all_seeds(8)
        function = const_cnn_cifar10()
        x = torch.tensor([[-1., 7., 6., 4., 1.2]])
        expected = torch.tensor([[-1., 7., 6., 4., 1.2]])

        result = function.discretise_inputs(x.clone())

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

    def test_cnn_benchmark_takena22_optimal_location(self):
        set_all_seeds(8)
        function = const_cnn_cifar10()
        expected_original_inputs = torch.tensor([[0., 6., 6., 6., 1.5]])
        normalized_best_recommended_point = function._transform_hypers_to_cube(expected_original_inputs)
        expected_best_fval = torch.tensor([0.8349473328872954], dtype=torch.float64)

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
        actual_full_vector = function.evaluate_black_box(normalized_best_recommended_point, False)

        self.assertAllClose(expected_best_fval, actual_best_fval, rtol=1e-4)
        self.assertEqual(True, is_location_feasible1)
        self.assertEqual(True, is_location_feasible2)
        self.assertEqual(True, is_location_feasible3)
        self.assertEqual(True, is_location_feasible4)
        self.assertEqual(True, is_location_feasible5)
        self.assertEqual(True, is_location_feasible6)
        self.assertEqual(True, is_location_feasible7)
        self.assertEqual(True, is_location_feasible8)
        self.assertEqual(True, is_location_feasible9)
        self.assertEqual(True, is_location_feasible10)
        self.assertAllClose(expected_best_fval.item(), actual_full_vector[:, 0].item(), atol=1e-3)
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
        self.assertFalse(function.is_expensive())
        self.assertFalse(function.is_noisy())
        self.assertEqual(actual_full_vector.shape[1], 11)

    def test_constrained_func3_redundant_noise(self):
        f = ConstrainedFunc3Redundant(noise_std=1e-2, negate=True)
        torch.manual_seed(0)
        X = torch.rand(10, 2, dtype=torch.double)
        npo = f.get_noise_per_output()

        # evaluate_black_box: is_repeated=True must be deterministic for all outputs
        det1 = f.evaluate_black_box(X, is_repeated=True)
        det2 = f.evaluate_black_box(X, is_repeated=True)
        self.assertAllClose(det1, det2)

        def _is_noisy(v):
            return v is not None and float(v) > 1e-6

        # evaluate_black_box: is_repeated=False — float outputs noisy, None near-deterministic
        noisy1 = f.evaluate_black_box(X, is_repeated=False)
        noisy2 = f.evaluate_black_box(X, is_repeated=False)
        for i, v in enumerate(npo):
            if _is_noisy(v):
                self.assertFalse(
                    torch.allclose(noisy1[:, i], noisy2[:, i]),
                    msg=f"evaluate_black_box output {i} should be stochastic",
                )
            else:
                self.assertAllClose(noisy1[:, i], noisy2[:, i])

        # evaluate_task: float outputs noisy, None near-deterministic
        for i, v in enumerate(npo):
            t1 = f.evaluate_task(X, i)
            t2 = f.evaluate_task(X, i)
            if _is_noisy(v):
                self.assertFalse(
                    torch.allclose(t1, t2),
                    msg=f"evaluate_task({i}) should be stochastic",
                )
            else:
                self.assertAllClose(t1, t2)

    def test_constrained_func3_redundant_optimal_values(self):
        expected_best_fval = 0.7483
        best_recommended_point = torch.tensor([[0.2018, 0.8332]], dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        normalized_best_point = normalize(best_recommended_point, bounds=bounds)

        f = ConstrainedFunc3Redundant(noise_std=1e-2, negate=True)
        bb = f.evaluate_black_box(normalized_best_point, is_repeated=True)

        self.assertEqual(bb.shape, torch.Size([1, 6]))
        self.assertAllClose(bb[:, 0], torch.tensor([expected_best_fval], dtype=torch.double), atol=1e-3)
        for i in range(1, 6):
            self.assertTrue((bb[:, i] <= 0).all(), msg=f"constraint {i} should be feasible at optimum")
        self.assertAllClose(bb[:, 4], torch.tensor([-100.0], dtype=torch.double))
        self.assertAllClose(bb[:, 5], torch.tensor([-100.0], dtype=torch.double))

        self.assertTrue(f.is_noisy())
        self.assertFalse(f.is_expensive())
        self.assertEqual(f.get_number_of_constraints(), 5)
        self.assertEqual(f.get_name(), "test_function_3_redundant")

    def test_constrained_func3_redundant_noisy_active_constraint_variants(self):
        # noisy_active_constraint selects which of c1/c2/c3 carries observation noise;
        # obj (index 0) and c5 (index 5) are noisy in every variant, c4 (index 4) never is.
        expected_name = {1: "test_function_3_redundant_c1noisy",
                         2: "test_function_3_redundant",
                         3: "test_function_3_redundant_c3noisy"}
        for k in (1, 2, 3):
            f = ConstrainedFunc3Redundant(noise_std=0.0, negate=True, noisy_active_constraint=k)
            self.assertEqual(f.get_name(), expected_name[k])

            npo = f.get_noise_per_output()
            self.assertEqual(len(npo), 6)
            self.assertIsNotNone(npo[0])
            self.assertIsNone(npo[4])
            self.assertIsNotNone(npo[5])
            for c in (1, 2, 3):
                if c == k:
                    self.assertIsNotNone(npo[c], msg=f"constraint {c} should be noisy when k={k}")
                else:
                    self.assertIsNone(npo[c], msg=f"constraint {c} should be noiseless when k={k}")

    def test_gp_noise_converges_to_true_levels(self):
        torch.manual_seed(0)
        f = ConstrainedFunc3Redundant(noise_std=0, negate=True)
        npo = f.get_noise_per_output()
        n_outputs = 6

        N_small, N_large = 20, 200
        X_all = torch.rand(N_large, 2, dtype=torch.double)
        Y_all = f.evaluate_black_box(X_all, is_repeated=False)

        def fit_wrapper(X, Y_full):
            wrapper = ConstrainedDeoupledGPModelWrapper(
                num_constraints=n_outputs - 1, is_noisy=True
            )
            wrapper.fit([X] * n_outputs, [Y_full[:, i] for i in range(n_outputs)])
            wrapper.optimize()
            noise_orig = []
            for i in range(n_outputs):
                stdvs = wrapper.model.models[i].outcome_transform.stdvs.squeeze().item()
                noise_orig.append(wrapper.model.models[i].likelihood.noise.item() * stdvs ** 2)
            return noise_orig

        noise_small = fit_wrapper(X_all[:N_small], Y_all[:N_small])
        noise_large = fit_wrapper(X_all, Y_all)

        noisy_idxs = [i for i, v in enumerate(npo) if v is not None and float(v) > 1e-6]
        ndet_idxs = [i for i, v in enumerate(npo) if v is None]

        # Total error across all noisy outputs must decrease with more data
        total_err_small = sum(abs(noise_small[i] - npo[i]) for i in noisy_idxs)
        total_err_large = sum(abs(noise_large[i] - npo[i]) for i in noisy_idxs)
        self.assertLess(
            total_err_large, total_err_small,
            msg=(f"N={N_large} total noise error ({total_err_large:.4f}) should be smaller than "
                 f"N={N_small} ({total_err_small:.4f})")
        )

        # With N_large, noisy outputs should have clearly higher noise than near-deterministic
        min_noisy = min(noise_large[i] for i in noisy_idxs)
        max_ndet = max(noise_large[i] for i in ndet_idxs)
        self.assertGreater(
            min_noisy, max_ndet,
            msg="Noisy outputs should have higher learned noise than near-deterministic outputs"
        )
