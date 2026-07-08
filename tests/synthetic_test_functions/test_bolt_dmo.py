import torch
from botorch.utils.testing import BotorchTestCase

from Launcher import run_experiment_coupled_acquisition_functions, run_experiment_decoupled_acquisition_functions
from bo.bo_loops.bayesian_optimization_loop_type import BayesianOptimizationLoopType
from bo.synthetic_test_functions.bolt_dmo_benchmark import DecoupledBOLTDMO

device = torch.device("cpu")
dtype = torch.double
torch.set_default_dtype(dtype)


class TestDecoupledBOLTDMO(BotorchTestCase):

    @classmethod
    def setUpClass(cls):
        cls.function = DecoupledBOLTDMO(negate=False)

    def test_simplex_map(self):
        Z = torch.rand(20, 4, dtype=dtype)
        corners = torch.tensor([[0., 0., 0., 0.],
                                [1., 1., 1., 1.],
                                [0., 1., 1., 0.],
                                [1., 0., 0., 1.]], dtype=dtype)
        X_bolt = self.function.transform_inputs(torch.cat([Z, corners]))
        self.assertEqual(X_bolt.shape, torch.Size([24, 6]))
        self.assertTrue(torch.allclose(X_bolt[:, :3].sum(dim=1), torch.ones(24, dtype=dtype)))
        self.assertTrue(torch.allclose(X_bolt[:, 3:].sum(dim=1), torch.ones(24, dtype=dtype)))
        self.assertTrue((X_bolt >= 0).all() and (X_bolt <= 1).all())

    def test_black_box_matches_tasks(self):
        X = torch.rand(5, 4, dtype=dtype)
        black_box = self.function.evaluate_black_box(X)
        self.assertEqual(black_box.shape, torch.Size([5, 3]))
        for task_idx in range(3):
            self.assertTrue(torch.allclose(self.function.evaluate_task(X, task_idx), black_box[:, task_idx]))

    def test_reference_optimum_is_feasible(self):
        values = self.function.evaluate_black_box(self.function.x_star_ref.unsqueeze(0))
        self.assertAlmostEqual(values[0, 0].item(), self.function.CONSTRAINED_MAX, places=6)
        self.assertTrue((values[0, 1:] <= 0).all())
        self.assertLess(self.function.CONSTRAINED_MAX, self.function.GLOBAL_MAX)

    def test_problem_definition(self):
        self.assertEqual(self.function.dim, 4)
        self.assertEqual(self.function.get_number_of_constraints(), 2)
        self.assertEqual(self.function.get_name(), "bolt_dmo")
        self.assertFalse(self.function.is_noisy())
        self.assertFalse(self.function.is_expensive())
        # penalty must sit below the worst objective value on the reference set
        self.assertGreater(self.function.get_penalty(), abs(self.function.GLOBAL_MAX))


class TestBOLTDMOSmokeRun(BotorchTestCase):

    def test_decoupled_run(self):
        black_box_function = DecoupledBOLTDMO(negate=False)
        experiment_args = {"black_box_function": black_box_function,
                           "budget": 2,
                           "seed": 0,
                           "bayesian_optimization_algorithm": BayesianOptimizationLoopType.DCKG_INDEPENDENT,
                           "number_of_initial_designs": 1,
                           "cost": None}
        run_experiment_decoupled_acquisition_functions(**experiment_args)

    def test_coupled_run(self):
        black_box_function = DecoupledBOLTDMO(negate=False)
        experiment_args = {"black_box_function": black_box_function,
                           "budget": 2,
                           "seed": 0,
                           "bayesian_optimization_algorithm": BayesianOptimizationLoopType.CEI,
                           "number_of_initial_designs": 1}
        run_experiment_coupled_acquisition_functions(**experiment_args)
