import os
import pickle
import shutil
import tempfile

import torch
from botorch.acquisition import ConstrainedMCObjective
from botorch.utils.testing import BotorchTestCase

from Launcher import run_experiment_coupled_acquisition_functions
from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType
from bo.bo_loops.bayesian_optimization_factory import BayesianOptimizationLoopFactory
from bo.bo_loops.bayesian_optimization_loop_type import BayesianOptimizationLoopType
from bo.bo_loops.bo_loop import OptimizationLoop
from bo.model.Model import (
    ConstrainedDeoupledGPModelWrapper,
    constraint_callable_wrapper,
    obj_callable,
)
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import MysteryFunctionSuperRedundant


class _TmpCwdMixin:
    """Run each test inside a temp directory so the real results/ folder is untouched."""

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.mkdtemp()
        self._original_cwd = os.getcwd()
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._original_cwd)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().tearDown()


class TestResultsLoadFromFile(_TmpCwdMixin, BotorchTestCase):

    def test_roundtrip_preserves_fields(self):
        r = Results(filename="roundtrip.pkl")
        r.random_seed(42)
        r.save_budget(20)
        r.save_input_data([torch.tensor([[0.1, 0.2]]), torch.tensor([[0.3, 0.4]])])
        r.save_output_data([torch.tensor([1.0]), torch.tensor([2.0])])
        r.save_number_initial_points(6)
        r.save_performance_type("model")
        r.save_best_predicted_location(torch.tensor([[0.5, 0.5]]))
        r.save_best_predicted_location_true_value(0.7)
        r.save_acqf_recommended_location(torch.tensor([[0.6, 0.6]]))
        r.save_acqf_recommended_location_true_value(0.8)
        r.save_acqf_recommended_output_index(torch.tensor(0))
        r.save_budget_consumed(torch.tensor(5.0))
        r.save_acqf_values([0.123])
        r.save_model_length_scales([0.4, 0.5])
        r.save_cost_configurations(torch.ones(3))
        r.save_failing_constraint(-1)
        r.save_evaluated_functions(torch.tensor([1.0, 2.0, 3.0]))
        r.generate_pkl_file()

        loaded = Results.load_from_file(r.filepath)

        self.assertEqual(loaded.seed, 42)
        self.assertEqual(loaded.budget, 20)
        self.assertEqual(loaded.number_initial_samples, 6)
        self.assertEqual(loaded.performance_type, "model")
        self.assertEqual(loaded.budget_consumed, [5.0])
        self.assertEqual(len(loaded.best_predicted_location), 1)
        self.assertEqual(len(loaded.acqf_recommended_location), 1)
        self.assertEqual(len(loaded.acqf_values), 1)
        self.assertEqual(len(loaded.evals), 1)
        self.assertEqual(loaded.failing_constraint, ["None"])

    def test_loaded_lists_are_independent_copies(self):
        # Mutating the loaded result should not corrupt the on-disk dict
        # if we re-load it later.
        r = Results(filename="indep.pkl")
        r.random_seed(0)
        r.save_budget(5)
        r.save_input_data([torch.zeros(1, 2)])
        r.save_output_data([torch.zeros(1)])
        r.save_number_initial_points(1)
        r.save_performance_type("model")
        r.save_cost_configurations(None)
        r.save_budget_consumed(torch.tensor(1.0))
        r.generate_pkl_file()

        loaded = Results.load_from_file(r.filepath)
        loaded.budget_consumed.append(999.0)

        reloaded = Results.load_from_file(r.filepath)
        self.assertEqual(reloaded.budget_consumed, [1.0])


class TestLoadOrSkip(_TmpCwdMixin, BotorchTestCase):
    """Verifies the skip/resume/fresh decision branches in the factory."""

    def setUp(self):
        super().setUp()
        os.makedirs("results", exist_ok=True)
        bb = MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True,
                                           redundant_constraints=False)
        ncons = bb.get_number_of_constraints()
        # _load_or_skip only touches self.budget and filesystem state, so
        # model/objective can be None for this isolated unit test.
        self.factory = BayesianOptimizationLoopFactory(
            black_box_function=bb,
            constrained_obj=None, model=None, seed=0, budget=10,
            penalty_value=torch.tensor([0.0]), costs=torch.ones(ncons + 1),
            number_of_constraints=ncons, base_file_name="test",
        )

    def _write_pkl(self, filename, budget_consumed_list=None,
                   acqf_locations=None, stored_budget=10):
        r = Results(filename=filename)
        r.save_budget(stored_budget)
        r.save_input_data([torch.zeros(8, 2), torch.zeros(8, 2)])
        r.save_output_data([torch.zeros(8), torch.zeros(8)])
        r.random_seed(0)
        r.save_number_initial_points(6)
        r.save_performance_type("model")
        r.save_cost_configurations(None)
        for bc in (budget_consumed_list or []):
            r.save_budget_consumed(torch.tensor(float(bc)))
        for loc in (acqf_locations or []):
            r.save_acqf_recommended_location(loc)
        r.generate_pkl_file()

    def test_no_existing_file_returns_fresh(self):
        results, resume_state, skip = self.factory._load_or_skip(
            "nofile.pkl", effective_budget=10,
        )
        self.assertFalse(skip)
        self.assertIsNone(resume_state)
        self.assertEqual(results.filename, "nofile.pkl")
        self.assertIsNone(results.input_data)

    def test_complete_file_returns_skip(self):
        self._write_pkl("done.pkl", budget_consumed_list=[1, 3, 7, 10])
        results, resume_state, skip = self.factory._load_or_skip(
            "done.pkl", effective_budget=10,
        )
        self.assertTrue(skip)
        self.assertIsNone(resume_state)

    def test_overshoot_file_returns_skip(self):
        # consumed=12 against target=10 -> skip
        self._write_pkl("over.pkl", budget_consumed_list=[4, 8, 12])
        _, resume_state, skip = self.factory._load_or_skip(
            "over.pkl", effective_budget=10,
        )
        self.assertTrue(skip)
        self.assertIsNone(resume_state)

    def test_partial_file_returns_resume_state(self):
        self._write_pkl("partial.pkl", budget_consumed_list=[1, 3, 4])
        _, resume_state, skip = self.factory._load_or_skip(
            "partial.pkl", effective_budget=10,
        )
        self.assertFalse(skip)
        self.assertIsNotNone(resume_state)
        self.assertEqual(resume_state["budget_consumed"], 4.0)
        self.assertEqual(len(resume_state["train_x"]), 2)
        self.assertEqual(len(resume_state["train_y"]), 2)

    def test_coupled_file_without_budget_consumed_uses_acqf_length(self):
        # Coupled CEI loop never calls save_budget_consumed, so budget_consumed
        # is []. _load_or_skip must fall back to len(acqf_recommended_location).
        self._write_pkl(
            "coupled_partial.pkl",
            budget_consumed_list=None,
            acqf_locations=[torch.zeros(1, 2) for _ in range(3)],
        )
        _, resume_state, skip = self.factory._load_or_skip(
            "coupled_partial.pkl", effective_budget=5,
        )
        self.assertFalse(skip)
        self.assertEqual(resume_state["budget_consumed"], 3.0)

    def test_coupled_file_at_target_iter_count_skips(self):
        self._write_pkl(
            "coupled_done.pkl",
            budget_consumed_list=None,
            acqf_locations=[torch.zeros(1, 2) for _ in range(5)],
        )
        _, resume_state, skip = self.factory._load_or_skip(
            "coupled_done.pkl", effective_budget=5,
        )
        self.assertTrue(skip)
        self.assertIsNone(resume_state)


class TestInitializeState(_TmpCwdMixin, BotorchTestCase):
    """Verifies the two branches of OptimizationLoop._initialize_state."""

    def _make_loop(self, **resume_kwargs):
        bb = MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True,
                                           redundant_constraints=False)
        ncons = bb.get_number_of_constraints()
        model = ConstrainedDeoupledGPModelWrapper(
            num_constraints=ncons, is_noisy=bb.is_noisy(),
        )
        obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(i) for i in range(1, ncons + 1)],
        )
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        return OptimizationLoop(
            black_box_func=bb, model=model, objective=obj,
            ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
            seed=0, budget=10, performance_type="model",
            bounds=bounds, results=Results(filename="x.pkl"),
            number_initial_designs=3,
            **resume_kwargs,
        )

    def test_fresh_path_generates_initial_data(self):
        loop = self._make_loop()
        train_x, train_y, _, budget_consumed = loop._initialize_state()
        self.assertEqual(budget_consumed, 0.0)
        # Mystery (1 constraint) -> 2 outputs
        self.assertEqual(len(train_x), 2)
        for tx in train_x:
            self.assertEqual(tx.shape[0], 3)

    def test_resume_path_returns_provided_state(self):
        initial_x = [torch.rand(5, 2, dtype=torch.double) for _ in range(2)]
        initial_y = [torch.rand(5, dtype=torch.double) for _ in range(2)]
        loop = self._make_loop(
            initial_train_x=initial_x,
            initial_train_y=initial_y,
            initial_budget_consumed=3.5,
        )
        train_x, train_y, _, budget_consumed = loop._initialize_state()
        self.assertEqual(budget_consumed, 3.5)
        self.assertIs(train_x, initial_x)
        self.assertIs(train_y, initial_y)
        for tx in train_x:
            self.assertEqual(tx.shape[0], 5)


class TestEndToEndResume(_TmpCwdMixin, BotorchTestCase):
    """Drive the full launcher path. Uses coupled CEI which works on CPU."""

    def test_skip_then_resume_extends_history(self):
        bb = MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True,
                                           redundant_constraints=False)
        common = dict(
            black_box_function=bb,
            seed=0,
            bayesian_optimization_algorithm=BayesianOptimizationLoopType.CEI,
            number_of_initial_designs=1,
        )
        result_path = "results/mystery_equal_costs__cei0.pkl"

        # 1. Fresh run (budget=2 -> 1 iter for CEI with 1 constraint).
        run_experiment_coupled_acquisition_functions(budget=2, **common)
        self.assertTrue(os.path.exists(result_path))
        with open(result_path, "rb") as f:
            d1 = pickle.load(f)
        n1 = len(d1["acqf_recommended_location"])
        x1 = d1["input_data"][0].shape[0]
        self.assertGreater(n1, 0)

        # 2. Same budget => skip; history must be unchanged.
        run_experiment_coupled_acquisition_functions(budget=2, **common)
        with open(result_path, "rb") as f:
            d2 = pickle.load(f)
        self.assertEqual(len(d2["acqf_recommended_location"]), n1)
        self.assertEqual(d2["input_data"][0].shape[0], x1)

        # 3. Larger budget => resume; history must grow.
        run_experiment_coupled_acquisition_functions(budget=6, **common)
        with open(result_path, "rb") as f:
            d3 = pickle.load(f)
        self.assertGreater(len(d3["acqf_recommended_location"]), n1)
        self.assertGreater(d3["input_data"][0].shape[0], x1)
        self.assertEqual(d3["budget"], 3)  # int(6 / (1 + 1)) = 3
