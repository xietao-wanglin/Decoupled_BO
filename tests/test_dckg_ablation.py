import os
import pickle

import numpy as np
import torch
from botorch.utils.testing import BotorchTestCase

from Launcher import run_experiment_decoupled_acquisition_functions
from bo.bo_loops.bayesian_optimization_loop_type import BayesianOptimizationLoopType
from bo.synthetic_test_functions.synthetic_test_functions import ConstrainedFunc3

torch.set_default_dtype(torch.double)

# Seed kept out of the 0-39 experiment range so the budget=2 pkl files these
# tests write can never be picked up as resume state by a real run.
SMOKE_SEED = 999
N_CONSTRAINTS = 3


class TestDcKGAblationSmokeRun(BotorchTestCase):

    def _run(self, algorithm, suffix):
        path = f"results/test_function_3_equal_cost_{suffix}_{SMOKE_SEED}.pkl"
        if os.path.exists(path):
            os.remove(path)
        run_experiment_decoupled_acquisition_functions(
            black_box_function=ConstrainedFunc3(noise_std=1e-6, negate=True),
            budget=2,
            seed=SMOKE_SEED,
            bayesian_optimization_algorithm=algorithm,
            number_of_initial_designs=1,
            cost=None,
        )
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        os.remove(path)
        return d

    @staticmethod
    def _evaluated_sources(d):
        return [np.atleast_1d(i).astype(int) for i in d["acqf_recommended_output_index:"]]

    def test_no_coupled_candidate(self):
        d = self._run(BayesianOptimizationLoopType.DCKG_NO_COUPLED, "dckg_nocoupled")
        sources = self._evaluated_sources(d)
        self.assertGreater(len(sources), 0)
        # Only real output indices are recorded; no coupled acqf slot leaks through.
        self.assertTrue(all(s.max() <= N_CONSTRAINTS for s in sources))
        # K+1 competing sources instead of the baseline's K+2.
        self.assertEqual(len(d["acqf_values"][0]), N_CONSTRAINTS + 1)

    def test_fully_decoupled(self):
        d = self._run(BayesianOptimizationLoopType.DCKG_PURE, "dckg_pure")
        sources = self._evaluated_sources(d)
        self.assertGreater(len(sources), 0)
        self.assertTrue(all(s.max() <= N_CONSTRAINTS for s in sources))
        self.assertEqual(len(d["acqf_values"][0]), N_CONSTRAINTS + 1)
        # No iteration may evaluate more than one source.
        self.assertTrue(all(s.size == 1 for s in sources))
