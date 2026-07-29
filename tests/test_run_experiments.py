import glob
import os

import torch
from botorch.utils.testing import BotorchTestCase

from Launcher import run_experiment_coupled_acquisition_functions, run_experiment_decoupled_acquisition_functions
from bo.bo_loops.bayesian_optimization_loop_type import BayesianOptimizationLoopType
from bo.synthetic_test_functions.cnn_takena22_benchmark import const_cnn_cifar10
from bo.synthetic_test_functions.synthetic_test_functions import MysteryFunctionSuperRedundant, \
    ConstrainedFunc3, ConstrainedBraninNew, PressureVessel, SpeedReducer

device = torch.device("cpu")
dtype = torch.double
torch.set_default_dtype(dtype)

# Seed kept out of the 0-39 experiment range so the budget=2 pkl files these tests
# write can never be picked up as resume state by a real run.
SMOKE_SEED = 999
BUDGET = 2
N_INITIAL_DESIGNS = 1

COUPLED_ALGORITHMS = [BayesianOptimizationLoopType.CEI,
                      BayesianOptimizationLoopType.CKG_V2]
DECOUPLED_ALGORITHMS = [BayesianOptimizationLoopType.DCKG_INDEPENDENT]


def _cleanup(black_box_function):
    for path in glob.glob(f"results/{black_box_function.get_name()}*{SMOKE_SEED}.pkl"):
        os.remove(path)


class TestCoupledAcquisitionFunctions(BotorchTestCase):

    def _run(self, black_box_function):
        _cleanup(black_box_function)
        try:
            for algorithm in COUPLED_ALGORITHMS:
                run_experiment_coupled_acquisition_functions(
                    black_box_function=black_box_function,
                    budget=BUDGET,
                    seed=SMOKE_SEED,
                    bayesian_optimization_algorithm=algorithm,
                    number_of_initial_designs=N_INITIAL_DESIGNS)
        finally:
            _cleanup(black_box_function)

    def test_mystery(self):
        self._run(MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True,
                                                redundant_constraints=False))

    def test_MysteryRedundant(self):
        self._run(MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True,
                                                redundant_constraints=True))

    def test_TestFunc3(self):
        self._run(ConstrainedFunc3(noise_std=1e-6, negate=True))

    def test_ConstrainedBraninNew(self):
        self._run(ConstrainedBraninNew(noise_std=1e-6, negate=True))

    def test_PressureVessel(self):
        self._run(PressureVessel(noise_std=1e-6, negate=True))

    def test_SpeedReducer(self):
        self._run(SpeedReducer(noise_std=1e-6, negate=True))

    def test_cnn(self):
        self._run(const_cnn_cifar10(negate=False))


class TestDecoupledAcquisitionFunctions(BotorchTestCase):

    def _run(self, black_box_function):
        _cleanup(black_box_function)
        try:
            for algorithm in DECOUPLED_ALGORITHMS:
                run_experiment_decoupled_acquisition_functions(
                    black_box_function=black_box_function,
                    budget=BUDGET,
                    seed=SMOKE_SEED,
                    bayesian_optimization_algorithm=algorithm,
                    number_of_initial_designs=N_INITIAL_DESIGNS,
                    cost=None)
        finally:
            _cleanup(black_box_function)

    def test_mystery(self):
        self._run(MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True,
                                                redundant_constraints=False))

    def test_MysteryRedundant(self):
        self._run(MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True,
                                                redundant_constraints=True))

    def test_TestFunc3(self):
        self._run(ConstrainedFunc3(noise_std=1e-6, negate=True))

    def test_ConstrainedBraninNew(self):
        self._run(ConstrainedBraninNew(noise_std=1e-6, negate=True))

    def test_PressureVessel(self):
        self._run(PressureVessel(noise_std=1e-6, negate=True))

    def test_SpeedReducer(self):
        self._run(SpeedReducer(noise_std=1e-6, negate=True))

    def test_cnn(self):
        self._run(const_cnn_cifar10(negate=False))
