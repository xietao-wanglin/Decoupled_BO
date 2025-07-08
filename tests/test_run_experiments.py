import torch
from botorch.utils.testing import BotorchTestCase

from Launcher import run_experiment_coupled_acquisition_functions, run_experiment_decoupled_acquisition_functions
from bo.bo_loops.bayesian_optimization_loop_type import BayesianOptimizationLoopType
from bo.synthetic_test_functions.synthetic_test_functions import MysteryFunctionSuperRedundant, \
    ConstrainedFunc3, ConstrainedBraninNew, WeldedBeamSO, PressureVessel, TwoLayerCNN_train

device = torch.device("cpu")
dtype = torch.double
torch.set_default_dtype(dtype)


class TestCoupledAcquisitionFunctions(BotorchTestCase):

    def test_mystery(self):
        black_box_function = MysteryFunctionSuperRedundant(noise_std=1e-6,
                                                           negate=True,
                                                           redundant_constraints=False)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.CEI, BayesianOptimizationLoopType.CKG]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1}
            run_experiment_coupled_acquisition_functions(**experiment_args)

    def test_MysteryRedundant(self):
        black_box_function = MysteryFunctionSuperRedundant(noise_std=1e-6,
                                                           negate=True,
                                                           redundant_constraints=True)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.CEI, BayesianOptimizationLoopType.CKG]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1}
            run_experiment_coupled_acquisition_functions(**experiment_args)

    def test_TestFunc3(self):
        black_box_function = ConstrainedFunc3(noise_std=1e-6,
                                              negate=True)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.CEI, BayesianOptimizationLoopType.CKG]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1}
            run_experiment_coupled_acquisition_functions(**experiment_args)

    def test_ConstrainedBraninNew(self):
        black_box_function = ConstrainedBraninNew(noise_std=1e-6,
                                                  negate=True)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.CEI, BayesianOptimizationLoopType.CKG]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1}
            run_experiment_coupled_acquisition_functions(**experiment_args)

    def test_WeldedBeamSO(self):
        black_box_function = WeldedBeamSO(noise_std=1e-6,
                                          negate=True)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.CEI, BayesianOptimizationLoopType.CKG]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1}
            run_experiment_coupled_acquisition_functions(**experiment_args)

    def test_PressureVessel(self):
        black_box_function = PressureVessel(noise_std=1e-6,
                                            negate=True)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.CEI, BayesianOptimizationLoopType.CKG]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1}
            run_experiment_coupled_acquisition_functions(**experiment_args)

    def test_TwoLayerCNN(self):
        black_box_function = TwoLayerCNN_train(negate=False)
        black_box_function.number_of_epochs = 1
        black_box_function.samples_per_class = {i: 5 if i < 5 else 5 for i in range(10)}

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.CEI, BayesianOptimizationLoopType.CKG]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1}
            run_experiment_coupled_acquisition_functions(**experiment_args)


class TestDecoupledAcquisitionFunctions(BotorchTestCase):
    def test_mystery(self):
        black_box_function = MysteryFunctionSuperRedundant(noise_std=1e-6,
                                                           negate=True,
                                                           redundant_constraints=False)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.DCKG_CKG,
                                                BayesianOptimizationLoopType.DCKG,
                                                BayesianOptimizationLoopType.EIKG,
                                                BayesianOptimizationLoopType.DEI]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1,
                               "cost": None}
            run_experiment_decoupled_acquisition_functions(**experiment_args)

    def test_MysteryRedundant(self):
        black_box_function = MysteryFunctionSuperRedundant(noise_std=1e-6,
                                                           negate=True,
                                                           redundant_constraints=True)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.DCKG_CKG,
                                                BayesianOptimizationLoopType.DCKG,
                                                BayesianOptimizationLoopType.EIKG,
                                                BayesianOptimizationLoopType.DEI]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1,
                               "cost": None}
            run_experiment_decoupled_acquisition_functions(**experiment_args)

    def test_TestFunc3(self):
        black_box_function = ConstrainedFunc3(noise_std=1e-6,
                                              negate=True)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.DCKG_CKG,
                                                BayesianOptimizationLoopType.DCKG,
                                                BayesianOptimizationLoopType.EIKG,
                                                BayesianOptimizationLoopType.DEI]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1,
                               "cost": None}
            run_experiment_decoupled_acquisition_functions(**experiment_args)

    def test_ConstrainedBraninNew(self):
        black_box_function = ConstrainedBraninNew(noise_std=1e-6,
                                                  negate=True)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.DCKG_CKG,
                                                BayesianOptimizationLoopType.DCKG,
                                                BayesianOptimizationLoopType.EIKG,
                                                BayesianOptimizationLoopType.DEI]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1,
                               "cost": None}
            run_experiment_decoupled_acquisition_functions(**experiment_args)

    def test_WeldedBeamSO(self):
        black_box_function = WeldedBeamSO(noise_std=1e-6,
                                          negate=True)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.DCKG_CKG,
                                                BayesianOptimizationLoopType.DCKG,
                                                BayesianOptimizationLoopType.EIKG,
                                                BayesianOptimizationLoopType.DEI]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1,
                               "cost": None}
            run_experiment_decoupled_acquisition_functions(**experiment_args)

    def test_PressureVessel(self):
        black_box_function = PressureVessel(noise_std=1e-6,
                                            negate=True)

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.DCKG_CKG,
                                                BayesianOptimizationLoopType.DCKG,
                                                BayesianOptimizationLoopType.EIKG,
                                                BayesianOptimizationLoopType.DEI]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1,
                               "cost": None}
            run_experiment_decoupled_acquisition_functions(**experiment_args)

    def test_TwoLayerCNN(self):
        black_box_function = TwoLayerCNN_train(negate=False)
        black_box_function.number_of_epochs = 1
        black_box_function.samples_per_class = {i: 5 if i < 5 else 5 for i in range(10)}

        for bayesian_optimization_algorithm in [BayesianOptimizationLoopType.DCKG_CKG,
                                                BayesianOptimizationLoopType.DCKG,
                                                BayesianOptimizationLoopType.EIKG,
                                                BayesianOptimizationLoopType.DEI]:
            experiment_args = {"black_box_function": black_box_function,
                               "budget": 2,
                               "seed": 0,
                               "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
                               "number_of_initial_designs": 1,
                               "cost": None}
            run_experiment_decoupled_acquisition_functions(**experiment_args)
