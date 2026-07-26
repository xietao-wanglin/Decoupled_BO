import argparse
import itertools
import logging
import random
from typing import Optional

import numpy as np
import torch
from botorch.acquisition import ConstrainedMCObjective

from bo.bo_loops.bayesian_optimization_factory import BayesianOptimizationLoopFactory
from bo.bo_loops.bayesian_optimization_loop_type import BayesianOptimizationLoopType
from bo.model.Model import ConstrainedDeoupledGPModelWrapper, obj_callable, constraint_callable_wrapper
from bo.synthetic_test_functions.bolt_dmo_benchmark import DecoupledBOLTDMO
from bo.synthetic_test_functions.cnn_takena22_benchmark import const_cnn_cifar10
from bo.synthetic_test_functions.synthetic_test_functions import ConstrainedFunc3, ConstrainedBraninNew, \
    MysteryFunctionSuperRedundant, WeldedBeamSO, PressureVessel, TwoLayerCNN_train, SingleObjectiveProblem, \
    TensionCompression, SpeedReducer, BraninHoo, BraninHoo2, BraninHoo3, ConstrainedFunc3Redundant
from bo.device_utils import DEVICE as device, DTYPE as dtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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


def run_experiment_decoupled_acquisition_functions(black_box_function: SingleObjectiveProblem,
                                                   bayesian_optimization_algorithm: BayesianOptimizationLoopType,
                                                   number_of_initial_designs,
                                                   budget=100,
                                                   cost=1,
                                                   seed=1):
    filename_pf = black_box_function.get_name()
    number_of_constraints = black_box_function.get_number_of_constraints()
    if cost is None:
        costs = torch.ones(number_of_constraints + 1)
        cost_label = "_equal_cost"

        model = ConstrainedDeoupledGPModelWrapper(num_constraints=number_of_constraints,
                                                  is_noisy=black_box_function.is_noisy())
        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable_wrapper(idx) for idx in range(1, number_of_constraints + 1)],
        )
        bo_loop_factory = BayesianOptimizationLoopFactory(black_box_function=black_box_function,
                                                          constrained_obj=constrained_obj, model=model, seed=seed,
                                                          budget=budget,
                                                          penalty_value=torch.tensor(
                                                              [black_box_function.get_penalty()]),
                                                          costs=costs, number_of_constraints=number_of_constraints,
                                                          base_file_name=filename_pf + cost_label)

        bo_loop = bo_loop_factory.create(bayesian_optimization_algorithm, number_of_initial_designs)
        if bo_loop is not None:
            bo_loop.run()
    else:
        for i in range(number_of_constraints + 1):
            costs = torch.ones(number_of_constraints + 1)
            costs[i] = cost
            if i == 0:
                cost_label = "_expensive_objective_with_" + str(cost)
            else:
                cost_label = "_expensive_constraint_" + str(i) + "_with_" + str(cost)

            model = ConstrainedDeoupledGPModelWrapper(num_constraints=number_of_constraints,
                                                      is_noisy=black_box_function.is_noisy())
            constrained_obj = ConstrainedMCObjective(
                objective=obj_callable,
                constraints=[constraint_callable_wrapper(idx) for idx in range(1, number_of_constraints + 1)],
            )
            bo_loop_factory = BayesianOptimizationLoopFactory(black_box_function=black_box_function,
                                                              constrained_obj=constrained_obj, model=model, seed=seed,
                                                              budget=budget,
                                                              penalty_value=torch.tensor(
                                                                  [black_box_function.get_penalty()]),
                                                              costs=costs, number_of_constraints=number_of_constraints,
                                                              base_file_name=filename_pf + cost_label)

            bo_loop = bo_loop_factory.create(bayesian_optimization_algorithm, number_of_initial_designs)
            if bo_loop is not None:
                bo_loop.run()


def run_experiment_coupled_acquisition_functions(black_box_function: SingleObjectiveProblem,
                                                 budget,
                                                 seed,
                                                 bayesian_optimization_algorithm,
                                                 number_of_initial_designs):
    filename_pf = black_box_function.get_name()
    number_of_constraints = black_box_function.get_number_of_constraints()
    costs = torch.ones(number_of_constraints + 1)
    cost_label = "_equal_costs_"
    model = ConstrainedDeoupledGPModelWrapper(num_constraints=number_of_constraints,
                                              is_noisy=black_box_function.is_noisy())
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable_wrapper(idx) for idx in range(1, number_of_constraints + 1)],
    )
    bo_loop_factory = BayesianOptimizationLoopFactory(black_box_function=black_box_function,
                                                      constrained_obj=constrained_obj, model=model, seed=seed,
                                                      budget=budget,
                                                      penalty_value=torch.tensor([black_box_function.get_penalty()]),
                                                      costs=costs, number_of_constraints=number_of_constraints,
                                                      base_file_name=filename_pf + cost_label)

    bo_loop = bo_loop_factory.create(bayesian_optimization_algorithm, number_of_initial_designs)
    if bo_loop is not None:
        bo_loop.run()


ABLATION_ALGORITHMS = {
    "nocoupled": [BayesianOptimizationLoopType.DCKG_NO_COUPLED],
    "pure": [BayesianOptimizationLoopType.DCKG_PURE],
    "both": [BayesianOptimizationLoopType.DCKG_NO_COUPLED,
             BayesianOptimizationLoopType.DCKG_PURE],
}


def get_bo_algorithms(decoupled: bool, ablation: Optional[str] = None):
    """Returns the appropriate Bayesian Optimization algorithms based on acquisition function type."""
    if decoupled and ablation:
        return ABLATION_ALGORITHMS[ablation]
    if decoupled:
        return [
            # BayesianOptimizationLoopType.DCKG_ALL_SOURCES,
            BayesianOptimizationLoopType.DCKG_INDEPENDENT,
            # BayesianOptimizationLoopType.DCKG,
            # BayesianOptimizationLoopType.EIKG,
            # BayesianOptimizationLoopType.DEI,
            # BayesianOptimizationLoopType.OPTIMISTIC_UCB
        ]
    return [
        # BayesianOptimizationLoopType.CEI,
        BayesianOptimizationLoopType.CKG_V2,
    ]


# ============================
# HOW TO USE THIS SCRIPT
# ============================

# This script runs Bayesian optimization experiments using different black-box functions
# and acquisition function types (coupled or decoupled).

# 1. Run with a specific function and a seed range:
#    Example:
#        python script.py --function Mystery --min-seed 0 --max-seed 5
#    This will run the experiment using the 'Mystery' function, with seeds from 0 to 5.

# 2. Enable decoupled acquisition functions (default is coupled):
#    Example:
#        python script.py --function Branin --min-seed 0 --max-seed 3 --decoupled
#    This enables decoupled acquisition functions.


# 4. Run coupled acquisition functions explicitly (not required since default is False):
#    Example:
#        python script.py --function MysteryRedundant --min-seed 1 --max-seed 3 --no-decoupled
#    This will run coupled acquisition functions explicitly.

# ============================
# ARGUMENT DETAILS
# ============================
# --function:         The black-box function to use (Mystery, MysteryRedundant, Branin, TestFunc3).
# --min-seed:        Minimum seed value (integer).
# --max-seed:        Maximum seed value (integer, inclusive).
# --decoupled:  If provided, enables decoupled acquisition functions otherwise it runs only coupled algorithms.

# ============================
# SCRIPT ENTRY POINT
# ============================
# The script calls `get_bo_algorithms()` based on `--decoupled`
# and iterates over all parameter combinations using `itertools.product`
# to run the optimization experiments.
# If you need to run any particular subset of bayesian optimization algorithms just
# modify accordingly the lists inside get_bo_algorithms().
# the file runs with the following defaults
# 1. budget = 150
# 2. for different cost acqf: 5
# 3. for same cost acqf: 1
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments with --function --min-seed and --max-seed.")
    parser.add_argument("--function", type=str, choices=["Mystery",
                                                         "MysteryRedundant",
                                                         "Branin",
                                                         "TestFunc3",
                                                         "TestFunc3Redundant",
                                                         "TestFunc3RedundantC1",
                                                         "TestFunc3RedundantC3",
                                                         "TestFunc3RedundantNoNoise",
                                                         "WeldedBeam",
                                                         "BraninHoo",
                                                         "BraninHoo2",
                                                         "BraninHoo3",
                                                         "TensionCompression",
                                                         "PressureVessel",
                                                         "SpeedReducer",
                                                         "two_layer_cnn",
                                                         "two_layer_cnn_discrete",
                                                         "bolt_dmo",
                                                         "bolt_dmo_10"],
                        required=True,
                        help="Choose the function")

    parser.add_argument(
        "--decoupled",
        action="store_true",
        help="Enable decoupled acquisition functions"
    )

    parser.add_argument(
        "--ablation",
        nargs="?",
        const="both",
        default=None,
        choices=["nocoupled", "pure", "both"],
        help="With --decoupled, run a dcKG ablation instead of the default decoupled "
             "algorithms. 'nocoupled' drops the coupled cKG candidate; 'pure' also "
             "replaces the all-zero fallback with a single round-robin source; "
             "'both' (the bare flag) runs the two in sequence."
    )

    parser.add_argument("--min-seed", type=int, default=0,
                        help="Minimum seed value (default: 0)")
    parser.add_argument("--max-seed", type=int, default=39,
                        help="Maximum seed value (inclusive)")

    parser.add_argument("--cost", type=float, default=None,
                        help="Expensive-source cost for decoupled runs. When set, each source "
                             "is made expensive in turn (objective, then each constraint). "
                             "Ignored for coupled runs.")

    args = parser.parse_args()

    # Select the function based on the argument
    if args.function == "Mystery":
        black_box_function = MysteryFunctionSuperRedundant(noise_std=1e-6,
                                                           negate=True,
                                                           redundant_constraints=False)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "MysteryRedundant":
        black_box_function = MysteryFunctionSuperRedundant(noise_std=1e-6,
                                                           negate=True,
                                                           redundant_constraints=True)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "TestFunc3":
        black_box_function = ConstrainedFunc3(noise_std=1e-6,
                                              negate=True)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "TestFunc3Redundant":
        black_box_function = ConstrainedFunc3Redundant(noise_std=0.0,
                                                       negate=True,
                                                       noisy_active_constraint=2)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "TestFunc3RedundantC1":
        black_box_function = ConstrainedFunc3Redundant(noise_std=0.0,
                                                       negate=True,
                                                       noisy_active_constraint=1)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "TestFunc3RedundantC3":
        black_box_function = ConstrainedFunc3Redundant(noise_std=0.0,
                                                       negate=True,
                                                       noisy_active_constraint=3)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "TestFunc3RedundantNoNoise":
        black_box_function = ConstrainedFunc3Redundant(noise_std=0.0,
                                                       negate=True,
                                                       noisy_active_constraint=None)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "Branin":
        black_box_function = ConstrainedBraninNew(noise_std=1e-6,
                                                  negate=True)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "BraninHoo":
        black_box_function = BraninHoo(noise_std=1e-2, negate=True)
        number_initial_designs = 3
        budgets = [160]
    elif args.function == "BraninHoo2":
        black_box_function = BraninHoo2(noise_std=1e-2, negate=True)
        number_initial_designs = 5
        budgets = [160]
    elif args.function == "BraninHoo3":
        black_box_function = BraninHoo3(noise_std=1e-2, negate=True)
        number_initial_designs = 5
        budgets = [160]
    elif args.function == "WeldedBeam":
        black_box_function = WeldedBeamSO(noise_std=1e-6,
                                          negate=True)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "TensionCompression":
        black_box_function = TensionCompression(noise_std=1e-6,
                                                negate=True)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "PressureVessel":
        black_box_function = PressureVessel(noise_std=1e-6,
                                            negate=True)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "SpeedReducer":
        black_box_function = SpeedReducer(noise_std=1e-6,
                                          negate=True)
        number_initial_designs = 6
        budgets = [160]
    elif args.function == "two_layer_cnn_discrete":
        black_box_function = const_cnn_cifar10(negate=False)
        number_initial_designs = 30
        budgets = [300]
    elif args.function == "bolt_dmo":
        black_box_function = DecoupledBOLTDMO(negate=False)
        number_initial_designs = 30
        budgets = [300]
    elif args.function == "bolt_dmo_10":
        # Quantiles 0.775/0.775 give a ~10% joint feasible rate on the reference
        # set (vs ~20% for the default 0.6/0.6).
        black_box_function = DecoupledBOLTDMO(negate=False,
                                              quantile_if=0.775,
                                              quantile_mbpp=0.775)
        number_initial_designs = 30
        budgets = [300]
    else:
        raise ValueError(f"Function {args.function} is not supported.")

    # Parameters
    costs = [args.cost]
    seeds = list(range(args.min_seed, args.max_seed + 1))
    bayesian_optimization_algorithms = get_bo_algorithms(decoupled=args.decoupled,
                                                         ablation=args.ablation)

    # Run experiments
    for bayesian_optimization_algorithm, budget, cost, seed in itertools.product(bayesian_optimization_algorithms,
                                                                                 budgets, costs, seeds):
        logging.info(
            f"Running experiment | Algorithm: {bayesian_optimization_algorithm.name}, Budget: {budget}, "
            f"Seed: {seed}, Cost: {cost if args.decoupled else 'N/A'}, "
            f"Decoupled: {args.decoupled}"
        )

        set_all_seeds(seed)

        experiment_fn = (
            run_experiment_decoupled_acquisition_functions
            if args.decoupled
            else run_experiment_coupled_acquisition_functions
        )

        experiment_args = {
            "black_box_function": black_box_function,
            "budget": budget,
            "seed": seed,
            "bayesian_optimization_algorithm": bayesian_optimization_algorithm,
            "number_of_initial_designs": number_initial_designs
        }

        if args.decoupled:
            experiment_args["cost"] = cost  # Only needed for decoupled case

        experiment_fn(**experiment_args)
