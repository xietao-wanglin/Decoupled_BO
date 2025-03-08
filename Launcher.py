import argparse
import itertools
import logging

import torch
from botorch.acquisition import ConstrainedMCObjective

from bo.bo_loops.bayesian_optimization_factory import BayesianOptimizationLoopFactory
from bo.bo_loops.bayesian_optimization_loop_type import BayesianOptimizationLoopType
from bo.model.Model import ConstrainedDeoupledGPModelWrapper, obj_callable, constraint_callable_wrapper
from bo.synthetic_test_functions.synthetic_test_functions import ConstrainedFunc3, ConstrainedBraninNew, \
    MysteryFunctionSuperRedundant

device = torch.device("cpu")
dtype = torch.double
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_experiment_decoupled_acquisition_functions(black_box_function,
                                                   bayesian_optimization_algorithm: BayesianOptimizationLoopType,
                                                   budget=100,
                                                   cost=1,
                                                   seed=1):
    filename_pf = black_box_function.get_name()
    number_of_constraints = black_box_function.get_number_of_constraints()
    costs = torch.ones(number_of_constraints + 1)
    for i in range(number_of_constraints + 1):
        costs[i] = cost
        if i == 0:
            cost_label = "_expensive_objective_with_" + str(cost)
        else:
            cost_label = "_expensive_constraint_" + str(i) + "_with_" + str(cost)

        model = ConstrainedDeoupledGPModelWrapper(num_constraints=number_of_constraints)
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

        bo_loop = bo_loop_factory.create(bayesian_optimization_algorithm)
        bo_loop.run()


def run_experiment_coupled_acquisition_functions(black_box_function, budget, seed,
                                                 bayesian_optimization_algorithm):
    filename_pf = black_box_function.get_name()
    number_of_constraints = black_box_function.get_number_of_constraints()
    costs = torch.ones(number_of_constraints + 1)
    cost_label = "_equal_costs_"
    model = ConstrainedDeoupledGPModelWrapper(num_constraints=number_of_constraints)
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

    bo_loop = bo_loop_factory.create(bayesian_optimization_algorithm)
    bo_loop.run()


def get_bo_algorithms(decoupled: bool):
    """Returns the appropriate Bayesian Optimization algorithms based on acquisition function type."""
    if decoupled:
        return [
            BayesianOptimizationLoopType.DCKG_CKG,
            BayesianOptimizationLoopType.DCKG,
            BayesianOptimizationLoopType.DEI,
            BayesianOptimizationLoopType.EIKG,
        ]
    return [
        BayesianOptimizationLoopType.CEI,
        BayesianOptimizationLoopType.CKG,
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
    parser = argparse.ArgumentParser(description="Run experiments with --function --min-seed and --max--seed.")
    parser.add_argument("--function", type=str, choices=["Mystery",
                                                         "MysteryRedundant",
                                                         "Branin",
                                                         "TestFunc3"], required=True,
                        help="Choose the function: Mystery, MysteryRedundant, TestFunc3 or Branin")

    parser.add_argument(
        "--decoupled",
        action="store_true",
        help="Enable decoupled acquisition functions"
    )

    parser.add_argument("--min-seed", type=int, default=0,
                        help="Minimum seed value (default: 0)")
    parser.add_argument("--max-seed", type=int, required=True,
                        help="Maximum seed value (inclusive)")

    args = parser.parse_args()

    # Select the function based on the argument
    if args.function == "Mystery":
        black_box_function = MysteryFunctionSuperRedundant(noise_std=1e-6,
                                                           negate=True,
                                                           redundant_constraints=False)
    elif args.function == "MysteryRedundant":
        black_box_function = MysteryFunctionSuperRedundant(noise_std=1e-6,
                                                           negate=True,
                                                           redundant_constraints=True)
    elif args.function == "TestFunc3":
        black_box_function = ConstrainedFunc3(noise_std=1e-6,
                                              negate=True)
    elif args.function == "Branin":
        black_box_function = ConstrainedBraninNew(noise_std=1e-6,
                                                  negate=True)
    else:
        raise ValueError(f"Function {args.function} is not supported.")

    # Parameters
    budgets = [150]
    costs = [5]
    seeds = list(range(args.min_seed, args.max_seed + 1))
    bayesian_optimization_algorithms = get_bo_algorithms(decoupled=args.decoupled)

    # Run experiments
    for bayesian_optimization_algorithm, budget, cost, seed in itertools.product(bayesian_optimization_algorithms,
                                                                                 budgets, costs, seeds):

        logging.info(
            f"Running experiment | Algorithm: {bayesian_optimization_algorithm.name}, Budget: {budget}, "
            f"Seed: {seed}, Cost: {cost if args.decoupled else 'N/A'}, "
            f"Decoupled: {args.decoupled}"
        )
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
        }

        if args.decoupled:
            experiment_args["cost"] = cost  # Only needed for decoupled case

        experiment_fn(**experiment_args)