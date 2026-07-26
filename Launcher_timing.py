"""CPU timing benchmark for the dcKG and cEI acquisitions on the Mystery benchmark.

Measures three quantities as a function of the training-set size ``n_train``
(the same number of observations is used for the objective GP and for the
constraint GP):

1. **Steady-state forward evaluation** of each acquisition (``acqf(X)`` under
   ``no_grad`` after warm-up forwards have built the GP posterior caches).
2. **Complete acquisition optimisation** of each acquisition separately: one
   timer around the production optimisation call, which covers raw-sample
   generation, restart initialisation, the adaptive discretisation, every
   forward/backward pass of every L-BFGS-B restart, and the selection of the
   best candidate. GP fitting and acquisition-object construction happen before
   the timer starts and are excluded.
3. **One complete sequential BO iteration**, measured with a single outer
   wall-clock timer -- separately for the decoupled dcKG algorithm and for the
   coupled cEI baseline. The dcKG iteration covers GP fitting, acquisition
   construction, sequential optimisation of the three dcKG acquisitions, cost
   normalisation and source selection (including the delta feasibility rule for
   the coupled proposal), evaluation of the selected Mystery source(s), and the
   dataset update. The cEI iteration covers GP fitting, acquisition
   construction, optimisation, the coupled evaluation of every output, and the
   dataset update. Neither total is reconstructed by summing component medians.

Mystery has K = 1 constraint, so ``AllSourcesDcKG.sources`` is exactly
``[ObjectiveDcKG, ConstraintDcKG(0), CoupledCKG]`` -- the three acquisitions
named ``dckg_objective``, ``dckg_constraint`` and ``coupled_ckg``. ``cei`` is
the coupled constrained-EI baseline, which is optimised on a single candidate
(q = 1) rather than on a candidate plus discretisation (q = 65).

Everything runs on CPU with the production acquisition implementations and
their default settings (see the ``PROD_*`` constants, which mirror
``IndependentSourcesOptimizationLoop.run`` and ``EI_OptimizationLoop.run`` in
``bo/bo_loops/bo_loop.py``).

Usage:
    python Launcher_timing.py                # full benchmark
    python Launcher_timing.py --smoke-test   # minimal end-to-end check

Outputs (in ``results/timing_study/``): ``mystery_forward_raw.csv``,
``mystery_forward_summary.csv``, ``mystery_opt_raw.csv``,
``mystery_opt_summary.csv``, ``mystery_bo_loop_raw.csv``,
``mystery_bo_loop_summary.csv``, ``mystery_cei_loop_raw.csv``,
``mystery_cei_loop_summary.csv`` and ``mystery_timing_metadata.json``.
Render the paper table with ``python timing_table.py``.
"""
import os

# CPU-only benchmark. This must run before torch (and any bo module) is
# imported: bo.device_utils resolves DEVICE at import time.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
import json
import logging
import statistics
import time
import traceback
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BENCHMARK = "Mystery"
DEVICE_NAME = "cpu"

N_TRAIN_VALUES = [10, 40, 160, 300]

# Order matches AllSourcesDcKG.sources for a single-constraint problem:
# [ObjectiveDcKG, ConstraintDcKG(0), CoupledCKG].
DCKG_ACQUISITIONS = ["dckg_objective", "dckg_constraint", "coupled_ckg"]
CEI = "cei"
ACQUISITIONS = DCKG_ACQUISITIONS + [CEI]

# Production settings of the dcKG loop (IndependentSourcesOptimizationLoop.run).
PROD_N_FANTASIES = 7             # bo_loop.py:1496
PROD_N_CONSTRAINT_SAMPLES = 5    # bo_loop.py:1497
PROD_N_DISC_ACQF = 128           # bo_loop.py:1498 (acquisition construction)
PROD_N_DISC_OPT = 64             # bo_loop.py:1442 (what the optimiser actually uses)
PROD_Q = 1 + PROD_N_DISC_OPT     # candidate + discretisation slots per forward
PROD_NUM_RESTARTS = 15           # bo_loop.py:1510
PROD_RAW_SAMPLES = 72            # bo_loop.py:1510
PROD_INFEASIBILITY_THRESHOLD = 1e-7  # delta in compute_important_idxs, bo_loop.py:251

# Production settings of the cEI loop (EI_OptimizationLoop.compute_next_sample).
PROD_CEI_Q = 1                   # bo_loop.py:976
PROD_CEI_NUM_RESTARTS = 15       # bo_loop.py:977
PROD_CEI_RAW_SAMPLES = 72        # bo_loop.py:978

# cEI evaluates a single candidate; the dcKG acquisitions evaluate a candidate
# plus a 64-point discretisation. Per-call forward times are not like-for-like.
ACQUISITION_Q = {name: PROD_Q for name in DCKG_ACQUISITIONS}
ACQUISITION_Q[CEI] = PROD_CEI_Q

WARMUP_SEED_OFFSET = 10_000

FORWARD_COLUMNS = ["benchmark", "device", "dtype", "threads", "n_train", "acquisition",
                   "repeat", "seed", "q", "forward_seconds",
                   "status", "error_type", "error_message"]

OPT_COLUMNS = ["benchmark", "device", "dtype", "threads", "n_train", "acquisition",
               "repeat", "seed", "opt_seed", "optimise_seconds", "acquisition_value",
               "status", "error_type", "error_message"]

LOOP_COLUMNS = ["benchmark", "device", "dtype", "threads", "n_train", "repeat", "seed",
                "fit_models_seconds", "construct_acquisitions_seconds",
                "optimise_dckg_objective_seconds", "optimise_dckg_constraint_seconds",
                "optimise_coupled_ckg_seconds", "select_action_seconds",
                "evaluate_sources_seconds", "update_datasets_seconds",
                "total_loop_seconds",
                "selected_action", "selected_location", "selected_sources",
                "objective_acquisition_value", "constraint_acquisition_value",
                "coupled_acquisition_value",
                "n_objective_before", "n_constraint_before",
                "n_objective_after", "n_constraint_after",
                "status", "error_type", "error_message"]

CEI_LOOP_COLUMNS = ["benchmark", "device", "dtype", "threads", "n_train", "repeat", "seed",
                    "fit_models_seconds", "construct_acquisition_seconds",
                    "optimise_cei_seconds", "evaluate_sources_seconds",
                    "update_datasets_seconds", "total_loop_seconds",
                    "selected_location", "selected_sources", "cei_acquisition_value",
                    "n_objective_before", "n_constraint_before",
                    "n_objective_after", "n_constraint_after",
                    "status", "error_type", "error_message"]

LOOP_DEFINITION = (
    "fit the objective and constraint GPs -> recommend x_best -> construct dckg_objective, "
    "dckg_constraint and coupled_ckg -> optimise them sequentially in that order -> cost "
    "normalisation, delta feasibility rule and source selection -> evaluate the selected "
    "Mystery source(s) at the selected location -> append the observation(s) to the "
    "corresponding dataset(s)"
)

CEI_LOOP_DEFINITION = (
    "fit the objective and constraint GPs -> recommend x_best (supplies the incumbent value) -> "
    "construct the constrained EI acquisition -> optimise it (production two-stage "
    "optimize_acqf: raw-sample restarts plus the smart-initialisation restart, best of the two) "
    "-> evaluate every Mystery output at the selected location (coupled) -> append one "
    "observation to every dataset"
)

GP_REFIT_STRATEGY = "refit_from_scratch"
GP_REFIT_STRATEGY_DETAIL = (
    "ConstrainedDeoupledGPModelWrapper.fit builds new SingleTaskGP objects on every call "
    "(bo/model/Model.py:338-360) and optimize() runs fit_gpytorch_mll on a fresh "
    "SumMarginalLogLikelihood (bo/model/Model.py:362-367): the hyperparameters are re-optimised "
    "from their default initialisation every iteration, with no warm start and no cache-only "
    "posterior refresh. This holds for both the dcKG and the cEI loop."
)

RECOMMENDATION_STEP = (
    "compute_best_posterior_mean (optimize_acqf with num_restarts=20, raw_samples=2048) is "
    "required to build the acquisitions -- it supplies x_best for dcKG and the incumbent value "
    "for cEI. It runs inside the outer loop timer and its duration is attributed to "
    "construct_acquisitions_seconds (dcKG) / construct_acquisition_seconds (cEI)."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU timing benchmark for the dcKG and cEI acquisitions on Mystery.")
    parser.add_argument("--n-train", type=int, nargs="+", default=N_TRAIN_VALUES,
                        help=f"Training-set sizes per GP (default: {N_TRAIN_VALUES})")
    parser.add_argument("--forward-warmups", type=int, default=3,
                        help="Untimed forward calls per acquisition (default: 3)")
    parser.add_argument("--forward-repeats", type=int, default=20,
                        help="Timed forward calls per acquisition (default: 20)")
    parser.add_argument("--opt-warmups", type=int, default=1,
                        help="Untimed acquisition optimisations per acquisition (default: 1)")
    parser.add_argument("--opt-repeats", type=int, default=10,
                        help="Timed acquisition optimisations per acquisition (default: 10)")
    parser.add_argument("--loop-warmups", type=int, default=3,
                        help="Untimed sequential BO iterations, per algorithm (default: 3)")
    parser.add_argument("--loop-repeats", type=int, default=10,
                        help="Timed sequential BO iterations, per algorithm (default: 10)")
    parser.add_argument("--data-seed", type=int, default=0,
                        help="Seed of the fixed base dataset (default: 0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base optimisation seed; repetition r uses seed + r (default: 0)")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch.set_num_threads value (default: leave the torch default)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Minimal end-to-end run: n_train=10, 1 forward warm-up + 2 timed "
                             "forwards, 1 optimisation, 1 loop warm-up + 1 timed loop.")
    parser.add_argument("--output-dir", type=str, default=os.path.join("results", "timing_study"),
                        help="Output directory (default: results/timing_study)")
    args = parser.parse_args()
    if args.smoke_test:
        args.n_train = [10]
        args.forward_warmups, args.forward_repeats = 1, 2
        args.opt_warmups, args.opt_repeats = 0, 1
        args.loop_warmups, args.loop_repeats = 1, 1
    return args


# ----------------------------------------------------------------------------
# Production objects
# ----------------------------------------------------------------------------

def make_mystery():
    from bo.synthetic_test_functions.synthetic_test_functions import MysteryFunctionSuperRedundant
    return MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True, redundant_constraints=False)


def _make_loop(loop_class, ei_type, black_box_function, seed):
    """Build the loop object BayesianOptimizationLoopFactory.create() would build,
    without touching any results file on disk. ``run()`` is never called."""
    import torch
    from botorch.acquisition import ConstrainedMCObjective

    from bo.device_utils import DEVICE as device, DTYPE as dtype
    from bo.model.Model import (ConstrainedDeoupledGPModelWrapper, obj_callable,
                                constraint_callable_wrapper)
    from bo.result_utils.result_container import Results

    number_of_constraints = black_box_function.get_number_of_constraints()
    model = ConstrainedDeoupledGPModelWrapper(num_constraints=number_of_constraints,
                                              is_noisy=black_box_function.is_noisy())
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable_wrapper(idx) for idx in range(1, number_of_constraints + 1)],
    )
    bounds = torch.zeros(2, black_box_function.dim, device=device, dtype=dtype)
    bounds[1] = 1.0
    # Same double-wrapping as Launcher.py -> BayesianOptimizationLoopFactory.create()
    penalty_value = torch.tensor([torch.tensor([black_box_function.get_penalty()])])

    return loop_class(
        black_box_func=black_box_function,
        objective=constrained_obj,
        bounds=bounds,
        performance_type="model",
        model=model,
        ei_type=ei_type,
        seed=seed,
        budget=1,  # never used: we do not call run()
        number_initial_designs=6,
        results=Results(filename="timing_study_dummy.pkl"),  # never serialised
        costs=torch.ones(number_of_constraints + 1),
        penalty_value=penalty_value,
    )


def make_loop(black_box_function, seed):
    """The production decoupled dcKG loop (DCKG_INDEPENDENT)."""
    from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType
    from bo.bo_loops.bo_loop import IndependentSourcesOptimizationLoop
    return _make_loop(IndependentSourcesOptimizationLoop,
                      AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2,
                      black_box_function, seed)


def make_cei_loop(black_box_function, seed):
    """The production coupled cEI loop (CEI)."""
    from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType
    from bo.bo_loops.bo_loop import EI_OptimizationLoop
    return _make_loop(EI_OptimizationLoop,
                      AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                      black_box_function, seed)


def make_base_state(loop, n_train):
    """Fixed base datasets: n_train observations for the objective and for each
    constraint. Always built outside every timer."""
    return loop.generate_initial_data(n=n_train)


def build_acquisitions(loop, model, seed):
    """Production dcKG acquisition construction: recommend x_best, then build the
    K+2 sources (objective dcKG, constraint dcKG, coupled cKG)."""
    from bo.acquisition_functions.refactored_acquisition_functions import AllSourcesDcKG

    x_best, _ = loop.best_observed(
        best_value_computation_type=loop.performance_type,
        train_x=None, train_y=None, model=model, bounds=loop.bounds)
    all_dckg = AllSourcesDcKG(
        model,
        penalty_value=loop.penalty_value,
        x_best_location=x_best,
        objective=loop.objective,
        n_fantasies=PROD_N_FANTASIES,
        n_constraint_samples=PROD_N_CONSTRAINT_SAMPLES,
        n_disc=PROD_N_DISC_ACQF,
        seed=seed,
    )
    return x_best, all_dckg


def build_cei_acquisition(loop, model, iteration):
    """Production cEI acquisition construction: recommend x_best (which supplies
    the incumbent value), then build the acquisition (bo_loop.py:879-895)."""
    from bo.acquisition_functions.acquisition_functions import acquisition_function_factory

    x_best, best_value = loop.best_observed(
        best_value_computation_type=loop.performance_type,
        train_x=None, train_y=None, model=model, bounds=loop.bounds)
    acquisition_function = acquisition_function_factory(
        model=model,
        type=loop.acquisition_function_type,
        objective=loop.objective,
        best_value=best_value,
        idx=1,
        number_of_outputs=loop.number_of_outputs,
        penalty_value=loop.penalty_value,
        iteration=iteration,
        initial_condition_internal_optimizer=x_best)
    return x_best, acquisition_function


def optimise_cei(loop, model, acquisition_function, x_best):
    """The complete production cEI optimisation (bo_loop.py:896-898): smart
    restart initialisation followed by the two-stage optimize_acqf, returning the
    better of the raw-sample restarts and the smart-initialisation restart."""
    initialization = loop.get_smart_initialization(acquisition_function, model, x_best)
    return loop.compute_next_sample(acquisition_function=acquisition_function,
                                    smart_initial_locations=initialization)


def prepare_state(black_box_function, n_train, seed):
    """Untimed prelude shared by the forward and optimisation measurements:
    fixed dataset -> fitted GPs -> constructed acquisitions."""
    from Launcher import set_all_seeds

    set_all_seeds(seed)
    loop = make_loop(black_box_function, seed)
    train_x, train_y = make_base_state(loop, n_train)
    model = loop.update_model(X=train_x, y=train_y)
    _, all_dckg = build_acquisitions(loop, model, seed)
    return loop, model, all_dckg


def prepare_cei_state(black_box_function, n_train, seed):
    """Untimed prelude for the cEI forward and optimisation measurements."""
    from Launcher import set_all_seeds

    set_all_seeds(seed)
    loop = make_cei_loop(black_box_function, seed)
    train_x, train_y = make_base_state(loop, n_train)
    model = loop.update_model(X=train_x, y=train_y)
    x_best, acquisition_function = build_cei_acquisition(loop, model, iteration=seed)
    return loop, model, x_best, acquisition_function


# ----------------------------------------------------------------------------
# Measurement 1: steady-state forward evaluation
# ----------------------------------------------------------------------------

def _time_forwards(acquisition_function, X, warmups, repeats):
    """Individually timed steady-state forwards, after untimed warm-ups."""
    import torch
    elapsed = []
    with torch.no_grad():
        for _ in range(warmups):
            acquisition_function(X)
        for _ in range(repeats):
            t0 = time.perf_counter()
            acquisition_function(X)
            elapsed.append(time.perf_counter() - t0)
    return elapsed


def measure_forward(black_box_function, n_train, args, base_record):
    """One timed row per (acquisition, repeat): a single steady-state acqf(X).

    The dcKG acquisitions are evaluated on their production candidate tensor
    (q = 65: candidate + discretisation); cEI on its production q = 1.
    """
    import torch
    from bo.device_utils import DEVICE, DTYPE

    rows = []

    def _row(name, repeat, **fields):
        return dict(base_record, n_train=n_train, acquisition=name, repeat=repeat,
                    seed=args.seed, q=ACQUISITION_Q[name], **fields)

    # --- dcKG sources -------------------------------------------------------
    try:
        loop, _, all_dckg = prepare_state(black_box_function, n_train, args.seed)
        named_acquisitions = list(zip(DCKG_ACQUISITIONS, all_dckg.sources))
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        named_acquisitions = []
        rows += [_row(name, 0, forward_seconds="", **_error_fields(exc))
                 for name in DCKG_ACQUISITIONS]

    # --- cEI ----------------------------------------------------------------
    try:
        cei_loop, _, _, cei_acqf = prepare_cei_state(black_box_function, n_train, args.seed)
        named_acquisitions.append((CEI, cei_acqf))
        dim_x = cei_loop.dim_x
    except Exception as exc:  # noqa: BLE001
        rows.append(_row(CEI, 0, forward_seconds="", **_error_fields(exc)))
        dim_x = None

    for name, acquisition_function in named_acquisitions:
        dim = dim_x if dim_x is not None else loop.dim_x
        X = torch.rand(1, ACQUISITION_Q[name], dim, device=DEVICE, dtype=DTYPE)
        try:
            elapsed = _time_forwards(acquisition_function, X,
                                     args.forward_warmups, args.forward_repeats)
            rows += [_row(name, repeat, forward_seconds=value,
                          status="ok", error_type="", error_message="")
                     for repeat, value in enumerate(elapsed)]
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(name, 0, forward_seconds="", **_error_fields(exc)))
        logging.info(f"  forward n_train={n_train} {name} (q={ACQUISITION_Q[name]}): "
                     f"{_median_ms(rows, name):.3f} ms (median)")
    return rows


# ----------------------------------------------------------------------------
# Measurement 2: complete acquisition optimisation
# ----------------------------------------------------------------------------

def measure_optimisation(black_box_function, n_train, args, base_record):
    """One timed row per (acquisition, repeat) around the complete production
    optimisation call. GP fitting and acquisition construction are excluded:
    they happen in the untimed prelude."""
    from Launcher import set_all_seeds
    from bo.bo_loops.bo_loop import IndependentSourcesOptimizationLoop

    rows = []

    def _row(name, repeat, opt_seed, **fields):
        return dict(base_record, n_train=n_train, acquisition=name, repeat=repeat,
                    seed=args.seed, opt_seed=opt_seed, **fields)

    def _repeats():
        """(repeat, is_warmup) over the warm-ups followed by the timed repetitions."""
        return [(r, r < 0) for r in range(-args.opt_warmups, args.opt_repeats)]

    # --- dcKG sources -------------------------------------------------------
    try:
        loop, _, all_dckg = prepare_state(black_box_function, n_train, args.seed)
    except Exception as exc:  # noqa: BLE001
        rows += [_row(name, 0, "", optimise_seconds="", acquisition_value="",
                      **_error_fields(exc)) for name in DCKG_ACQUISITIONS]
        all_dckg = None

    if all_dckg is not None:
        for source_index, (name, source_acqf) in enumerate(zip(DCKG_ACQUISITIONS, all_dckg.sources)):
            for repeat, is_warmup in _repeats():
                opt_seed = (args.seed + max(repeat, 0) + source_index * 100
                            + (WARMUP_SEED_OFFSET if is_warmup else 0))
                try:
                    t0 = time.perf_counter()
                    _, value = IndependentSourcesOptimizationLoop._optimize_single_source(
                        source_acqf, loop.bounds,
                        x_best=all_dckg.x_best,
                        num_restarts=PROD_NUM_RESTARTS, raw_samples=PROD_RAW_SAMPLES,
                        seed=opt_seed, warm_start_ic=None)
                    elapsed = time.perf_counter() - t0
                    if not is_warmup:
                        rows.append(_row(name, repeat, opt_seed, optimise_seconds=elapsed,
                                         acquisition_value=float(value),
                                         status="ok", error_type="", error_message=""))
                except Exception as exc:  # noqa: BLE001
                    if not is_warmup:
                        rows.append(_row(name, repeat, opt_seed, optimise_seconds="",
                                         acquisition_value="", **_error_fields(exc)))
            logging.info(f"  optimise n_train={n_train} {name}: "
                         f"{_median_s(rows, name):.3f} s (median)")

    # --- cEI ----------------------------------------------------------------
    try:
        cei_loop, cei_model, x_best, cei_acqf = prepare_cei_state(
            black_box_function, n_train, args.seed)
    except Exception as exc:  # noqa: BLE001
        rows.append(_row(CEI, 0, "", optimise_seconds="", acquisition_value="",
                         **_error_fields(exc)))
        return rows

    for repeat, is_warmup in _repeats():
        opt_seed = args.seed + max(repeat, 0) + (WARMUP_SEED_OFFSET if is_warmup else 0)
        try:
            # compute_next_sample takes no seed argument, so the global RNG is
            # what makes repetitions distinct.
            set_all_seeds(opt_seed)
            t0 = time.perf_counter()
            _, value = optimise_cei(cei_loop, cei_model, cei_acqf, x_best)
            elapsed = time.perf_counter() - t0
            if not is_warmup:
                rows.append(_row(CEI, repeat, opt_seed, optimise_seconds=elapsed,
                                 acquisition_value=float(value),
                                 status="ok", error_type="", error_message=""))
        except Exception as exc:  # noqa: BLE001
            if not is_warmup:
                rows.append(_row(CEI, repeat, opt_seed, optimise_seconds="",
                                 acquisition_value="", **_error_fields(exc)))
    logging.info(f"  optimise n_train={n_train} {CEI}: {_median_s(rows, CEI):.3f} s (median)")
    return rows


# ----------------------------------------------------------------------------
# Measurement 3a: one complete sequential dcKG BO iteration
# ----------------------------------------------------------------------------

def run_one_bo_iteration(loop, train_x, train_y, seed):
    """One complete sequential dcKG BO iteration, reproducing
    IndependentSourcesOptimizationLoop.run() (bo/bo_loops/bo_loop.py:1487-1552).

    ``train_x``/``train_y`` are mutated in place (step 9). The outer timer is
    started before the first component and stopped after the dataset update; it
    is measured independently and never assembled from the component timers.
    Everything runs sequentially: no multiprocessing, threads, futures or async.
    """
    import torch

    components = {}
    total_start = time.perf_counter()

    # (2) Fit the objective and constraint GPs (production procedure).
    t0 = time.perf_counter()
    model = loop.update_model(X=train_x, y=train_y)
    components["fit_models_seconds"] = time.perf_counter() - t0

    # (3) Construct the three acquisitions. The recommendation of x_best is a
    # prerequisite of AllSourcesDcKG and is accounted for here.
    t0 = time.perf_counter()
    x_best, all_dckg = build_acquisitions(loop, model, seed)
    components["construct_acquisitions_seconds"] = time.perf_counter() - t0

    # (4) Optimise dckg_objective, dckg_constraint, coupled_ckg -- in that order.
    kg_values = torch.zeros(all_dckg.n_sources)
    best_xs = []
    for source_index, source_acqf in enumerate(all_dckg.sources):
        t0 = time.perf_counter()
        best_x_s, value_s = loop._optimize_single_source(
            source_acqf, loop.bounds,
            x_best=all_dckg.x_best,
            num_restarts=PROD_NUM_RESTARTS, raw_samples=PROD_RAW_SAMPLES,
            seed=seed + source_index * 100, warm_start_ic=None)
        components[f"optimise_{DCKG_ACQUISITIONS[source_index]}_seconds"] = (
            time.perf_counter() - t0)
        kg_values[source_index] = value_s
        best_xs.append(best_x_s)

    # (5-7) Cost normalisation, delta feasibility rule, source selection.
    t0 = time.perf_counter()
    coupled_index = all_dckg.n_sources - 1
    x_ckg = best_xs[coupled_index][:, 0:1, :].reshape(1, -1)
    idx_to_eval = loop.compute_important_idxs(model, x_ckg)
    coupled_cost = torch.sum(loop.costs[idx_to_eval])
    costs_with_ckg = torch.cat([loop.costs, coupled_cost.unsqueeze(0)])
    if (kg_values == 0).all():
        new_x = x_best.detach().reshape(1, -1)
        sources_to_eval = loop.compute_important_idxs(model, new_x)
        index = coupled_index
    else:
        index = int(torch.argmax(kg_values / costs_with_ckg))
        new_x = best_xs[index][:, 0:1, :].reshape(1, -1)
        sources_to_eval = idx_to_eval if index == coupled_index else [index]
    components["select_action_seconds"] = time.perf_counter() - t0

    # (8) Evaluate the selected Mystery source(s) at the selected location.
    t0 = time.perf_counter()
    new_ys = [loop.evaluate_black_box_func(new_x, src_idx) for src_idx in sources_to_eval]
    components["evaluate_sources_seconds"] = time.perf_counter() - t0

    # (9) Append the observation(s) to the corresponding dataset(s).
    t0 = time.perf_counter()
    for src_idx, new_y in zip(sources_to_eval, new_ys):
        train_x[src_idx] = torch.cat([train_x[src_idx].cpu(), new_x.cpu()])
        train_y[src_idx] = torch.cat([train_y[src_idx].cpu(), new_y.cpu()])
    components["update_datasets_seconds"] = time.perf_counter() - t0

    # (10) Stop the outer timer only after the dataset update has completed.
    components["total_loop_seconds"] = time.perf_counter() - total_start

    components.update(
        selected_action=DCKG_ACQUISITIONS[index],
        selected_location=_join_location(new_x),
        selected_sources=";".join(str(int(s)) for s in sources_to_eval),
        objective_acquisition_value=float(kg_values[0]),
        constraint_acquisition_value=float(kg_values[1]),
        coupled_acquisition_value=float(kg_values[coupled_index]),
        n_objective_after=int(train_x[0].shape[0]),
        n_constraint_after=int(train_x[1].shape[0]),
    )
    return components


# ----------------------------------------------------------------------------
# Measurement 3b: one complete sequential cEI BO iteration
# ----------------------------------------------------------------------------

def run_one_cei_iteration(loop, train_x, train_y, seed):
    """One complete coupled cEI BO iteration, reproducing
    EI_OptimizationLoop.run() (bo/bo_loops/bo_loop.py:879-904).

    cEI is coupled: there is no source selection, and every output is evaluated
    at the selected location, so one observation is appended to every dataset.
    """
    import torch

    components = {}
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    model = loop.update_model(X=train_x, y=train_y)
    components["fit_models_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_best, acquisition_function = build_cei_acquisition(loop, model, iteration=seed)
    components["construct_acquisition_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_x, value = optimise_cei(loop, model, acquisition_function, x_best)
    components["optimise_cei_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_y = loop.black_box_func.evaluate_black_box(new_x.cpu(), False)
    components["evaluate_sources_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for output_index in range(loop.model_wrapper.getNumberOfOutputs()):
        train_x[output_index] = torch.cat([train_x[output_index].cpu(), new_x.cpu()])
        train_y[output_index] = torch.cat([train_y[output_index].cpu(),
                                          new_y[:, output_index].cpu()])
    components["update_datasets_seconds"] = time.perf_counter() - t0

    components["total_loop_seconds"] = time.perf_counter() - total_start

    components.update(
        selected_location=_join_location(new_x),
        selected_sources=";".join(str(i) for i in range(loop.model_wrapper.getNumberOfOutputs())),
        cei_acquisition_value=float(value),
        n_objective_after=int(train_x[0].shape[0]),
        n_constraint_after=int(train_x[1].shape[0]),
    )
    return components


def measure_loop(black_box_function, n_train, args, base_record, *,
                 loop_factory, iteration_fn, columns, label):
    """Timed sequential BO iterations, each starting from the same base state.

    The base datasets are built once, outside every timer. Before each
    repetition a fresh loop (hence a fresh, unfitted model wrapper) is created
    and the base datasets are cloned, so the observation(s) appended by one
    repetition are discarded before the next one.
    """
    from Launcher import set_all_seeds

    rows = []
    base_loop = loop_factory(black_box_function, args.data_seed)
    base_train_x, base_train_y = make_base_state(base_loop, n_train)

    for repeat in range(-args.loop_warmups, args.loop_repeats):
        is_warmup = repeat < 0
        seed = args.seed + max(repeat, 0) + (WARMUP_SEED_OFFSET if is_warmup else 0)
        set_all_seeds(seed)
        loop = loop_factory(black_box_function, args.data_seed)
        train_x = [t.clone() for t in base_train_x]
        train_y = [t.clone() for t in base_train_y]
        n_objective_before = int(train_x[0].shape[0])
        n_constraint_before = int(train_x[1].shape[0])

        try:
            components = iteration_fn(loop, train_x, train_y, seed)
            status_fields = dict(status="ok", error_type="", error_message="")
        except Exception as exc:  # noqa: BLE001
            components = {}
            status_fields = _error_fields(exc)

        if is_warmup:
            logging.info(f"  {label} warm-up n_train={n_train} done")
            continue

        row = dict(base_record, n_train=n_train, repeat=repeat, seed=seed,
                   n_objective_before=n_objective_before,
                   n_constraint_before=n_constraint_before,
                   **status_fields)
        for column in columns:
            row.setdefault(column, components.get(column, ""))
        rows.append(row)
        total = row["total_loop_seconds"] if row["status"] == "ok" else "FAILED"
        logging.info(f"  {label} n_train={n_train} repeat={repeat}: {total} s")
    return rows


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def _join_location(new_x):
    return ";".join(f"{v:.6f}" for v in new_x.detach().cpu().reshape(-1).tolist())


def _error_fields(exc):
    return {"status": "error", "error_type": type(exc).__name__,
            "error_message": str(exc).replace("\n", " ")[:500]}


def _values(rows, column, acquisition=None):
    return [row[column] for row in rows
            if row["status"] == "ok" and (acquisition is None or row["acquisition"] == acquisition)]


def _median_ms(rows, acquisition):
    values = _values(rows, "forward_seconds", acquisition)
    return statistics.median(values) * 1e3 if values else float("nan")


def _median_s(rows, acquisition):
    values = _values(rows, "optimise_seconds", acquisition)
    return statistics.median(values) if values else float("nan")


def _quantiles(values):
    """Median, 25th and 75th percentile (linear interpolation, numpy convention)."""
    import numpy as np
    if not values:
        return "", "", ""
    array = np.asarray(values, dtype=float)
    return (float(np.median(array)), float(np.percentile(array, 25)),
            float(np.percentile(array, 75)))


def _median(values):
    import numpy as np
    return float(np.median(np.asarray(values, dtype=float))) if values else ""


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Wrote {path} ({len(rows)} rows)")


def _summarise_per_acquisition(rows, column, scale, names):
    summary = []
    for n_train in sorted({row["n_train"] for row in rows}):
        for acquisition in ACQUISITIONS:
            subset = [r for r in rows
                      if r["n_train"] == n_train and r["acquisition"] == acquisition]
            ok = [r[column] * scale for r in subset if r["status"] == "ok"]
            median, q25, q75 = _quantiles(ok)
            summary.append({"n_train": n_train, "acquisition": acquisition,
                            names[0]: median, names[1]: q25, names[2]: q75,
                            "successes": len(ok), "failures": len(subset) - len(ok)})
    return summary


FORWARD_SUMMARY_COLUMNS = ["n_train", "acquisition", "forward_median_ms", "forward_q25_ms",
                           "forward_q75_ms", "successes", "failures"]
OPT_SUMMARY_COLUMNS = ["n_train", "acquisition", "optimise_median_s", "optimise_q25_s",
                       "optimise_q75_s", "successes", "failures"]


def summarise_forward(rows):
    return _summarise_per_acquisition(rows, "forward_seconds", 1e3,
                                      FORWARD_SUMMARY_COLUMNS[2:5])


def summarise_optimisation(rows):
    return _summarise_per_acquisition(rows, "optimise_seconds", 1.0,
                                      OPT_SUMMARY_COLUMNS[2:5])


LOOP_COMPONENT_COLUMNS = ["fit_models_seconds", "construct_acquisitions_seconds",
                          "optimise_dckg_objective_seconds", "optimise_dckg_constraint_seconds",
                          "optimise_coupled_ckg_seconds", "select_action_seconds",
                          "evaluate_sources_seconds", "update_datasets_seconds"]

CEI_LOOP_COMPONENT_COLUMNS = ["fit_models_seconds", "construct_acquisition_seconds",
                              "optimise_cei_seconds", "evaluate_sources_seconds",
                              "update_datasets_seconds"]


def summarise_loop(rows, component_columns, count_actions):
    summary = []
    for n_train in sorted({row["n_train"] for row in rows}):
        subset = [r for r in rows if r["n_train"] == n_train]
        ok = [r for r in subset if r["status"] == "ok"]
        median, q25, q75 = _quantiles([r["total_loop_seconds"] for r in ok])
        record = {"n_train": n_train, "total_loop_median_s": median,
                  "total_loop_q25_s": q25, "total_loop_q75_s": q75}
        for column in component_columns:
            record[column.replace("_seconds", "_median_s")] = _median([r[column] for r in ok])
        record["successes"] = len(ok)
        record["failures"] = len(subset) - len(ok)
        if count_actions:
            for action, key in zip(DCKG_ACQUISITIONS, ["selected_objective_count",
                                                       "selected_constraint_count",
                                                       "selected_coupled_count"]):
                record[key] = sum(1 for r in ok if r["selected_action"] == action)
        summary.append(record)
    return summary


def _loop_summary_columns(component_columns, count_actions):
    columns = (["n_train", "total_loop_median_s", "total_loop_q25_s", "total_loop_q75_s"]
               + [c.replace("_seconds", "_median_s") for c in component_columns]
               + ["successes", "failures"])
    if count_actions:
        columns += ["selected_objective_count", "selected_constraint_count",
                    "selected_coupled_count"]
    return columns


LOOP_SUMMARY_COLUMNS = _loop_summary_columns(LOOP_COMPONENT_COLUMNS, True)
CEI_LOOP_SUMMARY_COLUMNS = _loop_summary_columns(CEI_LOOP_COMPONENT_COLUMNS, False)


def write_metadata(path, args, threads, torch_version):
    metadata = {
        "benchmark": BENCHMARK,
        "device": DEVICE_NAME,
        "dtype": "torch.float64",
        "threads": threads,
        "torch_version": torch_version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_train_values": list(args.n_train),
        "acquisitions": list(ACQUISITIONS),
        "production_settings": {
            "n_fantasies": PROD_N_FANTASIES,
            "n_constraint_samples": PROD_N_CONSTRAINT_SAMPLES,
            "n_disc_acqf": PROD_N_DISC_ACQF,
            "n_disc_optimiser": PROD_N_DISC_OPT,
            "q": PROD_Q,
            "num_restarts": PROD_NUM_RESTARTS,
            "raw_samples": PROD_RAW_SAMPLES,
            "infeasibility_threshold_delta": PROD_INFEASIBILITY_THRESHOLD,
            "cei_q": PROD_CEI_Q,
            "cei_num_restarts": PROD_CEI_NUM_RESTARTS,
            "cei_raw_samples": PROD_CEI_RAW_SAMPLES,
        },
        "forward_warmups": args.forward_warmups,
        "forward_repeats": args.forward_repeats,
        "forward_q": {name: ACQUISITION_Q[name] for name in ACQUISITIONS},
        "forward_q_note": (
            "The dcKG acquisitions are evaluated on a candidate plus a 64-point discretisation "
            "(q=65); cEI is evaluated on a single candidate (q=1). Per-call forward times are "
            "therefore not directly comparable across the two families."
        ),
        "opt_warmups": args.opt_warmups,
        "opt_repeats": args.opt_repeats,
        "sequential_execution": True,
        "loop_definition": LOOP_DEFINITION,
        "cei_loop_definition": CEI_LOOP_DEFINITION,
        "loop_warmups": args.loop_warmups,
        "loop_repeats": args.loop_repeats,
        "loop_includes_gp_fitting": True,
        "loop_includes_source_evaluation": True,
        "loop_includes_dataset_update": True,
        "individual_optimisation_includes_gp_fitting": False,
        "gp_refit_strategy": GP_REFIT_STRATEGY,
        "gp_refit_strategy_detail": GP_REFIT_STRATEGY_DETAIL,
        "recommendation_step": RECOMMENDATION_STEP,
        "smoke_test": bool(args.smoke_test),
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Wrote {path}")


# ----------------------------------------------------------------------------

def main(args):
    import torch
    from gpytorch import settings

    from bo.device_utils import DEVICE, DTYPE

    if torch.cuda.is_available():
        raise SystemExit("ERROR: this is a CPU-only benchmark but CUDA is still visible; "
                         "CUDA_VISIBLE_DEVICES was not applied before torch import.")

    # Match the module-level settings applied by bayesian_optimization_factory.py
    torch.set_default_dtype(DTYPE)
    settings.min_fixed_noise._global_double_value = 1e-6
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    threads = torch.get_num_threads()

    logging.info(f"Mystery timing benchmark on {DEVICE} ({threads} threads), "
                 f"torch {torch.__version__}, n_train={args.n_train}")

    black_box_function = make_mystery()
    os.makedirs(args.output_dir, exist_ok=True)
    base_record = {"benchmark": BENCHMARK, "device": DEVICE_NAME, "dtype": str(DTYPE),
                   "threads": threads}

    forward_rows, opt_rows, loop_rows, cei_loop_rows = [], [], [], []
    for n_train in args.n_train:
        logging.info(f"n_train={n_train}: forward evaluation")
        forward_rows += measure_forward(black_box_function, n_train, args, base_record)
        logging.info(f"n_train={n_train}: acquisition optimisation")
        opt_rows += measure_optimisation(black_box_function, n_train, args, base_record)
        logging.info(f"n_train={n_train}: sequential dcKG BO iteration")
        loop_rows += measure_loop(black_box_function, n_train, args, base_record,
                                  loop_factory=make_loop, iteration_fn=run_one_bo_iteration,
                                  columns=LOOP_COLUMNS, label="dckg loop")
        logging.info(f"n_train={n_train}: sequential cEI BO iteration")
        cei_loop_rows += measure_loop(black_box_function, n_train, args, base_record,
                                      loop_factory=make_cei_loop,
                                      iteration_fn=run_one_cei_iteration,
                                      columns=CEI_LOOP_COLUMNS, label="cei loop")

    out = args.output_dir
    write_csv(os.path.join(out, "mystery_forward_raw.csv"), FORWARD_COLUMNS, forward_rows)
    write_csv(os.path.join(out, "mystery_forward_summary.csv"), FORWARD_SUMMARY_COLUMNS,
              summarise_forward(forward_rows))
    write_csv(os.path.join(out, "mystery_opt_raw.csv"), OPT_COLUMNS, opt_rows)
    write_csv(os.path.join(out, "mystery_opt_summary.csv"), OPT_SUMMARY_COLUMNS,
              summarise_optimisation(opt_rows))
    write_csv(os.path.join(out, "mystery_bo_loop_raw.csv"), LOOP_COLUMNS, loop_rows)
    write_csv(os.path.join(out, "mystery_bo_loop_summary.csv"), LOOP_SUMMARY_COLUMNS,
              summarise_loop(loop_rows, LOOP_COMPONENT_COLUMNS, count_actions=True))
    write_csv(os.path.join(out, "mystery_cei_loop_raw.csv"), CEI_LOOP_COLUMNS, cei_loop_rows)
    write_csv(os.path.join(out, "mystery_cei_loop_summary.csv"), CEI_LOOP_SUMMARY_COLUMNS,
              summarise_loop(cei_loop_rows, CEI_LOOP_COMPONENT_COLUMNS, count_actions=False))
    write_metadata(os.path.join(out, "mystery_timing_metadata.json"), args, threads,
                   torch.__version__)

    all_rows = forward_rows + opt_rows + loop_rows + cei_loop_rows
    failures = [r for r in all_rows if r["status"] != "ok"]
    if failures:
        logging.warning(f"{len(failures)} measurement(s) failed; see the status columns.")
        for row in failures:
            logging.warning(f"  {row['error_type']}: {row['error_message']}")
    logging.info(f"Done. Render the paper table with: python timing_table.py --input {out}")


if __name__ == "__main__":
    try:
        main(parse_args())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
