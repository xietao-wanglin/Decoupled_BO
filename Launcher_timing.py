"""Timing study for acquisition-function forward evaluation.

Measures the wall-clock time of a single forward pass of the acquisition
function for DCKG_INDEPENDENT, CKG_V2 and CEI, on CPU or GPU, as a function of
the data budget (number of training points per GP) and of the acquisition
function's complexity knobs: the domain discretisation size (``--n-disc``), the
fantasy-sample count (``--n-fantasies``) and the constraint-sample count used by
the coupled cKG source (``--n-constraint-samples``). The complexity knobs are
swept one at a time around a fixed baseline (see the ``BASELINE_*`` constants);
cEI has no such knobs and is timed once per data budget.

Each run fits fresh GPs on a Latin-Hypercube design of ``n_train`` points per
output and computes the recommended location (untimed pre-steps, recorded in
separate columns), constructs the acquisition function, then times ``acqf(X)``
under ``torch.no_grad()`` on random candidate batches. Two forward metrics are
recorded, because they scale very differently with the data budget:

  - ``acqf_first_forward_time_s``: the FIRST forward after a model update,
    which builds the GP posterior caches (O(n^3)); the BO loop pays this once
    per iteration after refitting. This is where n_train scaling shows.
  - ``acqf_forward_time_s``: steady-state forward reusing the caches, averaged
    over ``--forwards-per-run`` calls (after 2 untimed warm-up forwards).

For large ``--n-train`` use ``--no-fit`` to skip GP hyperparameter
optimisation (the forward cost does not depend on the hyperparameter values).

Input shapes match how each acquisition is evaluated during optimisation:
  - CEI:              X of shape (batch, 1, d)
  - CKG_V2:           X of shape (batch, 1 + n_disc, d)  (candidate + discretisation)
  - DCKG_INDEPENDENT: one forward per source (K+2 sources), each on
                      (batch, 1 + n_disc, d); ``acqf_forward_time_s`` is their sum
                      and per-source times are recorded separately.

Usage:
    python Launcher_timing.py --function Mystery --device cpu
    python Launcher_timing.py --function Mystery --device cuda

Output: one CSV row per (algorithm, n_train, seed) appended to
``results/timing_study/timing_<function>_<device>.csv`` (plus a .pkl with the
same records). Aggregate with ``python timing_table.py``.
"""
import argparse
import csv
import logging
import os
import pickle
import sys
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ALGORITHMS = ["DCKG_INDEPENDENT", "CKG_V2", "CEI"]

FUNCTIONS = ["Mystery", "MysteryRedundant", "Branin", "TestFunc3", "WeldedBeam",
             "BraninHoo", "BraninHoo2", "BraninHoo3", "TensionCompression",
             "PressureVessel", "SpeedReducer"]

CSV_COLUMNS = ["function", "algorithm", "device", "device_name", "n_train",
               "n_disc", "n_fantasies", "n_constraint_samples", "seed",
               "warmup", "dim", "n_constraints", "forward_batch_size", "q_per_forward",
               "forwards_per_run", "hyperparams_fitted",
               "fit_time_s", "recommendation_time_s", "acqf_construction_time_s",
               "acqf_forward_time_s",
               "dckg_single_source_forward_time_s", "dckg_coupled_forward_time_s",
               "per_source_forward_times_s",
               "acqf_first_forward_time_s",
               "dckg_single_source_first_forward_time_s", "dckg_coupled_first_forward_time_s",
               "per_source_first_forward_times_s",
               "torch_version", "timestamp"]

# Baseline acquisition-function complexity knobs. When a sweep varies one axis
# (--n-disc / --n-fantasies / --n-constraint-samples), the other two are held at
# these values. The discretisation baseline matches the q = 1 + N_DISC the V2 KG
# optimisers use per forward (see _optimize_fast_ckg / _optimize_single_source in
# bo/bo_loops/bo_loop.py); the sample baselines match the production loop classes.
BASELINE_N_DISC = 64
BASELINE_N_FANTASIES = 7
BASELINE_N_CONSTRAINT_SAMPLES = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Acquisition-function timing study (CPU/GPU).")
    parser.add_argument("--function", type=str, choices=FUNCTIONS, default="Mystery",
                        help="Black-box benchmark (default: Mystery)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], required=True,
                        help="Device to run on. 'cpu' hides CUDA from the whole process.")
    parser.add_argument("--algorithms", type=str, nargs="+", choices=ALGORITHMS,
                        default=ALGORITHMS,
                        help="Algorithms to time (default: all three)")
    parser.add_argument("--n-train", type=int, nargs="+", default=[10, 40, 160, 640, 1280],
                        help="Data budgets: training points per GP (default: 10 40 160 640 1280)")
    parser.add_argument("--n-disc", type=int, nargs="+", default=[32, 64, 128],
                        help="Discretisation sizes to sweep for the KG methods; the candidate "
                             "tensor has q = 1 + n_disc (default: 32 64 128). Ignored by cEI.")
    parser.add_argument("--n-fantasies", type=int, nargs="+", default=[3, 7, 15],
                        help="Fantasy (Z_y) sample counts to sweep for the KG methods "
                             "(default: 3 7 15). Ignored by cEI.")
    parser.add_argument("--n-constraint-samples", type=int, nargs="+", default=[3, 5, 10],
                        help="Constraint (Z_c) sample counts to sweep; only the coupled cKG "
                             "source uses these (default: 3 5 10). Ignored by cEI.")
    parser.add_argument("--min-seed", type=int, default=0, help="Minimum seed (default: 0)")
    parser.add_argument("--max-seed", type=int, default=9, help="Maximum seed, inclusive (default: 9)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warm-up runs per algorithm, flagged warmup=1 in the CSV (default: 1)")
    parser.add_argument("--forward-batch-size", type=int, default=1,
                        help="Batch dimension of the candidate tensor passed to each forward (default: 1)")
    parser.add_argument("--forwards-per-run", type=int, default=10,
                        help="Timed forward calls per run; the recorded time is their mean (default: 10)")
    parser.add_argument("--no-fit", action="store_true",
                        help="Skip GP hyperparameter optimisation (L-BFGS). Forward cost does not "
                             "depend on hyperparameter values, and fitting dominates runtime for "
                             "large --n-train; recommended for n >= 1280.")
    parser.add_argument("--output-dir", type=str, default=os.path.join("results", "timing_study"),
                        help="Output directory (default: results/timing_study)")
    return parser.parse_args()


def make_black_box(name):
    from bo.synthetic_test_functions.synthetic_test_functions import (
        ConstrainedFunc3, ConstrainedBraninNew, MysteryFunctionSuperRedundant, WeldedBeamSO,
        PressureVessel, TensionCompression, SpeedReducer, BraninHoo, BraninHoo2, BraninHoo3,
    )
    if name == "Mystery":
        return MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True, redundant_constraints=False)
    if name == "MysteryRedundant":
        return MysteryFunctionSuperRedundant(noise_std=1e-6, negate=True, redundant_constraints=True)
    if name == "Branin":
        return ConstrainedBraninNew(noise_std=1e-6, negate=True)
    if name == "TestFunc3":
        return ConstrainedFunc3(noise_std=1e-6, negate=True)
    if name == "WeldedBeam":
        return WeldedBeamSO(noise_std=1e-6, negate=True)
    if name == "BraninHoo":
        return BraninHoo(noise_std=1e-2, negate=True)
    if name == "BraninHoo2":
        return BraninHoo2(noise_std=1e-2, negate=True)
    if name == "BraninHoo3":
        return BraninHoo3(noise_std=1e-2, negate=True)
    if name == "TensionCompression":
        return TensionCompression(noise_std=1e-6, negate=True)
    if name == "PressureVessel":
        return PressureVessel(noise_std=1e-6, negate=True)
    if name == "SpeedReducer":
        return SpeedReducer(noise_std=1e-6, negate=True)
    raise ValueError(f"Function {name} is not supported.")


def make_loop(algorithm, black_box_function, seed):
    """Build the same loop object BayesianOptimizationLoopFactory.create() would,
    without touching any results files on disk."""
    import torch
    from botorch.acquisition import ConstrainedMCObjective

    from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType
    from bo.bo_loops.bo_loop import EI_OptimizationLoop, IndependentSourcesOptimizationLoop
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
    dim = black_box_function.dim
    bounds = torch.zeros(2, dim, device=device, dtype=dtype)
    bounds[1] = 1.0
    # Same double-wrapping as Launcher.py -> BayesianOptimizationLoopFactory.create()
    penalty_value = torch.tensor([torch.tensor([black_box_function.get_penalty()])])

    common = dict(
        black_box_func=black_box_function,
        objective=constrained_obj,
        bounds=bounds,
        performance_type="model",
        model=model,
        seed=seed,
        budget=1,  # never used: we do not call run()
        number_initial_designs=6,
        results=Results(filename="timing_study_dummy.pkl"),  # never serialised
        costs=torch.ones(number_of_constraints + 1),
        penalty_value=penalty_value,
    )
    if algorithm == "CEI":
        return EI_OptimizationLoop(ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                   **common)
    if algorithm == "CKG_V2":
        return EI_OptimizationLoop(ei_type=AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2,
                                   **common)
    if algorithm == "DCKG_INDEPENDENT":
        return IndependentSourcesOptimizationLoop(
            ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2, **common)
    raise ValueError(f"Algorithm {algorithm} is not supported.")


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _candidate_batch(dim, q, batch_size):
    """Random candidate tensor of shape (batch_size, q, d) on DEVICE."""
    import torch
    from bo.device_utils import DEVICE, DTYPE
    return torch.rand(batch_size, q, dim, device=DEVICE, dtype=DTYPE)


def time_forward(acquisition_function, X, n_forwards):
    """Mean wall-clock time of one steady-state acqf(X) forward under no_grad.

    Two untimed warm-up forwards absorb lazy posterior caches / CUDA kernel
    compilation before the timed loop.
    """
    import torch
    with torch.no_grad():
        for _ in range(2):
            acquisition_function(X)
        _sync()
        t0 = time.perf_counter()
        for _ in range(n_forwards):
            acquisition_function(X)
        _sync()
    return (time.perf_counter() - t0) / n_forwards


def _clear_posterior_caches(model):
    """Reset gpytorch prediction caches, as happens after every model refit."""
    model.train()
    model.eval()


N_COLD_REPS = 3


def time_first_forward(acquisition_function, model, X):
    """Mean wall-clock time of the FIRST acqf(X) forward after a model update.

    This includes building the GP posterior caches (O(n^3) in the training-set
    size), which the BO loop pays once per iteration after refitting the model.
    Steady-state forwards (time_forward) reuse those caches and only show the
    O(n)/O(n^2) per-query cost.
    """
    import torch
    total = 0.0
    with torch.no_grad():
        for _ in range(N_COLD_REPS):
            _clear_posterior_caches(model)
            _sync()
            t0 = time.perf_counter()
            acquisition_function(X)
            _sync()
            total += time.perf_counter() - t0
    return total / N_COLD_REPS


def time_coupled_forward(loop, model, best_observed_location, best_observed_value, iteration,
                         batch_size, n_forwards, n_disc, n_fantasies, n_constraint_samples):
    """Forward-pass time for CEI (q=1) and CKG_V2 (q=1+n_disc).

    CKG_V2 is a coupled cKG, so its ``CoupledCKG`` is constructed here directly
    (rather than via acquisition_function_factory, which hardcodes the sample
    counts) so ``n_fantasies`` and ``n_constraint_samples`` can be swept.
    """
    from bo.acquisition_functions.acquisition_functions import (
        AcquisitionFunctionType, acquisition_function_factory)
    from bo.acquisition_functions.refactored_acquisition_functions import (
        CoupledCKG, FastConstrainedKG)

    is_ckg_v2 = (loop.acquisition_function_type
                 is AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT_V2)

    _sync()
    t0 = time.perf_counter()
    if is_ckg_v2:
        acquisition_function = CoupledCKG(
            model,
            penalty_value=loop.penalty_value,
            x_best_location=best_observed_location,
            objective=loop.objective,
            n_fantasies=n_fantasies,
            n_constraint_samples=n_constraint_samples,
            seed=iteration)
    else:
        acquisition_function = acquisition_function_factory(
            model=model,
            type=loop.acquisition_function_type,
            objective=loop.objective,
            best_value=best_observed_value,
            idx=1,
            number_of_outputs=loop.number_of_outputs,
            penalty_value=loop.penalty_value,
            iteration=iteration,
            initial_condition_internal_optimizer=best_observed_location)
    _sync()
    construction_time = time.perf_counter() - t0

    q = 1 + n_disc if isinstance(acquisition_function, FastConstrainedKG) else 1
    X = _candidate_batch(loop.dim_x, q, batch_size)
    first_forward_time = time_first_forward(acquisition_function, model, X)
    forward_time = time_forward(acquisition_function, X, n_forwards)
    return {
        "construction_time": construction_time,
        "forward_time": forward_time,
        "first_forward_time": first_forward_time,
        "q": q,
        "per_source_times": [],
        "per_source_first_times": [],
    }


def time_dckg_forward(loop, model, best_observed_location, iteration, batch_size, n_forwards,
                      n_disc, n_fantasies, n_constraint_samples):
    """Per-source forward-pass times for dcKG (IndependentSourcesOptimizationLoop).

    AllSourcesDcKG holds K+2 sources: objective dcKG, K constraint dcKGs, and
    the full coupled cKG as the last source. Each is timed on its own
    (batch, 1+n_disc, d) candidate tensor. Returns the total (sum over
    sources) plus the per-source breakdown, from which the single-source and
    coupled-cKG columns are derived.
    """
    from bo.acquisition_functions.refactored_acquisition_functions import AllSourcesDcKG

    _sync()
    t0 = time.perf_counter()
    all_dckg = AllSourcesDcKG(
        model,
        penalty_value=loop.penalty_value,
        x_best_location=best_observed_location,
        objective=loop.objective,
        n_fantasies=n_fantasies,
        n_constraint_samples=n_constraint_samples,
        n_disc=n_disc,
        seed=iteration,
    )
    _sync()
    construction_time = time.perf_counter() - t0

    q = 1 + n_disc
    per_source_times = []
    per_source_first_times = []
    for source_acqf in all_dckg.sources:
        X = _candidate_batch(loop.dim_x, q, batch_size)
        per_source_first_times.append(time_first_forward(source_acqf, model, X))
        per_source_times.append(time_forward(source_acqf, X, n_forwards))
    return {
        "construction_time": construction_time,
        "forward_time": sum(per_source_times),
        "first_forward_time": sum(per_source_first_times),
        "q": q,
        "per_source_times": per_source_times,
        "per_source_first_times": per_source_first_times,
    }


def _build_model(loop, train_x, train_y, no_fit):
    """Fit the GPs. With no_fit, skip hyperparameter optimisation (the forward
    cost does not depend on hyperparameter values) but keep the same copula
    transform and device placement as loop.update_model."""
    from bo.device_utils import DEVICE
    from bo.model.Model import gaussian_copula_transform

    if not no_fit:
        return loop.update_model(train_x, train_y)
    y = train_y
    if loop.black_box_func.get_objective_transform() is not None:
        y = list(y)
        y[0] = gaussian_copula_transform(y[0])
    loop.model_wrapper.fit(train_x, y)
    loop.model_wrapper.to_device(DEVICE)
    return loop.model_wrapper.model


def run_once(algorithm, black_box_function, n_train, seed, batch_size, n_forwards, no_fit,
             n_disc, n_fantasies, n_constraint_samples):
    """One timing run: data generation, GP fit, recommendation, acqf forward timing.

    ``n_disc``/``n_fantasies``/``n_constraint_samples`` are the acquisition-function
    complexity knobs; they are ignored for CEI (which has no discretisation).
    """
    from Launcher import set_all_seeds

    set_all_seeds(seed)
    loop = make_loop(algorithm, black_box_function, seed)

    train_x, train_y = loop.generate_initial_data(n=n_train)

    _sync()
    t0 = time.perf_counter()
    model = _build_model(loop, train_x, train_y, no_fit)
    _sync()
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    best_observed_location, best_observed_value = loop.best_observed(
        best_value_computation_type=loop.performance_type,
        train_x=train_x, train_y=train_y, model=model, bounds=loop.bounds)
    _sync()
    recommendation_time = time.perf_counter() - t0

    if algorithm == "DCKG_INDEPENDENT":
        timings = time_dckg_forward(
            loop, model, best_observed_location, iteration=seed,
            batch_size=batch_size, n_forwards=n_forwards,
            n_disc=n_disc, n_fantasies=n_fantasies,
            n_constraint_samples=n_constraint_samples)
    else:
        timings = time_coupled_forward(
            loop, model, best_observed_location, best_observed_value, iteration=seed,
            batch_size=batch_size, n_forwards=n_forwards,
            n_disc=n_disc, n_fantasies=n_fantasies,
            n_constraint_samples=n_constraint_samples)

    return {
        "fit_time_s": fit_time,
        "recommendation_time_s": recommendation_time,
        "acqf_construction_time_s": timings["construction_time"],
        "acqf_forward_time_s": timings["forward_time"],
        "acqf_first_forward_time_s": timings["first_forward_time"],
        "q_per_forward": timings["q"],
        "per_source_times": timings["per_source_times"],
        "per_source_first_times": timings["per_source_first_times"],
    }


def _knob_points(algorithm, n_disc_grid, n_fantasies_grid, n_constraint_grid):
    """(n_disc, n_fantasies, n_constraint_samples) points to time for one algorithm.

    cEI has no complexity knobs, so it gets a single ``(None, None, None)`` point.
    The KG methods sweep one knob at a time around the baseline (disc, then
    fantasy, then constraint); the shared baseline point is emitted only once.
    """
    if algorithm == "CEI":
        return [(None, None, None)]
    sweep = ([(nd, BASELINE_N_FANTASIES, BASELINE_N_CONSTRAINT_SAMPLES) for nd in n_disc_grid]
             + [(BASELINE_N_DISC, nf, BASELINE_N_CONSTRAINT_SAMPLES) for nf in n_fantasies_grid]
             + [(BASELINE_N_DISC, BASELINE_N_FANTASIES, nc) for nc in n_constraint_grid])
    ordered, seen = [], set()
    for point in sweep:
        if point not in seen:
            seen.add(point)
            ordered.append(point)
    return ordered


def append_record(csv_path, record):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def main(args):
    import torch
    from gpytorch import settings

    from bo.device_utils import DEVICE, DTYPE

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("ERROR: --device cuda requested but CUDA is not available on this machine.")
    if args.device == "cpu" and torch.cuda.is_available():
        sys.exit("ERROR: --device cpu requested but CUDA is still visible; "
                 "CUDA_VISIBLE_DEVICES was not applied before torch import.")

    # Match the module-level settings applied by bayesian_optimization_factory.py
    torch.set_default_dtype(DTYPE)
    settings.min_fixed_noise._global_double_value = 1e-6

    device_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    logging.info(f"Timing study on device={DEVICE} ({device_name}), torch {torch.__version__}")

    black_box_function = make_black_box(args.function)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"timing_{args.function}_{args.device}.csv")
    pkl_path = os.path.join(args.output_dir, f"timing_{args.function}_{args.device}.pkl")

    records = []
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            records = pickle.load(f)

    seeds = list(range(args.min_seed, args.max_seed + 1))

    for algorithm in args.algorithms:
        knob_points = _knob_points(algorithm, args.n_disc, args.n_fantasies,
                                   args.n_constraint_samples)
        baseline_point = (None, None, None) if algorithm == "CEI" else (
            BASELINE_N_DISC, BASELINE_N_FANTASIES, BASELINE_N_CONSTRAINT_SAMPLES)
        # Warm-ups: smallest budget at the baseline knobs, flagged warmup=1.
        warmup_runs = [(min(args.n_train), *baseline_point, 10_000 + w, 1)
                       for w in range(args.warmup)]
        # Timed: every (n_train, knob point, seed).
        timed_runs = [(n, nd, nf, nc, seed, 0)
                      for n in args.n_train
                      for (nd, nf, nc) in knob_points
                      for seed in seeds]

        for n_train, n_disc, n_fantasies, n_constraint_samples, seed, warmup in \
                warmup_runs + timed_runs:
            label = "warmup" if warmup else "timed"
            logging.info(f"[{label}] algorithm={algorithm}, n_train={n_train}, "
                         f"n_disc={n_disc}, n_fantasies={n_fantasies}, "
                         f"n_constraint_samples={n_constraint_samples}, seed={seed}")
            timings = run_once(algorithm, black_box_function, n_train, seed,
                               batch_size=args.forward_batch_size,
                               n_forwards=args.forwards_per_run,
                               no_fit=args.no_fit,
                               n_disc=n_disc, n_fantasies=n_fantasies,
                               n_constraint_samples=n_constraint_samples)

            # dcKG sources: [objective, constraint_1..K, coupled cKG]. The
            # single-source time is the mean over the K+1 decoupled sources;
            # the coupled time is the last source (full coupled cKG).
            def _split_sources(times):
                if not times:
                    return "", ""
                return sum(times[:-1]) / len(times[:-1]), times[-1]

            per_source_times = timings["per_source_times"]
            per_source_first_times = timings["per_source_first_times"]
            single_source_time, coupled_time = _split_sources(per_source_times)
            single_source_first_time, coupled_first_time = _split_sources(per_source_first_times)
            record = {
                "function": args.function,
                "algorithm": algorithm,
                "device": args.device,
                "device_name": device_name,
                "n_train": n_train,
                "n_disc": "NA" if n_disc is None else n_disc,
                "n_fantasies": "NA" if n_fantasies is None else n_fantasies,
                "n_constraint_samples": "NA" if n_constraint_samples is None else n_constraint_samples,
                "seed": seed,
                "warmup": warmup,
                "dim": black_box_function.dim,
                "n_constraints": black_box_function.get_number_of_constraints(),
                "forward_batch_size": args.forward_batch_size,
                "q_per_forward": timings["q_per_forward"],
                "forwards_per_run": args.forwards_per_run,
                "hyperparams_fitted": 0 if args.no_fit else 1,
                "fit_time_s": timings["fit_time_s"],
                "recommendation_time_s": timings["recommendation_time_s"],
                "acqf_construction_time_s": timings["acqf_construction_time_s"],
                "acqf_forward_time_s": timings["acqf_forward_time_s"],
                "dckg_single_source_forward_time_s": single_source_time,
                "dckg_coupled_forward_time_s": coupled_time,
                "per_source_forward_times_s": ";".join(f"{t:.6f}" for t in per_source_times),
                "acqf_first_forward_time_s": timings["acqf_first_forward_time_s"],
                "dckg_single_source_first_forward_time_s": single_source_first_time,
                "dckg_coupled_first_forward_time_s": coupled_first_time,
                "per_source_first_forward_times_s": ";".join(f"{t:.6f}" for t in per_source_first_times),
                "torch_version": torch.__version__,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            append_record(csv_path, record)
            records.append(record)
            with open(pkl_path, "wb") as f:
                pickle.dump(records, f)
            logging.info(f"    first_forward={timings['acqf_first_forward_time_s'] * 1e3:.2f}ms, "
                         f"steady_forward={timings['acqf_forward_time_s'] * 1e3:.3f}ms "
                         f"(fit={timings['fit_time_s']:.3f}s)")

    logging.info(f"Done. Results in {csv_path}")


if __name__ == "__main__":
    _args = parse_args()
    if _args.device == "cpu":
        # Must happen before torch (and any bo module) is imported: bo.device_utils
        # resolves DEVICE at import time.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main(_args)
