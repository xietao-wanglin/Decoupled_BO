# Decoupled Constrained Bayesian Optimisation

Reference implementation for the decoupled constrained Knowledge Gradient (**dcKG**) and the
coupled baselines it is compared against. In a decoupled problem the objective and each constraint
can be queried independently, at its own cost, so the acquisition function has to choose *where* to
sample and *which source* to reveal.

## Install

Requires Python 3.10 and BoTorch 0.10.0.

```bash
conda env create -f environment.yml
conda activate py3.10
```

## Algorithms

| `BayesianOptimizationLoopType` | Paper name | Type |
|---|---|---|
| `CEI` | cEI — constrained Expected Improvement | coupled |
| `CKG_V2` | cKG — constrained Knowledge Gradient | coupled |
| `DCKG_INDEPENDENT` | dcKG — decoupled constrained KG | decoupled |
| `DCKG_NO_COUPLED` | dcKG without the coupled cKG candidate (ablation) | decoupled |
| `DCKG_PURE` | fully decoupled dcKG (ablation) | decoupled |

Coupled methods evaluate every output at the chosen location; dcKG optimises one acquisition
function per source and picks the source with the best KG-per-unit-cost, keeping the coupled cKG
candidate as an additional option. `CKG_V2` and the dcKG sources use the reparametrisation-based
implementation in `bo/acquisition_functions/refactored_acquisition_functions.py`.

## Benchmarks

| `--function` | Problem | Constraints |
|---|---|---|
| `Mystery` | Mystery | 1 |
| `MysteryRedundant` | Mystery + 8 redundant constraints | 9 |
| `Branin` | Constrained Branin | 1 |
| `TestFunc3` | Test Function 2 | 3 |
| `TestFunc3Redundant` | Test Function 2, noise on the active constraint c2 | 3 |
| `TestFunc3RedundantC1` | Test Function 2, noise on c1 | 3 |
| `TestFunc3RedundantC3` | Test Function 2, noise on c3 | 3 |
| `TestFunc3RedundantNoNoise` | Test Function 2, deterministic reference | 3 |
| `PressureVessel` | Pressure vessel design | 3 |
| `SpeedReducer` | Speed reducer design | 11 |
| `two_layer_cnn_discrete` | CNN hyper-parameters on class-imbalanced CIFAR-10 | 10 |
| `bolt_dmo`, `bolt_dmo_10` | Modified BOLT data-mixture benchmark | 2 |

The CNN benchmark reads a precomputed table of trained-network results
(`bo/synthetic_test_functions/AutoML_data_CBO/cnn_CIFAR10_data/`). The BOLT benchmark uses the
vendored emulator in `bo/synthetic_test_functions/bolt_emulator.py`, which downloads its weights
from the Hugging Face Hub on first use.

## Running experiments

**`Launcher_replic.py` is the main entry point.** It reproduces every experiment in the paper:

```bash
python Launcher_replic.py --smoke    # seeds 0-1 — start here, checks the whole sweep runs
python Launcher_replic.py            # seeds 0-39 — the full paper sweep
```

### How it works

`Launcher_replic.py` holds the sweep as a list of configurations and runs `Launcher.py` once per
entry in a subprocess, so every configuration starts from a clean interpreter. Each entry is a
4-tuple:

```python
runs = [
    # (function_name, decoupled, cost, ablation)
    ("Mystery",   False, None, None),    # cEI and cKG on Mystery, equal costs
    ("Mystery",   True,  None, None),    # dcKG on Mystery, equal costs
    ("Mystery",   True,  5,    None),    # dcKG, each source made expensive in turn
    ("TestFunc3", True,  None, "both"),  # the two dcKG ablations
]
```

- `function_name` — one of the `--function` values in the benchmark table above.
- `decoupled` — `False` runs the coupled baselines (cEI **and** cKG); `True` runs dcKG.
- `cost` — `None` for equal costs; a number (the paper uses `5`) repeats the run with each source
  made expensive in turn, i.e. `K + 1` runs for `K` constraints. Decoupled only.
- `ablation` — `None`, or `"nocoupled"` / `"pure"` / `"both"` to run the supplement's
  dcKG-without-cKG variants instead of dcKG. Decoupled only.

The shipped list is exactly the paper's sweep: the eight benchmarks coupled and decoupled at equal
cost, the noisy-constraint variants of Test Function 2, the heterogeneous-cost runs on Mystery,
Branin and Test Function 2, and the ablations. To run a subset, comment out the entries you do not
need; to add a configuration, append a tuple. `--smoke` only narrows the seed range (0–1), it still
walks the whole list.

Expect the full sweep to take a long time: 24 configurations × 40 seeds, and a single dcKG iteration
costs roughly 7–12 s of overhead on top of the benchmark evaluation (see the timing study below).
A configuration that exits non-zero aborts the sweep rather than being skipped
(`subprocess.run(..., check=True)`). Because runs resume (next section), it is safe to interrupt and
restart, or to split the list across machines: fix the cause and re-run, and the driver picks up
where it stopped instead of redoing finished work.

### Results, resuming and single runs

Per-iteration histories are pickled under `results/`, one file per (benchmark, cost setting,
algorithm, seed). Default budget is 160 evaluations, 300 for the CNN and BOLT benchmarks. On startup
each run inspects its own file: it **skips** the run if the budget is already spent and **resumes**
from the stored data otherwise — so re-running `Launcher_replic.py` after an interruption only does
the work that is left.

To run a single configuration directly — for debugging, or to parallelise the sweep by hand — call
`Launcher.py` with the same options:

```bash
# Coupled baselines (cEI and cKG)
python Launcher.py --function Mystery --min-seed 0 --max-seed 39

# Decoupled dcKG
python Launcher.py --function Mystery --min-seed 0 --max-seed 39 --decoupled

# Heterogeneous evaluation costs: each source is made expensive in turn
python Launcher.py --function Mystery --min-seed 0 --max-seed 39 --decoupled --cost 5

# Supplement ablations of dcKG
python Launcher.py --function TestFunc3 --min-seed 0 --max-seed 39 --decoupled --ablation both
```

Which algorithms a `Launcher.py` invocation runs is decided by `get_bo_algorithms()` in that file:
`[CEI, CKG_V2]` when coupled, `[DCKG_INDEPENDENT]` when `--decoupled`. Edit those lists to add or
drop a method across the whole sweep.

## Timing study

Single-threaded CPU wall-clock cost of the acquisition functions and of a complete BO iteration on
the Mystery problem:

```bash
python Launcher_timing.py               # writes results/timing_study/*.csv + metadata JSON
python Launcher_timing.py --smoke-test  # minimal end-to-end check
python timing_table.py                  # prints markdown, writes mystery_timing_table.tex
```

The benchmark pins every CPU thread pool to a single thread by default (`--threads`) so the numbers
are stable and comparable.

## Tests

```bash
pytest tests/
```

The experiment tests run each algorithm for two evaluations on every benchmark; the acquisition
tests check shapes, non-negativity and the epigraph/KGCB implementations.

## Layout

```
Launcher_replic.py   main entry point: the whole paper sweep, one subprocess per configuration
Launcher.py          a single configuration: one benchmark, coupled or decoupled, a range of seeds
Launcher_timing.py   timing benchmark; timing_table.py renders its LaTeX table
bo/acquisition_functions/  cEI / cKG / dcKG acquisition functions
bo/bo_loops/               BO loops and the factory that wires them up
bo/model/                  GP wrapper and the feasibility-weighted posterior mean
bo/synthetic_test_functions/  benchmark problems
bo/result_utils/           per-iteration result container
```
