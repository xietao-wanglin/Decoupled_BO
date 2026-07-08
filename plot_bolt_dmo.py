"""Plots for the BoLT DMCurriculumMO benchmark: opportunity cost and source allocation.

The OC reference optimum is the best feasible logit(MATH) over the fixed reference set
(see bo/synthetic_test_functions/bolt_dmo_data/generate_reference_set.py), recomputed
here from the committed pickle so the two never drift apart.
"""
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
REFERENCE_PICKLE = "bo/synthetic_test_functions/bolt_dmo_data/bolt_dmo_reference.pickle"

METHODS = {
    "dckg_indep": ("dckg_indep", "dcKG"),
    "ei_kg": ("ei_kg", "cEI+"),
    "ckg_v2": ("ckg_v2", "cKG"),
    "cei": ("cei", "cEI"),
}
COLORS = {"dckg_indep": "C0", "ei_kg": "C1", "ckg_v2": "C2", "cei": "C3"}
DECOUPLED_METHODS = ["dckg_indep", "ei_kg"]
SOURCE_LABELS = ["MATH-500 (objective)", "IFEval (constraint)", "MBPP+ (constraint)"]

# Common budget grid for aligning realisations with differing budget_consumed.
GRID = np.arange(1, 301)


def compute_optimum():
    with open(REFERENCE_PICKLE, "rb") as f:
        ref = pickle.load(f)
    Y = ref["Y"]
    if_scores, math_scores, mbpp_scores = Y[:, 0], Y[:, 1], Y[:, 2]
    tau_if = np.quantile(if_scores, ref["quantile_if"])
    tau_mbpp = np.quantile(mbpp_scores, ref["quantile_mbpp"])
    feasible = (if_scores >= tau_if) & (mbpp_scores >= tau_mbpp)
    math_logit = np.log(np.clip(math_scores, 1e-5, 1 - 1e-5) / (1 - np.clip(math_scores, 1e-5, 1 - 1e-5)))
    return np.max(math_logit[feasible])


OPTIMUM = compute_optimum()


def load_files(pattern):
    files = sorted(glob.glob(f"{RESULTS_DIR}/bolt_dmo*_{pattern}*.pkl"))
    data = []
    for f in files:
        with open(f, "rb") as fh:
            data.append(pickle.load(fh))
    return data


def oc_curves(pattern):
    """Return interpolated OC curves (one row per realisation) over GRID."""
    curves = []
    for d in load_files(pattern):
        vals = np.array([float(v.reshape(-1)[0]) for v in d["best_predicted_location_value"]])
        if vals.size == 0:
            continue
        budget = np.asarray(d["budget_consumed"], dtype=float)
        if budget.size == 0:
            # Coupled loops (ckg_v2, cei) don't record budget_consumed: each
            # iteration jointly evaluates all sources, costing costs.sum() =
            # number_of_outputs under equal costs.
            n_outputs = len(d["input_data"])
            budget = np.arange(1, vals.size + 1, dtype=float) * n_outputs
        oc = np.abs(OPTIMUM - vals)
        order = np.argsort(budget)
        budget, oc = budget[order], oc[order]
        curves.append(np.interp(GRID, budget, oc))
    return np.vstack(curves)


def source_fraction_curves(pattern):
    """Cumulative fraction of evaluations per source, aligned on the iteration index.

    Returns (iterations, fractions) where fractions has shape (n_seeds, n_iters, 3).
    """
    per_seed = []
    for d in load_files(pattern):
        # Entries are scalars/tensors for single-source picks, or lists such as
        # [0, 1, 2] when the loop falls back to evaluating every source.
        entries = [np.atleast_1d(np.asarray(i)).astype(int) for i in d["acqf_recommended_output_index:"]]
        if not entries:
            continue
        counts = np.array([[np.sum(e == src) for src in range(3)] for e in entries]).cumsum(axis=0)
        per_seed.append(counts / counts.sum(axis=1, keepdims=True))
    n_iters = min(f.shape[0] for f in per_seed)
    return np.arange(1, n_iters + 1), np.stack([f[:n_iters] for f in per_seed])


def plot_oc():
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, (pattern, label) in METHODS.items():
        curves = oc_curves(pattern)
        med = np.median(curves, axis=0)
        q25 = np.percentile(curves, 25, axis=0)
        q75 = np.percentile(curves, 75, axis=0)
        ax.plot(GRID, med, label=f"{label} (n={curves.shape[0]})", color=COLORS[key])
        ax.fill_between(GRID, q25, q75, alpha=0.2, color=COLORS[key])
    ax.set_xlabel("Budget consumed")
    ax.set_ylabel("Opportunity cost")
    ax.set_yscale("log")
    ax.set_title("BoLT DMCurriculumMO: Opportunity Cost vs Budget")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/bolt_dmo_oc.png", dpi=150)
    print("saved results/bolt_dmo_oc.png")


def plot_source_allocation():
    fig, axes = plt.subplots(1, len(DECOUPLED_METHODS), figsize=(6 * len(DECOUPLED_METHODS), 5), sharey=True)
    for ax, key in zip(np.atleast_1d(axes), DECOUPLED_METHODS):
        pattern, label = METHODS[key]
        iterations, fractions = source_fraction_curves(pattern)
        for src in range(3):
            med = np.median(fractions[:, :, src], axis=0)
            q25 = np.percentile(fractions[:, :, src], 25, axis=0)
            q75 = np.percentile(fractions[:, :, src], 75, axis=0)
            ax.plot(iterations, med, label=SOURCE_LABELS[src], color=f"C{src}")
            ax.fill_between(iterations, q25, q75, alpha=0.2, color=f"C{src}")
        ax.set_xlabel("Iteration")
        ax.set_title(f"{label} (n={fractions.shape[0]})")
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Cumulative fraction of evaluations")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle("BoLT DMCurriculumMO: Source Allocation")
    fig.tight_layout()
    fig.savefig("results/bolt_dmo_sources.png", dpi=150)
    print("saved results/bolt_dmo_sources.png")


if __name__ == "__main__":
    plot_oc()
    plot_source_allocation()
