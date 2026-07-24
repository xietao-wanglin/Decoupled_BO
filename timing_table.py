"""Aggregate Launcher_timing.py results into paper-ready tables.

Reads all ``timing_*.csv`` files in the input directory (default
``results/timing_study/``), drops warm-up runs, and reports the mean +- std of
the acquisition-function forward-pass time (milliseconds) per method, device
and data budget. dcKG is split into its single decoupled source (mean over the
objective/constraint sources) and its full coupled cKG source.

For each benchmark function and each of the two forward metrics (see below), a
separate table is produced for every acquisition-function complexity axis swept
by Launcher_timing.py — discretisation size (``n_disc``), fantasy samples
(``n_fantasies``) and constraint samples (``n_constraint_samples``). Within one
axis table the other two knobs are held at their baseline; cEI has no such
knobs and appears as a single reference row (axis value ``--``).

The two metrics:
  - first forward after a model update (includes the O(n^3) posterior-cache
    build the BO loop pays once per iteration — this is where n_train scaling
    shows), and
  - steady-state forward (reuses the caches).

For each (function, metric, axis) it prints a Markdown table to stdout and
writes ``timing_table_<function>_<metric>_<axis>.tex`` (booktabs LaTeX). A
single ``timing_summary.csv`` (aggregated mean/std/count across every
function/metric/axis) is also written.

Usage:
    python timing_table.py
    python timing_table.py --input results/timing_study
"""
import argparse
import glob
import os

import pandas as pd

from Launcher_timing import (BASELINE_N_CONSTRAINT_SAMPLES, BASELINE_N_DISC,
                             BASELINE_N_FANTASIES)

# Two metrics, reported as separate tables:
#  - "first forward": the first acqf(X) call after a model update; includes
#    building the GP posterior caches (O(n^3)) and is where n_train scaling shows.
#  - "steady-state forward": subsequent calls reusing the caches.
# Each maps to: [(algorithm in CSV, column holding the time in seconds, display name)]
METRICS = {
    "first": {
        "title": "first forward after model update",
        "suffix": "first",
        "methods": [
            ("DCKG_INDEPENDENT", "dckg_single_source_first_forward_time_s", "dcKG (single source)"),
            ("DCKG_INDEPENDENT", "dckg_coupled_first_forward_time_s", "dcKG (coupled cKG)"),
            ("CKG_V2", "acqf_first_forward_time_s", "cKG"),
            ("CEI", "acqf_first_forward_time_s", "cEI"),
        ],
    },
    "steady": {
        "title": "steady-state forward",
        "suffix": "steady",
        "methods": [
            ("DCKG_INDEPENDENT", "dckg_single_source_forward_time_s", "dcKG (single source)"),
            ("DCKG_INDEPENDENT", "dckg_coupled_forward_time_s", "dcKG (coupled cKG)"),
            ("CKG_V2", "acqf_forward_time_s", "cKG"),
            ("CEI", "acqf_forward_time_s", "cEI"),
        ],
    },
}

# Complexity axes swept by Launcher_timing.py. For each axis the OTHER two knobs
# are pinned to their baseline so the table isolates the effect of that one axis.
KNOB_COLUMNS = ["n_disc", "n_fantasies", "n_constraint_samples"]
AXES = {
    "n_disc": {
        "title": "discretisation size",
        "latex": r"$n_\mathrm{disc}$",
        "others": {"n_fantasies": BASELINE_N_FANTASIES,
                   "n_constraint_samples": BASELINE_N_CONSTRAINT_SAMPLES},
    },
    "n_fantasies": {
        "title": "fantasy samples",
        "latex": r"$n_y$",
        "others": {"n_disc": BASELINE_N_DISC,
                   "n_constraint_samples": BASELINE_N_CONSTRAINT_SAMPLES},
    },
    "n_constraint_samples": {
        "title": "constraint samples",
        "latex": r"$n_c$",
        "others": {"n_disc": BASELINE_N_DISC,
                   "n_fantasies": BASELINE_N_FANTASIES},
    },
}
DEVICE_ORDER = ["cpu", "cuda"]
DEVICE_LABELS = {"cpu": "CPU", "cuda": "GPU"}


def load_results(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "timing_*.csv")))
    if not files:
        raise SystemExit(f"No timing_*.csv files found in {input_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["warmup"] == 0]
    if df.empty:
        raise SystemExit("Only warm-up rows found; run more seeds first.")
    # The knob columns are "NA" for cEI (no discretisation / samples); coerce to
    # nullable numeric so cEI rows become NaN and form their own axis group.
    for col in KNOB_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_to_axis(df, axis_col, others):
    """Rows relevant to one axis: the other two knobs at baseline, plus cEI
    (all-NaN knobs) which is a knob-independent reference line."""
    mask = pd.Series(True, index=df.index)
    for col, baseline in others.items():
        mask &= (df[col] == baseline) | df[col].isna()
    return df[mask]


def aggregate(df, methods, metric_name, axis_col):
    """Long-format aggregation: one row per
    (function, metric, method, axis, device, n_train, axis_value),
    times converted to milliseconds. Returns None if nothing matches."""
    parts = []
    for algorithm, column, method in methods:
        if column not in df.columns:
            continue
        sub = df[df["algorithm"] == algorithm].dropna(subset=[column])
        if sub.empty:
            continue
        grouped = (sub.groupby(["function", "device", "n_train", axis_col], dropna=False)[column]
                      .agg(mean="mean", std="std", runs="count")
                      .reset_index())
        grouped["mean"] = grouped["mean"] * 1e3
        grouped["std"] = grouped["std"].fillna(0.0) * 1e3
        grouped = grouped.rename(columns={axis_col: "axis_value"})
        grouped.insert(1, "method", method)
        grouped.insert(1, "axis", axis_col)
        grouped.insert(1, "metric", metric_name)
        parts.append(grouped)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def format_cell(mean, std):
    if mean < 1:
        return f"{mean:.3f} ± {std:.3f}"
    if mean < 100:
        return f"{mean:.2f} ± {std:.2f}"
    return f"{mean:.1f} ± {std:.1f}"


def _axis_label(value):
    return "--" if pd.isna(value) else str(int(value))


def build_pivot(agg_fn, methods, axis_meta):
    """Rows: (method, device, axis value); columns: n_train; cells: 'mean ± std' ms."""
    n_trains = sorted(agg_fn["n_train"].unique())
    method_order = [m[2] for m in methods]
    axis_header = axis_meta["title"]
    rows = []
    for method in dict.fromkeys(method_order):
        for device in DEVICE_ORDER:
            sub_md = agg_fn[(agg_fn["method"] == method) & (agg_fn["device"] == device)]
            if sub_md.empty:
                continue
            axis_values = sorted(sub_md["axis_value"].dropna().unique())
            if sub_md["axis_value"].isna().any():
                axis_values = axis_values + [float("nan")]
            for value in axis_values:
                if pd.isna(value):
                    sub = sub_md[sub_md["axis_value"].isna()]
                else:
                    sub = sub_md[sub_md["axis_value"] == value]
                row = {"Method": method, "Device": DEVICE_LABELS.get(device, device),
                       axis_header: _axis_label(value)}
                for n in n_trains:
                    cell = sub[sub["n_train"] == n]
                    row[f"n={n}"] = (format_cell(cell["mean"].iloc[0], cell["std"].iloc[0])
                                     if not cell.empty else "--")
                rows.append(row)
    return pd.DataFrame(rows), n_trains


def to_markdown(pivot, function, runs, metric_title, axis_title):
    cols = list(pivot.columns)
    widths = [max(len(str(c)), *(len(str(v)) for v in pivot[c])) for c in cols]
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    rows = ["| " + " | ".join(str(row[c]).ljust(w) for c, w in zip(cols, widths)) + " |"
            for _, row in pivot.iterrows()]
    return "\n".join([f"### {function} — {metric_title} vs {axis_title}, "
                      f"mean ± std (ms), {runs} runs",
                      "", header, sep, *rows])


def to_latex(pivot, function, n_trains, runs, metric_title, suffix, axis_meta):
    axis_title = axis_meta["title"]
    axis_latex = axis_meta["latex"]
    cols = "lll" + "r" * len(n_trains)
    header = ("Method & Device & " + axis_latex + " & "
              + " & ".join(f"$n={n}$" for n in n_trains) + r" \\")
    body = []
    for _, row in pivot.iterrows():
        cells = " & ".join(str(row[f"n={n}"]).replace("±", r"$\pm$") for n in n_trains)
        body.append(f"{row['Method']} & {row['Device']} & {row[axis_title]} & {cells}" + r" \\")
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Mean $\pm$ std wall-clock time (milliseconds) of the {metric_title} "
        rf"of the acquisition function on the \textsc{{{function}}} benchmark, as a "
        rf"function of the number of training points $n$ per GP and of the {axis_title} "
        rf"{axis_latex} (the other complexity knobs held at baseline), averaged over "
        rf"{runs} runs. dcKG is split into one decoupled source and its full coupled cKG "
        rf"source; cEI has no {axis_title} knob and is shown as a reference row.}}",
        rf"\label{{tab:timing_{function.lower()}_{suffix}_{axis_meta['col']}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def main():
    parser = argparse.ArgumentParser(description="Build timing-study tables.")
    parser.add_argument("--input", type=str, default=os.path.join("results", "timing_study"),
                        help="Directory containing timing_*.csv files (default: results/timing_study)")
    args = parser.parse_args()

    df = load_results(args.input)
    all_aggs = []
    for metric in METRICS.values():
        for axis_col, axis_meta in AXES.items():
            axis_meta = {**axis_meta, "col": axis_col}
            sub = filter_to_axis(df, axis_col, axis_meta["others"])
            agg = aggregate(sub, metric["methods"], metric["title"], axis_col)
            if agg is None or agg.empty:
                continue
            all_aggs.append(agg)
            for function in sorted(agg["function"].unique()):
                agg_fn = agg[agg["function"] == function]
                pivot, n_trains = build_pivot(agg_fn, metric["methods"], axis_meta)
                runs = int(agg_fn["runs"].max())
                print(to_markdown(pivot, function, runs, metric["title"], axis_meta["title"]))
                print()
                tex_path = os.path.join(
                    args.input,
                    f"timing_table_{function}_{metric['suffix']}_{axis_col}.tex")
                with open(tex_path, "w") as f:
                    f.write(to_latex(pivot, function, n_trains, runs,
                                     metric["title"], metric["suffix"], axis_meta) + "\n")
                print(f"LaTeX table written to {tex_path}\n")

    if not all_aggs:
        raise SystemExit("No matching timing rows found.")
    summary_path = os.path.join(args.input, "timing_summary.csv")
    pd.concat(all_aggs, ignore_index=True).to_csv(summary_path, index=False)
    print(f"Aggregated summary written to {summary_path}")


if __name__ == "__main__":
    main()
