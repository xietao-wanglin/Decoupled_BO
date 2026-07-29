"""Render the paper-ready Mystery timing table from Launcher_timing.py output.

Reads the summary CSVs and the metadata JSON written by ``Launcher_timing.py``
(no statistic is recomputed here), prints a Markdown version to stdout and
writes ``mystery_timing_table.tex``: a two-column-wide table with one body row
per training size, grouping the steady-state forward times (ms), the complete
acquisition-optimisation times (s) and the directly measured sequential BO
iteration times (s) of the dcKG and cEI algorithms. Every cell is
``median [q25, q75]``.

Usage:
    python timing_table.py
    python timing_table.py --input results/timing_study
    python timing_table.py --font-size footnotesize
"""
import argparse
import csv
import json
import os

from Launcher_timing import ACQUISITIONS, N_TRAIN_VALUES

# Compact column headings, explained in the caption.
ACQUISITION_LABELS = {"dckg_objective": "dcKG-f",
                      "dckg_constraint": "dcKG-c",
                      "coupled_ckg": "cKG",
                      "cei": "cEI"}

FORWARD_SUMMARY = "mystery_forward_summary.csv"
OPT_SUMMARY = "mystery_opt_summary.csv"
LOOP_SUMMARY = "mystery_bo_loop_summary.csv"
CEI_LOOP_SUMMARY = "mystery_cei_loop_summary.csv"
METADATA = "mystery_timing_metadata.json"
TABLE = "mystery_timing_table.tex"

FONT_SIZES = ["normalsize", "small", "footnotesize", "scriptsize"]


def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run Launcher_timing.py first.")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_metadata(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value):
    if value is None:
        return "--"
    if abs(value) < 1:
        return f"{value:.3f}"
    if abs(value) < 100:
        return f"{value:.2f}"
    return f"{value:.1f}"


def format_cell(median, q25, q75):
    """``median [q25, q75]``, or ``--`` when the measurement is missing."""
    median, q25, q75 = _float(median), _float(q25), _float(q75)
    if median is None:
        return "--"
    return f"{_fmt(median)} [{_fmt(q25)}, {_fmt(q75)}]"


def index_by_acquisition(rows, median_col, q25_col, q75_col):
    """{(n_train, acquisition): cell string}"""
    return {(int(row["n_train"]), row["acquisition"]):
            format_cell(row[median_col], row[q25_col], row[q75_col])
            for row in rows}


def index_by_n_train(rows):
    """{n_train: cell string} for a loop summary."""
    return {int(row["n_train"]): format_cell(row["total_loop_median_s"],
                                             row["total_loop_q25_s"],
                                             row["total_loop_q75_s"])
            for row in rows}


def build_rows(input_dir, n_train_values):
    forward = index_by_acquisition(read_csv(os.path.join(input_dir, FORWARD_SUMMARY)),
                                   "forward_median_ms", "forward_q25_ms", "forward_q75_ms")
    optimisation = index_by_acquisition(read_csv(os.path.join(input_dir, OPT_SUMMARY)),
                                        "optimise_median_s", "optimise_q25_s", "optimise_q75_s")
    dckg_loop = index_by_n_train(read_csv(os.path.join(input_dir, LOOP_SUMMARY)))
    cei_loop = index_by_n_train(read_csv(os.path.join(input_dir, CEI_LOOP_SUMMARY)))

    rows = []
    for n_train in n_train_values:
        cells = [forward.get((n_train, a), "--") for a in ACQUISITIONS]
        cells += [optimisation.get((n_train, a), "--") for a in ACQUISITIONS]
        cells += [dckg_loop.get(n_train, "--"), cei_loop.get(n_train, "--")]
        rows.append((n_train, cells))
    return rows


def to_markdown(rows):
    labels = [ACQUISITION_LABELS[a] for a in ACQUISITIONS]
    header = (["Training points"]
              + [f"fwd {label} [ms]" for label in labels]
              + [f"opt {label} [s]" for label in labels]
              + ["dcKG iteration [s]", "cEI iteration [s]"])
    table = [header] + [[str(n)] + cells for n, cells in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = ["| " + " | ".join(c.ljust(w) for c, w in zip(table[0], widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
              for row in table[1:]]
    return "\n".join(lines)


def to_latex(rows, metadata, font_size):
    labels = [ACQUISITION_LABELS[a] for a in ACQUISITIONS]
    n_acqf = len(labels)
    threads = metadata.get("threads")
    if threads == 1:
        thread_phrase = r"measured on a single CPU thread"
    elif threads:
        thread_phrase = rf"measured on CPU with {threads} threads"
    else:
        thread_phrase = r"measured on CPU"
    caption = (
        r"Wall-clock timings of the dcKG and cEI acquisitions on the \textsc{Mystery} benchmark, "
        + thread_phrase +
        r", as a function of the number of training points $n$. The same $n$ "
        r"observations are used for the objective GP and for the constraint GP. "
        r"\emph{Forward evaluation} reports a steady-state acquisition evaluation (the GP "
        r"posterior caches are already built); note that the dcKG acquisitions evaluate a "
        r"candidate plus a 64-point discretisation ($q=65$) whereas cEI evaluates a single "
        r"candidate ($q=1$), so per-call forward times are not directly comparable across the "
        r"two families. \emph{Acquisition optimisation} reports the complete production "
        r"optimisation of one acquisition -- raw-sample generation, restart initialisation, "
        r"candidate and adaptive-discretisation optimisation, all L-BFGS-B restarts and the "
        r"selection of the best candidate -- and excludes GP fitting and acquisition "
        r"construction. The \emph{sequential BO iteration} columns each report one complete "
        r"iteration of the corresponding algorithm, timed with a single outer timer: GP fitting, "
        r"acquisition construction, the sequential optimisation of every acquisition the "
        r"algorithm uses (all three for dcKG, one for cEI), source selection, evaluation of the "
        r"selected source(s) and the dataset update. Both are measured directly and are not the "
        r"sum of the individual columns. dcKG-f is the decoupled objective acquisition, dcKG-c "
        r"the decoupled constraint acquisition, cKG the coupled knowledge-gradient acquisition "
        r"and cEI the coupled constrained expected improvement. Every entry reports the median "
        r"and, in brackets, the interquartile range $[q_{25}, q_{75}]$"
    )
    repeats = metadata.get("loop_repeats")
    if repeats:
        plural = "repetition" if repeats == 1 else "repetitions"
        caption += rf" over {repeats} {plural} of the BO iteration."
    else:
        caption += "."

    forward_span = rf"\multicolumn{{{n_acqf}}}{{c}}{{Forward evaluation [ms]}}"
    opt_span = rf"\multicolumn{{{n_acqf}}}{{c}}{{Acquisition optimisation [s]}}"
    loop_span = r"\multicolumn{2}{c}{Sequential BO iteration [s]}"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:mystery_timing}",
        rf"\{font_size}",
        rf"\begin{{tabular}}{{{'r' * (1 + 2 * n_acqf + 2)}}}",
        r"\toprule",
        rf"Training & {forward_span} & {opt_span} & {loop_span} \\",
        (rf"\cmidrule(lr){{2-{1 + n_acqf}}} "
         rf"\cmidrule(lr){{{2 + n_acqf}-{1 + 2 * n_acqf}}} "
         rf"\cmidrule(lr){{{2 + 2 * n_acqf}-{3 + 2 * n_acqf}}}"),
        ("points & " + " & ".join(labels + labels + ["dcKG", "cEI"]) + r" \\"),
        r"\midrule",
    ]
    lines += [f"{n} & " + " & ".join(cells) + r" \\" for n, cells in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Render the Mystery timing table.")
    parser.add_argument("--input", type=str, default=os.path.join("results", "timing_study"),
                        help="Directory with the Launcher_timing.py output "
                             "(default: results/timing_study)")
    parser.add_argument("--font-size", type=str, choices=FONT_SIZES, default="small",
                        help="LaTeX size command applied to the tabular; ten timing cells is a "
                             "wide row (default: small)")
    args = parser.parse_args()

    metadata = read_metadata(os.path.join(args.input, METADATA))
    n_train_values = metadata.get("n_train_values") or N_TRAIN_VALUES
    # The table always shows the four paper training sizes, even if a partial
    # run only covered some of them (missing cells become "--").
    n_train_values = sorted(set(N_TRAIN_VALUES) | set(int(n) for n in n_train_values))

    rows = build_rows(args.input, n_train_values)
    print(to_markdown(rows))
    print()

    tex_path = os.path.join(args.input, TABLE)
    with open(tex_path, "w") as f:
        f.write(to_latex(rows, metadata, args.font_size) + "\n")
    print(f"LaTeX table written to {tex_path}")


if __name__ == "__main__":
    main()
