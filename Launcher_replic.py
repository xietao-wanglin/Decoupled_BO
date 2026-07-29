"""Batch driver: runs the full paper sweep by calling Launcher.py once per configuration."""
import argparse
import subprocess
import sys
from pathlib import Path

TARGET_SCRIPT = Path(__file__).with_name("Launcher.py")


def run_command(function_name: str, decoupled: bool, min_seed: int, max_seed: int, cost=None,
                ablation=None):
    cmd = [
        sys.executable,
        str(TARGET_SCRIPT),
        "--function",
        function_name,
        "--min-seed",
        str(min_seed),
        "--max-seed",
        str(max_seed),
    ]

    if decoupled:
        cmd.append("--decoupled")

    if cost is not None:
        cmd += ["--cost", str(cost)]

    if ablation is not None:
        cmd += ["--ablation", ablation]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test: run the whole sweep with seeds 0-1 only",
    )
    args = parser.parse_args()

    min_seed = 0
    max_seed = 1 if args.smoke else 39

    runs = [
        # (function_name, decoupled, cost, ablation)
        # decoupled=False runs cEI + cKG, decoupled=True runs dcKG.
        # ablation: None (default algorithms), "pure", "nocoupled" or "both".

        # --- Equal-cost benchmarks: coupled baselines and dcKG ---
        ("Mystery", False, None, None),
        ("Mystery", True, None, None),
        ("Branin", False, None, None),
        ("Branin", True, None, None),
        ("TestFunc3", False, None, None),
        ("TestFunc3", True, None, None),
        ("MysteryRedundant", False, None, None),
        ("MysteryRedundant", True, None, None),
        ("PressureVessel", False, None, None),
        ("PressureVessel", True, None, None),
        ("SpeedReducer", False, None, None),
        ("SpeedReducer", True, None, None),
        ("two_layer_cnn_discrete", False, None, None),
        ("two_layer_cnn_discrete", True, None, None),
        ("bolt_dmo_10", False, None, None),
        ("bolt_dmo_10", True, None, None),

        # --- Noisy-constraint variants of Test Function 2 ---
        ("TestFunc3Redundant", True, None, None),
        ("TestFunc3RedundantC1", True, None, None),
        ("TestFunc3RedundantC3", True, None, None),
        ("TestFunc3RedundantNoNoise", True, None, None),

        # --- Heterogeneous evaluation costs (cost=5 makes each source expensive in turn) ---
        ("Mystery", True, 5, None),      # 1 constraint -> [5,1] / [1,5]
        ("Branin", True, 5, None),       # 1 constraint -> [5,1] / [1,5]
        ("TestFunc3", True, 5, None),    # 3 constraints -> [5,1,1,1] / [1,5,1,1] / [1,1,5,1] / [1,1,1,5]

        # --- Supplement ablation: dcKG without the coupled cKG candidate ---
        ("TestFunc3", True, None, "both"),
    ]

    for function_name, decoupled, cost, ablation in runs:
        run_command(
            function_name=function_name,
            decoupled=decoupled,
            min_seed=min_seed,
            max_seed=max_seed,
            cost=cost,
            ablation=ablation,
        )


if __name__ == "__main__":
    main()
