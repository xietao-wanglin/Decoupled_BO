import argparse
import subprocess
import sys
from pathlib import Path

# Change this to the actual filename of your current experiment script
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
        help="Quick smoke test: run only seeds 0-2",
    )
    args = parser.parse_args()

    min_seed = 0
    max_seed = 1 if args.smoke else 40

    runs = [
        # (function_name, decoupled, cost, ablation)
        # ablation: None (default algorithms), "pure", "nocoupled" or "both"
        ("TestFunc3", True, None, "pure"),
        # ("TestFunc3", True, None, "nocoupled"),
        # ("MysteryRedundant", True, None, None),
        # ("MysteryRedundant", False, None, None)
        # ("TestFunc3RedundantNoNoise", True, None, None),
        # Unequal-cost decoupled experiments (cost=5 makes each source expensive in turn).
        # ("Mystery", True, 5, None),      # 1 constraint -> [5,1] / [1,5]
        # ("TestFunc3", True, 5, None),    # 3 constraints -> [5,1,1,1] / [1,5,1,1] / [1,1,5,1] / [1,1,1,5]
        # ("Branin", True, 5, None),       # 1 constraint -> [5,1] / [1,5]
        # ("PressureVessel", False, None, None),
        # ("SpeedReducer", True, None, None),
        # ("two_layer_cnn_discrete", True, None, None),
        # ("WeldedBeam", True, None, None),
        # ("TensionCompression", True, None, None),
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
