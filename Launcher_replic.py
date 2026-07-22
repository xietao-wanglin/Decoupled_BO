import argparse
import subprocess
import sys
from pathlib import Path

# Change this to the actual filename of your current experiment script
TARGET_SCRIPT = Path(__file__).with_name("Launcher.py")


def run_command(function_name: str, decoupled: bool, min_seed: int, max_seed: int, cost=None):
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
        # (function_name, decoupled, cost)
        # ("MysteryRedundant", True, None),
        # ("MysteryRedundant", False, None)
        ("TestFunc3RedundantNoNoise", True, None)
        # Unequal-cost decoupled experiments (cost=5 makes each source expensive in turn).
        # ("Mystery", True, 5),      # 1 constraint -> [5,1] / [1,5]
        # ("TestFunc3", True, 5),    # 3 constraints -> [5,1,1,1] / [1,5,1,1] / [1,1,5,1] / [1,1,1,5]
        # ("Branin", True, 5),       # 1 constraint -> [5,1] / [1,5]
        # ("PressureVessel", False, None),
        # ("SpeedReducer", True, None),
        # ("two_layer_cnn_discrete", True, None),
        # ("WeldedBeam", True, None),
        # ("TensionCompression", True, None),
    ]

    for function_name, decoupled, cost in runs:
        run_command(
            function_name=function_name,
            decoupled=decoupled,
            min_seed=min_seed,
            max_seed=max_seed,
            cost=cost,
        )


if __name__ == "__main__":
    main()
