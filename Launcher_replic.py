import subprocess
import sys
from pathlib import Path

# Change this to the actual filename of your current experiment script
TARGET_SCRIPT = Path(__file__).with_name("Launcher.py")


def run_command(function_name: str, decoupled: bool, min_seed: int, max_seed: int):
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

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    min_seed = 0
    max_seed = 40

    runs = [
        # ("PressureVessel", False),
        ("Mystery", True)
        # ("MysteryRedundant", True),
        # ("TestFunc3", True),
        # ("Branin", True),
        # ("PressureVessel", True),
        # ("SpeedReducer", True),
        # ("two_layer_cnn_discrete", True),
        # ("WeldedBeam", True),
        # ("TensionCompression", True),
        # ("WeldedBeam", False),
        # ("WeldedBeam", True),
        # ("SpeedReducer", False),
        # ("TensionCompression", False)
        # ("TensionCompression", True),
        # ("SpeedReducer", True)
    ]

    for function_name, decoupled in runs:
        run_command(
            function_name=function_name,
            decoupled=decoupled,
            min_seed=min_seed,
            max_seed=max_seed,
        )


if __name__ == "__main__":
    main()