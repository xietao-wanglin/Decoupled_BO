import subprocess

def run_script_with_seed(seed):
    """Run the optimization script with a specific seed."""
    print(f"Running script with seed: {seed}")
    try:
        subprocess.run(["python", "Launcher_All.py", str(seed)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred for seed {seed}: {e}")
    print(f"Finished running seed: {seed}")

if __name__ == "__main__":
    for seed in range(20):  # Loop over seeds 1 through 15
        run_script_with_seed(seed)