"""One-off script to generate the fixed reference set for the BoLT DMCurriculumMO benchmark.

Sobol-samples the 4D box, maps each point to BoLT's 6D two-simplex representation,
evaluates the noise-free emulator once, and saves everything to bolt_dmo_reference.pickle.
Prints a threshold calibration table so the quantiles can be chosen to give a joint
feasible rate in the 10-30% range.

Run from the repo root:
    python bo/synthetic_test_functions/bolt_dmo_data/generate_reference_set.py
"""
import os
import pickle

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from bo.synthetic_test_functions.bolt_emulator import DMCurriculumMOEmulator

N_POINTS = 20_000
SEED = 0
# Quantiles used for the committed thresholds (adjust after inspecting the calibration table).
QUANTILE_IF = 0.6
QUANTILE_MBPP = 0.6

# Emulator output column order.
IFEVAL_COL, MATH_COL, MBPP_COL = 0, 1, 2


def map_box_to_simplices(Z: torch.Tensor) -> torch.Tensor:
    """Map z = [u1, v1, u2, v2] in [0,1]^4 to BoLT's 6D input [IF_1, Math_1, Code_1, IF_2, Math_2, Code_2].

    Each (u, v) pair parameterises one 3-component simplex: p = [u, (1-u)v, (1-u)(1-v)].
    """
    u1, v1, u2, v2 = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
    return torch.stack([
        u1, (1 - u1) * v1, (1 - u1) * (1 - v1),
        u2, (1 - u2) * v2, (1 - u2) * (1 - v2),
    ], dim=-1)


def logit(value):
    value = np.clip(value, 1e-5, 1 - 1e-5)
    return np.log(value / (1 - value))


def main():
    sobol = SobolEngine(dimension=4, scramble=True, seed=SEED)
    Z = sobol.draw(N_POINTS, dtype=torch.double)
    X_bolt = map_box_to_simplices(Z)

    problem = DMCurriculumMOEmulator()
    Y = problem(X_bolt).numpy()  # (N, 3): [IFEval, MATH-500, MBPP+]

    if_scores, math_scores, mbpp_scores = Y[:, IFEVAL_COL], Y[:, MATH_COL], Y[:, MBPP_COL]

    print(f"Reference set: {N_POINTS} Sobol points, seed {SEED}")
    for name, s in [("IFEval", if_scores), ("MATH-500", math_scores), ("MBPP+", mbpp_scores)]:
        print(f"  {name}: min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f}")

    print("\nThreshold calibration (joint feasible rate):")
    print(f"{'q_if':>6} {'q_mbpp':>7} {'tau_if':>8} {'tau_mbpp':>9} {'feasible':>9}")
    for q_if in (0.5, 0.6, 0.7, 0.8):
        for q_mbpp in (0.5, 0.6, 0.7, 0.8):
            tau_if = np.quantile(if_scores, q_if)
            tau_mbpp = np.quantile(mbpp_scores, q_mbpp)
            rate = np.mean((if_scores >= tau_if) & (mbpp_scores >= tau_mbpp))
            print(f"{q_if:>6} {q_mbpp:>7} {tau_if:>8.4f} {tau_mbpp:>9.4f} {rate:>9.2%}")

    tau_if = np.quantile(if_scores, QUANTILE_IF)
    tau_mbpp = np.quantile(mbpp_scores, QUANTILE_MBPP)
    feasible = (if_scores >= tau_if) & (mbpp_scores >= tau_mbpp)
    feasible_rate = feasible.mean()
    best_idx = np.argmax(np.where(feasible, math_scores, -np.inf))

    print(f"\nChosen quantiles: q_if={QUANTILE_IF}, q_mbpp={QUANTILE_MBPP}")
    print(f"tau_if={tau_if:.6f}, tau_mbpp={tau_mbpp:.6f}, feasible rate={feasible_rate:.2%}")
    print(f"CONSTRAINED_MAX (raw MATH) = {math_scores[best_idx]:.6f}")
    print(f"CONSTRAINED_MAX (logit)    = {logit(math_scores[best_idx]):.6f}")
    print(f"GLOBAL_MAX (logit)         = {logit(math_scores.max()):.6f}")
    print(f"logit(MATH) range: [{logit(math_scores.min()):.4f}, {logit(math_scores.max()):.4f}]"
          f" -> check get_penalty() is below the feasible optimum minus this range")
    print(f"x_star_ref (4D box)  = {Z[best_idx].numpy()}")
    print(f"x_star_ref (6D bolt) = {X_bolt[best_idx].numpy()}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bolt_dmo_reference.pickle")
    with open(out_path, "wb") as f:
        pickle.dump({
            "Z": Z.numpy(),
            "X_bolt": X_bolt.numpy(),
            "Y": Y,
            "output_order": ["IFEval", "MATH-500", "MBPP+"],
            "quantile_if": QUANTILE_IF,
            "quantile_mbpp": QUANTILE_MBPP,
            "seed": SEED,
        }, f)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
