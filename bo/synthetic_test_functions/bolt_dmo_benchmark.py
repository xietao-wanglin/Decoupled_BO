import os
import pickle
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from bo.device_utils import DTYPE as dtype
from bo.synthetic_test_functions.bolt_emulator import DMCurriculumMOEmulator
from bo.synthetic_test_functions.synthetic_test_functions import SingleObjectiveProblem


class DecoupledBOLTDMO(SingleObjectiveProblem):
    """Decoupled constrained benchmark built from BoLT's DMCurriculumMO emulator.

    maximize MATH-500(x) s.t. IFEval(x) >= tau_if, MBPP+(x) >= tau_mbpp.

    Sources: 0 = MATH-500 objective, 1 = IFEval constraint, 2 = MBPP+ constraint.
    The emulator returns all three scores jointly, but evaluate_task reveals only
    the queried source. The 4D box input [u1, v1, u2, v2] is mapped to BoLT's 6D
    two-simplex representation, so the known simplex constraints are handled by
    parameterization rather than modelled as black-box constraints.
    Thresholds and the reference optimum come from a fixed Sobol reference set
    (see bolt_dmo_data/generate_reference_set.py).
    """
    _bounds = [(0., 1.), (0., 1.), (0., 1.), (0., 1.)]

    # Emulator output column order.
    IFEVAL_COL, MATH_COL, MBPP_COL = 0, 1, 2

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 4
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=dtype).transpose(-1, -2)
        self.C = 2
        self.M = self.C + 1

        self.problem = DMCurriculumMOEmulator()

        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, "bolt_dmo_data", "bolt_dmo_reference.pickle")
        with open(file_path, 'rb') as f:
            reference = pickle.load(f)

        Y = reference["Y"]
        if_scores, math_scores, mbpp_scores = Y[:, self.IFEVAL_COL], Y[:, self.MATH_COL], Y[:, self.MBPP_COL]
        tau_if = np.quantile(if_scores, reference["quantile_if"])
        tau_mbpp = np.quantile(mbpp_scores, reference["quantile_mbpp"])

        feasible = (if_scores >= tau_if) & (mbpp_scores >= tau_mbpp)
        print('number of feasible points:', feasible.sum(), 'of', len(feasible))

        self.g_thresholds = torch.tensor(self.output_data_transform(np.array([tau_if, tau_mbpp])), dtype=dtype)
        math_logit = self.output_data_transform(math_scores)
        self.CONSTRAINED_MAX = np.max(math_logit[feasible])
        self.GLOBAL_MAX = np.max(math_logit)
        best_idx = np.argmax(np.where(feasible, math_logit, -np.inf))
        self.x_star_ref = torch.tensor(reference["Z"][best_idx], dtype=dtype)

        print('CONSTRAINED_MAX:', self.CONSTRAINED_MAX, 'GLOBAL_MAX:', self.GLOBAL_MAX)

    def output_data_transform(self, value):
        value = np.clip(value, 1e-5, 1 - 1e-5)
        return np.log(value / (1 - value))

    def transform_inputs(self, Z: Tensor) -> Tensor:
        """Map z = [u1, v1, u2, v2] in [0,1]^4 to the 6D two-simplex input
        [IF_1, Math_1, Code_1, IF_2, Math_2, Code_2]; p = [u, (1-u)v, (1-u)(1-v)] per stage."""
        Z = torch.atleast_2d(Z)
        u1, v1, u2, v2 = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
        return torch.stack([
            u1, (1 - u1) * v1, (1 - u1) * (1 - v1),
            u2, (1 - u2) * v2, (1 - u2) * (1 - v2),
        ], dim=-1)

    def _evaluate_emulator(self, X: Tensor) -> Tensor:
        """Single internal emulator call; returns (n, 3) columns [objective, c1, c2]:
        [logit(MATH), tau_if - logit(IFEval), tau_mbpp - logit(MBPP+)]."""
        X_bolt = self.transform_inputs(X).to(torch.double).cpu()
        scores = self.problem(X_bolt)
        scores = torch.tensor(self.output_data_transform(scores.numpy()), dtype=dtype)
        return torch.stack([
            scores[:, self.MATH_COL],
            self.g_thresholds[0] - scores[:, self.IFEVAL_COL],
            self.g_thresholds[1] - scores[:, self.MBPP_COL],
        ], dim=-1)

    def evaluate_task(self, X: Tensor, task_idx) -> Tensor:
        if task_idx not in (0, 1, 2):
            raise ValueError(f"task_idx must be in {{0, 1, 2}}, got {task_idx}")
        # The emulator computes all sources jointly, but only the queried one is revealed.
        return self._evaluate_emulator(X)[:, task_idx].reshape(-1)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        return self._evaluate_emulator(X)

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        return self.evaluate_slack_true(X)

    def evaluate_true(self, X: Tensor) -> Tensor:
        pass

    def is_expensive(self):
        return False

    def is_noisy(self):
        return False

    def get_name(self):
        return "bolt_dmo"

    def get_number_of_constraints(self):
        return self.C

    def get_penalty(self):
        return 3.0
