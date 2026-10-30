"""Plain numpy wrapper around DecoupledBOLTDMO's MATH-500 objective.

DecoupledBOLTDMO is a torch/BoTorch ConstrainedBaseTestProblem; this exposes
its objective (dropping the constraint outputs) as a scalar-in/scalar-out
numpy function, e.g. for use with optimizers that expect a signature like
mystery_objective(x1, x2, ...).
"""
import numpy as np
import torch

from bo.synthetic_test_functions.bolt_dmo_benchmark import DecoupledBOLTDMO

_problem = None


def _get_problem():
    global _problem
    if _problem is None:
        _problem = DecoupledBOLTDMO(negate=False)
    return _problem


def bolt_dmo(x1, x2, x3, x4):
    """MATH-500 objective of DecoupledBOLTDMO at box input [x1, x2, x3, x4] in [0, 1]^4.

    Each argument is a scalar or a numpy array of matching shape; returns the
    objective value with the same shape (float if all inputs were scalar).
    """
    problem = _get_problem()
    X = np.stack(np.broadcast_arrays(
        np.asarray(x1, dtype=np.float64), np.asarray(x2, dtype=np.float64),
        np.asarray(x3, dtype=np.float64), np.asarray(x4, dtype=np.float64),
    ), axis=-1)
    scalar_input = X.ndim == 1
    X_t = torch.as_tensor(np.atleast_2d(X), dtype=torch.double)
    y = problem.evaluate_black_box(X_t)[:, 0].detach().cpu().numpy()
    return float(y[0]) if scalar_input else y
