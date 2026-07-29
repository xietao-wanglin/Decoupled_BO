import math
from abc import abstractmethod
from typing import Optional

import torch
from botorch.models.transforms import Bilog
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.test_functions.utils import round_nearest
from botorch.utils.transforms import unnormalize
from torch import Tensor


class SingleObjectiveProblem(ConstrainedBaseTestProblem):

    @abstractmethod
    def is_noisy(self):
        pass

    @abstractmethod
    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        pass

    @abstractmethod
    def is_expensive(self):
        pass

    def get_objective_transform(self):
        """Return an OutcomeTransform for the objective GP, or None for default (Standardize)."""
        return None


class HeterogeneousNoiseProblem(SingleObjectiveProblem):
    """Base for problems with per-output heterogeneous noise.

    Subclasses implement get_noise_per_output(). Returns is_noisy()=True so the
    GP model wrapper uses SingleTaskGP without train_Yvar and learns the noise.

    _apply_noise() adds observation noise to outputs whose get_noise_per_output()
    entry is a float > 1e-6 (STD = sqrt(v)). Outputs with None are near-deterministic
    and returned unchanged.
    """

    @abstractmethod
    def get_noise_per_output(self):
        pass

    def is_noisy(self):
        return True

    def _apply_noise(self, val: Tensor, output_idx: int, is_repeated: bool = False) -> Tensor:
        if is_repeated:
            return val
        v = self.get_noise_per_output()[output_idx]
        if v is not None and float(v) > 1e-6:
            return val + torch.randn_like(val) * math.sqrt(float(v))
        return val


class ConstrainedBraninNew(SingleObjectiveProblem):
    _bounds = [(-5.0, 10.0), (0.0, 15.0)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def is_expensive(self):
        return False

    def get_number_of_constraints(self):
        return 1

    def get_penalty(self):
        return 4.0

    def get_name(self):
        return "constrained_branin"

    def is_noisy(self):
        return False

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return -(X_tf[..., 0] - 10) ** 2 - (X_tf[..., 1] - 15) ** 2

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        t1 = ((X_tf[..., 1]
               - 5.1 / (4 * math.pi ** 2) * (X_tf[..., 0] ** 2)
               + (5 / math.pi) * X_tf[..., 0]) - 6) ** 2
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X_tf[..., 0])
        return t1 + t2 + 5

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        y = self.forward(X).reshape(-1, 1)
        c1 = self.evaluate_slack_true(X).reshape(-1, 1)  #
        return torch.concat([y, c1], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert task_index <= 1, "Maximum of 2 Outputs allowed (task_index <= 1)"
        assert task_index >= 0, "No negative values for task_index allowed"
        if task_index == 0:
            return self.forward(X)
        elif task_index == 1:
            return self.evaluate_slack_true(X)
        else:
            print("Error evaluate_task")
            raise


class MysteryFunctionSuperRedundant(SingleObjectiveProblem):
    _bounds = [(0.0, 5.0), (0.0, 5.0)]

    def __init__(self, noise_std=0.0, negate=False, redundant_constraints=False):
        self.redundant_constraints = redundant_constraints
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def is_noisy(self):
        return False

    def is_expensive(self):
        return False

    def get_number_of_constraints(self):
        if self.redundant_constraints:
            return 9
        else:
            return 1

    def get_penalty(self):
        return 40.0

    def get_name(self):
        if self.redundant_constraints:
            return "mystery_redundant_constraints"
        return "mystery"

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        X_1 = X_tf[..., 0]
        X_2 = X_tf[..., 1]

        t1 = 2.0 + 0.01 * ((X_2 - X_1.pow(2)).pow(2))
        t2 = (1 - X_1).pow(2)
        t3 = 2 * ((2 - X_2).pow(2))
        t4 = 7 * torch.sin(0.5 * X_1) * torch.sin(0.7 * X_1 * X_2)
        return t1 + t2 + t3 + t4

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        pass

    def evaluate_slack1_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return -torch.sin(X_tf[..., 0] - X_tf[..., 1] - math.pi / 8)

    def evaluate_slack2_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[..., 0] * 0.0 + X_tf[..., 0] * 0.0 - 100

    def evaluate_slack3_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[..., 0] * 0.0 + X_tf[..., 0] * 0.0 - 100

    def evaluate_slack4_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[..., 0] * 0.0 + X_tf[..., 0] * 0.0 - 100

    def evaluate_slack5_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[..., 0] * 0.0 + X_tf[..., 0] * 0.0 - 100

    def evaluate_slack6_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[..., 0] * 0.0 + X_tf[..., 0] * 0.0 - 100

    def evaluate_slack7_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[..., 0] * 0.0 + X_tf[..., 0] * 0.0 - 100

    def evaluate_slack8_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[..., 0] * 0.0 + X_tf[..., 0] * 0.0 - 100

    def evaluate_slack9_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[..., 0] * 0.0 + X_tf[..., 0] * 0.0 - 100

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        y = self.forward(X).reshape(-1, 1)
        c1 = self.evaluate_slack1_true(X).reshape(-1, 1)
        c2 = self.evaluate_slack2_true(X).reshape(-1, 1)
        c3 = self.evaluate_slack3_true(X).reshape(-1, 1)
        c4 = self.evaluate_slack4_true(X).reshape(-1, 1)
        c5 = self.evaluate_slack5_true(X).reshape(-1, 1)
        c6 = self.evaluate_slack6_true(X).reshape(-1, 1)
        c7 = self.evaluate_slack7_true(X).reshape(-1, 1)
        c8 = self.evaluate_slack8_true(X).reshape(-1, 1)
        c9 = self.evaluate_slack9_true(X).reshape(-1, 1)
        if self.redundant_constraints:
            return torch.concat([y, c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1)
        return torch.concat([y, c1, c2], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert task_index <= 9, "Maximum of 3 Outputs allowed (task_index <= 2)"
        assert task_index >= 0, "No negative values for task_index allowed"
        if task_index == 0:
            return self.forward(X)
        elif task_index == 1:
            return self.evaluate_slack1_true(X)
        elif task_index == 2:
            return self.evaluate_slack2_true(X)
        elif task_index == 3:
            return self.evaluate_slack3_true(X)
        elif task_index == 4:
            return self.evaluate_slack4_true(X)
        elif task_index == 5:
            return self.evaluate_slack5_true(X)
        elif task_index == 6:
            return self.evaluate_slack6_true(X)
        elif task_index == 7:
            return self.evaluate_slack7_true(X)
        elif task_index == 8:
            return self.evaluate_slack8_true(X)
        elif task_index == 9:
            return self.evaluate_slack9_true(X)
        else:
            print("Error evaluate_task")
            raise


class ConstrainedFunc3(SingleObjectiveProblem):
    _bounds = [(0.0, 1.0), (0.0, 1.0)]

    def get_number_of_constraints(self):
        return 3

    def is_noisy(self):
        return False

    def is_expensive(self):
        return False

    def get_penalty(self):
        return 4.0

    def get_name(self):
        return "test_function_3"

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        t1 = (X_tf[..., 0] - 1) ** 2
        t2 = (X_tf[..., 1] - 0.5) ** 2
        return -t1 - t2

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        pass

    def evaluate_slack1_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return ((X_tf[..., 0] - 3) ** 2 + (X_tf[..., 1] + 2) ** 2) * torch.exp(-(X_tf[..., 1]) ** 7) - 12

    def evaluate_slack2_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return 10 * X_tf[..., 0] + X_tf[..., 1] - 7

    def evaluate_slack3_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return (X_tf[..., 0] - 0.5) ** 2 + (X_tf[..., 1] - 0.5) ** 2 - 0.2

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        y = self.forward(X).reshape(-1, 1)
        c1 = self.evaluate_slack1_true(X).reshape(-1, 1)
        c2 = self.evaluate_slack2_true(X).reshape(-1, 1)
        c3 = self.evaluate_slack3_true(X).reshape(-1, 1)
        return torch.concat([y, c1, c2, c3], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert task_index <= 3, "Maximum of 4 Outputs allowed (task_index <= 3)"
        assert task_index >= 0, "No negative values for task_index allowed"
        if task_index == 0:
            return self.forward(X)
        elif task_index == 1:
            return self.evaluate_slack1_true(X)
        elif task_index == 2:
            return self.evaluate_slack2_true(X)
        elif task_index == 3:
            return self.evaluate_slack3_true(X)
        else:
            print("Error evaluate_task")
            raise


class ConstrainedFunc3Redundant(HeterogeneousNoiseProblem, ConstrainedFunc3):
    """ConstrainedFunc3 extended with 2 redundant constraints and heterogeneous noise.

    At most one active constraint (c1, c2, or c3 — chosen via `noisy_active_constraint`)
    carries observation noise at ~20% of its signal std; the other active constraints
    are noiseless. Pass `noisy_active_constraint=None` to keep all three active
    constraints noiseless. Redundant constraints: c4 (noiseless, always -100), c5
    (noisy, always -100). Objective: always noisy (~20% of its signal std).
    """

    # Per-active-constraint noise variance ~= (0.2 * signal_std)^2, estimated by sampling.
    _ACTIVE_NOISE_VAR = {1: 0.1642, 2: 0.3367, 3: 0.0004}

    def __init__(self, noise_std=0.0, negate=False, noisy_active_constraint=2):
        assert noisy_active_constraint in (1, 2, 3, None), \
            "noisy_active_constraint must be 1, 2, 3, or None"
        self.noisy_active_constraint = noisy_active_constraint
        super().__init__(noise_std=noise_std, negate=negate)

    def get_number_of_constraints(self):
        return 5

    def get_name(self):
        if self.noisy_active_constraint == 2:
            return "test_function_3_redundant"
        if self.noisy_active_constraint is None:
            return "test_function_3_redundant_no_noisy_constraint"
        return f"test_function_3_redundant_c{self.noisy_active_constraint}noisy"

    def get_noise_per_output(self):
        # [obj, c1_active, c2_active, c3_active, c4_redundant, c5_redundant]
        # float > 1e-6: noisy — obs noise STD = sqrt(v); fixed GP noise variance v
        # None:         near-deterministic — no external observation noise
        npo = [0.0038, None, None, None, None, 4.0]
        if self.noisy_active_constraint is not None:
            npo[self.noisy_active_constraint] = self._ACTIVE_NOISE_VAR[self.noisy_active_constraint]
        return npo

    def evaluate_slack4_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[..., 0] * 0.0 - 100

    def evaluate_slack5_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[..., 0] * 0.0 - 100

    def _obj_true(self, X: Tensor) -> Tensor:
        val = self.evaluate_true(X)
        return -val if self.negate else val

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        y  = self._apply_noise(self._obj_true(X),                0, is_repeated).reshape(-1, 1)
        c1 = self._apply_noise(self.evaluate_slack1_true(X),     1, is_repeated).reshape(-1, 1)
        c2 = self._apply_noise(self.evaluate_slack2_true(X),     2, is_repeated).reshape(-1, 1)
        c3 = self._apply_noise(self.evaluate_slack3_true(X),     3, is_repeated).reshape(-1, 1)
        c4 = self._apply_noise(self.evaluate_slack4_true(X),     4, is_repeated).reshape(-1, 1)
        c5 = self._apply_noise(self.evaluate_slack5_true(X),     5, is_repeated).reshape(-1, 1)
        return torch.concat([y, c1, c2, c3, c4, c5], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert 0 <= task_index <= 5, "task_index must be in [0, 5]"
        fns = [
            lambda: self._obj_true(X),
            lambda: self.evaluate_slack1_true(X),
            lambda: self.evaluate_slack2_true(X),
            lambda: self.evaluate_slack3_true(X),
            lambda: self.evaluate_slack4_true(X),
            lambda: self.evaluate_slack5_true(X),
        ]
        return self._apply_noise(fns[task_index](), task_index)


class PressureVessel(SingleObjectiveProblem):
    _bounds = [(0.0, 10.0), (0.0, 10.0), (10.0, 50.0), (150.0, 200.0)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 4
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float)
        self.transformation = Bilog()

    def get_number_of_constraints(self):
        return 4

    def get_penalty(self):
        return 269000.0  # Maximum is around 268658.84375

    def get_name(self):
        return "pressure_vessel"

    def is_noisy(self):
        return False

    def is_expensive(self):
        return False

    def get_objective_transform(self):
        return "gaussian_copula"

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds.transpose(-1, -2))
        x1, x2, x3, x4 = X_tf[..., 0], X_tf[..., 1], X_tf[..., 2], X_tf[..., 3]
        x1 = round_nearest(x1, increment=0.0625, bounds=self._bounds[0])
        x2 = round_nearest(x2, increment=0.0625, bounds=self._bounds[1])
        return (
                0.6224 * x1 * x3 * x4
                + 1.7781 * x2 * x3.pow(2)
                + 3.1661 * x1.pow(2) * x4
                + 19.84 * x1.pow(2) * x3
        )

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        pass

    def evaluate_slack1_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds.transpose(-1, -2))
        x1, x3 = X_tf[..., 0], X_tf[..., 2]
        return -x1 + 0.0193 * x3

    def evaluate_slack2_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds.transpose(-1, -2))
        x2, x3 = X_tf[..., 1], X_tf[..., 2]
        return -x2 + 0.00954 * x3

    def evaluate_slack3_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds.transpose(-1, -2))
        x3, x4 = X_tf[..., 2], X_tf[..., 3]
        return -math.pi * x3.pow(2) * x4 - (4 / 3) * math.pi * x3.pow(3) + 1296000.0

    def evaluate_slack4_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds.transpose(-1, -2))
        x4 = X_tf[..., 3]
        return x4 - 240.0

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        y = self.forward(X).reshape(-1, 1)
        c1 = self.evaluate_slack1_true(X).reshape(-1, 1)
        c2 = self.evaluate_slack2_true(X).reshape(-1, 1)
        c3 = self.evaluate_slack3_true(X).reshape(-1, 1)
        c4 = self.evaluate_slack4_true(X).reshape(-1, 1)
        out = torch.concat([y, c1, c2, c3, c4], dim=1)
        out_transformed = out.clone()
        out_transformed[..., 1:] = self.transform_(out[..., 1:])
        return out_transformed

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert 0 <= task_index <= 4, "Task index must be between 0 and 4"
        if task_index == 0:
            return self.forward(X)
        elif task_index == 1:
            constraint_1_raw = self.evaluate_slack1_true(X)
            return self.transform_(constraint_1_raw)
        elif task_index == 2:
            constraint_2_raw = self.evaluate_slack2_true(X)
            return self.transform_(constraint_2_raw)
        elif task_index == 3:
            constraint_3_raw = self.evaluate_slack3_true(X)
            return self.transform_(constraint_3_raw)
        elif task_index == 4:
            constraint_4_raw = self.evaluate_slack4_true(X)
            return self.transform_(constraint_4_raw)
        else:
            raise ValueError("Invalid task index")

    def transform_(self, raw_values):
        logits = self.transformation(torch.atleast_1d(raw_values))
        logits = logits[0].view(raw_values.shape)
        return logits


class SpeedReducer(SingleObjectiveProblem):
    _bounds = [(2.6, 3.6), (0.7, 0.8), (17.0, 28.0), (7.3, 8.3), (7.8, 8.3), (2.9, 3.9), (5.0, 5.5)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 7
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)
        self.transformation = Bilog()

    def get_number_of_constraints(self):
        return 11

    def is_noisy(self):
        return False

    def get_penalty(self):
        return 4500.0

    def is_expensive(self):
        return False

    def get_name(self):
        return "speed_reducer   "

    def get_objective_transform(self):
        return "gaussian_copula"

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return 0.7854 * x1 * (x2 ** 2) * (3.3333 * (x3 ** 2) + 14.9334 * x3 - 43.0934) - 1.508 * x1 * (
                x6 ** 2 + x7 ** 2) + 7.4777 * (x6 ** 3 + x7 ** 3) + 0.7854 * (x4 * (x6 ** 2) + x5 * (x7 ** 2))

    def get_coordinates(self, X_tf):
        return X_tf[..., 0], X_tf[..., 1], X_tf[..., 2], X_tf[..., 3], X_tf[..., 4], X_tf[..., 5], X_tf[..., 6]

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        pass

    def evaluate_slack1_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return 27.0 * (1 / x1) * (1 / (x2 ** 2)) * (1 / x3) - 1

    def evaluate_slack2_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return 397.5 * (1 / x1) * (1 / (x2 ** 2)) * (1 / (x3 ** 2)) - 1

    def evaluate_slack3_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return 1.93 * (1 / x2) * (1 / x3) * (x4 ** 3) * (1 / (x6 ** 4)) - 1

    def evaluate_slack4_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return 1.93 * (1 / x2) * (1 / x3) * (x5 ** 3) * (1 / (x7 ** 4)) - 1

    def evaluate_slack5_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return 1 / (0.1 * (x6 ** 3)) * torch.sqrt((745 * x4 / (x2 * x3)) ** 2 + 16.9 * 1e6) - 1100

    def evaluate_slack6_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return 1 / (0.1 * (x7 ** 3)) * torch.sqrt((745 * x5 / (x2 * x3)) ** 2 + 157.5 * 1e6) - 850

    def evaluate_slack7_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return x2 * x3 - 40

    def evaluate_slack8_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return 5 - x1 / x2

    def evaluate_slack9_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return x1 / x2 - 12

    def evaluate_slack10_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return (1.5 * x6 + 1.9) / x4 - 1

    def evaluate_slack11_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4, x5, x6, x7 = self.get_coordinates(X_tf)
        return (1.1 * x7 + 1.9) / x5 - 1

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        y = self.forward(X).reshape(-1, 1)
        c1 = self.evaluate_slack1_true(X).reshape(-1, 1)
        c2 = self.evaluate_slack2_true(X).reshape(-1, 1)
        c3 = self.evaluate_slack3_true(X).reshape(-1, 1)
        c4 = self.evaluate_slack4_true(X).reshape(-1, 1)
        c5 = self.evaluate_slack5_true(X).reshape(-1, 1)
        c6 = self.evaluate_slack6_true(X).reshape(-1, 1)
        c7 = self.evaluate_slack7_true(X).reshape(-1, 1)
        c8 = self.evaluate_slack8_true(X).reshape(-1, 1)
        c9 = self.evaluate_slack9_true(X).reshape(-1, 1)
        c10 = self.evaluate_slack10_true(X).reshape(-1, 1)
        c11 = self.evaluate_slack11_true(X).reshape(-1, 1)
        out = torch.concat([y, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11], dim=1)
        out_transformed = out.clone()
        out_transformed[..., 1:] = self.transform_(out[..., 1:])
        return out_transformed

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert 0 <= task_index <= 11, "Task index must be between 0 and 11"
        if task_index == 0:
            return self.forward(X)
        elif task_index == 1:
            constraint_1_raw = self.evaluate_slack1_true(X)
            return self.transform_(constraint_1_raw)
        elif task_index == 2:
            constraint_2_raw = self.evaluate_slack2_true(X)
            return self.transform_(constraint_2_raw)
        elif task_index == 3:
            constraint_3_raw = self.evaluate_slack3_true(X)
            return self.transform_(constraint_3_raw)
        elif task_index == 4:
            constraint_4_raw = self.evaluate_slack4_true(X)
            return self.transform_(constraint_4_raw)
        elif task_index == 5:
            constraint_5_raw = self.evaluate_slack5_true(X)
            return self.transform_(constraint_5_raw)
        elif task_index == 6:
            constraint_6_raw = self.evaluate_slack6_true(X)
            return self.transform_(constraint_6_raw)
        elif task_index == 7:
            constraint_7_raw = self.evaluate_slack7_true(X)
            return self.transform_(constraint_7_raw)
        elif task_index == 8:
            constraint_8_raw = self.evaluate_slack8_true(X)
            return self.transform_(constraint_8_raw)
        elif task_index == 9:
            constraint_9_raw = self.evaluate_slack9_true(X)
            return self.transform_(constraint_9_raw)
        elif task_index == 10:
            constraint_10_raw = self.evaluate_slack10_true(X)
            return self.transform_(constraint_10_raw)
        elif task_index == 11:
            constraint_11_raw = self.evaluate_slack11_true(X)
            return self.transform_(constraint_11_raw)
        else:
            raise ValueError("Invalid task index")

    def transform_(self, raw_values):
        logits = self.transformation(torch.atleast_1d(raw_values))
        logits = logits[0].view(raw_values.shape)
        return logits
