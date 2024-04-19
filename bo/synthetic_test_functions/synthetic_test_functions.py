import math

import torch
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.utils.transforms import unnormalize
from torch import Tensor


class ConstrainedBranin(ConstrainedBaseTestProblem):
    _bounds = [(-5.0, 10.0), (0.0, 15.0)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        t1 = (
                X_tf[..., 1]
                - 5.1 / (4 * math.pi ** 2) * X_tf[..., 0] ** 2
                + 5 / math.pi * X_tf[..., 0]
                - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X_tf[..., 0])
        return t1 ** 2 + t2 + 10

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return 50 - (X_tf[..., 0:1] - 2.5).pow(2) - (X_tf[..., 1:2] - 7.5).pow(2)

    def evaluate_black_box(self, X: Tensor) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
        c1 = self.evaluate_slack_true(X)
        return torch.concat([y, c1], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert task_index <= 1, "Maximum of 2 Outputs allowed (task_index <= 1)"
        assert task_index >= 0, "No negative values for task_index allowed"
        if task_index == 0:
            return self.evaluate_true(X)
        elif task_index == 1:
            return self.evaluate_slack_true(X)
        else:
            print("Error evaluate_task")
            raise

class MysteryFunction(ConstrainedBaseTestProblem):
    _bounds = [(0.0, 5.0), (0.0, 5.0)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        X_1 = X_tf[..., 0]
        X_2 = X_tf[..., 1]

        t1 = 2.0 + 0.01*((X_2 - X_1.pow(2)).pow(2))
        t2 = (1 - X_1).pow(2)
        t3 = 2*((2-X_2).pow(2))
        t4 = 7*torch.sin(0.5*X_1)*torch.sin(0.7*X_1*X_2)
        return t1 + t2 + t3 + t4

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return -torch.sin(X_tf[..., 0] - X_tf[..., 1] - math.pi/8)

    def evaluate_black_box(self, X: Tensor) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
        c1 = self.evaluate_slack_true(X).reshape(-1, 1)
        print(y.shape, c1.shape)
        return torch.concat([y, c1], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert task_index <= 1, "Maximum of 2 Outputs allowed (task_index <= 1)"
        assert task_index >= 0, "No negative values for task_index allowed"
        if task_index == 0:
            return self.evaluate_true(X)
        elif task_index == 1:
            return self.evaluate_slack_true(X)
        else:
            print("Error evaluate_task")
            raise