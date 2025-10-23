import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from botorch.models.transforms import Bilog
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.test_functions.utils import round_nearest
from botorch.utils.transforms import unnormalize
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from bo.synthetic_test_functions.CNN_Model import TwoLayerCNN


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


class MOPTA08(ConstrainedBaseTestProblem):
    _bounds = [(0.0, 1.0)] * 124

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 124
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)
        sysarch = 64 if sys.maxsize > 2 ** 32 else 32
        machine = platform.machine().lower()
        if machine == "armv7l":
            assert sysarch == 32, "Not supported"
            self.mopta_exectutable = "mopta08_armhf.bin"
        elif machine == "x86_64":
            assert sysarch == 64, "Not supported"
            self.mopta_exectutable = "mopta08_elf64.bin"
        elif machine == "i386":
            assert sysarch == 32, "Not supported"
            self.mopta_exectutable = "mopta08_elf32.bin"
        else:
            raise RuntimeError("Machine with this architecture is not supported")

        self.mopta_full_path = os.path.join(
            Path(__file__).parent, "mopta08", self.mopta_exectutable
        )

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        pass

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        directory_file_descriptor = tempfile.TemporaryDirectory()
        directory_name = directory_file_descriptor.name
        with open(os.path.join(directory_name, "input.txt"), "w+") as tmp_file:
            for _x in X_tf:
                tmp_file.write(f"{_x}\n")
        popen = subprocess.Popen(
            self.mopta_full_path,
            stdout=subprocess.PIPE,
            cwd=directory_name,
        )
        popen.wait()
        output = (
            open(os.path.join(directory_name, "output.txt"), "r")
            .read()
            .split("\n")
        )
        output = [x.strip() for x in output]
        output = torch.tensor([float(x) for x in output if len(x) > 0])
        return output

    def evaluate_black_box(self, X: Tensor) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
        c1 = self.evaluate_slack_true(X).reshape(-1, 1)  #
        return torch.concat([y, c1], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert task_index <= 68, "Maximum of 69 Outputs allowed (task_index <= 68)"
        assert task_index >= 0, "No negative values for task_index allowed"
        return self.evaluate_true(X)[task_index]


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


class MysteryFunction(SingleObjectiveProblem):
    _bounds = [(0.0, 5.0), (0.0, 5.0)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def is_expensive(self):
        return False

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        X_1 = X_tf[..., 0]
        X_2 = X_tf[..., 1]

        t1 = 2.0 + 0.01 * ((X_2 - X_1.pow(2)).pow(2))
        t2 = (1 - X_1).pow(2)
        t3 = 2 * ((2 - X_2).pow(2))
        t4 = 7 * torch.sin(0.5 * X_1) * torch.sin(0.7 * X_1 * X_2)
        return t1 + t2 + t3 + t4

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        y = self.forward(X).reshape(-1, 1)
        c1 = self.evaluate_slack_true(X).reshape(-1, 1)
        print(y.shape, c1.shape)
        return torch.concat([y, c1], dim=1)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return -torch.sin(X_tf[..., 0] - X_tf[..., 1] - math.pi / 8)

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

    def is_noisy(self):
        return False


class MysteryFunctionRedundant(SingleObjectiveProblem):
    _bounds = [(0.0, 5.0), (0.0, 5.0)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def is_noisy(self):
        return False

    def is_expensive(self):
        return False

    def get_number_of_constraints(self):
        return 1

    def get_penalty(self):
        return 40.0

    def get_name(self):
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

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
        c1 = self.evaluate_slack1_true(X).reshape(-1, 1)
        c2 = self.evaluate_slack2_true(X).reshape(-1, 1)
        return torch.concat([y, c1, c2], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert task_index <= 2, "Maximum of 3 Outputs allowed (task_index <= 2)"
        assert task_index >= 0, "No negative values for task_index allowed"
        if task_index == 0:
            return -self.evaluate_true(X)
        elif task_index == 1:
            return self.evaluate_slack1_true(X)
        elif task_index == 2:
            return self.evaluate_slack2_true(X)
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

class BraninHoo(SingleObjectiveProblem):
    _bounds = [(-5.0, 10.0), (0.0, 15.0)]

    def get_number_of_constraints(self):
        return 1
    
    def get_penalty(self):
        return 1
    
    def get_name(self):
        return "braninhoo"
    
    def is_noisy(self):
        return False

    def is_expensive(self):
        return False

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)
    
    def func(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        X_1 = X_tf[..., 0]
        X_2 = X_tf[..., 1]
        a = 1
        b = 5.1/(4*torch.pi**2)
        c = 5 / torch.pi
        r = 6
        s = 10
        t = 1 / (8*torch.pi)
        return a*(X_2 - b*X_1*X_1 + c*X_1 - r)**2 + s*(1-t)*torch.cos(X_1) + s

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self.func(X)

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        y = self.forward(X).reshape(-1, 1)
        c1 = self.evaluate_slack_true(X).reshape(-1, 1)
        print(y.shape, c1.shape)
        return torch.concat([y, c1], dim=1)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        slack = -self.func(X) + 0.6 
        return slack

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


class PressureVessel(SingleObjectiveProblem):
    _bounds = [(0.0, 10.0), (0.0, 10.0), (10.0, 50.0), (150.0, 200.0)]

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

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 4
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float)

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
        print(X)
        y = self.forward(X).reshape(-1, 1)
        print(y)
        c1 = self.evaluate_slack1_true(X).reshape(-1, 1)
        print(c1)
        c2 = self.evaluate_slack2_true(X).reshape(-1, 1)
        c3 = self.evaluate_slack3_true(X).reshape(-1, 1)
        c4 = self.evaluate_slack4_true(X).reshape(-1, 1)
        print(torch.concat([y, c1, c2, c3, c4], dim=1))
        return torch.concat([y, c1, c2, c3, c4], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert 0 <= task_index <= 4, "Task index must be between 0 and 4"
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
        else:
            raise ValueError("Invalid task index")

# TODO: TensionCompresion is not working as it should. Not really optimizing. The EI values are all equal to the penalty.

class TensionCompression(SingleObjectiveProblem):
    # _bounds = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)] # bounds from original paper
    _bounds = [(0.01, 1.0), (0.01, 1.0), (0.01, 20.0)] #botorch bounds
    def get_number_of_constraints(self):
        return 4

    def get_penalty(self):
        return 0.3

    def get_name(self):
        return "tension-compression-string"

    def is_noisy(self):
        return False

    def is_expensive(self):
        return False

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 3
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float)
        self.bilog = Bilog()

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds.transpose(-1, -2))
        x1, x2, x3 = X_tf[..., 0], X_tf[..., 1], X_tf[..., 2]
        Bilog()
        return (x1 ** 2) * x2 * (x3 + 2)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        pass

    def evaluate_slack1_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds.transpose(-1, -2))
        x1, x2, x3 = X_tf[..., 0], X_tf[..., 1], X_tf[..., 2]
        return self.bilog(1 - (x2 ** 3) * x3 / (71785 * (x1 ** 4)))[0]

    def evaluate_slack2_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds.transpose(-1, -2))
        x1, x2, x3 = X_tf[..., 0], X_tf[..., 1], X_tf[..., 2]
        return self.bilog(torch.clip((4 * (x2 ** 2) - x1 * x2) / (12566 * (x1 ** 3) * (x2 - x1)) + 1 / (5108 * (x1 ** 2)) - 1, max=5000))[0]

    def evaluate_slack3_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds.transpose(-1, -2))
        x1, x2, x3 = X_tf[..., 0], X_tf[..., 1], X_tf[..., 2]
        return self.bilog(1 - 140.45 * x1 / (x3 * (x2 ** 2)))[0]

    def evaluate_slack4_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds.transpose(-1, -2))
        x1, x2, x3 = X_tf[..., 0], X_tf[..., 1], X_tf[..., 2]
        return self.bilog((x1 + x2) / 1.5 - 1)[0]

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        y = self.forward(X).reshape(-1, 1)
        c1 = self.evaluate_slack1_true(X).reshape(-1, 1)
        c2 = self.evaluate_slack2_true(X).reshape(-1, 1)
        c3 = self.evaluate_slack3_true(X).reshape(-1, 1)
        c4 = self.evaluate_slack4_true(X).reshape(-1, 1)
        return torch.concat([y, c1, c2, c3, c4], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert 0 <= task_index <= 4, "Task index must be between 0 and 4"
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
        else:
            raise ValueError("Invalid task index")


class SpeedReducer(SingleObjectiveProblem):
    _bounds = [(2.6, 3.6), (0.7, 0.8), (17.0, 28.0), (7.3, 8.3), (7.8, 8.3), (2.9, 3.9), (5.0, 5.5)]

    def get_number_of_constraints(self):
        return 7

    def is_noisy(self):
        return False

    def get_penalty(self):
        return 4500.0

    def is_expensive(self):
        return False

    def get_name(self):
        return "speed_reducer   "

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 7
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

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
        return torch.concat([y, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert 0 <= task_index <= 11, "Task index must be between 0 and 11"
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
        elif task_index == 10:
            return self.evaluate_slack10_true(X)
        elif task_index == 11:
            return self.evaluate_slack11_true(X)
        else:
            raise ValueError("Invalid task index")


class WeldedBeamSO(SingleObjectiveProblem):
    _bounds = [(0.125, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]

    def get_number_of_constraints(self):
        return 5

    def is_noisy(self):
        return False

    def get_penalty(self):
        return 150.0  # Maximum is around 1220.174

    def is_expensive(self):
        return False

    def get_name(self):
        return "welded_beam"

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 4
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4 = X_tf.unbind(-1)
        return 1.10471 * x1.pow(2) * x2 + 0.04811 * x3 * x4 * (14.0 + x2)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        pass

    def evaluate_slack1_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x2, x3, x4 = X_tf[..., 0], X_tf[..., 1], X_tf[..., 2], X_tf[..., 3]
        P, L, E, G, t_max = 6000.0, 14.0, 30e6, 12e6, 13600.0
        M = P * (L + x2 / 2)
        R = torch.sqrt(0.25 * (x2.pow(2) + (x1 + x3).pow(2)))
        J = 2 * math.sqrt(2) * x1 * x2 * (x2.pow(2) / 12 + 0.25 * (x1 + x3).pow(2))
        t1 = P / (math.sqrt(2) * x1 * x2)
        t2 = M * R / J
        t = torch.sqrt(t1.pow(2) + t1 * t2 * x2 / R + t2.pow(2))
        return t - t_max

    def evaluate_slack2_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x2, x3, x4 = X_tf[..., 1], X_tf[..., 2], X_tf[..., 3]
        P, L, s_max = 6000.0, 14.0, 30000.0
        s = 6 * P * L / (x4 * x3.pow(2))
        return s - s_max

    def evaluate_slack3_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x1, x4 = X_tf[..., 0], X_tf[..., 3]
        return x1 - x4

    def evaluate_slack4_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x3, x4 = X_tf[..., 2], X_tf[..., 3]
        P, L, E, d_max = 6000.0, 14.0, 30e6, 0.25
        d = 4 * P * L ** 3 / (E * x3.pow(3) * x4)
        return d - d_max

    def evaluate_slack5_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        x3, x4 = X_tf[..., 2], X_tf[..., 3]
        P, L, E, G = 6000.0, 14.0, 30e6, 12e6
        P_c = (
                4.013 * E * x3 * x4.pow(3) * 6 / (L ** 2)
                * (1 - 0.25 * x3 * math.sqrt(E / G) / L)
        )
        return P - P_c

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        y = self.forward(X).reshape(-1, 1)
        c1 = self.evaluate_slack1_true(X).reshape(-1, 1)
        c2 = self.evaluate_slack2_true(X).reshape(-1, 1)
        c3 = self.evaluate_slack3_true(X).reshape(-1, 1)
        c4 = self.evaluate_slack4_true(X).reshape(-1, 1)
        c5 = self.evaluate_slack5_true(X).reshape(-1, 1)
        return torch.concat([y, c1, c2, c3, c4, c5], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert 0 <= task_index <= 5, "Task index must be between 0 and 6"
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
        else:
            raise ValueError("Invalid task index")


class TwoLayerCNN_train(SingleObjectiveProblem):
    _bounds = [(0.0005, 0.05), (0, 1.9), (8, 64), (8, 64), (1 / 16, 1 / 2)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 5
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)
        self.number_of_epochs = 20
        self.batch_size = 128
        self.number_of_replications = 10
        self.number_of_classes = 10
        self.minimum_accuracy_level_percentage = 50
        self.dtype = torch.float32
        self.samples_per_class = {i: 2500 if i < 5 else 5000 for i in range(self.number_of_classes)}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_loader = self.prepare_train_set(transform)
        self.test_loader = self.prepare_test_set(transform)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_train_set(self, transform):
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        targets = np.array(full_trainset.targets)
        selected_indices = []
        for cls, count in self.samples_per_class.items():
            cls_indices = np.where(targets == cls)[0]
            selected = np.random.choice(cls_indices, count, replace=False)
            selected_indices.extend(selected)
        random.shuffle(selected_indices)
        imbalanced_trainset = Subset(full_trainset, selected_indices)
        return DataLoader(imbalanced_trainset, batch_size=self.batch_size, shuffle=True, num_workers=2,
                          persistent_workers=True)

    def prepare_test_set(self, transform):
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        return DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    def get_number_of_constraints(self):
        return self.number_of_classes

    def get_penalty(self):
        return 100.0

    def get_name(self):
        return "two_layer_cnn"

    def is_noisy(self):
        return True

    def is_expensive(self):
        return True

    def train_network(self, learning_rate, rho, out_channels1, out_channels2, dropout_prob):
        print(f"--- Function Arguments ---")
        print(f"Learning Rate: {learning_rate}")
        print(f"Rho: {rho}")
        print(f"Out Channels 1: {out_channels1}")
        print(f"Out Channels 2: {out_channels2}")
        print(f"Dropout Probability: {dropout_prob}")
        print(f"--------------------------")
        # Build a class imbalanced dataset
        model = TwoLayerCNN(out_channels1=out_channels1,
                            out_channels2=out_channels2,
                            dropout_prob=dropout_prob).to(self.device, self.dtype)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Training loop
        criterion = self.get_cost_function(rho)
        for epoch in range(self.number_of_epochs):
            start = time.time()
            model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            stop = time.time()
            print(f"Epoch [{epoch + 1}/{self.number_of_epochs}] - Loss: {running_loss / len(self.train_loader):.4f}")
            print("time per epoch [seconds]: ", stop - start)

        # Evaluation
        accuracy = self.evaluate_model_accuracy(model, self.test_loader)

        # adapt constraints from accuracy >= 50 to 50 - accuracy <= 0. All constraints are as less and equal in the code
        for i in range(1, self.number_of_classes + 1):
            accuracy[i] = self.minimum_accuracy_level_percentage - accuracy[i]
        return accuracy

    def get_cost_function(self, rho):
        num_samples = sum(self.samples_per_class.values())
        class_weights = torch.tensor(
            [(num_samples / (self.samples_per_class[i] * self.number_of_classes)) ** rho for i in
             range(self.number_of_classes)]).to(self.device, self.dtype)
        return torch.nn.CrossEntropyLoss(weight=class_weights)

    def evaluate_model_accuracy(self, model, dataloader):
        model.eval()
        correct_per_class = [0 for _ in range(self.number_of_classes)]
        total_per_class = [0 for _ in range(self.number_of_classes)]
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    total_per_class[label] += 1
                    if pred == label:
                        correct_per_class[label] += 1
        print("Per-Class Accuracy:")
        per_class_accuracies = []
        for i in range(self.number_of_classes):
            per_class_accuracies.append(100 * correct_per_class[i] / total_per_class[i])
            print(f"Class {i}: {per_class_accuracies[i]:.2f}%")
        overall_accuracies = [100 * sum(correct_per_class) / sum(total_per_class)]
        accuracies = torch.tensor(overall_accuracies + per_class_accuracies).to("cpu")
        print("accuracies: " + str(accuracies))
        return accuracies

    def evaluate_true(self, X: Tensor) -> Tensor:
        predictions = torch.zeros(X.shape[0], self.number_of_classes + 1)
        for idx, x in enumerate(X):
            x1, x2, x3, x4, x5 = self.transform_cube_to_hypers(x, self._bounds)
            print(x1, x2, x3, x4, x5)
            predictions[idx, :] = self.train_network(learning_rate=x1,
                                                     rho=x2,
                                                     out_channels1=int(x3.item()),
                                                     out_channels2=int(x4.item()),
                                                     dropout_prob=x5)
        return predictions[:, 0]

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        number_of_replications = self.number_of_replications if is_repeated else 1
        predictions = torch.zeros(X.shape[0], self.number_of_classes + 1)
        for idx, x in enumerate(X):
            x1, x2, x3, x4, x5 = self.transform_cube_to_hypers(x, self._bounds)
            predictions_replications = torch.zeros(number_of_replications, self.number_of_classes + 1)
            for rep in range(number_of_replications):
                predictions_replications[rep, :] = self.train_network(learning_rate=x1,
                                                                      rho=x2,
                                                                      out_channels1=int(x3.item()),
                                                                      out_channels2=int(x4.item()),
                                                                      dropout_prob=x5)
            predictions[idx, :] = torch.mean(predictions_replications, dim=0)
        return predictions

    def transform_cube_to_hypers(self, x, bounds):
        # get bounds
        learning_rate_bounds = bounds[:, 0]
        rho_bounds = bounds[:, 1]
        out_channel_1_bounds = bounds[:, 2]
        out_channel_2_bounds = bounds[:, 3]
        dropout_bounds = bounds[:, 4]

        # get values in cube [0, 1]^d to hypers
        learning_rate = learning_rate_bounds[0] * np.exp(
            x[0] * np.log(learning_rate_bounds[1] / learning_rate_bounds[0]))
        rho_actual = (rho_bounds[1] - rho_bounds[0]) * x[1] + rho_bounds[0]
        out_channel_1 = (out_channel_1_bounds[1] - out_channel_1_bounds[0]) * x[2] + out_channel_1_bounds[0]
        out_channel_2 = (out_channel_2_bounds[1] - out_channel_2_bounds[0]) * x[3] + out_channel_2_bounds[0]
        dropout = (dropout_bounds[1] - dropout_bounds[0]) * x[4] + dropout_bounds[0]
        return learning_rate, rho_actual, out_channel_1, out_channel_2, dropout

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert 0 <= task_index <= 10, "Task index must be between 0 and 10"
        if task_index == 0:
            return self.forward(X)
        output = self.evaluate_slack_true_by_index(X.reshape(-1), task_index)
        return torch.atleast_1d(output)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        pass

    def evaluate_slack_true_by_index(self, x: Tensor, index: int) -> Tensor:
        x1, x2, x3, x4, x5 = self.transform_cube_to_hypers(x, self._bounds)
        return self.train_network(learning_rate=x1,
                                  rho=x2,
                                  out_channels1=int(x3.item()),
                                  out_channels2=int(x4.item()),
                                  dropout_prob=x5)[index]
