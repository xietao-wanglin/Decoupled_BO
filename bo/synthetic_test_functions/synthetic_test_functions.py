import math

import torch
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.utils.transforms import unnormalize
from torch import Tensor

import os
import subprocess
import sys
import tempfile
from pathlib import Path
import platform
from platform import machine

class MOPTA08(ConstrainedBaseTestProblem):
    _bounds = [(0.0, 1.0)]*124

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



class ConstrainedBraninNew(ConstrainedBaseTestProblem):
    _bounds = [(-5.0, 10.0), (0.0, 15.0)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return -(X_tf[..., 0] - 10) ** 2 - (X_tf[..., 1] - 15) ** 2

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        t1 = ((
                      X_tf[..., 1]
                      - 5.1 / (4 * math.pi ** 2) * (X_tf[..., 0] ** 2)
                      + (5 / math.pi) * X_tf[..., 0]) - 6) ** 2
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X_tf[..., 0])
        return t1 + t2 + 5

    def evaluate_black_box(self, X: Tensor) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
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

        t1 = 2.0 + 0.01 * ((X_2 - X_1.pow(2)).pow(2))
        t2 = (1 - X_1).pow(2)
        t3 = 2 * ((2 - X_2).pow(2))
        t4 = 7 * torch.sin(0.5 * X_1) * torch.sin(0.7 * X_1 * X_2)
        return t1 + t2 + t3 + t4

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return -torch.sin(X_tf[..., 0] - X_tf[..., 1] - math.pi / 8)

    def evaluate_black_box(self, X: Tensor) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
        c1 = self.evaluate_slack_true(X).reshape(-1, 1)
        print(y.shape, c1.shape)
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

class MysteryFunctionRedundant(ConstrainedBaseTestProblem):
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
        return (t1 + t2 + t3 + t4)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        pass
    
    def evaluate_slack1_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return -torch.sin(X_tf[..., 0] - X_tf[..., 1] - math.pi/8)
    
    def evaluate_slack2_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[...,0]*0.0 + X_tf[...,0]*0.0 - 100

    def evaluate_black_box(self, X: Tensor) -> Tensor:
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

class MysteryFunctionSuperRedundant(ConstrainedBaseTestProblem):
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
        return (t1 + t2 + t3 + t4)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        pass
    
    def evaluate_slack1_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return -torch.sin(X_tf[..., 0] - X_tf[..., 1] - math.pi/8)
    
    def evaluate_slack2_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[...,0]*0.0 + X_tf[...,0]*0.0 - 100
    
    def evaluate_slack3_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[...,0]*0.0 + X_tf[...,0]*0.0 - 100
    
    def evaluate_slack4_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[...,0]*0.0 + X_tf[...,0]*0.0 - 100
    
    def evaluate_slack5_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[...,0]*0.0 + X_tf[...,0]*0.0 - 100
    
    def evaluate_slack6_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[...,0]*0.0 + X_tf[...,0]*0.0 - 100
    
    def evaluate_slack7_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[...,0]*0.0 + X_tf[...,0]*0.0 - 100
    
    def evaluate_slack8_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[...,0]*0.0 + X_tf[...,0]*0.0 - 100
    
    def evaluate_slack9_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return X_tf[...,0]*0.0 + X_tf[...,0]*0.0 - 100

    def evaluate_black_box(self, X: Tensor) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
        c1 = self.evaluate_slack1_true(X).reshape(-1, 1)
        c2 = self.evaluate_slack2_true(X).reshape(-1, 1)
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

class ConstrainedFunc3(ConstrainedBaseTestProblem):
    _bounds = [(0.0, 1.0), (0.0, 1.0)]

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

    def evaluate_black_box(self, X: Tensor) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
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

class SimpleTest(ConstrainedBaseTestProblem):
    _bounds = [(0.0, 1.0), (0.0, 1.0)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        X_1 = X_tf[..., 0]
        X_2 = X_tf[..., 1]

        t1 = (X_2-0.5).pow(2) + (X_1-0.5).pow(2) - 1
        return t1

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        X_1 = X_tf[..., 0]
        X_2 = X_tf[..., 1]

        t1 = +(X_2-0.5).pow(2) + (X_1-0.5).pow(2) - 0.001
        return t1

    def evaluate_black_box(self, X: Tensor) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
        c1 = self.evaluate_slack_true(X).reshape(-1, 1)
        print(y.shape, c1.shape)
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
