import os
import pickle
from typing import Optional

import numpy as np
from sympy.printing.pytorch import torch
from torch import Tensor

from bo.synthetic_test_functions.synthetic_test_functions import SingleObjectiveProblem


def standard_length_scale(bounds):
    return (bounds[1] - bounds[0]) / 2.

from bo.device_utils import DEVICE as device, DTYPE as dtype

class const_cnn_cifar10(SingleObjectiveProblem):
    '''
    CNN CIFAR10 real data Function: d = 5, M = 3
    '''
    _bounds = [(-3., 0.), (5., 8.), (3., 6.), (3., 6.), (0., 1.9)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 5
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=dtype).transpose(-1, -2)
        self.noise_var = 0
        self.C = 10
        self.M = self.C + 1
        self.standard_length_scale = standard_length_scale(self._bounds)
        self.maximum = 0

        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, "AutoML_data_CBO", "cnn_CIFAR10_data", "cnn_CIFAR10_data.pickle")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.g_thresholds = 0.5 * np.ones(self.C)

        feasible_index = np.where(np.all(data[:, 5:15] >= self.g_thresholds, axis=1) == True)[0]
        print('number of feasible points:', np.size(feasible_index))

        self.X = data[:, 0:5]
        self.Y = [data[:, 15]]
        self.Y.extend([data[:, 5 + i] for i in range(self.C)])

        # logit transformation
        self.Y = [np.where(Y <= 0, 1e-5, Y) for Y in self.Y]
        self.Y = [np.where(Y >= 1, 1 - 1e-5, Y) for Y in self.Y]
        self.Y = [self.output_data_transform(Y) for Y in self.Y]
        self.g_thresholds = self.output_data_transform(self.g_thresholds)

        self.CONSTRAINED_MAX = np.max(self.Y[0][feasible_index])
        self.GLOBAL_MAX = np.max(self.Y[0])

        print(self.CONSTRAINED_MAX, np.max(self.Y[0]), self.GLOBAL_MAX)
        print("transformed")
        print(self.output_data_transform(self.CONSTRAINED_MAX), self.output_data_transform(np.max(self.Y[0])),
              self.output_data_transform(self.GLOBAL_MAX))


    def output_data_transform(self, value):
        return np.log(value / (1 - value))

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        func_values_list = []
        for m in range(self.M):
            func_values_list.append(self.evaluate_task(X, task_idx=m))
        return torch.vstack(func_values_list).T

    def evaluate_black_box(self, X: Tensor, is_repeated: Optional[bool] = False) -> Tensor:
        return self.evaluate_slack_true(X)

    def evaluate_true(self, X: Tensor) -> Tensor:
        pass

    def is_expensive(self):
        return False

    def is_noisy(self):
        return False

    def get_name(self):
        return "two_layer_cnn_discretised"

    def get_number_of_constraints(self):
        return self.C

    def get_penalty(self):
        return 3.0

    def evaluate_task(self, input, task_idx):
        input = torch.atleast_2d(input)
        hypers = self.transform_inputs(input)

        match_index = list()
        for i in range(np.shape(hypers)[0]):
            tmp_match_index = torch.where(torch.all(torch.abs(torch.tensor(self.X) - hypers[i]) < 1e-6, axis=1))[0]
            match_index.append(tmp_match_index)
        match_index = torch.tensor([match_index]).ravel()
        if task_idx > 0:
            return self.g_thresholds[task_idx - 1] - torch.tensor([self.Y[task_idx][match_index]], dtype=dtype,
                                                                  device=device).reshape(-1)
        return torch.tensor([self.Y[task_idx][match_index]], dtype=dtype, device=device).reshape(-1)

    def transform_inputs(self, input):
        hypers_transformed = self.transform_cube_to_hypers(input)
        return self.discretise_inputs(hypers_transformed)

    def _transform_hypers_to_cube(self, y):
        y = torch.atleast_2d(y)

        bounds = torch.tensor([
            [1 / 1000, 1.0],  # learning rate
            [32, 256],  # batch_size
            [8, 64],  # out_channel_1
            [8, 64],  # out_channel_2
            [0.0, 1.9]  # rho
        ], dtype=torch.float)

        lb_index = 0
        ub_index = 1

        # learning rate (log10)
        lr_lin = 10 ** y[:, 0]
        learning_rate = (lr_lin - bounds[0, lb_index]) / (bounds[0, ub_index] - bounds[0, lb_index])

        # batch size (log2)
        bs_lin = 2 ** y[:, 1]
        batch_size = (bs_lin - bounds[1, lb_index]) / (bounds[1, ub_index] - bounds[1, lb_index])

        # out_channel_1 (log2)
        oc1_lin = 2 ** y[:, 2]
        out_channel_1 = (oc1_lin - bounds[2, lb_index]) / (bounds[2, ub_index] - bounds[2, lb_index])

        # out_channel_2 (log2)
        oc2_lin = 2 ** y[:, 3]
        out_channel_2 = (oc2_lin - bounds[3, lb_index]) / (bounds[3, ub_index] - bounds[3, lb_index])

        # rho (linear)
        rho = (y[:, 4] - bounds[4, lb_index]) / (bounds[4, ub_index] - bounds[4, lb_index])

        return torch.stack(
            [learning_rate, batch_size, out_channel_1, out_channel_2, rho],
            dim=0
        ).T

    def transform_cube_to_hypers(self, x):
        # get bounds
        x = torch.atleast_2d(x)
        bounds = torch.tensor([
            [1 / 1000, 1.0],  # learning rate
            [32, 256],  # batch_size
            [8, 64],  # out_channel_1
            [8, 64],  # out_channel_2
            [0.0, 1.9]  # rho
        ], dtype=torch.float)

        ub_index = 1
        lb_index = 0
        learning_rate = torch.log10(x[:, lb_index] * (bounds[0, ub_index] - bounds[0, lb_index]) + bounds[0, lb_index])
        batch_size = torch.log2(x[:, ub_index] * (bounds[1, ub_index] - bounds[1, lb_index]) + bounds[1, lb_index])
        out_channel_1 = torch.log2(x[:, 2] * (bounds[2, ub_index] - bounds[2, lb_index]) + bounds[2, lb_index])
        out_channel_2 = torch.log2(x[:, 3] * (bounds[3, ub_index] - bounds[3, lb_index]) + bounds[3, lb_index])
        rho = x[:, 4] * (bounds[4, ub_index] - bounds[4, lb_index]) + bounds[4, lb_index]
        return torch.stack([learning_rate, batch_size, out_channel_1, out_channel_2, rho], dim=0).T

    def discretise_inputs(self, x):
        learning_rates = torch.tensor([-3, -2, -1, 0], dtype=dtype, device=x.device)
        batch_size = torch.tensor([5, 6., 7., 8.], dtype=dtype, device=x.device)
        channels_1 = torch.tensor([3, 4, 5, 6], dtype=dtype, device=x.device)
        channels_2 = torch.tensor([3, 4, 5, 6], dtype=dtype, device=x.device)
        rhos = torch.round(torch.arange(0.0, 2.0, 0.1, dtype=dtype, device=x.device) * 10) / 10

        # Discretize each column
        x[:, 0] = learning_rates[torch.abs(x[:, 0].unsqueeze(1) - learning_rates).argmin(dim=1)]
        x[:, 1] = batch_size[torch.abs(x[:, 1].unsqueeze(1) - batch_size).argmin(dim=1)]
        x[:, 2] = channels_1[torch.abs(x[:, 2].unsqueeze(1) - channels_1).argmin(dim=1)]
        x[:, 3] = channels_2[torch.abs(x[:, 3].unsqueeze(1) - channels_2).argmin(dim=1)]
        x[:, 4] = rhos[torch.abs(x[:, 4].unsqueeze(1) - rhos).argmin(dim=1)]
        return x