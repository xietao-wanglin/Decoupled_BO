from typing import Optional

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import AnalyticAcquisitionFunction, MCAcquisitionObjective
from botorch.models import ModelListGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Standardize
from botorch.utils import t_batch_mode_transform
from botorch.utils.probability.utils import (
    log_ndtr as log_Phi,
)
from gpytorch.mlls import SumMarginalLogLikelihood
from torch import Tensor

from bo.device_utils import DEVICE as device, DTYPE as dtype


# ──────────────────────────────────────────────────────────────────────
# Gaussian Copula outcome transform (SCBO, Eriksson & Poloczek 2021)
# ──────────────────────────────────────────────────────────────────────

def gaussian_copula_transform(Y: Tensor) -> Tensor:
    r"""
    Rank-based Gaussian copula transform:
        z_i = Φ^{-1}(avg_rank(y_i) / (n + 1))

    Uses averaged ranks for exact ties. Operates on a 1D tensor.
    """
    if Y.ndim != 1:
        Y = Y.reshape(-1)

    orig_dtype = Y.dtype
    device = Y.device

    if not torch.is_floating_point(Y):
        Y = Y.to(torch.double)
    else:
        Y = Y.to(dtype=torch.double)

    n = Y.numel()
    if n == 0:
        return Y.to(dtype=orig_dtype)

    # Stable sort
    sorted_vals, sort_idx = torch.sort(Y, stable=True)

    # Average ranks in sorted order
    avg_ranks_sorted = torch.empty(n, dtype=Y.dtype, device=device)

    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1

        # 1-based average rank for indices start,...,end-1
        avg_rank = 0.5 * ((start + 1) + end)
        avg_ranks_sorted[start:end] = avg_rank
        start = end

    # Scatter back to original order
    ranks = torch.empty(n, dtype=Y.dtype, device=device)
    ranks[sort_idx] = avg_ranks_sorted

    u = ranks / (n + 1.0)

    # Numerically safe, though theoretically unnecessary here
    eps = torch.finfo(Y.dtype).eps
    u = u.clamp(min=eps, max=1.0 - eps)

    z = torch.special.ndtri(u)
    return z.to(dtype=orig_dtype)


def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]


def constraint_callable_wrapper(constraint_idx):
    def constraint_callable(Z):
        return Z[..., constraint_idx]

    return constraint_callable


class ConstrainedPosteriorMean(AnalyticAcquisitionFunction):
    r"""Constrained Posterior Mean (feasibility-weighted).

    Computes the analytic Posterior Mean for a Normal posterior
    distribution, weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports the case `q=1`. The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.
    """

    def __init__(self, model: Model, objective: Optional[MCAcquisitionObjective] = None, maximize: bool = True,
                 penalty_value: Optional[Tensor] = torch.tensor([0.0], dtype=torch.float64),
                 fantasised_models: Optional[Model] = None, evaluation_mask: Optional[Tensor] = None) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.fantasised_models = fantasised_models
        self.objective = objective
        self.posterior_transform = None
        self.maximize = maximize
        # Move penalty to model device
        try:
            _dev = next(model.parameters()).device
        except StopIteration:
            _dev = torch.device("cpu")
        self.penalty_value = penalty_value.to(device=_dev, dtype=torch.float64)
        if fantasised_models is not None and evaluation_mask is None:
            raise print("provide evaluation mask when providing fantasised models")
        if fantasised_models is None and evaluation_mask is not None:
            raise print("provide fantasised models when providing evaluation mask")
        self.fantasised_models = fantasised_models
        self.evaluation_mask = evaluation_mask

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """

        means, sigmas = self.evaluate_posterior(X)
        mean_obj = means[..., 0]
        mean_constraints = means[..., 1:]
        sigma_constraints = sigmas[..., 1:]
        limits = torch.zeros(means.shape[-1] - 1, device=means.device, dtype=means.dtype)
        probability_feasibility = self.compute_feasibility(mean_constraints, limits, sigma_constraints)
        constrained_posterior_mean = (mean_obj * probability_feasibility) - self.penalty_value * (
                1 - probability_feasibility)
        return constrained_posterior_mean.squeeze(dim=-1)

    def _compute_feasibility(self, X):
        means, sigmas = self.evaluate_posterior(X)
        mean_constraints = means[..., 1:]
        sigma_constraints = sigmas[..., 1:]
        limits = torch.zeros(means.shape[-1] - 1, device=means.device, dtype=means.dtype)
        return self.compute_feasibility(mean_constraints, limits, sigma_constraints)

    def evaluate_feasibility_by_index(self, X: Tensor, index):
        means, sigmas = self.evaluate_posterior(X)
        mean_constraints = means[..., index]
        sigma_constraints = sigmas[..., index]
        limits = torch.zeros(1, device=means.device, dtype=means.dtype)
        z = (limits - mean_constraints) / sigma_constraints
        return log_Phi(z).exp()

    def _evaluate_objective(self, X: Tensor):
        means, sigmas = self.evaluate_posterior(X)
        return means[..., 0]

    @staticmethod
    def compute_feasibility(mean_constraints, limits, sigma_constraints):
        # Compute log-CDF to improve numerical stability, then sum
        z = (limits - mean_constraints) / sigma_constraints
        return log_Phi(z).sum(dim=-1).exp()

    def evaluate_posterior(self, X: Tensor) -> Tensor:
        if self.evaluation_mask is not None:
            posteriors = []
            for out in range(self.model.num_outputs):
                if self.evaluation_mask[..., out]:
                    posteriors.append(self.fantasised_models.models[out].posterior(X))
                else:
                    posteriors.append(self.model.models[out].posterior(X))
            means = torch.stack([posterior.mean.squeeze(dim=-1) for posterior in posteriors], dim=-1)
            sigmas = torch.stack(
                [posterior.variance.squeeze(dim=-1).clamp_min(1e-12).sqrt() for posterior in posteriors], dim=-1)
        else:
            posterior = self.model.posterior(X=X)
            means = posterior.mean.squeeze()  # (b) x m
            sigmas = posterior.variance.squeeze().clamp_min(1e-12).sqrt()  # (b) x m
        return means, sigmas


class ConstrainedDeoupledGPModelWrapper:
    def __init__(self, num_constraints: int, is_noisy: bool):
        self.model_f = None
        self.model = None
        self.num_constraints = num_constraints
        self.num_outputs = num_constraints + 1
        # Noise on CPU — fitting always happens on CPU; model moves to GPU after optimize()
        self.train_var_noise = None if is_noisy else torch.tensor(1e-9, dtype=dtype)

    def fit(self, X, Y):
        # Fit on CPU — model is moved to GPU in optimize()
        def _yvar(y_col):
            if self.train_var_noise is None:
                return None
            return self.train_var_noise.expand_as(y_col)

        X0 = X[0].cpu()
        Y0 = Y[0].reshape(-1, 1).cpu()
        self.model_f = SingleTaskGP(train_X=X0, train_Y=Y0,
                                    train_Yvar=_yvar(Y0),
                                    outcome_transform=Standardize(m=1))

        list_of_models = [self.model_f]
        for c in range(1, self.num_constraints + 1):
            Xc = X[c].cpu()
            Yc = Y[c].reshape(-1, 1).cpu()
            list_of_models.append(SingleTaskGP(train_X=Xc, train_Y=Yc,
                                               train_Yvar=_yvar(Yc),
                                               outcome_transform=Standardize(m=1)))

        self.model = ModelListGP(*list_of_models)
        return self.model

    def optimize(self):
        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        if device.type == "cuda":
            self.model = self.model.to(device)
        return self.model

    def get_model_length_scales(self):
        length_scales = []
        for i in range(self.num_constraints + 1):
            length_scales.append(self.model.models[i].covar_module.base_kernel.lengthscale.detach())
        return length_scales

    def getNumberOfOutputs(self):
        return self.num_outputs

    def to_device(self, target_device):
        """Move the fitted model to the specified device."""
        if self.model is not None:
            self.model = self.model.to(target_device)
            if self.train_var_noise is not None:
                self.train_var_noise = self.train_var_noise.to(target_device)
        return self
