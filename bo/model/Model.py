from typing import Optional

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import AnalyticAcquisitionFunction, MCAcquisitionObjective
from botorch.models import FixedNoiseGP, ModelListGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Standardize
from botorch.utils import t_batch_mode_transform
from botorch.utils.probability.utils import (
    log_ndtr as log_Phi,
)
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood
from torch import Tensor

# constants
device = torch.device("cpu")
dtype = torch.float64


class GPModelWrapper():
    def __init__(self):
        self.train_yvar = torch.tensor(1e-6, device=device, dtype=dtype)

    def fit(self, X, y):
        self.model = FixedNoiseGP(train_X=X,
                                  train_Y=y,
                                  train_Yvar=self.train_yvar.expand_as(y),
                                  outcome_transform=Standardize(m=1))
        return self.model

    def optimize(self):
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        return self.model


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
        self.penalty_value = penalty_value.to(torch.float64)
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
        limits = torch.tensor([0] * (means.shape[-1] - 1))
        probability_feasibility = self.compute_feasibility(mean_constraints, limits, sigma_constraints)
        constrained_posterior_mean = (mean_obj * probability_feasibility) - self.penalty_value * (
                1 - probability_feasibility)
        return constrained_posterior_mean.squeeze(dim=-1)


    def _evaluate_feasibility_by_index(self, X: Tensor, index):
        means, sigmas = self.evaluate_posterior(X)
        mean_constraints = means[..., index]
        sigma_constraints = sigmas[..., index]
        limits = torch.tensor([0])
        z = (limits - mean_constraints) / sigma_constraints
        return log_Phi(z).exp()

    def compute_feasibility(self, mean_constraints, limits, sigma_constraints):
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
            sigmas = torch.stack([posterior.variance.squeeze(dim=-1).clamp_min(1e-12).sqrt() for posterior in posteriors], dim=-1)
        else:
            posterior = self.model.posterior(X=X)
            means = posterior.mean.squeeze()  # (b) x m
            sigmas = posterior.variance.squeeze().clamp_min(1e-12).sqrt()  # (b) x m
        return means, sigmas


class DecoupledConstraintPosteriorMean(AnalyticAcquisitionFunction):
    r"""Constrained Posterior Mean (feasibility-weighted).

    Computes the analytic Posterior Mean for a Normal posterior
    distribution, weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports the case `q=1`. The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.
    """

    def __init__(
            self,
            model: Model,
            objective: Optional[MCAcquisitionObjective] = None,
            index: int = 0,
            maximize: bool = True,
            penalty_value: Optional[Tensor] = torch.tensor([0.0], dtype=torch.float64),
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.objective = objective
        self.posterior_transform = None
        self.maximize = maximize
        self.penalty_value = penalty_value.to(torch.float64)
        self.index = index

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
        constrained_posterior_mean = mean_obj
        return constrained_posterior_mean.squeeze(dim=-1) - self.penalty_value * torch.sum(
            torch.max(mean_constraints, torch.tensor([0])),
            dim=-1).squeeze()

    def evaluate_posterior(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X=X)
        means = posterior.mean.squeeze()  # (b) x m
        sigmas = posterior.variance.squeeze().clamp_min(1e-12).sqrt()  # (b) x m
        return means, sigmas


class CustomGaussianLikelihood(GaussianLikelihood):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_constraint("noise_constraint", GreaterThan(1e-4))


class ConstrainedGPModelWrapper():
    def __init__(self, num_constraints):
        self.model_f = None
        self.model = None
        self.num_constraints = num_constraints
        self.train_var_noise = torch.tensor(1e-4, device=device, dtype=dtype)

    def fit(self, X, Y):
        assert Y.shape[1] == self.num_constraints + 1, "missmatch constraint number"
        assert Y.shape[0] == X.shape[0], "missmatch number of evaluations"

        self.model_f = SingleTaskGP(train_X=X,
                                    train_Y=Y[:, 0].reshape(-1, 1),
                                    train_Yvar=self.train_var_noise.expand_as(Y[:, 0].reshape(-1, 1)),
                                    outcome_transform=Standardize(m=1))

        list_of_models = [self.model_f]
        for c in range(1, self.num_constraints + 1):
            list_of_models.append(SingleTaskGP(train_X=X,
                                               train_Y=Y[:, c].reshape(-1, 1),
                                               train_Yvar=self.train_var_noise.expand_as(Y[:, 0].reshape(-1, 1))))

            self.model = ModelListGP(*list_of_models)
            return self.model

    def optimize(self):
        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        return self.model


class ConstrainedDeoupledGPModelWrapper():
    def __init__(self, num_constraints):
        self.model_f = None
        self.model = None
        self.num_constraints = num_constraints
        self.train_var_noise = torch.tensor(1e-9, device=device, dtype=dtype)
        self.num_outputs = num_constraints + 1

    def getNumberOfOutputs(self):
        return self.num_outputs

    def fit(self, X, Y):
        self.model_f = SingleTaskGP(train_X=X[0],
                                    train_Y=Y[0].reshape(-1, 1),
                                    train_Yvar=self.train_var_noise.expand_as(Y[0].reshape(-1, 1)),
                                    outcome_transform=Standardize(m=1))

        list_of_models = [self.model_f]
        for c in range(1, self.num_constraints + 1):
            list_of_models.append(SingleTaskGP(train_X=X[c],
                                               train_Y=Y[c].reshape(-1, 1),
                                               train_Yvar=self.train_var_noise.expand_as(Y[c].reshape(-1, 1))))

        self.model = ModelListGP(*list_of_models)
        return self.model

    def optimize(self):
        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        return self.model

    def get_model_length_scales(self):
        length_scales = []
        for i in range(self.num_constraints + 1):
            length_scales.append(self.model.models[i].covar_module.base_kernel.lengthscale.detach())
        return length_scales