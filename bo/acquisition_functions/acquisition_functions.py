from enum import Enum, auto
from typing import Optional, Union, Any

import torch
from botorch import gen_candidates_torch
from botorch.acquisition import ExpectedImprovement, \
    qExpectedImprovement, MCAcquisitionObjective, qKnowledgeGradient, MCAcquisitionFunction, \
    DecoupledAcquisitionFunction
from botorch.acquisition.analytic import _ei_helper, PosteriorMean
from botorch.acquisition.knowledge_gradient import _split_fantasy_points
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.optim import optimize_acqf, initialize_q_batch
from botorch.sampling import MCSampler, SobolQMCNormalSampler, ListSampler
from botorch.utils import draw_sobol_samples
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform
from torch import Tensor

from bo.model.Model import ConstrainedPosteriorMean
from bo.samplers.samplers import quantileSampler


class AcquisitionFunctionType(Enum):
    MC_CONSTRAINED_KNOWLEDGE_GRADIENT = auto()
    ONESHOT_CONSTRAINED_KNOWLEDGE_GRADIENT = auto()
    DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT = auto()
    BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT = auto()
    BOTORCH_EXPECTED_IMPROVEMENT = auto()
    BOTORCH_MC_EXPECTED_IMPROVEMENT = auto()
    MATHSYS_EXPECTED_IMPROVEMENT = auto()
    MATHSYS_MC_EXPECTED_IMPROVEMENT = auto()


def compute_best_posterior_mean(model, bounds, objective):
    argmax_mean, max_mean = optimize_acqf(
        acq_function=PosteriorMean(model, posterior_transform=objective),
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=2048,
    )
    return argmax_mean, max_mean


def acquisition_function_factory(type, model, objective, best_value, idx, number_of_outputs, penalty_value, iteration,
                                 initial_condition_internal_optimizer):
    if type is AcquisitionFunctionType.BOTORCH_EXPECTED_IMPROVEMENT:
        return ExpectedImprovement(model=model, best_f=best_value)
    elif type is AcquisitionFunctionType.BOTORCH_MC_EXPECTED_IMPROVEMENT:
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1000]))
        return qExpectedImprovement(model=model, best_f=best_value, sampler=qmc_sampler)
    elif type is AcquisitionFunctionType.MATHSYS_EXPECTED_IMPROVEMENT:
        return MathsysExpectedImprovement(model=model, best_f=best_value)
    elif type is AcquisitionFunctionType.MATHSYS_MC_EXPECTED_IMPROVEMENT:
        return MathsysMCExpectedImprovement(model=model, best_f=best_value)
    elif type is AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT:
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([100]))
        return qExpectedImprovement(model=model, best_f=best_value, sampler=qmc_sampler, objective=objective)
    elif type is AcquisitionFunctionType.ONESHOT_CONSTRAINED_KNOWLEDGE_GRADIENT:
        return OneShotConstrainedKnowledgeGradient(model, num_fantasies=64, current_value=best_value,
                                                   objective=objective)
    elif type is AcquisitionFunctionType.MC_CONSTRAINED_KNOWLEDGE_GRADIENT:
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([5]))
        sampler_list = ListSampler(*[sampler] * number_of_outputs)
        x_eval_mask = torch.ones(1, number_of_outputs, dtype=torch.bool)
        return DecoupledConstrainedKnowledgeGradient(model, sampler=sampler_list, num_fantasies=5,
                                                     objective=objective,
                                                     number_of_raw_points=72,  # can make smaller if speed
                                                     number_of_restarts=15,  # this too, not too small though
                                                     seed=iteration,
                                                     # play with this if it gets too slow...get it down a little...
                                                     X_evaluation_mask=x_eval_mask,
                                                     penalty_value=penalty_value,
                                                     x_best_location=initial_condition_internal_optimizer)
    elif type is AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT:
        sampler = quantileSampler(sample_shape=torch.Size([5]))
        sampler_list = ListSampler(*[sampler] * number_of_outputs)
        x_eval_mask = torch.zeros(1, number_of_outputs,
                                  dtype=torch.bool)  # 2 outputs, 1 == True # Change torch.zeros to ones for coupled cKG
        x_eval_mask[0, idx] = 1
        return DecoupledConstrainedKnowledgeGradient(model, sampler=sampler_list, num_fantasies=5,
                                                     objective=objective,
                                                     number_of_raw_points=72,  # can make smaller if speed
                                                     number_of_restarts=15,  # this too, not too small though
                                                     seed=iteration,
                                                     # play with this if it gets too slow...get it down a little...
                                                     X_evaluation_mask=x_eval_mask,
                                                     penalty_value=penalty_value,
                                                     x_best_location=initial_condition_internal_optimizer)


class DecoupledConstrainedKnowledgeGradient(DecoupledAcquisitionFunction, MCAcquisitionFunction):

    def __init__(self, model: Model,
                 sampler: Optional[MCSampler] = None,
                 num_fantasies: Optional[int] = 5,
                 current_value: Optional[Tensor] = None,
                 objective: Optional[MCAcquisitionObjective] = None,
                 posterior_transform: Optional[PosteriorTransform] = None,
                 X_pending: Optional[Tensor] = None,
                 number_of_raw_points: Optional[int] = 64,
                 number_of_restarts: Optional[int] = 20,
                 X_evaluation_mask: Optional[Tensor] = None,
                 seed: Optional[int] = 0,
                 penalty_value: Optional[Tensor] = None,
                 x_best_location: Optional[Tensor] = None) -> None:
        super().__init__(model=model, sampler=sampler, objective=objective,
                         posterior_transform=posterior_transform, X_pending=X_pending,
                         X_evaluation_mask=X_evaluation_mask)
        self.current_value = current_value
        self.num_fantasies = num_fantasies
        self.penalty_value = penalty_value
        self.number_of_restarts = number_of_restarts
        self.number_of_raw_points = number_of_raw_points
        self.seed = seed
        self.x_best_location = x_best_location.reshape(-1)
        assert len(self.x_best_location.shape) == 1, "include a single best location"

    def forward(self, X: Tensor) -> Tensor:
        fantasy_model = self.model.fantasize(X=X, sampler=self.sampler,
                                             evaluation_mask=self.construct_evaluation_mask(X))
        bounds = torch.tensor([[0.0] * X.shape[-1], [1.0] * X.shape[-1]], dtype=torch.double)
        constrained_posterior_mean_model = ConstrainedPosteriorMean(fantasy_model, penalty_value=self.penalty_value)
        batch_shape = constrained_posterior_mean_model.model.batch_shape
        raw_points = draw_sobol_samples(bounds=bounds,
                                        n=self.number_of_raw_points,
                                        q=1,
                                        batch_shape=batch_shape,
                                        seed=self.seed)

        best_location_adapted_dimensions = self.adapt_x_location_dim(batch_shape)
        restart_points = initialize_q_batch(X=raw_points,
                                            Y=constrained_posterior_mean_model(raw_points),
                                            n=self.number_of_restarts, eta=2.0)
        restart_points = torch.cat([restart_points, best_location_adapted_dimensions], dim=0)
        with torch.enable_grad():
            bestx, _ = gen_candidates_torch(initial_conditions=restart_points,
                                            acquisition_function=constrained_posterior_mean_model,
                                            lower_bounds=bounds[0],
                                            upper_bounds=bounds[1],
                                            options={"maxiter": 100})
            bestvals = constrained_posterior_mean_model(bestx)

        bestval_sample = bestvals.max(dim=0)[0]
        kgvals = bestval_sample.mean(dim=0)
        return kgvals

    def evaluate_kg_value(self, X: Tensor, number_of_restarts, number_of_raw_points):
        self.number_of_restarts = number_of_restarts
        self.number_of_raw_points = number_of_raw_points
        best_predictive_mean = self.forward(X)
        fantasy_model = self.model.fantasize(X=X, sampler=self.sampler,
                                             evaluation_mask=self.construct_evaluation_mask(X))
        constrained_posterior_mean_model = ConstrainedPosteriorMean(fantasy_model, penalty_value=self.penalty_value)
        batch_shape = constrained_posterior_mean_model.model.batch_shape
        best_location_adapted_dimensions = self.adapt_x_location_dim(batch_shape)
        fval_best_location = constrained_posterior_mean_model(best_location_adapted_dimensions).max(dim=0)[0]
        kgvals = best_predictive_mean - fval_best_location.mean(dim=0)
        return kgvals

    def adapt_x_location_dim(self, batch_shape):
        best_location_adapted_dimensions = torch.ones(1, *batch_shape, 1, len(self.x_best_location))
        best_location_adapted_dimensions[..., :] = self.x_best_location
        return best_location_adapted_dimensions


class MCConstrainedKnowledgeGradient(MCAcquisitionFunction):
    def __init__(self, model: Model, num_fantasies: Optional[int] = 64, sampler: Optional[MCSampler] = None,
                 objective: Optional[MCAcquisitionObjective] = None,
                 posterior_transform: Optional[PosteriorTransform] = None,
                 current_value: Optional[Tensor] = None) -> None:
        super().__init__(model, sampler=sampler, objective=objective, posterior_transform=posterior_transform)
        if sampler is None:
            self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_fantasies]))
        self.current_value = current_value
        self.num_fantasies = num_fantasies

    def forward(self, X: Tensor) -> Tensor:
        # construct the fantasy model of shape `num_fantasies x b`
        kgvals = torch.zeros(X.shape[0], dtype=torch.double)
        for xi, xnew in enumerate(X):
            fantasy_model = self.model.fantasize(
                X=xnew,
                sampler=self.sampler,
            )
            bounds = torch.tensor([[0.0] * X.shape[-1], [1.0] * X.shape[-1]])
            batch_shape = ConstrainedPosteriorMean(fantasy_model).model.batch_shape
            with torch.enable_grad():
                num_init_points = 5
                initial_conditions = draw_sobol_samples(bounds=bounds, n=num_init_points, q=1, batch_shape=batch_shape)
                best_x, _ = gen_candidates_torch(
                    initial_conditions=initial_conditions.contiguous(),
                    acquisition_function=ConstrainedPosteriorMean(fantasy_model),
                    lower_bounds=bounds[0],
                    upper_bounds=bounds[1],
                    options={"maxiter": 60}
                )
            kgvals[xi] = self.compute_discrete_kg(model=fantasy_model, x_new=xnew,
                                                  optimal_discretisation=best_x.reshape(
                                                      num_init_points * self.num_fantasies,
                                                      X.shape[-1]))
        if self.current_value is not None:
            kgvals = kgvals - self.current_value
        return kgvals

    def compute_discrete_kg(self, model: Model, x_new: Tensor, optimal_discretisation: Tensor) -> Tensor:
        """

        Args:
        xnew: A `1 x 1 x d` Tensor with `1` acquisition function evaluations of
            `d` dimensions.
            optimal_discretisation: num_fantasies x d Tensor. Optimal X values for each z in zvalues.

        """
        # Augment the discretisation with the designs.
        concatenated_xnew_discretisation = torch.cat(
            [x_new, optimal_discretisation], dim=0
        ).squeeze()  # (m + num_X_disc, d)

        # Compute posterior mean, variance, and covariance.
        posterior = model.posterior(concatenated_xnew_discretisation)
        means = posterior.mean
        sigmas = posterior.variance.sqrt().clamp_min(1e-9)

        mean_obj = means[..., 0]
        std_obj = sigmas[..., 0]
        mean_constraints = means[..., 1:]
        sigma_constraints = sigmas[..., 1:]
        limits = torch.tensor([0] * (means.shape[-1] - 1))
        z = (limits - mean_constraints) / sigma_constraints
        probability_feasibility = torch.distributions.Normal(0, 1).cdf(z).prod(dim=-1)
        constrained_posterior_mean = mean_obj * probability_feasibility
        constrained_posterior_std = std_obj * probability_feasibility
        # initialise empty kgvals torch.tensor
        return self.kgcb(a=constrained_posterior_mean, b=constrained_posterior_std)

    def kgcb(self, a: Tensor, b: Tensor) -> Tensor:
        r"""
        Calculates the linear epigraph, i.e. the boundary of the set of points
        in 2D lying above a collection of straight lines y=a+bx.
        Parameters
        ----------
        a
            Vector of intercepts describing a set of straight lines
        b
            Vector of slopes describing a set of straight lines
        Returns
        -------
        KGCB
            average height of the epigraph
        """

        a = a.squeeze()
        b = b.squeeze()
        assert len(a) > 0, "must provide slopes"
        assert len(a) == len(b), f"#intercepts != #slopes, {len(a)}, {len(b)}"

        maxa = torch.max(a)

        if torch.all(torch.abs(b) < 0.000000001):
            return torch.Tensor([0])  # , np.zeros(a.shape), np.zeros(b.shape)

        # Order by ascending b and descending a. There should be an easier way to do this
        # but it seems that pytorch sorts everything as a 1D Tensor

        ab_tensor = torch.vstack([-a, b]).T
        ab_tensor_sort_a = ab_tensor[ab_tensor[:, 0].sort()[1]]
        ab_tensor_sort_b = ab_tensor_sort_a[ab_tensor_sort_a[:, 1].sort()[1]]
        a = -ab_tensor_sort_b[:, 0]
        b = ab_tensor_sort_b[:, 1]

        # exclude duplicated b (or super duper similar b)
        threshold = (b[-1] - b[0]) * 0.00001
        diff_b = b[1:] - b[:-1]
        keep = diff_b > threshold
        keep = torch.cat([torch.Tensor([True]), keep])
        keep[torch.argmax(a)] = True
        keep = keep.bool()  # making sure 0 1's are transformed to booleans

        a = a[keep]
        b = b[keep]

        # initialize
        idz = [0]
        i_last = 0
        x = [-torch.inf]

        n_lines = len(a)
        while i_last < n_lines - 1:
            i_mask = torch.arange(i_last + 1, n_lines)
            x_mask = -(a[i_last] - a[i_mask]) / (b[i_last] - b[i_mask])

            best_pos = torch.argmin(x_mask)
            idz.append(i_mask[best_pos])
            x.append(x_mask[best_pos])

            i_last = idz[-1]

        x.append(torch.inf)

        x = torch.Tensor(x)
        idz = torch.LongTensor(idz)
        # found the epigraph, now compute the expectation
        a = a[idz]
        b = b[idz]

        normal = torch.distributions.Normal(torch.zeros_like(x), torch.ones_like(x))

        pdf = torch.exp(normal.log_prob(x))
        cdf = normal.cdf(x)

        kg = torch.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:]))
        kg -= maxa
        return kg


class OneShotConstrainedKnowledgeGradient(qKnowledgeGradient):

    def __init__(self, model: Model, num_fantasies: Optional[int] = 64, sampler: Optional[MCSampler] = None,
                 objective: Optional[MCAcquisitionObjective] = None,
                 posterior_transform: Optional[PosteriorTransform] = None, inner_sampler: Optional[MCSampler] = None,
                 X_pending: Optional[Tensor] = None, current_value: Optional[Tensor] = None) -> None:
        super().__init__(model, num_fantasies, sampler, objective, posterior_transform, inner_sampler, X_pending,
                         current_value)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=self.num_fantasies)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual,
            sampler=self.sampler,
        )

        posterior = fantasy_model.posterior(X_fantasies)
        posterior_objective_mean = posterior.mean[..., 0]
        posterior_constraint_mean = posterior.mean[..., 1:]
        posterior_constraint_std = posterior.variance[..., 1:].sqrt().clamp_min(1e-9)
        limits = torch.tensor([0] * (posterior.mean.shape[-1] - 1))
        z = (limits - posterior_constraint_mean) / posterior_constraint_std

        probability_feasibility = torch.distributions.Normal(0, 1).cdf(z).prod(dim=-1)
        values = posterior_objective_mean * probability_feasibility
        if self.current_value is not None:
            values = values - self.current_value

        # return average over the fantasy samples
        return values.mean(dim=0).reshape(-1)

    def evaluate(self, X: Tensor, bounds: Tensor, **kwargs: Any) -> Tensor:
        return super().evaluate(X, bounds, **kwargs)


class MathsysExpectedImprovement(ExpectedImprovement):

    def __init__(self, model: Model, best_f: Union[float, Tensor],
                 posterior_transform: Optional[PosteriorTransform] = None, maximize: bool = True, **kwargs):
        super().__init__(model, best_f, posterior_transform, maximize, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        posterior = self.model.posterior(X)
        mu = posterior.mean
        sigma = torch.sqrt(posterior.variance)
        Z = (mu - self.best_f) / sigma
        ei_value = sigma * _ei_helper(Z)
        return ei_value.reshape(-1)


class MathsysMCExpectedImprovement(qExpectedImprovement):
    def __init__(self, model: Model, best_f: Union[float, Tensor], sampler: Optional[MCSampler] = None,
                 objective: Optional[MCAcquisitionObjective] = None,
                 posterior_transform: Optional[PosteriorTransform] = None, X_pending: Optional[Tensor] = None,
                 **kwargs: Any) -> None:
        super().__init__(model, best_f, sampler, objective, posterior_transform, X_pending, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        pass
