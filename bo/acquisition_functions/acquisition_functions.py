from enum import Enum, auto
from typing import Optional, Union, Any

import torch
from botorch import gen_candidates_torch, gen_candidates_scipy
from botorch.acquisition import ExpectedImprovement, \
    qExpectedImprovement, MCAcquisitionObjective, qKnowledgeGradient, MCAcquisitionFunction, \
    DecoupledAcquisitionFunction
from botorch.acquisition.analytic import _ei_helper, PosteriorMean
from botorch.acquisition.knowledge_gradient import _split_fantasy_points
from botorch.acquisition.objective import PosteriorTransform
from botorch.models import ModelListGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf, initialize_q_batch
from botorch.sampling import MCSampler, SobolQMCNormalSampler, ListSampler
from botorch.utils import draw_sobol_samples
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform
from torch import Tensor

from bo.model.Model import ConstrainedPosteriorMean
from bo.samplers.samplers import quantileSampler
from debug_utils.utils import record_io


class AcquisitionFunctionType(Enum):
    COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT = auto()
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
        raw_samples=1024,
    )
    return argmax_mean, max_mean


def filter_a_b(a, b, threshold=1e-9):
    sorted_pairs = torch.stack(sort_a_b(a.clone(), b.clone()), dim=1)
    filtered_pairs = [sorted_pairs[0]]
    for i in range(1, len(sorted_pairs)):
        if torch.abs(sorted_pairs[i][1] - sorted_pairs[i - 1][1]) > threshold:
            filtered_pairs.append(sorted_pairs[i])
    filtered_pairs = torch.stack(filtered_pairs)
    final_a, final_b = filtered_pairs[:, 0], filtered_pairs[:, 1]
    return final_a, final_b


def sort_a_b(a, b):
    _, a_sort_indices = torch.sort(-a)
    sorted_indices = torch.argsort(b[a_sort_indices])
    final_sorted_indices = a_sort_indices[sorted_indices]
    sorted_a = a[final_sorted_indices]
    sorted_b = b[final_sorted_indices]
    return sorted_a, sorted_b


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
    elif type is AcquisitionFunctionType.COUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT:
        number_of_fantasies = 7
        samplers = []
        samplers.append(quantileSampler(sample_shape=torch.Size([number_of_fantasies])))
        for _ in range(number_of_outputs - 1):
            samplers.append(SobolQMCNormalSampler(sample_shape=torch.Size([number_of_fantasies])))
        sampler_list = ListSampler(*samplers)
        x_eval_mask = torch.ones(1, number_of_outputs, dtype=torch.bool)
        torch.manual_seed(iteration)
        return DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                          num_fantasies=number_of_fantasies,
                                                          objective=objective, number_of_raw_points=500,
                                                          number_of_restarts=15, X_evaluation_mask=x_eval_mask,
                                                          seed=iteration, penalty_value=penalty_value,
                                                          x_best_location=initial_condition_internal_optimizer,
                                                          evaluate_all_sources=True)

    elif type is AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT:
        number_of_fantasies = 7
        sampler = quantileSampler(sample_shape=torch.Size([number_of_fantasies]))
        sampler_list = ListSampler(*[sampler] * number_of_outputs)
        x_eval_mask = torch.zeros(1, number_of_outputs,
                                  dtype=torch.bool)  # 2 outputs, 1 == True # Change torch.zeros to ones for coupled cKG
        x_eval_mask[0, idx] = 1
        torch.manual_seed(iteration)
        return DecopledHybridConstrainedKnowledgeGradient(model, sampler=sampler_list,
                                                          num_fantasies=number_of_fantasies,
                                                          objective=objective, number_of_raw_points=100,
                                                          evaluate_all_sources=False,
                                                          source_index=idx,
                                                          number_of_restarts=15, X_evaluation_mask=x_eval_mask,
                                                          seed=iteration, penalty_value=penalty_value,
                                                          x_best_location=initial_condition_internal_optimizer)


class DecopledHybridConstrainedKnowledgeGradient(DecoupledAcquisitionFunction, MCAcquisitionFunction):

    def __init__(self, model: Model, sampler: Optional[MCSampler] = None, num_fantasies: Optional[int] = 5,
                 current_value: Optional[Tensor] = None, objective: Optional[MCAcquisitionObjective] = None,
                 posterior_transform: Optional[PosteriorTransform] = None, X_pending: Optional[Tensor] = None,
                 number_of_raw_points: Optional[int] = 64, number_of_restarts: Optional[int] = 20,
                 X_evaluation_mask: Optional[Tensor] = None, seed: Optional[int] = 0,
                 penalty_value: Optional[Tensor] = None, x_best_location: Optional[Tensor] = None,
                 evaluate_all_sources=None, source_index: Optional[int] = None) -> None:

        super().__init__(model=model, sampler=sampler, objective=objective,
                         posterior_transform=posterior_transform, X_pending=X_pending,
                         X_evaluation_mask=X_evaluation_mask)
        self.current_value = current_value
        self.num_fantasies = num_fantasies
        self.penalty_value = penalty_value
        self.number_of_restarts = number_of_restarts
        self.number_of_raw_points = number_of_raw_points
        self.seed = seed
        self.x_best_location = x_best_location
        self.evaluate_all_sources = evaluate_all_sources
        self.source_index = source_index
        self.cached_bestx = {}
        self.cached_fantasy_model = None
        self.use_scipy = False
        if self.source_index is not None:
            if source_index + 1 > self.X_evaluation_mask.shape[1]:
                print("number of outputs does not coincide with source index")
                raise

        if self.num_fantasies not in [1, 3, 5, 7, 9, 11]:
            print("make sure the number of fantasies is odd..as in 3, 5, 7, 9 ,11")
            raise

    def forward(self, X: Tensor) -> Tensor:
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        x_discretisation, fantasised_model = self.compute_optimized_X_discretisation(X)
        if (self.evaluate_all_sources is False) and (self.source_index != 0):
            return self.compute_constraints_kg(x_discretisation, fantasised_model)
        elif (self.evaluate_all_sources is False) and (self.source_index == 0):
            return self.compute_objective_kg(X, x_discretisation)
        else:
            kg_values = self.compute_coupled_kg(X, fantasised_model, x_discretisation)
        return kg_values

    def compute_objective_kg(self, X, x_discretisation):
        if len(x_discretisation.shape) == 4:
            x_discretisation = x_discretisation.unsqueeze(-2)
        feasibility_discretisation = self.precompute_feasibility(x_discretisation, self.model)
        feasibility_X = self.precompute_feasibility(X, self.model).reshape(-1)
        feasibility_best_location = self.precompute_feasibility(self.x_best_location, self.model).reshape(-1)
        number_of_samples = X.shape[0]
        input_dimensions = X.shape[-1]
        kg_values = torch.zeros(number_of_samples)
        for sample_idx in range(number_of_samples):
            assert x_discretisation.shape[2] == number_of_samples;
            assert X.shape[0] == number_of_samples;
            assert feasibility_X.shape[0] == number_of_samples;
            assert feasibility_discretisation.shape[2] == number_of_samples;
            optimal_discretisation_adapted = x_discretisation[:, :, sample_idx, :, :].reshape(-1, input_dimensions)
            xnew = X[sample_idx, ...]
            feasibility_xnew = feasibility_X[sample_idx].reshape(-1)
            concatenated_xnew_discretisation = torch.cat(
                [xnew, self.x_best_location, optimal_discretisation_adapted], dim=0)
            probability_feasibility = torch.cat(
                [feasibility_xnew, feasibility_best_location,
                 feasibility_discretisation[:, :, sample_idx, :].reshape(-1)], dim=0)
            if torch.sum(probability_feasibility) == 0.0:
                kg_values[sample_idx] = torch.sum(probability_feasibility)
            else:
                kg_values[sample_idx] = self.compute_discrete_kg_values(model=self.model,
                                                                        probability_of_feasibility=probability_feasibility,
                                                                        optimal_discretisation=concatenated_xnew_discretisation,
                                                                        x_new=xnew)
        return kg_values

    def compute_coupled_kg(self, X, fantasised_model, x_discretisation):
        concatenated_xnew_discretisation = self.get_concatenated_discretisation(X, x_discretisation)
        feasibility_discretisation = self.precompute_feasibility(concatenated_xnew_discretisation,
                                                                 fantasised_model,
                                                                 squeeze_dim_=None)
        objective_model = self.get_objective_model(self.model)
        # Compute posterior mean, variance, and covariance.
        x_adapted_1 = torch.vstack(
            [concatenated_xnew_discretisation[:, i, ...][None, ...] for i in range(self.num_fantasies)])
        x_adapted_2 = torch.vstack([x_adapted_1[:, :, i, ...][None, ...] for i in range(X.shape[0])])

        objective_posterior = objective_model.posterior(x_adapted_2.squeeze(-2),
                                                        observation_noise=False)
        objective_mean = objective_posterior.mean
        objective_variance = objective_posterior.variance  # (1, )
        objective_noise_variance = self.get_objective_noise(objective_model)  # check this
        objective_posterior_covariance = objective_posterior.mvn.covariance_matrix

        number_of_samples = X.shape[0]
        kg_values = torch.zeros(number_of_samples)
        for sample_idx in range(number_of_samples):
            kg_per_fantasy = torch.zeros(self.num_fantasies)
            for f in range(self.num_fantasies):
                objective_mean_per_realisation = objective_mean[sample_idx, f].squeeze()
                objective_variance_per_realisation = objective_variance[sample_idx, f].squeeze()
                objective_posterior_covariance_per_realisation = objective_posterior_covariance[sample_idx, f].squeeze()
                probability_of_feasibility_per_realisation = feasibility_discretisation[:, f, sample_idx, :].squeeze()
                assert objective_posterior_covariance_per_realisation.shape[-1] == \
                       objective_posterior_covariance_per_realisation.shape[-2];
                if torch.sum(probability_of_feasibility_per_realisation) == 0.0:
                    kg_per_fantasy[f] = torch.sum(probability_of_feasibility_per_realisation)
                else:
                    kg_per_fantasy[f] = self.compute_discrete_kg_values_fast(
                        objective_mean=objective_mean_per_realisation,
                        objective_variance=objective_variance_per_realisation,
                        objective_noise_variance=objective_noise_variance.squeeze(),
                        objective_posterior_covariance=objective_posterior_covariance_per_realisation,
                        probability_of_feasibility=probability_of_feasibility_per_realisation,
                        x_new=X[sample_idx],
                        fantasised_model=fantasised_model,
                        discretisation_per_realisation = concatenated_xnew_discretisation[:, f].squeeze()
                    )
            kg_values[sample_idx] = torch.mean(kg_per_fantasy)
        return kg_values

    def get_concatenated_discretisation(self, X, x_discretisation):
        if len(x_discretisation.shape) == 4:
            x_discretisation = x_discretisation.unsqueeze(-2)
        concatenated_xnew_discretisation = torch.vstack(
            [self.adapt_x_location_dim(X, torch.Size([self.num_fantasies, X.shape[0]])),
             self.adapt_x_location_dim(self.x_best_location, torch.Size([self.num_fantasies, X.shape[0]])),
             x_discretisation])
        return concatenated_xnew_discretisation

    def set_scipy_as_internal_optimizer(self):
        self.use_scipy = True

    def compute_random_discretisation(self, X):
        torch.manual_seed(self.seed)
        fantasy_model = self.model.fantasize(X=X, sampler=self.sampler,
                                             evaluation_mask=self.construct_evaluation_mask(X))
        bounds = torch.tensor([[0.0] * X.shape[-1], [1.0] * X.shape[-1]], dtype=torch.double)
        constrained_posterior_mean_model = ConstrainedPosteriorMean(fantasised_models=fantasy_model,
                                                                    model=self.model,
                                                                    evaluation_mask=self.construct_evaluation_mask(X),
                                                                    penalty_value=self.penalty_value)
        batch_shape = fantasy_model.batch_shape
        best_location_adapted_dimensions = self.adapt_x_location_dim(self.x_best_location, batch_shape)
        if (X.shape[0] in self.cached_bestx):
            restart_points = self.cached_bestx[X.shape[0]]
            return restart_points, fantasy_model
        raw_points = draw_sobol_samples(bounds=bounds,
                                        n=self.number_of_raw_points,
                                        q=1,
                                        batch_shape=batch_shape,
                                        seed=self.seed)

        restart_points = initialize_q_batch(X=raw_points,
                                            Y=constrained_posterior_mean_model(raw_points),
                                            n=self.number_of_restarts, eta=2.0)

        restart_points = torch.cat([restart_points, best_location_adapted_dimensions], dim=0)
        return restart_points, fantasy_model

    def precompute_feasibility(self, X, fantasised_model, squeeze_dim_=None):
        if squeeze_dim_ is not None:
            X = X.squeeze(squeeze_dim_)
        mean, std = self.precompute_constraints_posterior_mean(X, fantasised_model)
        feasibility = self.compute_probability_of_feasibility(mean, std)
        feasibility = torch.atleast_3d(feasibility)
        return feasibility

    @record_io(enabled=False)
    def compute_discrete_kg_values_fast(self, objective_mean,
                                        objective_variance,
                                        objective_noise_variance,
                                        objective_posterior_covariance,
                                        x_new: Tensor,
                                        probability_of_feasibility: Optional[Tensor] = None, **kwargs) -> Tensor:
        """

        Args:
        xnew: A `1 x 1 x d` Tensor with `1` acquisition function evaluations of
            `d` dimensions.
            optimal_discretisation: num_fantasies x d Tensor. Optimal X values for each z in zvalues.

        """
        objective_variance = objective_variance[0]  # (1, )
        objective_posterior_cov_xnew_discretisation = objective_posterior_covariance[: len(x_new), :].reshape(-1,
                                                                                                              1)  # ( 1 + num_X_disc,)
        full_predictive_covariance = (objective_posterior_cov_xnew_discretisation / (
                objective_variance + objective_noise_variance).sqrt())

        predictive_mean = objective_mean.squeeze() * probability_of_feasibility.squeeze() - self.penalty_value * (
                1 - probability_of_feasibility.squeeze())
        predictive_variance = full_predictive_covariance.squeeze() * probability_of_feasibility.squeeze()
        return self.kgcb(a=predictive_mean,
                         b=predictive_variance)

    def precompute_constraints_posterior_mean(self, X, modelList):
        model = self.get_constraints_model(modelList)
        posterior = model.posterior(X)
        mean_sample_idx_ = posterior.mean
        std_sample_idx_ = posterior.variance.sqrt()
        return mean_sample_idx_, std_sample_idx_

    def compute_discrete_kg_values(self, model: Model, x_new: Tensor,
                                   optimal_discretisation: Tensor,
                                   probability_of_feasibility: Optional[Tensor] = None, ) -> Tensor:
        """

        Args:
        xnew: A `1 x 1 x d` Tensor with `1` acquisition function evaluations of
            `d` dimensions.
            optimal_discretisation: num_fantasies x d Tensor. Optimal X values for each z in zvalues.

        """
        objective_model = self.get_objective_model(model)
        # Compute posterior mean, variance, and covariance.
        objective_posterior = objective_model.posterior(optimal_discretisation, observation_noise=False)

        objective_mean = objective_posterior.mean
        objective_variance = objective_posterior.variance[0]  # (1, )
        objective_noise_variance = self.get_objective_noise(objective_model)  # check this
        objective_posterior_covariance = objective_posterior.mvn.covariance_matrix
        # (1 + num_X_disc , 1 + num_X_disc ) # check this.

        objective_posterior_cov_xnew_discretisation = objective_posterior_covariance[: len(x_new), :].reshape(-1,
                                                                                                              1)  # ( 1 + num_X_disc,)
        full_predictive_covariance = (objective_posterior_cov_xnew_discretisation / (
                objective_variance + objective_noise_variance).sqrt())

        predictive_mean = objective_mean.squeeze() * probability_of_feasibility.squeeze() - self.penalty_value * (
                1 - probability_of_feasibility.squeeze())
        predictive_variance = full_predictive_covariance.squeeze() * probability_of_feasibility.squeeze()
        return self.kgcb(a=predictive_mean,
                         b=predictive_variance)

    @staticmethod
    def compute_probability_of_feasibility(mean_constraints, sigma_constraints):
        z = -mean_constraints / sigma_constraints
        probability_feasibility = torch.distributions.Normal(0, 1).cdf(z).prod(dim=-1)
        return probability_feasibility

    def get_objective_model(self, models: ModelListGP):
        return models.models[0]

    def get_constraints_model(self, models: ModelListGP):
        return ModelListGP(*models.models[1:])

    def get_objective_noise(self, model):
        return torch.unique(model.likelihood.noise_covar.noise)

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

        a = a.squeeze().clone()
        b = b.squeeze().clone()
        assert len(a) > 0, "must provide slopes"
        assert len(a) == len(b), f"#intercepts != #slopes, {len(a)}, {len(b)}"

        maxa = torch.max(a)
        # exclude duplicated b (or super duper similar b)
        threshold = 1e-16
        a_0, b_0 = filter_a_b(a, b, threshold)
        # initialize
        idz = [0]
        i_last = 0
        x = [-torch.inf]
        n_lines = len(a_0)
        while i_last < n_lines - 1:
            i_mask = torch.arange(i_last + 1, n_lines)
            x_mask = -(a_0[i_last] - a_0[i_mask]) / (b_0[i_last] - b_0[i_mask])

            best_pos = torch.argmin(x_mask)
            idz.append(i_mask[best_pos])
            x.append(x_mask[best_pos])

            i_last = idz[-1]

        x.append(torch.inf)

        x = torch.Tensor(x)
        idz = torch.LongTensor(idz)
        # found the epigraph, now compute the expectation
        a = a_0[idz]
        b = b_0[idz]

        normal = torch.distributions.Normal(torch.zeros_like(x), torch.ones_like(x))

        pdf = torch.exp(normal.log_prob(x))
        cdf = normal.cdf(x)

        kg = torch.sum(a * (cdf[1:] - cdf[:-1]) + b * (pdf[:-1] - pdf[1:]))
        kg -= maxa
        return kg

    def compute_constraints_kg(self, X: Tensor, fantasy_model: Model) -> Tensor:
        constrained_posterior_mean_model = ConstrainedPosteriorMean(fantasised_models=fantasy_model,
                                                                    model=self.model,
                                                                    evaluation_mask=self.construct_evaluation_mask(X),
                                                                    penalty_value=self.penalty_value)
        current_model = ConstrainedPosteriorMean(self.model, penalty_value=self.penalty_value)
        with torch.enable_grad():
            bestvals = constrained_posterior_mean_model(X)
            assert bestvals.shape[1] == self.num_fantasies;
            current_best_value_estimation = torch.max(current_model(X.reshape(-1, 1, X.shape[-1])))
        diff = torch.max(bestvals, current_best_value_estimation) - current_best_value_estimation
        bestval_sample = diff.max(dim=0)[0]
        kgvals = bestval_sample.mean(dim=0)
        return torch.atleast_1d(kgvals)

    def compute_optimized_X_discretisation(self, X):
        if X.shape[0] == 1 and len(X.shape) == 3:
            X = X.squeeze(0)
        restart_points, fantasy_model = self.compute_random_discretisation(X)
        bounds = torch.tensor([[0.0] * X.shape[-1],
                               [1.0] * X.shape[-1]], dtype=torch.double)
        with torch.enable_grad():
            unconstrained_posterior_mean = ConstrainedPosteriorMean(
                model=fantasy_model,
                penalty_value=self.penalty_value)

            if self.use_scipy:
                bestx, _ = gen_candidates_scipy(initial_conditions=restart_points,
                                                acquisition_function=unconstrained_posterior_mean,
                                                lower_bounds=bounds[0],
                                                upper_bounds=bounds[1],
                                                options={"maxiter": 100})
            else:
                bestx, _ = gen_candidates_torch(initial_conditions=restart_points,
                                                acquisition_function=unconstrained_posterior_mean,
                                                lower_bounds=bounds[0],
                                                upper_bounds=bounds[1],
                                                options={"maxiter": 100})
        self.cached_bestx[X.shape[0]] = bestx.clone().detach()
        return bestx, fantasy_model,

    def adapt_x_location_dim(self, X, batch_shape):
        best_location_adapted_dimensions = torch.ones((1, *batch_shape, 1, self.x_best_location.shape[-1]),
                                                      dtype=torch.float64)
        best_location_adapted_dimensions[..., :] = X
        return best_location_adapted_dimensions


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
        constrained_posterior_mean_model = ConstrainedPosteriorMean(fantasised_models=fantasy_model,
                                                                    model=self.model,
                                                                    evaluation_mask=self.construct_evaluation_mask(X),
                                                                    penalty_value=self.penalty_value)
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
        constrained_posterior_mean_model = ConstrainedPosteriorMean(fantasised_models=fantasy_model,
                                                                    model=self.model,
                                                                    evaluation_mask=self.construct_evaluation_mask(X),
                                                                    penalty_value=self.penalty_value)
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
                    acquisition_function=ConstrainedPosteriorMean(fantasised_models=fantasy_model,
                                                                  model=self.model,
                                                                  evaluation_mask=self.construct_evaluation_mask(X)),
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
