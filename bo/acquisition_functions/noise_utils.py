"""Observation-noise helpers shared by the KG acquisition functions.

With an outcome transform such as ``Standardize``, BoTorch fits the GP on
transformed targets, so ``likelihood.noise_covar.noise`` is in *transformed*
units, whereas ``model.posterior(X)`` returns moments in *original* units.
Any formula that mixes the two (e.g. sigma_tilde = Sigma(x', x) /
sqrt(Sigma(x, x) + noise)) must convert the noise first.
"""
from botorch.models.transforms.outcome import Standardize
from torch import Tensor


def observation_noise_variance(gp_model) -> Tensor:
    """Homoscedastic observation-noise variance of ``gp_model`` in original units.

    Equivalent to ``posterior(X, observation_noise=True).variance -
    posterior(X).variance`` but costs a single scalar multiplication.

    For fixed-noise models (``train_Yvar`` given) the first training point's
    noise is returned, matching the previous ``.view(-1)[0]`` behaviour.

    Raises:
        NotImplementedError: for outcome transforms other than ``Standardize``,
            whose noise cannot be mapped back to original units by a scale
            factor. Failing loudly beats silently mixing units.
    """
    noise = gp_model.likelihood.noise_covar.noise.view(-1)[0]
    transform = getattr(gp_model, "outcome_transform", None)
    if transform is None:
        return noise
    if isinstance(transform, Standardize):
        if transform.stdvs.numel() != 1:
            raise NotImplementedError(
                "observation_noise_variance expects single-output Standardize(m=1); "
                f"got stdvs of shape {tuple(transform.stdvs.shape)}."
            )
        return noise * transform.stdvs.view(-1)[0] ** 2
    raise NotImplementedError(
        f"Cannot convert likelihood noise to original units for outcome transform "
        f"{type(transform).__name__}; only Standardize is supported."
    )
