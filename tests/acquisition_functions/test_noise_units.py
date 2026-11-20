"""Observation noise must be in the same (original) units as model.posterior()."""
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Log, Standardize
from botorch.sampling import SobolQMCNormalSampler

from bo.acquisition_functions.noise_utils import observation_noise_variance
from bo.acquisition_functions.refactored_acquisition_functions import BaseConstrainedKG

DT = torch.double


def _gp(scale, train_Yvar=None, outcome_transform="std"):
    g = torch.Generator().manual_seed(0)
    X = torch.rand(20, 2, generator=g, dtype=DT)
    Y = scale * (torch.sin(6 * X[:, :1]) + 0.3 * torch.randn(20, 1, generator=g, dtype=DT))
    tf = Standardize(m=1) if outcome_transform == "std" else outcome_transform
    gp = SingleTaskGP(X, Y, train_Yvar=train_Yvar, outcome_transform=tf)
    # Fixed, non-trivial hyperparameters instead of fitting: keeps the test fast and deterministic.
    if train_Yvar is None:
        gp.likelihood.noise = torch.tensor(0.05, dtype=DT)
    gp.eval()
    return gp


def _obs_noise_reference(gp, X):
    """What BoTorch itself adds for observation noise, in original units."""
    with torch.no_grad():
        return gp.posterior(X, observation_noise=True).variance - gp.posterior(X).variance


@pytest.mark.parametrize("scale", [0.1, 1.0, 30.0])
def test_learned_noise_matches_botorch(scale):
    gp = _gp(scale)
    X = torch.rand(5, 2, dtype=DT)
    ref = _obs_noise_reference(gp, X)
    assert torch.allclose(observation_noise_variance(gp).expand_as(ref), ref, rtol=1e-6)
    # Guard against the regression: with scale != 1 the raw noise is in the wrong units.
    if scale != 1.0:
        assert not torch.allclose(gp.likelihood.noise_covar.noise.view(-1)[0].expand_as(ref), ref, rtol=1e-2)


def test_fixed_noise_recovers_train_yvar():
    gp = _gp(30.0, train_Yvar=torch.full((20, 1), 1e-3, dtype=DT))
    assert torch.allclose(observation_noise_variance(gp), torch.tensor(1e-3, dtype=DT), rtol=1e-6)


def test_fantasy_model_keeps_original_units():
    gp = _gp(30.0)
    fm = gp.fantasize(torch.rand(1, 2, dtype=DT), sampler=SobolQMCNormalSampler(torch.Size([3])))
    assert torch.allclose(observation_noise_variance(fm), observation_noise_variance(gp), rtol=1e-6)


def test_no_transform_returns_raw_noise():
    gp = _gp(1.0, outcome_transform=None)
    assert torch.equal(observation_noise_variance(gp), gp.likelihood.noise_covar.noise.view(-1)[0])


def test_unsupported_transform_raises():
    gp = _gp(1.0, outcome_transform=None)
    gp.outcome_transform = Log()
    with pytest.raises(NotImplementedError):
        observation_noise_variance(gp)


@pytest.mark.parametrize("scale", [0.1, 30.0])
def test_sigma_tilde_matches_observation_noise_posterior(scale):
    gp = _gp(scale)
    X_full = torch.rand(4, 9, 2, dtype=DT)  # batch of (candidate + 8 disc points)
    with torch.no_grad():
        st, _, _ = BaseConstrainedKG._sigma_tilde(gp, X_full)
        cov = gp.posterior(X_full).mvn.covariance_matrix
        var_obs_cand = gp.posterior(X_full[:, :1], observation_noise=True).variance.squeeze(-1)
    expected = cov[:, :, 0] / var_obs_cand.sqrt()
    assert torch.allclose(st, expected, rtol=1e-6)
