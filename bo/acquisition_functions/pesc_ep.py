"""PESC with the paper's full factor set (the faithful variant).

Thin acquisition-function layer over :mod:`bo.acquisition_functions.pesc_sites`,
which runs expectation propagation over the (N+1)-dimensional vector
``[value(x*), value(u_1..u_N)]`` once per sampled optimum and supplies the
conditioned moments that Eqs. (36)-(37) in :mod:`bo.acquisition_functions.pesc`
consume.  Relative to the no-EP path in ``pesc.py`` this adds the ``h_n``
observed-point factors of supplementary Eq. (4) -- "``u_n`` does not dominate x*".

Measured against the paper's own rejection-sampling reference (Sec. 4.1, 5 seeds,
in ``tests/acquisition_functions/test_pesc_fidelity.py``):

    source      no-EP path            this module
    objective   0.73 mean (0.34 min)  **0.96 mean (0.94 min)**
    constraint  0.96 mean             0.96 mean

The objective gain is the point: without ``h_n`` nothing forces ``f(x*)`` to beat
the observed data, which is the dominant source of objective information.

History: an earlier version of this module ran EP on a 2-D ``(x, x*)`` block per
candidate x and substituted a single hard "incumbent" truncation of ``f(x*)`` at the
best predicted-feasible observed value in place of the ``h_n`` factors.  That hack
pushed in the right direction -- which the measurement above retrospectively
justifies -- but it is superseded and has been removed, along with the
``observed_factors`` switch that toggled it.
"""

from typing import Optional

from botorch.models.model import Model
from torch import Tensor

from bo.acquisition_functions.pesc import _PESCSource
from bo.acquisition_functions.pesc_sites import (
    _MAX_SWEEPS,
    EP_STATS,
    PESCFaithfulConditioner,
    pesc_faithful_conditional_variances,
)

__all__ = [
    "pesc_ep_conditional_variances",
    "PESCObjectiveEP",
    "PESCConstraintEP",
    "EP_STATS",
]


def pesc_ep_conditional_variances(
    model: Model,
    X: Tensor,
    x_star: Tensor,
    add_observation_noise: bool = True,
    run_ep: bool = True,
    max_sweeps: int = _MAX_SWEEPS,
    locations: Optional[Tensor] = None,
):
    """Conditioned predictive variances with the full factor set.

    Signature matches :func:`bo.acquisition_functions.pesc.pesc_conditional_variances`
    so the two paths are drop-in swappable for A/B measurement.

    Returns:
        ``(v_prior, v_cond)`` each ``(b, M, K+1)``.
    """
    return pesc_faithful_conditional_variances(
        model, X, x_star,
        add_observation_noise=add_observation_noise,
        run_ep=run_ep,
        max_sweeps=max_sweeps,
        locations=locations,
    )


class _PESCEPSource(_PESCSource):
    """Per-source acquisition backed by the (N+1)-dim EP.

    The EP is run once, lazily, and reused for every candidate batch -- otherwise
    ``optimize_acqf`` would pay for a full EP on each line-search evaluation.  A
    shared conditioner across sources of the same iteration would be better still,
    but ``acquisition_function_factory`` builds one acquisition per source.
    """

    def __init__(self, model: Model, output_index: int, x_star: Tensor,
                 max_sweeps: int = _MAX_SWEEPS, conditioner=None):
        super().__init__(model, source_model_index=output_index, x_star=x_star)
        self.output_index = output_index
        self.max_sweeps = max_sweeps
        self._conditioner = conditioner

    def _conditional_variances(self, X: Tensor):
        if self._conditioner is None:
            self._conditioner = PESCFaithfulConditioner(
                self.model, self.x_star, max_sweeps=self.max_sweeps
            )
        return self._conditioner.conditional_variances(X.squeeze(1))


def PESCObjectiveEP(model: Model, x_star: Tensor, conditioner=None, **kwargs):
    """Objective source term (source 0) with the full factor set.

    ``conditioner`` lets one converged EP be shared across all sources of an
    iteration, as Spearmint does (its ``predictEP`` returns every task's variance
    from a single EP solution).  Worth only a few percent here -- profiling put EP
    at 1.4-6.7% of iteration cost -- but it is free to pass through.
    """
    return _PESCEPSource(model, output_index=0, x_star=x_star, conditioner=conditioner,
                         max_sweeps=kwargs.get("max_sweeps", _MAX_SWEEPS))


def PESCConstraintEP(model: Model, constraint_index: int, x_star: Tensor,
                     conditioner=None, **kwargs):
    """Constraint ``constraint_index`` (0-based) with the full factor set."""
    return _PESCEPSource(model, output_index=constraint_index + 1, x_star=x_star,
                         conditioner=conditioner,
                         max_sweeps=kwargs.get("max_sweeps", _MAX_SWEEPS))
