# transfit/sbi/prior.py
"""Wrap TransFit's MixedBoundsPrior as a PyTorch Distribution for sbi."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.distributions import Distribution, constraints

from ..priors.common import MixedBoundsPrior


class TransFitPrior(Distribution):
    """Wraps MixedBoundsPrior as a PyTorch distribution for sbi.

    Supports both uniform and log-uniform (Jeffreys) dimensions via
    the MixedBoundsPrior log_flags mechanism.
    """

    has_rsample = False
    arg_constraints: dict = {}  # type: ignore[assignment]

    def __init__(self, mixed_prior: MixedBoundsPrior):
        self._prior = mixed_prior
        self._device = torch.device("cpu")
        self._low = torch.as_tensor(
            mixed_prior.bounds[:, 0], dtype=torch.float32, device=self._device
        )
        self._high = torch.as_tensor(
            mixed_prior.bounds[:, 1], dtype=torch.float32, device=self._device
        )
        # Cache log_flags as tensor for vectorized log_prob
        self._log_flags = torch.as_tensor(
            np.asarray(mixed_prior.log_flags, bool), dtype=torch.bool, device=self._device
        )
        batch_shape = torch.Size()
        event_shape = torch.Size([self._low.shape[0]])
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=False)

    @property
    def support(self):
        return constraints.independent(
            constraints.interval(self._low, self._high), 1
        )

    @property
    def param_names(self):
        return list(self._prior.param_names)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "TransFitPrior":
        device = torch.device(device)
        self._device = device
        self._low = self._low.to(device)
        self._high = self._high.to(device)
        self._log_flags = self._log_flags.to(device)
        return self

    def sample(self, sample_shape=torch.Size()):
        n = 1
        if len(sample_shape) > 0:
            n = int(sample_shape[0])
        samples = self._prior.sample(n)
        return torch.as_tensor(samples, dtype=torch.float32, device=self._device)

    def log_prob(self, value):
        """Vectorized log-probability.

        For uniform dims: 0 if in bounds, -inf otherwise.
        For log-uniform dims: -log(x) if in positive bounds, -inf otherwise.
        Normalization constants are dropped (only relative values matter for sbi).
        """
        value = torch.as_tensor(value, dtype=torch.float32, device=self._device)
        if value.dim() == 1:
            value = value.unsqueeze(0)

        # Bounds check
        in_bounds = (value >= self._low) & (value <= self._high)
        all_in = in_bounds.all(dim=-1).float()

        lp = torch.zeros(value.shape[0], dtype=torch.float32, device=self._device)

        # Jeffreys penalty for log-uniform dims: -log(x)
        if self._log_flags.any():
            log_dims = value[:, self._log_flags]
            positive = (log_dims > 0).float().all(dim=-1)
            lp = lp + all_in * positive * (-torch.log(
                torch.clamp(log_dims, min=1e-30)
            ).sum(dim=-1))
            # Invalidate if any log-uniform dim is non-positive
            neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=self._device)
            lp = torch.where(
                all_in > 0.5,
                torch.where(positive > 0.5, lp, neg_inf),
                neg_inf,
            )
        else:
            neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=self._device)
            lp = torch.where(all_in > 0.5, lp, neg_inf)

        return lp
