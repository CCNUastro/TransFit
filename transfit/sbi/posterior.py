# transfit/sbi/posterior.py
"""SBIPosterior container -- trained SBI posterior analogous to FitResult."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SBIPosterior:
    """Trained SBI posterior, analogous to FitResult.

    Wraps an sbi DirectPosterior together with the embedding network
    and metadata needed for inference on new observations.
    """

    model: str
    param_names: List[str]
    posterior: Any  # sbi.inference.posteriors.DirectPosterior
    embedding_net: nn.Module
    meta: Dict[str, Any] = field(default_factory=dict)
    band_vocabulary: Optional[List[str]] = None
    t_range: tuple = (0.0, 150.0)
    mode: str = "multiband"  # "bolometric" or "multiband"

    def sample(
        self,
        n: int,
        y_obs: np.ndarray,
        *,
        t_days: Optional[np.ndarray] = None,
        band: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw n posterior samples given observed data.

        Parameters
        ----------
        n : int
            Number of samples.
        y_obs : np.ndarray
            Observed values.
        t_days : np.ndarray, optional
            Observation times.
        band : np.ndarray, optional
            Band labels.
        mask : np.ndarray, optional
            Validity mask.
        seed : int, optional
            Random seed.

        Returns
        -------
        samples : np.ndarray, shape (n, ndim)
        """
        x_encoded = self._encode_observation(y_obs, t_days=t_days, band=band)
        if seed is not None:
            torch.manual_seed(seed)
        samples = self.posterior.sample(
            (n,), x=x_encoded, show_progress_bars=False
        )
        return np.asarray(samples.detach().cpu().numpy(), float)

    def log_prob(
        self,
        theta: np.ndarray,
        y_obs: np.ndarray,
        *,
        t_days: Optional[np.ndarray] = None,
        band: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate log posterior probability at given theta values.

        Returns
        -------
        log_prob : np.ndarray, shape (n_samples,)
        """
        x_encoded = self._encode_observation(y_obs, t_days=t_days, band=band)
        theta_t = torch.as_tensor(theta, dtype=torch.float32)
        if theta_t.dim() == 1:
            theta_t = theta_t.unsqueeze(0)
        lp = self.posterior.log_prob(theta_t, x=x_encoded)
        return np.asarray(lp.detach().cpu().numpy(), float)

    def map_estimate(
        self,
        y_obs: np.ndarray,
        *,
        t_days: Optional[np.ndarray] = None,
        band: Optional[np.ndarray] = None,
        n_candidates: int = 10000,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """MAP estimate via sample-and-pick-best.

        Draw n_candidates samples and return the one with highest log_prob.
        """
        samples = self.sample(
            n_candidates, y_obs, t_days=t_days, band=band, seed=seed
        )
        lp = self.log_prob(samples, y_obs, t_days=t_days, band=band)
        best_idx = int(np.argmax(lp))
        return {name: float(samples[best_idx, i]) for i, name in enumerate(self.param_names)}

    def median(
        self,
        y_obs: np.ndarray,
        *,
        t_days: Optional[np.ndarray] = None,
        band: Optional[np.ndarray] = None,
        n: int = 5000,
    ) -> Dict[str, float]:
        """Posterior median."""
        samples = self.sample(n, y_obs, t_days=t_days, band=band)
        med = np.median(samples, axis=0)
        return {name: float(med[i]) for i, name in enumerate(self.param_names)}

    def _encode_observation(
        self,
        y_obs: np.ndarray,
        *,
        t_days: Optional[np.ndarray] = None,
        band: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Encode a single observation into raw features for the embedding network.

        Returns the raw feature tensor (with validity column for SetSummaryNet),
        NOT the embedding output. The sbi DirectPosterior will internally call
        embedding_net(x) during sampling/evaluation, so we must not pre-apply it.
        """
        from .embedding import encode_observations, SetSummaryNet

        y_obs = np.asarray(y_obs, float).reshape(-1)

        if isinstance(self.embedding_net, SetSummaryNet):
            features, _ = encode_observations(
                y_obs,
                t_days=t_days,
                band=band,
                band_vocabulary=self.band_vocabulary,
                t_range=self.t_range,
            )
            # Append validity indicator column (all 1.0, no padding)
            validity = np.ones((features.shape[0], 1), dtype=np.float32)
            features_with_mask = np.concatenate([features, validity], axis=1)
            # Shape: (1, n_obs, feature_dim+1)
            return torch.as_tensor(
                features_with_mask[np.newaxis, :, :], dtype=torch.float32
            )

        if isinstance(self.embedding_net, nn.Module):
            features, _ = encode_observations(
                y_obs,
                t_days=t_days,
                band=band,
                band_vocabulary=self.band_vocabulary,
                t_range=self.t_range,
            )
            # Flatten to 1D: (1, n_obs * feature_dim)
            return torch.as_tensor(
                features.reshape(1, -1), dtype=torch.float32
            )

        # Fallback: raw 1D
        return torch.as_tensor(y_obs, dtype=torch.float32).unsqueeze(0)
