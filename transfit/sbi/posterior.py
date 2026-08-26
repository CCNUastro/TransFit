# transfit/sbi/posterior.py
"""SBIPosterior container -- trained SBI posterior analogous to FitResult."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def _module_device(module: Optional[nn.Module]) -> Optional[torch.device]:
    if not isinstance(module, nn.Module):
        return None
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


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

    @property
    def device(self) -> torch.device:
        for module in (
            self._posterior_estimator_module(),
            self.embedding_net if isinstance(self.embedding_net, nn.Module) else None,
        ):
            device = _module_device(module)
            if device is not None:
                return device
        return torch.device("cpu")

    def to(self, device: torch.device | str) -> "SBIPosterior":
        """Move the embedding network and underlying posterior estimator."""
        device = torch.device(device)
        device_str = str(device)

        if isinstance(self.embedding_net, nn.Module):
            self.embedding_net.to(device)

        if self.posterior is not None:
            for attr in ("_device", "device"):
                if hasattr(self.posterior, attr):
                    try:
                        setattr(self.posterior, attr, device_str)
                    except Exception:
                        pass

        posterior_module = self._posterior_estimator_module()
        if posterior_module is not None:
            posterior_module.to(device)

        potential_fn = getattr(self.posterior, "potential_fn", None)
        if potential_fn is not None:
            for attr in ("_device", "device"):
                if hasattr(potential_fn, attr):
                    try:
                        setattr(potential_fn, attr, device_str)
                    except Exception:
                        pass

        for prior_obj in (
            getattr(self.posterior, "prior", None),
            getattr(self.posterior, "_prior", None),
            getattr(potential_fn, "prior", None),
            getattr(potential_fn, "_prior", None),
        ):
            if prior_obj is not None and hasattr(prior_obj, "to"):
                prior_obj.to(device)

        potential_module = getattr(potential_fn, "posterior_estimator", None)
        if (
            isinstance(potential_module, nn.Module)
            and potential_module is not posterior_module
        ):
            potential_module.to(device)

        return self

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
        posterior = self._require_posterior()
        x_encoded = self._encode_observation(
            y_obs, t_days=t_days, band=band, mask=mask
        )
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        samples = posterior.sample(
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
        posterior = self._require_posterior()
        x_encoded = self._encode_observation(y_obs, t_days=t_days, band=band)
        theta_t = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        if theta_t.dim() == 1:
            theta_t = theta_t.unsqueeze(0)
        lp = posterior.log_prob(theta_t, x=x_encoded)
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
        mask: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Encode a single observation into raw features for the embedding network.

        Returns the raw feature tensor (with validity column for SetSummaryNet),
        NOT the embedding output. The sbi DirectPosterior will internally call
        embedding_net(x) during sampling/evaluation, so we must not pre-apply it.
        """
        from .embedding import encode_observations, SetSummaryNet

        y_obs = np.asarray(y_obs, float).reshape(-1)
        device = self.device

        if isinstance(self.embedding_net, SetSummaryNet):
            features, _ = encode_observations(
                y_obs,
                t_days=t_days,
                band=band,
                band_vocabulary=self.band_vocabulary,
                t_range=self.t_range,
            )
            if mask is None:
                validity = np.ones((features.shape[0], 1), dtype=np.float32)
            else:
                mask_array = np.asarray(mask, bool).reshape(-1)
                if len(mask_array) != features.shape[0]:
                    raise ValueError("mask and y_obs must have the same length.")
                validity = mask_array.astype(np.float32).reshape(-1, 1)
            features_with_mask = np.concatenate([features, validity], axis=1)
            # Shape: (1, n_obs, feature_dim+1)
            return self._pad_encoded_observation(torch.as_tensor(
                features_with_mask[np.newaxis, :, :],
                dtype=torch.float32,
                device=device,
            ))

        if isinstance(self.embedding_net, nn.Module):
            features, _ = encode_observations(
                y_obs,
                t_days=t_days,
                band=band,
                band_vocabulary=self.band_vocabulary,
                t_range=self.t_range,
            )
            # Flatten to 1D: (1, n_obs * feature_dim)
            return self._pad_encoded_observation(torch.as_tensor(
                features.reshape(1, -1),
                dtype=torch.float32,
                device=device,
            ))

        # Fallback: raw 1D
        return self._pad_encoded_observation(
            torch.as_tensor(y_obs, dtype=torch.float32, device=device).unsqueeze(0)
        )

    def _require_posterior(self):
        if self.posterior is None:
            raise RuntimeError(
                "This SBIPosterior does not include a serialized sbi posterior object. "
                "It can be inspected, but it cannot sample or evaluate log_prob."
            )
        return self.posterior

    def _posterior_estimator_module(self) -> Optional[nn.Module]:
        posterior = self.posterior
        if posterior is None:
            return None

        for candidate in (
            getattr(posterior, "posterior_estimator", None),
            getattr(posterior, "_posterior_estimator", None),
            getattr(getattr(posterior, "potential_fn", None), "posterior_estimator", None),
            getattr(getattr(posterior, "_potential_fn", None), "posterior_estimator", None),
        ):
            if isinstance(candidate, nn.Module):
                return candidate
        return None

    def _pad_encoded_observation(self, x: torch.Tensor) -> torch.Tensor:
        target_shape = self.meta.get("x_event_shape")
        if target_shape is None:
            return x

        target = tuple(int(v) for v in target_shape)
        actual = tuple(int(v) for v in x.shape[1:])
        if actual == target:
            return x

        if len(target) == 2:
            if actual[1] != target[1]:
                raise RuntimeError(
                    f"Encoded observation feature shape {actual} does not match trained feature shape {target}."
                )
            if actual[0] > target[0]:
                raise RuntimeError(
                    f"Observation has {actual[0]} epochs, but the trained SBI posterior expects at most {target[0]}."
                )
            padded = torch.zeros((x.shape[0], target[0], target[1]), dtype=x.dtype, device=x.device)
            padded[:, :actual[0], :] = x
            return padded

        if len(target) == 1:
            if actual[0] > target[0]:
                raise RuntimeError(
                    f"Encoded observation length {actual[0]} exceeds the trained SBI shape {target[0]}."
                )
            padded = torch.zeros((x.shape[0], target[0]), dtype=x.dtype, device=x.device)
            padded[:, :actual[0]] = x
            return padded

        raise RuntimeError(
            f"Unsupported trained SBI event shape {target}; expected a 1D or 2D event shape."
        )
