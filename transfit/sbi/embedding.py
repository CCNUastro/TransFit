# transfit/sbi/embedding.py
"""Variable-cadence embedding networks for SBI.

Provides DeepSet-style (SetSummaryNet) and MLP (MLPEmbeddingNet) architectures
to encode variable-size observation sets into fixed-size summary vectors.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class SetSummaryNet(nn.Module):
    """DeepSet-style embedding for variable-size observation sets.

    Input: (batch, N_obs, feature_dim)  where N_obs can vary
    Output: (batch, output_dim)         fixed-size summary

    The last column of the input is treated as a validity indicator:
    1.0 = valid, 0.0 = padded.  When an explicit mask is not passed
    to forward(), the mask is automatically inferred from this column.
    This allows the network to be used as a standard nn.Module inside
    sbi's posterior_nn, which only calls embedding_net(x).
    """

    def __init__(self, feature_dim: int, hidden_features: int = 64, output_dim: int = 32):
        # phi operates on feature_dim - 1 (excluding the mask channel)
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(feature_dim - 1, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, N_obs, feature_dim)
        # mask: (batch, N_obs) boolean, True=valid
        if mask is None:
            # Infer mask from the last feature column
            mask = (x[..., -1] != 0.0)
        features = x[..., :-1]  # strip mask channel
        h = self.phi(features)
        h = h * mask.unsqueeze(-1)
        n_valid = mask.sum(dim=-1, keepdim=True).clamp(min=1.0).float()
        h = h.sum(dim=1) / n_valid
        return self.rho(h)


class MLPEmbeddingNet(nn.Module):
    """Simple MLP for fixed-size observation vectors.

    Input: (batch, input_dim)
    Output: (batch, output_dim)
    """

    def __init__(self, input_dim: int, hidden_features: int = 64, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def encode_observations(
    y_values: np.ndarray,
    *,
    t_days: Optional[np.ndarray] = None,
    band: Optional[np.ndarray] = None,
    band_vocabulary: Optional[List[str]] = None,
    t_range: Tuple[float, float] = (0.0, 150.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode observations into a padded feature array + validity mask.

    For multiband: feature_dim = 1 + n_bands + 1 = [t_norm, band_onehot..., y_value]
    For bolometric (band=None): feature_dim = 2 = [t_norm, y_value]

    Parameters
    ----------
    y_values : np.ndarray
        Observed values (magnitudes or log10-luminosity).
    t_days : np.ndarray, optional
        Observation times. If None, uses sequential indices.
    band : np.ndarray, optional
        Band labels. If None, treats as bolometric (no band encoding).
    band_vocabulary : list[str], optional
        Ordered list of band names for one-hot encoding. Required if band is given.
    t_range : tuple
        (t_min, t_max) for normalizing time to [0, 1].

    Returns
    -------
    features : np.ndarray, shape (n_obs, feature_dim)
    mask : np.ndarray, shape (n_obs,), all True
    """
    y_values = np.asarray(y_values, float).reshape(-1)
    n_obs = len(y_values)

    if t_days is not None:
        t_days = np.asarray(t_days, float).reshape(-1)
        t_lo, t_hi = float(t_range[0]), float(t_range[1])
        if t_hi <= t_lo:
            t_hi = t_lo + 1.0
        t_norm = (t_days - t_lo) / (t_hi - t_lo)
    else:
        t_norm = np.zeros(n_obs, dtype=float)

    if band is not None and band_vocabulary is not None:
        band = np.asarray(band, object).reshape(-1)
        n_bands = len(band_vocabulary)
        band_idx = {b: i for i, b in enumerate(band_vocabulary)}
        onehot = np.zeros((n_obs, n_bands), dtype=float)
        for j, b in enumerate(band):
            if b in band_idx:
                onehot[j, band_idx[b]] = 1.0
        features = np.column_stack([t_norm, onehot, y_values])
    else:
        features = np.column_stack([t_norm, y_values])

    mask = np.ones(n_obs, dtype=bool)
    return features.astype(np.float32), mask


def encode_batch(
    batch_y: List[np.ndarray],
    *,
    t_days_list: Optional[List[np.ndarray]] = None,
    band_list: Optional[List[np.ndarray]] = None,
    band_vocabulary: Optional[List[str]] = None,
    t_range: Tuple[float, float] = (0.0, 150.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of variable-length observations into padded tensors.

    Returns
    -------
    features : torch.Tensor, shape (batch, max_n_obs, feature_dim + 1)
        The last column is a validity indicator (1.0 = valid, 0.0 = padded),
        used by SetSummaryNet to automatically infer the mask.
    mask : torch.Tensor, shape (batch, max_n_obs), True for valid entries
    """
    encoded = []
    for i, y in enumerate(batch_y):
        td = t_days_list[i] if t_days_list is not None else None
        bd = band_list[i] if band_list is not None else None
        f, m = encode_observations(
            y, t_days=td, band=bd,
            band_vocabulary=band_vocabulary, t_range=t_range,
        )
        encoded.append((f, m))

    max_n_obs = max(f.shape[0] for f, _ in encoded)
    if band_list is not None and band_vocabulary is not None:
        feature_dim = 1 + len(band_vocabulary) + 1
    else:
        feature_dim = encoded[0][0].shape[1]

    batch_size = len(encoded)
    # +1 column for validity indicator
    features = np.zeros((batch_size, max_n_obs, feature_dim + 1), dtype=np.float32)
    masks = np.zeros((batch_size, max_n_obs), dtype=bool)

    for i, (f, m) in enumerate(encoded):
        n = f.shape[0]
        features[i, :n, :-1] = f
        features[i, :n, -1] = 1.0  # validity indicator
        masks[i, :n] = m

    return (
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(masks, dtype=torch.bool),
    )
