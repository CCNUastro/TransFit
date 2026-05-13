# transfit/sbi/__init__.py
"""Simulation-Based Inference (SBI) for TransFit.

Wraps TransFit forward models as simulators and uses the sbi library
for amortized Neural Posterior Estimation (NPE).

Public API
----------
train_sbi : Train an NPE posterior for a given model and observation context.
infer_sbi : Quick inference helper (wraps SBIPosterior.sample).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..api import (
    _split_prior_specs,
    _apply_log10_priors,
    _split_sampling,
)
from ..model_registry import canonical_model_name
from ..priors import build_bounds, MixedBoundsPrior

from .embedding import SetSummaryNet, MLPEmbeddingNet, encode_batch
from .posterior import SBIPosterior
from .prior import TransFitPrior
from .simulator import make_bolometric_simulator, make_multiband_simulator
from .training import generate_training_data
from .io import save_posterior, load_posterior
from .diagnostics import simulation_based_calibration, posterior_predictive_check


def train_sbi(
    *,
    model: str,
    mode: str = "multiband",
    # Observation context
    z: Optional[float] = None,
    distance_modulus: Optional[float] = None,
    filters: Optional[Dict] = None,
    y_kind: str = "mag",
    mag_system: str = "ab",
    extinction=None,
    # Prior specification
    priors: Optional[Dict[str, Any]] = None,
    fixed: Optional[Dict[str, float]] = None,
    # Training data generation
    n_simulations: int = 5000,
    noise_sigma: Optional[float] = None,
    noise_model: Optional[Callable] = None,
    # Cadence templates for variable-length training
    cadence_templates: Optional[List[Dict[str, Any]]] = None,
    n_epochs_range: Tuple[int, int] = (5, 20),
    bands_pool: Optional[List[str]] = None,
    t_range: Tuple[float, float] = (1.0, 120.0),
    # Neural network
    embedding_net: Optional[nn.Module] = None,
    hidden_features: int = 64,
    num_transforms: int = 5,
    training_batch_size: int = 50,
    max_num_epochs: int = 100,
    learning_rate: float = 5e-4,
    device: Optional[str] = None,
    # Data generation
    n_workers: int = 1,
    seed: int = 42,
    cache_path: Optional[str] = None,
    show_progress: bool = True,
    # Forward model kwargs
    Nx: int = 20,
    Ny: int = 50,
    t_max_days: float = 150.0,
) -> SBIPosterior:
    """Train a Neural Posterior Estimator for amortized inference.

    Parameters
    ----------
    model : str
        TransFit model name ("nickel", "magnetar", "magnetar_ni").
    mode : str
        "bolometric" or "multiband".
    z : float, optional
        Redshift.
    distance_modulus : float, optional
        Distance modulus (required for multiband).
    filters : dict, optional
        Filter map for multiband.
    y_kind : str
        "mag" or "flux".
    mag_system : str
        "ab" or "vega".
    extinction : optional
        Extinction specification.
    priors : dict, optional
        Prior bounds per parameter (same format as fit_multiband).
    fixed : dict, optional
        Fixed parameter values.
    n_simulations : int
        Number of training simulations.
    noise_sigma : float, optional
        Homoscedastic Gaussian noise std.
    noise_model : callable, optional
        Custom noise injection.
    cadence_templates : list[dict], optional
        Explicit list of {"t_days": ..., "band": ...} patterns.
    n_epochs_range : tuple[int, int]
        (min, max) number of epochs for random cadence generation.
    bands_pool : list[str], optional
        Pool of band names for random cadence generation.
    t_range : tuple[float, float]
        (t_min, t_max) for random cadence generation.
    embedding_net : nn.Module, optional
        Custom embedding network. Default: SetSummaryNet.
    hidden_features : int
        Hidden layer size in embedding net and normalizing flow.
    num_transforms : int
        Number of normalizing flow transforms.
    training_batch_size : int
        Training batch size.
    max_num_epochs : int
        Maximum training epochs.
    learning_rate : float
        Learning rate.
    device : str, optional
        Torch device for NPE training/inference ("cpu", "cuda", "cuda:0", ...).
        Defaults to CUDA when available, otherwise CPU.
    n_workers : int
        Parallel workers for simulation.
    seed : int
        Random seed.
    cache_path : str, optional
        Path for caching training data.
    show_progress : bool
        Show progress bars.
    Nx, Ny : int
        Forward model grid resolution.
    t_max_days : float
        Max time for forward model grid.

    Returns
    -------
    SBIPosterior
        Trained posterior object.
    """
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn

    model = canonical_model_name(model, warn_legacy=False)
    z_val = float(z or 0.0)
    train_device = _resolve_sbi_device(device)

    # ---- Build prior ----
    priors_lin, priors_log10 = _split_prior_specs(priors)
    names_all, bounds_all = build_bounds(model, priors=priors_lin, include_t_shift=True)
    bounds_all, log_set_all = _apply_log10_priors(names_all, bounds_all, priors_log10)
    names_samp, bounds_samp, fixed_dict = _split_sampling(names_all, bounds_all, fixed=fixed)
    log_flags_samp = [n in log_set_all for n in names_samp]
    mixed_prior = MixedBoundsPrior(
        bounds=bounds_samp, param_names=names_samp, log_flags=log_flags_samp
    )
    tf_prior = TransFitPrior(mixed_prior)
    tf_prior_train = TransFitPrior(mixed_prior).to(train_device)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ---- Build cadence templates ----
    if cadence_templates is None:
        cadence_templates = _generate_random_cadences(
            n_simulations=n_simulations,
            n_epochs_range=n_epochs_range,
            bands_pool=bands_pool,
            t_range=t_range,
            mode=mode,
            rng=rng,
        )

    # ---- Determine band vocabulary ----
    band_vocabulary = None
    if mode == "multiband" and bands_pool is not None:
        band_vocabulary = sorted(set(bands_pool))
    elif mode == "multiband":
        all_bands = set()
        for tmpl in cadence_templates:
            if "band" in tmpl and tmpl["band"] is not None:
                all_bands.update(np.asarray(tmpl["band"], object).tolist())
        band_vocabulary = sorted(all_bands) if all_bands else None

    # ---- Generate training data with variable cadences ----
    all_theta = []
    all_x_raw = []
    all_t_days = []
    all_bands = [] if mode == "multiband" else None

    # Distribute simulations across cadence templates
    n_templates = len(cadence_templates)
    sims_per_template = max(1, n_simulations // n_templates)
    n_extra = n_simulations - sims_per_template * n_templates

    for t_idx, tmpl in enumerate(cadence_templates):
        n_sims = sims_per_template + (1 if t_idx < n_extra else 0)
        if n_sims <= 0:
            continue

        t_days_tmpl = np.asarray(tmpl["t_days"], float)
        band_tmpl = tmpl.get("band")
        if band_tmpl is not None:
            band_tmpl = np.asarray(band_tmpl, object)

        # Create simulator for this cadence
        if mode == "bolometric":
            sim = make_bolometric_simulator(
                model=model, z=z_val, t_days=t_days_tmpl,
                noise_sigma=noise_sigma, noise_model=noise_model,
                Nx=Nx, Ny=Ny, t_max_days=t_max_days,
                param_names=names_samp, names_all=names_all, fixed=fixed_dict,
            )
        else:
            sim = make_multiband_simulator(
                model=model, z=z_val,
                distance_modulus=distance_modulus, filters=filters,
                t_days=t_days_tmpl, band=band_tmpl,
                y_kind=y_kind, mag_system=mag_system,
                extinction=extinction,
                noise_sigma=noise_sigma, noise_model=noise_model,
                Nx=Nx, Ny=Ny, t_max_days=t_max_days,
                param_names=names_samp, names_all=names_all, fixed=fixed_dict,
            )

        # Generate data
        theta_batch, x_batch, _ = generate_training_data(
            simulator=sim,
            prior=tf_prior,
            n_simulations=n_sims,
            n_workers=n_workers if t_idx == 0 else 1,  # parallel only for first
            seed=seed + t_idx,
            cache_path=None,
            show_progress=show_progress and t_idx == 0,
        )
        if len(theta_batch) == 0:
            continue

        all_theta.append(theta_batch)
        all_x_raw.extend(np.asarray(x_batch[i], float) for i in range(len(x_batch)))
        all_t_days.extend([t_days_tmpl] * len(theta_batch))
        if all_bands is not None:
            all_bands.extend([band_tmpl] * len(theta_batch))

    if not all_theta:
        raise RuntimeError("No valid simulations produced. Check prior bounds and model parameters.")

    # Concatenate all training data and pad variable-length cadences globally.
    theta_train = torch.cat(all_theta, dim=0)
    x_train, _ = encode_batch(
        all_x_raw,
        t_days_list=all_t_days,
        band_list=all_bands,
        band_vocabulary=band_vocabulary,
        t_range=t_range,
    )

    # Filter NaN observations
    valid = torch.all(torch.isfinite(x_train.view(x_train.shape[0], -1)), dim=1)
    theta_train = theta_train[valid]
    x_train = x_train[valid]

    if len(theta_train) == 0:
        raise RuntimeError("No valid simulations produced. Check prior bounds and model parameters.")

    # ---- Build embedding net ----
    feature_dim = x_train.shape[2] if x_train.dim() == 3 else 1

    if embedding_net is None:
        embedding_net = SetSummaryNet(
            feature_dim=feature_dim,
            hidden_features=hidden_features,
            output_dim=hidden_features,
        )
    embedding_net = embedding_net.to(train_device)

    # ---- Train NPE ----
    density_estimator = posterior_nn(
        model="maf",
        embedding_net=embedding_net,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
    )

    inference = SNPE(
        prior=tf_prior_train,
        density_estimator=density_estimator,
        device=str(train_device),
        show_progress_bars=show_progress,
    )
    inference.append_simulations(theta_train, x_train)

    density_estimator = inference.train(
        training_batch_size=training_batch_size,
        max_num_epochs=max_num_epochs,
        learning_rate=learning_rate,
        show_train_summary=show_progress,
    )

    sbi_posterior = inference.build_posterior(density_estimator)

    posterior = SBIPosterior(
        model=model,
        param_names=names_samp,
        posterior=sbi_posterior,
        embedding_net=embedding_net,
        meta={
            "mode": mode,
            "z": z_val,
            "distance_modulus": distance_modulus,
            "filters": str(filters) if filters else None,
            "y_kind": y_kind,
            "mag_system": mag_system,
            "names_all": names_all,
            "bounds_samp": np.asarray(bounds_samp, float).tolist(),
            "fixed": fixed_dict,
            "n_simulations": n_simulations,
            "n_valid": len(theta_train),
            "Nx": Nx,
            "Ny": Ny,
            "t_max_days": t_max_days,
            "t_range": list(t_range),
            "x_event_shape": list(x_train.shape[1:]),
            "device": str(train_device),
        },
        band_vocabulary=band_vocabulary,
        t_range=t_range,
        mode=mode,
    )
    return posterior.to(train_device)


def infer_sbi(
    posterior: SBIPosterior,
    y_obs: np.ndarray,
    *,
    t_days: Optional[np.ndarray] = None,
    band: Optional[np.ndarray] = None,
    n_samples: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Quick inference helper: draw posterior samples and compute summary stats.

    Returns
    -------
    dict with keys "samples", "median", "map", "param_names"
    """
    samples = posterior.sample(
        n_samples, y_obs, t_days=t_days, band=band, seed=seed
    )
    median = posterior.median(y_obs, t_days=t_days, band=band, n=n_samples)
    map_est = posterior.map_estimate(
        y_obs, t_days=t_days, band=band, n_candidates=n_samples, seed=seed
    )
    return {
        "samples": samples,
        "median": median,
        "map": map_est,
        "param_names": posterior.param_names,
    }


def _generate_random_cadences(
    *,
    n_simulations: int,
    n_epochs_range: Tuple[int, int],
    bands_pool: Optional[List[str]],
    t_range: Tuple[float, float],
    mode: str,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Generate random observation cadence templates.

    Returns a list of cadence dicts, each with keys "t_days" and optionally "band".
    """
    n_templates = max(5, n_simulations // 100)
    cadences = []

    for _ in range(n_templates):
        n_obs = rng.integers(n_epochs_range[0], n_epochs_range[1] + 1)
        t_days = np.sort(rng.uniform(t_range[0], t_range[1], size=n_obs))

        tmpl: Dict[str, Any] = {"t_days": t_days}
        if mode == "multiband" and bands_pool:
            band = rng.choice(bands_pool, size=n_obs)
            tmpl["band"] = band

        cadences.append(tmpl)

    return cadences


def _resolve_sbi_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device}' was requested, but torch.cuda.is_available() is False. "
            "Check the installed PyTorch CUDA build and the NVIDIA driver."
        )
    return resolved


__all__ = [
    "train_sbi",
    "infer_sbi",
    "SBIPosterior",
    "TransFitPrior",
    "save_posterior",
    "load_posterior",
    "simulation_based_calibration",
    "posterior_predictive_check",
]
