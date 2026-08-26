# transfit/sbi/simulator.py
"""Simulator wrappers that turn TransFit forward models into theta -> x callables."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..api import (
    predict_bol,
    predict_multiband,
    _physical_constraints_lnprior,
    _param_values_from_sample,
    _assemble_model_params_from_values,
)
from ..model_registry import canonical_model_name


def make_bolometric_simulator(
    *,
    model: str,
    z: float = 0.0,
    t_days: np.ndarray,
    noise_sigma: Optional[float] = None,
    noise_model: Optional[Callable] = None,
    seed: Optional[int] = None,
    Nx: int = 20,
    Ny: int = 50,
    t_max_days: float = 150.0,
    param_names: List[str],
    names_all: List[str],
    fixed: Dict[str, float],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable that maps theta (tensor) -> simulated bolometric LC (tensor).

    Parameters
    ----------
    model : str
        Model name (e.g. "nickel").
    z : float
        Redshift.
    t_days : np.ndarray
        Observer-frame days at which to evaluate.
    noise_sigma : float, optional
        Homoscedastic Gaussian noise standard deviation (log10-space).
    noise_model : callable, optional
        Custom noise injection: y_clean -> y_noisy.
    Nx, Ny : int
        Grid resolution for the radiative diffusion solver.
    t_max_days : float
        Maximum observer-frame time for the solver grid.
    param_names : list[str]
        Names of free (sampled) parameters.
    names_all : list[str]
        All model parameter names (including t_shift).
    fixed : dict
        Fixed parameter values.
    """

    model = canonical_model_name(model, warn_legacy=False)
    t_days_np = np.asarray(t_days, float).copy()
    rng = np.random.default_rng(seed)

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        theta_np = np.asarray(theta.detach().cpu().numpy(), float)
        if theta_np.ndim == 1:
            theta_np = theta_np.reshape(1, -1)

        batch_size = theta_np.shape[0]
        outputs = np.full((batch_size, len(t_days_np)), np.nan, dtype=float)

        for i in range(batch_size):
            try:
                vals = _param_values_from_sample(
                    theta_np[i], param_names, fixed
                )
                lp_phys = _physical_constraints_lnprior(vals, model=model)
                if not np.isfinite(lp_phys):
                    continue

                model_params, t_shift = _assemble_model_params_from_values(
                    vals, names_all
                )
                t_eval = t_days_np + t_shift

                y = predict_bol(
                    model=model,
                    params=model_params,
                    z=z,
                    t_days=t_eval,
                    t_max_days=t_max_days,
                    interp_fill="nan",
                    solver_kwargs={"Nx": Nx, "Ny": Ny},
                )

                if np.any(~np.isfinite(y)):
                    continue

                # Convert to log10 for more uniform scale
                y_out = np.log10(np.clip(y, 1e-30, None))

                if noise_model is not None:
                    y_out = noise_model(y_out)
                elif noise_sigma is not None and noise_sigma > 0:
                    y_out = y_out + rng.normal(0.0, noise_sigma, size=len(y_out))

                outputs[i] = y_out
            except (ImportError, AttributeError, TypeError):
                raise
            except Exception:
                continue

        return torch.as_tensor(outputs, dtype=torch.float32)

    return simulator


def make_multiband_simulator(
    *,
    model: str,
    z: float = 0.0,
    distance_modulus: Optional[float] = None,
    filters: Optional[Dict] = None,
    t_days: np.ndarray,
    band: np.ndarray,
    y_kind: str = "mag",
    mag_system: str = "ab",
    extinction=None,
    noise_sigma: Optional[float] = None,
    noise_model: Optional[Callable] = None,
    seed: Optional[int] = None,
    Nx: int = 20,
    Ny: int = 50,
    t_max_days: float = 150.0,
    param_names: List[str],
    names_all: List[str],
    fixed: Dict[str, float],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable that maps theta (tensor) -> simulated multiband observations (tensor).

    Parameters
    ----------
    model : str
        Model name.
    z : float
        Redshift.
    distance_modulus : float, optional
        Distance modulus (mag).
    filters : dict, optional
        Filter map.
    t_days : np.ndarray
        Observer-frame observation times.
    band : np.ndarray
        Band labels for each observation.
    y_kind : str
        "mag" or "flux".
    mag_system : str
        "ab" or "vega".
    extinction : optional
        Extinction specification.
    noise_sigma : float, optional
        Homoscedastic Gaussian noise std.
    noise_model : callable, optional
        Custom noise injection.
    Nx, Ny : int
        Grid resolution.
    t_max_days : float
        Max solver time.
    param_names : list[str]
        Free parameter names.
    names_all : list[str]
        All parameter names.
    fixed : dict
        Fixed parameter values.
    """

    model = canonical_model_name(model, warn_legacy=False)
    t_days_np = np.asarray(t_days, float).copy()
    band_np = np.asarray(band, object).copy()
    n_obs = len(t_days_np)
    rng = np.random.default_rng(seed)

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        theta_np = np.asarray(theta.detach().cpu().numpy(), float)
        if theta_np.ndim == 1:
            theta_np = theta_np.reshape(1, -1)

        batch_size = theta_np.shape[0]
        outputs = np.full((batch_size, n_obs), np.nan, dtype=float)

        for i in range(batch_size):
            try:
                vals = _param_values_from_sample(
                    theta_np[i], param_names, fixed
                )
                lp_phys = _physical_constraints_lnprior(vals, model=model)
                if not np.isfinite(lp_phys):
                    continue

                model_params, t_shift = _assemble_model_params_from_values(
                    vals, names_all
                )
                t_eval = t_days_np + t_shift

                y = predict_multiband(
                    model=model,
                    params=model_params,
                    z=z,
                    distance_modulus=distance_modulus,
                    filters=filters,
                    t_days=t_eval,
                    band=band_np,
                    y_kind=y_kind,
                    mag_system=mag_system,
                    extinction=extinction,
                    t_max_days=t_max_days,
                    interp_fill="nan",
                    solver_kwargs={"Nx": Nx, "Ny": Ny},
                )

                if np.any(~np.isfinite(y)):
                    continue

                if noise_model is not None:
                    y = noise_model(y)
                elif noise_sigma is not None and noise_sigma > 0:
                    y = y + rng.normal(0.0, noise_sigma, size=len(y))

                outputs[i] = y
            except (ImportError, AttributeError, TypeError):
                raise
            except Exception:
                continue

        return torch.as_tensor(outputs, dtype=torch.float32)

    return simulator
