from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr

_MAG_TO_FRAC_FLUX = 0.4 * np.log(10.0)
DEFAULT_UPPER_LIMIT_NSIGMA = 5.0


def gaussian_lnlike(y_obs: np.ndarray, y_model: np.ndarray, y_err: np.ndarray) -> float:
    y_obs = np.asarray(y_obs, float)
    y_model = np.asarray(y_model, float)
    y_err = np.asarray(y_err, float)

    good = np.isfinite(y_obs) & np.isfinite(y_model) & np.isfinite(y_err) & (y_err > 0)
    if not np.any(good):
        return -np.inf

    r = (y_obs[good] - y_model[good]) / y_err[good]
    return -0.5 * float(np.sum(r * r))


def gaussian_lnlike_flux(y_obs: np.ndarray, y_model: np.ndarray, y_err: np.ndarray) -> float:
    """
    Gaussian likelihood in flux-density space.
    """
    return gaussian_lnlike(y_obs, y_model, y_err)


def gaussian_lnlike_mag(y_obs: np.ndarray, y_model: np.ndarray, y_err: np.ndarray) -> float:
    """
    Gaussian likelihood in magnitude space.
    """
    return gaussian_lnlike(y_obs, y_model, y_err)


def gaussian_lnlike_for_observation(
    *,
    y_kind: str,
    y_obs: np.ndarray,
    y_model: np.ndarray,
    y_err: np.ndarray,
) -> float:
    kind = str(y_kind).strip().lower()
    if kind == "flux":
        return gaussian_lnlike_flux(y_obs, y_model, y_err)
    if kind == "mag":
        return gaussian_lnlike_mag(y_obs, y_model, y_err)
    raise ValueError("y_kind must be 'mag' or 'flux'.")


def gaussian_lnlike_with_nuisance(
    *,
    y_kind: str,
    y_obs: np.ndarray,
    y_model: np.ndarray,
    y_err: np.ndarray,
    nuisance_params: dict[str, float] | None = None,
) -> float:
    """
    Gaussian likelihood with optional likelihood-only nuisance parameters.

    When no nuisance parameters are active, this preserves the historical
    TransFit likelihood exactly. When ``sigma_int`` is provided, the Gaussian
    normalization term is included because the scatter can be sampled.
    """
    sigma_int = dict(nuisance_params or {}).get("sigma_int")
    if sigma_int is None:
        return gaussian_lnlike_for_observation(
            y_kind=y_kind,
            y_obs=y_obs,
            y_model=y_model,
            y_err=y_err,
        )

    sigma_int = float(sigma_int)
    if not np.isfinite(sigma_int) or sigma_int < 0.0:
        return -np.inf

    y_obs = np.asarray(y_obs, float)
    y_model = np.asarray(y_model, float)
    y_err = np.asarray(y_err, float)
    kind = str(y_kind).strip().lower()

    good = np.isfinite(y_obs) & np.isfinite(y_model) & np.isfinite(y_err) & (y_err > 0)
    if not np.any(good):
        return -np.inf

    if kind == "mag":
        extra = np.full(np.sum(good), sigma_int, dtype=float)
    elif kind == "flux":
        extra = _MAG_TO_FRAC_FLUX * sigma_int * np.abs(y_obs[good])
    else:
        raise ValueError("y_kind must be 'mag' or 'flux'.")

    var = y_err[good] * y_err[good] + extra * extra
    if np.any(~np.isfinite(var)) or np.any(var <= 0.0):
        return -np.inf

    resid = y_obs[good] - y_model[good]
    return -0.5 * float(np.sum((resid * resid) / var + np.log(2.0 * np.pi * var)))


def upper_limit_gaussian_cdf_lnlike(
    *,
    y_kind: str,
    y_limit: np.ndarray,
    y_model: np.ndarray,
    y_err: np.ndarray,
    upper_limit_nsigma: np.ndarray | None = None,
    default_nsigma: float = DEFAULT_UPPER_LIMIT_NSIGMA,
) -> float:
    """One-sided Gaussian-CDF likelihood for non-detection limits.

    ``y_limit`` is the reported flux or magnitude limit. A finite ``y_err``
    supplies the local one-sigma noise. Otherwise ``upper_limit_nsigma`` gives
    the reported detection significance. When both are missing, the limit is
    interpreted as a ``default_nsigma`` detection threshold.

    Magnitude limits are evaluated through the model-to-limit flux ratio, so
    the result does not depend on the magnitude zero point.
    """
    y_limit = np.asarray(y_limit, float).reshape(-1)
    y_model = np.asarray(y_model, float).reshape(-1)
    y_err = np.asarray(y_err, float).reshape(-1)

    if upper_limit_nsigma is None:
        nsigma = np.full(y_limit.size, np.nan, dtype=float)
    else:
        nsigma_in = np.asarray(upper_limit_nsigma, float)
        nsigma = (
            np.full(y_limit.size, float(nsigma_in), dtype=float)
            if nsigma_in.ndim == 0
            else nsigma_in.reshape(-1)
        )

    if not (y_limit.size == y_model.size == y_err.size == nsigma.size):
        raise ValueError(
            "y_limit/y_model/y_err/upper_limit_nsigma must have the same length."
        )
    if y_limit.size == 0:
        return 0.0

    default_nsigma = float(default_nsigma)
    if not np.isfinite(default_nsigma) or default_nsigma <= 0.0:
        raise ValueError("default_nsigma must be finite and > 0.")
    if np.any(~np.isfinite(y_limit)) or np.any(~np.isfinite(y_model)):
        return -np.inf
    if np.any(np.isinf(y_err)) or np.any(np.isfinite(y_err) & (y_err <= 0.0)):
        return -np.inf
    if np.any(np.isinf(nsigma)) or np.any(np.isfinite(nsigma) & (nsigma <= 0.0)):
        return -np.inf

    kind = str(y_kind).strip().lower()
    has_error = np.isfinite(y_err)
    has_nsigma = np.isfinite(nsigma)
    if np.any(has_error & has_nsigma):
        return -np.inf

    effective_nsigma = np.full(y_limit.shape, default_nsigma, dtype=float)
    effective_nsigma[has_nsigma] = nsigma[has_nsigma]
    sigma_ratio = 1.0 / effective_nsigma

    if kind == "flux":
        if np.any(y_limit <= 0.0):
            return -np.inf
        model_to_limit = y_model / y_limit
        sigma_ratio[has_error] = y_err[has_error] / y_limit[has_error]
    elif kind == "mag":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            model_to_limit = np.power(10.0, -0.4 * (y_model - y_limit))
        # A magnitude error is converted locally to fractional flux error.
        sigma_ratio[has_error] = _MAG_TO_FRAC_FLUX * y_err[has_error]
    else:
        raise ValueError("y_kind must be 'mag' or 'flux'.")

    if np.any(~np.isfinite(sigma_ratio)) or np.any(sigma_ratio <= 0.0):
        return -np.inf

    z = (1.0 - model_to_limit) / sigma_ratio
    return float(np.sum(log_ndtr(z)))
