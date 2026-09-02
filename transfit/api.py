# transfit/api.py
from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Literal, Any, Tuple

import numpy as np

from .data import BolometricData, MultiBandData
from .exceptions import NonPhysicalModelError
from .modules.extinction import (
    ExtinctionSpec,
    apply_extinction_to_fnu_grid,
    normalize_extinction,
    validate_extinction_spec,
)
from .modules.filters import FilterProfile, normalize_filters, validate_filter_map
from .modules.fnu import evaluate_multiband_model_fnu
from .modules.interp import interp_fit
from .modules.labels import normalize_band_label
from .modules.likelihood import (
    DEFAULT_UPPER_LIMIT_NSIGMA,
    gaussian_lnlike_with_nuisance,
    upper_limit_gaussian_cdf_lnlike,
)
from .modules.photometry import (
    evaluate_multiband_observer_output,
    fnu_grid_to_observation_output,
    validate_observation_mode,
)
from .modules.sed import BlackbodySED, sed_to_dict
from .model_registry import canonical_model_name, forward_param_defaults
from .samplers import FitResult, run_emcee, run_zeus, run_dynesty
from .priors import MixedBoundsPrior, build_bounds
from .priors.nuisance import LIKELIHOOD_NUISANCE_PARAM_SPECS
from transfit.constants import DAY, MPC, PC, PI, SIGMA_SB


# -------------------------
# Internal forward metadata
# -------------------------

_BOL_INTERNAL_T_FLOOR = 1000.0
_FIT_T_MAX_DAYS_DEFAULT = 150.0
_FIT_T_MAX_DAYS_PADDING = 20.0
_CSM_INTERNAL_R_CSM_IN = 100.0


class _NonPhysicalModelOutput(NonPhysicalModelError):
    """Raised when a solved model state is not usable as a physical prediction."""


def _cosmo_luminosity_distance_cm(z: float) -> float:
    from astropy.cosmology import Planck15 as cosmo
    import astropy.units as u

    return cosmo.luminosity_distance(float(z)).to(u.cm).value


def _distance_cm_from_modulus(distance_modulus: float) -> float:
    return (10.0 ** ((float(distance_modulus) + 5.0) / 5.0)) * PC


def _distance_modulus_from_cm(distance_cm: float) -> float:
    d_pc = float(distance_cm) / PC
    return 5.0 * np.log10(d_pc) - 5.0


@dataclass(frozen=True)
class _Distance:
    z: Optional[float] = None
    DL_cm: Optional[float] = None
    source: Optional[str] = None

    def get_z(self) -> float:
        return float(self.z or 0.0)

    def get_DL_cm(self) -> float:
        if self.DL_cm is not None:
            dl = float(self.DL_cm)
            if self.z is not None:
                dl_cosmo = _cosmo_luminosity_distance_cm(float(self.z))
                if np.isfinite(dl_cosmo) and dl_cosmo > 0.0:
                    frac = abs(dl - dl_cosmo) / dl_cosmo
                    if frac > 0.05:
                        warnings.warn(
                            "Using a user-supplied explicit distance that differs from the Planck15 luminosity distance "
                            "implied by z by more than 5%. z is still used for time/frequency redshift terms.",
                            stacklevel=2,
                        )
            return dl
        if self.z is None:
            raise ValueError("Distance needs either an explicit distance or z.")
        return _cosmo_luminosity_distance_cm(float(self.z))


def _distance_from_public_inputs(
    *,
    z: Optional[float],
    distance_modulus: Optional[float],
    require_distance: bool,
) -> _Distance:
    z_norm = None if z is None else float(z)
    if z_norm is not None and not np.isfinite(z_norm):
        raise ValueError("z must be finite when provided.")
    if z_norm is not None and z_norm < 0.0:
        raise ValueError("z must be non-negative when provided.")

    dl_norm = None
    source = None
    if distance_modulus is not None:
        dl_norm = _distance_cm_from_modulus(float(distance_modulus))
        source = "distance_modulus"

    if dl_norm is not None and (not np.isfinite(dl_norm) or dl_norm <= 0.0):
        raise ValueError("Explicit distance must be positive and finite.")
    if require_distance and z_norm is None and dl_norm is None:
        raise ValueError(
            "Provide `z` or an explicit distance via `distance_modulus`."
        )
    if source is None and z_norm is not None:
        source = "from_z"
    return _Distance(z=z_norm, DL_cm=dl_norm, source=source)


@dataclass(frozen=True)
class _Context:
    """
    Internal context for forward-model evaluation.

    - Multi-band prediction: filters is required.
    """
    distance: _Distance
    filters: Optional[Dict[str, FilterProfile]] = None
    y_kind: Literal["mag", "flux"] = "mag"
    mag_system: Literal["ab", "vega"] = "ab"
    extinction: Optional[ExtinctionSpec] = None


def _effective_mag_system(y_kind: str, mag_system: str) -> str:
    return str(mag_system).strip().lower() if str(y_kind).strip().lower() == "mag" else "ab"


def _context_from_fit_inputs(
    *,
    z: Optional[float],
    distance_modulus: Optional[float],
    filters: Optional[Dict[str, Any]],
    y_kind: Literal["mag", "flux"],
    mag_system: Literal["ab", "vega"],
    extinction: Optional[Dict[str, Any] | ExtinctionSpec],
    require_filters: bool,
) -> _Context:
    """
    Build the internal Context used by fitting.

    Public fitting APIs accept direct scalar inputs instead of exposing
    Context/Distance as required user-facing concepts.
    """
    if require_filters and filters is None:
        raise ValueError(
            "filters is required for multiband fitting."
        )
    y_kind_n, mag_system_n = validate_observation_mode(y_kind, mag_system)
    return _Context(
        distance=_distance_from_public_inputs(
            z=z,
            distance_modulus=distance_modulus,
            require_distance=require_filters,
        ),
        filters=None if filters is None else normalize_filters(
            filters,
            mag_system=_effective_mag_system(y_kind_n, mag_system_n),
        ),
        y_kind=y_kind_n,
        mag_system=mag_system_n,
        extinction=normalize_extinction(extinction),
    )


def _context_from_forward_inputs(
    *,
    z: Optional[float],
    distance_modulus: Optional[float],
    filters: Optional[Dict[str, Any]],
    y_kind: Literal["mag", "flux"],
    mag_system: Literal["ab", "vega"],
    extinction: Optional[Dict[str, Any] | ExtinctionSpec],
    require_filters: bool,
    require_distance: bool,
) -> _Context:
    """
    Build internal forward metadata for public prediction/lightcurve helpers.

    Public forward APIs accept direct scalar inputs instead of requiring
    Context/Distance objects from users.
    """
    if require_filters and filters is None:
        raise ValueError("filters is required for multiband forward calculations.")
    y_kind_n, mag_system_n = validate_observation_mode(y_kind, mag_system)
    return _Context(
        distance=_distance_from_public_inputs(
            z=z,
            distance_modulus=distance_modulus,
            require_distance=require_distance,
        ),
        filters=None if filters is None else normalize_filters(
            filters,
            mag_system=_effective_mag_system(y_kind_n, mag_system_n),
        ),
        y_kind=y_kind_n,
        mag_system=mag_system_n,
        extinction=normalize_extinction(extinction),
    )



# -------------------------
# Light curve containers
# -------------------------

@dataclass(frozen=True)
class BolometricLC:
    t_days: np.ndarray
    Lbol: np.ndarray
    Teff: np.ndarray
    Rph: np.ndarray
    Lphotospheric: np.ndarray
    Ldirect: np.ndarray
    photosphere_valid: np.ndarray


@dataclass(frozen=True)
class MultiBandLC:
    t_days: np.ndarray
    bands: List[str]
    y: Dict[str, np.ndarray]                  # band -> mag/flux array


# -------------------------
# Small utilities
# -------------------------

def _as_1d_float(x, name: str) -> np.ndarray:
    arr = np.asarray(x, float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def _check_same_length(**arrays: np.ndarray) -> None:
    lens = {k: int(np.asarray(v).size) for k, v in arrays.items()}
    if len(set(lens.values())) != 1:
        raise ValueError(f"Input arrays must have the same length, got: {lens}")


def _apply_data_filter(data):
    """
    Apply container-level masking when available.

    This makes `mask` a first-class part of the public fitting API. Invalid
    unmasked values are validated later and must not be silently dropped.
    """
    if hasattr(data, "filtered"):
        return data.filtered()
    return data


def _validate_fit_time_and_errors(t_obs: np.ndarray, y_err: np.ndarray) -> None:
    if np.any(~np.isfinite(t_obs)):
        raise ValueError("data.t_days must be finite.")
    if np.any(~np.isfinite(y_err)) or np.any(y_err <= 0.0):
        raise ValueError("data.yerr must be finite and > 0.")


def _validate_multiband_fit_errors(
    y_err: np.ndarray,
    is_upper_limit: np.ndarray,
    upper_limit_nsigma: Optional[np.ndarray] = None,
) -> None:
    y_err = np.asarray(y_err, float).reshape(-1)
    upper = np.asarray(is_upper_limit, bool).reshape(-1)
    if y_err.size != upper.size:
        raise ValueError("data.yerr and data.is_upper_limit must have the same length.")

    if upper_limit_nsigma is None:
        nsigma = np.full(y_err.size, np.nan, dtype=float)
    else:
        nsigma = np.asarray(upper_limit_nsigma, float).reshape(-1)
        if nsigma.size != y_err.size:
            raise ValueError(
                "data.upper_limit_nsigma must have the same length as data.yerr."
            )

    detection_err = y_err[~upper]
    if np.any(~np.isfinite(detection_err)) or np.any(detection_err <= 0.0):
        raise ValueError("data.yerr must be finite and > 0 for detections.")

    upper_err = y_err[upper]
    if np.any(np.isinf(upper_err)) or np.any(
        np.isfinite(upper_err) & (upper_err <= 0.0)
    ):
        raise ValueError(
            "Upper-limit data.yerr must be finite and > 0 when provided; "
            "use np.nan when the one-sigma error is unavailable."
        )

    if np.any(np.isinf(nsigma)) or np.any(np.isfinite(nsigma) & (nsigma <= 0.0)):
        raise ValueError(
            "data.upper_limit_nsigma must be finite and > 0 when provided."
        )
    if np.any(np.isfinite(nsigma[~upper])):
        raise ValueError(
            "data.upper_limit_nsigma may only be set for upper-limit rows."
        )
    if np.any(np.isfinite(upper_err) & np.isfinite(nsigma[upper])):
        raise ValueError(
            "For each upper-limit row, provide either data.yerr or "
            "data.upper_limit_nsigma, not both."
        )


def _validate_multiband_fit_y(
    y_obs: np.ndarray,
    *,
    y_kind: str,
    is_upper_limit: Optional[np.ndarray] = None,
) -> None:
    kind = str(y_kind).strip().lower()
    if kind not in ("mag", "flux"):
        raise ValueError("y_kind must be 'mag' or 'flux'.")
    if np.any(~np.isfinite(y_obs)):
        raise ValueError(f"data.y must be finite when y_kind='{kind}'.")
    if is_upper_limit is None:
        return

    upper = np.asarray(is_upper_limit, bool).reshape(-1)
    if upper.size != np.asarray(y_obs).size:
        raise ValueError("data.y and data.is_upper_limit must have the same length.")
    if kind == "flux" and np.any(np.asarray(y_obs, float).reshape(-1)[upper] <= 0.0):
        raise ValueError("Upper-limit data.y must be positive when y_kind='flux'.")


def _validate_bolometric_fit_y(y_obs: np.ndarray) -> None:
    if np.any(~np.isfinite(y_obs)) or np.any(y_obs <= 0.0):
        raise ValueError("data.y must be positive and finite for bolometric fitting.")


# -------------------------
# Model registry
# -------------------------

_ENGINE_CACHE = {}

def _get_engine(model: str):
    m = canonical_model_name(model, warn_legacy=False)
    if m in _ENGINE_CACHE:
        return _ENGINE_CACHE[m]

    if m == "nickel":
        from .models.nickel import NickelModel
        eng = NickelModel()
        _ENGINE_CACHE[m] = eng
        return eng

    if m == "magnetar":
        from .models.magnetar import MagnetarModel
        eng = MagnetarModel()
        _ENGINE_CACHE[m] = eng
        return eng

    if m == "magnetar_ni":
        from .models.magnetar_ni import MagNiModel
        eng = MagNiModel()
        _ENGINE_CACHE[m] = eng
        return eng

    if m == "csm":
        from .models.csm import CSMModel
        eng = CSMModel()
        _ENGINE_CACHE[m] = eng
        return eng

    raise ValueError(f"Unknown model='{model}'")


def model_param_names(model: str, *, include_t_shift: bool = False) -> List[str]:
    model = canonical_model_name(model, warn_legacy=True)
    names, _ = build_bounds(model, include_t_shift=include_t_shift)
    return list(names)


def param_template(
    model: str,
    *,
    include_t_shift: bool = False,
    fill_value: Any = None,
) -> Dict[str, Any]:
    model = canonical_model_name(model, warn_legacy=True)
    return {name: fill_value for name in model_param_names(model, include_t_shift=include_t_shift)}


def _model_vector_from_params(
    model: str,
    params: Dict[str, Any],
    *,
    allow_missing_tfloor: bool,
):
    if not isinstance(params, dict):
        raise TypeError("params must be a dict mapping parameter names to values.")

    values = dict(params)
    # Allow direct reuse of res.best_params / res.median_params in forward helpers.
    values.pop("t_shift", None)

    model = canonical_model_name(model, warn_legacy=False)
    defaults = forward_param_defaults(model)
    names = model_param_names(model, include_t_shift=False)
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise KeyError(f"Unknown parameter(s) for model='{model}': {unknown}. Allowed: {names}")

    missing = [
        n for n in names
        if n not in values and n not in defaults and not (allow_missing_tfloor and n == "T_floor")
    ]
    if missing:
        raise KeyError(f"Missing parameter(s) for model='{model}': {missing}. Required: {names}")

    model_vector = []
    for n in names:
        if n not in values and n in defaults:
            model_vector.append(float(defaults[n]))
        elif n == "T_floor" and n not in values and allow_missing_tfloor:
            model_vector.append(_BOL_INTERNAL_T_FLOOR)
        else:
            model_vector.append(float(values[n]))
    return tuple(model_vector)


_NONNEGATIVE_PARAMS = {"E_Th_in", "M_ni", "delta", "s", "t_shift"}
_POSITIVE_PARAMS = {
    "M_csm",
    "M_ej",
    "v_ej",
    "E_sn",
    "n",
    "R_0",
    "R_csm_in",
    "R_csm_out",
    "kappa",
    "kappa_gamma",
    "T_floor",
    "P_ms",
    "B14",
}
_UNIT_INTERVAL_PARAMS = {"eps_sh", "f_ni", "f_mag"}


def _model_values_from_vector(model: str, model_vector) -> Dict[str, float]:
    names = model_param_names(model, include_t_shift=False)
    values_t = tuple(model_vector)
    if len(values_t) != len(names):
        raise ValueError(
            f"Model parameter vector for model='{model}' must have length {len(names)}."
        )
    return {str(n): float(v) for n, v in zip(names, values_t)}


def _physical_constraint_reason(
    vals: Dict[str, float],
    *,
    model: Optional[str] = None,
    enforce_ni_mixing_constraint: bool = True,
) -> Optional[str]:
    """
    Return a short reason when a parameter set violates model-independent
    physical constraints, otherwise None.
    """
    values = {str(k): float(v) for k, v in dict(vals or {}).items()}

    for name, value in values.items():
        if not np.isfinite(value):
            return f"{name} must be finite."

    for name in sorted(_POSITIVE_PARAMS & set(values)):
        if not (values[name] > 0.0):
            return f"{name} must be > 0."

    for name in sorted(_NONNEGATIVE_PARAMS & set(values)):
        if not (values[name] >= 0.0):
            return f"{name} must be >= 0."

    for name in sorted(_UNIT_INTERVAL_PARAMS & set(values)):
        if not (0.0 <= values[name] <= 1.0):
            return f"{name} must be in [0, 1]."

    if "s" in values and not (values["s"] < 3.0):
        return "s must be < 3."

    if "n" in values:
        if not (values["n"] > 5.0):
            return "n must be > 5."
        if "s" in values and not (values["n"] > values["s"]):
            return "n must be > s."

    if "delta" in values and not (values["delta"] < 3.0):
        return "delta must be < 3."

    if "M_ni" in values and "M_ej" in values:
        if values["M_ni"] > values["M_ej"]:
            return "M_ni must be <= M_ej."

    if (
        enforce_ni_mixing_constraint
        and
        model is not None
        and canonical_model_name(model, warn_legacy=False) == "nickel"
        and {"M_ej", "M_ni", "f_ni"}.issubset(values)
    ):
        mixed_mass = values["f_ni"] * values["M_ej"]
        tolerance = 1.0e-12 * max(abs(values["M_ni"]), abs(mixed_mass), 1.0)
        if values["M_ni"] > mixed_mass + tolerance:
            return (
                "Nickel mass-coordinate mixing requires "
                "M_ni <= f_ni*M_ej (equivalently f_ni >= M_ni/M_ej)."
            )

    if "R_csm_in" in values and "R_csm_out" in values:
        if not (values["R_csm_out"] > values["R_csm_in"]):
            return "R_csm_out must be > R_csm_in."
    elif "R_csm_out" in values:
        if not (values["R_csm_out"] > _CSM_INTERNAL_R_CSM_IN):
            return "R_csm_out must be > R_csm_in."

    return None


def _validate_physical_values(
    vals: Dict[str, float],
    *,
    model: Optional[str] = None,
    enforce_ni_mixing_constraint: bool = True,
) -> None:
    reason = _physical_constraint_reason(
        vals,
        model=model,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )
    if reason is not None:
        raise ValueError(f"Physical parameter constraints are invalid: {reason}")


def _validate_physical_model_vector(
    model: str,
    model_vector,
    *,
    enforce_ni_mixing_constraint: bool = True,
) -> None:
    _validate_physical_values(
        _model_values_from_vector(model, model_vector),
        model=model,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )


def _resolve_forward_params(
    model: str,
    *,
    params: Optional[Dict[str, Any]],
    allow_missing_tfloor: bool,
    enforce_ni_mixing_constraint: bool = True,
):
    if params is not None:
        model_vector = _model_vector_from_params(model, params, allow_missing_tfloor=allow_missing_tfloor)
        _validate_physical_model_vector(
            model,
            model_vector,
            enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
        )
        return model_vector
    raise ValueError("Provide `params` for forward-model evaluation.")


def _observer_days_to_rest_days(t_days_obs: float, z: float) -> float:
    """
    Public APIs accept observer-frame day scales.
    Internal engines solve in rest-frame / physical time.
    """
    return float(t_days_obs) / (1.0 + float(z))


def _solve_state(
    engine,
    model_vector,
    *,
    Nx: int,
    Ny: int,
    t_max_days_obs: float,
    z: float,
    **solver_options,
):
    from .models.csm import CSMModel
    from .models.nickel import NickelModel

    enforce_ni_mixing_constraint = bool(
        solver_options.pop("enforce_ni_mixing_constraint", True)
    )
    t_max_days_rest = _observer_days_to_rest_days(t_max_days_obs, z)
    if hasattr(engine, "calculate_transport"):
        transport_options = dict(solver_options)
        if isinstance(engine, NickelModel):
            transport_options["enforce_ni_mixing_constraint"] = bool(
                enforce_ni_mixing_constraint
            )
        transport = engine.calculate_transport(
            model_vector,
            Nx=Nx,
            Ny=Ny,
            t_max_days=t_max_days_rest,
            **transport_options,
        )
        return (
            transport.t_s,
            transport.Lbol,
            transport.Tph,
            transport.Rph,
            transport,
        )
    engine_vector = tuple(model_vector)
    if isinstance(engine, CSMModel):
        values = _model_values_from_vector("csm", model_vector)
        engine_vector = (
            values["M_ej"],
            values["E_sn"],
            values["M_csm"],
            values["R_csm_out"],
            _CSM_INTERNAL_R_CSM_IN,
            values["kappa"],
            values["s"],
            values["n"],
            values["delta"],
            values["eps_sh"],
            values["T_floor"],
        )
    t_s, Lbol, Teff, Rph = engine.calculate_light_curve(
        engine_vector,
        Nx=Nx,
        Ny=Ny,
        t_max_days=t_max_days_rest,
        **solver_options,
    )
    return t_s, Lbol, Teff, Rph, None


_BASE_SOLVER_KEYS = {"Nx", "Ny"}
_NICKEL_SOLVER_KEYS = {"density_profile"}
_CSM_SOLVER_KEYS = {"photosphere_mode", "reverse_shock"}
_DEFAULT_IA_R0_RSUN = 0.01


def _normalize_density_profile(value: Any) -> str:
    profile = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "uniform": "uniform",
        "bpl": "broken_power_law",
        "broken_power_law": "broken_power_law",
        "exp": "exponential",
        "exponential": "exponential",
        "ia": "exponential",
    }
    if profile not in aliases:
        raise ValueError(
            "solver_kwargs['density_profile'] must be one of "
            "'uniform', 'bpl'/'broken_power_law', or "
            "'exp'/'exponential'/'ia'."
        )
    return aliases[profile]


def _resolve_solver_kwargs(
    solver_kwargs: Optional[Dict[str, Any]],
    *,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    opts = dict(solver_kwargs or {})
    model_canonical = None
    if model is not None:
        model_canonical = canonical_model_name(model, warn_legacy=False)

    allowed = set(_BASE_SOLVER_KEYS)
    if model_canonical == "nickel":
        allowed.update(_NICKEL_SOLVER_KEYS)
    elif model_canonical == "csm":
        allowed.update(_CSM_SOLVER_KEYS)

    unknown = sorted(set(opts) - allowed)
    if unknown:
        raise KeyError(
            f"Unknown solver_kwargs key(s) for model={model_canonical!r}: {unknown}. "
            f"Allowed: {sorted(allowed)}"
        )

    out: Dict[str, Any] = {
        "Nx": int(opts.get("Nx", 100)),
        "Ny": int(opts.get("Ny", 1000)),
    }
    for name in ("Nx", "Ny"):
        value = out[name]
        if value <= 0:
            raise ValueError(f"solver_kwargs['{name}'] must be a positive integer.")

    if model_canonical == "nickel":
        if "density_profile" in opts:
            out["density_profile"] = _normalize_density_profile(opts["density_profile"])
    elif model_canonical == "csm":
        if "photosphere_mode" in opts:
            photosphere_mode = str(opts["photosphere_mode"]).strip().lower()
            if photosphere_mode not in {"tau", "outer"}:
                raise ValueError(
                    "solver_kwargs['photosphere_mode'] must be 'tau' or 'outer'."
                )
            out["photosphere_mode"] = photosphere_mode
        if "reverse_shock" in opts:
            reverse_shock = opts["reverse_shock"]
            if not isinstance(reverse_shock, (bool, np.bool_)):
                raise TypeError(
                    "solver_kwargs['reverse_shock'] must be a boolean."
                )
            out["reverse_shock"] = bool(reverse_shock)

    return out


def _require_positive_finite(name: str, arr: np.ndarray) -> None:
    values = np.asarray(arr, float)
    if values.size == 0:
        raise _NonPhysicalModelOutput(f"Model produced an empty {name} grid.")
    bad = ~np.isfinite(values) | (values <= 0.0)
    if np.any(bad):
        first = float(values.reshape(-1)[int(np.where(bad.reshape(-1))[0][0])])
        raise _NonPhysicalModelOutput(
            f"Model produced non-positive or non-finite {name}; first invalid value is {first!r}."
        )


def _validate_solved_state(Lbol, Teff, Rph, photosphere_valid=None) -> None:
    _require_positive_finite("Lbol", Lbol)
    if photosphere_valid is None:
        _require_positive_finite("Teff", Teff)
        _require_positive_finite("Rph", Rph)
        return

    valid = np.asarray(photosphere_valid, bool)
    temperature = np.asarray(Teff, float)
    radius = np.asarray(Rph, float)
    if valid.shape != temperature.shape or valid.shape != radius.shape:
        raise _NonPhysicalModelOutput(
            "photosphere_valid, Teff, and Rph must have the same shape."
        )
    if np.any(~np.isfinite(temperature[valid])) or np.any(temperature[valid] < 0.0):
        raise _NonPhysicalModelOutput(
            "Model produced invalid Teff on a valid photosphere."
        )
    if np.any(~np.isfinite(radius[valid])) or np.any(radius[valid] <= 0.0):
        raise _NonPhysicalModelOutput(
            "Model produced invalid Rph on a valid photosphere."
        )
    if np.any(~np.isnan(temperature[~valid])) or np.any(~np.isnan(radius[~valid])):
        raise _NonPhysicalModelOutput(
            "Invalid photosphere epochs must use NaN Teff and Rph."
        )


def _nickel_two_component_blackbody(transport, T_floor: float):
    """Return diluted photospheric and optically thin blackbody components.

    The physical photosphere uses ``Lphotospheric`` and ``Rph``.  When its
    effective temperature falls below ``T_floor``, the color temperature is
    fixed at the floor and the blackbody area is diluted so its frequency
    integral remains exactly ``Lphotospheric``.  ``Ldirect`` is normalized as
    a separate floor-temperature continuum.  The two components therefore
    integrate to ``Lbol`` without assigning optically thin power to ``Rph``.
    """
    floor = float(T_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("T_floor must be finite and > 0 for photospheric emission.")
    luminosity = np.asarray(transport.Lbol, float)
    photospheric_luminosity = np.asarray(transport.Lphotospheric, float)
    direct_luminosity = np.asarray(transport.Ldirect, float)
    radius_physical = np.asarray(transport.Rph, float)
    photosphere_valid = np.asarray(transport.photosphere_valid, bool)
    if not (
        luminosity.shape
        == photospheric_luminosity.shape
        == direct_luminosity.shape
        == radius_physical.shape
        == photosphere_valid.shape
    ):
        raise _NonPhysicalModelOutput(
            "Nickel Lbol, Lphotospheric, Ldirect, Rph, and photosphere_valid "
            "must have the same shape."
        )
    if (
        np.any(~np.isfinite(photospheric_luminosity))
        or np.any(photospheric_luminosity < 0.0)
        or np.any(~np.isfinite(direct_luminosity))
        or np.any(direct_luminosity < 0.0)
    ):
        raise _NonPhysicalModelOutput(
            "Nickel Lphotospheric and Ldirect must be finite and non-negative."
        )
    if not np.allclose(
        luminosity,
        photospheric_luminosity + direct_luminosity,
        rtol=5.0e-12,
        atol=0.0,
    ):
        raise _NonPhysicalModelOutput(
            "Nickel transport must satisfy Lbol=Lphotospheric+Ldirect."
        )

    photo_active = photosphere_valid & (photospheric_luminosity > 0.0)
    invalid_photo_radius = ~np.isfinite(radius_physical) | (radius_physical <= 0.0)
    if np.any(photo_active & invalid_photo_radius):
        raise _NonPhysicalModelOutput(
            "A luminous Nickel photospheric component requires finite positive Rph."
        )

    temperature_photo = np.full(luminosity.shape, floor, dtype=float)
    radius_photo = np.zeros(luminosity.shape, dtype=float)
    if np.any(photo_active):
        temperature_effective = (
            photospheric_luminosity[photo_active]
            / (4.0 * PI * SIGMA_SB * radius_physical[photo_active] ** 2)
        ) ** 0.25
        temperature_photo[photo_active] = np.maximum(temperature_effective, floor)
        # This effective area is Rph**2 times the usual dilution factor
        # (T_eff/T_color)**4.  Above the floor it is exactly the physical Rph.
        radius_photo[photo_active] = np.sqrt(
            photospheric_luminosity[photo_active]
            / (4.0 * PI * SIGMA_SB * temperature_photo[photo_active] ** 4)
        )

    temperature_direct = np.full(luminosity.shape, floor, dtype=float)
    radius_direct = np.sqrt(
        direct_luminosity / (4.0 * PI * SIGMA_SB * floor**4)
    )
    return temperature_photo, radius_photo, temperature_direct, radius_direct


def _evaluate_nickel_two_component_multiband(
    *,
    transport,
    T_floor: float,
    sed,
    filter_map,
    bands,
    DL_cm: float,
    z: float,
    y_kind: str,
    mag_system: str,
    extinction,
):
    """Add Nickel photospheric and direct continua in F_nu space."""
    if type(sed) is not BlackbodySED:
        raise ValueError(
            "BPL/Ia Nickel two-component emission currently requires the "
            "standard BlackbodySED."
        )
    (
        temperature_photo,
        radius_photo,
        temperature_direct,
        radius_direct,
    ) = _nickel_two_component_blackbody(transport, T_floor)

    photo_fnu = evaluate_multiband_model_fnu(
        sed=sed,
        filter_map=filter_map,
        bands=bands,
        Teff_K=temperature_photo,
        R_cm=radius_photo,
        DL_cm=DL_cm,
        z=z,
    )
    direct_fnu = evaluate_multiband_model_fnu(
        sed=sed,
        filter_map=filter_map,
        bands=bands,
        Teff_K=temperature_direct,
        R_cm=radius_direct,
        DL_cm=DL_cm,
        z=z,
    )
    photo_fnu[:, radius_photo <= 0.0] = 0.0
    direct_fnu[:, radius_direct <= 0.0] = 0.0
    total_fnu = photo_fnu + direct_fnu
    total_fnu = apply_extinction_to_fnu_grid(
        total_fnu,
        filter_map=filter_map,
        bands=bands,
        extinction=extinction,
        z=z,
    )
    return fnu_grid_to_observation_output(
        total_fnu,
        filter_map=filter_map,
        bands=bands,
        y_kind=y_kind,
        mag_system=mag_system,
    )


def _evaluate_multiband_solved_state(
    *,
    model: str,
    model_vector,
    transport,
    Teff,
    Rph,
    sed,
    filter_map,
    bands,
    DL_cm: float,
    z: float,
    y_kind: str,
    mag_system: str,
    extinction,
):
    """Apply the density-selected SED mapping to a solved state."""
    if model == "nickel":
        if transport is None:
            raise _NonPhysicalModelOutput("Nickel transport state is unavailable.")
        if getattr(transport, "density_profile", None) == "uniform":
            # The historical Uniform solver already returns its homologous
            # blackbody with the pointwise temperature-floor mapping applied.
            temperature = np.asarray(transport.Tph, float)
            radius = np.asarray(transport.Rph, float)
        else:
            values = _model_values_from_vector(model, model_vector)
            return _evaluate_nickel_two_component_multiband(
                transport=transport,
                T_floor=values["T_floor"],
                sed=sed,
                filter_map=filter_map,
                bands=bands,
                DL_cm=DL_cm,
                z=z,
                y_kind=y_kind,
                mag_system=mag_system,
                extinction=extinction,
            )
    else:
        temperature = np.asarray(Teff, float)
        radius = np.asarray(Rph, float)
    return evaluate_multiband_observer_output(
        sed=sed,
        filter_map=filter_map,
        bands=bands,
        Teff_K=temperature,
        R_cm=radius,
        DL_cm=DL_cm,
        z=z,
        y_kind=y_kind,
        mag_system=mag_system,
        extinction=extinction,
    )


def _t_grid_days_from_ts(t_s: np.ndarray, z: float) -> np.ndarray:
    # Public forward outputs use observer-frame days.
    return (np.asarray(t_s, float) * (1.0 + z)) / DAY


# -------------------------
# Forward model
# -------------------------

def lightcurve_bol(
    *,
    model: str,
    params: Optional[Dict[str, Any]] = None,
    z: Optional[float] = None,
    t_max_days: float = 150.0,
    solver_kwargs: Optional[Dict[str, Any]] = None,
    enforce_ni_mixing_constraint: bool = True,
) -> BolometricLC:
    """
    Return a bolometric light curve on an observer-frame time grid.

    User-facing time arguments are observer-frame days.
    Internal model evolution is solved in rest-frame time.
    """
    model = canonical_model_name(model, warn_legacy=True)
    ctx = _context_from_forward_inputs(
        z=z,
        distance_modulus=None,
        filters=None,
        y_kind="mag",
        mag_system="ab",
        extinction=None,
        require_filters=False,
        require_distance=False,
    )
    engine = _get_engine(model)
    model_vector = _resolve_forward_params(
        model,
        params=params,
        allow_missing_tfloor=True,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )
    solver = _resolve_solver_kwargs(solver_kwargs, model=model)
    if not enforce_ni_mixing_constraint:
        solver["enforce_ni_mixing_constraint"] = False
    z = ctx.distance.get_z()
    t_s, Lbol, Teff, Rph, transport = _solve_state(
        engine,
        model_vector,
        **solver,
        t_max_days_obs=t_max_days,
        z=z,
    )
    photosphere_valid = (
        np.asarray(transport.photosphere_valid, bool)
        if transport is not None
        else np.ones(np.asarray(Lbol).shape, dtype=bool)
    )
    _validate_solved_state(Lbol, Teff, Rph, photosphere_valid if transport is not None else None)
    t_days = _t_grid_days_from_ts(t_s, z=z)
    if transport is not None:
        Lphotospheric = np.asarray(transport.Lphotospheric, float)
        Ldirect = np.asarray(transport.Ldirect, float)
    else:
        Lphotospheric = np.asarray(Lbol, float).copy()
        Ldirect = np.zeros(np.asarray(Lbol).shape, dtype=float)
    return BolometricLC(
        t_days=np.asarray(t_days, float),
        Lbol=np.asarray(Lbol, float),
        Teff=np.asarray(Teff, float),
        Rph=np.asarray(Rph, float),
        Lphotospheric=Lphotospheric,
        Ldirect=Ldirect,
        photosphere_valid=photosphere_valid,
    )


def predict_bol(
    *,
    model: str,
    params: Optional[Dict[str, Any]] = None,
    z: Optional[float] = None,
    t_days: np.ndarray,
    t_max_days: float = 150.0,
    interp_fill: Literal["edge", "nan", "raise"] = "nan",
    solver_kwargs: Optional[Dict[str, Any]] = None,
    enforce_ni_mixing_constraint: bool = True,
) -> np.ndarray:
    """
    Predict a bolometric observable at observer-frame times `t_days`.

    `t_days` and `t_max_days` are interpreted in observer-frame days.
    Internal model evolution is solved in rest-frame time.
    """
    model = canonical_model_name(model, warn_legacy=True)
    ctx = _context_from_forward_inputs(
        z=z,
        distance_modulus=None,
        filters=None,
        y_kind="mag",
        mag_system="ab",
        extinction=None,
        require_filters=False,
        require_distance=False,
    )
    engine = _get_engine(model)
    model_vector = _resolve_forward_params(
        model,
        params=params,
        allow_missing_tfloor=True,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )
    solver = _resolve_solver_kwargs(solver_kwargs, model=model)
    if not enforce_ni_mixing_constraint:
        solver["enforce_ni_mixing_constraint"] = False
    z = ctx.distance.get_z()
    t_s, Lbol, Teff, Rph, transport = _solve_state(
        engine,
        model_vector,
        **solver,
        t_max_days_obs=t_max_days,
        z=z,
    )
    _validate_solved_state(
        Lbol,
        Teff,
        Rph,
        None if transport is None else transport.photosphere_valid,
    )
    t_grid_days = _t_grid_days_from_ts(t_s, z=z)

    # Lbol is strictly positive; log10 interpolation is more stable.
    return interp_fit(
        t_grid_days,
        np.asarray(Lbol, float),
        np.asarray(t_days, float),
        yscale="log10",
        fill=interp_fill,
    )


def lightcurve_multiband(
    *,
    model: str,
    params: Optional[Dict[str, Any]] = None,
    z: Optional[float] = None,
    distance_modulus: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    bands: Sequence[str],
    y_kind: Literal["mag", "flux"] = "mag",
    mag_system: Literal["ab", "vega"] = "ab",
    extinction: Optional[Dict[str, Any] | ExtinctionSpec] = None,
    t_max_days: float = 150.0,
    sed=None,
    solver_kwargs: Optional[Dict[str, Any]] = None,
    enforce_ni_mixing_constraint: bool = True,
) -> MultiBandLC:
    """
    Return a multi-band light curve on an observer-frame time grid.

    User-facing time arguments are observer-frame days.
    Internal model evolution is solved in rest-frame time.
    """
    model = canonical_model_name(model, warn_legacy=True)
    ctx = _context_from_forward_inputs(
        z=z,
        distance_modulus=distance_modulus,
        filters=filters,
        y_kind=y_kind,
        mag_system=mag_system,
        extinction=extinction,
        require_filters=True,
        require_distance=True,
    )
    sed = sed or BlackbodySED()
    bands = [normalize_band_label(b) for b in list(bands)]
    filter_map = validate_filter_map(
        ctx.filters or {},
        used_bands=bands,
        mag_system=_effective_mag_system(ctx.y_kind, ctx.mag_system),
    )
    z = ctx.distance.get_z()
    extinction_spec = validate_extinction_spec(
        ctx.extinction,
        used_bands=bands,
        filter_map=filter_map,
        z=z,
    )

    solver = _resolve_solver_kwargs(solver_kwargs, model=model)
    if not enforce_ni_mixing_constraint:
        solver["enforce_ni_mixing_constraint"] = False
    engine = _get_engine(model)
    model_vector = _resolve_forward_params(
        model,
        params=params,
        allow_missing_tfloor=False,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )
    DL_cm = ctx.distance.get_DL_cm()
    t_s, Lbol, Teff, Rph, transport = _solve_state(
        engine,
        model_vector,
        **solver,
        t_max_days_obs=t_max_days,
        z=z,
    )
    _validate_solved_state(
        Lbol,
        Teff,
        Rph,
        None if transport is None else transport.photosphere_valid,
    )
    t_days = _t_grid_days_from_ts(t_s, z=z)
    y_grid = _evaluate_multiband_solved_state(
        model=model,
        model_vector=model_vector,
        transport=transport,
        Teff=Teff,
        Rph=Rph,
        sed=sed,
        filter_map=filter_map,
        bands=bands,
        DL_cm=DL_cm,
        z=z,
        y_kind=ctx.y_kind,
        mag_system=ctx.mag_system,
        extinction=extinction_spec,
    )

    y = {b: np.asarray(y_grid[i], float).copy() for i, b in enumerate(bands)}
    return MultiBandLC(t_days=np.asarray(t_days, float), bands=bands, y=y)


def predict_multiband(
    *,
    model: str,
    params: Optional[Dict[str, Any]] = None,
    z: Optional[float] = None,
    distance_modulus: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    t_days: np.ndarray,
    band: np.ndarray,
    y_kind: Literal["mag", "flux"] = "mag",
    mag_system: Literal["ab", "vega"] = "ab",
    extinction: Optional[Dict[str, Any] | ExtinctionSpec] = None,
    t_max_days: float = 150.0,
    interp_fill: Literal["edge", "nan", "raise"] = "nan",
    sed=None,
    solver_kwargs: Optional[Dict[str, Any]] = None,
    enforce_ni_mixing_constraint: bool = True,
) -> np.ndarray:
    """
    Predict multi-band observables at observer-frame times `t_days`.

    `t_days` and `t_max_days` are interpreted in observer-frame days.
    Internal model evolution is solved in rest-frame time.
    """
    model = canonical_model_name(model, warn_legacy=True)
    ctx = _context_from_forward_inputs(
        z=z,
        distance_modulus=distance_modulus,
        filters=filters,
        y_kind=y_kind,
        mag_system=mag_system,
        extinction=extinction,
        require_filters=True,
        require_distance=True,
    )
    sed = sed or BlackbodySED()
    t_days = np.asarray(t_days, float).reshape(-1)

    # Keep band case; only strip whitespace.
    band = np.asarray([normalize_band_label(b) for b in np.asarray(band).reshape(-1)], dtype=object)
    _check_same_length(t_days=t_days, band=band)

    uniq = sorted(set(band.tolist()))
    filter_map = validate_filter_map(
        ctx.filters or {},
        used_bands=uniq,
        mag_system=_effective_mag_system(ctx.y_kind, ctx.mag_system),
    )
    z = ctx.distance.get_z()
    extinction_spec = validate_extinction_spec(
        ctx.extinction,
        used_bands=uniq,
        filter_map=filter_map,
        z=z,
    )

    solver = _resolve_solver_kwargs(solver_kwargs, model=model)
    if not enforce_ni_mixing_constraint:
        solver["enforce_ni_mixing_constraint"] = False
    engine = _get_engine(model)
    model_vector = _resolve_forward_params(
        model,
        params=params,
        allow_missing_tfloor=False,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )
    DL_cm = ctx.distance.get_DL_cm()
    t_s, Lbol, Teff, Rph, transport = _solve_state(
        engine,
        model_vector,
        **solver,
        t_max_days_obs=t_max_days,
        z=z,
    )
    _validate_solved_state(
        Lbol,
        Teff,
        Rph,
        None if transport is None else transport.photosphere_valid,
    )
    t_grid_days = _t_grid_days_from_ts(t_s, z=z)
    y_grid = _evaluate_multiband_solved_state(
        model=model,
        model_vector=model_vector,
        transport=transport,
        Teff=Teff,
        Rph=Rph,
        sed=sed,
        filter_map=filter_map,
        bands=uniq,
        DL_cm=DL_cm,
        z=z,
        y_kind=ctx.y_kind,
        mag_system=ctx.mag_system,
        extinction=extinction_spec,
    )
    itp_yscale = "linear" if ctx.y_kind == "mag" else "log10"

    out = np.empty_like(t_days, float)
    for i, b in enumerate(uniq):
        idx = np.where(band == b)[0]
        if idx.size == 0:
            continue
        out[idx] = interp_fit(
            t_grid_days,
            np.asarray(y_grid[i], float),
            t_days[idx],
            yscale=itp_yscale,
            fill=interp_fill,
        )
    return out


# -------------------------
# Fitting helpers
# -------------------------

_DEFAULT_FIXED_MODEL_PARAMS = {
    "nickel": {"delta": 0.0, "n": 10.0},
    "csm": {"n": 10.0, "delta": 0.0},
    "magnetar": {"f_mag": 0.2},
    "magnetar_ni": {"f_mag": 0.2},
}


def _apply_default_fixed_model_params(
    model: str,
    priors_model: Optional[Dict[str, Any]],
    fixed_model: Optional[Dict[str, float]],
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    fixed_out = dict(fixed_model or {})
    prior_names = set(dict(priors_model or {}).keys())
    model_canonical = canonical_model_name(model, warn_legacy=False)
    defaults = dict(_DEFAULT_FIXED_MODEL_PARAMS.get(model_canonical, {}))
    solver_kwargs = dict((model_kwargs or {}).get("solver_kwargs", {}) or {})
    if model_canonical == "nickel":
        density_profile = _normalize_density_profile(
            solver_kwargs.get("density_profile", "uniform")
        )
        if density_profile == "exponential":
            defaults["R_0"] = _DEFAULT_IA_R0_RSUN

    for name, value in defaults.items():
        if name not in fixed_out and name not in prior_names:
            fixed_out[name] = float(value)
    return fixed_out


def _split_sampling(
    names_all: List[str],
    bounds_all: np.ndarray,
    fixed: Optional[Dict[str, float]],
):
    fixed = dict(fixed or {})
    unknown_fixed = sorted(set(fixed.keys()) - set(names_all))
    if unknown_fixed:
        raise KeyError(
            f"Unknown fixed parameter(s): {unknown_fixed}. Allowed: {names_all}"
        )
    bounds_all = np.asarray(bounds_all, float)

    names_samp: List[str] = []
    bounds_samp: List[List[float]] = []

    for n, (lo, hi) in zip(names_all, bounds_all):
        lo = float(lo)
        hi = float(hi)
        if n in fixed:
            continue
        names_samp.append(n)
        bounds_samp.append([lo, hi])

    return names_samp, np.asarray(bounds_samp, float), fixed


def _assemble_model_params(
    sample_vec: np.ndarray,
    names_samp: List[str],
    fixed: Dict[str, float],
    names_all: List[str],
):
    vals = _param_values_from_sample(sample_vec, names_samp, fixed)
    return _assemble_model_params_from_values(vals, names_all)


def _param_values_from_sample(
    sample_vec: np.ndarray,
    names_samp: List[str],
    fixed: Dict[str, float],
) -> Dict[str, float]:
    vals = {str(k): float(v) for k, v in dict(fixed or {}).items()}
    vals.update({str(k): float(v) for k, v in zip(names_samp, np.asarray(sample_vec, float))})
    return vals


def _physical_constraints_lnprior(
    vals: Dict[str, float],
    *,
    model: Optional[str] = None,
    enforce_ni_mixing_constraint: bool = True,
) -> float:
    """
    Model-independent physical constraints that cannot be expressed as
    independent box priors.
    """
    reason = _physical_constraint_reason(
        vals,
        model=model,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )
    return -np.inf if reason is not None else 0.0


def _validate_fixed_physical_constraints(
    fixed: Dict[str, float],
    *,
    model: Optional[str] = None,
    enforce_ni_mixing_constraint: bool = True,
) -> None:
    reason = _physical_constraint_reason(
        fixed,
        model=model,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )
    if reason is not None:
        raise ValueError(f"Fixed physical constraints are invalid: {reason}")


def _validate_sampling_bounds_physical_constraints(names: Sequence[str], bounds: np.ndarray) -> None:
    b = np.asarray(bounds, float)
    for name, (lo, hi) in zip([str(n) for n in names], b):
        if name in _POSITIVE_PARAMS and lo <= 0.0:
            raise ValueError(f"Prior bounds for '{name}' must be > 0.")
        if name in _NONNEGATIVE_PARAMS and lo < 0.0:
            raise ValueError(f"Prior bounds for '{name}' must be >= 0.")
        if name in _UNIT_INTERVAL_PARAMS and (lo < 0.0 or hi > 1.0):
            raise ValueError(f"Prior bounds for '{name}' must stay within [0, 1].")
        if name == "delta" and hi >= 3.0:
            raise ValueError("Prior bounds for 'delta' must satisfy hi < 3.")
        if name == "n" and lo <= 5.0:
            raise ValueError("Prior bounds for 'n' must satisfy lo > 5.")


def _assemble_model_params_from_values(
    vals: Dict[str, float],
    names_all: List[str],
):
    t_shift = float(vals.get("t_shift", 0.0))

    model_params: Dict[str, float] = {}
    for n in names_all:
        if n == "t_shift":
            continue
        if n not in vals:
            raise KeyError(
                f"Missing parameter '{n}'. "
                f"Either provide it in priors (to sample) or in fixed."
            )
        model_params[str(n)] = float(vals[n])

    return model_params, t_shift


def _apply_log10_priors(
    names: Sequence[str],
    bounds: np.ndarray,
    priors_log10: Optional[Dict[str, Tuple[float, float]]],
):
    """
    Apply log10 prior bounds and return updated bounds + log-prior flags.

    `priors_log10` format:
      {"param": (log10_lo, log10_hi), ...}
    """
    b = np.asarray(bounds, float).copy()
    names_l = [str(n) for n in names]
    idx = {n: i for i, n in enumerate(names_l)}
    log_set = set()

    if not priors_log10:
        return b, log_set

    for k, vv in dict(priors_log10).items():
        if k not in idx:
            raise KeyError(f"Unknown log prior key '{k}'. Allowed: {names_l}")
        if vv is None or len(vv) != 2:
            raise ValueError(f"Invalid log10 bounds for '{k}': {vv!r}")

        lo_log10 = float(vv[0])
        hi_log10 = float(vv[1])
        if not (lo_log10 < hi_log10):
            raise ValueError(
                f"Invalid log10 bounds for '{k}': ({lo_log10}, {hi_log10})"
            )

        lo = 10.0 ** lo_log10
        hi = 10.0 ** hi_log10
        if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0.0 and hi > 0.0):
            raise ValueError(
                f"log10 bounds for '{k}' lead to invalid linear bounds: ({lo}, {hi})"
            )

        b[idx[k], 0] = lo
        b[idx[k], 1] = hi
        log_set.add(k)

    return b, log_set


def _validate_prior_bounds_pair(name: str, lo: float, hi: float) -> Tuple[float, float]:
    lo = float(lo)
    hi = float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"Invalid bounds for '{name}': ({lo}, {hi}); require finite lo < hi.")
    return lo, hi


def _split_prior_specs(
    priors: Optional[Dict[str, Any]],
):
    """
    Parse mixed prior specs into linear bounds and log10 bounds.

    Supported `priors` formats per parameter:
      - (lo, hi)                      -> linear uniform bounds
      - ("log10", lo_log10, hi_log10) -> log10 bounds, log-uniform prior
      - {"bounds": (lo, hi), "scale": "linear" | "log10"}
    """
    if not priors:
        return {}, {}

    pri_lin: Dict[str, Tuple[float, float]] = {}
    pri_log10: Dict[str, Tuple[float, float]] = {}

    for k, spec in dict(priors).items():
        if isinstance(spec, dict):
            if "bounds" not in spec:
                raise ValueError(
                    f"Prior spec for '{k}' must contain key 'bounds' when using dict format."
                )
            b = spec["bounds"]
            if b is None or len(b) != 2:
                raise ValueError(f"Invalid bounds for '{k}': {b!r}")
            lo, hi = _validate_prior_bounds_pair(k, b[0], b[1])

            scale = str(spec.get("scale", "linear")).strip().lower()
            if scale in ("linear", "lin"):
                pri_lin[k] = (lo, hi)
            elif scale in ("log10", "log"):
                pri_log10[k] = (lo, hi)
            else:
                raise ValueError(
                    f"Unknown prior scale '{scale}' for '{k}'. Use 'linear' or 'log10'."
                )
            continue

        if isinstance(spec, (tuple, list)):
            if len(spec) == 2 and not isinstance(spec[0], str):
                lo, hi = _validate_prior_bounds_pair(k, spec[0], spec[1])
                pri_lin[k] = (lo, hi)
                continue

            if len(spec) == 3 and isinstance(spec[0], str):
                mode = str(spec[0]).strip().lower()
                lo, hi = _validate_prior_bounds_pair(k, spec[1], spec[2])
                if mode in ("log10", "log"):
                    pri_log10[k] = (lo, hi)
                elif mode in ("linear", "lin"):
                    pri_lin[k] = (lo, hi)
                else:
                    raise ValueError(
                        f"Unknown prior mode '{mode}' for '{k}'. Use 'linear' or 'log10'."
                    )
                continue

        raise ValueError(
            f"Invalid prior spec for '{k}': {spec!r}. "
            "Use (lo, hi), ('log10', lo, hi), or {'bounds': (lo, hi), 'scale': 'linear|log10'}."
        )

    return pri_lin, pri_log10


def _log10_bounds_to_linear(name: str, bounds_log10: Tuple[float, float]) -> Tuple[float, float]:
    lo_log10 = float(bounds_log10[0])
    hi_log10 = float(bounds_log10[1])
    lo = 10.0 ** lo_log10
    hi = 10.0 ** hi_log10
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0.0 and hi > 0.0):
        raise ValueError(
            f"log10 bounds for '{name}' lead to invalid linear bounds: ({lo}, {hi})"
        )
    return float(lo), float(hi)


def _validate_likelihood_nuisance_value(
    value: float,
    spec: Dict[str, Any],
    *,
    label: str,
) -> float:
    out = float(value)
    minimum = spec.get("minimum")
    if not np.isfinite(out):
        raise ValueError(f"{label} must be finite.")
    if minimum is not None and out < float(minimum):
        raise ValueError(f"{label} must be >= {float(minimum):g}.")
    return out


def _validate_likelihood_nuisance_bounds(
    name: str,
    bounds: Tuple[float, float],
    spec: Dict[str, Any],
) -> Tuple[float, float]:
    lo = float(bounds[0])
    hi = float(bounds[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"Prior bounds for '{name}' must satisfy finite lo < hi.")
    minimum = spec.get("minimum")
    if minimum is not None and lo < float(minimum):
        raise ValueError(f"Prior bounds for '{name}' must satisfy lo >= {float(minimum):g}.")
    return lo, hi


def _split_likelihood_nuisance_fit_inputs(
    priors: Optional[Dict[str, Any]],
    fixed: Optional[Dict[str, float]],
):
    """
    Separate likelihood-only nuisance parameters from model parameters.

    Nuisance parameters are sampled/fixed through the fit API but are never
    part of the forward-model parameter vector.
    """
    priors_model = dict(priors or {})
    fixed_model = dict(fixed or {})
    nuisance_cfgs: Dict[str, Dict[str, Any]] = {}

    for name, spec in LIKELIHOOD_NUISANCE_PARAM_SPECS.items():
        prior_spec = priors_model.pop(name, None)
        fixed_present = name in fixed_model
        fixed_value = None
        if fixed_present:
            fixed_value = _validate_likelihood_nuisance_value(
                fixed_model.pop(name),
                spec,
                label=f"fixed['{name}']",
            )

        cfg: Dict[str, Any] = dict(
            enabled=prior_spec is not None or fixed_present,
            sampled=False,
            fixed=fixed_present,
            value=fixed_value,
            bounds=None,
            log_flag=False,
            units=spec.get("units"),
            likelihood=spec.get("likelihood", "gaussian"),
        )

        if prior_spec is not None:
            nuisance_lin, nuisance_log10 = _split_prior_specs({name: prior_spec})
            if name in nuisance_log10:
                bounds = _log10_bounds_to_linear(name, nuisance_log10[name])
                cfg["log_flag"] = True
            else:
                bounds = nuisance_lin[name]

            bounds = _validate_likelihood_nuisance_bounds(name, bounds, spec)
            cfg["bounds"] = bounds

            if fixed_present:
                lo, hi = bounds
                if not (lo <= float(fixed_value) <= hi):
                    raise ValueError(
                        f"fixed['{name}']={fixed_value} out of bounds ({lo}, {hi})"
                    )
            else:
                cfg["sampled"] = True

        nuisance_cfgs[name] = cfg

    return priors_model, fixed_model, nuisance_cfgs


def _add_fixed_likelihood_nuisance(
    fixed: Dict[str, float],
    nuisance_cfgs: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    out = dict(fixed or {})
    for name, cfg in nuisance_cfgs.items():
        if bool(cfg.get("fixed", False)):
            out[name] = float(cfg["value"])
    return out


def _likelihood_nuisance_meta(
    nuisance_cfgs: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {
        name: dict(
            enabled=bool(cfg["enabled"]),
            sampled=bool(cfg["sampled"]),
            fixed=bool(cfg["fixed"]),
            value=cfg["value"],
            bounds=cfg["bounds"],
            scale="log10" if cfg["log_flag"] else "linear",
            units=cfg.get("units"),
        )
        for name, cfg in nuisance_cfgs.items()
    }


def _append_likelihood_nuisance_sampling(
    names_samp: List[str],
    bounds_samp: np.ndarray,
    log_flags_samp: List[bool],
    nuisance_cfgs: Dict[str, Dict[str, Any]],
):
    out_names = list(names_samp)
    out_log_flags = list(log_flags_samp)

    bounds_arr = np.asarray(bounds_samp, float)
    if bounds_arr.size == 0:
        bounds_arr = np.empty((0, 2), dtype=float)

    for name, cfg in nuisance_cfgs.items():
        if not bool(cfg.get("sampled", False)):
            continue
        bounds_arr = np.vstack(
            [bounds_arr, np.asarray(cfg["bounds"], float).reshape(1, 2)]
        )
        out_names.append(name)
        out_log_flags.append(bool(cfg.get("log_flag", False)))

    return out_names, bounds_arr, out_log_flags


def _likelihood_nuisance_values(
    vals: Dict[str, float],
    nuisance_cfgs: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    return {
        name: float(vals[name])
        for name, cfg in nuisance_cfgs.items()
        if bool(cfg.get("enabled", False)) and name in vals
    }


def _fit_likelihood_name(nuisance_cfgs: Dict[str, Dict[str, Any]]) -> str:
    active = [
        str(cfg.get("likelihood", "gaussian"))
        for cfg in nuisance_cfgs.values()
        if bool(cfg.get("enabled", False))
    ]
    if not active:
        return "gaussian"
    unique = sorted(set(active))
    return unique[0] if len(unique) == 1 else "gaussian_with_nuisance"


@dataclass(frozen=True)
class _MultibandPredictor:
    model: str
    z: Optional[float]
    distance_modulus: Optional[float]
    filters: Dict[str, FilterProfile]
    band: np.ndarray
    y_kind: str
    mag_system: str
    extinction_spec: Optional[ExtinctionSpec]
    sed: Any
    interp_fill_fit: str
    model_kwargs_pred: Dict[str, Any]
    enforce_ni_mixing_constraint: bool = True

    def __call__(self, model_params: Dict[str, float], t_eval: np.ndarray) -> np.ndarray:
        return predict_multiband(
            model=self.model,
            params=model_params,
            z=self.z,
            distance_modulus=self.distance_modulus,
            filters=self.filters,
            t_days=t_eval,
            band=self.band,
            y_kind=self.y_kind,
            mag_system=self.mag_system,
            extinction=self.extinction_spec,
            sed=self.sed,
            interp_fill=self.interp_fill_fit,
            enforce_ni_mixing_constraint=self.enforce_ni_mixing_constraint,
            **self.model_kwargs_pred,
        )


@dataclass(frozen=True)
class _BolometricPredictor:
    model: str
    z: float
    interp_fill_fit: str
    model_kwargs_pred: Dict[str, Any]
    enforce_ni_mixing_constraint: bool = True

    def __call__(self, model_params: Dict[str, float], t_eval: np.ndarray) -> np.ndarray:
        return predict_bol(
            model=self.model,
            params=model_params,
            z=self.z,
            t_days=t_eval,
            interp_fill=self.interp_fill_fit,
            enforce_ni_mixing_constraint=self.enforce_ni_mixing_constraint,
            **self.model_kwargs_pred,
        )


@dataclass(frozen=True)
class _FitLnProb:
    model: str
    prior: MixedBoundsPrior
    names_samp: List[str]
    fixed: Dict[str, float]
    names_all: List[str]
    t_obs: np.ndarray
    y_obs: np.ndarray
    y_err: np.ndarray
    predictor: Callable[[Dict[str, float], np.ndarray], np.ndarray]
    likelihood_y_kind: str
    nuisance_cfgs: Dict[str, Dict[str, Any]]
    enforce_ni_mixing_constraint: bool = True
    is_upper_limit: Optional[np.ndarray] = None
    upper_limit_nsigma: Optional[np.ndarray] = None

    def __post_init__(self):
        n_obs = np.asarray(self.t_obs).size
        if self.is_upper_limit is None:
            upper = np.zeros(n_obs, dtype=bool)
        else:
            upper = np.asarray(self.is_upper_limit, bool).reshape(-1)
            if upper.size != n_obs:
                raise ValueError("is_upper_limit must have the same length as t_obs.")

        if self.upper_limit_nsigma is None:
            nsigma = np.full(n_obs, np.nan, dtype=float)
        else:
            nsigma = np.asarray(self.upper_limit_nsigma, float).reshape(-1)
            if nsigma.size != n_obs:
                raise ValueError(
                    "upper_limit_nsigma must have the same length as t_obs."
                )
        object.__setattr__(self, "is_upper_limit", upper)
        object.__setattr__(self, "upper_limit_nsigma", nsigma)

    def __call__(self, sample_vec: np.ndarray) -> float:
        lp = self.prior.lnprior(sample_vec)
        if not np.isfinite(lp):
            return -np.inf

        vals = _param_values_from_sample(sample_vec, self.names_samp, self.fixed)
        lp_phys = _physical_constraints_lnprior(
            vals,
            model=self.model,
            enforce_ni_mixing_constraint=self.enforce_ni_mixing_constraint,
        )
        if not np.isfinite(lp_phys):
            return -np.inf

        model_params, t_shift = _assemble_model_params_from_values(vals, self.names_all)
        # Shift model time axis by t_shift and compare on observed t_obs.
        # Legacy convention:
        # y_model_shifted(t_obs) = y_model_raw(t_obs + t_shift)
        t_eval = self.t_obs + t_shift

        try:
            y_mod = self.predictor(model_params, t_eval)
        except NonPhysicalModelError:
            return -np.inf

        y_mod = np.asarray(y_mod, float).reshape(-1)
        if y_mod.size != np.asarray(self.t_obs).size:
            return -np.inf
        if np.any(~np.isfinite(y_mod)):
            return -np.inf

        upper = np.asarray(self.is_upper_limit, bool)
        detection = ~upper

        lnlike_detection = 0.0
        if np.any(detection):
            lnlike_detection = gaussian_lnlike_with_nuisance(
                y_kind=self.likelihood_y_kind,
                y_obs=self.y_obs[detection],
                y_model=y_mod[detection],
                y_err=self.y_err[detection],
                nuisance_params=_likelihood_nuisance_values(vals, self.nuisance_cfgs),
            )
            if not np.isfinite(lnlike_detection):
                return -np.inf

        lnlike_upper = 0.0
        if np.any(upper):
            lnlike_upper = upper_limit_gaussian_cdf_lnlike(
                y_kind=self.likelihood_y_kind,
                y_limit=self.y_obs[upper],
                y_model=y_mod[upper],
                y_err=self.y_err[upper],
                upper_limit_nsigma=self.upper_limit_nsigma[upper],
            )
            if not np.isfinite(lnlike_upper):
                return -np.inf

        return lp + lp_phys + lnlike_detection + lnlike_upper


def _run_sampler(
    *,
    sampler: str,
    lnprob,
    prior,
    sampler_kwargs: Dict[str, Any],
):
    """
    Dispatch sampler backend and return (samples, logp, meta, sampler_name).
    """
    key = str(sampler).lower().strip()
    kw = dict(sampler_kwargs or {})

    if key == "emcee":
        ndim = len(prior.param_names)
        nwalkers = int(kw.pop("nwalkers", max(32, 4 * max(ndim, 1))))
        nsteps = int(kw.pop("nsteps", 2000))
        burnin = int(kw.pop("burnin", nsteps // 2))
        thin = int(kw.pop("thin", 1))
        seed = kw.pop("seed", None)
        init = kw.pop("init", "prior")
        pool = kw.pop("pool", None)
        progress = bool(kw.pop("progress", True))

        samples, logp, meta = run_emcee(
            lnprob=lnprob,
            prior=prior,
            nwalkers=nwalkers,
            nsteps=nsteps,
            burnin=burnin,
            thin=thin,
            seed=seed,
            init=init,
            pool=pool,
            progress=progress,
            **kw,
        )
        return samples, logp, meta, "emcee"

    if key == "zeus":
        ndim = len(prior.param_names)
        nwalkers = int(kw.pop("nwalkers", max(32, 4 * max(ndim, 1))))
        nsteps = int(kw.pop("nsteps", 2000))
        burnin = int(kw.pop("burnin", nsteps // 2))
        thin = int(kw.pop("thin", 1))
        seed = kw.pop("seed", None)
        init = kw.pop("init", "prior")
        pool = kw.pop("pool", None)
        progress = bool(kw.pop("progress", True))

        samples, logp, meta = run_zeus(
            lnprob=lnprob,
            prior=prior,
            nwalkers=nwalkers,
            nsteps=nsteps,
            burnin=burnin,
            thin=thin,
            seed=seed,
            init=init,
            pool=pool,
            progress=progress,
            **kw,
        )
        return samples, logp, meta, "zeus"

    if key == "dynesty":
        nlive = int(kw.pop("nlive", 200))
        sample = kw.pop("sample", "rwalk")
        bound = kw.pop("bound", "multi")
        dlogz = float(kw.pop("dlogz", 0.1))
        maxiter = kw.pop("maxiter", None)
        maxcall = kw.pop("maxcall", None)
        seed = kw.pop("seed", None)
        progress = bool(kw.pop("progress", True))
        nsamples = kw.pop("nsamples", None)
        add_live = bool(kw.pop("add_live", True))
        pool = kw.pop("pool", None)
        queue_size = kw.pop("queue_size", None)

        samples, logp, meta = run_dynesty(
            lnprob=lnprob,
            prior=prior,
            nlive=nlive,
            sample=sample,
            bound=bound,
            dlogz=dlogz,
            maxiter=maxiter,
            maxcall=maxcall,
            seed=seed,
            progress=progress,
            nsamples=nsamples,
            add_live=add_live,
            pool=pool,
            queue_size=queue_size,
            **kw,
        )
        return samples, logp, meta, "dynesty"

    raise ValueError(
        f"Unknown sampler='{sampler}'. Supported samplers: 'emcee', 'zeus', 'dynesty'."
    )


# -------------------------
# Fit-time interpolation policy
# -------------------------

def _split_fit_model_kwargs(
    model_kwargs: Optional[Dict[str, Any]],
    *,
    model: str,
):
    """
    Extract prediction kwargs and interpolation fill policy used by fit lnprob.

    For fitting, edge-clamped extrapolation is disallowed because it evaluates
    points outside model time range using boundary values.
    """
    mk = dict(model_kwargs or {})
    fill = str(mk.pop("interp_fill", "nan")).strip().lower()

    if fill not in ("edge", "nan", "raise"):
        raise ValueError(
            "model_kwargs['interp_fill'] must be one of: 'edge', 'nan', 'raise'."
        )
    if fill == "edge":
        raise ValueError(
            "model_kwargs['interp_fill']='edge' is not allowed in fitting. "
            "Use 'nan' or 'raise' to avoid out-of-range edge extrapolation."
        )

    solver_kwargs = dict(mk.pop("solver_kwargs", {}) or {})
    for key in ("Nx", "Ny"):
        if key in mk:
            solver_kwargs[key] = mk.pop(key)
    if solver_kwargs:
        mk["solver_kwargs"] = _resolve_solver_kwargs(solver_kwargs, model=model)

    return mk, fill


def _validate_density_structure_fit_configuration(
    *,
    model: str,
    priors: Optional[Dict[str, Any]],
    fixed: Optional[Dict[str, float]],
    model_kwargs: Dict[str, Any],
) -> None:
    if canonical_model_name(model, warn_legacy=False) != "nickel":
        return

    solver_kwargs = dict((model_kwargs or {}).get("solver_kwargs", {}) or {})
    density_profile = str(solver_kwargs.get("density_profile", "uniform"))
    if density_profile == "exponential" and "R_0" in dict(priors or {}):
        raise ValueError(
            "Nickel Ia/exponential density fixes R_0 instead of sampling it; "
            "remove R_0 from priors and optionally override it through fixed."
        )
    if density_profile == "broken_power_law":
        return

    sampled_structure = sorted({"delta", "n"}.intersection(dict(priors or {})))
    if sampled_structure:
        raise ValueError(
            f"Nickel BPL parameter prior(s) {sampled_structure} require "
            "model_kwargs['solver_kwargs']['density_profile']='bpl'."
        )

    fixed_values = dict(fixed or {})
    nondefault_fixed = []
    if "delta" in fixed_values and float(fixed_values["delta"]) != 0.0:
        nondefault_fixed.append("delta")
    if "n" in fixed_values and float(fixed_values["n"]) != 10.0:
        nondefault_fixed.append("n")
    if nondefault_fixed:
        raise ValueError(
            f"Non-default nickel BPL fixed parameter(s) {nondefault_fixed} require "
            "model_kwargs['solver_kwargs']['density_profile']='bpl'."
        )


def _csm_photosphere_mode(model_kwargs: Optional[Dict[str, Any]]) -> str:
    solver_kwargs = dict((model_kwargs or {}).get("solver_kwargs", {}) or {})
    return str(solver_kwargs.get("photosphere_mode", "tau")).strip().lower()


def _validate_csm_photosphere_fit_configuration(
    *,
    model: str,
    priors: Optional[Dict[str, Any]],
    model_kwargs: Dict[str, Any],
) -> None:
    if canonical_model_name(model, warn_legacy=False) != "csm":
        return
    if _csm_photosphere_mode(model_kwargs) == "tau" and "T_floor" in dict(
        priors or {}
    ):
        raise ValueError(
            "CSM photosphere_mode='tau' determines temperature from the "
            "tau=2/3 radius, so T_floor is inactive and cannot be sampled. "
            "Use photosphere_mode='outer' to fit T_floor."
        )


def _apply_multiband_csm_photosphere_defaults(
    *,
    model: str,
    priors_model: Optional[Dict[str, Any]],
    fixed_model: Dict[str, float],
    model_kwargs: Dict[str, Any],
) -> Dict[str, float]:
    fixed_out = dict(fixed_model)
    if (
        canonical_model_name(model, warn_legacy=False) == "csm"
        and _csm_photosphere_mode(model_kwargs) == "tau"
        and "T_floor" not in dict(priors_model or {})
        and "T_floor" not in fixed_out
    ):
        fixed_out["T_floor"] = 5000.0
    return fixed_out


def _t_shift_upper_for_fit(names_all: Sequence[str], bounds_all: np.ndarray, fixed: Dict[str, float]) -> float:
    if "t_shift" in fixed:
        return float(fixed["t_shift"])
    names = [str(n) for n in names_all]
    if "t_shift" not in names:
        return 0.0
    idx = names.index("t_shift")
    return float(np.asarray(bounds_all, float)[idx, 1])


def _resolve_fit_t_max_days(
    model_kwargs: Dict[str, Any],
    *,
    t_obs: np.ndarray,
    names_all: Sequence[str],
    bounds_all: np.ndarray,
    fixed: Dict[str, float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Ensure fit-time model grids cover all possible t_obs + t_shift values.

    Explicit user values are accepted when they cover the allowed time range.
    Automatic values add a small padding for interpolation/plot reuse.
    """
    mk = dict(model_kwargs or {})
    t_max_obs = float(np.nanmax(np.asarray(t_obs, float)))
    t_shift_upper = max(0.0, _t_shift_upper_for_fit(names_all, bounds_all, fixed))
    required = max(0.0, t_max_obs + t_shift_upper)

    if "t_max_days" in mk:
        try:
            user_t_max = float(mk["t_max_days"])
        except Exception as exc:
            raise ValueError("model_kwargs['t_max_days'] must be a positive finite number.") from exc
        if not np.isfinite(user_t_max) or user_t_max <= 0.0:
            raise ValueError("model_kwargs['t_max_days'] must be a positive finite number.")
        if user_t_max + 1e-12 < required:
            raise ValueError(
                "model_kwargs['t_max_days'] is too small for data.t_days and the allowed t_shift range. "
                f"Need at least {required:.6g} observer-frame days, got {user_t_max:.6g}."
            )
        mk["t_max_days"] = user_t_max
        return mk, dict(
            t_max_days=user_t_max,
            t_max_days_auto=False,
            t_max_days_required=required,
            t_shift_upper=t_shift_upper,
        )

    auto_t_max = max(_FIT_T_MAX_DAYS_DEFAULT, required + _FIT_T_MAX_DAYS_PADDING)
    mk["t_max_days"] = float(auto_t_max)
    return mk, dict(
        t_max_days=float(auto_t_max),
        t_max_days_auto=True,
        t_max_days_required=required,
        t_shift_upper=t_shift_upper,
        t_max_days_padding=_FIT_T_MAX_DAYS_PADDING,
    )


def fit_multiband(
    *,
    data: MultiBandData,
    model: str,
    z: Optional[float] = None,
    distance_modulus: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    y_kind: Literal["mag", "flux"] = "mag",
    mag_system: Literal["ab", "vega"] = "ab",
    extinction: Optional[Dict[str, Any] | ExtinctionSpec] = None,
    priors: Optional[Dict[str, Any]] = None,
    fixed: Optional[Dict[str, float]] = None,
    sampler: str = "emcee",
    sed=None,
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    enforce_ni_mixing_constraint: bool = True,
) -> FitResult:
    model = canonical_model_name(model, warn_legacy=True)
    ctx = _context_from_fit_inputs(
        z=z,
        distance_modulus=distance_modulus,
        filters=filters,
        y_kind=y_kind,
        mag_system=mag_system,
        extinction=extinction,
        require_filters=True,
    )
    sampler_kwargs = dict(sampler_kwargs or {})
    model_kwargs = dict(model_kwargs or {})
    if sed is None:
        sed = BlackbodySED()
    model_kwargs_pred, interp_fill_fit = _split_fit_model_kwargs(
        model_kwargs,
        model=model,
    )
    _validate_density_structure_fit_configuration(
        model=model,
        priors=priors,
        fixed=fixed,
        model_kwargs=model_kwargs_pred,
    )
    _validate_csm_photosphere_fit_configuration(
        model=model,
        priors=priors,
        model_kwargs=model_kwargs_pred,
    )
    data = _apply_data_filter(data)

    # ---- data ----
    t_obs = _as_1d_float(data.t_days, "data.t_days")
    y_obs = _as_1d_float(data.y, "data.y")
    y_err = _as_1d_float(data.yerr, "data.yerr")
    upper_raw = getattr(data, "is_upper_limit", None)
    is_upper_limit = (
        np.zeros(t_obs.size, dtype=bool)
        if upper_raw is None
        else np.asarray(upper_raw, bool).reshape(-1)
    )
    nsigma_raw = getattr(data, "upper_limit_nsigma", None)
    upper_limit_nsigma = (
        np.full(t_obs.size, np.nan, dtype=float)
        if nsigma_raw is None
        else np.asarray(nsigma_raw, float).reshape(-1)
    )

    # Keep band case; only strip whitespace.
    band = np.asarray([normalize_band_label(b) for b in np.asarray(data.band).reshape(-1)], dtype=object)

    _check_same_length(
        t_days=t_obs,
        band=band,
        y=y_obs,
        yerr=y_err,
        is_upper_limit=is_upper_limit,
        upper_limit_nsigma=upper_limit_nsigma,
    )

    if np.any(~np.isfinite(t_obs)):
        raise ValueError("data.t_days must be finite.")
    _validate_multiband_fit_errors(
        y_err,
        is_upper_limit,
        upper_limit_nsigma,
    )
    _validate_multiband_fit_y(
        y_obs,
        y_kind=ctx.y_kind,
        is_upper_limit=is_upper_limit,
    )

    # ---- bounds/prior ----
    priors_model, fixed_model, nuisance_cfgs = _split_likelihood_nuisance_fit_inputs(
        priors,
        fixed,
    )
    fixed_model = _apply_default_fixed_model_params(
        model,
        priors_model,
        fixed_model,
        model_kwargs_pred,
    )
    fixed_model = _apply_multiband_csm_photosphere_defaults(
        model=model,
        priors_model=priors_model,
        fixed_model=fixed_model,
        model_kwargs=model_kwargs_pred,
    )
    priors_lin, priors_log10 = _split_prior_specs(priors_model)
    names_all, bounds_all = build_bounds(model, priors=priors_lin, include_t_shift=True)
    bounds_all, log_set_all = _apply_log10_priors(names_all, bounds_all, priors_log10)
    _validate_sampling_bounds_physical_constraints(names_all, bounds_all)
    names_samp, bounds_samp, fixed = _split_sampling(names_all, bounds_all, fixed=fixed_model)
    _validate_fixed_physical_constraints(
        fixed,
        model=model,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )
    model_kwargs_pred, tmax_meta = _resolve_fit_t_max_days(
        model_kwargs_pred,
        t_obs=t_obs,
        names_all=names_all,
        bounds_all=bounds_all,
        fixed=fixed,
    )
    log_flags_samp = [n in log_set_all for n in names_samp]
    names_samp, bounds_samp, log_flags_samp = _append_likelihood_nuisance_sampling(
        names_samp,
        bounds_samp,
        log_flags_samp,
        nuisance_cfgs,
    )
    fixed = _add_fixed_likelihood_nuisance(fixed, nuisance_cfgs)
    prior = MixedBoundsPrior(bounds=bounds_samp, param_names=names_samp, log_flags=log_flags_samp)

    # ---- band check (case-sensitive) ----
    if ctx.filters is None:
        raise ValueError("ctx.filters is required for multiband. For bolometric you can omit it.")
    uniq_b = sorted(set(band.tolist()))
    filters = validate_filter_map(
        ctx.filters,
        used_bands=uniq_b,
        mag_system=_effective_mag_system(ctx.y_kind, ctx.mag_system),
    )
    extinction_spec = validate_extinction_spec(
        ctx.extinction,
        used_bands=uniq_b,
        filter_map=filters,
        z=ctx.distance.get_z(),
    )

    distance_modulus_fit = (
        None
        if ctx.distance.DL_cm is None
        else _distance_modulus_from_cm(float(ctx.distance.DL_cm))
    )
    predictor = _MultibandPredictor(
        model=model,
        z=ctx.distance.z,
        distance_modulus=distance_modulus_fit,
        filters=filters,
        band=band,
        y_kind=ctx.y_kind,
        mag_system=ctx.mag_system,
        extinction_spec=extinction_spec,
        sed=sed,
        interp_fill_fit=interp_fill_fit,
        model_kwargs_pred=model_kwargs_pred,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )
    lnprob = _FitLnProb(
        model=model,
        prior=prior,
        names_samp=names_samp,
        fixed=fixed,
        names_all=names_all,
        t_obs=t_obs,
        y_obs=y_obs,
        y_err=y_err,
        predictor=predictor,
        likelihood_y_kind=ctx.y_kind,
        nuisance_cfgs=nuisance_cfgs,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
        is_upper_limit=is_upper_limit,
        upper_limit_nsigma=upper_limit_nsigma,
    )

    samples, logp, meta, sampler_used = _run_sampler(
        sampler=sampler,
        lnprob=lnprob,
        prior=prior,
        sampler_kwargs=sampler_kwargs,
    )

    sed_config = sed_to_dict(sed)
    meta.update(
        dict(
            model=model,
            enforce_ni_mixing_constraint=bool(enforce_ni_mixing_constraint),
            y_kind=ctx.y_kind,
            mag_system=ctx.mag_system,
            names_all=names_all,
            bounds_all=np.asarray(bounds_all, float),
            bounds_samp=np.asarray(bounds_samp, float),
            priors_input=dict(priors or {}),
            priors_linear=dict(priors_lin or {}),
            priors_log10=dict(priors_log10 or {}),
            nuisance_priors=_likelihood_nuisance_meta(nuisance_cfgs),
            log_prior_names=sorted(
                [n for n, log_flag in zip(names_samp, log_flags_samp) if log_flag]
            ),
            likelihood=_fit_likelihood_name(nuisance_cfgs),
            upper_limit_likelihood="gaussian_cdf",
            upper_limit_default_nsigma=DEFAULT_UPPER_LIMIT_NSIGMA,
            n_detections=int(np.sum(~is_upper_limit)),
            n_upper_limits=int(np.sum(is_upper_limit)),
            n_upper_limits_with_error=int(
                np.sum(is_upper_limit & np.isfinite(y_err))
            ),
            n_upper_limits_with_nsigma=int(
                np.sum(
                    is_upper_limit
                    & ~np.isfinite(y_err)
                    & np.isfinite(upper_limit_nsigma)
                )
            ),
            n_upper_limits_default_nsigma=int(
                np.sum(
                    is_upper_limit
                    & ~np.isfinite(y_err)
                    & ~np.isfinite(upper_limit_nsigma)
                )
            ),
            sed=sed_config["name"],
            sed_config=sed_config,
            interp_fill_fit=interp_fill_fit,
            model_kwargs=model_kwargs_pred,
            t_max_days_policy=tmax_meta,
        )
    )

    return FitResult(
        model=model,
        ctx=ctx,
        sampler=sampler_used,
        param_names=names_samp,
        fixed=fixed,
        all_param_names=names_all,
        samples=samples,
        log_prob=logp,
        meta=meta,
    )


def fit_bol(
    *,
    data: BolometricData,
    model: str,
    z: Optional[float] = None,
    priors: Optional[Dict[str, Any]] = None,
    fixed: Optional[Dict[str, float]] = None,
    sampler: str = "emcee",
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    enforce_ni_mixing_constraint: bool = True,
) -> FitResult:
    model = canonical_model_name(model, warn_legacy=True)
    if priors and "T_floor" in priors:
        raise ValueError(
            f"`T_floor` is not a bolometric fit parameter in `fit_bol()`. "
            f"TransFit keeps an internal temperature floor of {_BOL_INTERNAL_T_FLOOR:.0f} K only for numerical stability."
        )
    if fixed and "T_floor" in fixed:
        raise ValueError(
            f"`T_floor` is not a bolometric fit parameter in `fit_bol()`. "
            f"TransFit keeps an internal temperature floor of {_BOL_INTERNAL_T_FLOOR:.0f} K only for numerical stability."
        )

    ctx = _context_from_fit_inputs(
        z=z,
        distance_modulus=None,
        filters=None,
        y_kind="mag",
        mag_system="ab",
        extinction=None,
        require_filters=False,
    )
    sampler_kwargs = dict(sampler_kwargs or {})
    model_kwargs = dict(model_kwargs or {})
    model_kwargs_pred, interp_fill_fit = _split_fit_model_kwargs(
        model_kwargs,
        model=model,
    )
    _validate_density_structure_fit_configuration(
        model=model,
        priors=priors,
        fixed=fixed,
        model_kwargs=model_kwargs_pred,
    )
    data = _apply_data_filter(data)

    t_obs = _as_1d_float(data.t_days, "data.t_days")
    y_obs = _as_1d_float(data.y, "data.y")
    y_err = _as_1d_float(data.yerr, "data.yerr")
    _check_same_length(t_days=t_obs, y=y_obs, yerr=y_err)

    _validate_fit_time_and_errors(t_obs, y_err)
    _validate_bolometric_fit_y(y_obs)

    priors_model, fixed_model, nuisance_cfgs = _split_likelihood_nuisance_fit_inputs(
        priors,
        fixed,
    )
    fixed_model = _apply_default_fixed_model_params(
        model,
        priors_model,
        fixed_model,
        model_kwargs_pred,
    )
    # Bolometric likelihoods do not constrain the blackbody color floor. The
    # internal forward vector receives _BOL_INTERNAL_T_FLOOR later, but it must
    # not remain in the reduced bolometric fit state.
    fixed_model.pop("T_floor", None)
    priors_lin, priors_log10 = _split_prior_specs(priors_model)
    names_all, bounds_all = build_bounds(model, priors=priors_lin, include_t_shift=True)
    bounds_all, log_set_all = _apply_log10_priors(names_all, bounds_all, priors_log10)
    # For bolometric fitting, T_floor stays internal and is not part of the fit state.
    if "T_floor" in names_all:
        i_tf = names_all.index("T_floor")
        names_all = [n for n in names_all if n != "T_floor"]
        bounds_all = np.asarray(bounds_all, float)
        bounds_all = np.delete(bounds_all, i_tf, axis=0)
        if "T_floor" in log_set_all:
            log_set_all.remove("T_floor")
    _validate_sampling_bounds_physical_constraints(names_all, bounds_all)

    names_samp, bounds_samp, fixed = _split_sampling(names_all, bounds_all, fixed=fixed_model)
    _validate_fixed_physical_constraints(
        fixed,
        model=model,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )
    model_kwargs_pred, tmax_meta = _resolve_fit_t_max_days(
        model_kwargs_pred,
        t_obs=t_obs,
        names_all=names_all,
        bounds_all=bounds_all,
        fixed=fixed,
    )
    log_flags_samp = [n in log_set_all for n in names_samp]
    names_samp, bounds_samp, log_flags_samp = _append_likelihood_nuisance_sampling(
        names_samp,
        bounds_samp,
        log_flags_samp,
        nuisance_cfgs,
    )
    fixed = _add_fixed_likelihood_nuisance(fixed, nuisance_cfgs)
    prior = MixedBoundsPrior(bounds=bounds_samp, param_names=names_samp, log_flags=log_flags_samp)

    predictor = _BolometricPredictor(
        model=model,
        z=ctx.distance.get_z(),
        interp_fill_fit=interp_fill_fit,
        model_kwargs_pred=model_kwargs_pred,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )
    lnprob = _FitLnProb(
        model=model,
        prior=prior,
        names_samp=names_samp,
        fixed=fixed,
        names_all=names_all,
        t_obs=t_obs,
        y_obs=y_obs,
        y_err=y_err,
        predictor=predictor,
        likelihood_y_kind="flux",
        nuisance_cfgs=nuisance_cfgs,
        enforce_ni_mixing_constraint=enforce_ni_mixing_constraint,
    )

    samples, logp, meta, sampler_used = _run_sampler(
        sampler=sampler,
        lnprob=lnprob,
        prior=prior,
        sampler_kwargs=sampler_kwargs,
    )

    meta.update(
        dict(
            model=model,
            enforce_ni_mixing_constraint=bool(enforce_ni_mixing_constraint),
            y_kind="bol",
            names_all=names_all,
            bounds_all=np.asarray(bounds_all, float),
            bounds_samp=np.asarray(bounds_samp, float),
            priors_input=dict(priors or {}),
            priors_linear=dict(priors_lin or {}),
            priors_log10=dict(priors_log10 or {}),
            nuisance_priors=_likelihood_nuisance_meta(nuisance_cfgs),
            log_prior_names=sorted(
                [n for n, log_flag in zip(names_samp, log_flags_samp) if log_flag]
            ),
            likelihood=_fit_likelihood_name(nuisance_cfgs),
            interp_fill_fit=interp_fill_fit,
            model_kwargs=model_kwargs_pred,
            t_max_days_policy=tmax_meta,
            internal_t_floor=_BOL_INTERNAL_T_FLOOR,
        )
    )

    return FitResult(
        model=model,
        ctx=ctx,
        sampler=sampler_used,
        param_names=names_samp,
        fixed=fixed,
        all_param_names=names_all,
        samples=samples,
        log_prob=logp,
        meta=meta,
    )


__all__ = [
    "BolometricLC", "MultiBandLC",
    "BolometricData", "MultiBandData",
    "model_param_names", "param_template",
    "lightcurve_bol", "predict_bol",
    "lightcurve_multiband", "predict_multiband",
    "fit_bol", "fit_multiband",
]
