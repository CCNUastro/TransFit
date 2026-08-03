"""Scientific release gate for physical transport plus homologous SED."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transfit as tf
from transfit.constants import (
    DAY,
    EPSILON_CO,
    EPSILON_NI,
    M_SUN,
    PI,
    R_SUN,
    SIGMA_SB,
    TAU_CO,
    TAU_NI,
)
from transfit.models.nickel import (
    NickelModel,
    _PHOTOSPHERE_TAU,
    _PROFILE_Q_MAX,
    _PROFILE_Q_MIN,
    _finite_profile_scales,
    _q_profile_moment,
)


PARAMS = {
    "M_ej": 3.0,
    "v_ej": 1.0,
    "E_Th_in": 0.0,
    "M_ni": 0.08,
    "R_0": 1.0,
    "f_ni": 0.8,
    "kappa": 0.12,
    "kappa_gamma": 0.03,
    "T_floor": 4500.0,
    "delta": 0.0,
    "n": 10.0,
}
THETA = tuple(PARAMS[name] for name in tf.model_param_names("nickel"))
PROFILES = ("uniform", "broken_power_law", "exponential")
NX = 100
NY = 1000
T_MAX_DAYS = 300.0
METRICS_PATH = ROOT / "result/tables/nickel-photosphere-homologous-release-metrics.json"

THRESHOLDS = {
    "component_closure": 1.0e-13,
    "stefan_boltzmann": 1.0e-12,
    "photosphere_tau": 1.0e-5,
    "thin_heating": 1.0e-11,
    "spatial_convergence_after_5d": 5.0e-3,
    "temporal_convergence_after_10d": 5.0e-3,
    "temporal_peak_normalized_at_1p5d": 5.0e-4,
    "homologous_floor_flux_ratio": 1.0e-11,
}


def _profile_scales(profile: str):
    mass = PARAMS["M_ej"] * M_SUN
    energy = 0.5 * mass * (PARAMS["v_ej"] * 1.0e9) ** 2
    mass_moment = _q_profile_moment(
        _PROFILE_Q_MIN,
        _PROFILE_Q_MAX,
        2.0,
        profile,
        PARAMS["delta"],
        PARAMS["n"],
    )
    kinetic_moment = _q_profile_moment(
        _PROFILE_Q_MIN,
        _PROFILE_Q_MAX,
        4.0,
        profile,
        PARAMS["delta"],
        PARAMS["n"],
    )
    return _finite_profile_scales(
        mass,
        energy,
        PARAMS["R_0"] * R_SUN,
        _PROFILE_Q_MAX,
        mass_moment,
        kinetic_moment,
    )


def _deposited_heating(t_s: np.ndarray) -> np.ndarray:
    t_s = np.asarray(t_s, float)
    mass = PARAMS["M_ej"] * M_SUN
    nickel_mass = PARAMS["M_ni"] * M_SUN
    velocity = PARAMS["v_ej"] * 1.0e9
    t_gamma = np.sqrt(
        3.0 * PARAMS["kappa_gamma"] * mass / (4.0 * PI * velocity**2)
    )
    deposition = np.ones_like(t_s)
    positive = t_s > 0.0
    deposition[positive] = 1.0 - np.exp(-(t_gamma / t_s[positive]) ** 2)
    intrinsic = nickel_mass * (
        (EPSILON_NI - EPSILON_CO) * np.exp(-t_s / TAU_NI)
        + EPSILON_CO * np.exp(-t_s / TAU_CO)
    )
    return intrinsic * deposition


def _max_relative(candidate, reference, mask=None) -> float:
    candidate = np.asarray(candidate, float)
    reference = np.asarray(reference, float)
    if mask is None:
        mask = np.ones(reference.shape, dtype=bool)
    relative = np.abs(candidate[mask] / reference[mask] - 1.0)
    return float(np.max(relative)) if relative.size else 0.0


def _transport_metrics(profile: str) -> dict:
    model = NickelModel()
    state = model.calculate_transport(
        THETA,
        Nx=NX,
        Ny=NY,
        t_max_days=T_MAX_DAYS,
        density_profile=profile,
    )
    valid = state.photosphere_valid
    thin = ~valid
    closure = _max_relative(
        state.Lphotospheric + state.Ldirect,
        state.Lbol,
    )
    sb = 4.0 * PI * SIGMA_SB * state.Rph[valid] ** 2 * state.Tph[valid] ** 4
    sb_error = _max_relative(sb, state.Lphotospheric[valid])

    density_scale, _, radius_scale = _profile_scales(profile)
    tau_scale = PARAMS["kappa"] * density_scale * radius_scale
    expansion = state.Rhom / radius_scale
    tau_ph = np.array([
        tau_scale
        / expansion_i**2
        * _q_profile_moment(
            q_ph,
            _PROFILE_Q_MAX,
            0.0,
            profile,
            PARAMS["delta"],
            PARAMS["n"],
        )
        for expansion_i, q_ph in zip(expansion[valid], state.q_ph[valid])
    ])
    tau_error = float(np.max(np.abs(tau_ph / _PHOTOSPHERE_TAU - 1.0)))

    heating = _deposited_heating(state.t_s)
    thin_heating_error = max(
        _max_relative(state.Ldirect, heating, thin),
        _max_relative(state.Lbol, heating, thin),
    )
    assert np.all(state.Lphotospheric[thin] == 0.0)
    assert np.all(np.isnan(state.Rph[thin]))
    assert np.all(np.isnan(state.Tph[thin]))

    convergence_state = model.calculate_transport(
        THETA,
        Nx=NX,
        Ny=NY,
        t_max_days=150.0,
        density_profile=profile,
    )
    fine_space = model.calculate_transport(
        THETA,
        Nx=2 * NX,
        Ny=NY,
        t_max_days=150.0,
        density_profile=profile,
    )
    fine_time = model.calculate_transport(
        THETA,
        Nx=NX,
        Ny=2 * NY,
        t_max_days=150.0,
        density_profile=profile,
    )
    significant = (
        (fine_space.Lbol > 0.01 * np.max(fine_space.Lbol))
        & (fine_space.t_s >= 5.0 * DAY)
    )
    spatial_error = _max_relative(
        convergence_state.Lbol,
        fine_space.Lbol,
        significant,
    )
    early_significant = (
        (fine_space.Lbol > 0.01 * np.max(fine_space.Lbol))
        & (fine_space.t_s < 5.0 * DAY)
    )
    early_spatial_error = _max_relative(
        convergence_state.Lbol,
        fine_space.Lbol,
        early_significant,
    )
    temporal_reference = np.interp(
        convergence_state.t_s,
        fine_time.t_s,
        fine_time.Lbol,
    )
    temporal_significant = (
        (temporal_reference > 0.01 * np.max(temporal_reference))
        & (convergence_state.t_s >= 10.0 * DAY)
    )
    early_temporal_significant = (
        (temporal_reference > 0.01 * np.max(temporal_reference))
        & (convergence_state.t_s < 10.0 * DAY)
    )
    temporal_error = _max_relative(
        convergence_state.Lbol,
        temporal_reference,
        temporal_significant,
    )
    early_temporal_error = _max_relative(
        convergence_state.Lbol,
        temporal_reference,
        early_temporal_significant,
    )
    temporal_convergence_by_day = {}
    temporal_peak_normalized_by_day = {}
    temporal_peak = float(np.max(temporal_reference))
    for day_value in (1.5, 3.0, 5.0, 10.0):
        time_index = int(
            np.argmin(np.abs(convergence_state.t_s / DAY - day_value))
        )
        temporal_convergence_by_day[f"{day_value:g}"] = float(
            abs(
                convergence_state.Lbol[time_index]
                / temporal_reference[time_index]
                - 1.0
            )
        )
        temporal_peak_normalized_by_day[f"{day_value:g}"] = float(
            abs(
                convergence_state.Lbol[time_index]
                - temporal_reference[time_index]
            )
            / temporal_peak
        )

    bolometric = tf.lightcurve_bol(
        model="nickel",
        params=PARAMS,
        z=0.0,
        t_max_days=T_MAX_DAYS,
        solver_kwargs={"Nx": NX, "Ny": NY, "density_profile": profile},
    )
    multiband = tf.lightcurve_multiband(
        model="nickel",
        params=PARAMS,
        z=0.0,
        distance_modulus=0.0,
        filters={"V": "johnson_cousins.V"},
        bands=["V"],
        y_kind="flux",
        t_max_days=T_MAX_DAYS,
        solver_kwargs={"Nx": NX, "Ny": NY, "density_profile": profile},
    )
    ratio = multiband.y["V"][thin] / bolometric.Lbol[thin]
    homologous_floor_flux_ratio_error = (
        float(np.max(np.abs(ratio / ratio[0] - 1.0))) if ratio.size else 0.0
    )

    metrics = {
        "component_closure": closure,
        "stefan_boltzmann": sb_error,
        "photosphere_tau": tau_error,
        "thin_heating": thin_heating_error,
        "spatial_convergence_after_5d": spatial_error,
        "spatial_convergence_before_5d_diagnostic": early_spatial_error,
        "temporal_convergence_after_10d": temporal_error,
        "temporal_convergence_before_10d_diagnostic": early_temporal_error,
        "temporal_convergence_by_day_diagnostic": temporal_convergence_by_day,
        "temporal_peak_normalized_at_1p5d": (
            temporal_peak_normalized_by_day["1.5"]
        ),
        "temporal_peak_normalized_by_day_diagnostic": (
            temporal_peak_normalized_by_day
        ),
        "homologous_floor_flux_ratio": homologous_floor_flux_ratio_error,
        "first_thin_day": float(state.t_s[np.argmax(thin)] / DAY) if np.any(thin) else None,
        "valid_photosphere_steps": int(np.count_nonzero(valid)),
    }
    for key, threshold in THRESHOLDS.items():
        if metrics[key] > threshold:
            raise RuntimeError(
                f"{profile}: {key}={metrics[key]:.6e} exceeds {threshold:.6e}"
            )
    return metrics


def main() -> Path:
    metrics = {
        "parameters": PARAMS,
        "grid": {
            "Nx": NX,
            "Ny": NY,
            "t_max_days": T_MAX_DAYS,
            "time_grid": "quadratic_nested",
        },
        "thresholds": THRESHOLDS,
        "profiles": {profile: _transport_metrics(profile) for profile in PROFILES},
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics["profiles"], indent=2, sort_keys=True))
    print(METRICS_PATH)
    return METRICS_PATH


if __name__ == "__main__":
    main()
