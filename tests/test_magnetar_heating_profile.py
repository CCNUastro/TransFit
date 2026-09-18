from __future__ import annotations

import numpy as np
import pytest

import transfit as tf
import transfit.api as api
from transfit.models.magnetar import (
    MagnetarModel,
    _magnetar_spatial_profile,
)
from transfit.models.magnetar_ni import MagNiModel


MAGNETAR_THETA = (5.0, 1.0, 0.0, 3.0, 1.0, 0.2, 1.0, 0.1, 0.03, 4000.0)
MAGNETAR_NI_THETA = (5.0, 1.0, 3.0, 1.0, 0.2, 0.1, 0.1, 0.1, 0.03, 4000.0)

MAGNETAR_PARAMS = {
    "M_ej": 5.0,
    "v_ej": 1.0,
    "E_Th_in": 0.0,
    "P_ms": 3.0,
    "B14": 1.0,
    "f_mag": 0.2,
    "R_0": 1.0,
    "kappa": 0.1,
    "kappa_gamma": 0.03,
    "T_floor": 4000.0,
}

MAGNETAR_NI_PARAMS = {
    "M_ej": 5.0,
    "v_ej": 1.0,
    "P_ms": 3.0,
    "B14": 1.0,
    "f_mag": 0.2,
    "M_ni": 0.1,
    "f_ni": 0.1,
    "kappa": 0.1,
    "kappa_gamma": 0.03,
    "T_floor": 4000.0,
}


@pytest.mark.parametrize(
    ("engine", "theta"),
    [(MagnetarModel(), MAGNETAR_THETA), (MagNiModel(), MAGNETAR_NI_THETA)],
)
def test_default_heating_profile_is_exactly_legacy_top_hat(engine, theta):
    default = engine.calculate_light_curve(theta, Nx=30, Ny=80, t_max_days=20.0)
    explicit = engine.calculate_light_curve(
        theta,
        Nx=30,
        Ny=80,
        t_max_days=20.0,
        heating_profile="top_hat",
    )
    for actual, expected in zip(default, explicit):
        np.testing.assert_array_equal(actual, expected)


def test_arnett_kernel_is_positive_monotonic_and_volume_normalised():
    x = np.linspace(1.0, 1.0e4, 100001)
    I_M = (x[-1] ** 3 - x[0] ** 3) / 3.0
    xi = _magnetar_spatial_profile(
        x,
        I_M=I_M,
        f_mag=0.2,
        heating_profile="arnett",
    )

    assert np.all(np.isfinite(xi))
    assert np.all(xi >= 0.0)
    assert np.all(np.diff(xi) <= 0.0)
    assert xi[-1] == 0.0
    assert np.trapz(xi * x**2, x) == pytest.approx(I_M, rel=2.0e-10)


@pytest.mark.parametrize(
    ("model", "params"),
    [("magnetar", MAGNETAR_PARAMS), ("magnetar_ni", MAGNETAR_NI_PARAMS)],
)
def test_public_bolometric_api_supports_arnett_heating_profile(model, params):
    legacy = tf.lightcurve_bol(
        model=model,
        params=params,
        z=0.0,
        t_max_days=30.0,
        solver_kwargs={"Nx": 30, "Ny": 100},
    )
    arnett = tf.lightcurve_bol(
        model=model,
        params=params,
        z=0.0,
        t_max_days=30.0,
        heating_profile="arnett",
        solver_kwargs={"Nx": 30, "Ny": 100},
    )

    assert np.all(np.isfinite(arnett.Lbol))
    assert np.all(arnett.Lbol > 0.0)
    assert not np.array_equal(legacy.Lbol, arnett.Lbol)


@pytest.mark.parametrize(
    ("engine", "theta", "f_mag_index"),
    [
        (MagnetarModel(), MAGNETAR_THETA, 5),
        (MagNiModel(), MAGNETAR_NI_THETA, 4),
    ],
)
def test_f_mag_is_ignored_only_for_arnett_profile(engine, theta, f_mag_index):
    theta_a = list(theta)
    theta_b = list(theta)
    theta_a[f_mag_index] = 0.1
    theta_b[f_mag_index] = 0.8

    arnett_a = engine.calculate_light_curve(
        theta_a, Nx=30, Ny=80, t_max_days=20.0, heating_profile="arnett"
    )[1]
    arnett_b = engine.calculate_light_curve(
        theta_b, Nx=30, Ny=80, t_max_days=20.0, heating_profile="arnett"
    )[1]
    top_hat_a = engine.calculate_light_curve(
        theta_a, Nx=30, Ny=80, t_max_days=20.0, heating_profile="top_hat"
    )[1]
    top_hat_b = engine.calculate_light_curve(
        theta_b, Nx=30, Ny=80, t_max_days=20.0, heating_profile="top_hat"
    )[1]

    np.testing.assert_array_equal(arnett_a, arnett_b)
    assert not np.array_equal(top_hat_a, top_hat_b)


def test_heating_profile_validation_is_explicit():
    with pytest.raises(ValueError, match="top_hat.*arnett"):
        MagnetarModel().calculate_light_curve(
            MAGNETAR_THETA,
            Nx=20,
            Ny=40,
            t_max_days=5.0,
            heating_profile="uniform",
        )

    with pytest.raises(ValueError, match="only supported"):
        tf.lightcurve_bol(
            model="nickel",
            params={
                "M_ej": 3.0,
                "v_ej": 1.0,
                "E_Th_in": 0.0,
                "M_ni": 0.1,
                "R_0": 10.0,
                "f_ni": 0.1,
                "kappa": 0.1,
                "kappa_gamma": 0.03,
                "T_floor": 4000.0,
            },
            heating_profile="arnett",
            solver_kwargs={"Nx": 20, "Ny": 40},
        )

    nonzero_thermal = list(MAGNETAR_THETA)
    nonzero_thermal[2] = 1.0
    with pytest.raises(ValueError, match="requires E_Th_in=0"):
        MagnetarModel().calculate_light_curve(
            nonzero_thermal,
            Nx=20,
            Ny=40,
            t_max_days=5.0,
            heating_profile="arnett",
        )


def test_fit_routes_arnett_profile_and_rejects_unidentifiable_f_mag(monkeypatch):
    def fake_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert lnprob.predictor.model_kwargs_pred["heating_profile"] == "arnett"
        sample = np.empty(0, dtype=float)
        return sample.reshape(1, 0), np.array([0.0]), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_sampler)
    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0]),
        y=np.array([1.0e41, 1.1e41, 1.2e41]),
        yerr=np.full(3, 1.0e40),
    )
    fixed = {
        k: v
        for k, v in MAGNETAR_PARAMS.items()
        if k not in {"T_floor", "E_Th_in"}
    }
    fixed["t_shift"] = 0.0

    result = tf.fit_bol(
        data=data,
        model="magnetar",
        fixed=fixed,
        model_kwargs={
            "heating_profile": "arnett",
            "solver_kwargs": {"Nx": 20, "Ny": 40},
        },
    )
    assert result.meta["model_kwargs"]["heating_profile"] == "arnett"
    assert result.fixed["E_Th_in"] == 0.0

    fixed_without_f_mag = dict(fixed)
    fixed_without_f_mag.pop("f_mag")
    with pytest.raises(ValueError, match="f_mag does not affect"):
        tf.fit_bol(
            data=data,
            model="magnetar",
            fixed=fixed_without_f_mag,
            priors={"f_mag": (0.1, 0.8)},
            model_kwargs={"heating_profile": "arnett"},
        )

    with pytest.raises(ValueError, match="requires E_Th_in=0"):
        tf.fit_bol(
            data=data,
            model="magnetar",
            fixed={**fixed, "E_Th_in": 1.0},
            model_kwargs={"heating_profile": "arnett"},
        )
