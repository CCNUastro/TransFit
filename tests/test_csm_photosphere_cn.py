from __future__ import annotations

import numpy as np
import pytest

import transfit as tf
from transfit import api
from transfit.exceptions import NonPhysicalModelError
from transfit.models.csm import (
    CSMModel,
    R_SUN,
    _build_y_grid,
    _fast_pde_loop,
    _integral_power_law,
)


THETA = (5.0, 1.0, 1.0, 10000.0, 0.34, 2.0, 1.0, 5000.0)


def _dense_cn_reference(
    dy_steps,
    expansion,
    beta,
    source,
    coeff_left,
    coeff_right,
    dx,
    luminosity_factor,
):
    n_times, n_pts = source.shape
    e_now = np.zeros(n_pts, dtype=float)
    history = np.zeros((n_times, n_pts), dtype=float)
    luminosity = np.zeros(n_times, dtype=float)

    for n in range(1, n_times):
        half_dy = 0.5 * dy_steps[n - 1]
        matrix = np.zeros((n_pts, n_pts), dtype=float)
        rhs = np.zeros(n_pts, dtype=float)
        matrix[0, 0] = 1.0
        matrix[0, 1] = -1.0

        for i in range(n_pts - 2):
            idx = i + 1
            left = coeff_left[i]
            right = coeff_right[i]
            matrix[idx, idx - 1] = -half_dy * expansion[n] * left
            matrix[idx, idx] = 1.0 + half_dy * expansion[n] * (left + right)
            matrix[idx, idx + 1] = -half_dy * expansion[n] * right
            old_diffusion = expansion[n - 1] * (
                left * e_now[idx - 1]
                - (left + right) * e_now[idx]
                + right * e_now[idx + 1]
            )
            rhs[idx] = e_now[idx] + half_dy * (
                old_diffusion + source[n - 1, idx] + source[n, idx]
            )

        matrix[-1, -2] = -beta[n] / dx
        matrix[-1, -1] = 1.0 + beta[n] / dx
        e_now = np.linalg.solve(matrix, rhs)
        history[n] = e_now
        luminosity[n] = luminosity_factor * (e_now[-2] - e_now[-1])

    return luminosity, history


def test_csm_numba_loop_matches_dense_crank_nicolson_reference():
    dy_steps = np.array([0.03, 0.05], dtype=float)
    expansion = np.array([1.0, 1.1, 1.25], dtype=float)
    beta = np.array([0.2, 0.23, 0.28], dtype=float)
    source = np.zeros((3, 5), dtype=float)
    source[0, 1:4] = (0.1, 0.3, 0.2)
    source[1, 1:4] = (0.4, 0.2, 0.5)
    source[2, 1:4] = (0.2, 0.6, 0.1)
    coeff_left = np.array([0.8, 1.0, 1.2], dtype=float)
    coeff_right = np.array([0.9, 1.1, 1.3], dtype=float)
    dx = 0.25
    luminosity_factor = 2.5

    expected_luminosity, expected_history = _dense_cn_reference(
        dy_steps,
        expansion,
        beta,
        source,
        coeff_left,
        coeff_right,
        dx,
        luminosity_factor,
    )
    luminosity, history = _fast_pde_loop(
        dy_steps,
        expansion,
        beta,
        source,
        coeff_left,
        coeff_right,
        dx,
        luminosity_factor,
        True,
    )

    np.testing.assert_allclose(history, expected_history, rtol=2.0e-13, atol=2.0e-14)
    np.testing.assert_allclose(
        luminosity, expected_luminosity, rtol=2.0e-13, atol=2.0e-14
    )


def test_csm_time_grid_remains_strict_when_transition_rounds_past_end():
    grid = _build_y_grid(
        1.0,
        np.nextafter(10.0, np.inf),
        10.0,
        40,
        y_extra=[3.0, np.nextafter(10.0, np.inf)],
    )
    assert grid[0] == pytest.approx(1.0)
    assert grid[-1] == pytest.approx(10.0)
    assert np.all(np.diff(grid) > 0.0)


def test_tau_photosphere_and_radiative_phases_retain_diffusion_cooling():
    result = CSMModel().calculate_light_curve(
        THETA,
        Nx=30,
        Ny=500,
        t_max_days=40.0,
        return_full=True,
    )
    scales = result["scales"]
    params = result["params"]
    tau_above = (
        params["kappa"]
        * scales["rho_0"]
        * params["R_in"]
        * _integral_power_law(
            scales["x_diff_max"], params["x_max"], -params["s"]
        )
    )
    assert tau_above == pytest.approx(2.0 / 3.0, rel=2.0e-12)
    assert result["x_grid"][-1] == pytest.approx(scales["x_diff_max"])
    assert np.all(np.diff(result["t_day"]) > 0.0)

    diffusion = result["diffusion_phase"]
    following = result["shock_following_phase"]
    cooling = result["cooling_phase"]
    assert np.any(diffusion) and np.any(following) and np.any(cooling)
    assert np.allclose(result["R_ph"][diffusion], scales["R_diff"])
    assert np.allclose(
        result["R_ph"][following],
        params["R_in"] * result["x_sh"][following],
    )
    assert np.allclose(result["R_ph"][cooling], result["R_out"][cooling])
    assert np.all(result["L_sh_heat_diffusion"][following] == 0.0)
    assert np.allclose(
        result["L_bol"][cooling], result["L_bol_diffusion"][cooling]
    )
    assert np.all(result["radiative_phase_code"][diffusion] == 0)
    assert np.all(result["radiative_phase_code"][following] == 1)
    assert np.all(result["radiative_phase_code"][cooling] == 2)
    dx = float(result["x_grid"][1] - result["x_grid"][0])
    beta = scales["beta_int"] * result["expansion_factor"] ** 2
    boundary_residual = result["e_hist"][:, -1] + beta * (
        result["e_hist"][:, -1] - result["e_hist"][:, -2]
    ) / dx
    assert np.allclose(boundary_residual, 0.0, rtol=1.0e-8, atol=1.0e-10)

    assert result["breakout_shock_luminosity"] + result[
        "breakout_matching_excess"
    ] == pytest.approx(
        result["breakout_diffusion_luminosity"], rel=1.0e-12
    )
    assert result["cooling_law"] == "source-free expanding Crank--Nicolson diffusion"


def test_csm_nx100_ny1000_is_stable_against_twofold_refinement():
    model = CSMModel()
    baseline = model.calculate_light_curve(
        THETA, Nx=100, Ny=1000, t_max_days=40.0, return_full=True
    )
    refined = model.calculate_light_curve(
        THETA, Nx=200, Ny=2000, t_max_days=40.0, return_full=True
    )
    baseline_on_refined = np.interp(
        refined["t_day"], baseline["t_day"], baseline["L_bol"]
    )
    resolved = refined["L_bol"] > 1.0e-4 * np.max(refined["L_bol"])
    relative = np.abs(
        baseline_on_refined[resolved] / refined["L_bol"][resolved] - 1.0
    )
    assert np.quantile(relative, 0.95) < 0.05

    baseline_peak = int(np.argmax(baseline["L_bol"]))
    refined_peak = int(np.argmax(refined["L_bol"]))
    assert baseline["t_day"][baseline_peak] == pytest.approx(
        refined["t_day"][refined_peak], rel=0.01
    )
    assert baseline["L_bol"][baseline_peak] == pytest.approx(
        refined["L_bol"][refined_peak], rel=0.01
    )


def test_tau_mode_rejects_optically_thin_csm_but_outer_mode_remains_available():
    thin = (4.93390298, 0.487128794, 1.94459766, 451983.422, 0.19533592,
            0.253202397, 0.796726885, 4651.12318)
    with pytest.raises(NonPhysicalModelError, match="optically thin"):
        CSMModel().calculate_light_curve(thin, Nx=20, Ny=40, t_max_days=150.0)

    t_s, luminosity, temperature, radius = CSMModel().calculate_light_curve(
        thin,
        Nx=20,
        Ny=40,
        t_max_days=150.0,
        photosphere_mode="outer",
    )
    assert np.all(np.diff(t_s) > 0.0)
    assert np.all(np.isfinite(luminosity))
    assert np.all(np.isfinite(temperature))
    assert np.all(np.isfinite(radius))


def test_public_api_accepts_csm_photosphere_mode_and_rejects_unknown_value():
    lc = tf.lightcurve_bol(
        model="csm",
        params={
            "M_ej": 5.0,
            "E_sn": 1.0,
            "M_csm": 1.0,
            "R_csm_out": 10000.0,
            "kappa": 0.34,
            "s": 2.0,
            "eps_sh": 1.0,
            "T_floor": 5000.0,
        },
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 40, "photosphere_mode": "outer"},
    )
    assert np.all(np.diff(lc.t_days) > 0.0)

    with pytest.raises(ValueError, match="photosphere_mode"):
        tf.lightcurve_bol(
            model="csm",
            params={
                "M_ej": 5.0,
                "E_sn": 1.0,
                "M_csm": 1.0,
                "R_csm_out": 10000.0,
                "kappa": 0.34,
                "s": 2.0,
                "eps_sh": 1.0,
                "T_floor": 5000.0,
            },
            t_max_days=20.0,
            solver_kwargs={"photosphere_mode": "invalid"},
        )


def test_optically_thin_csm_fit_sample_returns_minus_infinity(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == []
        sample = np.empty(0, dtype=float)
        assert lnprob(sample) == -np.inf
        return sample.reshape(1, 0), np.array([-np.inf]), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)
    fixed = {
        "M_ej": 4.93390298,
        "E_sn": 0.487128794,
        "M_csm": 1.94459766,
        "R_csm_out": 451983.422,
        "kappa": 0.19533592,
        "s": 0.253202397,
        "eps_sh": 0.796726885,
        "t_shift": 0.0,
    }
    result = tf.fit_bol(
        data=tf.BolometricData(
            t_days=np.array([10.0, 20.0]),
            y=np.array([1.0e41, 1.0e41]),
            yerr=np.array([1.0e40, 1.0e40]),
        ),
        model="csm",
        fixed=fixed,
        sampler="emcee",
        model_kwargs={
            "t_max_days": 150.0,
            "solver_kwargs": {"Nx": 20, "Ny": 40},
        },
    )
    assert result.log_prob[0] == -np.inf


def test_tau_photosphere_does_not_sample_inactive_temperature_floor():
    model_kwargs, _ = api._split_fit_model_kwargs({}, model="csm")
    fixed = api._apply_multiband_csm_photosphere_defaults(
        model="csm",
        priors_model={},
        fixed_model={},
        model_kwargs=model_kwargs,
    )
    assert fixed["T_floor"] == pytest.approx(5000.0)

    with pytest.raises(ValueError, match="T_floor is inactive"):
        api._validate_csm_photosphere_fit_configuration(
            model="csm",
            priors={"T_floor": (3000.0, 10000.0)},
            model_kwargs=model_kwargs,
        )

    outer_kwargs, _ = api._split_fit_model_kwargs(
        {"solver_kwargs": {"photosphere_mode": "outer"}}, model="csm"
    )
    api._validate_csm_photosphere_fit_configuration(
        model="csm",
        priors={"T_floor": (3000.0, 10000.0)},
        model_kwargs=outer_kwargs,
    )
