from __future__ import annotations

import numpy as np
import pytest

import transfit as tf
from transfit import api
from transfit.exceptions import NonPhysicalModelError
from transfit.models.csm import (
    CSMModel,
    PI,
    R_SUN,
    _build_y_grid,
    _deposit_shock_source,
    _fast_cooling_pde_loop,
    _fast_pde_loop,
    _integral_power_law,
    _precompute_sources,
    _rho_csm_of_x,
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
    luminosity_indices,
    luminosity_weights,
    luminosity_factors,
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
        luminosity_index = luminosity_indices[n]
        luminosity_weight = luminosity_weights[n]
        gradient = (1.0 - luminosity_weight) * (
            e_now[luminosity_index] - e_now[luminosity_index + 1]
        )
        if luminosity_weight > 0.0 and luminosity_index < n_pts - 2:
            gradient += luminosity_weight * (
                e_now[luminosity_index + 1] - e_now[luminosity_index + 2]
            )
        luminosity[n] = luminosity_factors[n] * gradient

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
    luminosity_indices = np.array([2, 1, 3], dtype=np.int64)
    luminosity_weights = np.array([0.0, 0.35, 0.0], dtype=float)
    luminosity_factors = np.array([2.5, 3.0, 4.0], dtype=float)

    expected_luminosity, expected_history = _dense_cn_reference(
        dy_steps,
        expansion,
        beta,
        source,
        coeff_left,
        coeff_right,
        dx,
        luminosity_indices,
        luminosity_weights,
        luminosity_factors,
    )
    luminosity, history, final_energy = _fast_pde_loop(
        dy_steps,
        expansion,
        beta,
        source,
        coeff_left,
        coeff_right,
        dx,
        luminosity_indices,
        luminosity_weights,
        luminosity_factors,
        True,
    )

    np.testing.assert_allclose(history, expected_history, rtol=2.0e-13, atol=2.0e-14)
    np.testing.assert_allclose(
        luminosity, expected_luminosity, rtol=2.0e-13, atol=2.0e-14
    )
    np.testing.assert_allclose(final_energy, expected_history[-1])


def test_csm_cooling_loop_applies_exact_homologous_pdv_integrating_factor():
    dy_steps = np.array([0.03, 0.05], dtype=float)
    expansion = np.array([1.0, 1.1, 1.25], dtype=float)
    beta = np.array([0.2, 0.23, 0.28], dtype=float)
    initial_energy = np.array([1.0, 1.0, 0.8, 0.4, 0.2], dtype=float)
    coeff_left = np.array([0.8, 1.0, 1.2], dtype=float)
    coeff_right = np.array([0.9, 1.1, 1.3], dtype=float)
    dx = 0.25
    luminosity_factor = 2.5

    n_times = expansion.size
    n_pts = initial_energy.size
    expected_history = np.zeros((n_times, n_pts), dtype=float)
    expected_history[0] = initial_energy
    expected_luminosity = np.zeros(n_times, dtype=float)
    expected_luminosity[0] = luminosity_factor * (
        initial_energy[-2] - initial_energy[-1]
    )
    comoving_energy = initial_energy.copy()

    half_dy_start = 0.5 * dy_steps[0]
    for substep in range(2):
        fraction = 0.5 * (substep + 1)
        expansion_sub = expansion[0] + fraction * (expansion[1] - expansion[0])
        if substep == 0:
            beta_sub = beta[0] * (expansion_sub / expansion[0]) ** 2
        else:
            beta_sub = beta[1]
        matrix = np.zeros((n_pts, n_pts), dtype=float)
        rhs = np.zeros(n_pts, dtype=float)
        matrix[0, 0] = 1.0
        matrix[0, 1] = -1.0
        for i in range(n_pts - 2):
            idx = i + 1
            left = coeff_left[i]
            right = coeff_right[i]
            matrix[idx, idx - 1] = -half_dy_start * expansion_sub * left
            matrix[idx, idx] = (
                1.0 + half_dy_start * expansion_sub * (left + right)
            )
            matrix[idx, idx + 1] = -half_dy_start * expansion_sub * right
            rhs[idx] = comoving_energy[idx]
        matrix[-1, -2] = -beta_sub / dx
        matrix[-1, -1] = 1.0 + beta_sub / dx
        comoving_energy = np.linalg.solve(matrix, rhs)
    expected_history[1] = comoving_energy / expansion[1] ** 4
    expected_luminosity[1] = luminosity_factor * (
        comoving_energy[-2] - comoving_energy[-1]
    )

    for n in range(2, n_times):
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
            matrix[idx, idx] = (
                1.0
                + half_dy * expansion[n] * (left + right)
            )
            matrix[idx, idx + 1] = -half_dy * expansion[n] * right
            old_diffusion = expansion[n - 1] * (
                left * comoving_energy[idx - 1]
                - (left + right) * comoving_energy[idx]
                + right * comoving_energy[idx + 1]
            )
            rhs[idx] = comoving_energy[idx] + half_dy * old_diffusion

        matrix[-1, -2] = -beta[n] / dx
        matrix[-1, -1] = 1.0 + beta[n] / dx
        comoving_energy = np.linalg.solve(matrix, rhs)
        expected_history[n] = comoving_energy / expansion[n] ** 4
        expected_luminosity[n] = (
            luminosity_factor
            * (comoving_energy[-2] - comoving_energy[-1])
        )

    luminosity, history = _fast_cooling_pde_loop(
        dy_steps,
        expansion,
        beta,
        initial_energy,
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


def test_csm_time_grid_remains_strict_after_scaling_to_physical_seconds():
    # This walker previously produced adjacent dimensionless values separated
    # by one ULP that rounded to the same t_s after multiplication by t_d.
    theta = (
        3.0,
        1.0,
        1.215460975116053,
        80942.26207822951,
        0.2,
        2.0,
        0.03288067294806389,
        5000.0,
    )
    result = CSMModel().calculate_light_curve(
        theta,
        Nx=100,
        Ny=1000,
        t_max_days=110.0,
        return_full=True,
        photosphere_mode="tau",
    )

    assert np.all(np.diff(result["y_grid"]) > 0.0)
    assert np.all(np.diff(result["t_s"]) > 0.0)
    observer_days = result["t_s"] * (1.0 + 0.001728) / (24.0 * 3600.0)
    assert np.all(np.diff(observer_days) > 0.0)
    assert result["t_day"][-1] == pytest.approx(result["shock_end_day"])

    prediction = tf.predict_multiband(
        model="csm",
        params={
            "M_ej": theta[0],
            "E_sn": theta[1],
            "M_csm": theta[2],
            "R_csm_out": theta[3],
            "kappa": theta[4],
            "s": theta[5],
            "eps_sh": theta[6],
            "T_floor": theta[7],
        },
        z=0.001728,
        filters={"B": "johnson_cousins.B"},
        t_days=np.array([20.0, 60.0, 90.0]),
        band=np.array(["B", "B", "B"]),
        y_kind="mag",
        mag_system="vega",
        t_max_days=110.0,
        solver_kwargs={"Nx": 100, "Ny": 1000, "photosphere_mode": "tau"},
    )
    assert np.all(np.isfinite(prediction))


@pytest.mark.parametrize(
    "kernel_kwargs",
    [
        {"shock_kernel_cells": 2, "shock_kernel_width_Rsun": 0.0},
        {"shock_kernel_cells": 5, "shock_kernel_width_Rsun": 0.0},
        {"shock_kernel_cells": 2, "shock_kernel_width_Rsun": 200.0},
    ],
)
def test_numba_shock_source_history_matches_python_reference(kernel_kwargs):
    result = CSMModel().calculate_light_curve(
        THETA,
        Nx=30,
        Ny=80,
        t_max_days=40.0,
        return_full=True,
        **kernel_kwargs,
    )
    params = result["params"]
    scales = result["scales"]
    actual = _precompute_sources(
        result["y_grid"],
        result["x_grid"],
        params,
        scales,
        result["shock_active"],
        result["x_sh"],
        result["w_sh"],
        source_x=result["shock_source_x"],
    )

    expected_source = np.zeros_like(actual["S_total"])
    expected_amplitude = np.zeros_like(actual["A_sh"])
    expected_luminosity = np.zeros_like(actual["L_sh_heat_raw"])
    for i in range(result["y_grid"].size):
        if not result["shock_active"][i]:
            continue
        expected_source[i], expected_amplitude[i] = _deposit_shock_source(
            result["x_grid"],
            float(result["x_sh"][i]),
            float(result["w_sh"][i]),
            params,
            source_x=float(result["shock_source_x"][i]),
        )
        expected_luminosity[i] = (
            params["eps_sh"]
            * 2.0
            * PI
            * (params["R_in"] * result["x_sh"][i]) ** 2
            * _rho_csm_of_x(result["x_sh"][i], params, scales)
            * (scales["v_max"] * result["w_sh"][i]) ** 3
        )

    np.testing.assert_allclose(actual["S_total"], expected_source, rtol=2.0e-14, atol=0.0)
    np.testing.assert_allclose(actual["A_sh"], expected_amplitude, rtol=2.0e-14, atol=0.0)
    np.testing.assert_allclose(
        actual["L_sh_heat_raw"], expected_luminosity, rtol=2.0e-14, atol=0.0
    )
    np.testing.assert_array_equal(actual["L_sh_heat"], actual["L_sh_heat_raw"])
    np.testing.assert_array_equal(
        actual["L_sh_heat_diffusion"], actual["L_sh_heat_raw"]
    )
    buffer_x = float(result["shock_source_buffer_x"])
    buffered = result["x_sh"] > result["x_grid"][0] + buffer_x
    np.testing.assert_allclose(
        result["x_sh"][buffered] - result["shock_source_x"][buffered],
        buffer_x,
    )


def test_reverse_shock_switch_uses_forward_shock_timing_and_deposition():
    model = CSMModel()
    default = model.calculate_light_curve(
        THETA, Nx=30, Ny=200, t_max_days=40.0, return_full=True
    )
    explicit_off = model.calculate_light_curve(
        THETA,
        Nx=30,
        Ny=200,
        t_max_days=40.0,
        return_full=True,
        reverse_shock=False,
    )
    enabled = model.calculate_light_curve(
        THETA,
        Nx=30,
        Ny=200,
        t_max_days=40.0,
        return_full=True,
        reverse_shock=True,
    )

    np.testing.assert_array_equal(default["L_bol"], explicit_off["L_bol"])
    assert not default["reverse_shock"]
    assert enabled["reverse_shock"]
    np.testing.assert_array_equal(default["x_sh"], enabled["x_sh"])
    np.testing.assert_array_equal(default["w_sh"], enabled["w_sh"])
    np.testing.assert_array_equal(default["shock_active"], enabled["shock_active"])

    active = enabled["shock_active"]
    inactive = ~active
    assert np.any(enabled["L_reverse_shock"][active] > 0.0)
    assert np.all(enabled["L_reverse_shock"][inactive] == 0.0)
    assert np.all(default["L_reverse_shock"] == 0.0)
    np.testing.assert_allclose(
        enabled["L_sh_heat_raw"],
        enabled["L_forward_shock"] + enabled["L_reverse_shock"],
        rtol=2.0e-14,
        atol=0.0,
    )

    sources = _precompute_sources(
        enabled["y_grid"],
        enabled["x_grid"],
        enabled["params"],
        enabled["scales"],
        enabled["shock_active"],
        enabled["x_sh"],
        enabled["w_sh"],
        source_x=enabled["shock_source_x"],
    )
    dx = float(enabled["x_grid"][1] - enabled["x_grid"][0])
    deposited_amplitude = (
        np.sum(enabled["x_grid"][None, :] ** 2 * sources["S_total"], axis=1)
        * dx
        / enabled["x_sh"] ** 2
    )
    np.testing.assert_allclose(
        deposited_amplitude, sources["A_sh"], rtol=2.0e-14, atol=1.0e-16
    )
    np.testing.assert_allclose(
        sources["A_sh"],
        sources["A_forward_shock"] + sources["A_reverse_shock"],
        rtol=2.0e-14,
        atol=0.0,
    )
    assert np.max(enabled["L_bol"]) > np.max(default["L_bol"])


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
    assert result["x_grid"][-1] == pytest.approx(params["x_max"])
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
    assert np.allclose(result["luminosity_x"][diffusion], scales["x_diff_max"])
    assert np.allclose(
        result["luminosity_x"][following], result["x_sh"][following]
    )
    assert np.allclose(result["luminosity_x"][cooling], params["x_max"])
    assert np.allclose(
        result["L_sh_heat_diffusion"][following],
        result["L_sh_heat_raw"][following],
    )
    diffusion_nonnegative = result["L_bol_diffusion"] >= 0.0
    assert np.array_equal(
        result["L_bol"][diffusion_nonnegative],
        result["L_bol_diffusion"][diffusion_nonnegative],
    )
    assert np.all(result["L_bol"][~diffusion_nonnegative] == 0.0)
    assert np.all(result["radiative_phase_code"][diffusion] == 0)
    assert np.all(result["radiative_phase_code"][following] == 1)
    assert np.all(result["radiative_phase_code"][cooling] == 2)
    dx = float(result["x_grid"][1] - result["x_grid"][0])
    beta = scales["beta_int"] * result["expansion_factor"] ** 2
    boundary_residual = result["e_hist"][:, -1] + beta * (
        result["e_hist"][:, -1] - result["e_hist"][:, -2]
    ) / dx
    assert np.allclose(boundary_residual, 0.0, rtol=1.0e-8, atol=1.0e-10)
    assert result["radiation_outer_boundary"]["tau"] == pytest.approx(0.0)

    assert result["breakout_matching_excess"] == 0.0
    assert result["cooling_law"] == (
        "homologous Rannacher--Crank--Nicolson diffusion with exact PdV "
        "integrating factor"
    )


def test_full_csm_grid_tracks_photosphere_shock_and_cooling_surfaces():
    theta = (5.0, 1.0, 1.0, 50_000.0, 0.2, 2.0, 1.0, 5_000.0)
    result = CSMModel().calculate_light_curve(
        theta,
        Nx=100,
        Ny=1000,
        t_max_days=150.0,
        return_full=True,
    )
    params = result["params"]
    scales = result["scales"]
    time = result["t_day"]

    assert result["x_grid"][-1] == pytest.approx(params["x_max"])
    assert np.all(result["x_sh"] <= result["x_grid"][-1] + 1.0e-12)

    photosphere_index = int(
        np.argmin(np.abs(time - result["diffusion_boundary_crossing_day"]))
    )
    exit_index = int(np.argmin(np.abs(time - result["shock_end_day"])))
    assert time[photosphere_index] == pytest.approx(
        result["diffusion_boundary_crossing_day"], abs=1.0e-12
    )
    assert result["luminosity_x"][photosphere_index] == pytest.approx(
        scales["x_diff_max"]
    )
    assert result["R_ph"][photosphere_index] == pytest.approx(scales["R_diff"])
    assert result["luminosity_x"][exit_index] == pytest.approx(params["x_max"])
    assert result["R_ph"][exit_index] == pytest.approx(params["R_csm"])
    assert result["R_ph"][exit_index + 1] > result["R_ph"][exit_index]

    # The cooling solver starts from the complete interaction-stage energy
    # profile at the exact CSM-exit event.  Its initial surface flux must
    # therefore be the same luminosity, without empirical rescaling or a
    # duplicated splice point.
    dx = float(result["x_grid"][1] - result["x_grid"][0])
    cooling_luminosity_factor = (
        scales["L0"] * params["x_max"] ** (2.0 + params["s"]) / dx
    )
    cooling_start_luminosity = cooling_luminosity_factor * (
        result["e_hist"][exit_index, -2] - result["e_hist"][exit_index, -1]
    )
    assert cooling_start_luminosity == pytest.approx(
        result["L_bol_diffusion"][exit_index], rel=2.0e-14
    )
    assert (
        abs(
            result["L_bol_diffusion"][exit_index + 1]
            / result["L_bol_diffusion"][exit_index]
            - 1.0
        )
        < 0.01
    )

    luminosity_scale = float(np.max(result["L_bol_diffusion"]))
    assert np.min(result["L_bol_diffusion"]) > -1.0e-8 * luminosity_scale
    following_luminosity = result["L_bol_diffusion"][result["shock_following_phase"]]
    relative_steps = np.abs(
        np.diff(following_luminosity) / following_luminosity[:-1]
    )
    assert np.max(relative_steps) < 0.01


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
        refined["L_bol"][refined_peak], rel=0.03
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

    reverse = tf.lightcurve_bol(
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
        solver_kwargs={
            "Nx": 20,
            "Ny": 40,
            "photosphere_mode": "outer",
            "reverse_shock": True,
        },
    )
    assert np.max(reverse.Lbol) > np.max(lc.Lbol)

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

    with pytest.raises(TypeError, match="reverse_shock"):
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
            solver_kwargs={"reverse_shock": "yes"},
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
