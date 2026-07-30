# CSM.py
# -*- coding: utf-8 -*-

import numpy as np
import numba
from scipy.integrate import solve_ivp
from transfit.exceptions import NonPhysicalModelError

try:
    from transfit.modules.interp import interp_fit
except ImportError:
    def interp_fit(x_grid, y_grid, x_obs, yscale="linear", fill="edge"):
        x_grid = np.asarray(x_grid, dtype=float)
        y_grid = np.asarray(y_grid, dtype=float)
        x_obs = np.asarray(x_obs, dtype=float)
        if yscale == "log10":
            y_safe = np.maximum(y_grid, 1.0e-300)
            out = 10.0 ** np.interp(
                x_obs,
                x_grid,
                np.log10(y_safe),
                left=np.log10(y_safe[0]),
                right=np.log10(y_safe[-1]),
            )
        else:
            out = np.interp(x_obs, x_grid, y_grid, left=y_grid[0], right=y_grid[-1])
        return float(out) if np.ndim(x_obs) == 0 else out


# CGS constants
PI = np.pi
C_LIGHT = 2.99792458e10
DAY = 86400.0
M_SUN = 1.98847e33
R_SUN = 6.957e10
SIGMA_SB = 5.670374419e-5

DEFAULT_R_CSM_IN_RSUN = 100.0
DEFAULT_EJECTA_N = 10.0
DEFAULT_EJECTA_DELTA = 0.0


# -----------------------------------------------------------------------------
# Thin-shell dynamics and CSM shock heating
# -----------------------------------------------------------------------------

def _integral_power_law(x_lo, x_hi, power):
    if not (x_hi > x_lo > 0.0):
        raise ValueError("Integral bounds must satisfy x_hi > x_lo > 0.")
    if np.isclose(power, -1.0):
        return float(np.log(x_hi / x_lo))
    return float((x_hi ** (power + 1.0) - x_lo ** (power + 1.0)) / (power + 1.0))


def _find_photosphere_x(p, rho_0, tau_target=2.0 / 3.0):
    """Return the radius where the outward CSM optical depth is ``tau_target``."""
    tau_scale = p["kappa"] * float(rho_0) * p["R_in"]
    tau_total = tau_scale * _integral_power_law(
        1.0, p["x_max"], -p["s"]
    )
    if tau_total <= tau_target * (1.0 + 1.0e-12):
        raise NonPhysicalModelError(
            "The CSM is optically thin: total radial optical depth "
            f"{tau_total:.6g} does not enclose a diffusion region below "
            f"the photosphere depth {tau_target:.6g}."
        )

    low = 1.0
    high = float(p["x_max"])
    for _ in range(72):
        middle = 0.5 * (low + high)
        tau_above = tau_scale * _integral_power_law(
            middle, p["x_max"], -p["s"]
        )
        if tau_above > tau_target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high), float(tau_total)


def _build_params(
    theta,
    Nx,
    Ny,
    t_max_days,
    s,
    n,
    delta_inner,
    T_floor,
    rtol_ode,
    atol_ode,
    shock_max_step,
    shock_ode_solver,
    shock_rk_dx,
    shock_kernel_cells,
    shock_kernel_width_Rsun,
    photosphere_mode,
    reverse_shock,
):
    theta = tuple(theta)
    if len(theta) == 7:
        if float(theta[4]) <= 1.0:
            M_ej, E_sn, M_csm, R_csm, kappa, s, eps_sh = theta
            R_in = DEFAULT_R_CSM_IN_RSUN
        elif float(theta[3]) > float(theta[4]):
            M_ej, E_sn, M_csm, R_csm, R_in, kappa, eps_sh = theta
        else:
            M_csm, M_ej, E_sn, R_in, R_csm, kappa, eps_sh = theta
    elif len(theta) == 8:
        if float(theta[4]) <= 1.0:
            M_ej, E_sn, M_csm, R_csm, kappa, s, eps_sh, T_floor = theta
            R_in = DEFAULT_R_CSM_IN_RSUN
        elif float(theta[3]) > float(theta[4]):
            M_ej, E_sn, M_csm, R_csm, R_in, kappa, s, eps_sh = theta
        else:
            M_csm, M_ej, E_sn, R_in, R_csm, kappa, eps_sh, extra = theta
            if float(extra) > 100.0:
                T_floor = extra
            else:
                s = extra
    elif len(theta) == 9:
        if float(theta[3]) > float(theta[4]):
            M_ej, E_sn, M_csm, R_csm, R_in, kappa, s, eps_sh, T_floor = theta
        else:
            M_csm, M_ej, E_sn, R_in, R_csm, kappa, eps_sh, s, T_floor = theta
    elif len(theta) == 10:
        M_ej, E_sn, M_csm, R_csm, R_in, kappa, s, n, delta_inner, eps_sh = theta
    elif len(theta) == 11:
        M_ej, E_sn, M_csm, R_csm, R_in, kappa, s, n, delta_inner, eps_sh, T_floor = theta
    else:
        raise ValueError(
            "CSMModel theta must have length 7, 8, 9, 10, or 11. "
            "Public helpers accept named parameters. Advanced direct calls use "
            "(M_ej, E_sn, M_csm, R_csm_out, R_csm_in, kappa, s, n, "
            "delta, eps_sh[, T_floor]). "
            "Legacy shorter forms may omit s and/or T_floor."
        )

    p = {
        "M_csm": float(M_csm) * M_SUN,
        "M_ej": float(M_ej) * M_SUN,
        "E_sn": float(E_sn) * 1.0e51,
        "R_in": float(R_in) * R_SUN,
        "R_csm": float(R_csm) * R_SUN,
        "kappa": float(kappa),
        "eps_sh": float(eps_sh),
        "T_floor": float(T_floor),
        "s": float(s),
        "n": float(n),
        "delta_inner": float(delta_inner),
        "N_x": int(Nx),
        "N_t": int(Ny),
        "t_max_days": float(t_max_days),
        "rtol_ode": float(rtol_ode),
        "atol_ode": float(atol_ode),
        "shock_max_step": float(shock_max_step),
        "shock_ode_solver": str(shock_ode_solver).strip().lower(),
        "shock_rk_dx": float(shock_rk_dx),
        "shock_kernel_cells": int(shock_kernel_cells),
        "shock_kernel_width": float(shock_kernel_width_Rsun) * R_SUN,
        "photosphere_mode": str(photosphere_mode).strip().lower(),
        "reverse_shock": reverse_shock,
    }
    p["x_max"] = p["R_csm"] / p["R_in"]
    _validate_params(p)
    return p


def _validate_params(p):
    if not isinstance(p["reverse_shock"], (bool, np.bool_)):
        raise TypeError("reverse_shock must be a boolean.")
    for name, value in p.items():
        if name in {
            "N_x",
            "N_t",
            "shock_kernel_cells",
            "shock_ode_solver",
            "photosphere_mode",
            "reverse_shock",
        }:
            continue
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if not (p["x_max"] > 1.0):
        raise ValueError("R_csm must be larger than R_in.")
    if not (p["M_csm"] > 0.0 and p["M_ej"] > 0.0 and p["E_sn"] > 0.0):
        raise ValueError("M_csm, M_ej, and E_sn must be positive.")
    if not (p["kappa"] > 0.0):
        raise ValueError("kappa must be positive.")
    if not (0.0 <= p["eps_sh"] <= 1.0):
        raise ValueError("eps_sh must satisfy 0 <= eps_sh <= 1.")
    if not (0.0 <= p["s"] < 3.0):
        raise ValueError("s must satisfy 0 <= s < 3.")
    if not (p["n"] > 5.0 and p["n"] > p["s"]):
        raise ValueError("The ejecta/CSM model requires n > 5 and n > s.")
    if not (0.0 <= p["delta_inner"] < 3.0):
        raise ValueError("delta must satisfy 0 <= delta < 3.")
    if not (p["T_floor"] > 0.0):
        raise ValueError("T_floor must be positive.")
    if not (p["N_x"] >= 10 and p["N_t"] >= 10):
        raise ValueError("Nx and Ny must be at least 10.")
    if not (p["t_max_days"] > 0.0):
        raise ValueError("t_max_days must be positive.")
    if not (p["shock_max_step"] > 0.0):
        raise ValueError("shock_max_step must be positive.")
    if p["shock_ode_solver"] not in {"numba", "scipy"}:
        raise ValueError("shock_ode_solver must be 'numba' or 'scipy'.")
    if p["photosphere_mode"] not in {"outer", "tau"}:
        raise ValueError("photosphere_mode must be 'outer' or 'tau'.")
    if not (p["shock_rk_dx"] > 0.0):
        raise ValueError("shock_rk_dx must be positive.")
    if not (p["shock_kernel_cells"] >= 0):
        raise ValueError("shock_kernel_cells must be non-negative.")
    if not (p["shock_kernel_width"] >= 0.0):
        raise ValueError("shock_kernel_width_Rsun must be non-negative.")


def _compute_csm_scales(p):
    x_max = p["x_max"]
    I_M_csm = _integral_power_law(1.0, x_max, 2.0 - p["s"])
    rho_0 = p["M_csm"] / (4.0 * PI * p["R_in"] ** 3 * I_M_csm)
    tau_in = p["kappa"] * rho_0 * p["R_in"]
    tau_total = tau_in * _integral_power_law(1.0, x_max, -p["s"])
    if p["photosphere_mode"] == "tau":
        x_diff_max, tau_total = _find_photosphere_x(p, rho_0)
    else:
        x_diff_max = x_max
    R_diff = p["R_in"] * x_diff_max
    # The PDE always covers the complete CSM.  In tau mode R_diff is an
    # internal luminosity/photosphere diagnostic, not the numerical boundary.
    photosphere_tau = 2.0 / 3.0 if p["photosphere_mode"] == "tau" else 0.0
    boundary_tau = 0.0
    eddington_extrapolation_tau = boundary_tau + 2.0 / 3.0
    surface_flux_factor = 1.0 / (3.0 * eddington_extrapolation_tau)
    beta_int = eddington_extrapolation_tau / (
        tau_in * x_max ** (-p["s"])
    )
    t_d = 3.0 * p["kappa"] * rho_0 * p["R_csm"] ** 2 / C_LIGHT

    zeta = (p["n"] - 3.0) * (3.0 - p["delta_inner"])
    zeta /= 4.0 * PI * (p["n"] - p["delta_inner"])
    f_v = 2.0 * (5.0 - p["delta_inner"]) * (p["n"] - 5.0)
    f_v /= (3.0 - p["delta_inner"]) * (p["n"] - 3.0)
    v_tr = np.sqrt(f_v * p["E_sn"] / p["M_ej"])
    v_ej_mean = np.sqrt(2.0 * p["E_sn"] / p["M_ej"])
    v_max = 3.0 * v_tr

    alpha = 1.0 / (p["n"] - 3.0)
    eta_mass = p["M_csm"] / p["M_ej"]
    lambda_ = (p["n"] - 3.0) / (p["n"] - p["s"])
    t_se = eta_mass ** alpha * p["R_csm"] / v_ej_mean
    t_in = p["R_in"] / v_max
    y_in = t_in / t_d
    w_tr = v_tr / v_max
    rho_ej_tr = zeta * p["M_ej"] / p["R_in"] ** 3
    A1 = rho_ej_tr / rho_0

    u0 = t_d * rho_0 * v_max ** 3 / (2.0 * p["R_in"])
    L0 = 4.0 * PI * C_LIGHT * p["R_in"] * u0 / (3.0 * p["kappa"] * rho_0)

    return {
        "I_M_csm": I_M_csm,
        "rho_0": rho_0,
        "tau_in": tau_in,
        "tau_total": tau_total,
        "tau_photosphere": photosphere_tau,
        "boundary_tau": boundary_tau,
        "eddington_extrapolation_tau": eddington_extrapolation_tau,
        "surface_flux_factor": surface_flux_factor,
        "x_diff_max": x_diff_max,
        "R_diff": R_diff,
        "t_d": t_d,
        "zeta": zeta,
        "f_v": f_v,
        "v_tr": v_tr,
        "v_ej_mean": v_ej_mean,
        "v_max": v_max,
        "alpha": alpha,
        "eta_mass": eta_mass,
        "lambda_": lambda_,
        "t_se": t_se,
        "t_in": t_in,
        "y_in": y_in,
        "w_tr": w_tr,
        "rho_ej_tr": rho_ej_tr,
        "A1": A1,
        "u0": u0,
        "beta_int": beta_int,
        "L0": L0,
    }


def _rho_csm_of_x(x, p, scales):
    x_arr = np.asarray(x, dtype=float)
    values = scales["rho_0"] * x_arr ** (-p["s"])
    return float(values) if np.ndim(x_arr) == 0 else values


def _f_rho_ej(x, y_sh, p, scales):
    x_arr = np.asarray(x, dtype=float)
    x_break = scales["w_tr"] * y_sh
    core = x_break ** (-3.0) * (x_arr / x_break) ** (-p["delta_inner"])
    env = x_break ** (-3.0) * (x_arr / x_break) ** (-p["n"])
    values = np.where(x_arr < x_break, core, env)
    return float(values) if np.ndim(x_arr) == 0 else values


def _shock_rhs(y_sh, state, p, scales):
    x_sh, z, w = state
    z_eff = max(float(z), 1.0e-12)
    rel_speed = x_sh / y_sh - w
    f_ej = _f_rho_ej(x_sh, y_sh, p, scales)

    dx_dysh = w
    dz_dysh = scales["A1"] * x_sh ** 2 * f_ej * rel_speed + x_sh ** (2.0 - p["s"]) * w
    dw_dysh = (
        scales["A1"] * x_sh ** 2 * f_ej * rel_speed ** 2
        - x_sh ** (2.0 - p["s"]) * w ** 2
    ) / z_eff
    return np.array([dx_dysh, dz_dysh, dw_dysh], dtype=float)


@numba.njit(cache=True)
def _f_rho_ej_scalar_numba(x, y_sh, w_tr, n, delta_inner):
    y_eff = y_sh
    if y_eff < 1.0e-12:
        y_eff = 1.0e-12
    x_break = w_tr * y_eff
    if x_break <= 0.0:
        return 0.0
    base = x_break ** -3.0
    ratio = x / x_break
    if ratio < 1.0e-300:
        ratio = 1.0e-300
    if x < x_break:
        return base * ratio ** (-delta_inner)
    return base * ratio ** (-n)


@numba.njit(cache=True)
def _shock_rhs_x_numba(x, y_sh, z, w, s, n, delta_inner, w_tr, A1):
    y_eff = y_sh
    if y_eff < 1.0e-12:
        y_eff = 1.0e-12
    w_eff = w
    if w_eff < 1.0e-12:
        w_eff = 1.0e-12
    z_eff = z
    if z_eff < 1.0e-12:
        z_eff = 1.0e-12

    rel_speed = x / y_eff - w
    f_ej = _f_rho_ej_scalar_numba(x, y_eff, w_tr, n, delta_inner)
    csm_term = x ** (2.0 - s)
    ejecta_term = A1 * x * x * f_ej

    dz_dysh = ejecta_term * rel_speed + csm_term * w
    dw_dysh = (ejecta_term * rel_speed * rel_speed - csm_term * w * w) / z_eff

    dy_dx = 1.0 / w_eff
    dz_dx = dz_dysh / w_eff
    dw_dx = dw_dysh / w_eff
    return dy_dx, dz_dx, dw_dx


@numba.njit(cache=True)
def _rk4_step_shock_x_numba(x, y, z, w, h, s, n, delta_inner, w_tr, A1):
    k1_y, k1_z, k1_w = _shock_rhs_x_numba(x, y, z, w, s, n, delta_inner, w_tr, A1)
    k2_y, k2_z, k2_w = _shock_rhs_x_numba(
        x + 0.5 * h,
        y + 0.5 * h * k1_y,
        z + 0.5 * h * k1_z,
        w + 0.5 * h * k1_w,
        s,
        n,
        delta_inner,
        w_tr,
        A1,
    )
    k3_y, k3_z, k3_w = _shock_rhs_x_numba(
        x + 0.5 * h,
        y + 0.5 * h * k2_y,
        z + 0.5 * h * k2_z,
        w + 0.5 * h * k2_w,
        s,
        n,
        delta_inner,
        w_tr,
        A1,
    )
    k4_y, k4_z, k4_w = _shock_rhs_x_numba(
        x + h,
        y + h * k3_y,
        z + h * k3_z,
        w + h * k3_w,
        s,
        n,
        delta_inner,
        w_tr,
        A1,
    )

    y_next = y + (h / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    z_next = z + (h / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
    w_next = w + (h / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
    return y_next, z_next, w_next


@numba.njit(cache=True)
def _shock_step_error_norm(y_full, z_full, w_full, y_half, z_half, w_half, rtol, atol):
    err_y = abs(y_half - y_full) / (atol + rtol * max(abs(y_half), abs(y_full), 1.0))
    err_z = abs(z_half - z_full) / (atol + rtol * max(abs(z_half), abs(z_full), 1.0))
    err_w = abs(w_half - w_full) / (atol + rtol * max(abs(w_half), abs(w_full), 1.0))
    return max(err_y, err_z, err_w)


@numba.njit(cache=True)
def _shock_step_factor(err_norm):
    if err_norm <= 1.0e-16:
        return 5.0
    factor = 0.9 * err_norm ** -0.2
    if factor < 0.2:
        return 0.2
    if factor > 5.0:
        return 5.0
    return factor


@numba.njit(cache=True)
def _solve_shock_ode_numba_arrays(x_max, s, n, delta_inner, w_tr, A1, rk_dx, rtol, atol):
    x0 = 0.9999
    y0 = 1.0
    z0 = 1.0e-3
    w0 = 0.999

    base_steps = int(np.ceil((x_max - x0) / rk_dx))
    if base_steps < 1:
        base_steps = 1
    n_steps = base_steps * 8 + 1024
    if n_steps < 2048:
        n_steps = 2048
    if n_steps > 1000000:
        n_steps = 1000000

    x_values = np.empty(n_steps + 1, dtype=np.float64)
    y_values = np.empty(n_steps + 1, dtype=np.float64)
    z_values = np.empty(n_steps + 1, dtype=np.float64)
    w_values = np.empty(n_steps + 1, dtype=np.float64)

    x_values[0] = x0
    y_values[0] = y0
    z_values[0] = z0
    w_values[0] = w0

    x = x0
    y = y0
    z = z0
    w = w0

    status = 0
    used = 0
    h = min(rk_dx, max((x_max - x0) * 1.0e-4, 1.0e-5))
    min_h = max((x_max - x0) * 1.0e-14, 1.0e-12)

    while x < x_max:
        if used + 1 >= n_steps:
            status = -2
            break
        if x + h > x_max:
            h = x_max - x
        if h <= min_h:
            status = -3
            break

        y_full, z_full, w_full = _rk4_step_shock_x_numba(
            x, y, z, w, h, s, n, delta_inner, w_tr, A1
        )
        half_h = 0.5 * h
        y_mid, z_mid, w_mid = _rk4_step_shock_x_numba(
            x, y, z, w, half_h, s, n, delta_inner, w_tr, A1
        )
        y_half, z_half, w_half = _rk4_step_shock_x_numba(
            x + half_h,
            y_mid,
            z_mid,
            w_mid,
            half_h,
            s,
            n,
            delta_inner,
            w_tr,
            A1,
        )

        err_norm = _shock_step_error_norm(
            y_full, z_full, w_full, y_half, z_half, w_half, rtol, atol
        )

        if (
            not np.isfinite(err_norm)
            or not np.isfinite(y_half)
            or not np.isfinite(z_half)
            or not np.isfinite(w_half)
            or y_half <= 0.0
            or z_half <= 0.0
        ):
            h *= 0.25
            if h <= min_h:
                status = -1
                break
            continue

        factor = _shock_step_factor(err_norm)
        if err_norm > 1.0:
            h *= factor
            if h <= min_h:
                status = -3
                break
            continue

        x += h
        y = y_half
        z = z_half
        w = w_half

        used += 1
        x_values[used] = x
        y_values[used] = y
        z_values[used] = z
        w_values[used] = w

        h *= factor
        if h > rk_dx:
            h = rk_dx

    return status, used + 1, x_values, y_values, z_values, w_values


def _solve_shock_ode_numba(p):
    scales = _compute_csm_scales(p)
    status, n_used, x_values, y_values, z_values, w_values = _solve_shock_ode_numba_arrays(
        float(p["x_max"]),
        float(p["s"]),
        float(p["n"]),
        float(p["delta_inner"]),
        float(scales["w_tr"]),
        float(scales["A1"]),
        float(p["shock_rk_dx"]),
        float(p["rtol_ode"]),
        float(p["atol_ode"]),
    )
    if status < 0 or n_used < 2:
        raise RuntimeError("Numba shock ODE failed before reaching x_sh = R_csm / R_in.")

    x_values = np.asarray(x_values[:n_used], dtype=float)
    y_values = np.asarray(y_values[:n_used], dtype=float)
    z_values = np.asarray(z_values[:n_used], dtype=float)
    w_values = np.asarray(w_values[:n_used], dtype=float)

    y_sh_end = float(y_values[-1])
    y_end = float(scales["y_in"] * y_sh_end)
    t_end = float(scales["t_d"] * y_end)

    return {
        "solver": None,
        "solver_kind": "numba",
        "scales": scales,
        "x_sh_grid": x_values,
        "y_sh_grid": y_values,
        "z_sh_grid": z_values,
        "w_sh_grid": w_values,
        "y_sh_end": y_sh_end,
        "y_end": y_end,
        "t_end": t_end,
        "t_end_days": t_end / DAY,
        "x_end": float(x_values[-1]),
        "z_end": float(z_values[-1]),
        "w_end": float(w_values[-1]),
    }


def _solve_shock_ode_scipy(p):
    scales = _compute_csm_scales(p)

    def rhs(y_sh, state):
        return _shock_rhs(y_sh, state, p, scales)

    def event_xmax(y_sh, state):
        del y_sh
        return float(state[0] - p["x_max"])

    event_xmax.terminal = True
    event_xmax.direction = 1.0

    initial_state = np.array([0.9999, 1.0e-3, 0.999], dtype=float)
    sol = solve_ivp(
        rhs,
        (1.0, 1.0e5),
        initial_state,
        method="BDF",
        events=event_xmax,
        dense_output=True,
        rtol=p["rtol_ode"],
        atol=p["atol_ode"],
        max_step=p["shock_max_step"],
    )
    if sol.status < 0:
        raise RuntimeError(f"Shock ODE failed: {sol.message}")
    if sol.t_events[0].size == 0:
        raise RuntimeError("Shock ODE did not reach x_sh = R_csm / R_in.")
    if sol.sol is None:
        raise RuntimeError("Shock ODE did not return dense output.")

    y_sh_end = float(sol.t_events[0][0])
    final_state = np.asarray(sol.sol(y_sh_end), dtype=float)
    y_end = float(scales["y_in"] * y_sh_end)
    t_end = float(scales["t_d"] * y_end)

    return {
        "solver": sol,
        "solver_kind": "scipy",
        "scales": scales,
        "y_sh_end": y_sh_end,
        "y_end": y_end,
        "t_end": t_end,
        "t_end_days": t_end / DAY,
        "x_end": float(final_state[0]),
        "z_end": float(final_state[1]),
        "w_end": float(final_state[2]),
    }


def _attach_diffusion_boundary_crossing(shock, p):
    """Record when the forward shock crosses the tau-defined photosphere."""
    scales = shock["scales"]
    x_target = float(scales["x_diff_max"])
    if np.isclose(x_target, p["x_max"], rtol=1.0e-12, atol=1.0e-12):
        y_sh = float(shock["y_sh_end"])
        z_value = float(shock["z_end"])
        w_value = float(shock["w_end"])
    elif shock.get("solver_kind") == "numba":
        x_grid = np.asarray(shock["x_sh_grid"], dtype=float)
        y_grid = np.asarray(shock["y_sh_grid"], dtype=float)
        z_grid = np.asarray(shock["z_sh_grid"], dtype=float)
        w_grid = np.asarray(shock["w_sh_grid"], dtype=float)
        y_sh = float(np.interp(x_target, x_grid, y_grid))
        z_value = float(np.interp(x_target, x_grid, z_grid))
        w_value = float(np.interp(x_target, x_grid, w_grid))
    else:
        low = 1.0
        high = float(shock["y_sh_end"])
        solver = shock["solver"]
        for _ in range(72):
            middle = 0.5 * (low + high)
            x_middle = float(np.asarray(solver.sol(middle), dtype=float)[0])
            if x_middle < x_target:
                low = middle
            else:
                high = middle
        y_sh = 0.5 * (low + high)
        state = np.asarray(solver.sol(y_sh), dtype=float)
        z_value = float(state[1])
        w_value = float(state[2])

    y_diff_end = float(scales["y_in"] * y_sh)
    shock.update(
        {
            "x_diff_end": x_target,
            "y_sh_diff_end": y_sh,
            "y_diff_end": y_diff_end,
            "t_diff_end": float(scales["t_d"] * y_diff_end),
            "t_diff_end_days": float(scales["t_d"] * y_diff_end / DAY),
            "z_diff_end": z_value,
            "w_diff_end": w_value,
        }
    )
    return shock


def _solve_shock_ode(p):
    if p["shock_ode_solver"] == "scipy":
        shock = _solve_shock_ode_scipy(p)
    else:
        shock = _solve_shock_ode_numba(p)
    return _attach_diffusion_boundary_crossing(shock, p)


def _evaluate_shock_state(shock, y_sh_grid):
    if shock.get("solver_kind") == "numba":
        y_src = np.asarray(shock["y_sh_grid"], dtype=float)
        x_src = np.asarray(shock["x_sh_grid"], dtype=float)
        z_src = np.asarray(shock["z_sh_grid"], dtype=float)
        w_src = np.asarray(shock["w_sh_grid"], dtype=float)
        return np.vstack([
            np.interp(y_sh_grid, y_src, x_src),
            np.interp(y_sh_grid, y_src, z_src),
            np.interp(y_sh_grid, y_src, w_src),
        ])

    return np.asarray(shock["solver"].sol(y_sh_grid), dtype=float)


def _coalesce_scaled_time_duplicates(y_grid, time_scale, protected_points):
    """Merge y values separated by only a few physical-time ULPs.

    Event points are preferred over ordinary base-grid points so removing a
    machine-precision duplicate does not move a shock transition.
    """
    grid = np.asarray(y_grid, dtype=float)
    scale = float(time_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("CSM time-grid scale must be positive and finite.")

    scaled = grid * scale
    protected = {float(value) for value in protected_points}
    merged = [float(grid[0])]
    merged_scaled = [float(scaled[0])]
    for value, physical_time in zip(grid[1:], scaled[1:]):
        value = float(value)
        physical_time = float(physical_time)
        previous_time = merged_scaled[-1]
        separation = physical_time - previous_time
        resolution = 4.0 * max(
            abs(float(np.spacing(previous_time))),
            abs(float(np.spacing(physical_time))),
        )
        if separation > resolution:
            merged.append(value)
            merged_scaled.append(physical_time)
            continue
        if separation < 0.0:
            raise RuntimeError("Internal CSM physical time grid is decreasing.")
        if value in protected and merged[-1] not in protected:
            merged[-1] = value
            merged_scaled[-1] = physical_time

    out = np.asarray(merged, dtype=float)
    if out.size < 2 or np.any(np.diff(out * scale) <= 0.0):
        raise RuntimeError("Internal CSM physical time grid is not strictly increasing.")
    return out


def _build_y_grid(
    y_start,
    y_end,
    y_max,
    n_steps,
    y_extra=None,
    *,
    time_scale=1.0,
):
    """Build a strictly increasing grid with resolved physical transitions."""
    y_start = float(y_start)
    y_end = float(y_end)
    y_max = float(y_max)
    n_steps = int(n_steps)
    if not (np.isfinite(y_start) and np.isfinite(y_end) and np.isfinite(y_max)):
        raise ValueError("CSM time-grid boundaries must be finite.")
    if not (y_max > y_start):
        raise NonPhysicalModelError(
            "CSM time grid requires the final time to be later than the start time."
        )
    if n_steps < 1:
        raise ValueError("CSM time grid requires at least one step.")

    y_base = np.linspace(y_start, y_max, n_steps + 1)
    dy0 = float((y_max - y_start) / n_steps)
    boundaries = [float(np.clip(y_end, y_start, y_max))]
    if y_extra is not None:
        boundaries.extend(
            np.clip(np.asarray(y_extra, dtype=float).reshape(-1), y_start, y_max).tolist()
        )

    pieces = [y_base, np.array([y_start, y_max], dtype=float)]
    # Resolve both sides of every radiative transition geometrically.  The
    # state and flux are continuous when the shock source switches off, but
    # the luminosity derivative changes abruptly and the outermost diffusion
    # modes relax on a timescale much shorter than the global Ny spacing.
    # Returning those early post-event points prevents that resolved transient
    # from being drawn as a single discontinuous-looking jump.
    positive_event_offsets = 2.0 ** np.arange(-12, 2, dtype=float)
    event_offset_fractions = np.concatenate(
        (-positive_event_offsets[::-1], np.array([0.0]), positive_event_offsets)
    )
    offsets = dy0 * event_offset_fractions
    for boundary in boundaries:
        pieces.append(np.clip(boundary + offsets, y_start, y_max))
        pieces.append(np.array([boundary], dtype=float))

    y_grid = np.unique(np.concatenate(pieces))
    y_grid = _coalesce_scaled_time_duplicates(
        y_grid,
        time_scale,
        protected_points=[y_start, y_max, *boundaries],
    )
    if y_grid.size < 2 or np.any(np.diff(y_grid) <= 0.0):
        raise RuntimeError("Internal CSM time grid is not strictly increasing.")
    return y_grid


def _build_expansion_profile(y_grid, shock, p, scales):
    y_end = float(shock["y_end"])
    y_sh_end = float(shock["y_sh_end"])
    w_end = max(float(shock["w_end"]), 1.0e-10)
    v_sh_end = scales["v_max"] * w_end
    t_sc = p["R_csm"] / v_sh_end
    cooling_rate = scales["t_d"] / t_sc

    expansion_factor = np.ones_like(y_grid, dtype=float)
    after_end = y_grid > y_end
    expansion_factor[after_end] = 1.0 + cooling_rate * (y_grid[after_end] - y_end)
    surface_beta = scales["beta_int"] * expansion_factor ** 2
    shock_active = y_grid <= (y_end + 1.0e-14)

    y_sh_grid = np.clip(y_grid / scales["y_in"], 1.0, y_sh_end)
    dense_state = _evaluate_shock_state(shock, y_sh_grid)
    x_sh = np.asarray(dense_state[0], dtype=float)
    z_sh = np.asarray(dense_state[1], dtype=float)
    w_sh = np.maximum(np.asarray(dense_state[2], dtype=float), 0.0)
    x_sh[~shock_active] = p["x_max"]
    z_sh[~shock_active] = float(shock["z_end"])
    w_sh[~shock_active] = w_end

    return (
        expansion_factor,
        surface_beta,
        shock_active,
        x_sh,
        z_sh,
        w_sh,
    )


def _deposit_shock_source(x_grid, x_sh, w, p, source_x=None):
    dx = float(x_grid[1] - x_grid[0])
    source = np.zeros_like(x_grid, dtype=float)
    n_cells = x_grid.size - 1
    A_sh = p["eps_sh"] * x_sh ** (-p["s"]) * max(float(w), 0.0) ** 3
    deposit_x = float(x_sh if source_x is None else source_x)
    if A_sh == 0.0:
        return source, A_sh

    if p["shock_kernel_width"] > 0.0:
        kernel_width_x = p["shock_kernel_width"] / p["R_in"]
        kernel_cells = max(3, int(np.ceil(6.0 * kernel_width_x / dx)))
        use_delta_kernel = False
    else:
        kernel_width_x = 0.0
        kernel_cells = int(p["shock_kernel_cells"])
        use_delta_kernel = kernel_cells <= 2

    if use_delta_kernel:
        if deposit_x <= x_grid[0]:
            source[1] = A_sh / dx
        elif deposit_x >= x_grid[-1]:
            source[n_cells - 1] = A_sh / dx
        else:
            j = int(np.searchsorted(x_grid, deposit_x, side="right") - 1)
            j = int(np.clip(j, 0, n_cells - 1))
            theta = (deposit_x - x_grid[j]) / dx
            source[j] += A_sh * (1.0 - theta) / dx
            source[j + 1] += A_sh * theta / dx
            if j == 0:
                source[1] += source[0]
                source[0] = 0.0
            if j + 1 == n_cells:
                source[n_cells - 1] += source[n_cells]
                source[n_cells] = 0.0
    elif n_cells <= 2:
        source[1] = A_sh / dx
    else:
        valid_start = 1
        valid_stop = n_cells - 1
        width = min(kernel_cells, valid_stop - valid_start + 1)
        center = int(
            np.argmin(np.abs(x_grid[valid_start : valid_stop + 1] - deposit_x))
        ) + valid_start
        left = center - width // 2
        right = left + width - 1
        if left < valid_start:
            right += valid_start - left
            left = valid_start
        if right > valid_stop:
            left -= right - valid_stop
            right = valid_stop
        indices = np.arange(left, right + 1)
        sigma_x = max(kernel_width_x, 0.5 * float(kernel_cells) * dx, 1.0e-300)
        weights = np.exp(-0.5 * ((x_grid[indices] - deposit_x) / sigma_x) ** 2)
        weight_norm = float(np.sum(x_grid[indices] ** 2 * weights) * dx)
        if weight_norm <= 0.0 or not np.isfinite(weight_norm):
            source[center] = A_sh / dx
        else:
            source[indices] = A_sh * x_sh ** 2 * weights / weight_norm

    physical_integral = float(np.sum(x_grid ** 2 * source) * dx / max(x_sh ** 2, 1.0e-300))
    if physical_integral > 0.0 and np.isfinite(physical_integral):
        source *= A_sh / physical_integral
    return source, A_sh


@numba.njit(cache=True)
def _searchsorted_right_numba(values, target):
    low = 0
    high = values.size
    while low < high:
        middle = (low + high) // 2
        if target < values[middle]:
            high = middle
        else:
            low = middle + 1
    return low


@numba.njit(cache=True)
def _precompute_sources_numba(
    x_grid,
    shock_active,
    x_sh,
    y_sh,
    source_x,
    w_sh,
    eps_sh,
    s,
    n,
    delta_inner,
    w_tr,
    A1,
    reverse_shock,
    R_in,
    rho_0,
    v_max,
    shock_kernel_cells,
    kernel_width_x,
):
    """Build the complete shock source history without Python time loops."""
    n_times = x_sh.size
    n_points = x_grid.size
    n_cells = n_points - 1
    dx = x_grid[1] - x_grid[0]

    shock_source = np.zeros((n_times, n_points), dtype=np.float64)
    A_sh = np.zeros(n_times, dtype=np.float64)
    A_forward = np.zeros(n_times, dtype=np.float64)
    A_reverse = np.zeros(n_times, dtype=np.float64)
    L_sh_heat_raw = np.zeros(n_times, dtype=np.float64)
    L_forward_shock_raw = np.zeros(n_times, dtype=np.float64)
    L_reverse_shock_raw = np.zeros(n_times, dtype=np.float64)

    if kernel_width_x > 0.0:
        kernel_cells = max(3, int(np.ceil(6.0 * kernel_width_x / dx)))
        use_delta_kernel = False
    else:
        kernel_cells = shock_kernel_cells
        use_delta_kernel = kernel_cells <= 2

    for i in range(n_times):
        if not shock_active[i]:
            continue

        x_value = x_sh[i]
        y_value = max(y_sh[i], 1.0e-12)
        deposit_value = source_x[i]
        w_value = max(w_sh[i], 0.0)
        forward_amplitude = eps_sh * x_value ** (-s) * w_value ** 3
        reverse_amplitude = 0.0
        if reverse_shock:
            ejecta_speed = x_value / y_value
            relative_speed = max(ejecta_speed - w_value, 0.0)
            ejecta_density = _f_rho_ej_scalar_numba(
                x_value, y_value, w_tr, n, delta_inner
            )
            reverse_amplitude = (
                eps_sh * A1 * ejecta_density * relative_speed ** 3
            )
        amplitude = forward_amplitude + reverse_amplitude
        A_sh[i] = amplitude
        A_forward[i] = forward_amplitude
        A_reverse[i] = reverse_amplitude
        dimensional_factor = (
            2.0 * PI * (R_in * x_value) ** 2 * rho_0 * v_max ** 3
        )
        L_forward_shock_raw[i] = dimensional_factor * forward_amplitude
        L_reverse_shock_raw[i] = dimensional_factor * reverse_amplitude
        L_sh_heat_raw[i] = dimensional_factor * amplitude
        if amplitude == 0.0:
            continue

        if use_delta_kernel:
            if deposit_value <= x_grid[0]:
                shock_source[i, 1] = amplitude / dx
            elif deposit_value >= x_grid[n_points - 1]:
                shock_source[i, n_cells - 1] = amplitude / dx
            else:
                j = _searchsorted_right_numba(x_grid, deposit_value) - 1
                if j < 0:
                    j = 0
                elif j > n_cells - 1:
                    j = n_cells - 1
                theta = (deposit_value - x_grid[j]) / dx
                shock_source[i, j] += amplitude * (1.0 - theta) / dx
                shock_source[i, j + 1] += amplitude * theta / dx
                if j == 0:
                    shock_source[i, 1] += shock_source[i, 0]
                    shock_source[i, 0] = 0.0
                if j + 1 == n_cells:
                    shock_source[i, n_cells - 1] += shock_source[i, n_cells]
                    shock_source[i, n_cells] = 0.0
        elif n_cells <= 2:
            shock_source[i, 1] = amplitude / dx
        else:
            valid_start = 1
            valid_stop = n_cells - 1
            width = min(kernel_cells, valid_stop - valid_start + 1)

            center = valid_start
            center_distance = abs(x_grid[center] - deposit_value)
            for j in range(valid_start + 1, valid_stop + 1):
                distance = abs(x_grid[j] - deposit_value)
                if distance < center_distance:
                    center = j
                    center_distance = distance

            left = center - width // 2
            right = left + width - 1
            if left < valid_start:
                right += valid_start - left
                left = valid_start
            if right > valid_stop:
                left -= right - valid_stop
                right = valid_stop

            sigma_x = max(kernel_width_x, 0.5 * float(kernel_cells) * dx, 1.0e-300)
            weight_norm = 0.0
            for j in range(left, right + 1):
                weight = np.exp(
                    -0.5 * ((x_grid[j] - deposit_value) / sigma_x) ** 2
                )
                weight_norm += x_grid[j] ** 2 * weight * dx

            if weight_norm <= 0.0 or not np.isfinite(weight_norm):
                shock_source[i, center] = amplitude / dx
            else:
                for j in range(left, right + 1):
                    weight = np.exp(
                        -0.5 * ((x_grid[j] - deposit_value) / sigma_x) ** 2
                    )
                    shock_source[i, j] = amplitude * x_value ** 2 * weight / weight_norm

        physical_integral = 0.0
        for j in range(n_points):
            physical_integral += x_grid[j] ** 2 * shock_source[i, j]
        physical_integral *= dx / max(x_value ** 2, 1.0e-300)
        if physical_integral > 0.0 and np.isfinite(physical_integral):
            factor = amplitude / physical_integral
            for j in range(n_points):
                shock_source[i, j] *= factor

    return (
        shock_source,
        A_sh,
        A_forward,
        A_reverse,
        L_sh_heat_raw,
        L_forward_shock_raw,
        L_reverse_shock_raw,
    )


def _precompute_sources(
    y_grid,
    x_grid,
    p,
    scales,
    shock_active,
    x_sh,
    w_sh,
    source_x=None,
):
    n_times = np.asarray(y_grid).size
    if not (
        np.asarray(shock_active).size == n_times
        and np.asarray(x_sh).size == n_times
        and np.asarray(w_sh).size == n_times
    ):
        raise ValueError("CSM shock source histories must share the time-grid length.")
    if source_x is None:
        source_x_values = np.asarray(x_sh, dtype=float)
    else:
        source_x_values = np.asarray(source_x, dtype=float)
        if source_x_values.size != n_times:
            raise ValueError("CSM shock source positions must share the time-grid length.")
    kernel_width_x = p["shock_kernel_width"] / p["R_in"]
    y_sh_values = np.asarray(y_grid, dtype=float) / float(scales["y_in"])
    (
        shock_source,
        A_sh,
        A_forward,
        A_reverse,
        L_sh_heat_raw,
        L_forward_shock_raw,
        L_reverse_shock_raw,
    ) = _precompute_sources_numba(
        np.asarray(x_grid, dtype=float),
        np.asarray(shock_active, dtype=np.bool_),
        np.asarray(x_sh, dtype=float),
        y_sh_values,
        source_x_values,
        np.asarray(w_sh, dtype=float),
        float(p["eps_sh"]),
        float(p["s"]),
        float(p["n"]),
        float(p["delta_inner"]),
        float(scales["w_tr"]),
        float(scales["A1"]),
        bool(p["reverse_shock"]),
        float(p["R_in"]),
        float(scales["rho_0"]),
        float(scales["v_max"]),
        int(p["shock_kernel_cells"]),
        float(kernel_width_x),
    )

    return {
        "S_total": shock_source,
        "A_sh": A_sh,
        "A_forward_shock": A_forward,
        "A_reverse_shock": A_reverse,
        "L_sh_heat": L_sh_heat_raw.copy(),
        "L_sh_heat_raw": L_sh_heat_raw,
        "L_sh_heat_diffusion": L_sh_heat_raw.copy(),
        "L_forward_shock": L_forward_shock_raw,
        "L_reverse_shock": L_reverse_shock_raw,
    }


# -----------------------------------------------------------------------------
# Crank--Nicolson diffusion loop (Numba JIT)
# -----------------------------------------------------------------------------

@numba.njit(fastmath=True, cache=True)
def thomas_algorithm(a, b, c_up, d, c_prime, d_prime, x_out):
    n = len(d)
    c_prime[0] = c_up[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = c_up[i] / denom
        else:
            c_prime[i] = 0.0
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    x_out[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x_out[i] = d_prime[i] - c_prime[i] * x_out[i + 1]


@numba.njit(fastmath=True, cache=True)
def _fast_pde_loop(
    dy_steps,
    expansion_vals,
    beta_vals,
    total_source,
    coeff_imh_base,
    coeff_iph_base,
    dx,
    luminosity_indices,
    luminosity_weights,
    luminosity_factors,
    store_history,
):
    n_times, n_pts = total_source.shape
    n_inner = n_pts - 2

    e_now = np.zeros(n_pts, dtype=np.float64)
    e_next = np.zeros(n_pts, dtype=np.float64)
    L_bol = np.zeros(n_times, dtype=np.float64)

    if store_history:
        e_hist = np.zeros((n_times, n_pts), dtype=np.float64)
        e_hist[0, :] = e_now
    else:
        e_hist = np.zeros((1, 1), dtype=np.float64)

    a = np.zeros(n_pts, dtype=np.float64)
    b_diag = np.zeros(n_pts, dtype=np.float64)
    c_up = np.zeros(n_pts, dtype=np.float64)
    rhs = np.zeros(n_pts, dtype=np.float64)
    c_prime = np.zeros(n_pts, dtype=np.float64)
    d_prime = np.zeros(n_pts, dtype=np.float64)

    for n in range(1, n_times):
        dy = dy_steps[n - 1]
        half_dy = 0.5 * dy
        expansion_old = expansion_vals[n - 1]
        expansion_new = expansion_vals[n]
        beta = beta_vals[n]

        a[:] = 0.0
        b_diag[:] = 0.0
        c_up[:] = 0.0
        rhs[:] = 0.0

        b_diag[0] = 1.0
        c_up[0] = -1.0

        for i in range(n_inner):
            idx = i + 1
            coeff_left = coeff_imh_base[i]
            coeff_right = coeff_iph_base[i]
            lower = -half_dy * expansion_new * coeff_left
            upper = -half_dy * expansion_new * coeff_right
            a[idx] = lower
            b_diag[idx] = 1.0 - lower - upper
            c_up[idx] = upper
            diffusion_old = expansion_old * (
                coeff_left * e_now[idx - 1]
                - (coeff_left + coeff_right) * e_now[idx]
                + coeff_right * e_now[idx + 1]
            )
            source_avg = total_source[n - 1, idx] + total_source[n, idx]
            rhs[idx] = e_now[idx] + half_dy * (diffusion_old + source_avg)

        a[n_pts - 1] = -beta / dx
        b_diag[n_pts - 1] = 1.0 + beta / dx

        thomas_algorithm(a, b_diag, c_up, rhs, c_prime, d_prime, e_next)
        luminosity_index = luminosity_indices[n]
        luminosity_weight = luminosity_weights[n]
        gradient = (1.0 - luminosity_weight) * (
            e_next[luminosity_index] - e_next[luminosity_index + 1]
        )
        if luminosity_weight > 0.0 and luminosity_index < n_pts - 2:
            gradient += luminosity_weight * (
                e_next[luminosity_index + 1]
                - e_next[luminosity_index + 2]
            )
        L_bol[n] = luminosity_factors[n] * gradient
        if store_history:
            e_hist[n, :] = e_next

        e_now, e_next = e_next, e_now

    return L_bol, e_hist, e_now.copy()


@numba.njit(fastmath=True, cache=True)
def _fast_cooling_pde_loop(
    dy_steps,
    expansion_vals,
    beta_vals,
    initial_energy,
    coeff_imh_base,
    coeff_iph_base,
    dx,
    Lfac,
    store_history,
):
    """Solve post-CSM diffusion with the homologous PdV loss integrated exactly.

    In the comoving coordinate the dimensionless radiation energy density
    ``e = E_rad/u0`` obeys

        de/dy = a D[e] - 4 (d ln(a)/dy) e,

    where ``y=t/t_d`` and ``a = R/R_breakout``.  Evolving the dimensionless
    integrating-factor variable ``q = a**4 e`` removes only the analytically
    known PdV term and gives

        dq/dy = a D[q].

    This retains the adiabatic loss exactly and avoids loss of accuracy after
    ``e`` has fallen by many orders of magnitude.  A two-half-step Rannacher
    start damps the source-switch discontinuity before standard CN stepping.
    Stored histories are converted back to ``e``; dimensional energy density
    is ``u0*e``.
    """
    n_times = expansion_vals.size
    n_pts = initial_energy.size
    n_inner = n_pts - 2

    u_now = initial_energy.copy()
    u_next = np.zeros(n_pts, dtype=np.float64)
    L_bol = np.zeros(n_times, dtype=np.float64)
    L_bol[0] = Lfac * (u_now[n_pts - 2] - u_now[n_pts - 1])

    if store_history:
        e_hist = np.zeros((n_times, n_pts), dtype=np.float64)
        e_hist[0, :] = u_now
    else:
        e_hist = np.zeros((1, 1), dtype=np.float64)

    a = np.zeros(n_pts, dtype=np.float64)
    b_diag = np.zeros(n_pts, dtype=np.float64)
    c_up = np.zeros(n_pts, dtype=np.float64)
    rhs = np.zeros(n_pts, dtype=np.float64)
    c_prime = np.zeros(n_pts, dtype=np.float64)
    d_prime = np.zeros(n_pts, dtype=np.float64)

    # The shock source switches off discontinuously at CSM exit.  Two
    # backward-Euler half steps damp the unresolved high-frequency modes in
    # that hand-off; subsequent steps retain second-order CN accuracy.
    if n_times > 1:
        dy_start = dy_steps[0]
        half_dy_start = 0.5 * dy_start
        expansion_start = expansion_vals[0]
        expansion_finish = expansion_vals[1]
        beta_start = beta_vals[0]
        beta_finish = beta_vals[1]
        for substep in range(2):
            fraction = 0.5 * (substep + 1)
            expansion_sub = expansion_start + fraction * (
                expansion_finish - expansion_start
            )
            if substep == 0:
                beta_sub = beta_start * (expansion_sub / expansion_start) ** 2
            else:
                beta_sub = beta_finish

            a[:] = 0.0
            b_diag[:] = 0.0
            c_up[:] = 0.0
            rhs[:] = 0.0
            b_diag[0] = 1.0
            c_up[0] = -1.0

            for i in range(n_inner):
                idx = i + 1
                coeff_left = coeff_imh_base[i]
                coeff_right = coeff_iph_base[i]
                lower = -half_dy_start * expansion_sub * coeff_left
                upper = -half_dy_start * expansion_sub * coeff_right
                a[idx] = lower
                b_diag[idx] = 1.0 - lower - upper
                c_up[idx] = upper
                rhs[idx] = u_now[idx]

            a[n_pts - 1] = -beta_sub / dx
            b_diag[n_pts - 1] = 1.0 + beta_sub / dx
            thomas_algorithm(a, b_diag, c_up, rhs, c_prime, d_prime, u_next)
            u_now, u_next = u_next, u_now

        L_bol[1] = Lfac * (u_now[n_pts - 2] - u_now[n_pts - 1])
        if store_history:
            e_hist[1, :] = u_now / expansion_finish ** 4

    for n in range(2, n_times):
        dy = dy_steps[n - 1]
        half_dy = 0.5 * dy
        expansion_old = expansion_vals[n - 1]
        expansion_new = expansion_vals[n]
        beta = beta_vals[n]

        a[:] = 0.0
        b_diag[:] = 0.0
        c_up[:] = 0.0
        rhs[:] = 0.0

        b_diag[0] = 1.0
        c_up[0] = -1.0

        for i in range(n_inner):
            idx = i + 1
            coeff_left = coeff_imh_base[i]
            coeff_right = coeff_iph_base[i]
            lower = -half_dy * expansion_new * coeff_left
            upper = -half_dy * expansion_new * coeff_right
            a[idx] = lower
            b_diag[idx] = 1.0 - lower - upper
            c_up[idx] = upper
            diffusion_old = expansion_old * (
                coeff_left * u_now[idx - 1]
                - (coeff_left + coeff_right) * u_now[idx]
                + coeff_right * u_now[idx + 1]
            )
            rhs[idx] = u_now[idx] + half_dy * diffusion_old

        a[n_pts - 1] = -beta / dx
        b_diag[n_pts - 1] = 1.0 + beta / dx

        thomas_algorithm(a, b_diag, c_up, rhs, c_prime, d_prime, u_next)
        L_bol[n] = Lfac * (u_next[n_pts - 2] - u_next[n_pts - 1])
        if store_history:
            e_hist[n, :] = u_next / expansion_new ** 4

        u_now, u_next = u_next, u_now

    return L_bol, e_hist


def _compose_radiative_phases(t_s, L_diffusion, L_sh_heat, shock):
    """Classify radius phases without altering the diffusion luminosity."""
    t_s = np.asarray(t_s, dtype=float)
    L_diffusion = np.asarray(L_diffusion, dtype=float)
    L_sh_heat = np.asarray(L_sh_heat, dtype=float)

    t_ph = float(shock["t_diff_end"])
    t_out = float(shock["t_end"])
    diffusion_phase = t_s <= (t_ph + 1.0e-10)
    shock_following_phase = (t_s > t_ph) & (t_s <= (t_out + 1.0e-10))
    cooling_phase = t_s > t_out

    L_diffusion_nonnegative = np.maximum(L_diffusion, 0.0)
    L_sh_heat_nonnegative = np.maximum(L_sh_heat, 0.0)
    L_emergent = L_diffusion_nonnegative.copy()

    L_diffusion_at_breakout = float(np.interp(t_ph, t_s, L_diffusion_nonnegative))
    L_sh_heat_at_breakout = float(np.interp(t_ph, t_s, L_sh_heat_nonnegative))
    # Crossing the tau=2/3 surface changes only the emitting radius.  Shock
    # heating remains in the interaction equation until the forward shock
    # exits the CSM.  The post-CSM luminosity is supplied by the separate
    # source-free cooling equation with homologous PdV losses.
    diffusion_tail_weight = np.ones(t_s.size, dtype=float)

    phase_code = np.zeros(t_s.size, dtype=np.int8)
    phase_code[shock_following_phase] = 1
    phase_code[cooling_phase] = 2
    return {
        "L_bol": L_emergent,
        "phase_code": phase_code,
        "diffusion_phase": diffusion_phase,
        "shock_following_phase": shock_following_phase,
        "cooling_phase": cooling_phase,
        "diffusion_tail_weight": diffusion_tail_weight,
        "breakout_diffusion_luminosity": L_diffusion_at_breakout,
        "breakout_shock_luminosity": L_sh_heat_at_breakout,
        "breakout_matching_excess": 0.0,
        "cooling_law": "homologous Rannacher--Crank--Nicolson diffusion with exact PdV integrating factor",
    }


def _photosphere(L_bol, expansion_factor, x_sh, radiative_phase_code, p, scales):
    R_out = p["R_csm"] * expansion_factor
    L_pos = np.maximum(np.asarray(L_bol, dtype=float), 1.0e-300)

    if p["photosphere_mode"] == "tau":
        phase_code = np.asarray(radiative_phase_code, dtype=np.int8)
        R_ph = np.full(L_pos.shape, scales["R_diff"], dtype=float)
        shock_following = phase_code == 1
        cooling = phase_code == 2
        R_ph[shock_following] = (
            p["R_in"] * np.asarray(x_sh, dtype=float)[shock_following]
        )
        # After the forward shock exits the CSM, the shocked shell cools while
        # expanding homologously.  Its emitting radius follows the expanding
        # CSM outer radius; the luminosity still comes from source-free diffusion.
        R_ph[cooling] = R_out[cooling]
        T_eff = (
            L_pos
            / (4.0 * PI * SIGMA_SB * np.maximum(R_ph, 1.0e-300) ** 2)
        ) ** 0.25
        return T_eff, R_ph, R_out

    T_surface = (L_pos / (4.0 * PI * SIGMA_SB * np.maximum(R_out, 1.0e-300) ** 2)) ** 0.25
    R_floor = np.sqrt(L_pos / (4.0 * PI * SIGMA_SB * p["T_floor"] ** 4))

    floor_active = T_surface < p["T_floor"]
    R_ph = R_out.copy()
    R_ph[floor_active] = R_floor[floor_active]
    T_eff = (L_pos / (4.0 * PI * SIGMA_SB * np.maximum(R_ph, 1.0e-300) ** 2)) ** 0.25
    return T_eff, R_ph, R_out


class CSMModel:
    """
    Pure CSM-interaction light-curve model.

    Public API parameter order:
    (M_ej, E_sn, M_csm, R_csm_out, kappa, s, n, delta, eps_sh, T_floor)

    The public API inserts the internal R_csm_in value before calling this
    engine. Advanced direct calls use:
    (M_ej, E_sn, M_csm, R_csm_out, R_csm_in, kappa, s, n, delta, eps_sh,
    T_floor)

    Units:
    - M_csm, M_ej: solar mass
    - E_sn: 1e51 erg
    - R_csm_out: solar radius
    - kappa: cm^2 g^-1
    - s: CSM density power-law index
    - eps_sh: dimensionless

    Public API defaults:
    - R_csm_in = 100 R_sun
    - n = 10 and delta = 0 when omitted in forward calls
    - n = 10 and delta = 0 are fixed in fits unless explicit priors are supplied

    Legacy shorter theta:
    (M_csm, M_ej, E_sn_51, R_in, R_csm, kappa, eps_sh)
    (M_csm, M_ej, E_sn_51, R_in, R_csm, kappa, eps_sh, s)
    """

    _warmup_theta = (5.0, 1.0, 1.0, 10000.0, 0.34, 2.0, 1.0, 5000.0)
    _warmup_kwargs = {"Nx": 20, "Ny": 40, "t_max_days": 20.0}

    def __init__(self, *, warmup=False):
        if warmup:
            self.warmup()

    def warmup(self, **kwargs):
        warmup_kwargs = dict(self._warmup_kwargs)
        warmup_kwargs.update(kwargs)
        self.calculate_light_curve(self._warmup_theta, **warmup_kwargs)
        return self

    def _params_from_theta(
        self,
        theta,
        Nx,
        Ny,
        t_max_days,
        s=2.0,
        n=DEFAULT_EJECTA_N,
        delta_inner=DEFAULT_EJECTA_DELTA,
        T_floor=5000.0,
        rtol_ode=1.0e-6,
        atol_ode=1.0e-8,
        shock_max_step=0.8,
        shock_ode_solver="numba",
        shock_rk_dx=0.05,
        shock_kernel_cells=2,
        shock_kernel_width_Rsun=0.0,
        photosphere_mode="tau",
        reverse_shock=False,
    ):
        return _build_params(
            theta,
            Nx,
            Ny,
            t_max_days,
            s,
            n,
            delta_inner,
            T_floor,
            rtol_ode,
            atol_ode,
            shock_max_step,
            shock_ode_solver,
            shock_rk_dx,
            shock_kernel_cells,
            shock_kernel_width_Rsun,
            photosphere_mode,
            reverse_shock,
        )

    def calculate_light_curve(self, theta, Nx=140, Ny=1000, t_max_days=150.0, return_full=False, **kwargs):
        p = self._params_from_theta(theta, Nx, Ny, t_max_days, **kwargs)
        shock = _solve_shock_ode(p)
        scales = shock["scales"]

        y_max = max(p["t_max_days"] * DAY / scales["t_d"], shock["y_end"])
        x_grid = np.linspace(1.0, p["x_max"], p["N_x"] + 1, dtype=float)
        y_grid = _build_y_grid(
            scales["y_in"],
            shock["y_end"],
            y_max,
            p["N_t"],
            y_extra=[shock["y_diff_end"]],
            time_scale=scales["t_d"],
        )
        t_s = y_grid * scales["t_d"]
        dx = float(x_grid[1] - x_grid[0])
        dy_steps = np.diff(y_grid).astype(float)

        (
            expansion_factor,
            surface_beta,
            shock_active,
            x_sh,
            z_sh,
            w_sh,
        ) = _build_expansion_profile(y_grid, shock, p, scales)

        # Deposit the forward and optional reverse-shock powers at the same
        # thin-shell source position, one Nx=100 reference cell downstream of
        # the forward-shock surface.  Both use the forward-shock activity mask;
        # keeping the buffer at 1% of the full CSM width makes it
        # resolution-independent (one cell for Nx=100, two for Nx=200).
        shock_source_buffer_x = (p["x_max"] - 1.0) / 100.0
        shock_source_x = np.maximum(
            x_sh - shock_source_buffer_x, x_grid[0]
        )

        sources = _precompute_sources(
            y_grid,
            x_grid,
            p,
            scales,
            shock_active,
            x_sh,
            w_sh,
            source_x=shock_source_x,
        )

        if p["photosphere_mode"] == "tau":
            luminosity_x = np.where(
                y_grid <= (shock["y_diff_end"] + 1.0e-14),
                scales["x_diff_max"],
                x_sh,
            )
        else:
            luminosity_x = np.full(y_grid.shape, p["x_max"], dtype=float)
        luminosity_x = np.clip(luminosity_x, 1.0, p["x_max"])
        # Use the same fractional position as the two-node shock deposition to
        # interpolate the two outward cell-face fluxes.  When the shock crosses
        # a node, the old second face becomes the new first face continuously.
        luminosity_indices = (
            np.searchsorted(x_grid, luminosity_x, side="right") - 1
        ).astype(np.int64)
        luminosity_indices = np.clip(
            luminosity_indices, 0, x_grid.size - 2
        )
        luminosity_weights = (
            luminosity_x - x_grid[luminosity_indices]
        ) / dx
        luminosity_weights = np.clip(luminosity_weights, 0.0, 1.0)
        at_outer_edge = luminosity_indices >= x_grid.size - 2
        luminosity_weights[at_outer_edge] = 0.0
        luminosity_factors = (
            scales["L0"] * luminosity_x ** (2.0 + p["s"]) / dx
        )

        x_inner = x_grid[1:-1]
        xi_sq = (1.0 / p["x_max"]) ** 2
        dx_sq = dx ** 2
        coeff_imh_base = (x_inner - 0.5 * dx) ** (2.0 + p["s"]) / (xi_sq * x_inner ** 2 * dx_sq)
        coeff_iph_base = (x_inner + 0.5 * dx) ** (2.0 + p["s"]) / (xi_sq * x_inner ** 2 * dx_sq)
        Lfac_cooling = scales["L0"] * p["x_max"] ** (2.0 + p["s"]) / dx

        shock_end_index = int(np.searchsorted(y_grid, shock["y_end"]))
        if not np.isclose(
            y_grid[shock_end_index], shock["y_end"], rtol=0.0, atol=0.0
        ):
            raise RuntimeError("CSM time grid does not contain the shock exit event.")

        # Stage 1: interaction diffusion.  The CSM is stationary and shock
        # heating remains active through the exact outer-boundary event.
        interaction_stop = shock_end_index + 1
        L_interaction, e_hist_interaction, e_at_exit = _fast_pde_loop(
            dy_steps[:shock_end_index],
            np.ones(interaction_stop, dtype=float),
            np.full(interaction_stop, scales["beta_int"], dtype=float),
            sources["S_total"][:interaction_stop].astype(float),
            coeff_imh_base.astype(float),
            coeff_iph_base.astype(float),
            dx,
            luminosity_indices[:interaction_stop],
            luminosity_weights[:interaction_stop].astype(float),
            luminosity_factors[:interaction_stop].astype(float),
            bool(return_full),
        )

        # Stage 2: after the forward shock exits the CSM, pass the complete
        # radiation-energy profile into a source-free homologous cooling solve.
        # The exact integrating factor q=a**4 e applies the homologous PdV loss
        # without sacrificing late-time numerical accuracy.
        if shock_end_index < y_grid.size - 1:
            cooling_expansion = expansion_factor[shock_end_index:].astype(float)
            cooling_beta = surface_beta[shock_end_index:].astype(float)
            L_cooling, e_hist_cooling = _fast_cooling_pde_loop(
                dy_steps[shock_end_index:],
                cooling_expansion,
                cooling_beta,
                e_at_exit,
                coeff_imh_base.astype(float),
                coeff_iph_base.astype(float),
                dx,
                Lfac_cooling,
                bool(return_full),
            )
            L_bol_diffusion = np.concatenate(
                (L_interaction, L_cooling[1:])
            )
            if bool(return_full):
                e_hist = np.concatenate(
                    (e_hist_interaction, e_hist_cooling[1:]), axis=0
                )
            else:
                e_hist = e_hist_interaction
        else:
            L_bol_diffusion = L_interaction
            e_hist = e_hist_interaction
        if p["photosphere_mode"] == "tau":
            radiative_phases = _compose_radiative_phases(
                t_s,
                L_bol_diffusion,
                sources["L_sh_heat_raw"],
                shock,
            )
        else:
            n_times = t_s.size
            radiative_phases = {
                "L_bol": L_bol_diffusion,
                "phase_code": np.zeros(n_times, dtype=np.int8),
                "diffusion_phase": np.ones(n_times, dtype=bool),
                "shock_following_phase": np.zeros(n_times, dtype=bool),
                "cooling_phase": np.zeros(n_times, dtype=bool),
                "diffusion_tail_weight": np.ones(n_times, dtype=float),
                "breakout_diffusion_luminosity": float(
                    np.interp(shock["t_end"], t_s, L_bol_diffusion)
                ),
                "breakout_shock_luminosity": float(
                    np.interp(shock["t_end"], t_s, sources["L_sh_heat_raw"])
                ),
                "breakout_matching_excess": 0.0,
                "cooling_law": "legacy outer-boundary diffusion",
            }
        L_bol = radiative_phases["L_bol"]
        T_eff, R_ph, R_out = _photosphere(
            L_bol,
            expansion_factor,
            x_sh,
            radiative_phases["phase_code"],
            p,
            scales,
        )

        out = slice(1, None)
        t_s_out = t_s[out]
        y_grid_out = y_grid[out]
        L_bol_out = L_bol[out]
        L_bol_diffusion_out = L_bol_diffusion[out]
        L_sh_heat_out = sources["L_sh_heat"][out]
        L_sh_heat_raw_out = sources["L_sh_heat_raw"][out]
        L_sh_heat_diffusion_out = sources["L_sh_heat_diffusion"][out]
        L_forward_shock_out = sources["L_forward_shock"][out]
        L_reverse_shock_out = sources["L_reverse_shock"][out]
        T_eff_out = T_eff[out]
        R_ph_out = R_ph[out]
        R_out_out = R_out[out]
        x_sh_out = x_sh[out]
        z_sh_out = z_sh[out]
        w_sh_out = w_sh[out]
        shock_source_x_out = shock_source_x[out]
        shock_active_out = shock_active[out]
        diffusion_source_active_out = shock_active_out.copy()
        expansion_factor_out = expansion_factor[out]
        luminosity_x_out = luminosity_x[out]
        e_hist_out = e_hist[out] if bool(return_full) else e_hist

        if return_full:
            return {
                "theta": tuple(theta),
                "params": p,
                "scales": scales,
                "t_s": t_s_out,
                "t_day": t_s_out / DAY,
                "x_grid": x_grid,
                "y_grid": y_grid_out,
                "L_bol": L_bol_out,
                "L_bol_diffusion": L_bol_diffusion_out,
                "L_sh_heat": L_sh_heat_out,
                "L_sh_heat_raw": L_sh_heat_raw_out,
                "L_sh_heat_diffusion": L_sh_heat_diffusion_out,
                "L_forward_shock": L_forward_shock_out,
                "L_reverse_shock": L_reverse_shock_out,
                "reverse_shock": bool(p["reverse_shock"]),
                "T_eff": T_eff_out,
                "R_ph": R_ph_out,
                "R_out": R_out_out,
                "x_sh": x_sh_out,
                "z_sh": z_sh_out,
                "w_sh": w_sh_out,
                "shock_source_x": shock_source_x_out,
                "shock_source_buffer_x": float(shock_source_buffer_x),
                "v_sh": scales["v_max"] * w_sh_out,
                "shock_active": shock_active_out,
                "diffusion_source_active": diffusion_source_active_out,
                "radiative_phase_code": radiative_phases["phase_code"][out],
                "diffusion_phase": radiative_phases["diffusion_phase"][out],
                "shock_following_phase": radiative_phases[
                    "shock_following_phase"
                ][out],
                "cooling_phase": radiative_phases["cooling_phase"][out],
                "diffusion_tail_weight": radiative_phases[
                    "diffusion_tail_weight"
                ][out],
                "breakout_diffusion_luminosity": radiative_phases[
                    "breakout_diffusion_luminosity"
                ],
                "breakout_shock_luminosity": radiative_phases[
                    "breakout_shock_luminosity"
                ],
                "breakout_matching_excess": radiative_phases[
                    "breakout_matching_excess"
                ],
                "cooling_law": radiative_phases["cooling_law"],
                "shock_end_day": shock["t_end_days"],
                "diffusion_boundary_crossing_day": shock["t_diff_end_days"],
                "radiation_outer_boundary": {
                    "type": "grey_eddington_robin",
                    "tau": scales["boundary_tau"],
                    "flux_over_cE": scales["surface_flux_factor"],
                    "beta_initial": scales["beta_int"],
                },
                "expansion_factor": expansion_factor_out,
                "luminosity_x": luminosity_x_out,
                "e_hist": e_hist_out,
            }

        return t_s_out, L_bol_out, T_eff_out, R_ph_out

    def calculate_shock_diagnostics(
        self, theta, Nx=140, Ny=1000, t_max_days=150.0, **kwargs
    ):
        """Return shock and optical-depth diagnostics without solving diffusion."""
        p = self._params_from_theta(theta, Nx, Ny, t_max_days, **kwargs)
        shock = _solve_shock_ode(p)
        scales = shock["scales"]
        return {
            "tau_total": float(scales["tau_total"]),
            "x_ph": float(scales["x_diff_max"]),
            "R_ph_Rsun": float(scales["R_diff"] / R_SUN),
            "R_csm_out_Rsun": float(p["R_csm"] / R_SUN),
            "diffusion_boundary_crossing_day": float(shock["t_diff_end_days"]),
            "shock_end_day": float(shock["t_end_days"]),
            "reverse_shock": bool(p["reverse_shock"]),
            "v_ej_mean_km_s": float(scales["v_ej_mean"] / 1.0e5),
        }

    def calculate_dynamics(self, theta, Nx=140, Ny=1000, t_max_days=150.0, **kwargs):
        result = self.calculate_light_curve(theta, Nx=Nx, Ny=Ny, t_max_days=t_max_days, return_full=True, **kwargs)
        return result["t_s"], result["v_sh"], result["L_sh_heat"]

    def L_bol(self, t_obs, theta, z=0.0, **kwargs):
        t_s, L_series, _, _ = self.calculate_light_curve(theta, **kwargs)
        t_obs_days = np.asarray(t_obs, dtype=float)
        t_obs_grid_days = (t_s * (1.0 + z)) / DAY
        return interp_fit(
            t_obs_grid_days,
            np.maximum(np.asarray(L_series, dtype=float), 1.0e-300),
            t_obs_days,
            yscale="log10",
            fill="edge",
        )
