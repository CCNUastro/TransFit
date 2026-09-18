# transfit/models/magnetar_ni.py
# -*- coding: utf-8 -*-

import numpy as np
import numba
from transfit.modules.interp import interp_fit
from transfit.models.magnetar import (
    _magnetar_spatial_profile,
    _normalize_magnetar_heating_profile,
)

from transfit.constants import (
    PI, C_LIGHT, DAY,
    M_SUN, R_SUN,
    SIGMA_SB,
    EPSILON_NI, EPSILON_CO, TAU_NI, TAU_CO,
)


@numba.njit(fastmath=True, cache=True)
def thomas_algorithm(a, b, c_up, d, c_prime, d_prime, x_out):
    n = len(d)

    c_prime[0] = c_up[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c_up[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    x_out[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x_out[i] = d_prime[i] - c_prime[i] * x_out[i + 1]


@numba.njit(fastmath=True, cache=True)
def _fast_time_loop_numba(
    Ny, Nx, dx, dy,
    fR_vals, f_ob_vals,
    heat_mag_vals, xi_mag_vals,
    heat_ni_vals, xi_ni_vals,
    mu_const, up_const, lo_const, diag_const,
    e_initial, Lfac,
):
    e_now = e_initial.copy()
    e_next = np.empty_like(e_now)
    L_bol_out = np.zeros(Ny)

    a = np.zeros(Nx + 1)
    b_diag = np.zeros(Nx + 1)
    c_up = np.zeros(Nx + 1)
    rhs = np.zeros(Nx + 1)
    c_prime = np.zeros(Nx + 1)
    d_prime = np.zeros(Nx + 1)

    i_mid = slice(1, Nx)
    im1 = slice(0, Nx - 1)
    ip1 = slice(2, Nx + 1)
    xi_mag_inner = xi_mag_vals[1:-1]
    xi_ni_inner = xi_ni_vals[1:-1]

    for n in range(Ny):
        fR_now, fR_next = fR_vals[n], fR_vals[n + 1]

        mu_fr_next = mu_const * fR_next
        b_diag[i_mid] = 1.0 + mu_fr_next * diag_const
        c_up[i_mid] = -mu_fr_next * up_const
        a[i_mid] = -mu_fr_next * lo_const

        b_diag[0] = -1.0
        c_up[0] = 1.0
        a[0] = 0.0

        b_diag[Nx] = dx - f_ob_vals[n + 1]
        a[Nx] = f_ob_vals[n + 1]
        c_up[Nx] = 0.0

        S_now_inner = fR_now * (
            xi_mag_inner * heat_mag_vals[n]
            + xi_ni_inner * heat_ni_vals[n]
        )
        S_next_inner = fR_next * (
            xi_mag_inner * heat_mag_vals[n + 1]
            + xi_ni_inner * heat_ni_vals[n + 1]
        )

        rhs[i_mid] = (
            e_now[i_mid]
            + 0.5 * dy * (S_now_inner + S_next_inner)
            + (mu_const * fR_now) * (
                up_const * (e_now[ip1] - e_now[i_mid])
                - lo_const * (e_now[i_mid] - e_now[im1])
            )
        )
        rhs[0] = 0.0
        rhs[Nx] = 0.0

        thomas_algorithm(a, b_diag, c_up, rhs, c_prime, d_prime, e_next)
        L_bol_out[n] = Lfac * (e_next[Nx - 1] - e_next[Nx])
        e_now, e_next = e_next, e_now

    return L_bol_out


class MagNiModel:
    """
    Magnetar + Nickel model.

    theta order:
    (M_ej, v_ej, P_ms, B14, f_mag, M_ni, f_ni, kappa0, kappa_gamma, T_floor)

    Internal fixed values:
    - E_Th_in = 0
    - R_max_in = 1 R_sun
    """

    _warmup_theta = (10.0, 1.0, 3.0, 1.0, 0.2, 0.1, 0.05, 0.2, 0.03, 4000.0)
    _warmup_kwargs = {"Nx": 10, "Ny": 20}

    def __init__(self, *, warmup: bool = False):
        if warmup:
            self.warmup()

    def warmup(self, **kwargs):
        """
        Explicitly trigger a small solve to precompile the JIT path.
        """
        warmup_kwargs = dict(self._warmup_kwargs)
        warmup_kwargs.update(kwargs)
        self.calculate_light_curve(self._warmup_theta, **warmup_kwargs)
        return self

    def calculate_light_curve(
        self,
        theta,
        Nx=100,
        Ny=1000,
        t_max_days=150.0,
        heating_profile="top_hat",
    ):
        pi, c, day = PI, C_LIGHT, DAY
        eNi, eCo = EPSILON_NI, EPSILON_CO
        tau_Ni, tau_Co = TAU_NI, TAU_CO
        heating_profile = _normalize_magnetar_heating_profile(heating_profile)

        theta = tuple(theta)
        if len(theta) == 10:
            (M_ej, v_ej, P_ms, B14, f_mag, M_ni, f_ni, kappa0, kappa_gamma, T_floor) = theta
        elif len(theta) == 9:
            (M_ej, v_ej, P_ms, B14, M_ni, f_ni, kappa0, kappa_gamma, T_floor) = theta
            f_mag = 0.2
        else:
            raise ValueError("MagNiModel theta must have length 10 (or legacy length 9).")
        E_Th_in = 0.0
        R_max_in = 1.0

        M_ej = float(M_ej) * M_SUN
        E_Th_in = float(E_Th_in) * 1.0e49
        M_ni = float(M_ni)
        if not np.isfinite(M_ni) or M_ni < 0.0:
            raise ValueError("M_ni must be finite and >= 0.")
        M_ni = M_ni * M_SUN
        R_max_in = float(R_max_in) * R_SUN
        f_mag = float(f_mag)
        if not np.isfinite(f_mag) or not (0.0 <= f_mag <= 1.0):
            raise ValueError("f_mag must be finite and in [0, 1].")
        f_ni = float(f_ni)
        if not np.isfinite(f_ni) or not (0.0 <= f_ni <= 1.0):
            raise ValueError("f_ni must be finite and in [0, 1].")
        kappa0 = float(kappa0)
        kappa_g = float(kappa_gamma)
        v_ej = float(v_ej) * 1e9

        x_min, x_max = 1.0, 1.0e4
        x_heat_ni = np.clip(f_ni * x_max, x_min, x_max)

        E_K = 0.5 * M_ej * v_ej * v_ej
        I_M = (x_max**3 - x_min**3) / 3.0
        I_K = (x_max**5 - x_min**5) / 5.0

        R_min_in = R_max_in / x_max
        rho_in = M_ej / (4.0 * pi * I_M * R_min_in**3)
        v_min = np.sqrt(2.0 * I_M * E_K / (I_K * M_ej))
        t_ex = R_min_in / v_min
        t_diff = 3.0 * kappa0 * rho_in * R_min_in**2 / c
        tau_in = kappa0 * rho_in * R_min_in
        t_gamma = np.sqrt((3.0 * kappa_g * M_ej) / (4.0 * pi * v_ej * v_ej))

        I_ns = 1.0e45
        R_ns = 1.0e6

        P = float(P_ms) * 1.0e-3
        B = float(B14) * 1.0e14
        Omega = 2.0 * pi / P

        E_p = 0.5 * I_ns * Omega * Omega
        t_p = (6.0 * I_ns * c**3) / (B * B * R_ns**6 * Omega * Omega)
        eps0 = (E_p / t_p) / M_ej

        u0 = rho_in * eps0 * t_diff
        L0 = (4.0 * pi * R_min_in * c * u0) / (3.0 * kappa0 * rho_in)
        e0_coeff = E_Th_in / (2.0 * pi * u0 * x_max**2 * R_min_in**3)

        if x_heat_ni <= x_min + 1e-14:
            xi0_ni = 0.0
        else:
            denom_heat_ni = (x_heat_ni**3 - x_min**3) / 3.0
            xi0_ni = I_M / denom_heat_ni
        xi0_ni = max(xi0_ni, 0.0)

        Nx, Ny = int(Nx), int(Ny)
        x_vals = np.linspace(x_min, x_max, Nx + 1)
        dx = (x_max - x_min) / Nx
        x2 = x_vals * x_vals

        t_max = float(t_max_days) * day
        y_max = t_max / t_diff
        t_min = 1e-6 * day
        y_min = t_min / t_diff
        y_vals = np.linspace(y_min, y_max, Ny + 1)
        dy = y_vals[1] - y_vals[0]

        fR_vals = 1.0 + (y_vals * t_diff / t_ex)
        f_ob_vals = -(4.0 / (3.0 * tau_in)) * (fR_vals * fR_vals)

        t_phys = y_vals * t_diff
        dep = np.zeros_like(t_phys)
        mask = t_phys > 0.0
        dep[mask] = 1.0 - np.exp(-(t_gamma / t_phys[mask])**2)

        # Two power sources are added in the same PDE source term.
        heat_mag = dep / (1.0 + t_phys / t_p)**2
        eCo_ratio = eCo / (eNi - eCo)
        heat_ni = np.exp(-t_phys / tau_Ni) + eCo_ratio * np.exp(-t_phys / tau_Co) * dep
        ni_scale = (M_ni / M_ej) * (eNi - eCo) / eps0
        heat_ni = ni_scale * heat_ni

        xi_mag_vals = _magnetar_spatial_profile(
            x_vals,
            I_M=I_M,
            f_mag=f_mag,
            heating_profile=heating_profile,
        )

        xi_ni_vals = np.zeros_like(x_vals)
        if x_heat_ni > x_min:
            mask_heat_ni = (x_vals >= x_min) & (x_vals <= x_heat_ni)
            xi_ni_vals[mask_heat_ni] = xi0_ni

        i_mid = slice(1, Nx)
        im1 = slice(0, Nx - 1)
        ip1 = slice(2, Nx + 1)
        x_inner = x_vals[1:-1]
        inv_xdx2 = 1.0 / (x_inner * dx)**2
        mu_const = 0.25 * dy * inv_xdx2
        up_const = (x2[ip1] + x2[i_mid])
        lo_const = (x2[i_mid] + x2[im1])
        diag_const = (x2[ip1] + 2.0 * x2[i_mid] + x2[im1])

        e_initial = e0_coeff / x_vals
        Lfac = L0 * (x_max * x_max) / dx

        L_out = _fast_time_loop_numba(
            Ny, Nx, dx, dy,
            fR_vals, f_ob_vals,
            heat_mag, xi_mag_vals,
            heat_ni, xi_ni_vals,
            mu_const, up_const, lo_const, diag_const,
            e_initial, Lfac,
        )

        t_s = (y_vals * t_diff)[1:]
        R_nom = R_max_in * fR_vals[1:]

        L_pos = np.where(L_out > 0.0, L_out, 0.0)
        Teff_try = (L_pos / (4.0 * pi * R_nom * R_nom * SIGMA_SB))**0.25
        R_floor = np.sqrt(L_pos / (4.0 * pi * SIGMA_SB * (T_floor**4)))

        T_eff_values = np.where(Teff_try > T_floor, Teff_try, T_floor)
        R_outer_values = np.where(Teff_try > T_floor, R_nom, R_floor)

        return t_s, L_out, T_eff_values, R_outer_values

    def L_bol(self, t_obs, theta, z=0.0, **kwargs):
        t_s, L_series, _, _ = self.calculate_light_curve(theta, **kwargs)
        t_obs_days = np.asarray(t_obs, float)
        t_obs_grid_days = (t_s * (1.0 + z)) / DAY
        return interp_fit(
            t_obs_grid_days,
            np.asarray(L_series, float),
            t_obs_days,
            yscale="log10",
            fill="edge",
        )
