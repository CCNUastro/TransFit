# transfit/models/nickel.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np
import numba  # Numba JIT
from transfit.modules.interp import interp_fit

# unified constants (CGS, Numba-friendly)
from transfit.constants import (
    PI, C_LIGHT, DAY,
    M_SUN, R_SUN,
    SIGMA_SB,
    EPSILON_NI, EPSILON_CO, TAU_NI, TAU_CO,
)

# Finite ejecta domains.  R_0 is always the initial outer radius; these
# dimensionless cutoffs set the density scale radius inside that boundary.
_BPL_V_MAX_OVER_V_T = 3.0
_BPL_DEFAULT_DELTA = 0.0
_BPL_DEFAULT_N = 10.0
_EXPONENTIAL_V_MAX_OVER_V_E = 12.0

# Common normalized radius/velocity grid used by every density profile.
# Profile-specific scales are encoded only in eta(q): q_t=v_t/v_max for BPL
# and q_e=v_e/v_max for the exponential profile.
_PROFILE_Q_MIN = 1.0e-4
_PROFILE_Q_MAX = 1.0
_BPL_Q_T = 1.0 / _BPL_V_MAX_OVER_V_T
_EXPONENTIAL_Q_E = 1.0 / _EXPONENTIAL_V_MAX_OVER_V_E
_PHOTOSPHERE_TAU = 2.0 / 3.0
_TIME_GRID_POWER = 2.0


@dataclass(frozen=True)
class NickelTransportState:
    """Solved bolometric transport state for the Nickel model.

    Uniform density preserves the historical fixed-outer-boundary solve and
    homologous effective blackbody.  BPL and exponential density use the
    physical ``tau=2/3`` diffusion boundary; power deposited outside that
    domain is carried by ``Ldirect`` and never enters the diffusion matrix.
    """

    t_s: np.ndarray
    Lbol: np.ndarray
    Lphotospheric: np.ndarray
    Ldirect: np.ndarray
    q_ph: np.ndarray
    Rph: np.ndarray
    Tph: np.ndarray
    Rhom: np.ndarray
    photosphere_valid: np.ndarray
    density_profile: str

# -----------------------------------------------------------------------------
# Core solver functions (Numba JIT)
# -----------------------------------------------------------------------------


def _gamma_deposition_fraction(t_phys, t_gamma):
    """Return ``1 - exp[-(t_gamma/t)^2]`` with its ``t=0`` limit."""
    t_phys = np.asarray(t_phys, dtype=float)
    deposition = np.ones_like(t_phys)
    positive_time = t_phys > 0.0
    deposition[positive_time] = 1.0 - np.exp(
        -(float(t_gamma) / t_phys[positive_time]) ** 2
    )
    return deposition


def _radioactive_heating_shape(t_phys, t_gamma):
    """Return deposited Ni-to-Co heating normalized by ``epsilon_Ni-epsilon_Co``."""
    t_phys = np.asarray(t_phys, dtype=float)
    e_co_ratio = EPSILON_CO / (EPSILON_NI - EPSILON_CO)
    radioactive = (
        np.exp(-t_phys / TAU_NI)
        + e_co_ratio * np.exp(-t_phys / TAU_CO)
    )
    return radioactive * _gamma_deposition_fraction(t_phys, t_gamma)


def _build_time_grid(y_max, Ny):
    """Return the fixed early-time-refined dimensionless transport grid.

    Nickel diffusion rises most rapidly during the first few days, especially
    for centrally concentrated BPL and exponential ejecta.  A quadratic grid
    resolves that phase without adding a fit parameter or changing the public
    meaning of ``Ny``.  Grids with ``2*Ny`` contain every node of the ``Ny``
    grid, which makes temporal convergence checks unambiguous.
    """
    y_max = float(y_max)
    Ny = int(Ny)
    if not np.isfinite(y_max) or y_max <= 0.0:
        raise ValueError("Nickel time-grid extent must be finite and positive.")
    if Ny < 1:
        raise ValueError("Nickel time grids require Ny >= 1.")

    fraction = np.linspace(0.0, 1.0, Ny + 1)
    return y_max * fraction**_TIME_GRID_POWER


def _integral_power_law(x_lo, x_hi, power):
    """Return ``integral(x**power dx, x_lo, x_hi)`` for positive bounds."""
    if not (np.isfinite(x_lo) and np.isfinite(x_hi) and 0.0 < x_lo < x_hi):
        raise ValueError("Power-law integration bounds must satisfy 0 < x_lo < x_hi.")

    exponent = float(power) + 1.0
    if abs(exponent) < 1.0e-12:
        return float(np.log(x_hi / x_lo))
    return float((x_hi**exponent - x_lo**exponent) / exponent)


@numba.njit(cache=True)
def _integral_power_law_scalar_numba(x_lo, x_hi, power):
    """Numba-friendly power-law integral for the cut-cell kernels."""
    exponent = power + 1.0
    if abs(exponent) < 1.0e-12:
        return np.log(x_hi / x_lo)
    return (x_hi**exponent - x_lo**exponent) / exponent


@numba.njit(cache=True)
def _q_profile_moment_scalar_numba(
    q_min,
    q_max,
    order,
    density_profile,
    delta,
    n,
):
    """Scalar profile moment used by the Numba cut-cell geometry kernel."""
    if density_profile == "uniform":
        return _integral_power_law_scalar_numba(q_min, q_max, order)

    if density_profile == "exponential":
        q_e = _EXPONENTIAL_Q_E
        x_min = q_min / q_e
        x_max = q_max / q_e
        # Geometry only requests the order-two mass moment.
        tail_min = np.exp(-x_min) * (x_min * x_min + 2.0 * x_min + 2.0)
        tail_max = np.exp(-x_max) * (x_max * x_max + 2.0 * x_max + 2.0)
        return q_e**3 * (tail_min - tail_max)

    q_t = _BPL_Q_T
    if q_max <= q_t:
        return q_t**delta * _integral_power_law_scalar_numba(
            q_min, q_max, order - delta
        )
    if q_min >= q_t:
        return q_t**n * _integral_power_law_scalar_numba(
            q_min, q_max, order - n
        )
    return (
        q_t**delta * _integral_power_law_scalar_numba(
            q_min, q_t, order - delta
        )
        + q_t**n * _integral_power_law_scalar_numba(
            q_t, q_max, order - n
        )
    )


def _legacy_uniform_ni_source_profile(x_vals, x_heat, xi0):
    """Return the exact historical control-volume Uniform Ni source."""
    x_vals = np.asarray(x_vals, dtype=float)
    source_profile = np.zeros_like(x_vals)
    if xi0 <= 0.0 or x_heat <= x_vals[0] or x_vals.size < 3:
        return source_profile

    x_inner = x_vals[1:-1]
    edges = np.empty(x_inner.size + 1, dtype=float)
    edges[0] = x_vals[0]
    edges[-1] = x_vals[-1]
    if x_inner.size > 1:
        edges[1:-1] = 0.5 * (x_inner[:-1] + x_inner[1:])

    lower = edges[:-1]
    upper = edges[1:]
    mixed_upper = np.minimum(upper, float(x_heat))
    active = mixed_upper > lower
    mixed_mass = np.zeros_like(lower)
    mixed_mass[active] = (
        mixed_upper[active] ** 3 - lower[active] ** 3
    ) / 3.0
    cell_volume = (upper**3 - lower**3) / 3.0
    source_profile[1:-1] = float(xi0) * mixed_mass / cell_volume
    return source_profile


def _exponential_tail_moment(x, order):
    """Return ``integral(x**order * exp(-x) dx, x, infinity)``.

    Only non-negative integer orders are needed here, so the upper incomplete
    gamma function has this exact finite-series form and requires no SciPy.
    This is only an analytic antiderivative: physical normalizations always
    subtract its values at the two finite ejecta boundaries.
    """
    x = float(x)
    order = int(order)
    if not np.isfinite(x) or x < 0.0:
        raise ValueError("Exponential-profile coordinates must be finite and non-negative.")
    if order < 0:
        raise ValueError("Exponential-profile moment order must be non-negative.")

    term = 1.0
    series = 1.0
    factorial = 1.0
    for k in range(1, order + 1):
        term *= x / float(k)
        series += term
        factorial *= float(k)
    return float(factorial * np.exp(-x) * series)


def _eta_q(q, density_profile, delta, n):
    """Evaluate a density profile on the common ``q=v/v_max=r/R_out`` grid."""
    q = np.asarray(q, dtype=float)
    if density_profile == "uniform":
        return np.ones_like(q)
    if density_profile == "exponential":
        return np.exp(-q / _EXPONENTIAL_Q_E)

    scaled = q / _BPL_Q_T
    return np.where(
        q < _BPL_Q_T,
        scaled ** (-float(delta)),
        scaled ** (-float(n)),
    )


def _q_profile_moment(q_min, q_max, order, density_profile, delta, n):
    """Return ``integral(q**order * eta(q) dq)`` on the common grid."""
    q_min = float(q_min)
    q_max = float(q_max)
    order = float(order)
    if not (
        np.isfinite(q_min)
        and np.isfinite(q_max)
        and 0.0 < q_min < q_max <= 1.0
    ):
        raise ValueError("Common profile bounds must satisfy 0 < q_min < q_max <= 1.")

    if density_profile == "uniform":
        return _integral_power_law(q_min, q_max, order)

    if density_profile == "exponential":
        q_e = _EXPONENTIAL_Q_E
        x_min = q_min / q_e
        x_max = q_max / q_e
        return float(
            q_e ** (order + 1.0)
            * (
                _exponential_tail_moment(x_min, int(order))
                - _exponential_tail_moment(x_max, int(order))
            )
        )

    if not (np.isfinite(delta) and 0.0 <= delta < 3.0):
        raise ValueError("delta must be finite and satisfy 0 <= delta < 3.")
    if not (np.isfinite(n) and n > 5.0):
        raise ValueError("n must be finite and > 5.")

    q_t = _BPL_Q_T
    if q_max <= q_t:
        return float(q_t**delta * _integral_power_law(q_min, q_max, order - delta))
    if q_min >= q_t:
        return float(q_t**n * _integral_power_law(q_min, q_max, order - n))
    return float(
        q_t**delta * _integral_power_law(q_min, q_t, order - delta)
        + q_t**n * _integral_power_law(q_t, q_max, order - n)
    )


def _q_enclosed_profile_mass(q, q_min, density_profile, delta, n):
    """Vectorized dimensionless mass enclosed between ``q_min`` and ``q``."""
    q = np.asarray(q, dtype=float)
    q_min = float(q_min)

    if density_profile == "uniform":
        return (q**3 - q_min**3) / 3.0

    if density_profile == "exponential":
        q_e = _EXPONENTIAL_Q_E
        x_min = q_min / q_e
        x = q / q_e
        tail_min = np.exp(-x_min) * (x_min * x_min + 2.0 * x_min + 2.0)
        tail_x = np.exp(-x) * (x * x + 2.0 * x + 2.0)
        return q_e**3 * (tail_min - tail_x)

    q_t = _BPL_Q_T
    inner_exponent = 3.0 - float(delta)
    outer_exponent = 3.0 - float(n)
    inner_mass = q_t**delta * (
        q_t**inner_exponent - q_min**inner_exponent
    ) / inner_exponent
    enclosed_inner = q_t**delta * (
        q**inner_exponent - q_min**inner_exponent
    ) / inner_exponent
    enclosed_outer = inner_mass + q_t**n * (
        q**outer_exponent - q_t**outer_exponent
    ) / outer_exponent
    return np.where(q <= q_t, enclosed_inner, enclosed_outer)


@numba.njit(cache=True)
def _q_enclosed_profile_mass_scalar_numba(q, q_min, density_profile, delta, n):
    """Scalar enclosed profile mass for the Numba cut-cell geometry kernel."""
    if density_profile == "uniform":
        return (np.power(q, 3.0) - np.power(q_min, 3.0)) / 3.0

    if density_profile == "exponential":
        q_e = _EXPONENTIAL_Q_E
        x_min = q_min / q_e
        x = q / q_e
        tail_min = np.exp(-x_min) * (x_min * x_min + 2.0 * x_min + 2.0)
        tail_x = np.exp(-x) * (x * x + 2.0 * x + 2.0)
        return q_e**3 * (tail_min - tail_x)

    q_t = _BPL_Q_T
    inner_exponent = 3.0 - delta
    outer_exponent = 3.0 - n
    inner_mass = q_t**delta * (
        q_t**inner_exponent - q_min**inner_exponent
    ) / inner_exponent
    enclosed_inner = q_t**delta * (
        q**inner_exponent - q_min**inner_exponent
    ) / inner_exponent
    enclosed_outer = inner_mass + q_t**n * (
        q**outer_exponent - q_t**outer_exponent
    ) / outer_exponent
    if q <= q_t:
        return enclosed_inner
    return enclosed_outer


def _q_mass_fraction_to_radius(f_mass, q_min, density_profile, delta, n):
    """Invert the enclosed mass fraction on the common normalized grid."""
    f_mass = float(f_mass)
    if not np.isfinite(f_mass) or not (0.0 <= f_mass <= 1.0):
        raise ValueError("Mass fraction must be finite and in [0, 1].")
    if f_mass <= 0.0:
        return float(q_min)
    if f_mass >= 1.0:
        return _PROFILE_Q_MAX

    total_mass = _q_profile_moment(
        q_min, _PROFILE_Q_MAX, 2.0, density_profile, delta, n
    )
    target_mass = f_mass * total_mass
    lo = float(q_min)
    hi = _PROFILE_Q_MAX
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        enclosed = float(
            _q_enclosed_profile_mass(mid, q_min, density_profile, delta, n)
        )
        if enclosed < target_mass:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _q_photosphere_radius(
    expansion_factor,
    tau_scale,
    density_profile,
    delta,
    n,
    tau_target=_PHOTOSPHERE_TAU,
):
    """Return the ``tau=tau_target`` radius on the common ``q`` grid.

    Homologous expansion gives

    ``tau(q, t) = tau_scale/f_R(t)**2 * integral_q^1 eta(q') dq'``.

    The inversion is analytic for every built-in profile.  Once the complete
    ejecta column is below ``tau_target``, the returned coordinate is clamped
    to the inner represented boundary, signalling a fully nebular ejecta.
    """
    expansion_factor = np.asarray(expansion_factor, dtype=float)
    if np.any(~np.isfinite(expansion_factor)) or np.any(expansion_factor <= 0.0):
        raise ValueError("Expansion factors must be finite and positive.")
    tau_scale = float(tau_scale)
    tau_target = float(tau_target)
    if not np.isfinite(tau_scale) or tau_scale <= 0.0:
        raise ValueError("Optical-depth scale must be finite and positive.")
    if not np.isfinite(tau_target) or tau_target <= 0.0:
        raise ValueError("Photosphere optical depth must be finite and positive.")

    column_target = tau_target * expansion_factor**2 / tau_scale

    if density_profile == "uniform":
        q_ph = _PROFILE_Q_MAX - column_target
    elif density_profile == "exponential":
        edge_term = np.exp(-_PROFILE_Q_MAX / _EXPONENTIAL_Q_E)
        argument = column_target / _EXPONENTIAL_Q_E + edge_term
        q_ph = -_EXPONENTIAL_Q_E * np.log(argument)
    else:
        q_t = _BPL_Q_T
        delta = float(delta)
        n = float(n)
        outer_column = q_t**n * (
            q_t ** (1.0 - n) - _PROFILE_Q_MAX ** (1.0 - n)
        ) / (n - 1.0)
        in_outer = column_target <= outer_column

        outer_power = (
            _PROFILE_Q_MAX ** (1.0 - n)
            + (n - 1.0) * column_target / q_t**n
        )
        q_outer = outer_power ** (1.0 / (1.0 - n))

        inner_column = np.maximum(column_target - outer_column, 0.0)
        if abs(1.0 - delta) < 1.0e-12:
            q_inner = q_t * np.exp(-inner_column / q_t)
        else:
            inner_power = (
                q_t ** (1.0 - delta)
                - (1.0 - delta) * inner_column / q_t**delta
            )
            inner_power = np.maximum(inner_power, np.finfo(float).tiny)
            q_inner = inner_power ** (1.0 / (1.0 - delta))
        q_ph = np.where(in_outer, q_outer, q_inner)

    return np.clip(q_ph, _PROFILE_Q_MIN, _PROFILE_Q_MAX)


def _finite_profile_scales(M_ej, E_K, R_max_in, x_max, i_mass, i_kin):
    """Normalize a finite homologous profile to exact mass and kinetic energy.

    With ``r = R_scale*x`` and ``v = v_scale*x`` over the represented domain,
    the returned scales satisfy

    ``M_ej = 4*pi*rho_scale*R_scale**3*i_mass`` and
    ``E_K = 2*pi*rho_scale*R_scale**3*v_scale**2*i_kin``.
    """
    values = (M_ej, E_K, R_max_in, x_max, i_mass, i_kin)
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(
            "Finite ejecta mass, energy, radius, and profile integrals must be positive."
        )

    R_scale = float(R_max_in) / float(x_max)
    rho_scale = float(M_ej) / (4.0 * PI * float(i_mass) * R_scale**3)
    v_scale = np.sqrt(
        2.0 * float(i_mass) * float(E_K) / (float(i_kin) * float(M_ej))
    )
    return float(rho_scale), float(v_scale), float(R_scale)


def _finite_volume_q_cell_profiles(
    q_faces,
    q_heat,
    xi0,
    density_profile,
    delta,
    n,
):
    """Return exact shell averages on the common normalized ``q`` grid."""
    q_faces = np.asarray(q_faces, dtype=float)
    if q_faces.ndim != 1 or q_faces.size < 3:
        raise ValueError("Finite-volume grids require at least two cells.")
    if not np.all(np.isfinite(q_faces)) or not np.all(np.diff(q_faces) > 0.0):
        raise ValueError("Finite-volume cell faces must be finite and increasing.")

    lower = q_faces[:-1]
    upper = q_faces[1:]
    shell_volume = (upper**3 - lower**3) / 3.0
    enclosed_lower = _q_enclosed_profile_mass(
        lower, q_faces[0], density_profile, delta, n
    )
    enclosed_upper = _q_enclosed_profile_mass(
        upper, q_faces[0], density_profile, delta, n
    )
    shell_mass = enclosed_upper - enclosed_lower
    density_cell = shell_mass / shell_volume

    source_cell = np.zeros_like(density_cell)
    if xi0 > 0.0 and q_heat > q_faces[0]:
        mixed_upper = np.minimum(upper, q_heat)
        active = mixed_upper > lower
        if np.any(active):
            mixed_mass = (
                _q_enclosed_profile_mass(
                    mixed_upper[active],
                    q_faces[0],
                    density_profile,
                    delta,
                    n,
                )
                - enclosed_lower[active]
            )
            source_cell[active] = xi0 * mixed_mass / shell_volume[active]

    return density_cell, source_cell, shell_volume


@numba.njit(cache=True)
def _photospheric_cell_geometry_numba(
    q_faces,
    q_heat,
    xi0,
    q_photosphere,
    density_profile,
    delta,
    n,
):
    """Return moving cut-cell geometry and sources with a Numba kernel."""
    n_cells = q_faces.size - 1
    n_times = q_photosphere.size
    q_min = q_faces[0]

    thermal_upper = np.zeros((n_times, n_cells), dtype=np.float64)
    thermal_volume = np.zeros((n_times, n_cells), dtype=np.float64)
    density = np.ones((n_times, n_cells), dtype=np.float64)
    centres = np.zeros((n_times, n_cells), dtype=np.float64)
    source_density = np.zeros((n_times, n_cells), dtype=np.float64)
    direct_fraction = np.zeros(n_times, dtype=np.float64)
    active_count = np.zeros(n_times, dtype=np.int64)

    enclosed_lower = np.zeros(n_cells, dtype=np.float64)
    full_centres = np.zeros(n_cells, dtype=np.float64)
    for cell in range(n_cells):
        lower = q_faces[cell]
        upper = q_faces[cell + 1]
        enclosed_lower[cell] = _q_enclosed_profile_mass_scalar_numba(
            lower, q_min, density_profile, delta, n
        )
        full_centres[cell] = 0.5 * (lower + upper)

    for time_index in range(n_times):
        q_ph = q_photosphere[time_index]
        count = 0
        thermal_source = 0.0
        for cell in range(n_cells):
            lower = q_faces[cell]
            upper = q_faces[cell + 1]
            cut_upper = min(upper, q_ph)
            thermal_upper[time_index, cell] = cut_upper
            centres[time_index, cell] = full_centres[cell]

            if cut_upper <= lower:
                continue

            count += 1
            volume = (
                np.power(cut_upper, 3.0) - np.power(lower, 3.0)
            ) / 3.0
            thermal_volume[time_index, cell] = volume
            thermal_mass = (
                _q_enclosed_profile_mass_scalar_numba(
                    cut_upper, q_min, density_profile, delta, n
                )
                - enclosed_lower[cell]
            )
            density[time_index, cell] = thermal_mass / volume
            centres[time_index, cell] = 0.75 * (
                np.power(cut_upper, 4.0) - np.power(lower, 4.0)
            ) / (
                np.power(cut_upper, 3.0) - np.power(lower, 3.0)
            )

            source_upper = min(cut_upper, q_heat)
            if source_upper > lower:
                source_mass = (
                    _q_enclosed_profile_mass_scalar_numba(
                        source_upper, q_min, density_profile, delta, n
                    )
                    - enclosed_lower[cell]
                )
                source_density[time_index, cell] = float(xi0) * source_mass / volume
                thermal_source += source_density[time_index, cell] * volume

        active_count[time_index] = count

        total_source = 0.0
        if float(xi0) > 0.0 and float(q_heat) > q_min:
            total_source = float(xi0) * _q_profile_moment_scalar_numba(
                q_min,
                float(q_heat),
                2.0,
                density_profile,
                delta,
                n,
            )
        if total_source > 0.0:
            direct_fraction[time_index] = min(
                max(1.0 - thermal_source / total_source, 0.0),
                1.0,
            )
    return (
        thermal_upper,
        thermal_volume,
        density,
        centres,
        source_density,
        direct_fraction,
        active_count,
    )


def _photospheric_cell_geometry(
    q_faces,
    q_heat,
    xi0,
    q_photosphere,
    density_profile,
    delta,
    n,
):
    """Normalize inputs and dispatch cut-cell geometry to the Numba kernel."""
    q_faces = np.ascontiguousarray(np.asarray(q_faces, dtype=float))
    q_photosphere = np.ascontiguousarray(np.asarray(q_photosphere, dtype=float))
    return _photospheric_cell_geometry_numba(
        q_faces, q_heat, xi0, q_photosphere, density_profile, delta, n
    )


@numba.njit(cache=True)
def _photospheric_transport_coefficients_numba(
    q_faces,
    q_photosphere,
    expansion_factor,
    tau_scale,
    thermal_volume,
    density,
    centres,
    boundary_density,
    active_count,
    dy_values,
):
    """Numba kernel for moving-cut-cell diffusion coefficients."""
    n_times, n_cells = thermal_volume.shape
    lower = np.zeros_like(thermal_volume)
    upper = np.zeros_like(thermal_volume)
    boundary = np.zeros_like(thermal_volume)
    luminosity_transport = np.zeros(n_times)

    for time_index in range(n_times):
        dy_step = dy_values[time_index]
        count = int(active_count[time_index])
        if count > 1:
            for cell in range(count - 1):
                spacing = (
                    centres[time_index, cell + 1]
                    - centres[time_index, cell]
                )
                inv_density_face = 2.0 / (
                    density[time_index, cell]
                    + density[time_index, cell + 1]
                )
                face_transport = (
                    q_faces[cell + 1] ** 2
                    * inv_density_face
                    / spacing
                )
                upper[time_index, cell] = (
                    dy_step
                    * face_transport
                    / thermal_volume[time_index, cell]
                )
                lower[time_index, cell + 1] = (
                    dy_step
                    * face_transport
                    / thermal_volume[time_index, cell + 1]
                )

        if count > 0:
            # Apply the same Marshak relation at the moving tau=2/3 surface.
            boundary_distance = max(
                q_photosphere[time_index]
                - centres[time_index, count - 1],
                0.0,
            )
            boundary_alpha = (
                4.0
                * expansion_factor[time_index] ** 2
                / (
                    3.0
                    * tau_scale
                    * boundary_density[time_index]
                )
            )
            transport = (
                q_photosphere[time_index] ** 2
                / (
                    boundary_density[time_index]
                    * (boundary_distance + boundary_alpha)
                )
            )
            boundary[time_index, count - 1] = (
                dy_step
                * transport
                / thermal_volume[time_index, count - 1]
            )
            luminosity_transport[time_index] = transport

    return lower, upper, boundary, luminosity_transport


def _photospheric_transport_coefficients(
    q_faces,
    q_photosphere,
    expansion_factor,
    tau_scale,
    thermal_volume,
    density,
    centres,
    boundary_density,
    active_count,
    dy,
):
    """Build conservative diffusion coefficients on moving cut cells."""
    q_faces = np.asarray(q_faces, dtype=float)
    q_photosphere = np.asarray(q_photosphere, dtype=float)
    expansion_factor = np.asarray(expansion_factor, dtype=float)
    thermal_volume = np.asarray(thermal_volume, dtype=float)
    density = np.asarray(density, dtype=float)
    centres = np.asarray(centres, dtype=float)
    boundary_density = np.asarray(boundary_density, dtype=float)
    active_count = np.asarray(active_count, dtype=np.int64)
    n_times, n_cells = thermal_volume.shape
    dy_values = np.asarray(dy, dtype=float)
    if dy_values.ndim == 0:
        dy_values = np.full(n_times, float(dy_values), dtype=float)
    elif dy_values.shape != (n_times,):
        raise ValueError(
            "dy must be a scalar or contain one step size per time-grid node."
        )
    if np.any(~np.isfinite(dy_values)) or np.any(dy_values <= 0.0):
        raise ValueError("Nickel time steps must be finite and positive.")

    return _photospheric_transport_coefficients_numba(
        q_faces,
        q_photosphere,
        expansion_factor,
        float(tau_scale),
        thermal_volume,
        density,
        centres,
        boundary_density,
        active_count,
        np.ascontiguousarray(dy_values),
    )


@numba.njit(fastmath=True, cache=True)
def thomas_algorithm(a, b, c_up, d, c_prime, d_prime, x_out):
    """
    Numba-jitted Thomas algorithm for solving a tridiagonal system Ax=d.
    a: lower diagonal (a[0] is ignored)
    b: main diagonal
    c_up: upper diagonal (c_up[-1] is ignored)
    d: right-hand side vector
    Writes the solution into x_out.
    """
    n = len(d)

    # Forward elimination
    c_prime[0] = c_up[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c_up[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    # Backward substitution
    x_out[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x_out[i] = d_prime[i] - c_prime[i] * x_out[i + 1]


@numba.njit(fastmath=True, cache=True)
def _fast_time_loop_uniform_legacy_numba(
    Ny,
    Nx,
    dx,
    dy,
    fR_vals,
    f_ob_vals,
    heat_vals,
    source_profile,
    upper_coeff,
    lower_coeff,
    e_initial,
    Lfac,
):
    """Run the historical Crank--Nicolson Uniform outer-boundary solve."""
    implicit_weight = 0.5
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
    source_inner = source_profile[1:-1]

    for time_index in range(Ny):
        fR_now = fR_vals[time_index]
        fR_next = fR_vals[time_index + 1]

        b_diag[i_mid] = 1.0 + implicit_weight * fR_next * (
            upper_coeff + lower_coeff
        )
        c_up[i_mid] = -implicit_weight * fR_next * upper_coeff
        a[i_mid] = -implicit_weight * fR_next * lower_coeff

        b_diag[0] = -1.0
        c_up[0] = 1.0
        a[0] = 0.0

        b_diag[Nx] = dx - f_ob_vals[time_index + 1]
        a[Nx] = f_ob_vals[time_index + 1]
        c_up[Nx] = 0.0

        S_now_inner = source_inner * (fR_now * heat_vals[time_index])
        S_next_inner = source_inner * (fR_next * heat_vals[time_index + 1])
        rhs[i_mid] = (
            e_now[i_mid]
            + dy
            * (
                (1.0 - implicit_weight) * S_now_inner
                + implicit_weight * S_next_inner
            )
            + (1.0 - implicit_weight)
            * fR_now
            * (
                upper_coeff * (e_now[ip1] - e_now[i_mid])
                - lower_coeff * (e_now[i_mid] - e_now[im1])
            )
        )
        rhs[0] = 0.0
        rhs[Nx] = 0.0

        thomas_algorithm(a, b_diag, c_up, rhs, c_prime, d_prime, e_next)
        L_bol_out[time_index] = Lfac * (e_next[Nx - 1] - e_next[Nx])
        e_now, e_next = e_next, e_now

    return L_bol_out


@numba.njit(fastmath=True, cache=True)
def _fast_time_loop_photosphere_kernel(
    Ny,
    n_cells,
    dy_steps,
    fR_vals,
    heat_vals,
    source_vals,
    volume_vals,
    upper_coeff_vals,
    lower_coeff_vals,
    boundary_coeff_vals,
    active_count_vals,
    e_initial,
    luminosity_coeff_vals,
    L0,
):
    """Advance diffusion inside a receding cut-cell photosphere."""
    e_now = e_initial.copy()
    e_next = np.empty_like(e_now)
    L_out = np.zeros(Ny)
    a = np.zeros(n_cells)
    b_diag = np.zeros(n_cells)
    c_up = np.zeros(n_cells)
    rhs = np.zeros(n_cells)
    c_prime = np.zeros(n_cells)
    d_prime = np.zeros(n_cells)

    for time_index in range(Ny):
        dy_step = dy_steps[time_index]
        fR_next = fR_vals[time_index + 1]
        active_next = active_count_vals[time_index + 1]
        swept_energy = 0.0

        for cell in range(n_cells):
            volume_lost = (
                volume_vals[time_index, cell]
                - volume_vals[time_index + 1, cell]
            )
            if volume_lost > 0.0:
                swept_energy += max(e_now[cell], 0.0) * volume_lost

            if cell < active_next:
                lower = lower_coeff_vals[time_index + 1, cell]
                upper = upper_coeff_vals[time_index + 1, cell]
                boundary = boundary_coeff_vals[time_index + 1, cell]
                a[cell] = -fR_next * lower
                c_up[cell] = -fR_next * upper
                b_diag[cell] = 1.0 + fR_next * (
                    lower + upper + boundary
                )
                rhs[cell] = (
                    e_now[cell]
                    + dy_step
                    * fR_next
                    * heat_vals[time_index + 1]
                    * source_vals[time_index + 1, cell]
                )
            else:
                a[cell] = 0.0
                b_diag[cell] = 1.0
                c_up[cell] = 0.0
                rhs[cell] = 0.0

        thomas_algorithm(a, b_diag, c_up, rhs, c_prime, d_prime, e_next)

        boundary_luminosity = 0.0
        if active_next > 0:
            boundary_luminosity = (
                luminosity_coeff_vals[time_index + 1]
                * e_next[active_next - 1]
            )
        sweep_luminosity = L0 * swept_energy / (fR_next * dy_step)
        if active_next > 0:
            L_out[time_index] = boundary_luminosity + sweep_luminosity
        else:
            # The terminal cut-cell sweep occurs at the crossing instant, not
            # at an output node in the already fully thin state.  Assign its
            # interval-integrated energy to the last valid photospheric node.
            L_out[time_index] = 0.0
            if time_index > 0:
                L_out[time_index - 1] += sweep_luminosity
        e_now, e_next = e_next, e_now

    return L_out


def _fast_time_loop_photosphere_numba(
    Ny,
    n_cells,
    dy,
    fR_vals,
    heat_vals,
    source_vals,
    volume_vals,
    upper_coeff_vals,
    lower_coeff_vals,
    boundary_coeff_vals,
    active_count_vals,
    e_initial,
    luminosity_coeff_vals,
    L0,
):
    """Normalize scalar/variable steps and run the compiled transport loop."""
    dy_steps = np.asarray(dy, dtype=float)
    if dy_steps.ndim == 0:
        dy_steps = np.full(int(Ny), float(dy_steps), dtype=float)
    elif dy_steps.shape != (int(Ny),):
        raise ValueError("dy must be a scalar or contain exactly Ny step sizes.")
    if np.any(~np.isfinite(dy_steps)) or np.any(dy_steps <= 0.0):
        raise ValueError("Nickel time steps must be finite and positive.")
    return _fast_time_loop_photosphere_kernel(
        Ny,
        n_cells,
        np.ascontiguousarray(dy_steps),
        fR_vals,
        heat_vals,
        source_vals,
        volume_vals,
        upper_coeff_vals,
        lower_coeff_vals,
        boundary_coeff_vals,
        active_count_vals,
        e_initial,
        luminosity_coeff_vals,
        L0,
    )


def _calculate_uniform_legacy_transport(
    *,
    M_ej,
    v_ej,
    E_Th_in,
    M_ni,
    R_max_in,
    f_ni,
    kappa0,
    kappa_gamma,
    T_floor,
    Nx,
    Ny,
    t_max_days,
):
    """Reproduce the historical Uniform outer-boundary Nickel solver."""
    Nx, Ny = int(Nx), int(Ny)
    if Nx < 2 or Ny < 1:
        raise ValueError("NickelModel requires Nx >= 2 and Ny >= 1.")

    x_min, x_max = 1.0, 1.0e4
    I_M = _integral_power_law(x_min, x_max, 2.0)
    I_K = _integral_power_law(x_min, x_max, 4.0)
    E_K = 0.5 * float(M_ej) * float(v_ej) ** 2
    rho_scale, v_scale, R_scale = _finite_profile_scales(
        M_ej,
        E_K,
        R_max_in,
        x_max,
        I_M,
        I_K,
    )
    t_ex = R_scale / v_scale
    t_diff = 3.0 * float(kappa0) * rho_scale * R_scale**2 / C_LIGHT
    t_gamma = np.sqrt(
        3.0
        * float(kappa_gamma)
        * float(M_ej)
        / (4.0 * PI * float(v_ej) ** 2)
    )

    u0 = rho_scale * (EPSILON_NI - EPSILON_CO) * t_diff
    L0 = (
        4.0 * PI * R_scale * C_LIGHT * u0
        / (3.0 * float(kappa0) * rho_scale)
    )
    tau_scale = float(kappa0) * rho_scale * R_scale
    e0_coeff = float(E_Th_in) / (
        2.0 * PI * u0 * x_max**2 * R_scale**3
    )

    if float(f_ni) <= 0.0:
        x_heat = x_min
    elif float(f_ni) >= 1.0:
        x_heat = x_max
    else:
        x_heat = (
            x_min**3 + float(f_ni) * (x_max**3 - x_min**3)
        ) ** (1.0 / 3.0)
    if x_heat <= x_min + 1.0e-14:
        xi0 = 0.0
    else:
        mixed_mass = _integral_power_law(x_min, x_heat, 2.0)
        xi0 = (I_M * (float(M_ni) / float(M_ej))) / mixed_mass
    xi0 = max(float(xi0), 0.0)

    x_vals = np.linspace(x_min, x_max, Nx + 1)
    dx = (x_max - x_min) / Nx
    x2 = x_vals * x_vals

    t_max = float(t_max_days) * DAY
    y_max = t_max / t_diff
    y_vals = np.linspace(0.0, y_max, Ny + 1)
    dy = y_vals[1] - y_vals[0]
    fR_vals = 1.0 + (y_vals * t_diff / t_ex)
    f_ob_vals = -(4.0 / (3.0 * tau_scale)) * (fR_vals * fR_vals)

    t_phys = y_vals * t_diff
    e_co_ratio = EPSILON_CO / (EPSILON_NI - EPSILON_CO)
    heat = np.exp(-t_phys / TAU_NI)
    co_deposition = np.zeros_like(t_phys)
    positive_time = t_phys > 0.0
    co_deposition[positive_time] = 1.0 - np.exp(
        -(t_gamma / t_phys[positive_time]) ** 2
    )
    heat += e_co_ratio * np.exp(-t_phys / TAU_CO) * co_deposition

    source_profile = _legacy_uniform_ni_source_profile(x_vals, x_heat, xi0)
    x_inner = x_vals[1:-1]
    face_area = 0.5 * (x2[:-1] + x2[1:])
    coeff_norm = dy / (x_inner * dx) ** 2
    lower_coeff = coeff_norm * face_area[:-1]
    upper_coeff = coeff_norm * face_area[1:]

    e_initial = e0_coeff / x_vals
    Lfac = L0 * x_max**2 / dx
    L_out = _fast_time_loop_uniform_legacy_numba(
        Ny,
        Nx,
        dx,
        dy,
        fR_vals,
        f_ob_vals,
        heat,
        source_profile,
        upper_coeff,
        lower_coeff,
        e_initial,
        Lfac,
    )

    t_s = (y_vals * t_diff)[1:]
    R_hom = float(R_max_in) * fR_vals[1:]
    L_positive = np.where(L_out > 0.0, L_out, 0.0)
    T_try = (
        L_positive / (4.0 * PI * SIGMA_SB * R_hom**2)
    ) ** 0.25
    R_floor = np.sqrt(
        L_positive / (4.0 * PI * SIGMA_SB * float(T_floor) ** 4)
    )
    above_floor = T_try > float(T_floor)
    T_effective = np.where(above_floor, T_try, float(T_floor))
    R_effective = np.where(above_floor, R_hom, R_floor)

    return NickelTransportState(
        t_s=np.asarray(t_s, float),
        Lbol=np.asarray(L_out, float),
        Lphotospheric=np.asarray(L_out, float).copy(),
        Ldirect=np.zeros(Ny, dtype=float),
        q_ph=np.ones(Ny, dtype=float),
        Rph=np.asarray(R_effective, float),
        Tph=np.asarray(T_effective, float),
        Rhom=np.asarray(R_hom, float),
        photosphere_valid=np.ones(Ny, dtype=bool),
        density_profile="uniform",
    )


class NickelModel:
    """
    Canonical nickel-powered model.

    Canonical theta order:
    (M_ej, v_ej, E_Th_in, M_ni, R_0, f_ni, kappa0, kappa_gamma,
     T_floor, delta, n)

    Backward compatibility:
    - the old shorter pure-nickel form
      (M_ej, v_ej, M_ni, f_ni, kappa0, kappa_gamma, T_floor)
      is still accepted and mapped to E_Th_in=0, R_0=10.
    """

    _warmup_theta = (5.0, 1.0, 1.0, 0.2, 100.0, 0.5, 0.2, 0.03, 4000.0)
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

    def calculate_transport(
        self,
        theta,
        Nx=100,
        Ny=1000,
        t_max_days=150.0,
        density_profile="uniform",
        enforce_ni_mixing_constraint=True,
    ) -> NickelTransportState:
        """Solve the profile-selected Nickel transport problem.

        Parameters
        ----------
        density_profile : {"uniform", "bpl", "broken_power_law", "exp", "exponential", "ia", "auto"}, optional
            ``uniform`` is the backward-compatible default and selects the
            historical fixed-outer-boundary Crank--Nicolson solver. ``bpl`` and
            ``broken_power_law`` select the current physical-photosphere BPL
            solve; ``exponential`` (or ``exp``/``ia``) selects the corresponding
            finite exponential Type-Ia-like solve.
            The legacy direct-call value ``auto`` selects BPL for the canonical
            parameter form and uniform density for legacy input.
        Notes
        -----
        Uniform reproduces the historical ``x=1..1e4`` grid, linear time grid,
        Co-only gamma leakage, outer-boundary luminosity, and homologous
        temperature-floor mapping. BPL and exponential use the common
        ``q=v/v_max=r/R_out`` coordinate from ``1e-4`` to ``1``, a quadratic
        time grid, and the moving ``tau=2/3`` finite-volume boundary.

        ``f_ni`` is the Lagrangian mass coordinate of the outer edge of the
        Ni-mixed region: ``f_ni=M(<q_Ni)/M_ej``.  The solver derives the
        profile-dependent radius/velocity coordinate ``q_Ni`` from this mass
        fraction.  The Ni abundance is constant inside that cutoff and zero
        outside, with mass-integral normalization to ``M_ni``.

        For BPL and exponential, spatial diffusion is discretized with spherical finite
        volumes. Density and radioactive heating are exact shell averages and
        internal face fluxes use a harmonic diffusion coefficient. In the
        production solve the outermost active cell is cut continuously by the
        receding photosphere; radiation exposed as that cut cell shrinks is
        released explicitly, while the remaining stored energy is conserved.
        There is no fitted or forced late-time matching factor.

        For BPL and exponential, the physical radius is ``Rph=Rhom*q_ph`` and its temperature is
        determined only from ``Lphotospheric`` through the Stefan--Boltzmann
        relation.  No temperature floor is applied.  After the represented
        ejecta becomes completely optically thin, ``Rph`` and ``Tph`` are NaN
        while ``Lbol=Ldirect`` remains finite.
        """
        # constants shortcut
        pi, c, day = PI, C_LIGHT, DAY
        eNi, eCo = EPSILON_NI, EPSILON_CO

        theta = tuple(theta)
        if len(theta) == 11:
            (
                M_ej,
                v_ej,
                E_Th_in,
                M_ni,
                R_max_in,
                f_ni,
                kappa0,
                kappa_gamma,
                _T_floor,
                delta,
                n,
            ) = theta
            is_legacy_theta = False
        elif len(theta) == 9:
            (M_ej, v_ej, E_Th_in, M_ni, R_max_in, f_ni, kappa0, kappa_gamma, _T_floor) = theta
            delta = _BPL_DEFAULT_DELTA
            n = _BPL_DEFAULT_N
            # Public helpers expand legacy seven-parameter dictionaries to
            # this canonical form before reaching the solver.
            is_legacy_theta = bool(
                float(E_Th_in) == 0.0 and float(R_max_in) == 10.0
            )
        elif len(theta) == 7:
            (M_ej, v_ej, M_ni, f_ni, kappa0, kappa_gamma, _T_floor) = theta
            E_Th_in = 0.0
            R_max_in = 10.0
            delta = _BPL_DEFAULT_DELTA
            n = _BPL_DEFAULT_N
            is_legacy_theta = True
        else:
            raise ValueError(
                "NickelModel theta must have canonical length 11 "
                "(or legacy length 9 or 7)."
            )

        M_ej = float(M_ej) * M_SUN
        E_Th_in = float(E_Th_in) * 1.0e49
        M_ni = float(M_ni) * M_SUN
        R_max_in = float(R_max_in) * R_SUN

        f_ni = float(f_ni)
        if not np.isfinite(f_ni) or not (0.0 <= f_ni <= 1.0):
            raise ValueError("f_ni must be finite and in [0, 1].")
        mixed_mass_limit = f_ni * float(M_ej)
        tolerance = 1.0e-12 * max(abs(float(M_ni)), abs(mixed_mass_limit), 1.0)
        if enforce_ni_mixing_constraint and float(M_ni) > mixed_mass_limit + tolerance:
            raise ValueError(
                "Nickel mass-coordinate mixing requires "
                "M_ni <= f_ni*M_ej (equivalently f_ni >= M_ni/M_ej)."
            )
        kappa0 = float(kappa0)
        kappa_g = float(kappa_gamma)
        v_ej = float(v_ej) * 1e9

        density_profile = (
            str(density_profile).strip().lower().replace("-", "_").replace(" ", "_")
        )
        if density_profile == "auto":
            density_profile = "uniform" if is_legacy_theta else "broken_power_law"
        density_profile = {
            "bpl": "broken_power_law",
            "exp": "exponential",
            "ia": "exponential",
        }.get(density_profile, density_profile)
        if density_profile not in {"broken_power_law", "exponential", "uniform"}:
            raise ValueError(
                "density_profile must be 'uniform', 'bpl'/'broken_power_law', "
                "'exp'/'exponential'/'ia', or 'auto'."
            )

        delta = float(delta)
        n = float(n)

        if density_profile == "uniform":
            return _calculate_uniform_legacy_transport(
                M_ej=M_ej,
                v_ej=v_ej,
                E_Th_in=E_Th_in,
                M_ni=M_ni,
                R_max_in=R_max_in,
                f_ni=f_ni,
                kappa0=kappa0,
                kappa_gamma=kappa_g,
                T_floor=float(_T_floor),
                Nx=Nx,
                Ny=Ny,
                t_max_days=t_max_days,
            )

        q_min = _PROFILE_Q_MIN
        q_max = _PROFILE_Q_MAX
        I_M = _q_profile_moment(
            q_min, q_max, 2.0, density_profile, delta, n
        )
        I_K = _q_profile_moment(
            q_min, q_max, 4.0, density_profile, delta, n
        )

        E_K = 0.5 * M_ej * v_ej * v_ej
        rho_scale, v_scale, R_scale = _finite_profile_scales(
            M_ej,
            E_K,
            R_max_in,
            q_max,
            I_M,
            I_K,
        )
        t_ex = R_scale / v_scale
        t_diff = 3.0 * kappa0 * rho_scale * R_scale**2 / c
        t_gamma = np.sqrt((3.0 * kappa_g * M_ej) / (4.0 * pi * v_ej * v_ej))

        u0 = rho_scale * (eNi - eCo) * t_diff
        L0 = (4.0 * pi * R_scale * c * u0) / (3.0 * kappa0 * rho_scale)
        tau_scale = kappa0 * rho_scale * R_scale
        thermal_integral = q_max**2 - q_min**2
        e0_coeff = E_Th_in / (2.0 * pi * u0 * thermal_integral * R_scale**3)

        q_heat = _q_mass_fraction_to_radius(
            f_ni,
            q_min,
            density_profile,
            delta,
            n,
        )
        if q_heat <= q_min + 1e-14:
            xi0 = 0.0
        else:
            denom_heat = _q_profile_moment(
                q_min, q_heat, 2.0, density_profile, delta, n
            )
            # Normalize against the same finite mass domain used for rho_scale,
            # so every profile deposits exactly the requested M_ni.
            xi0 = (I_M * (M_ni / M_ej)) / denom_heat
        xi0 = max(xi0, 0.0)

        Nx, Ny = int(Nx), int(Ny)
        if Nx < 2 or Ny < 1:
            raise ValueError("NickelModel requires Nx >= 2 and Ny >= 1.")

        # Cell-centred spherical finite-volume mesh.  Every spatial term below
        # is built from these same faces and shell volumes.
        q_faces = np.linspace(q_min, q_max, Nx + 1)

        t_max = float(t_max_days) * day
        y_max = t_max / t_diff
        # The model time origin is the explosion epoch at R_out=R_0.  Starting
        # at a fixed positive dimensionless y makes the corresponding physical
        # time grow as R_0 shrinks and can even reverse the grid for compact
        # white-dwarf radii.  Start exactly at t=0; the gamma-leakage expression
        # below already handles that endpoint explicitly.
        y_vals = _build_time_grid(y_max, Ny)
        dy_steps = np.diff(y_vals)
        # Coefficients live on time-grid nodes; row i uses the step ending at
        # that node.  Row zero is not advanced but receives the first positive
        # step so the coefficient builder stays well-defined and testable.
        dy_by_node = np.empty_like(y_vals)
        dy_by_node[0] = dy_steps[0]
        dy_by_node[1:] = dy_steps

        fR_vals = 1.0 + (y_vals * t_diff / t_ex)

        t_phys = y_vals * t_diff
        heat = _radioactive_heating_shape(t_phys, t_gamma)
        deposited_heating = M_ni * (eNi - eCo) * heat[1:]
        q_photosphere = _q_photosphere_radius(
            fR_vals,
            tau_scale,
            density_profile,
            delta,
            n,
        )
        (
            thermal_upper,
            thermal_volume,
            thermal_density,
            thermal_centres,
            source_vals,
            direct_fraction_vals,
            active_count_vals,
        ) = _photospheric_cell_geometry(
            q_faces,
            q_heat,
            xi0,
            q_photosphere,
            density_profile,
            delta,
            n,
        )
        (
            lower_coeff_vals,
            upper_coeff_vals,
            photosphere_coeff_vals,
            luminosity_transport_vals,
        ) = _photospheric_transport_coefficients(
            q_faces,
            q_photosphere,
            fR_vals,
            tau_scale,
            thermal_volume,
            thermal_density,
            thermal_centres,
            _eta_q(q_photosphere, density_profile, delta, n),
            active_count_vals,
            dy_by_node,
        )
        e_initial = np.zeros(Nx)
        active_initial = thermal_volume[0] > 0.0
        lower_faces = q_faces[:-1]
        e_initial[active_initial] = (
            e0_coeff
            * 0.5
            * (
                thermal_upper[0, active_initial] ** 2
                - lower_faces[active_initial] ** 2
            )
            / thermal_volume[0, active_initial]
        )
        L_photospheric = _fast_time_loop_photosphere_numba(
            Ny,
            Nx,
            dy_steps,
            fR_vals,
            heat,
            np.ascontiguousarray(source_vals),
            np.ascontiguousarray(thermal_volume),
            np.ascontiguousarray(upper_coeff_vals),
            np.ascontiguousarray(lower_coeff_vals),
            np.ascontiguousarray(photosphere_coeff_vals),
            np.ascontiguousarray(active_count_vals),
            e_initial,
            L0 * luminosity_transport_vals,
            L0,
        )
        L_direct = deposited_heating * direct_fraction_vals[1:]

        photosphere_valid = active_count_vals[1:] > 0
        # The time loop assigns the terminal cut-cell sweep to the last valid
        # photospheric node. Output nodes after that event describe the fully
        # thin state, whose luminosity is the instantaneous deposited power.
        L_photospheric = np.where(photosphere_valid, L_photospheric, 0.0)
        L_bol = L_photospheric + L_direct
        t_s = (y_vals * t_diff)[1:]

        R_hom = R_scale * fR_vals[1:]
        q_ph = q_photosphere[1:]
        R_ph = np.full(Ny, np.nan, dtype=float)
        T_ph = np.full(Ny, np.nan, dtype=float)
        R_ph[photosphere_valid] = R_hom[photosphere_valid] * q_ph[photosphere_valid]
        T_ph[photosphere_valid] = (
            np.maximum(L_photospheric[photosphere_valid], 0.0)
            / (
                4.0
                * pi
                * SIGMA_SB
                * R_ph[photosphere_valid] ** 2
            )
        ) ** 0.25

        return NickelTransportState(
            t_s=np.asarray(t_s, float),
            Lbol=np.asarray(L_bol, float),
            Lphotospheric=np.asarray(L_photospheric, float),
            Ldirect=np.asarray(L_direct, float),
            q_ph=np.asarray(q_ph, float),
            Rph=R_ph,
            Tph=T_ph,
            Rhom=np.asarray(R_hom, float),
            photosphere_valid=np.asarray(photosphere_valid, bool),
            density_profile=density_profile,
        )

    def calculate_light_curve(self, theta, **kwargs):
        """Compatibility wrapper returning the physical photosphere tuple."""
        state = self.calculate_transport(theta, **kwargs)
        return state.t_s, state.Lbol, state.Tph, state.Rph

    def L_bol(self, t_obs, theta, z=0.0, **kwargs):
        t_s, L_series, T_eff_values, R_outer_values = self.calculate_light_curve(theta, **kwargs)
        t_obs_days = np.asarray(t_obs, float)
        t_obs_grid_days = (t_s * (1.0 + z)) / DAY
        return interp_fit(
            t_obs_grid_days,
            np.asarray(L_series, float),
            t_obs_days,
            yscale="log10",
            fill="edge",
        )
