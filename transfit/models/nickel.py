# sn_broken_power_law_fast.py
# -*- coding: utf-8 -*-

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
_BPL_X_MIN = 1.0e-3
_BPL_DEFAULT_DELTA = 0.0
_BPL_DEFAULT_N = 10.0
_EXPONENTIAL_V_MAX_OVER_V_E = 12.0
_EXPONENTIAL_X_MIN = 1.0e-3

# -----------------------------------------------------------------------------
# Core solver functions (Numba JIT)
# -----------------------------------------------------------------------------


def _integral_power_law(x_lo, x_hi, power):
    """Return ``integral(x**power dx, x_lo, x_hi)`` for positive bounds."""
    if not (np.isfinite(x_lo) and np.isfinite(x_hi) and 0.0 < x_lo < x_hi):
        raise ValueError("Power-law integration bounds must satisfy 0 < x_lo < x_hi.")

    exponent = float(power) + 1.0
    if abs(exponent) < 1.0e-12:
        return float(np.log(x_hi / x_lo))
    return float((x_hi**exponent - x_lo**exponent) / exponent)


def _broken_power_law_integrals(x_min, x_max, delta, n):
    """Dimensionless mass and kinetic-energy integrals for a BPL profile.

    The profile is ``rho/rho_t = x**(-delta)`` below the break ``x=1`` and
    ``x**(-n)`` above it.  The returned integrals are respectively
    ``integral(x**2 rho/rho_t dx)`` and
    ``integral(x**4 rho/rho_t dx)`` over the finite computational domain.
    """
    if not (0.0 < x_min < 1.0 < x_max):
        raise ValueError("Broken-power-law grid must satisfy x_min < 1 < x_max.")
    if not (np.isfinite(delta) and 0.0 <= delta < 3.0):
        raise ValueError("delta must be finite and satisfy 0 <= delta < 3.")
    if not (np.isfinite(n) and n > 5.0):
        raise ValueError("n must be finite and > 5.")

    i_mass = (
        _integral_power_law(x_min, 1.0, 2.0 - delta)
        + _integral_power_law(1.0, x_max, 2.0 - n)
    )
    i_kin = (
        _integral_power_law(x_min, 1.0, 4.0 - delta)
        + _integral_power_law(1.0, x_max, 4.0 - n)
    )
    return i_mass, i_kin


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


def _exponential_integrals(x_min, x_max):
    """Finite mass and kinetic-energy integrals for ``rho/rho_e=exp(-x)``."""
    if not (
        np.isfinite(x_min)
        and np.isfinite(x_max)
        and 0.0 <= x_min < x_max
    ):
        raise ValueError(
            "Exponential-profile grid must satisfy 0 <= x_min < x_max."
        )
    i_mass = _exponential_tail_moment(x_min, 2) - _exponential_tail_moment(x_max, 2)
    i_kin = _exponential_tail_moment(x_min, 4) - _exponential_tail_moment(x_max, 4)
    return float(i_mass), float(i_kin)


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


def _density_mass_integral(x_min, x_max, density_profile, delta, n):
    """Dimensionless enclosed mass integral used to normalize Ni mixing."""
    if density_profile == "uniform":
        return _integral_power_law(x_min, x_max, 2.0)
    if density_profile == "exponential":
        return _exponential_tail_moment(x_min, 2) - _exponential_tail_moment(x_max, 2)

    if x_max <= 1.0:
        return _integral_power_law(x_min, x_max, 2.0 - delta)
    if x_min >= 1.0:
        return _integral_power_law(x_min, x_max, 2.0 - n)
    return (
        _integral_power_law(x_min, 1.0, 2.0 - delta)
        + _integral_power_law(1.0, x_max, 2.0 - n)
    )


def _mass_fraction_to_radius(
    f_mass,
    x_min,
    x_max,
    density_profile,
    delta,
    n,
):
    """Return the radius enclosing a requested fraction of ejecta mass."""
    f_mass = float(f_mass)
    if not np.isfinite(f_mass) or not (0.0 <= f_mass <= 1.0):
        raise ValueError("Mass fraction must be finite and in [0, 1].")
    if f_mass <= 0.0:
        return float(x_min)
    if f_mass >= 1.0:
        return float(x_max)

    total_mass = _density_mass_integral(
        x_min, x_max, density_profile, delta, n
    )
    target_mass = f_mass * total_mass

    if density_profile == "uniform":
        return float((x_min**3 + 3.0 * target_mass) ** (1.0 / 3.0))

    if density_profile == "broken_power_law":
        inner_mass = _integral_power_law(x_min, 1.0, 2.0 - delta)
        if target_mass <= inner_mass:
            exponent = 3.0 - delta
            return float(
                (x_min**exponent + exponent * target_mass) ** (1.0 / exponent)
            )

        exponent = 3.0 - n
        return float(
            (1.0 + exponent * (target_mass - inner_mass)) ** (1.0 / exponent)
        )

    # The exponential cumulative mass has no elementary inverse.
    lo = float(x_min)
    hi = float(x_max)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        enclosed = _density_mass_integral(
            x_min, mid, density_profile, delta, n
        )
        if enclosed < target_mass:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _enclosed_profile_mass(x, x_min, density_profile, delta, n):
    """Vectorized dimensionless mass enclosed between ``x_min`` and ``x``."""
    x = np.asarray(x, dtype=float)

    if density_profile == "uniform":
        return (x**3 - x_min**3) / 3.0

    if density_profile == "exponential":
        tail_min = _exponential_tail_moment(x_min, 2)
        tail_x = np.exp(-x) * (x * x + 2.0 * x + 2.0)
        return tail_min - tail_x

    inner_exponent = 3.0 - delta
    outer_exponent = 3.0 - n
    inner_mass = (1.0 - x_min**inner_exponent) / inner_exponent
    enclosed_inner = (x**inner_exponent - x_min**inner_exponent) / inner_exponent
    enclosed_outer = inner_mass + (x**outer_exponent - 1.0) / outer_exponent
    return np.where(x <= 1.0, enclosed_inner, enclosed_outer)


def _conservative_ni_source_profile(
    x_vals,
    x_heat,
    xi0,
    density_profile,
    delta,
    n,
):
    """Return a control-volume-averaged, mass-conservative Ni source.

    The outer Ni boundary can cut through a numerical control volume.  Its
    contribution is included by the exact mixed mass inside that partial cell,
    avoiding resolution-dependent whole-cell switching while preserving the
    requested integrated nickel mass.
    """
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
    mixed_upper = np.minimum(upper, x_heat)
    active = mixed_upper > lower

    mixed_mass = np.zeros_like(lower)
    if np.any(active):
        mixed_mass[active] = (
            _enclosed_profile_mass(
                mixed_upper[active],
                x_vals[0],
                density_profile,
                delta,
                n,
            )
            - _enclosed_profile_mass(
                lower[active],
                x_vals[0],
                density_profile,
                delta,
                n,
            )
        )

    cell_volume = (upper**3 - lower**3) / 3.0
    source_profile[1:-1] = xi0 * mixed_mass / cell_volume
    return source_profile

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
def _fast_time_loop_numba(
    # Grid parameters
    Ny, Nx, dx, dy,
    # Pre-calculated physics arrays
    fR_vals, f_ob_vals, heat_vals, source_profile,
    # Pre-calculated matrix components
    upper_coeff, lower_coeff, implicit_weight,
    # Initial condition & constants
    e_initial, Lfac
):
    """
    The entire time-evolution loop, JIT-compiled with Numba.
    This function contains the performance-critical part of the calculation.
    """
    # Memory: only two evolving state vectors are needed.
    e_now = e_initial.copy()
    e_next = np.empty_like(e_now)

    # Output array
    L_bol_out = np.zeros(Ny)

    # Pre-allocate tridiagonal diagonals and RHS.
    a = np.zeros(Nx + 1)       # lower
    b_diag = np.zeros(Nx + 1)  # main
    c_up = np.zeros(Nx + 1)    # upper
    rhs = np.zeros(Nx + 1)
    c_prime = np.zeros(Nx + 1)  # Thomas workspace
    d_prime = np.zeros(Nx + 1)  # Thomas workspace

    # Index slices
    i_mid = slice(1, Nx)
    im1 = slice(0, Nx - 1)
    ip1 = slice(2, Nx + 1)
    source_inner = source_profile[1:-1]

    # --- Main time loop ---
    for n in range(Ny):
        fR_now, fR_next = fR_vals[n], fR_vals[n + 1]

        # --- Assemble A matrix (Ax=d) ---
        b_diag[i_mid] = 1.0 + implicit_weight * fR_next * (
            upper_coeff + lower_coeff
        )
        c_up[i_mid] = -implicit_weight * fR_next * upper_coeff
        a[i_mid] = -implicit_weight * fR_next * lower_coeff

        # Left boundary: -e_0 + e_1 = 0
        b_diag[0] = -1.0
        c_up[0] = 1.0
        a[0] = 0.0  # ignored

        # Right boundary: (dx - f_ob)*e_N + f_ob*e_{N-1} = 0
        b_diag[Nx] = dx - f_ob_vals[n + 1]
        a[Nx] = f_ob_vals[n + 1]
        c_up[Nx] = 0.0  # ignored

        # --- Assemble RHS vector ---
        S_now_inner = source_inner * (fR_now * heat_vals[n])
        S_next_inner = source_inner * (fR_next * heat_vals[n + 1])

        rhs[i_mid] = (
            e_now[i_mid]
            + dy * (
                (1.0 - implicit_weight) * S_now_inner
                + implicit_weight * S_next_inner
            )
            + (1.0 - implicit_weight) * fR_now * (
                upper_coeff * (e_now[ip1] - e_now[i_mid])
                - lower_coeff * (e_now[i_mid] - e_now[im1])
            )
        )
        rhs[0] = 0.0
        rhs[Nx] = 0.0

        # --- Solve ---
        thomas_algorithm(a, b_diag, c_up, rhs, c_prime, d_prime, e_next)

        # --- Luminosity ---
        L_bol_out[n] = Lfac * (e_next[Nx - 1] - e_next[Nx])

        # swap
        e_now, e_next = e_next, e_now

    return L_bol_out


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

    def calculate_light_curve(
        self,
        theta,
        Nx=100,
        Ny=1000,
        t_max_days=150.0,
        density_profile="uniform",
    ):
        """Solve the nickel-powered diffusion model.

        Parameters
        ----------
        density_profile : {"uniform", "bpl", "broken_power_law", "exp", "exponential", "ia", "auto"}, optional
            ``uniform`` is the backward-compatible default.  ``bpl`` and
            ``broken_power_law`` select homologously expanding ejecta with
            ``rho/rho_t = x**(-delta)`` for ``x < 1`` and ``x**(-n)`` for
            ``x >= 1``.  ``exponential`` (or ``exp``/``ia``) uses
            ``rho/rho_e=exp(-x)``.
            The legacy direct-call value ``auto`` selects BPL for the canonical
            parameter form and uniform density for legacy input.
        Notes
        -----
        ``delta`` and ``n`` are physical BPL structure parameters in ``theta``.
        Nine- and seven-parameter legacy vectors default to ``delta=0`` and
        ``n=10``.  The finite boundaries are fixed internally at ``v_max=3 v_t``
        for BPL and ``v_max=12 v_e`` for the exponential profile.  Both
        non-uniform profiles use backward Euler; uniform density uses
        Crank--Nicolson.  Every represented finite domain is normalized inside
        the initial outer radius ``R_0`` to the requested ejecta mass and energy.

        ``f_ni`` is the Lagrangian mass coordinate of the outer edge of the
        Ni-mixed region: ``f_ni=M(<x_Ni)/M_ej``.  The solver derives the
        profile-dependent radius/velocity coordinate ``x_Ni`` from this mass
        fraction.  The Ni abundance is constant inside that cutoff and zero
        outside, with mass-integral normalization to ``M_ni``.
        """
        # constants shortcut
        pi, c, day = PI, C_LIGHT, DAY
        eNi, eCo = EPSILON_NI, EPSILON_CO
        tau_Ni, tau_Co = TAU_NI, TAU_CO

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
                T_floor,
                delta,
                n,
            ) = theta
            is_legacy_theta = False
        elif len(theta) == 9:
            (M_ej, v_ej, E_Th_in, M_ni, R_max_in, f_ni, kappa0, kappa_gamma, T_floor) = theta
            delta = _BPL_DEFAULT_DELTA
            n = _BPL_DEFAULT_N
            # Public helpers expand legacy seven-parameter dictionaries to
            # this canonical form before reaching the solver.
            is_legacy_theta = bool(
                float(E_Th_in) == 0.0 and float(R_max_in) == 10.0
            )
        elif len(theta) == 7:
            (M_ej, v_ej, M_ni, f_ni, kappa0, kappa_gamma, T_floor) = theta
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
        if float(M_ni) > mixed_mass_limit + tolerance:
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
        if density_profile == "bpl":
            density_profile = "broken_power_law"
        if density_profile == "exp":
            density_profile = "exponential"
        if density_profile == "ia":
            density_profile = "exponential"
        if density_profile == "auto":
            density_profile = "uniform" if is_legacy_theta else "broken_power_law"
        if density_profile not in {"broken_power_law", "exponential", "uniform"}:
            raise ValueError(
                "density_profile must be 'uniform', 'bpl'/'broken_power_law', "
                "'exp'/'exponential'/'ia', or 'auto'."
            )

        delta = float(delta)
        n = float(n)
        implicit_weight = 0.5 if density_profile == "uniform" else 1.0

        if density_profile == "uniform":
            # Legacy grid and normalization.  Keeping this branch makes the
            # old uniform-density calculation available for A/B comparisons.
            x_min, x_max = 1.0, 1.0e4
            I_M = _density_mass_integral(x_min, x_max, density_profile, delta, n)
            I_K = _integral_power_law(x_min, x_max, 4.0)
        elif density_profile == "broken_power_law":
            # x = r/R_t = v/v_t.  R_0 is the finite outer boundary, so both
            # normalization integrals stop at x_max rather than infinity.
            x_min, x_max = _BPL_X_MIN, _BPL_V_MAX_OVER_V_T
            I_M, I_K = _broken_power_law_integrals(x_min, x_max, delta, n)
        else:
            # x = r/R_e = v/v_e for a finite exponential Ia-like profile.
            x_min, x_max = _EXPONENTIAL_X_MIN, _EXPONENTIAL_V_MAX_OVER_V_E
            I_M, I_K = _exponential_integrals(x_min, x_max)

        E_K = 0.5 * M_ej * v_ej * v_ej
        rho_scale, v_scale, R_scale = _finite_profile_scales(
            M_ej,
            E_K,
            R_max_in,
            x_max,
            I_M,
            I_K,
        )
        t_ex = R_scale / v_scale
        t_diff = 3.0 * kappa0 * rho_scale * R_scale**2 / c
        t_gamma = np.sqrt((3.0 * kappa_g * M_ej) / (4.0 * pi * v_ej * v_ej))

        eCo_ratio = eCo / (eNi - eCo)
        u0 = rho_scale * (eNi - eCo) * t_diff
        L0 = (4.0 * pi * R_scale * c * u0) / (3.0 * kappa0 * rho_scale)
        tau_scale = kappa0 * rho_scale * R_scale
        thermal_integral = (
            x_max**2
            if density_profile == "uniform"
            else x_max**2 - x_min**2
        )
        e0_coeff = E_Th_in / (2.0 * pi * u0 * thermal_integral * R_scale**3)

        x_heat = _mass_fraction_to_radius(
            f_ni,
            x_min,
            x_max,
            density_profile,
            delta,
            n,
        )
        if x_heat <= x_min + 1e-14:
            xi0 = 0.0
        else:
            denom_heat = _density_mass_integral(
                x_min, x_heat, density_profile, delta, n
            )
            # Normalize against the same finite mass domain used for rho_scale,
            # so every profile deposits exactly the requested M_ni.
            xi0 = (I_M * (M_ni / M_ej)) / denom_heat
        xi0 = max(xi0, 0.0)

        Nx, Ny = int(Nx), int(Ny)
        x_vals = np.linspace(x_min, x_max, Nx + 1)
        dx = (x_max - x_min) / Nx
        x2 = x_vals * x_vals

        if density_profile == "uniform":
            density_shape = np.ones_like(x_vals)
        elif density_profile == "broken_power_law":
            density_shape = np.where(
                x_vals < 1.0,
                x_vals ** (-delta),
                x_vals ** (-n),
            )
        else:
            density_shape = np.exp(-x_vals)

        t_max = float(t_max_days) * day
        y_max = t_max / t_diff
        # The model time origin is the explosion epoch at R_out=R_0.  Starting
        # at a fixed positive dimensionless y makes the corresponding physical
        # time grow as R_0 shrinks and can even reverse the grid for compact
        # white-dwarf radii.  Start exactly at t=0; the gamma-leakage expression
        # below already handles that endpoint explicitly.
        y_vals = np.linspace(0.0, y_max, Ny + 1)
        dy = y_vals[1] - y_vals[0]

        fR_vals = 1.0 + (y_vals * t_diff / t_ex)
        f_ob_vals = -(4.0 / (3.0 * tau_scale * density_shape[-1])) * (
            fR_vals * fR_vals
        )

        t_phys = y_vals * t_diff
        heat = np.exp(-t_phys / tau_Ni)
        leak = np.zeros_like(t_phys)
        mask = t_phys > 0.0
        leak[mask] = 1.0 - np.exp(-(t_gamma / t_phys[mask])**2)
        heat += eCo_ratio * np.exp(-t_phys / tau_Co) * leak

        source_profile = _conservative_ni_source_profile(
            x_vals,
            x_heat,
            xi0,
            density_profile,
            delta,
            n,
        )

        x_inner = x_vals[1:-1]
        # Face coefficients implement the conservative spherical operator
        # (1/x^2) d/dx [ x^2 / rho(x) * de/dx ].  The face diffusion
        # coefficient uses the harmonic mean of D propto 1/rho.
        face_area = 0.5 * (x2[:-1] + x2[1:])
        inv_density_face = 2.0 / (density_shape[:-1] + density_shape[1:])
        face_transport = face_area * inv_density_face
        coeff_norm = dy / (x_inner * dx)**2
        lower_coeff = coeff_norm * face_transport[:-1]
        upper_coeff = coeff_norm * face_transport[1:]

        e_initial = e0_coeff / x_vals
        Lfac = L0 * (x_max * x_max) / (density_shape[-1] * dx)

        L_out = _fast_time_loop_numba(
            Ny, Nx, dx, dy,
            fR_vals, f_ob_vals, heat, source_profile,
            upper_coeff, lower_coeff, implicit_weight,
            e_initial, Lfac
        )

        t_s = (y_vals * t_diff)[1:]

        # Nominal outer radius
        R_nom = R_max_in * fR_vals[1:]

        # guard against negative luminosity (numerical noise) to avoid invalid warnings
        L_pos = np.where(L_out > 0.0, L_out, 0.0)
        Teff_try = (L_pos / (4.0 * pi * R_nom * R_nom * SIGMA_SB))**0.25
        R_floor = np.sqrt(L_pos / (4.0 * pi * SIGMA_SB * (T_floor**4)))

        T_eff_values = np.where(Teff_try > T_floor, Teff_try, T_floor)
        R_outer_values = np.where(Teff_try > T_floor, R_nom, R_floor)

        return t_s, L_out, T_eff_values, R_outer_values

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
