from __future__ import annotations

import numpy as np
import pytest

from transfit.models.nickel import (
    _BPL_Q_T,
    _BPL_V_MAX_OVER_V_T,
    _EXPONENTIAL_Q_E,
    _EXPONENTIAL_V_MAX_OVER_V_E,
    _PHOTOSPHERE_TAU,
    _PROFILE_Q_MAX,
    _PROFILE_Q_MIN,
    _TIME_GRID_POWER,
    NickelModel,
    _build_time_grid,
    _eta_q,
    _finite_volume_q_cell_profiles,
    _finite_profile_scales,
    _photospheric_cell_geometry,
    _photospheric_transport_coefficients,
    _q_photosphere_radius,
    _q_mass_fraction_to_radius,
    _q_profile_moment,
    _radioactive_heating_shape,
)
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


PARAMS = (3.0, 1.0, 1.5, 0.08, 120.0, 0.2, 0.12, 0.03, 4500.0)
RELEASE_PARAMS = (
    3.0,
    1.0,
    0.0,
    0.08,
    1.0,
    0.8,
    0.12,
    0.03,
    4500.0,
    0.0,
    10.0,
)
TAIL_PARAMS = (
    5.0,
    np.sqrt(2.0e51 / (5.0 * M_SUN)) / 1.0e9,
    1.0,
    0.2,
    1.0,
    0.8,
    0.2,
    0.03,
    4500.0,
    0.0,
    10.0,
)
THIN_TAIL_PARAMS = (
    1.0,
    np.sqrt(2.0e51 / M_SUN) / 1.0e9,
    1.0,
    0.2,
    1.0,
    0.8,
    0.1,
    0.03,
    4500.0,
    0.0,
    10.0,
)


def _tail_deposited_heating(t_days, params=TAIL_PARAMS):
    """Deposited Ni/Co power for a canonical nickel parameter tuple."""
    t_s = np.asarray(t_days, dtype=float) * DAY
    mass = params[0] * M_SUN
    velocity = params[1] * 1.0e9
    nickel_mass = params[3] * M_SUN
    t_gamma = np.sqrt(
        3.0 * params[7] * mass / (4.0 * np.pi * velocity**2)
    )
    leakage = np.ones_like(t_s)
    positive = t_s > 0.0
    leakage[positive] = 1.0 - np.exp(-(t_gamma / t_s[positive]) ** 2)
    intrinsic_heating = nickel_mass * (
        (EPSILON_NI - EPSILON_CO) * np.exp(-t_s / TAU_NI)
        + EPSILON_CO * np.exp(-t_s / TAU_CO)
    )
    return intrinsic_heating * leakage


def test_gamma_leakage_multiplies_the_full_ni_co_heating_chain():
    t_days = np.array([0.0, 5.0, 20.0, 80.0])
    t_s = t_days * DAY
    t_gamma = 12.0 * DAY
    deposition = np.ones_like(t_s)
    positive = t_s > 0.0
    deposition[positive] = 1.0 - np.exp(-(t_gamma / t_s[positive]) ** 2)
    intrinsic_shape = (
        np.exp(-t_s / TAU_NI)
        + EPSILON_CO / (EPSILON_NI - EPSILON_CO) * np.exp(-t_s / TAU_CO)
    )

    actual = _radioactive_heating_shape(t_s, t_gamma)

    np.testing.assert_allclose(actual, intrinsic_shape * deposition, rtol=1.0e-15)
    assert actual[0] == pytest.approx(intrinsic_shape[0], rel=0.0, abs=0.0)


@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
def test_eta_profiles_share_the_same_q_domain(density_profile):
    q = np.linspace(_PROFILE_Q_MIN, _PROFILE_Q_MAX, 10_001)
    eta = _eta_q(q, density_profile, delta=0.0, n=10.0)

    assert q[0] == _PROFILE_Q_MIN
    assert q[-1] == _PROFILE_Q_MAX
    assert eta.shape == q.shape
    assert np.all(np.isfinite(eta))
    assert np.all(eta > 0.0)


@pytest.mark.parametrize("order", [0, 2, 4])
@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
def test_q_profile_moments_match_direct_quadrature(order, density_profile):
    expected = _q_profile_moment(
        _PROFILE_Q_MIN,
        _PROFILE_Q_MAX,
        order,
        density_profile,
        0.0,
        10.0,
    )
    q = np.linspace(_PROFILE_Q_MIN, _PROFILE_Q_MAX, 500_001)
    numerical = np.trapz(q**order * _eta_q(q, density_profile, 0.0, 10.0), q)
    assert expected == pytest.approx(numerical, rel=3.0e-8)


@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
@pytest.mark.parametrize("f_ni", [0.0, 0.2, 0.7, 1.0])
def test_q_f_ni_is_a_lagrangian_mass_coordinate(density_profile, f_ni):
    q_ni = _q_mass_fraction_to_radius(
        f_ni,
        _PROFILE_Q_MIN,
        density_profile,
        delta=0.0,
        n=10.0,
    )
    enclosed = (
        _q_profile_moment(
            _PROFILE_Q_MIN,
            q_ni,
            2.0,
            density_profile,
            0.0,
            10.0,
        )
        if q_ni > _PROFILE_Q_MIN
        else 0.0
    )
    total = _q_profile_moment(
        _PROFILE_Q_MIN,
        _PROFILE_Q_MAX,
        2.0,
        density_profile,
        0.0,
        10.0,
    )
    assert enclosed / total == pytest.approx(f_ni, rel=2.0e-13, abs=2.0e-13)


def test_nickel_solver_rejects_ni_abundance_above_one():
    params = list(PARAMS)
    params[5] = 0.01  # M_ni/M_ej = 0.0267, so this mixed mass is too small.
    with pytest.raises(ValueError, match=r"M_ni <= f_ni\*M_ej"):
        NickelModel().calculate_light_curve(
            tuple(params),
            Nx=40,
            Ny=160,
            t_max_days=20.0,
            density_profile="ia",
        )


@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
@pytest.mark.parametrize("nx", [20, 51, 100])
def test_common_q_cells_preserve_profile_and_nickel_mass(
    density_profile,
    nx,
):
    delta, n = 0.0, 10.0
    f_ni = 0.73
    nickel_to_ejecta = 0.04
    total_mass = _q_profile_moment(
        _PROFILE_Q_MIN,
        _PROFILE_Q_MAX,
        2.0,
        density_profile,
        delta,
        n,
    )
    q_heat = _q_mass_fraction_to_radius(
        f_ni,
        _PROFILE_Q_MIN,
        density_profile,
        delta,
        n,
    )
    mixed_mass = _q_profile_moment(
        _PROFILE_Q_MIN,
        q_heat,
        2.0,
        density_profile,
        delta,
        n,
    )
    xi0 = nickel_to_ejecta * total_mass / mixed_mass
    q_faces = np.linspace(_PROFILE_Q_MIN, _PROFILE_Q_MAX, nx + 1)

    density, source, volume = _finite_volume_q_cell_profiles(
        q_faces,
        q_heat,
        xi0,
        density_profile,
        delta,
        n,
    )

    assert np.sum(density * volume) == pytest.approx(
        total_mass, rel=3.0e-13, abs=3.0e-13
    )
    assert np.sum(source * volume) / total_mass == pytest.approx(
        nickel_to_ejecta, rel=3.0e-13, abs=3.0e-13
    )


@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
def test_q_photosphere_inverts_the_expanding_optical_depth(density_profile):
    mass = 5.0 * M_SUN
    energy = 1.0e51
    i_mass = _q_profile_moment(
        _PROFILE_Q_MIN, 1.0, 2.0, density_profile, 0.0, 10.0
    )
    i_kin = _q_profile_moment(
        _PROFILE_Q_MIN, 1.0, 4.0, density_profile, 0.0, 10.0
    )
    rho_scale, _, radius_scale = _finite_profile_scales(
        mass, energy, R_SUN, 1.0, i_mass, i_kin
    )
    tau_scale = 0.2 * rho_scale * radius_scale
    expansion = np.array([1.0, 2.0e4, 1.0e5, 4.0e5])
    q_ph = _q_photosphere_radius(
        expansion,
        tau_scale,
        density_profile,
        delta=0.0,
        n=10.0,
    )

    assert np.all(np.diff(q_ph) <= 0.0)
    for f_r, q_value in zip(expansion, q_ph):
        optical_depth = (
            tau_scale
            / f_r**2
            * _q_profile_moment(
                q_value,
                _PROFILE_Q_MAX,
                0.0,
                density_profile,
                0.0,
                10.0,
            )
        )
        if q_value > _PROFILE_Q_MIN * (1.0 + 1.0e-12):
            assert optical_depth == pytest.approx(
                _PHOTOSPHERE_TAU, rel=5.0e-6
            )
        else:
            assert optical_depth <= _PHOTOSPHERE_TAU


@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
def test_moving_photosphere_cut_cells_conserve_source_and_internal_flux(
    density_profile,
):
    q_faces = np.linspace(_PROFILE_Q_MIN, _PROFILE_Q_MAX, 61)
    q_ph = np.array([0.91, 0.53, 0.17, _PROFILE_Q_MIN])
    q_heat = _q_mass_fraction_to_radius(
        0.8, _PROFILE_Q_MIN, density_profile, 0.0, 10.0
    )
    xi0 = 0.23
    (
        thermal_upper,
        volume,
        density,
        centres,
        source,
        direct_fraction,
        active_count,
    ) = _photospheric_cell_geometry(
        q_faces,
        q_heat,
        xi0,
        q_ph,
        density_profile,
        0.0,
        10.0,
    )
    total_source = xi0 * _q_profile_moment(
        _PROFILE_Q_MIN, q_heat, 2.0, density_profile, 0.0, 10.0
    )
    np.testing.assert_allclose(
        np.sum(source * volume, axis=1) + direct_fraction * total_source,
        total_source,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    lower_faces = q_faces[:-1]
    for time_index, count in enumerate(active_count):
        if count == 0:
            assert np.all(volume[time_index] == 0.0)
            continue
        assert thermal_upper[time_index, count - 1] == pytest.approx(
            q_ph[time_index], rel=0.0, abs=2.0e-15
        )
        assert np.all(thermal_upper[time_index, count:] <= lower_faces[count:])
        assert np.all(source[time_index, count:] == 0.0)

    lower, upper, _, _ = _photospheric_transport_coefficients(
        q_faces,
        q_ph,
        np.array([1.0, 2.0, 4.0, 8.0]),
        tau_scale=3.0,
        thermal_volume=volume,
        density=density,
        centres=centres,
        boundary_density=_eta_q(q_ph, density_profile, 0.0, 10.0),
        active_count=active_count,
        dy=0.17,
    )
    for time_index, count in enumerate(active_count):
        if count < 2:
            continue
        state = np.sin(np.linspace(0.2, 2.1, count)) + 1.4
        operator = -(lower[time_index, :count] + upper[time_index, :count]) * state
        operator[1:] += lower[time_index, 1:count] * state[:-1]
        operator[:-1] += upper[time_index, : count - 1] * state[1:]
        weighted = volume[time_index, :count] * operator
        assert np.sum(weighted) == pytest.approx(
            0.0,
            abs=5.0e-14 * np.sum(np.abs(weighted)),
        )


def test_moving_photosphere_one_step_closes_discrete_energy_budget():
    q_faces = np.linspace(_PROFILE_Q_MIN, _PROFILE_Q_MAX, 31)
    q_ph = np.array([0.82, 0.76])
    q_heat = 0.70
    xi0 = 0.21
    dy = 0.013
    f_r = np.array([2.0, 2.1])
    heat = np.array([1.0, 0.91])
    (
        _,
        volume,
        density,
        centres,
        source,
        _,
        active_count,
    ) = _photospheric_cell_geometry(
        q_faces,
        q_heat,
        xi0,
        q_ph,
        "broken_power_law",
        0.0,
        10.0,
    )
    lower, upper, boundary, _ = _photospheric_transport_coefficients(
        q_faces,
        q_ph,
        f_r,
        tau_scale=3.2,
        thermal_volume=volume,
        density=density,
        centres=centres,
        boundary_density=_eta_q(q_ph, "broken_power_law", 0.0, 10.0),
        active_count=active_count,
        dy=dy,
    )
    count_now = int(active_count[0])
    count_next = int(active_count[1])
    e_now = np.zeros(q_faces.size - 1)
    e_now[:count_now] = np.linspace(1.4, 0.7, count_now)

    matrix = np.eye(count_next)
    rhs = np.empty(count_next)
    for cell in range(count_next):
        lo = lower[1, cell]
        up = upper[1, cell]
        edge = boundary[1, cell]
        matrix[cell, cell] += f_r[1] * (lo + up + edge)
        if cell > 0:
            matrix[cell, cell - 1] -= f_r[1] * lo
        if cell + 1 < count_next:
            matrix[cell, cell + 1] -= f_r[1] * up
        rhs[cell] = (
            e_now[cell]
            + dy * f_r[1] * heat[1] * source[1, cell]
        )
    e_next = np.linalg.solve(matrix, rhs)

    stored_now = np.sum(volume[0] * e_now)
    stored_next = np.sum(volume[1, :count_next] * e_next)
    swept = np.sum((volume[0] - volume[1]) * e_now)
    source_added = (
        dy
        * f_r[1]
        * heat[1]
        * np.sum(volume[1, :count_next] * source[1, :count_next])
    )
    boundary_loss = (
        f_r[1]
        * volume[1, count_next - 1]
        * boundary[1, count_next - 1]
        * e_next[count_next - 1]
    )

    assert stored_next - stored_now == pytest.approx(
        -swept + source_added - boundary_loss,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_photospheric_coefficients_use_each_local_time_step():
    q_faces = np.linspace(_PROFILE_Q_MIN, _PROFILE_Q_MAX, 21)
    q_ph = np.array([0.92, 0.81, 0.66])
    (
        _,
        volume,
        density,
        centres,
        _,
        _,
        active_count,
    ) = _photospheric_cell_geometry(
        q_faces,
        q_heat=0.7,
        xi0=0.2,
        q_photosphere=q_ph,
        density_profile="uniform",
        delta=0.0,
        n=10.0,
    )
    common = dict(
        q_faces=q_faces,
        q_photosphere=q_ph,
        expansion_factor=np.array([1.0, 1.1, 1.2]),
        tau_scale=3.0,
        thermal_volume=volume,
        density=density,
        centres=centres,
        boundary_density=np.ones(q_ph.shape),
        active_count=active_count,
    )
    unit = _photospheric_transport_coefficients(**common, dy=1.0)
    steps = np.array([0.01, 0.04, 0.09])
    variable = _photospheric_transport_coefficients(**common, dy=steps)

    for unit_coefficients, variable_coefficients in zip(unit[:3], variable[:3]):
        np.testing.assert_allclose(
            variable_coefficients,
            unit_coefficients * steps[:, None],
            rtol=2.0e-15,
            atol=0.0,
        )
    # The luminosity conversion coefficient is an instantaneous boundary
    # transport and therefore must remain independent of the integration step.
    np.testing.assert_allclose(variable[3], unit[3], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
def test_finite_profiles_recover_input_mass_energy_and_outer_radius(density_profile):
    mass = 3.0e33
    energy = 8.0e50
    outer_radius = 2.5e13
    x_max = _PROFILE_Q_MAX
    i_mass = _q_profile_moment(
        _PROFILE_Q_MIN, x_max, 2.0, density_profile, 0.0, 10.0
    )
    i_kin = _q_profile_moment(
        _PROFILE_Q_MIN, x_max, 4.0, density_profile, 0.0, 10.0
    )

    rho_scale, velocity_scale, radius_scale = _finite_profile_scales(
        mass,
        energy,
        outer_radius,
        x_max,
        i_mass,
        i_kin,
    )
    recovered_mass = 4.0 * PI * rho_scale * radius_scale**3 * i_mass
    recovered_energy = (
        2.0 * PI * rho_scale * radius_scale**3 * velocity_scale**2 * i_kin
    )

    assert radius_scale * x_max == pytest.approx(outer_radius, rel=1.0e-15)
    assert recovered_mass == pytest.approx(mass, rel=1.0e-15)
    assert recovered_energy == pytest.approx(energy, rel=1.0e-15)


def test_broken_power_law_light_curve_is_positive_and_differs_from_uniform():
    model = NickelModel()
    bpl = model.calculate_light_curve(
        PARAMS,
        Nx=60,
        Ny=300,
        t_max_days=30.0,
        density_profile="bpl",
    )
    uniform = model.calculate_light_curve(
        PARAMS,
        Nx=60,
        Ny=300,
        t_max_days=30.0,
        density_profile="uniform",
    )

    for values in bpl:
        assert np.all(np.isfinite(values))
    assert np.all(bpl[1] > 0.0)
    assert np.all(uniform[1] > 0.0)
    assert not np.allclose(bpl[1], uniform[1], rtol=1.0e-3, atol=0.0)


def test_exponential_light_curve_is_positive_distinct_and_has_exact_alias():
    model = NickelModel()
    exponential = model.calculate_light_curve(
        PARAMS,
        Nx=60,
        Ny=300,
        t_max_days=30.0,
        density_profile="exponential",
    )
    alias = model.calculate_light_curve(
        PARAMS,
        Nx=60,
        Ny=300,
        t_max_days=30.0,
        density_profile="exp",
    )
    ia_alias = model.calculate_light_curve(
        PARAMS,
        Nx=60,
        Ny=300,
        t_max_days=30.0,
        density_profile="ia",
    )
    uniform = model.calculate_light_curve(
        PARAMS,
        Nx=60,
        Ny=300,
        t_max_days=30.0,
        density_profile="uniform",
    )

    for values in exponential:
        assert np.all(np.isfinite(values))
    assert np.all(exponential[1] > 0.0)
    for canonical_values, alias_values in zip(exponential, alias):
        np.testing.assert_allclose(canonical_values, alias_values, rtol=0.0, atol=0.0)
    for canonical_values, alias_values in zip(exponential, ia_alias):
        np.testing.assert_allclose(canonical_values, alias_values, rtol=0.0, atol=0.0)
    assert not np.allclose(exponential[1], uniform[1], rtol=1.0e-3, atol=0.0)


@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
def test_default_fully_thin_tail_equals_deposited_heating(density_profile):
    state = NickelModel().calculate_transport(
        THIN_TAIL_PARAMS,
        Nx=80,
        Ny=1000,
        t_max_days=300.0,
        density_profile=density_profile,
    )
    t_days = state.t_s / DAY
    fully_thin = ~state.photosphere_valid

    assert np.any(fully_thin)
    np.testing.assert_allclose(
        state.Ldirect[fully_thin],
        _tail_deposited_heating(t_days[fully_thin], THIN_TAIL_PARAMS),
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        state.Lbol[fully_thin], state.Ldirect[fully_thin], rtol=0.0, atol=0.0
    )
    assert np.all(state.Lphotospheric[fully_thin] == 0.0)
    assert np.all(np.isnan(state.Tph[fully_thin]))
    assert np.all(np.isnan(state.Rph[fully_thin]))


@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
def test_transport_is_independent_of_homologous_temperature_floor(
    density_profile,
):
    low_floor = list(THIN_TAIL_PARAMS)
    high_floor = list(THIN_TAIL_PARAMS)
    low_floor[8] = 1000.0
    high_floor[8] = 15000.0
    common = {
        "Nx": 50,
        "Ny": 400,
        "t_max_days": 160.0,
        "density_profile": density_profile,
    }

    low = NickelModel().calculate_transport(tuple(low_floor), **common)
    high = NickelModel().calculate_transport(tuple(high_floor), **common)

    for name in (
        "t_s",
        "Lbol",
        "Lphotospheric",
        "Ldirect",
        "q_ph",
        "Rph",
        "Tph",
        "Rhom",
        "photosphere_valid",
    ):
        np.testing.assert_allclose(
            getattr(low, name),
            getattr(high, name),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
def test_physical_photosphere_obeys_stefan_boltzmann_without_floor(
    density_profile,
):
    state = NickelModel().calculate_transport(
        THIN_TAIL_PARAMS,
        Nx=50,
        Ny=400,
        t_max_days=160.0,
        density_profile=density_profile,
    )
    valid = state.photosphere_valid
    reconstructed = (
        4.0 * PI * SIGMA_SB * state.Rph[valid] ** 2 * state.Tph[valid] ** 4
    )
    np.testing.assert_allclose(
        reconstructed, state.Lphotospheric[valid], rtol=5.0e-15, atol=0.0
    )
    np.testing.assert_allclose(
        state.Rph[valid],
        state.Rhom[valid] * state.q_ph[valid],
        rtol=3.0e-15,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
def test_photospheric_tail_converges_with_spatial_resolution(density_profile):
    model = NickelModel()
    coarse = model.calculate_light_curve(
        THIN_TAIL_PARAMS,
        Nx=100,
        Ny=1000,
        t_max_days=300.0,
        density_profile=density_profile,
    )
    fine = model.calculate_light_curve(
        THIN_TAIL_PARAMS,
        Nx=200,
        Ny=1000,
        t_max_days=300.0,
        density_profile=density_profile,
    )
    tail = fine[0] / DAY >= 50.0
    relative = np.abs(coarse[1][tail] / fine[1][tail] - 1.0)

    assert np.max(relative) < 5.0e-3


def test_calculate_light_curve_is_a_physical_transport_compatibility_wrapper():
    model = NickelModel()
    common = dict(
        Nx=60,
        Ny=600,
        t_max_days=300.0,
        density_profile="exponential",
    )
    state = model.calculate_transport(TAIL_PARAMS, **common)
    legacy_tuple = model.calculate_light_curve(TAIL_PARAMS, **common)
    for actual, expected in zip(
        legacy_tuple,
        (state.t_s, state.Lbol, state.Tph, state.Rph),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_compact_ia_radius_has_forward_finite_time_grid_from_explosion_epoch():
    params = list(PARAMS)
    params[4] = 0.01
    t_s, luminosity, temperature, radius = NickelModel().calculate_light_curve(
        tuple(params),
        Nx=60,
        Ny=300,
        t_max_days=30.0,
        density_profile="ia",
    )

    assert np.all(np.diff(t_s) > 0.0)
    assert t_s[0] / 86400.0 == pytest.approx(
        30.0 * (1.0 / 300.0) ** _TIME_GRID_POWER
    )
    assert t_s[-1] / 86400.0 == pytest.approx(30.0)
    for values in (luminosity, temperature, radius):
        assert np.all(np.isfinite(values))


def test_nickel_time_grid_is_quadratic_and_nested_under_refinement():
    coarse = _build_time_grid(7.5, 100)
    fine = _build_time_grid(7.5, 200)

    assert coarse[0] == 0.0
    assert coarse[-1] == pytest.approx(7.5)
    assert np.all(np.diff(coarse) > 0.0)
    np.testing.assert_allclose(coarse, fine[::2], rtol=0.0, atol=0.0)
    assert np.diff(coarse)[0] < np.diff(coarse)[-1]


def test_radioactive_ia_main_light_curve_is_insensitive_to_fixed_wd_radius():
    model = NickelModel()
    curves = {}
    for initial_radius in (0.01, 1.0):
        params = list(PARAMS)
        params[2] = 0.0
        params[4] = initial_radius
        curves[initial_radius] = model.calculate_light_curve(
            tuple(params),
            Nx=160,
            Ny=1000,
            t_max_days=30.0,
            density_profile="ia",
        )

    compact = curves[0.01]
    extended = curves[1.0]
    after_five_days = compact[0] / 86400.0 >= 5.0
    np.testing.assert_allclose(
        compact[1][after_five_days],
        extended[1][after_five_days],
        rtol=2.0e-2,
        atol=0.0,
    )


@pytest.mark.parametrize("density_profile", ["bpl", "exponential"])
def test_nonuniform_density_light_curves_converge_with_spatial_resolution(density_profile):
    model = NickelModel()
    coarse = model.calculate_light_curve(
        RELEASE_PARAMS,
        Nx=100,
        Ny=1500,
        t_max_days=150.0,
        density_profile=density_profile,
    )
    fine = model.calculate_light_curve(
        RELEASE_PARAMS,
        Nx=200,
        Ny=1500,
        t_max_days=150.0,
        density_profile=density_profile,
    )

    relative = np.abs(coarse[1] - fine[1]) / np.maximum(
        np.abs(fine[1]),
        1.0e-300,
    )
    # The release configuration has no injected shock-cooling energy.  Test
    # the physically informative nickel-powered interval from five days onward,
    # including the late tail that exposed the old node/control-volume mismatch.
    significant = (
        (fine[1] > 0.01 * np.max(fine[1]))
        & (fine[0] >= 5.0 * DAY)
    )
    assert np.any(significant)
    assert np.max(relative[significant]) < 0.005
    assert np.median(relative[significant]) < 0.002


def test_bpl_indices_are_validated():
    model = NickelModel()

    with pytest.raises(ValueError, match="delta"):
        model.calculate_light_curve(PARAMS + (3.0, 10.0), density_profile="bpl")
    with pytest.raises(ValueError, match="n must"):
        model.calculate_light_curve(PARAMS + (0.0, 5.0), density_profile="bpl")


def test_bpl_structure_indices_are_read_from_model_parameters():
    model = NickelModel()
    legacy_defaults = model.calculate_light_curve(
        PARAMS,
        Nx=40,
        Ny=160,
        t_max_days=20.0,
        density_profile="bpl",
    )
    explicit_defaults = model.calculate_light_curve(
        PARAMS + (0.0, 10.0),
        Nx=40,
        Ny=160,
        t_max_days=20.0,
        density_profile="bpl",
    )
    different_structure = model.calculate_light_curve(
        PARAMS + (1.0, 8.0),
        Nx=40,
        Ny=160,
        t_max_days=20.0,
        density_profile="bpl",
    )

    np.testing.assert_allclose(
        legacy_defaults[1], explicit_defaults[1], rtol=0.0, atol=0.0
    )
    assert not np.allclose(
        different_structure[1], explicit_defaults[1], rtol=1.0e-3, atol=0.0
    )


def test_legacy_auto_profile_remains_uniform_for_compatibility():
    model = NickelModel()
    canonical_legacy = (3.0, 1.0, 0.0, 0.08, 10.0, 0.2, 0.12, 0.03, 4500.0)

    auto = model.calculate_light_curve(
        canonical_legacy,
        Nx=30,
        Ny=100,
        t_max_days=5.0,
        density_profile="auto",
    )
    uniform = model.calculate_light_curve(
        canonical_legacy,
        Nx=30,
        Ny=100,
        t_max_days=5.0,
        density_profile="uniform",
    )

    for auto_values, uniform_values in zip(auto, uniform):
        np.testing.assert_allclose(auto_values, uniform_values, rtol=0.0, atol=0.0)


def test_uniform_is_the_backward_compatible_default():
    model = NickelModel()
    default = model.calculate_light_curve(PARAMS, Nx=30, Ny=100, t_max_days=5.0)
    uniform = model.calculate_light_curve(
        PARAMS,
        Nx=30,
        Ny=100,
        t_max_days=5.0,
        density_profile="uniform",
    )

    for default_values, uniform_values in zip(default, uniform):
        np.testing.assert_allclose(default_values, uniform_values, rtol=0.0, atol=0.0)
