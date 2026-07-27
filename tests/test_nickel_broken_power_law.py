from __future__ import annotations

import numpy as np
import pytest

from transfit.models.nickel import (
    _BPL_V_MAX_OVER_V_T,
    _BPL_X_MIN,
    _EXPONENTIAL_V_MAX_OVER_V_E,
    _EXPONENTIAL_X_MIN,
    NickelModel,
    _broken_power_law_integrals,
    _conservative_ni_source_profile,
    _density_mass_integral,
    _exponential_integrals,
    _finite_profile_scales,
    _mass_fraction_to_radius,
)
from transfit.constants import PI


PARAMS = (3.0, 1.0, 1.5, 0.08, 120.0, 0.2, 0.12, 0.03, 4500.0)


def test_broken_power_law_integrals_match_direct_quadrature():
    x_min, x_max, delta, n = 1.0e-3, 3.0, 0.0, 10.0
    i_mass, i_kin = _broken_power_law_integrals(x_min, x_max, delta, n)

    x = np.linspace(x_min, x_max, 200_001)
    density = np.where(x < 1.0, x ** (-delta), x ** (-n))

    assert i_mass == pytest.approx(np.trapz(x**2 * density, x), rel=2.0e-8)
    assert i_kin == pytest.approx(np.trapz(x**4 * density, x), rel=2.0e-8)


def test_exponential_integrals_match_direct_quadrature():
    x_min = _EXPONENTIAL_X_MIN
    x_max = _EXPONENTIAL_V_MAX_OVER_V_E
    i_mass, i_kin = _exponential_integrals(x_min, x_max)

    x = np.linspace(x_min, x_max, 500_001)
    density = np.exp(-x)

    assert i_mass == pytest.approx(np.trapz(x**2 * density, x), rel=2.0e-10)
    assert i_kin == pytest.approx(np.trapz(x**4 * density, x), rel=2.0e-10)


@pytest.mark.parametrize(
    ("density_profile", "x_min", "x_max"),
    [
        ("uniform", 1.0, 1.0e4),
        ("broken_power_law", _BPL_X_MIN, _BPL_V_MAX_OVER_V_T),
        ("exponential", _EXPONENTIAL_X_MIN, _EXPONENTIAL_V_MAX_OVER_V_E),
    ],
)
@pytest.mark.parametrize("f_ni", [0.0, 0.2, 0.7, 1.0])
def test_f_ni_is_a_lagrangian_mass_coordinate(
    density_profile,
    x_min,
    x_max,
    f_ni,
):
    delta, n = 0.0, 10.0
    x_ni = _mass_fraction_to_radius(
        f_ni, x_min, x_max, density_profile, delta, n
    )
    enclosed = _density_mass_integral(
        x_min, x_ni, density_profile, delta, n
    ) if x_ni > x_min else 0.0
    total = _density_mass_integral(
        x_min,
        x_max,
        density_profile,
        delta,
        n,
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
    ("density_profile", "x_min", "x_max"),
    [
        ("uniform", 1.0, 1.0e4),
        ("broken_power_law", _BPL_X_MIN, _BPL_V_MAX_OVER_V_T),
        ("exponential", _EXPONENTIAL_X_MIN, _EXPONENTIAL_V_MAX_OVER_V_E),
    ],
)
@pytest.mark.parametrize("nx", [40, 100, 160])
def test_discrete_ni_source_preserves_requested_mass(
    density_profile,
    x_min,
    x_max,
    nx,
):
    delta, n = 0.0, 10.0
    f_ni = 0.23
    nickel_to_ejecta = 0.04
    x_ni = _mass_fraction_to_radius(
        f_ni, x_min, x_max, density_profile, delta, n
    )
    total_mass = _density_mass_integral(
        x_min,
        x_max,
        density_profile,
        delta,
        n,
    )
    mixed_mass = _density_mass_integral(x_min, x_ni, density_profile, delta, n)
    xi0 = nickel_to_ejecta * total_mass / mixed_mass
    x_vals = np.linspace(x_min, x_max, nx + 1)
    source = _conservative_ni_source_profile(
        x_vals,
        x_ni,
        xi0,
        density_profile,
        delta,
        n,
    )

    x_inner = x_vals[1:-1]
    edges = np.empty(x_inner.size + 1)
    edges[0] = x_min
    edges[-1] = x_max
    edges[1:-1] = 0.5 * (x_inner[:-1] + x_inner[1:])
    cell_volume = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    represented_ni = np.sum(source[1:-1] * cell_volume)
    assert represented_ni / total_mass == pytest.approx(
        nickel_to_ejecta,
        rel=3.0e-13,
        abs=3.0e-13,
    )


@pytest.mark.parametrize(
    "density_profile",
    ["uniform", "broken_power_law", "exponential"],
)
def test_finite_profiles_recover_input_mass_energy_and_outer_radius(density_profile):
    mass = 3.0e33
    energy = 8.0e50
    outer_radius = 2.5e13

    if density_profile == "uniform":
        x_min, x_max = 1.0, 1.0e4
        i_mass = (x_max**3 - x_min**3) / 3.0
        i_kin = (x_max**5 - x_min**5) / 5.0
    elif density_profile == "broken_power_law":
        x_min, x_max = _BPL_X_MIN, _BPL_V_MAX_OVER_V_T
        i_mass, i_kin = _broken_power_law_integrals(
            x_min, x_max, delta=0.0, n=10.0
        )
    else:
        x_min, x_max = _EXPONENTIAL_X_MIN, _EXPONENTIAL_V_MAX_OVER_V_E
        i_mass, i_kin = _exponential_integrals(x_min, x_max)

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
    assert t_s[0] / 86400.0 == pytest.approx(0.1)
    assert t_s[-1] / 86400.0 == pytest.approx(30.0)
    for values in (luminosity, temperature, radius):
        assert np.all(np.isfinite(values))


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
        rtol=2.0e-3,
        atol=0.0,
    )


@pytest.mark.parametrize("density_profile", ["bpl", "exponential"])
def test_nonuniform_density_light_curves_converge_with_spatial_resolution(density_profile):
    model = NickelModel()
    coarse = model.calculate_light_curve(
        PARAMS,
        Nx=80,
        Ny=400,
        t_max_days=40.0,
        density_profile=density_profile,
    )
    fine = model.calculate_light_curve(
        PARAMS,
        Nx=160,
        Ny=400,
        t_max_days=40.0,
        density_profile=density_profile,
    )

    relative = np.abs(coarse[1] - fine[1]) / np.maximum(
        np.abs(fine[1]),
        1.0e-300,
    )
    # Relative errors before the model reaches one percent of peak are
    # dominated by division by a nearly zero luminosity.  Test convergence in
    # the physically informative light-curve interval instead.
    significant = fine[1] > 0.01 * np.max(fine[1])
    assert np.max(relative[significant]) < 0.06
    assert np.median(relative[significant]) < 0.04


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
