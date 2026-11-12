import numpy as np
import pytest
from scipy.integrate import quad

from transfit.constants import M_SUN, PI
from transfit.models import magnetar_ni, nickel


@pytest.mark.parametrize("profile,delta", [("uniform", 0.0), ("broken_power_law", 0.0), ("broken_power_law", 0.5), ("broken_power_law", 1.0), ("exponential", 0.0)])
def test_gamma_timescale_matches_radial_column(profile, delta):
    q_min = 1e-4
    def density(q):
        if profile == "uniform":
            return 1.0
        if profile == "exponential":
            return np.exp(-12 * q)
        return (q * 3) ** (-delta if q < 1 / 3 else -10.0)

    moments = [quad(lambda q: density(q) * q**j, q_min, 1.0,
                    points=[1 / 3], epsabs=1e-12)[0] for j in (0, 2, 4)]
    expected = moments[0] * moments[2] / (4 * PI * moments[1]**2)
    actual = nickel._central_gamma_timescale(1.0, 1.0, 1.0, profile, delta, 10.0, q_min, 1.0)
    assert actual**2 == pytest.approx(expected, rel=1e-10)


def test_uniform_central_column_limit():
    actual = nickel._central_gamma_timescale(1.0, 1.0, 1.0, "uniform", 0.0, 10.0, 1e-9, 1.0)
    assert actual**2 == pytest.approx(9 / (20 * PI), rel=2e-9)


@pytest.mark.parametrize("model_name", ["nickel", "magnetar_ni"])
def test_radioactive_source_uses_full_chain_deposition(monkeypatch, model_name):
    """Check the source passed to each solver, including partial trapping."""
    sources = []
    if model_name == "nickel":
        module = nickel
        loop_name = "_fast_time_loop_uniform_legacy_numba"
        source_index = 6
        theta = [1.0, 2.0, 0.0, 0.08, 10.0, 0.5, 0.1, 0.001, 4500.0]
        gamma_index = 7
        model = nickel.NickelModel()
    else:
        module = magnetar_ni
        loop_name = "_fast_time_loop_numba"
        source_index = 8
        theta = [1.0, 2.0, 3.0, 1.0, 0.5, 0.08, 0.5, 0.1, 0.001, 4500.0]
        gamma_index = 8
        model = magnetar_ni.MagNiModel()

    def capture(*args):
        sources.append(np.array(args[source_index], copy=True))
        return np.zeros(args[0])

    monkeypatch.setattr(module, loop_name, capture)
    with np.errstate(divide="ignore", invalid="ignore"):
        times, *_ = model.calculate_light_curve(tuple(theta), Nx=10, Ny=100, t_max_days=50)
        theta[gamma_index] = 1e6
        model.calculate_light_curve(tuple(theta), Nx=10, Ny=100, t_max_days=50)

    q_min = 1e-4
    coefficient = ((1-q_min) * (1-q_min**5) / 5
                   / ((1-q_min**3) / 3)**2 / (4 * PI))
    t_gamma = np.sqrt(coefficient * 0.001 * M_SUN / (2e9)**2)
    expected = -np.expm1(-(t_gamma / times) ** 2)
    np.testing.assert_allclose(sources[0][1:] / sources[1][1:], expected, rtol=1e-12)
    assert expected[19] < 0.9  # Includes epochs when the Ni term remains important.
    if model_name == "nickel":
        assert sources[0][0] == sources[1][0]  # Fully trapped at t=0.
