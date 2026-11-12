import numpy as np
import pytest

from transfit.constants import M_SUN, PI
from transfit.models import magnetar_ni, nickel


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

    t_gamma = np.sqrt(3 * 0.001 * M_SUN / (4 * PI * (2e9)**2))
    expected = -np.expm1(-(t_gamma / times) ** 2)
    np.testing.assert_allclose(sources[0][1:] / sources[1][1:], expected, rtol=1e-12)
    assert expected[19] < 0.9  # Includes epochs when the Ni term remains important.
    if model_name == "nickel":
        assert sources[0][0] == sources[1][0]  # Fully trapped at t=0.
