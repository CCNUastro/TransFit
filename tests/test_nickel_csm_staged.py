import numpy as np
import pytest
import transfit as tf
from transfit.constants import DAY, M_SUN
from transfit.models.nickel import NickelModel
from transfit.models.nickel_csm_staged import NickelCSMStagedModel

THETA = NickelCSMStagedModel._warmup_theta

def calc(theta=THETA, **kw):
    return NickelCSMStagedModel().calculate_light_curve(theta, Nx=60, Ny=600,
        t_max_days=80., return_full=True, **kw)

def test_ejecta_output_is_the_ni_boundary_source():
    r = calc()
    v = np.sqrt(2.e51/(5.*M_SUN))/1.e9
    t, light, _, _ = NickelModel().calculate_light_curve(
        (5., v, 0., .05, 10., .5, .1, .03, 5000.), Nx=60, Ny=600, t_max_days=80.)
    expected = np.interp(r['t_s'], np.r_[0., t], np.r_[0., light])
    np.testing.assert_allclose(r['L_ni_ejecta'], expected, rtol=1.e-14)
    assert np.max(r['tau_csm']) > 1.
    assert np.any(np.abs(r['L_bol']-r['L_ni_ejecta']) > .01*np.max(expected))

def test_sources_and_reservoir_energy_balance():
    ni_only = list(THETA); ni_only[11] = ni_only[12] = 0.
    r = calc(tuple(ni_only))
    assert np.all(r['L_fs'] == 0.) and np.all(r['L_rs'] == 0.)
    assert np.max(r['L_ni_ejecta']) > 0.
    assert np.max(np.abs(r['relative_energy_residual'])) < 1.e-12
    shock_only = list(THETA); shock_only[2] = 0.
    r = calc(tuple(shock_only))
    assert np.all(r['L_ni_ejecta'] == 0.) and np.max(r['L_fs']) > 0.
    assert np.max(np.abs(r['relative_energy_residual'])) < 1.e-12
    assert np.any(r['shock_active']) and np.any(~r['shock_active'])
    assert np.all(np.isfinite(r['L_bol'])) and np.all(r['L_bol'] > 0.)

def test_grid_refinement_and_boundary_sweep():
    model = NickelCSMStagedModel()
    a = model.calculate_light_curve(THETA, Nx=100, Ny=1200, t_max_days=80., return_full=True)
    b = model.calculate_light_curve(THETA, Nx=200, Ny=2400, t_max_days=80., return_full=True)
    sample = np.array([10., 20., 30., 50., 70.])
    la = np.interp(sample, a['t_day'], a['L_bol'])
    lb = np.interp(sample, b['t_day'], b['L_bol'])
    assert np.max(np.abs(la/lb-1.)) < .06
    # The unswept column vanishes continuously at shock exit.
    i = np.searchsorted(b['t_day'], b['shock_end_day'])
    assert b['tau_csm'][i] < .1
    assert np.all(b['tau_csm'][i+1:] == 0.)

def test_public_api_and_uniform_mixing_guard():
    assert tf.model_param_names('nickel_csm_staged') == [
        'M_ej', 'E_sn', 'M_ni', 'f_ni', 'M_csm', 'R_csm_in', 'R_csm_out',
        'kappa_ej', 'kappa_csm', 'kappa_gamma', 's', 'eps_fs', 'eps_rs', 'T_floor',
    ]
    p = dict(zip(tf.model_param_names('nickel_csm_staged'), THETA))
    lc = tf.lightcurve_bol(model='nickel_csm_staged', params=p, z=0., t_max_days=20.,
                           solver_kwargs={'Nx': 30, 'Ny': 100})
    assert np.all(np.isfinite(lc.Lbol))
    bad = list(THETA); bad[3] = .1
    with pytest.raises(ValueError, match='uniform-density mass'):
        calc(tuple(bad))

def test_thin_csm_recovers_ejecta_light_at_representative_times():
    p = list(THETA); p[4] = .001; p[11] = p[12] = 0.
    r = calc(tuple(p))
    times = [20., 50., 70.]
    out = np.interp(times, r['t_day'], r['L_bol'])
    emitted = np.interp(times, r['t_day'], r['L_ni_ejecta'])
    np.testing.assert_allclose(out, emitted, rtol=.02)
