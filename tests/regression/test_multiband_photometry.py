from __future__ import annotations

import inspect
import pickle
import warnings
import numpy as np
import pytest

import transfit as tf
import transfit.api as api
from transfit.api import _physical_constraints_lnprior
from transfit.constants import C_LIGHT, MPC
from transfit.modules.extinction import (
    apply_extinction_to_fnu_grid,
    extinction_from_dict,
    extinction_to_dict,
    normalize_extinction,
    resolve_extinction_values_mag,
)
from transfit.modules.filters import (
    filters_from_dict,
    filters_to_dict,
    mono_effective_wavelength_A,
    normalize_filters,
)
from transfit.modules.fnu import evaluate_multiband_model_fnu
from transfit.modules.io import _validate_ctx_dict
from transfit.modules.likelihood import (
    gaussian_lnlike_flux,
    gaussian_lnlike_for_observation,
    gaussian_lnlike_mag,
    gaussian_lnlike_with_nuisance,
)
from transfit.modules.magnitudes import fnu_grid_to_vega_mag_grid
from transfit.modules.photometry import evaluate_multiband_observer_output
from transfit.modules.sed import BlackbodySED, CutoffBlackbodySED, sed_from_dict, sed_to_dict
from transfit.priors import MixedBoundsPrior, UniformBoundsPrior
from transfit.samplers import run_emcee, run_zeus

MU_7P5_MPC = 29.37530631695874
MU_Z1_PLANCK15 = api._distance_modulus_from_cm(api._cosmo_luminosity_distance_cm(1.0))


PARAMS_NI = {
    "M_ej": 3.0,
    "v_ej": 1.0,
    "E_Th_in": 1.5,
    "M_ni": 0.08,
    "R_0": 120.0,
    "f_ni": 0.2,
    "kappa": 0.12,
    "kappa_gamma": 0.03,
    "T_floor": 3000.0,
}

PARAMS_CSM = {
    "M_ej": 5.0,
    "E_sn": 1.0,
    "M_csm": 1.0,
    "R_csm_out": 10000.0,
    "kappa": 0.34,
    "s": 2.0,
    "eps_sh": 0.8,
    "T_floor": 5000.0,
}

LEGACY_PARAMS_NI = {
    "M_ej": 3.0,
    "v_ej": 1.0,
    "M_ni": 0.08,
    "f_ni": 0.2,
    "kappa": 0.12,
    "kappa_gamma": 0.03,
    "T_floor": 3000.0,
}


def _fixed_bol_params():
    fixed = dict(PARAMS_NI)
    fixed.pop("T_floor")
    fixed["t_shift"] = 0.0
    return fixed


def _standard_normal_lnprob(x):
    x = np.asarray(x, float)
    return -0.5 * float(np.sum(x * x))


def test_public_nickel_and_magnetar_use_canonical_full_parameter_sets():
    nickel_names = tf.model_param_names("nickel")
    magnetar_names = tf.model_param_names("magnetar")
    magnetar_ni_names = tf.model_param_names("magnetar_ni")
    csm_names = tf.model_param_names("csm")

    assert nickel_names == [
        "M_ej",
        "v_ej",
        "E_Th_in",
        "M_ni",
        "R_0",
        "f_ni",
        "kappa",
        "kappa_gamma",
        "T_floor",
        "delta",
        "n",
    ]
    assert magnetar_names == ["M_ej", "v_ej", "E_Th_in", "P_ms", "B14", "f_mag", "R_0", "kappa", "kappa_gamma", "T_floor"]
    assert magnetar_ni_names == ["M_ej", "v_ej", "P_ms", "B14", "f_mag", "M_ni", "f_ni", "kappa", "kappa_gamma", "T_floor"]
    assert csm_names == [
        "M_ej",
        "E_sn",
        "M_csm",
        "R_csm_out",
        "kappa",
        "s",
        "n",
        "delta",
        "eps_sh",
        "T_floor",
    ]
    assert tf.model_param_names("nickel", include_t_shift=True)[-1] == "t_shift"
    assert tf.model_param_names("magnetar_ni", include_t_shift=True)[-1] == "t_shift"
    assert tf.model_param_names("interaction") == csm_names
    assert tf.model_param_names("csm", include_t_shift=True)[-1] == "t_shift"


def test_magnetar_f_mag_defaults_to_fixed_unless_prior_is_given(monkeypatch):
    def fake_default_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert "f_mag" not in prior.param_names
        assert lnprob.fixed["f_mag"] == pytest.approx(0.2)
        sample = np.empty(0, dtype=float)
        return sample.reshape(1, 0), np.array([0.0], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_default_sampler)

    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.2e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )
    fixed = {
        "M_ej": 5.0,
        "v_ej": 1.0,
        "E_Th_in": 1.0,
        "P_ms": 3.0,
        "B14": 1.0,
        "R_0": 100.0,
        "kappa": 0.2,
        "kappa_gamma": 0.03,
        "t_shift": 0.0,
    }
    res = tf.fit_bol(data=data, model="magnetar", fixed=fixed)
    assert res.fixed["f_mag"] == pytest.approx(0.2)
    assert "f_mag" not in res.param_names

    def fake_prior_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == ["f_mag"]
        sample = np.array([0.35], float)
        return sample.reshape(1, 1), np.array([0.0], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_prior_sampler)
    res = tf.fit_bol(
        data=data,
        model="magnetar",
        fixed=fixed,
        priors={"f_mag": (0.1, 0.5)},
    )
    assert res.param_names == ["f_mag"]
    assert res.best_params["f_mag"] == pytest.approx(0.35)


def test_public_forward_signatures_use_params_not_theta():
    for func in (tf.lightcurve_bol, tf.predict_bol, tf.lightcurve_multiband, tf.predict_multiband):
        params = inspect.signature(func).parameters
        assert "params" in params
        assert "theta" not in params
        assert "Nx" not in params
        assert "Ny" not in params
        assert "solver_kwargs" in params


def test_csm_forward_api_outputs_positive_finite_bolometric_curve():
    params = dict(PARAMS_CSM)
    params.pop("s")
    params.pop("T_floor")

    lc = tf.lightcurve_bol(
        model="csm",
        params=params,
        z=0.0,
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 40},
    )

    assert lc.t_days.size >= 41
    assert np.all(np.diff(lc.t_days) > 0.0)
    assert np.all(np.isfinite(lc.Lbol))
    assert np.all(lc.Lbol > 0.0)
    assert np.all(np.isfinite(lc.Teff))
    assert np.all(lc.Teff > 0.0)
    assert np.all(np.isfinite(lc.Rph))
    assert np.all(lc.Rph > 0.0)

    pred = tf.predict_bol(
        model="csm",
        params=params,
        z=0.0,
        t_days=np.array([1.0, 3.0, 10.0], float),
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 40},
    )
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0.0)


def test_csm_forward_defaults_match_explicit_ejecta_structure():
    omitted = tf.lightcurve_bol(
        model="csm",
        params=PARAMS_CSM,
        z=0.0,
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 40},
    )
    explicit_params = dict(PARAMS_CSM)
    explicit_params.update(n=10.0, delta=0.0)
    explicit = tf.lightcurve_bol(
        model="csm",
        params=explicit_params,
        z=0.0,
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 40},
    )

    np.testing.assert_array_equal(omitted.t_days, explicit.t_days)
    np.testing.assert_allclose(omitted.Lbol, explicit.Lbol, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(omitted.Teff, explicit.Teff, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(omitted.Rph, explicit.Rph, rtol=0.0, atol=0.0)

    different_params = dict(PARAMS_CSM)
    different_params.update(n=8.0, delta=1.0)
    different = tf.lightcurve_bol(
        model="csm",
        params=different_params,
        z=0.0,
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 40},
    )
    different_on_default = np.interp(
        omitted.t_days, different.t_days, different.Lbol
    )
    assert not np.allclose(omitted.Lbol, different_on_default, rtol=1.0e-3)


def test_csm_fit_defaults_fix_structure_and_explicit_priors_sample_it(monkeypatch):
    data = tf.BolometricData(
        t_days=np.array([5.0, 10.0, 15.0], float),
        y=np.array([1.0e42, 1.1e42, 9.0e41], float),
        yerr=np.full(3, 1.0e41, dtype=float),
    )
    fixed = dict(PARAMS_CSM)
    fixed.pop("T_floor")
    fixed["t_shift"] = 0.0

    def fake_default_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == []
        assert lnprob.fixed["n"] == pytest.approx(10.0)
        assert lnprob.fixed["delta"] == pytest.approx(0.0)
        return np.empty((1, 0)), np.array([0.0]), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_default_sampler)
    default_result = tf.fit_bol(data=data, model="csm", fixed=fixed)
    assert default_result.fixed["n"] == pytest.approx(10.0)
    assert default_result.fixed["delta"] == pytest.approx(0.0)

    def fake_prior_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == ["n", "delta"]
        np.testing.assert_allclose(
            prior.bounds,
            np.array([[7.0, 14.0], [0.0, 2.0]], dtype=float),
        )
        sample = np.array([8.5, 0.4], dtype=float)
        return sample.reshape(1, 2), np.array([0.0]), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_prior_sampler)
    sampled_result = tf.fit_bol(
        data=data,
        model="csm",
        priors={"n": (7.0, 14.0), "delta": (0.0, 2.0)},
        fixed=fixed,
    )
    assert sampled_result.best_params_raw["n"] == pytest.approx(8.5)
    assert sampled_result.best_params_raw["delta"] == pytest.approx(0.4)

    explicit_fixed = dict(fixed)
    explicit_fixed.update(n=10.0, delta=0.0)
    monkeypatch.setattr(api, "_run_sampler", fake_default_sampler)
    fixed_result = tf.fit_bol(data=data, model="csm", fixed=explicit_fixed)
    assert fixed_result.fixed["n"] == pytest.approx(10.0)
    assert fixed_result.fixed["delta"] == pytest.approx(0.0)


def test_csm_multiband_forward_uses_canonical_full_params():
    lc = tf.lightcurve_multiband(
        model="csm",
        params=PARAMS_CSM,
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        bands=["B"],
        y_kind="mag",
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 40},
    )

    assert lc.bands == ["B"]
    finite = np.isfinite(lc.y["B"])
    assert np.any(finite)
    assert np.all(finite[np.flatnonzero(finite)[0] :])

    pred = tf.predict_multiband(
        model="csm",
        params=PARAMS_CSM,
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        t_days=np.array([5.0, 10.0, 15.0]),
        band=np.array(["B", "B", "B"]),
        y_kind="mag",
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 40},
    )
    assert np.all(np.isfinite(pred))


def test_fit_result_uses_params_for_public_best_fit_values():
    res = api.FitResult(
        model="nickel",
        ctx=None,
        sampler="fake",
        param_names=["M_ej", "t_shift"],
        fixed={},
        all_param_names=["M_ej", "t_shift"],
        samples=np.array([[2.0, 1.25]], float),
        log_prob=np.array([0.0], float),
        meta={},
    )

    assert not hasattr(res, "best_theta")
    assert not hasattr(res, "best_theta_and_shift")
    assert not hasattr(res, "best_t_shift")
    assert not hasattr(res, "best_fit_params")
    assert not hasattr(res, "best")
    assert not hasattr(res, "median")
    assert res.best_params["t_shift"] == pytest.approx(1.25)
    assert res.best_params_raw["t_shift"] == pytest.approx(1.25)
    assert res.median_params["t_shift"] == pytest.approx(1.25)


def test_removed_sc_alias_is_rejected():
    with pytest.raises(ValueError):
        tf.model_param_names("sc_ni")

    with pytest.raises(ValueError):
        tf.model_param_names("sc_magnetar")


def test_negative_redshift_is_rejected_by_public_forward_apis():
    with pytest.raises(ValueError, match="z must be non-negative"):
        tf.lightcurve_bol(
            model="nickel",
            params=LEGACY_PARAMS_NI,
            z=-0.1,
            solver_kwargs={"Nx": 20, "Ny": 50},
            t_max_days=5.0,
        )

    with pytest.raises(ValueError, match="z must be non-negative"):
        tf.predict_multiband(
            model="nickel",
            params=PARAMS_NI,
            z=-0.1,
            distance_modulus=MU_7P5_MPC,
            filters={"B": "johnson_cousins.B"},
            t_days=np.array([1.0], float),
            band=np.array(["B"], dtype=object),
            y_kind="flux",
            solver_kwargs={"Nx": 20, "Ny": 50},
            t_max_days=5.0,
        )


def test_physical_constraints_reject_ni_mass_larger_than_ejecta():
    assert _physical_constraints_lnprior({"M_ej": 1.0, "M_ni": 1.1}) == -np.inf
    assert _physical_constraints_lnprior({"M_ej": 1.0, "M_ni": 1.0}) == pytest.approx(0.0)
    assert _physical_constraints_lnprior({"M_ej": 1.0, "M_ni": 0.1}) == pytest.approx(0.0)
    assert _physical_constraints_lnprior(
        {"M_ej": 1.0, "M_ni": 0.3, "f_ni": 0.2},
        model="nickel",
    ) == -np.inf
    assert _physical_constraints_lnprior(
        {"M_ej": 1.0, "M_ni": 0.2, "f_ni": 0.2},
        model="nickel",
    ) == pytest.approx(0.0)
    # This new constraint is specific to the nickel mass-coordinate model.
    assert _physical_constraints_lnprior(
        {"M_ej": 1.0, "M_ni": 0.3, "f_ni": 0.2},
        model="magnetar_ni",
    ) == pytest.approx(0.0)
    assert _physical_constraints_lnprior({"kappa": -0.1}) == -np.inf
    assert _physical_constraints_lnprior({"kappa_gamma": 0.0}) == -np.inf
    assert _physical_constraints_lnprior({"f_ni": 1.1}) == -np.inf
    assert _physical_constraints_lnprior({"T_floor": np.nan}) == -np.inf
    assert _physical_constraints_lnprior({"t_shift": -0.1}) == -np.inf
    assert _physical_constraints_lnprior({"t_shift": 0.0}) == pytest.approx(0.0)
    assert _physical_constraints_lnprior({"M_csm": 0.0}) == -np.inf
    assert _physical_constraints_lnprior({"E_sn": -1.0}) == -np.inf
    assert _physical_constraints_lnprior({"eps_sh": 1.1}) == -np.inf
    assert _physical_constraints_lnprior({"s": 3.0}) == -np.inf
    assert _physical_constraints_lnprior({"n": 5.0}) == -np.inf
    assert _physical_constraints_lnprior({"s": 2.0, "n": 1.5}) == -np.inf
    assert _physical_constraints_lnprior({"delta": -0.1}) == -np.inf
    assert _physical_constraints_lnprior({"delta": 3.0}) == -np.inf
    assert _physical_constraints_lnprior({"R_csm_in": 100.0, "R_csm_out": 99.0}) == -np.inf
    assert _physical_constraints_lnprior({"R_csm_in": 100.0, "R_csm_out": 1000.0}) == pytest.approx(0.0)


def test_t_shift_prior_is_non_negative():
    names, bounds = api.build_bounds("nickel", include_t_shift=True)
    i = names.index("t_shift")
    assert bounds[i, 0] == pytest.approx(0.0)

    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.0e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )
    with pytest.raises(ValueError, match="t_shift.*>= 0"):
        tf.fit_bol(
            data=data,
            model="nickel",
            priors={"t_shift": (-1.0, 5.0)},
        )


def test_fixed_model_params_are_not_limited_by_default_sampling_bounds(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == []
        sample = np.empty((1, 0), dtype=float)
        return sample, np.array([0.0], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    data = tf.MultiBandData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        band=np.array(["B", "B", "B"], dtype=object),
        y=np.array([20.0, 20.1, 20.2], float),
        yerr=np.array([0.2, 0.2, 0.2], float),
    )
    fixed = dict(PARAMS_NI)
    fixed["E_Th_in"] = 0.0
    fixed["t_shift"] = 0.0

    res = tf.fit_multiband(
        data=data,
        model="nickel",
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        y_kind="mag",
        fixed=fixed,
        model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 8.0},
    )

    assert res.sampler == "fake"
    assert res.fixed["E_Th_in"] == pytest.approx(0.0)
    assert res.param_names == []


def test_fixed_model_params_still_obey_physical_constraints():
    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.0e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )
    fixed = _fixed_bol_params()
    fixed["E_Th_in"] = -1.0

    with pytest.raises(ValueError, match="E_Th_in must be >= 0"):
        tf.fit_bol(
            data=data,
            model="nickel",
            fixed=fixed,
            model_kwargs={"Nx": 20, "Ny": 50, "t_max_days": 5.0},
        )


def test_fit_bol_rejects_unmasked_bad_luminosity_values():
    for y in (
        np.array([np.nan, 1.1e41], float),
        np.array([-1.0e41, 1.1e41], float),
    ):
        data = tf.BolometricData(
            t_days=np.array([1.0, 2.0], float),
            y=y,
            yerr=np.array([1.0e40, 1.0e40], float),
        )
        with pytest.raises(ValueError, match="positive and finite"):
            tf.fit_bol(
                data=data,
                model="nickel",
                fixed=_fixed_bol_params(),
                model_kwargs={"Nx": 20, "Ny": 50, "t_max_days": 5.0},
            )


def test_fit_bol_allows_explicit_mask_to_exclude_bad_luminosity(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        sample = np.empty((1, len(prior.param_names)), dtype=float)
        return sample, np.array([0.0], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([np.nan, 1.1e41, -1.0e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
        mask=np.array([False, True, False], bool),
    )
    res = tf.fit_bol(
        data=data,
        model="nickel",
        fixed=_fixed_bol_params(),
        model_kwargs={"Nx": 20, "Ny": 50, "t_max_days": 5.0},
    )

    assert res.samples.shape == (1, 0)


def test_fit_auto_t_max_days_covers_t_shift_prior(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        sample = np.mean(np.asarray(prior.bounds, float), axis=1)
        return sample.reshape(1, -1), np.array([0.0], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    data = tf.BolometricData(
        t_days=np.array([10.0, 180.0], float),
        y=np.array([1.0e41, 1.1e41], float),
        yerr=np.array([1.0e40, 1.0e40], float),
    )
    res = tf.fit_bol(
        data=data,
        model="nickel",
        priors={"t_shift": (0.0, 80.0)},
        sampler_kwargs={"progress": False},
        model_kwargs={"Nx": 20, "Ny": 60},
    )

    assert res.meta["model_kwargs"]["t_max_days"] == pytest.approx(280.0)
    assert res.meta["t_max_days_policy"]["t_max_days_required"] == pytest.approx(260.0)
    assert res.meta["t_max_days_policy"]["t_shift_upper"] == pytest.approx(80.0)
    assert res.meta["t_max_days_policy"]["t_max_days_auto"] is True


def test_csm_fit_bol_reuses_public_fit_path(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == ["t_shift"]
        sample = np.array([1.0], float)
        logp = lnprob(sample)
        assert np.isfinite(logp)
        return sample.reshape(1, -1), np.array([logp], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    t_days = np.array([1.0, 3.0, 6.0], float)
    y = tf.predict_bol(
        model="csm",
        params=PARAMS_CSM,
        z=0.0,
        t_days=t_days,
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 40},
    )
    fixed = dict(PARAMS_CSM)
    fixed.pop("T_floor")

    res = tf.fit_bol(
        data=tf.BolometricData(t_days=t_days, y=y, yerr=0.1 * y),
        model="csm",
        fixed=fixed,
        model_kwargs={"solver_kwargs": {"Nx": 20, "Ny": 40}, "t_max_days": 30.0},
    )

    assert res.model == "csm"
    assert res.param_names == ["t_shift"]
    assert "T_floor" not in res.all_param_names
    assert res.best_params["t_shift"] == pytest.approx(1.0)


def test_fit_rejects_explicit_t_max_days_smaller_than_t_shift_range():
    data = tf.BolometricData(
        t_days=np.array([10.0, 100.0], float),
        y=np.array([1.0e41, 1.1e41], float),
        yerr=np.array([1.0e40, 1.0e40], float),
    )

    with pytest.raises(ValueError, match="t_max_days.*too small"):
        tf.fit_bol(
            data=data,
            model="nickel",
            priors={"t_shift": (0.0, 80.0)},
            model_kwargs={"t_max_days": 150.0},
        )


def test_public_forward_rejects_nonphysical_parameters_before_solving():
    bad = dict(PARAMS_NI)
    bad["f_ni"] = 1.1
    with pytest.raises(ValueError, match="f_ni must be in \\[0, 1\\]"):
        tf.lightcurve_bol(
            model="nickel",
            params=bad,
            solver_kwargs={"Nx": 20, "Ny": 50},
            t_max_days=5.0,
        )

    bad = dict(PARAMS_NI)
    bad["kappa"] = -0.12
    with pytest.raises(ValueError, match="kappa must be > 0"):
        tf.predict_bol(
            model="nickel",
            params=bad,
            t_days=np.array([1.0], float),
            solver_kwargs={"Nx": 20, "Ny": 50},
            t_max_days=5.0,
        )

    bad = dict(PARAMS_CSM)
    bad["R_csm_out"] = 100.0
    with pytest.raises(ValueError, match="R_csm_out must be > R_csm_in"):
        tf.lightcurve_bol(
            model="csm",
            params=bad,
            solver_kwargs={"Nx": 20, "Ny": 40},
            t_max_days=20.0,
        )


def test_prior_bounds_must_be_finite():
    with pytest.raises(ValueError, match="finite lo < hi"):
        api._split_prior_specs({"M_ej": (1.0, np.inf)})

    with pytest.raises(ValueError, match="bounds must be finite"):
        UniformBoundsPrior(bounds=[[0.0, np.inf]], param_names=["x"])

    with pytest.raises(ValueError, match="bounds must be finite"):
        MixedBoundsPrior(bounds=[[0.1, np.inf]], param_names=["x"], log_flags=[True])


def test_fit_multiband_rejects_invalid_observation_mode_before_sampling(monkeypatch):
    def fail_run_sampler(*args, **kwargs):
        raise AssertionError("sampler should not run for invalid observation mode")

    monkeypatch.setattr(api, "_run_sampler", fail_run_sampler)

    fixed = dict(PARAMS_NI)
    fixed["t_shift"] = 0.0
    data = tf.MultiBandData(
        t_days=np.array([1.0], float),
        band=np.array(["B"], dtype=object),
        y=np.array([20.0], float),
        yerr=np.array([0.1], float),
    )

    with pytest.raises(ValueError, match="y_kind"):
        tf.fit_multiband(
            data=data,
            model="nickel",
            z=0.001,
            distance_modulus=MU_7P5_MPC,
            filters={"B": "johnson_cousins.B"},
            y_kind="bad",
            fixed=fixed,
            model_kwargs={"Nx": 20, "Ny": 50, "t_max_days": 5.0},
        )

    with pytest.raises(ValueError, match="mag_system"):
        tf.fit_multiband(
            data=data,
            model="nickel",
            z=0.001,
            distance_modulus=MU_7P5_MPC,
            filters={"B": "johnson_cousins.B"},
            mag_system="bad",
            fixed=fixed,
            model_kwargs={"Nx": 20, "Ny": 50, "t_max_days": 5.0},
        )


def test_model_classes_reject_values_previously_silent_clamped():
    from transfit.models.nickel import NickelModel
    from transfit.models.magnetar_ni import MagNiModel

    bad_nickel = list(PARAMS_NI.values())
    bad_nickel[5] = 1.5
    with pytest.raises(ValueError, match="f_ni"):
        NickelModel().calculate_light_curve(bad_nickel, Nx=20, Ny=50, t_max_days=5.0)

    bad_magni = list(MagNiModel._warmup_theta)
    bad_magni[4] = 1.5
    with pytest.raises(ValueError, match="f_mag"):
        MagNiModel().calculate_light_curve(bad_magni, Nx=20, Ny=50, t_max_days=5.0)

    bad_magni = list(MagNiModel._warmup_theta)
    bad_magni[5] = -0.1
    with pytest.raises(ValueError, match="M_ni"):
        MagNiModel().calculate_light_curve(bad_magni, Nx=20, Ny=50, t_max_days=5.0)

    bad_magni = list(MagNiModel._warmup_theta)
    bad_magni[6] = 1.5
    with pytest.raises(ValueError, match="f_ni"):
        MagNiModel().calculate_light_curve(bad_magni, Nx=20, Ny=50, t_max_days=5.0)


def test_public_forward_rejects_nonphysical_solver_output(monkeypatch):
    def fake_solve_state(engine, theta, *, Nx, Ny, t_max_days_obs, z):
        return (
            np.array([1.0, 2.0], float),
            np.array([1.0e41, -1.0e40], float),
            np.array([6000.0, 6000.0], float),
            np.array([1.0e14, 1.1e14], float),
        )

    monkeypatch.setattr(api, "_solve_state", fake_solve_state)
    with pytest.raises(ValueError, match="non-positive or non-finite Lbol"):
        tf.predict_bol(
            model="nickel",
            params=PARAMS_NI,
            t_days=np.array([1.0], float),
            solver_kwargs={"Nx": 20, "Ny": 50},
            t_max_days=5.0,
        )


def test_fit_treats_nonphysical_solver_output_as_impossible(monkeypatch):
    def fake_solve_state(engine, theta, *, Nx, Ny, t_max_days_obs, z):
        return (
            np.array([1.0, 2.0], float),
            np.array([1.0e41, np.nan], float),
            np.array([6000.0, 6000.0], float),
            np.array([1.0e14, 1.1e14], float),
        )

    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        sample = np.mean(np.asarray(prior.bounds, float), axis=1)
        return sample.reshape(1, -1), np.array([lnprob(sample)], float), {}, "fake"

    monkeypatch.setattr(api, "_solve_state", fake_solve_state)
    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0], float),
        y=np.array([1.0e41, 1.1e41], float),
        yerr=np.array([1.0e40, 1.0e40], float),
    )
    res = tf.fit_bol(
        data=data,
        model="nickel",
        priors={"M_ej": (1.0, 5.0)},
        fixed={
            "v_ej": 1.0,
            "E_Th_in": 1.0,
            "M_ni": 0.1,
            "R_0": 100.0,
            "f_ni": 0.2,
            "kappa": 0.12,
            "kappa_gamma": 0.03,
            "t_shift": 0.0,
        },
        sampler_kwargs={"progress": False},
    )

    assert res.log_prob[0] == -np.inf


def test_public_fit_rejects_fixed_ni_mass_larger_than_ejecta():
    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.0e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )

    with pytest.raises(ValueError, match="M_ni must be <= M_ej"):
        tf.fit_bol(
            data=data,
            model="nickel",
            fixed={"M_ej": 0.5, "M_ni": 0.6},
        )


def test_uniform_bounds_prior_normalizes_list_bounds():
    prior = UniformBoundsPrior(bounds=[[0.0, 1.0]], param_names=["x"])

    assert isinstance(prior.bounds, np.ndarray)
    assert prior.lnprior([0.5]) == pytest.approx(0.0)

    samples = prior.sample(4, rng=np.random.default_rng(123))
    assert samples.shape == (4, 1)
    assert np.all(np.isfinite(samples))
    assert np.all((samples > 0.0) & (samples < 1.0))


def test_mixed_bounds_prior_normalizes_list_bounds_and_log_flags():
    prior = MixedBoundsPrior(
        bounds=[[0.1, 1.0], [1.0, 10.0]],
        param_names=["x", "y"],
        log_flags=[True, False],
    )

    assert isinstance(prior.bounds, np.ndarray)
    assert isinstance(prior.log_flags, np.ndarray)
    assert prior.lnprior([0.5, 2.0]) == pytest.approx(-np.log(0.5))

    samples = prior.sample(4, rng=np.random.default_rng(123))
    assert samples.shape == (4, 2)
    assert np.all(np.isfinite(samples))
    assert np.all((samples[:, 0] > 0.1) & (samples[:, 0] < 1.0))
    assert np.all((samples[:, 1] > 1.0) & (samples[:, 1] < 10.0))


def test_dynesty_parallel_helpers_are_pickleable():
    from transfit.samplers.dynesty import _DynestyLogLike, _build_prior_transform

    prior = MixedBoundsPrior(
        bounds=np.array([[0.1, 1.0], [1.0, 2.0]], float),
        param_names=["x", "y"],
        log_flags=[True, False],
    )

    prior_transform = pickle.loads(
        pickle.dumps(_build_prior_transform(prior.bounds, prior.log_flags))
    )
    loglike = pickle.loads(
        pickle.dumps(_DynestyLogLike(lnprob=_standard_normal_lnprob, prior=prior))
    )

    assert prior_transform(np.array([0.0, 0.5], float)).tolist() == pytest.approx([0.1, 1.5])
    assert np.isfinite(loglike(np.array([0.5, 1.5], float)))


def test_mcmc_backends_treat_nsteps_as_production_length():
    pytest.importorskip("emcee")
    pytest.importorskip("zeus")

    prior = MixedBoundsPrior(
        bounds=np.array([[-1.0, 1.0]], float),
        param_names=["x"],
    )

    def lnprob(x):
        x = np.asarray(x, float)
        return -0.5 * float(np.sum(x * x))

    common = dict(
        lnprob=lnprob,
        prior=prior,
        nwalkers=8,
        nsteps=6,
        burnin=3,
        thin=1,
        seed=2,
        init="prior",
        robust_init=False,
        progress=False,
    )
    samples_emcee, _, _ = run_emcee(**common)
    samples_zeus, _, _ = run_zeus(**common)

    assert samples_emcee.shape == (8 * 6, 1)
    assert samples_zeus.shape == (8 * 6, 1)


def test_legacy_short_nickel_param_dict_is_still_accepted_for_forward_calls():
    out = tf.predict_multiband(
        model="nickel",
        params=LEGACY_PARAMS_NI,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        t_days=np.array([1.0, 2.0], float),
        band=np.array(["B", "B"], dtype=object),
        y_kind="flux",
        mag_system="ab",
        solver_kwargs={"Nx": 20, "Ny": 50},
        t_max_days=5.0,
    )

    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_bolometric_forward_tmax_is_public_observer_frame():
    lc = tf.lightcurve_bol(
        model="nickel",
        params=LEGACY_PARAMS_NI,
        z=1.0,
        solver_kwargs={"Nx": 20, "Ny": 50},
        t_max_days=10.0,
    )
    pred = tf.predict_bol(
        model="nickel",
        params=LEGACY_PARAMS_NI,
        z=1.0,
        t_days=np.array([9.0, 11.0], float),
        solver_kwargs={"Nx": 20, "Ny": 50},
        t_max_days=10.0,
        interp_fill="nan",
    )

    assert float(np.nanmax(lc.t_days)) == pytest.approx(10.0)
    assert np.isfinite(pred[0])
    assert np.isnan(pred[1])


def test_multiband_forward_tmax_is_public_observer_frame():
    lc = tf.lightcurve_multiband(
        model="nickel",
        params=PARAMS_NI,
        z=1.0,
        distance_modulus=MU_Z1_PLANCK15,
        filters={"B": "johnson_cousins.B"},
        bands=["B"],
        y_kind="flux",
        solver_kwargs={"Nx": 20, "Ny": 50},
        t_max_days=10.0,
    )
    pred = tf.predict_multiband(
        model="nickel",
        params=PARAMS_NI,
        z=1.0,
        distance_modulus=MU_Z1_PLANCK15,
        filters={"B": "johnson_cousins.B"},
        t_days=np.array([9.0, 11.0], float),
        band=np.array(["B", "B"], dtype=object),
        y_kind="flux",
        solver_kwargs={"Nx": 20, "Ny": 50},
        t_max_days=10.0,
        interp_fill="nan",
    )

    assert float(np.nanmax(lc.t_days)) == pytest.approx(10.0)
    assert np.isfinite(pred[0])
    assert np.isnan(pred[1])


def test_flux_output_is_independent_of_mag_system():
    t_days = np.array([1.0, 2.0, 3.0], float)
    band = np.array(["B", "B", "B"], dtype=object)

    flux_ab = tf.predict_multiband(
        model="nickel",
        params=PARAMS_NI,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        t_days=t_days,
        band=band,
        y_kind="flux",
        mag_system="ab",
        solver_kwargs={"Nx": 20, "Ny": 50},
        t_max_days=5.0,
    )
    flux_vega = tf.predict_multiband(
        model="nickel",
        params=PARAMS_NI,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        t_days=t_days,
        band=band,
        y_kind="flux",
        mag_system="vega",
        solver_kwargs={"Nx": 20, "Ny": 50},
        t_max_days=5.0,
    )

    assert np.allclose(flux_ab, flux_vega, equal_nan=True)


def test_public_multiband_signatures_hide_dl_cm():
    for func in (tf.lightcurve_multiband, tf.predict_multiband, tf.fit_multiband):
        params = inspect.signature(func).parameters
        assert "DL_cm" not in params
        assert "DL_Mpc" not in params
        assert "distance_modulus" in params


def test_photometry_pipeline_is_fnu_then_extinction_then_mag():
    filters = normalize_filters({"V": "johnson_cousins.V"})
    extinction = normalize_extinction({"mw": {"ebv": 0.04, "rv": 3.1, "law": "ccm89"}})
    sed = BlackbodySED()
    bands = ["V"]
    teff = np.array([7000.0, 6500.0], float)
    rph = np.array([4.0e14, 4.3e14], float)

    fnu_model = evaluate_multiband_model_fnu(
        sed=sed,
        filter_map=filters,
        bands=bands,
        Teff_K=teff,
        R_cm=rph,
        DL_cm=7.5 * MPC,
        z=0.01,
    )
    fnu_ext = apply_extinction_to_fnu_grid(
        fnu_model,
        filter_map=filters,
        bands=bands,
        extinction=extinction,
        z=0.01,
    )
    expected = fnu_grid_to_vega_mag_grid(
        fnu_ext,
        filter_map=filters,
        bands=bands,
    )

    out = evaluate_multiband_observer_output(
        sed=sed,
        filter_map=filters,
        bands=bands,
        Teff_K=teff,
        R_cm=rph,
        DL_cm=7.5 * MPC,
        z=0.01,
        y_kind="mag",
        mag_system="vega",
        extinction=extinction,
    )

    assert np.allclose(out, expected, equal_nan=True)


def test_cutoff_blackbody_suppresses_only_short_wavelength_flux():
    bb = BlackbodySED()
    sed = CutoffBlackbodySED(3000.0, 2.0)
    assert sed.cutoff_wavelength_A == pytest.approx(3000.0)
    assert sed.uv_slope == pytest.approx(2.0)

    wavelengths_A = np.array([2000.0, 4000.0], float)
    nu_obs = C_LIGHT / (wavelengths_A * 1.0e-8)
    teff = np.array([10000.0, 8000.0], float)
    rph = np.array([1.0e15, 1.1e15], float)

    f_bb = bb.fnu(nu_obs, teff, rph, DL_cm=7.5 * MPC, z=0.0)
    f_cut = sed.fnu(nu_obs, teff, rph, DL_cm=7.5 * MPC, z=0.0)

    assert np.allclose(f_cut[0] / f_bb[0], (2000.0 / 3000.0) ** 2)
    assert np.allclose(f_cut[1], f_bb[1])


def test_builtin_sed_config_roundtrips_cutoff_blackbody():
    sed = CutoffBlackbodySED(
        cutoff_wavelength_A=2500.0,
        uv_slope=3.0,
        min_factor=0.2,
        Tmin=2.0,
        Rmin=3.0,
    )

    cfg = sed_to_dict(sed)
    restored = sed_from_dict(cfg)

    assert cfg["name"] == "CutoffBlackbodySED"
    assert cfg["builtin"] is True
    assert cfg["params"]["cutoff_wavelength_A"] == pytest.approx(2500.0)
    assert isinstance(restored, CutoffBlackbodySED)
    assert restored.cutoff_wavelength_A == pytest.approx(2500.0)
    assert restored.uv_slope == pytest.approx(3.0)
    assert restored.min_factor == pytest.approx(0.2)
    assert restored.Tmin == pytest.approx(2.0)
    assert restored.Rmin == pytest.approx(3.0)


def test_fit_multiband_accepts_custom_sed(monkeypatch):
    class HalfFluxSED(BlackbodySED):
        def fnu(self, *args, **kwargs):
            called["sed"] = True
            return 0.5 * super().fnu(*args, **kwargs)

    called = {"sed": False}

    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == []
        sample = np.empty(0, dtype=float)
        logp = lnprob(sample)
        assert np.isfinite(logp)
        assert called["sed"] is True
        return sample.reshape(1, 0), np.array([logp], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    data = tf.MultiBandData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        band=np.array(["B", "B", "B"], dtype=object),
        y=np.array([20.0, 20.2, 20.5], float),
        yerr=np.array([0.2, 0.2, 0.2], float),
    )
    fixed = dict(PARAMS_NI)
    fixed["t_shift"] = 0.0

    res = tf.fit_multiband(
        data=data,
        model="nickel",
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        y_kind="mag",
        fixed=fixed,
        sed=HalfFluxSED(),
        model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 8.0},
    )

    assert res.param_names == []
    assert res.meta["sed"] == "HalfFluxSED"
    assert res.meta["sed_config"]["name"] == "HalfFluxSED"
    assert res.meta["sed_config"]["builtin"] is False

    with pytest.raises(ValueError, match="Pass the original SED object"):
        tf.plot.fit_multiband(res, data, n_t=5)


def test_plot_fit_multiband_reuses_saved_cutoff_sed(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == []
        sample = np.empty(0, dtype=float)
        logp = lnprob(sample)
        assert np.isfinite(logp)
        return sample.reshape(1, 0), np.array([logp], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    data = tf.MultiBandData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        band=np.array(["B", "B", "B"], dtype=object),
        y=np.array([20.0, 20.2, 20.5], float),
        yerr=np.array([0.2, 0.2, 0.2], float),
    )
    fixed = dict(PARAMS_NI)
    fixed["t_shift"] = 0.0
    sed = CutoffBlackbodySED(2500.0, 3.0, min_factor=0.2)

    res = tf.fit_multiband(
        data=data,
        model="nickel",
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        y_kind="mag",
        fixed=fixed,
        sed=sed,
        model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 8.0},
    )

    seen = {}

    def fake_lightcurve_multiband(**kwargs):
        seen["sed"] = kwargs.get("sed")
        return type(
            "LC",
            (),
            {
                "t_days": np.array([0.5, 1.0, 1.5], float),
                "bands": ["B"],
                "y": {"B": np.zeros(3, float)},
            },
        )()

    monkeypatch.setattr(api, "lightcurve_multiband", fake_lightcurve_multiband)

    fig = tf.plot.fit_multiband(res, data, n_t=5)
    assert fig is not None
    assert isinstance(seen["sed"], CutoffBlackbodySED)
    assert seen["sed"].cutoff_wavelength_A == pytest.approx(2500.0)
    assert seen["sed"].uv_slope == pytest.approx(3.0)
    assert seen["sed"].min_factor == pytest.approx(0.2)


def test_plot_fit_multiband_starts_model_at_negative_t_shift(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == []
        sample = np.empty(0, dtype=float)
        logp = lnprob(sample)
        assert np.isfinite(logp)
        return sample.reshape(1, 0), np.array([logp], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    data = tf.MultiBandData(
        t_days=np.array([0.0, 2.0, 5.0], float),
        band=np.array(["B", "B", "B"], dtype=object),
        y=np.array([20.0, 19.5, 19.0], float),
        yerr=np.array([0.2, 0.2, 0.2], float),
    )
    fixed = dict(PARAMS_NI)
    fixed["t_shift"] = 4.0

    res = tf.fit_multiband(
        data=data,
        model="nickel",
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        y_kind="mag",
        fixed=fixed,
        model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 12.0},
    )

    raw_t = np.array([0.5, 1.0, 2.0, 3.0], float)
    raw_y = np.array([21.0, 20.0, 19.0, 18.5], float)

    def fake_lightcurve_multiband(**kwargs):
        assert kwargs["bands"] == ["B"]
        return type(
            "LC",
            (),
            {"t_days": raw_t, "bands": ["B"], "y": {"B": raw_y}},
        )()

    monkeypatch.setattr(api, "lightcurve_multiband", fake_lightcurve_multiband)

    fig = tf.plot.fit_multiband(res, data, n_t=6, t_pad=1.0)
    ax = fig.axes[0]
    model_line = next(line for line in ax.lines if line.get_label() == "B best")

    assert np.asarray(model_line.get_xdata(), float).tolist() == pytest.approx((raw_t - 4.0).tolist())
    assert np.asarray(model_line.get_ydata(), float).tolist() == pytest.approx(raw_y.tolist())
    assert ax.get_xlim()[0] < raw_t[0] - 4.0
    assert np.any(np.isclose(np.asarray(ax.get_xticks(), float), 0.0))


def test_plot_fit_bol_starts_model_at_negative_t_shift(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == []
        sample = np.empty(0, dtype=float)
        logp = lnprob(sample)
        assert np.isfinite(logp)
        return sample.reshape(1, 0), np.array([logp], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    data = tf.BolometricData(
        t_days=np.array([0.0, 2.0, 5.0], float),
        y=np.array([1.0e41, 1.2e41, 1.1e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )
    fixed = dict(PARAMS_NI)
    fixed.pop("T_floor")
    fixed["t_shift"] = 3.5

    res = tf.fit_bol(
        data=data,
        model="nickel",
        z=0.001728,
        fixed=fixed,
        model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 12.0},
    )

    raw_t = np.array([0.25, 0.75, 1.5, 3.0], float)
    raw_l = np.array([0.8e41, 0.9e41, 1.0e41, 1.2e41], float)

    def fake_lightcurve_bol(**kwargs):
        return type(
            "LC",
            (),
            {
                "t_days": raw_t,
                "Lbol": raw_l,
                "Teff": np.ones_like(raw_t),
                "Rph": np.ones_like(raw_t),
            },
        )()

    monkeypatch.setattr(api, "lightcurve_bol", fake_lightcurve_bol)

    fig = tf.plot.fit_bol(res, data, n_t=6, t_pad=1.0)
    ax = fig.axes[0]
    model_line = next(line for line in ax.lines if line.get_label() == "best-fit model")

    assert np.asarray(model_line.get_xdata(), float).tolist() == pytest.approx((raw_t - 3.5).tolist())
    assert np.asarray(model_line.get_ydata(), float).tolist() == pytest.approx(raw_l.tolist())
    assert ax.get_xlim()[0] < raw_t[0] - 3.5
    assert np.any(np.isclose(np.asarray(ax.get_xticks(), float), 0.0))


def test_observation_likelihood_dispatch_is_explicit():
    y_obs = np.array([1.0, 2.0, 3.0], float)
    y_model = np.array([1.1, 1.9, 3.2], float)
    y_err = np.array([0.1, 0.2, 0.2], float)

    assert gaussian_lnlike_for_observation(
        y_kind="flux",
        y_obs=y_obs,
        y_model=y_model,
        y_err=y_err,
    ) == pytest.approx(gaussian_lnlike_flux(y_obs, y_model, y_err))
    assert gaussian_lnlike_for_observation(
        y_kind="mag",
        y_obs=y_obs,
        y_model=y_model,
        y_err=y_err,
    ) == pytest.approx(gaussian_lnlike_mag(y_obs, y_model, y_err))


def test_sigma_int_likelihood_preserves_legacy_behavior_when_disabled():
    y_obs = np.array([1.0, 2.0, 3.0], float)
    y_model = np.array([1.1, 1.9, 3.2], float)
    y_err = np.array([0.1, 0.2, 0.2], float)

    assert gaussian_lnlike_with_nuisance(
        y_kind="flux",
        y_obs=y_obs,
        y_model=y_model,
        y_err=y_err,
        nuisance_params=None,
    ) == pytest.approx(
        gaussian_lnlike_for_observation(
            y_kind="flux",
            y_obs=y_obs,
            y_model=y_model,
            y_err=y_err,
        )
    )


def test_sigma_int_likelihood_uses_mag_scatter_and_normalization():
    y_obs = np.array([20.0, 20.2, 20.4], float)
    y_model = np.array([20.1, 20.3, 20.1], float)
    y_err = np.array([0.05, 0.06, 0.07], float)
    sigma_int = 0.2

    var_mag = y_err * y_err + sigma_int * sigma_int
    expected_mag = -0.5 * np.sum(
        ((y_obs - y_model) ** 2) / var_mag + np.log(2.0 * np.pi * var_mag)
    )
    assert gaussian_lnlike_with_nuisance(
        y_kind="mag",
        y_obs=y_obs,
        y_model=y_model,
        y_err=y_err,
        nuisance_params={"sigma_int": sigma_int},
    ) == pytest.approx(expected_mag)

    flux_obs = np.array([1.0e-27, 2.0e-27, 3.0e-27], float)
    flux_model = np.array([1.1e-27, 1.9e-27, 3.2e-27], float)
    flux_err = np.array([0.1e-27, 0.2e-27, 0.3e-27], float)
    frac = 0.4 * np.log(10.0) * sigma_int
    var_flux = flux_err * flux_err + (frac * np.abs(flux_obs)) ** 2
    expected_flux = -0.5 * np.sum(
        ((flux_obs - flux_model) ** 2) / var_flux + np.log(2.0 * np.pi * var_flux)
    )
    assert gaussian_lnlike_with_nuisance(
        y_kind="flux",
        y_obs=flux_obs,
        y_model=flux_model,
        y_err=flux_err,
        nuisance_params={"sigma_int": sigma_int},
    ) == pytest.approx(expected_flux)


def test_fit_multiband_can_sample_sigma_int_as_likelihood_parameter(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == ["sigma_int"]
        assert prior.log_flags.tolist() == [True]
        assert prior.bounds[0, 0] == pytest.approx(0.01)
        assert prior.bounds[0, 1] == pytest.approx(1.0)
        sample = np.array([0.2], float)
        lnprob = pickle.loads(pickle.dumps(lnprob))
        logp = lnprob(sample)
        assert np.isfinite(logp)
        return sample.reshape(1, -1), np.array([logp], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    data = tf.MultiBandData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        band=np.array(["B", "B", "B"], dtype=object),
        y=np.array([20.0, 20.2, 20.5], float),
        yerr=np.array([0.05, 0.05, 0.05], float),
    )
    fixed = dict(PARAMS_NI)
    fixed["t_shift"] = 0.0

    res = tf.fit_multiband(
        data=data,
        model="nickel",
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        y_kind="mag",
        priors={"sigma_int": ["log10", -2.0, 0.0]},
        fixed=fixed,
        model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 8.0},
    )

    assert res.param_names == ["sigma_int"]
    assert "sigma_int" not in res.all_param_names
    assert res.best_params_raw["sigma_int"] == pytest.approx(0.2)
    assert res.meta["log_prior_names"] == ["sigma_int"]
    assert res.meta["nuisance_priors"]["sigma_int"]["sampled"] is True
    assert res.meta["likelihood"] == "gaussian_with_sigma_int_mag"


def test_fit_bol_can_sample_sigma_int_as_likelihood_parameter(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == ["sigma_int"]
        assert prior.log_flags.tolist() == [True]
        sample = np.array([0.3], float)
        lnprob = pickle.loads(pickle.dumps(lnprob))
        logp = lnprob(sample)
        assert np.isfinite(logp)
        return sample.reshape(1, -1), np.array([logp], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.2e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )
    fixed = dict(PARAMS_NI)
    fixed.pop("T_floor")
    fixed["t_shift"] = 0.0

    res = tf.fit_bol(
        data=data,
        model="nickel",
        priors={"sigma_int": ["log10", -2.0, 0.30103]},
        fixed=fixed,
        model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 8.0},
    )

    assert res.param_names == ["sigma_int"]
    assert "sigma_int" not in res.all_param_names
    assert res.best_params_raw["sigma_int"] == pytest.approx(0.3)
    assert res.meta["log_prior_names"] == ["sigma_int"]


def test_sigma_int_prior_accepts_mapping_form_without_becoming_model_param():
    priors_model, fixed_model, cfgs = api._split_likelihood_nuisance_fit_inputs(
        priors={
            "M_ej": (1.0, 5.0),
            "sigma_int": {"bounds": [-2.0, 0.30103], "scale": "log10"},
        },
        fixed={"sigma_int": 0.2},
    )
    cfg = cfgs["sigma_int"]

    assert priors_model == {"M_ej": (1.0, 5.0)}
    assert fixed_model == {}
    assert cfg["enabled"] is True
    assert cfg["sampled"] is False
    assert cfg["fixed"] is True
    assert cfg["value"] == pytest.approx(0.2)
    assert cfg["log_flag"] is True
    assert cfg["bounds"][0] == pytest.approx(0.01)
    assert cfg["bounds"][1] == pytest.approx(2.0, rel=1.0e-5)


def test_fit_rejects_misspelled_sigma_int_prior():
    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.2e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )
    fixed = dict(PARAMS_NI)
    fixed.pop("T_floor")
    fixed["t_shift"] = 0.0

    with pytest.raises(KeyError, match="simga_int"):
        tf.fit_bol(
            data=data,
            model="nickel",
            priors={"simga_int": ["log10", -2.0, 0.0]},
            fixed=fixed,
            model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 8.0},
        )


def test_custom_filter_accepts_effective_wavelength_in_angstrom():
    filters = normalize_filters({"custom_g": {"lambda_eff_A": 4770.0}})

    profile = filters["custom_g"]
    assert profile.filter_id == "user:custom_g"
    assert profile.source == "user"
    assert profile.kind == "mono"
    assert profile.nu_eff_hz == pytest.approx(C_LIGHT / (4770.0e-8))
    assert mono_effective_wavelength_A(profile) == pytest.approx(4770.0)
    assert profile.meta["lambda_eff_A"] == pytest.approx(4770.0)


def test_custom_filter_accepts_effective_wavelength_unit_aliases():
    filters = normalize_filters(
        {
            "r_nm": {"lambda_eff_nm": 623.1},
            "i_um": {"lambda_eff_um": 0.7625},
        }
    )

    assert mono_effective_wavelength_A(filters["r_nm"]) == pytest.approx(6231.0)
    assert mono_effective_wavelength_A(filters["i_um"]) == pytest.approx(7625.0)


def test_custom_filter_keeps_vega_zero_point_and_filter_id():
    filters = normalize_filters(
        {
            "B": {
                "filter_id": "user:B_eff",
                "lambda_eff_A": 4400.0,
                "vega_zero_point_jy": 4260.0,
            },
        },
        mag_system="vega",
    )

    profile = filters["B"]
    assert profile.filter_id == "user:B_eff"
    assert profile.zero_points_jy["vega"] == pytest.approx(4260.0)


def test_custom_filter_effective_wavelength_roundtrips_through_serialization():
    filters = normalize_filters({"custom": {"lambda_eff_A": 5000.0}})
    payload = filters_to_dict(filters)

    assert payload["custom"]["lambda_eff_A"] == pytest.approx(5000.0)
    assert payload["custom"]["nu_eff_hz"] == pytest.approx(C_LIGHT / (5000.0e-8))

    restored = filters_from_dict(payload)
    assert mono_effective_wavelength_A(restored["custom"]) == pytest.approx(5000.0)


def test_custom_filter_rejects_ambiguous_effective_wavelength_keys():
    with pytest.raises(ValueError, match="Specify only one effective wavelength key"):
        normalize_filters({"custom": {"lambda_eff_A": 5000.0, "lambda_eff_nm": 500.0}})


def test_custom_filter_rejects_preset_mixed_with_effective_wavelength():
    with pytest.raises(ValueError, match="preset cannot be combined"):
        normalize_filters({"custom": {"preset": "johnson_cousins.V", "lambda_eff_A": 5500.0}})


def test_structured_extinction_component_resolves_standard_av():
    filters = normalize_filters({"V": "johnson_cousins.V"})
    extinction = normalize_extinction({"mw": {"ebv": 0.04, "rv": 3.1, "law": "ccm89"}})
    values = resolve_extinction_values_mag(
        extinction,
        filter_map=filters,
        used_bands=["V"],
        z=0.0,
    )

    assert values["V"] == pytest.approx(0.124, rel=0.08)


def test_host_extinction_uses_rest_frame_wavelength():
    filters = normalize_filters({"g": "sdss.g"})
    obs = normalize_extinction({"mw": {"ebv": 0.04, "rv": 3.1, "law": "odonnell94"}})
    host = normalize_extinction({"host": {"ebv": 0.04, "rv": 3.1, "law": "odonnell94"}})

    obs_vals = resolve_extinction_values_mag(obs, filter_map=filters, used_bands=["g"], z=0.5)
    host_vals = resolve_extinction_values_mag(host, filter_map=filters, used_bands=["g"], z=0.5)

    assert host_vals["g"] > obs_vals["g"]


def test_structured_extinction_roundtrip_and_band_map_addition():
    filters = normalize_filters({"B": "johnson_cousins.B", "V": "johnson_cousins.V"})
    extinction = normalize_extinction(
        {
            "band_map": {"B": 0.01},
            "mw": {"ebv": 0.04, "rv": 3.1, "law": "ccm89"},
        }
    )
    payload = extinction_to_dict(extinction)
    restored = extinction_from_dict(payload)
    values = resolve_extinction_values_mag(
        restored,
        filter_map=filters,
        used_bands=["B", "V"],
        z=0.0,
    )

    assert values["B"] > values["V"]
    assert values["B"] - values["V"] == pytest.approx(0.05, rel=0.4)


def test_predict_multiband_accepts_structured_extinction():
    t_days = np.array([2.0, 4.0, 6.0], float)
    band = np.array(["V", "V", "V"], dtype=object)

    baseline = tf.predict_multiband(
        model="nickel",
        params=PARAMS_NI,
        distance_modulus=MU_7P5_MPC,
        filters={"V": "johnson_cousins.V"},
        t_days=t_days,
        band=band,
        y_kind="mag",
        mag_system="vega",
        solver_kwargs={"Nx": 20, "Ny": 50},
        t_max_days=8.0,
    )
    reddened = tf.predict_multiband(
        model="nickel",
        params=PARAMS_NI,
        distance_modulus=MU_7P5_MPC,
        filters={"V": "johnson_cousins.V"},
        t_days=t_days,
        band=band,
        y_kind="mag",
        mag_system="vega",
        extinction={"mw": {"ebv": 0.04, "rv": 3.1, "law": "ccm89"}},
        solver_kwargs={"Nx": 20, "Ny": 50},
        t_max_days=8.0,
    )

    assert np.all(reddened > baseline)


def test_vega_requires_zero_points_for_all_used_bands():
    with pytest.raises(ValueError):
        tf.lightcurve_multiband(
            model="nickel",
            params=PARAMS_NI,
            z=0.001728,
            filters={"B": 6.8e14},
            bands=["B"],
            y_kind="mag",
            mag_system="vega",
            solver_kwargs={"Nx": 20, "Ny": 50},
            t_max_days=5.0,
        )


def test_explicit_distance_and_extinction_roundtrip(tmp_path):
    mb = tf.lightcurve_multiband(
        model="nickel",
        params=PARAMS_NI,
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B", "V": "johnson_cousins.V"},
        bands=["B", "V"],
        y_kind="mag",
        mag_system="vega",
        extinction={"B": 0.1, "V": 0.05},
        solver_kwargs={"Nx": 20, "Ny": 60},
        t_max_days=8.0,
    )
    idx = np.linspace(0, len(mb.t_days) - 1, 5, dtype=int)
    data = tf.MultiBandData(
        t_days=np.repeat(mb.t_days[idx], 2),
        band=np.array(["B", "V"] * len(idx), dtype=object),
        y=np.column_stack([mb.y["B"][idx], mb.y["V"][idx]]).reshape(-1),
        yerr=np.full(10, 0.1),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
            module=r"emcee\.autocorr",
        )
        res = tf.fit_multiband(
            data=data,
            model="nickel",
            z=0.001728,
            distance_modulus=MU_7P5_MPC,
            filters={"B": "johnson_cousins.B", "V": "johnson_cousins.V"},
            y_kind="mag",
            mag_system="vega",
            extinction={"B": 0.1, "V": 0.05},
            priors={
                "M_ej": (1.0, 5.0),
                "v_ej": (0.5, 2.0),
                "E_Th_in": (0.05, 8.0),
                "M_ni": (0.01, 0.2),
                "R_0": (10.0, 400.0),
                "T_floor": (2000.0, 6000.0),
            },
            fixed={"f_ni": 0.2, "kappa": 0.12, "kappa_gamma": 0.03, "t_shift": 0.0},
            sampler="emcee",
            sampler_kwargs={"nwalkers": 16, "nsteps": 10, "burnin": 2, "thin": 1, "seed": 1, "progress": False},
            model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 8.0},
        )

    path = tmp_path / "fit_multiband_photometry.npz"
    saved = tf.save(res, path=path)
    loaded = tf.load(saved)

    assert loaded["ctx"]["photometry"]["mag_system"] == "vega"
    assert loaded["ctx"]["distance"]["DL_cm"] is not None
    assert "B" in loaded["ctx"]["filters"]
    assert loaded["ctx"]["extinction"]["band_map"]["values_mag"]["B"] == pytest.approx(0.1)


def test_legacy_context_is_upgraded_to_schema_one():
    ctx = {
        "distance": {"z": 0.001728},
        "filters": {"B": 6.8e14},
        "y_kind": "mag",
    }
    out = _validate_ctx_dict(ctx)

    assert out["schema_version"] == 1
    assert out["photometry"]["mag_system"] == "ab"
    assert out["filters"]["B"]["filter_id"] == "legacy:B"
