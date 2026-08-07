from __future__ import annotations

import numpy as np
import pytest

import transfit as tf
import transfit.api as api
from transfit.constants import C_LIGHT, M_SUN, MPC
from transfit.modules.sed import BlackbodySED, CutoffBlackbodySED


PARAMS_NICKEL = {
    "M_ej": 3.0,
    "v_ej": 1.0,
    "E_Th_in": 1.5,
    "M_ni": 0.08,
    "R_0": 120.0,
    "f_ni": 0.2,
    "kappa": 0.12,
    "kappa_gamma": 0.03,
    "T_floor": 4500.0,
    "delta": 0.0,
    "n": 10.0,
}


def test_model_parameter_helpers_are_small_and_public():
    assert tf.model_param_names("nickel") == [
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
    assert tf.model_param_names("csm") == [
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
    assert tf.model_param_names("magnetar") == [
        "M_ej",
        "v_ej",
        "E_Th_in",
        "P_ms",
        "B14",
        "f_mag",
        "R_0",
        "kappa",
        "kappa_gamma",
        "T_floor",
    ]
    assert tf.model_param_names("magnetar_ni") == [
        "M_ej",
        "v_ej",
        "P_ms",
        "B14",
        "f_mag",
        "M_ni",
        "f_ni",
        "kappa",
        "kappa_gamma",
        "T_floor",
    ]
    assert "t_shift" in tf.param_template("nickel", include_t_shift=True)


def test_forward_bolometric_light_curve_is_finite():
    lc = tf.lightcurve_bol(
        model="nickel",
        params=PARAMS_NICKEL,
        z=0.001728,
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 80},
    )

    assert lc.t_days.ndim == 1
    assert np.all(np.diff(lc.t_days) > 0.0)
    assert np.all(np.isfinite(lc.Lbol))
    assert np.all(lc.Lbol > 0.0)
    assert np.all(np.isfinite(lc.Teff))
    assert np.all(np.isfinite(lc.Rph))


def test_nickel_bolometric_forward_is_independent_of_temperature_floor():
    params_default = dict(PARAMS_NICKEL)
    params_default.pop("T_floor")
    default = tf.lightcurve_bol(
        model="nickel",
        params=params_default,
        z=0.0,
        t_max_days=100.0,
        solver_kwargs={"Nx": 20, "Ny": 100},
    )
    explicit = tf.lightcurve_bol(
        model="nickel",
        params=dict(params_default, T_floor=4500.0),
        z=0.0,
        t_max_days=100.0,
        solver_kwargs={"Nx": 20, "Ny": 100},
    )

    np.testing.assert_allclose(default.Lbol, explicit.Lbol, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(default.Teff, explicit.Teff, rtol=0.0, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(default.Rph, explicit.Rph, rtol=0.0, atol=0.0, equal_nan=True)


def test_public_api_can_select_uniform_bpl_and_exponential_density_profiles():
    common = dict(
        model="nickel",
        params=PARAMS_NICKEL,
        z=0.001728,
        t_max_days=20.0,
    )
    default = tf.lightcurve_bol(
        **common,
        solver_kwargs={"Nx": 30, "Ny": 120},
    )
    uniform = tf.lightcurve_bol(
        **common,
        solver_kwargs={"Nx": 30, "Ny": 120, "density_profile": "uniform"},
    )
    bpl = tf.lightcurve_bol(
        **common,
        solver_kwargs={
            "Nx": 30,
            "Ny": 120,
            "density_profile": "bpl",
        },
    )
    bpl_canonical = tf.lightcurve_bol(
        **common,
        solver_kwargs={
            "Nx": 30,
            "Ny": 120,
            "density_profile": "broken_power_law",
        },
    )
    exponential = tf.lightcurve_bol(
        **common,
        solver_kwargs={
            "Nx": 30,
            "Ny": 120,
            "density_profile": "exponential",
        },
    )
    exponential_alias = tf.lightcurve_bol(
        **common,
        solver_kwargs={
            "Nx": 30,
            "Ny": 120,
            "density_profile": "exp",
        },
    )
    ia_alias = tf.lightcurve_bol(
        **common,
        solver_kwargs={
            "Nx": 30,
            "Ny": 120,
            "density_profile": "ia",
        },
    )

    np.testing.assert_allclose(default.Lbol, uniform.Lbol, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        uniform.Lphotospheric, uniform.Lbol, rtol=0.0, atol=0.0
    )
    assert np.all(uniform.Ldirect == 0.0)
    assert np.all(uniform.photosphere_valid)
    assert np.all(np.isfinite(uniform.Teff))
    assert np.all(np.isfinite(uniform.Rph))
    np.testing.assert_allclose(bpl.Lbol, bpl_canonical.Lbol, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        exponential.Lbol,
        exponential_alias.Lbol,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(exponential.Lbol, ia_alias.Lbol, rtol=0.0, atol=0.0)
    assert not np.allclose(bpl.Lbol, uniform.Lbol, rtol=1.0e-3, atol=0.0)
    assert not np.allclose(exponential.Lbol, uniform.Lbol, rtol=1.0e-3, atol=0.0)


def test_public_api_validates_model_specific_bpl_solver_options():
    assert api._resolve_solver_kwargs(
        {"density_profile": "exp"},
        model="nickel",
    )["density_profile"] == "exponential"
    for removed_key, value in (
        ("late_time_mode", "photospheric"),
        ("transport_mode", "fld"),
        ("fld_max_iterations", 8),
        ("fld_tolerance", 1.0e-7),
    ):
        with pytest.raises(KeyError, match=removed_key):
            api._resolve_solver_kwargs({removed_key: value}, model="nickel")
    with pytest.raises(KeyError, match="late_time_mode"):
        api._resolve_solver_kwargs(
            {"late_time_mode": "instant"},
            model="nickel",
        )
    with pytest.raises(KeyError, match="transport_mode"):
        api._resolve_solver_kwargs(
            {"transport_mode": "streaming"},
            model="nickel",
        )
    with pytest.raises(ValueError, match="density_profile"):
        api._resolve_solver_kwargs(
            {"density_profile": "powerlaw"},
            model="nickel",
        )
    with pytest.raises(KeyError, match="delta"):
        api._resolve_solver_kwargs(
            {"density_profile": "bpl", "delta": 3.0},
            model="nickel",
        )
    with pytest.raises(KeyError, match="n"):
        api._resolve_solver_kwargs(
            {"density_profile": "bpl", "n": 5.0},
            model="nickel",
        )
    with pytest.raises(KeyError, match="bpl_vmax_over_vt"):
        api._resolve_solver_kwargs(
            {"density_profile": "bpl", "bpl_vmax_over_vt": 4.0},
            model="nickel",
        )
    with pytest.raises(KeyError, match="time_scheme"):
        api._resolve_solver_kwargs(
            {"density_profile": "bpl", "time_scheme": "cn"},
            model="nickel",
        )
    with pytest.raises(KeyError, match="density_profile"):
        api._resolve_solver_kwargs(
            {"density_profile": "bpl"},
            model="magnetar",
        )


def test_forward_multiband_light_curve_is_finite():
    lc = tf.lightcurve_multiband(
        model="nickel",
        params=PARAMS_NICKEL,
        z=0.001728,
        distance_modulus=29.84,
        filters={"B": "johnson_cousins.B", "V": "johnson_cousins.V"},
        bands=["B", "V"],
        y_kind="mag",
        mag_system="vega",
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 80},
    )

    assert lc.bands == ["B", "V"]
    assert set(lc.y) == {"B", "V"}
    assert np.all(np.isfinite(lc.y["B"]))
    assert np.all(np.isfinite(lc.y["V"]))


def test_nickel_photospheric_floor_multiband_continues_after_photosphere_is_thin():
    params = {
        "M_ej": 1.0,
        "v_ej": np.sqrt(2.0e51 / M_SUN) / 1.0e9,
        "E_Th_in": 1.0,
        "M_ni": 0.2,
        "R_0": 1.0,
        "f_ni": 0.8,
        "kappa": 0.1,
        "kappa_gamma": 0.03,
        "T_floor": 4500.0,
        "delta": 0.0,
        "n": 10.0,
    }
    lc = tf.lightcurve_multiband(
        model="nickel",
        params=params,
        z=0.0,
        distance_modulus=0.0,
        filters={"g": "sdss.g", "r": "sdss.r", "i": "sdss.i"},
        bands=["g", "r", "i"],
        y_kind="mag",
        mag_system="ab",
        t_max_days=300.0,
        solver_kwargs={
            "Nx": 60,
            "Ny": 600,
            "density_profile": "exponential",
        },
    )

    for band in lc.bands:
        assert np.all(np.isfinite(lc.y[band]))

    bolometric = tf.lightcurve_bol(
        model="nickel",
        params=params,
        z=0.0,
        t_max_days=300.0,
        solver_kwargs={
            "Nx": 60,
            "Ny": 600,
            "density_profile": "exponential",
        },
    )
    assert np.any(~bolometric.photosphere_valid)
    np.testing.assert_allclose(
        bolometric.Lphotospheric + bolometric.Ldirect,
        bolometric.Lbol,
        rtol=0.0,
        atol=0.0,
    )
    assert np.all(np.isnan(bolometric.Teff[~bolometric.photosphere_valid]))
    assert np.all(np.isnan(bolometric.Rph[~bolometric.photosphere_valid]))
    prediction = tf.predict_multiband(
        model="nickel",
        params=params,
        z=0.0,
        distance_modulus=0.0,
        filters={"r": "sdss.r"},
        t_days=np.array([50.0, 200.0]),
        band=np.array(["r", "r"], dtype=object),
        y_kind="mag",
        mag_system="ab",
        t_max_days=300.0,
        solver_kwargs={
            "Nx": 60,
            "Ny": 600,
            "density_profile": "exponential",
        },
    )
    assert np.isfinite(prediction[0])
    assert np.isfinite(prediction[1])


def test_custom_effective_wavelength_filter_is_public():
    lc = tf.lightcurve_multiband(
        model="nickel",
        params=PARAMS_NICKEL,
        z=0.001728,
        filters={"custom_g": {"lambda_eff_A": 4770.0}},
        bands=["custom_g"],
        y_kind="mag",
        mag_system="ab",
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 80},
    )

    assert lc.bands == ["custom_g"]
    assert np.all(np.isfinite(lc.y["custom_g"]))


def test_fit_bol_and_save_load_smoke(monkeypatch, tmp_path):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert sampler == "emcee"
        assert list(prior.param_names) == []
        sample = np.empty(0, dtype=float)
        logp = lnprob(sample)
        assert np.isfinite(logp)
        return sample.reshape(1, 0), np.array([logp], float), {"fake": True}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    fixed = dict(PARAMS_NICKEL)
    fixed.pop("T_floor")
    fixed["t_shift"] = 0.0

    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.2e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )

    res = tf.fit_bol(
        data=data,
        model="nickel",
        z=0.001728,
        fixed=fixed,
        model_kwargs={
            "t_max_days": 10.0,
            "solver_kwargs": {"Nx": 20, "Ny": 80},
        },
    )

    assert res.sampler == "fake"
    assert res.best_params_raw["M_ej"] == pytest.approx(PARAMS_NICKEL["M_ej"])
    assert res.best_params_raw["t_shift"] == pytest.approx(0.0)

    out = tf.save(res, tmp_path / "fit_smoke.npz")
    loaded = tf.load(out)

    assert loaded["model"] == "nickel"
    assert loaded["samples"].shape == (1, 0)
    assert loaded["fixed"]["M_ej"] == pytest.approx(PARAMS_NICKEL["M_ej"])


def test_fit_bol_bpl_keeps_default_structure_fixed(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == []
        sample = np.empty(0, dtype=float)
        logp = lnprob(sample)
        assert np.isfinite(logp)
        return sample.reshape(1, 0), np.array([logp], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    fixed = dict(PARAMS_NICKEL)
    fixed.pop("T_floor")
    fixed.pop("delta")
    fixed.pop("n")
    fixed["t_shift"] = 0.0
    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.2e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )

    result = tf.fit_bol(
        data=data,
        model="nickel",
        z=0.001728,
        fixed=fixed,
        model_kwargs={
            "t_max_days": 10.0,
            "solver_kwargs": {
                "Nx": 20,
                "Ny": 80,
                "density_profile": "bpl",
            },
        },
    )

    solver = result.meta["model_kwargs"]["solver_kwargs"]
    assert solver == {
        "Nx": 20,
        "Ny": 80,
        "density_profile": "broken_power_law",
    }
    assert result.param_names == []
    assert result.fixed["delta"] == pytest.approx(0.0)
    assert result.fixed["n"] == pytest.approx(10.0)


def test_fit_multiband_bpl_samples_delta_and_n_when_priors_are_explicit(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == ["delta", "n"]
        np.testing.assert_allclose(
            prior.bounds,
            np.array([[0.0, 2.9], [5.1, 14.0]], dtype=float),
        )
        sample = np.array([0.4, 8.5], dtype=float)
        return sample.reshape(1, 2), np.array([0.0], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    fixed = dict(PARAMS_NICKEL)
    fixed.pop("delta")
    fixed.pop("n")
    fixed["t_shift"] = 0.0
    data = tf.MultiBandData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        band=np.array(["g", "g", "g"], object),
        y=np.array([19.0, 18.8, 18.6], float),
        yerr=np.array([0.1, 0.1, 0.1], float),
    )

    result = tf.fit_multiband(
        data=data,
        model="nickel",
        z=0.001728,
        distance_modulus=29.84,
        filters={"g": {"lambda_eff_A": 4770.0}},
        y_kind="mag",
        mag_system="ab",
        priors={"delta": (0.0, 2.9), "n": (5.1, 14.0)},
        fixed=fixed,
        model_kwargs={
            "t_max_days": 10.0,
            "solver_kwargs": {
                "Nx": 20,
                "Ny": 80,
                "density_profile": "bpl",
            },
        },
    )

    assert result.best_params_raw["delta"] == pytest.approx(0.4)
    assert result.best_params_raw["n"] == pytest.approx(8.5)


def test_fit_bol_forwards_and_records_exponential_density_profile(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == []
        sample = np.empty(0, dtype=float)
        logp = lnprob(sample)
        assert np.isfinite(logp)
        return sample.reshape(1, 0), np.array([logp], float), {}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    fixed = dict(PARAMS_NICKEL)
    fixed.pop("T_floor")
    fixed.pop("R_0")
    fixed["t_shift"] = 0.0
    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.2e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )

    result = tf.fit_bol(
        data=data,
        model="nickel",
        z=0.001728,
        fixed=fixed,
        model_kwargs={
            "t_max_days": 10.0,
            "solver_kwargs": {
                "Nx": 20,
                "Ny": 80,
                "density_profile": "ia",
            },
        },
    )

    assert result.meta["model_kwargs"]["solver_kwargs"] == {
        "Nx": 20,
        "Ny": 80,
        "density_profile": "exponential",
    }
    assert result.fixed["R_0"] == pytest.approx(0.01)


def test_fit_exponential_density_allows_fixed_radius_override_and_rejects_radius_prior():
    model_kwargs, _ = api._split_fit_model_kwargs(
        {"solver_kwargs": {"density_profile": "ia"}},
        model="nickel",
    )
    fixed = api._apply_default_fixed_model_params(
        "nickel",
        priors_model={},
        fixed_model={"R_0": 1.0},
        model_kwargs=model_kwargs,
    )
    assert fixed["R_0"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="fixes R_0 instead of sampling"):
        api._validate_density_structure_fit_configuration(
            model="nickel",
            priors={"R_0": (0.001, 0.1)},
            fixed=None,
            model_kwargs=model_kwargs,
        )


@pytest.mark.parametrize(
    ("density_profile", "expected_fixed"),
    [
        ("uniform", {"delta": 0.0, "n": 10.0}),
        ("bpl", {"delta": 0.0, "n": 10.0}),
        (
            "ia",
            {"delta": 0.0, "n": 10.0, "R_0": 0.01},
        ),
    ],
)
def test_density_profile_controls_default_fixed_structure_parameters(
    density_profile,
    expected_fixed,
):
    model_kwargs, _ = api._split_fit_model_kwargs(
        {"solver_kwargs": {"density_profile": density_profile}},
        model="nickel",
    )
    fixed = api._apply_default_fixed_model_params(
        "nickel",
        priors_model={},
        fixed_model={},
        model_kwargs=model_kwargs,
    )
    assert fixed == expected_fixed


def test_fit_rejects_active_bpl_parameters_in_uniform_mode():
    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0], float),
        y=np.array([1.0e41, 1.1e41], float),
        yerr=np.array([1.0e40, 1.0e40], float),
    )

    with pytest.raises(ValueError, match="prior.*density_profile.*bpl"):
        tf.fit_bol(
            data=data,
            model="nickel",
            priors={"delta": (0.0, 2.0)},
        )

    with pytest.raises(ValueError, match="fixed.*density_profile.*bpl"):
        tf.fit_bol(
            data=data,
            model="nickel",
            fixed={"n": 8.0},
        )


def test_fit_bol_reports_sampled_t_shift(monkeypatch):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert list(prior.param_names) == ["t_shift"]
        sample = np.array([1.25], dtype=float)
        logp = lnprob(sample)
        assert np.isfinite(logp)
        return sample.reshape(1, 1), np.array([logp], float), {"fake": True}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    fixed = dict(PARAMS_NICKEL)
    fixed.pop("T_floor")

    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.2e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )

    res = tf.fit_bol(
        data=data,
        model="nickel",
        z=0.001728,
        fixed=fixed,
        model_kwargs={
            "t_max_days": 30.0,
            "solver_kwargs": {"Nx": 20, "Ny": 80},
        },
    )

    assert res.param_names == ["t_shift"]
    assert res.best_params["t_shift"] == pytest.approx(1.25)
    assert res.best_params_raw["t_shift"] == pytest.approx(1.25)
    assert res.median_params["t_shift"] == pytest.approx(1.25)
    assert res.best_fit["params"]["t_shift"] == pytest.approx(1.25)


def test_cutoff_blackbody_suppresses_blue_flux_only():
    bb = BlackbodySED()
    sed = CutoffBlackbodySED(
        cutoff_wavelength_A=3000.0,
        uv_slope=2.0,
        min_factor=0.0,
    )

    wavelengths_A = np.array([2000.0, 4000.0], float)
    nu_obs = C_LIGHT / (wavelengths_A * 1.0e-8)
    teff = np.array([10000.0, 9000.0], float)
    rph = np.array([1.0e15, 1.1e15], float)

    f_bb = bb.fnu(nu_obs, teff, rph, DL_cm=7.5 * MPC, z=0.0)
    f_cut = sed.fnu(nu_obs, teff, rph, DL_cm=7.5 * MPC, z=0.0)

    assert np.allclose(f_cut[0] / f_bb[0], (2000.0 / 3000.0) ** 2)
    assert np.allclose(f_cut[1], f_bb[1])
