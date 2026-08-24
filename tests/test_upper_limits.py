from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy.special import log_ndtr

import transfit as tf
import transfit.api as api
from transfit.modules.likelihood import (
    DEFAULT_UPPER_LIMIT_NSIGMA,
    gaussian_lnlike_with_nuisance,
    upper_limit_gaussian_cdf_lnlike,
)
from transfit.priors import MixedBoundsPrior


MU_7P5_MPC = 29.37530631695874

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
    "t_shift": 0.0,
}


class _TimePredictor:
    def __call__(self, model_params, t_eval):
        return np.asarray(t_eval, float)


class _ArrayPredictor:
    def __init__(self, values):
        self.values = np.asarray(values, float)

    def __call__(self, model_params, t_eval):
        return self.values.copy()


def _empty_prior():
    return MixedBoundsPrior(
        bounds=np.empty((0, 2), dtype=float),
        param_names=[],
    )


def _fake_sampler(*, sampler, lnprob, prior, sampler_kwargs):
    restored = pickle.loads(pickle.dumps(lnprob))
    assert np.array_equal(restored.is_upper_limit, lnprob.is_upper_limit)
    assert np.array_equal(
        restored.upper_limit_nsigma,
        lnprob.upper_limit_nsigma,
        equal_nan=True,
    )
    sample = np.empty((1, 0), dtype=float)
    return sample, np.array([0.0], dtype=float), {}, "fake"


def test_multiband_data_normalizes_and_filters_upper_limit_flags():
    data = tf.MultiBandData(
        t_days=np.array([1.0, 2.0, 3.0]),
        band=np.array(["B", "V", "B"], dtype=object),
        y=np.array([20.0, 21.0, 22.0]),
        yerr=np.array([0.1, np.nan, 0.2]),
        mask=np.array([True, False, True]),
        is_upper_limit=np.array([False, True, True]),
        upper_limit_nsigma=np.array([np.nan, 3.0, 5.0]),
    )

    filtered = data.filtered()
    assert filtered.mask is None
    assert filtered.t_days.tolist() == [1.0, 3.0]
    assert filtered.is_upper_limit.tolist() == [False, True]
    assert np.isnan(filtered.upper_limit_nsigma[0])
    assert filtered.upper_limit_nsigma[1] == pytest.approx(5.0)

    detections = tf.MultiBandData(
        t_days=np.array([1.0, 2.0]),
        band=np.array(["B", "B"], dtype=object),
        y=np.array([20.0, 20.1]),
        yerr=np.array([0.1, 0.1]),
    )
    assert detections.is_upper_limit.tolist() == [False, False]

    all_limits = tf.MultiBandData(
        t_days=np.array([1.0, 2.0]),
        band=np.array(["B", "B"], dtype=object),
        y=np.array([20.0, 20.1]),
        yerr=np.array([np.nan, np.nan]),
        is_upper_limit=True,
        upper_limit_nsigma=3.0,
    )
    assert all_limits.is_upper_limit.tolist() == [True, True]
    assert all_limits.upper_limit_nsigma.tolist() == [3.0, 3.0]


def test_multiband_data_rejects_bad_upper_limit_shape():
    with pytest.raises(ValueError, match="is_upper_limit"):
        tf.MultiBandData(
            t_days=np.array([1.0, 2.0]),
            band=np.array(["B", "B"], dtype=object),
            y=np.array([20.0, 20.1]),
            yerr=np.array([0.1, 0.1]),
            is_upper_limit=np.array([True, False, True]),
        )


def test_flux_upper_limit_cdf_uses_nsigma_error_or_default_five_sigma():
    value = upper_limit_gaussian_cdf_lnlike(
        y_kind="flux",
        y_limit=np.array([10.0, 10.0, 10.0]),
        y_model=np.array([10.0, 12.0, 12.0]),
        y_err=np.array([np.nan, np.nan, 1.0]),
        upper_limit_nsigma=np.array([np.nan, 3.0, np.nan]),
    )

    # At the limit z=0. The second row is a reported 3-sigma limit, while the
    # final row uses its supplied one-sigma flux error of 1.
    expected_z = np.array([0.0, -0.6, -2.0])
    assert value == pytest.approx(float(np.sum(log_ndtr(expected_z))))


def test_upper_limit_cdf_rejects_error_and_nsigma_on_the_same_row():
    value = upper_limit_gaussian_cdf_lnlike(
        y_kind="flux",
        y_limit=np.array([10.0]),
        y_model=np.array([9.0]),
        y_err=np.array([1.0]),
        upper_limit_nsigma=np.array([5.0]),
    )
    assert value == -np.inf


def test_magnitude_upper_limit_cdf_has_the_correct_brightness_direction():
    limit = np.array([21.0, 21.0, 21.0])
    model = np.array([22.0, 21.0, 20.8])
    value = upper_limit_gaussian_cdf_lnlike(
        y_kind="mag",
        y_limit=limit,
        y_model=model,
        y_err=np.full(3, np.nan),
    )

    flux_ratio = 10.0 ** (-0.4 * (model - limit))
    expected = np.sum(log_ndtr(5.0 * (1.0 - flux_ratio)))
    assert value == pytest.approx(float(expected))

    fainter = upper_limit_gaussian_cdf_lnlike(
        y_kind="mag",
        y_limit=np.array([21.0]),
        y_model=np.array([22.0]),
        y_err=np.array([np.nan]),
    )
    brighter = upper_limit_gaussian_cdf_lnlike(
        y_kind="mag",
        y_limit=np.array([21.0]),
        y_model=np.array([20.0]),
        y_err=np.array([np.nan]),
    )
    assert fainter > brighter


def test_fit_lnprob_combines_detection_gaussian_and_upper_limit_cdf_with_shift():
    lnprob = api._FitLnProb(
        model="nickel",
        prior=_empty_prior(),
        names_samp=[],
        fixed={"t_shift": 3.0},
        names_all=["t_shift"],
        t_obs=np.array([1.0, 2.0]),
        y_obs=np.array([4.0, 5.0]),
        y_err=np.array([0.5, np.nan]),
        predictor=_TimePredictor(),
        likelihood_y_kind="flux",
        nuisance_cfgs={},
        is_upper_limit=np.array([False, True]),
    )

    # The shifted model is [4, 5]: zero detection residual and an upper-limit
    # model exactly on the threshold, which contributes log(Phi(0)).
    assert lnprob(np.empty(0)) == pytest.approx(float(log_ndtr(0.0)))


def test_fit_lnprob_without_upper_limits_preserves_detection_likelihood():
    y_obs = np.array([10.0, 11.0])
    y_model = np.array([9.5, 11.5])
    y_err = np.array([0.5, 0.25])
    lnprob = api._FitLnProb(
        model="nickel",
        prior=_empty_prior(),
        names_samp=[],
        fixed={},
        names_all=[],
        t_obs=np.array([1.0, 2.0]),
        y_obs=y_obs,
        y_err=y_err,
        predictor=_ArrayPredictor(y_model),
        likelihood_y_kind="flux",
        nuisance_cfgs={},
    )

    expected = gaussian_lnlike_with_nuisance(
        y_kind="flux",
        y_obs=y_obs,
        y_model=y_model,
        y_err=y_err,
        nuisance_params={},
    )
    assert lnprob(np.empty(0)) == pytest.approx(expected)


def test_fit_multiband_wires_upper_limits_and_records_metadata(monkeypatch):
    seen = {}

    def fake_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        seen["upper"] = np.asarray(lnprob.is_upper_limit, bool).copy()
        return _fake_sampler(
            sampler=sampler,
            lnprob=lnprob,
            prior=prior,
            sampler_kwargs=sampler_kwargs,
        )

    monkeypatch.setattr(api, "_run_sampler", fake_sampler)

    data = tf.MultiBandData(
        t_days=np.array([1.0, 2.0, 3.0]),
        band=np.array(["B", "B", "B"], dtype=object),
        y=np.array([20.0, 21.0, 22.0]),
        yerr=np.array([0.1, np.nan, 0.2]),
        is_upper_limit=np.array([False, True, True]),
        upper_limit_nsigma=np.array([np.nan, 3.0, np.nan]),
    )
    result = tf.fit_multiband(
        data=data,
        model="nickel",
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        y_kind="mag",
        fixed=PARAMS_NI,
        model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 8.0},
    )

    assert seen["upper"].tolist() == [False, True, True]
    assert result.meta["upper_limit_likelihood"] == "gaussian_cdf"
    assert result.meta["upper_limit_default_nsigma"] == pytest.approx(
        DEFAULT_UPPER_LIMIT_NSIGMA
    )
    assert result.meta["n_detections"] == 1
    assert result.meta["n_upper_limits"] == 2
    assert result.meta["n_upper_limits_with_error"] == 1
    assert result.meta["n_upper_limits_with_nsigma"] == 1
    assert result.meta["n_upper_limits_default_nsigma"] == 0


def test_fit_multiband_upper_limit_validation(monkeypatch):
    monkeypatch.setattr(api, "_run_sampler", _fake_sampler)

    common = dict(
        model="nickel",
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        fixed=PARAMS_NI,
        model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 8.0},
    )

    bad_detection_error = tf.MultiBandData(
        t_days=np.array([1.0, 2.0]),
        band=np.array(["B", "B"], dtype=object),
        y=np.array([20.0, 21.0]),
        yerr=np.array([np.nan, np.nan]),
        is_upper_limit=np.array([False, True]),
    )
    with pytest.raises(ValueError, match="for detections"):
        tf.fit_multiband(data=bad_detection_error, y_kind="mag", **common)

    bad_upper_error = tf.MultiBandData(
        t_days=np.array([1.0]),
        band=np.array(["B"], dtype=object),
        y=np.array([21.0]),
        yerr=np.array([0.0]),
        is_upper_limit=True,
    )
    with pytest.raises(ValueError, match="Upper-limit data.yerr"):
        tf.fit_multiband(data=bad_upper_error, y_kind="mag", **common)

    conflicting_upper_inputs = tf.MultiBandData(
        t_days=np.array([1.0]),
        band=np.array(["B"], dtype=object),
        y=np.array([21.0]),
        yerr=np.array([0.2]),
        is_upper_limit=True,
        upper_limit_nsigma=3.0,
    )
    with pytest.raises(ValueError, match="either data.yerr"):
        tf.fit_multiband(data=conflicting_upper_inputs, y_kind="mag", **common)

    nsigma_on_detection = tf.MultiBandData(
        t_days=np.array([1.0]),
        band=np.array(["B"], dtype=object),
        y=np.array([21.0]),
        yerr=np.array([0.2]),
        is_upper_limit=False,
        upper_limit_nsigma=np.array([3.0]),
    )
    with pytest.raises(ValueError, match="only be set for upper-limit"):
        tf.fit_multiband(data=nsigma_on_detection, y_kind="mag", **common)

    bad_flux_limit = tf.MultiBandData(
        t_days=np.array([1.0]),
        band=np.array(["B"], dtype=object),
        y=np.array([0.0]),
        yerr=np.array([np.nan]),
        is_upper_limit=True,
    )
    with pytest.raises(ValueError, match="positive"):
        tf.fit_multiband(data=bad_flux_limit, y_kind="flux", **common)


def test_plot_fit_multiband_marks_upper_limits(monkeypatch):
    monkeypatch.setattr(api, "_run_sampler", _fake_sampler)

    data = tf.MultiBandData(
        t_days=np.array([1.0, 2.0, 3.0]),
        band=np.array(["B", "B", "B"], dtype=object),
        y=np.array([20.0, 21.0, 22.0]),
        yerr=np.array([0.1, np.nan, np.nan]),
        is_upper_limit=np.array([False, True, True]),
    )
    result = tf.fit_multiband(
        data=data,
        model="nickel",
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        y_kind="mag",
        fixed=PARAMS_NI,
        model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 8.0},
    )

    def fake_lightcurve_multiband(**kwargs):
        return type(
            "LC",
            (),
            {
                "t_days": np.array([0.0, 2.0, 4.0]),
                "bands": ["B"],
                "y": {"B": np.array([22.0, 20.0, 21.0])},
            },
        )()

    monkeypatch.setattr(api, "lightcurve_multiband", fake_lightcurve_multiband)
    fig = tf.plot.fit_multiband(result, data, n_t=5)
    ax = fig.axes[0]
    upper_line = next(
        line for line in ax.lines if line.get_label() == "B upper limit"
    )

    assert upper_line.get_marker() == "v"
    assert np.asarray(upper_line.get_xdata(), float).tolist() == [2.0, 3.0]
    assert np.asarray(upper_line.get_ydata(), float).tolist() == [21.0, 22.0]
