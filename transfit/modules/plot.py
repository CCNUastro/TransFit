# transfit/modules/plot.py
# -*- coding: utf-8 -*-
"""
One-click plotting helpers for transfit.

Public:
  - fit(res, data, ...)
  - corner(res, ...)
  - fit_multiband(res, data, ...)
  - fit_bol(res, data, ...)

Accepted `res`:
  1) FitResult (from tf.fit_*)
  2) loaded dict (from io.load(...))

Notes:
  - This module DOES NOT read file paths.
  - It NEVER calls plt.show() / display().
    In notebooks: returning a Figure will be displayed once automatically.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List, Literal

import numpy as np
import matplotlib.pyplot as plt

from .extinction import extinction_from_dict
from .filters import filters_from_dict
from .io import _ctx_to_dict, _validate_ctx_dict
from .sed import sed_from_dict


# -----------------------------------------------------------------------------
# style (good journal-like defaults, only inside rc_context)
# -----------------------------------------------------------------------------

def _journal_rc() -> Dict[str, Any]:
    return {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    }


def _maybe_invert_mag_axis(ax, y_kind: str):
    if str(y_kind).lower() == "mag":
        ax.invert_yaxis()


def _mag_ylabel(y_kind: str, mag_system: str) -> str:
    if str(y_kind).lower() != "mag":
        return r"F$_\nu$"
    if str(mag_system).lower() == "vega":
        return "Vega mag"
    return "AB mag"


# Parameter labels with units (LaTeX-ready for matplotlib mathtext).
_PARAM_LABELS_LATEX = {
    "M_ej": r"$M_{\mathrm{ej}}\,[M_{\odot}]$",
    "v_ej": r"$v_{\mathrm{ej}}\,[10^{9}\,\mathrm{cm\,s^{-1}}]$",
    "M_ni": r"$M_{\mathrm{Ni}}\,[M_{\odot}]$",
    "f_ni": r"$f_{\mathrm{Ni}}$",
    "kappa": r"$\kappa\,[\mathrm{cm^{2}\,g^{-1}}]$",
    "kappa_gamma": r"$\kappa_{\gamma}\,[\mathrm{cm^{2}\,g^{-1}}]$",
    "T_floor": r"$T_{\mathrm{floor}}\,[\mathrm{K}]$",
    "E_Th_in": r"$E_{\mathrm{th,in}}\,[10^{49}\,\mathrm{erg}]$",
    "R_0": r"$R_{0}\,[R_{\odot}]$",
    "P_ms": r"$P_{0}\,[\mathrm{ms}]$",
    "B14": r"$B\,[10^{14}\,\mathrm{G}]$",
    "t_shift": r"$t_{\mathrm{shift}}\,[\mathrm{day}]$",
    "sigma_int": r"$\sigma_{\mathrm{int}}\,[\mathrm{mag}]$",
}


def _param_label_latex(name: str) -> str:
    n = str(name)
    if n in _PARAM_LABELS_LATEX:
        return _PARAM_LABELS_LATEX[n]
    if n.startswith("$") and n.endswith("$"):
        return n
    if "\\" in n:
        return "$" + n + "$"
    return "$" + n.replace("_", r"\_") + "$"


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _unique_in_order(seq: Sequence[Any]) -> List[Any]:
    """Unique values preserving first-seen order (case-sensitive)."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _to_loaded(res: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    Accept FitResult or loaded dict. NO PATH support.
    loaded dict recommended keys:
      samples, log_prob, param_names, all_param_names, fixed, meta, ctx, model
    """
    if isinstance(res, dict):
        loaded = dict(res)
        if "ctx" in loaded:
            loaded["ctx"] = _validate_ctx_dict(dict(loaded.get("ctx", {}) or {}))
        return loaded

    if hasattr(res, "samples") and hasattr(res, "param_names"):
        return dict(
            samples=np.asarray(res.samples, float),
            log_prob=np.asarray(getattr(res, "log_prob", None), float)
            if getattr(res, "log_prob", None) is not None
            else None,
            param_names=np.asarray(res.param_names, dtype=object),
            all_param_names=np.asarray(getattr(res, "all_param_names", []), dtype=object),
            fixed=dict(getattr(res, "fixed", {}) or {}),
            meta=dict(getattr(res, "meta", {}) or {}),
            ctx=_ctx_to_dict(getattr(res, "ctx", None)),
            model=str(getattr(res, "model", "")),
            sampler=str(getattr(res, "sampler", "")),
        )

    raise TypeError(
        "plot only accepts FitResult or loaded dict.\n"
        "If you have a .npz path, load it first via io.load(), then pass the loaded dict."
    )


def _forward_inputs_from_ctx_dict(
    ctx_dict: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Dict[str, Any], str, str, Any]:
    ctx_dict = _validate_ctx_dict(dict(ctx_dict or {}))
    dist = dict(ctx_dict.get("distance", {}) or {})
    z = dist.get("z", None)
    if z is not None:
        z = float(z)
    DL_cm = dist.get("DL_cm", None)
    distance_modulus = None
    if DL_cm is not None:
        from ..api import _distance_modulus_from_cm

        distance_modulus = _distance_modulus_from_cm(float(DL_cm))
    filters = filters_from_dict(dict(ctx_dict.get("filters", {}) or {}))
    phot = dict(ctx_dict.get("photometry", {}) or {})
    y_kind = str(phot.get("y_kind", "mag"))
    mag_system = str(phot.get("mag_system", "ab"))
    extinction = extinction_from_dict(ctx_dict.get("extinction"))
    return z, distance_modulus, filters, y_kind, mag_system, extinction


def _get_model_name(loaded: Dict[str, Any], fallback: Optional[str] = None) -> str:
    if loaded.get("model"):
        return str(loaded["model"])
    meta = dict(loaded.get("meta", {}) or {})
    if meta.get("model"):
        return str(meta["model"])
    if fallback:
        return str(fallback)
    raise ValueError("Cannot determine model name; pass model=... to fit_*().")


def _plot_sed_from_meta(loaded: Dict[str, Any]):
    meta = dict(loaded.get("meta", {}) or {})
    config = meta.get("sed_config")
    if config is None:
        return None
    return sed_from_dict(dict(config))


def _prepare_plot_model_kwargs(model_kwargs: Dict[str, Any], required_t_max_days: float) -> Dict[str, Any]:
    """
    Ensure plotting evaluation does not hit interpolation edge-clamping.
    If needed, enlarge t_max_days for forward-model generation.
    """
    mk = dict(model_kwargs or {})
    solver_kwargs = dict(mk.pop("solver_kwargs", {}) or {})
    for key in ("Nx", "Ny"):
        if key in mk:
            solver_kwargs[key] = mk.pop(key)
    if solver_kwargs:
        mk["solver_kwargs"] = solver_kwargs

    req = float(required_t_max_days)
    if not np.isfinite(req) or req <= 0.0:
        return mk

    cur_raw = mk.get("t_max_days", 150.0)
    try:
        cur = float(cur_raw)
    except Exception:
        cur = 150.0

    mk["t_max_days"] = max(cur, req + 1.0)
    return mk


def _best_subset(samples: np.ndarray, log_prob: Optional[np.ndarray], max_n: Optional[int]) -> np.ndarray:
    samples = np.asarray(samples, float)
    finite = np.all(np.isfinite(samples), axis=1)
    if log_prob is not None:
        lp = np.asarray(log_prob, float).reshape(-1)
        if lp.shape[0] == samples.shape[0]:
            finite &= np.isfinite(lp)
    samples = samples[finite]

    if max_n is None or samples.shape[0] <= int(max_n):
        return samples
    rng = np.random.default_rng(123)
    idx = rng.choice(samples.shape[0], size=int(max_n), replace=False)
    return samples[idx]


def _paramdict_from_samples(
    loaded: Dict[str, Any],
    samples: np.ndarray,
    *,
    use: str = "median",
) -> Dict[str, float]:
    """Full parameter dict including fixed + center of sampled parameters."""
    fixed = dict(loaded.get("fixed", {}) or {})
    pnames = [str(x) for x in loaded["param_names"]]
    samples = np.asarray(samples, float)

    vals = dict(fixed)
    center = np.nanmean(samples, axis=0) if use == "mean" else np.nanmedian(samples, axis=0)
    for i, n in enumerate(pnames):
        vals[n] = float(center[i])
    return vals


def _paramdict_best_sample(
    loaded: Dict[str, Any],
    samples: np.ndarray,
    log_prob: Optional[np.ndarray],
) -> Dict[str, float]:
    """Full parameter dict using the maximum-posterior sample."""
    fixed = dict(loaded.get("fixed", {}) or {})
    pnames = [str(x) for x in loaded["param_names"]]
    samp = np.asarray(samples, float)
    if samp.ndim != 2 or samp.shape[0] == 0:
        raise ValueError("No samples available for best-fit parameter extraction.")

    lp = np.asarray(log_prob, float).reshape(-1) if log_prob is not None else None
    if lp is None or lp.size != samp.shape[0]:
        idx = int(samp.shape[0] // 2)
    else:
        finite = np.isfinite(lp)
        if np.any(finite):
            irel = int(np.argmax(lp[finite]))
            idx = int(np.where(finite)[0][irel])
        else:
            idx = int(np.argmax(lp))

    vals = dict(fixed)
    for i, n in enumerate(pnames):
        vals[n] = float(samp[idx, i])
    return vals


def _params_and_shift_from_paramdict(loaded: Dict[str, Any], p: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    Model parameter dict, excluding t_shift, plus the fitted time shift.
    If all_param_names is missing, fallback to param_names order.
    """
    all_names = [str(x) for x in loaded.get("all_param_names", []) if str(x)]
    if not all_names:
        all_names = [str(x) for x in loaded["param_names"]]

    t_shift = float(p.get("t_shift", 0.0))

    model_params: Dict[str, float] = {}
    for n in [name for name in all_names if name != "t_shift"]:
        if n not in p:
            raise KeyError(f"Parameter '{n}' missing when building model parameters. Check all_param_names/fixed/param_names.")
        model_params[n] = float(p[n])

    return model_params, t_shift


def _model_zero_time_grid(
    t_obs: np.ndarray,
    t_shift: float,
    t_pad: float,
    n_t: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build a plotting grid whose physical model origin is visible on the
    observed-time axis.

    Fit convention:
        model(t_obs) = raw_model(t_obs + t_shift)

    Therefore raw model time zero belongs at x = -t_shift on a plot labelled
    "Since First Detection".
    """
    t_obs = np.asarray(t_obs, float).reshape(-1)
    t_shift = max(float(t_shift), 0.0)
    t_obs_max = float(np.nanmax(t_obs)) if t_obs.size else 0.0
    model_t_max = max(t_obs_max + t_shift, 0.0) + float(t_pad)
    u = np.linspace(0.0, 1.0, max(int(n_t), 2))
    t_model_plot = model_t_max * u * u
    t_plot = t_model_plot - t_shift
    return t_model_plot, t_plot, model_t_max


def _set_since_first_detection_xlim(ax, t_obs: np.ndarray, t_plot: np.ndarray) -> None:
    t_obs = np.asarray(t_obs, float).reshape(-1)
    t_plot = np.asarray(t_plot, float).reshape(-1)
    finite = np.concatenate([t_obs[np.isfinite(t_obs)], t_plot[np.isfinite(t_plot)]])
    if finite.size == 0:
        return

    x_min = float(np.nanmin(finite))
    x_max = float(np.nanmax(finite))
    span = max(x_max - x_min, 1.0)
    left_pad = max(1.0, 0.05 * span)
    right_pad = min(10.0, max(1.0, 0.02 * span))

    x_left = x_min - left_pad
    if x_min < 0.0:
        # Keep the model origin visible without wasting a large fraction of the
        # axis on pre-detection empty space for long light curves.
        origin_pad = max(1.0, min(5.0, 0.25 * abs(x_min), 0.02 * span))
        x_left = x_min - origin_pad
    x_right = x_max + right_pad
    ax.set_xlim(x_left, x_right)

    if x_min < 0.0:
        negative_tick = -_nice_pre_detection_tick(
            max(abs(x_min), 0.5 * abs(x_left))
        )
        min_tick_spacing = 0.08 * (x_right - x_left)
        if x_left <= negative_tick <= x_right and abs(negative_tick) >= min_tick_spacing:
            ticks = [
                float(t)
                for t in ax.get_xticks()
                if np.isfinite(t) and x_left <= float(t) <= x_right
            ]
            ticks.extend([negative_tick, 0.0])
            ticks = sorted(set(round(t, 10) for t in ticks))
            ax.set_xticks(ticks)


def _nice_pre_detection_tick(value: float) -> float:
    value = max(float(value), 0.0)
    for candidate in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        if value <= candidate:
            return candidate
    scale = 10.0 ** np.floor(np.log10(value))
    return float(np.ceil(value / scale) * scale)


def _anchor_model_origin_for_plot(
    y_line: np.ndarray,
    *,
    y_kind: str,
    y_reference: np.ndarray,
    t_plot: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Plot-only anchor for the physical model origin.

    Some model grids have no finite synthetic photometry exactly at t_model=0,
    while others return a finite value that is already close to the first data.
    The anchor is deliberately plot-only: it makes the displayed curve rise
    from the model origin without changing the fitted model or likelihood.
    """
    y = np.asarray(y_line, float).copy()
    if y.size == 0:
        return y
    finite_idx = np.flatnonzero(np.isfinite(y))
    if finite_idx.size == 0:
        return y

    kind = str(y_kind).lower()
    ref = np.asarray(y_reference, float).reshape(-1)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        ref = y[np.isfinite(y)]
    if ref.size == 0:
        return y

    if kind == "mag":
        anchor = float(np.nanmax(ref) + 0.5)
    elif kind == "flux":
        anchor = 0.0
    else:
        positive = ref[ref > 0.0]
        if positive.size:
            anchor = float(np.nanmin(positive) * 0.5)
        else:
            return y

    first = int(finite_idx[0])
    ramp_end = first
    if t_plot is not None:
        x = np.asarray(t_plot, float).reshape(-1)
        if x.size == y.size:
            nonnegative_finite = np.flatnonzero((x >= 0.0) & np.isfinite(y))
            if nonnegative_finite.size:
                ramp_end = int(nonnegative_finite[0])

    if ramp_end > 0 and np.isfinite(y[ramp_end]):
        y[:ramp_end] = np.linspace(anchor, y[ramp_end], ramp_end + 1)[:-1]
    else:
        y[0] = anchor
    return y


# -----------------------------------------------------------------------------
# public: corner
# -----------------------------------------------------------------------------

def fit(res, data, **kwargs):
    """
    User-facing plot entry point.

    Dispatches to the bolometric or multi-band plot helper based on `data`.
    """
    if hasattr(data, "band"):
        return fit_multiband(res, data, **kwargs)
    return fit_bol(res, data, **kwargs)

def corner(
    res: Union[Dict[str, Any], Any],
    *,
    truths: Optional[Sequence[float]] = None,
    q: Tuple[float, float] = (0.005, 0.995),
    pad_frac: float = 0.06,
    smooth: float = 1.0,
    bins: int = 35,
    max_points: Optional[int] = None,
    debug: bool = False,
    title_fmt: str = ".3f",
    levels: Tuple[float, float, float] = (0.68, 0.95, 0.997),
):
    """
    Corner plot of posterior samples (requires `corner` installed).
    Returns fig (does NOT show).
    """
    loaded = _to_loaded(res)
    samples = np.asarray(loaded["samples"], float)
    pnames = [str(x) for x in loaded["param_names"]]
    labels = [_param_label_latex(n) for n in pnames]
    logp = loaded.get("log_prob", None)
    logp = np.asarray(logp, float) if logp is not None else None

    samp = _best_subset(samples, logp, max_n=max_points)

    # truths default: median of posterior (including fixed if present)
    if truths is None:
        pmed = _paramdict_from_samples(loaded, samp, use="median")
        truths = [pmed.get(k, np.nan) for k in pnames]

    # robust ranges
    lo = np.quantile(samp, q[0], axis=0)
    hi = np.quantile(samp, q[1], axis=0)
    ranges = []
    for i in range(samp.shape[1]):
        w = float(hi[i] - lo[i])
        pad = pad_frac * w if w > 0 else 1e-6
        ranges.append((float(lo[i] - pad), float(hi[i] + pad)))

    if debug:
        for name, (a, b) in zip(pnames, ranges):
            print(f"{name:12s}  [{a:.4g}, {b:.4g}]  width={b-a:.4g}")

    with plt.rc_context(_journal_rc()):
        try:
            import corner as _corner
        except Exception as e:
            raise ImportError("`corner` package not found. Install with: pip install corner") from e

        fig = _corner.corner(
            samp,
            labels=labels,
            truths=truths,
            range=ranges,
            bins=bins,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=title_fmt,
            title_quantiles=[0.16, 0.5, 0.84],
            plot_datapoints=False,
            fill_contours=True,
            smooth=smooth,
            levels=levels,
        )

        # important: avoid notebook double-render
        plt.close(fig)
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        return fig


# -----------------------------------------------------------------------------
# public: fit_bol
# -----------------------------------------------------------------------------

def fit_bol(
    res: Union[Dict[str, Any], Any],
    data,
    *,
    model: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    # plotting controls
    t_pad: float = 50.0,
    n_t: int = 800,
    show_1sigma: bool = False,
    n_draws: int = 0,  # posterior draw count for 1sigma band; <=0 uses internal default
    summary: Literal["best", "median"] = "best",
    interp_fill_model: Literal["edge", "nan", "raise"] = "nan",
    model_color: str = "#4D4D4D",
    band_color: str = "#9ECAE1",
    data_edge_color: str = "#2F2F2F",
    data_face_color: str = "#E6E6E6",
    data_err_color: str = "#2F2F2F",
    alpha_band: float = 0.18,
    lw_model: float = 2.2,
    ms_data: float = 6.0,
    capsize: float = 2.0,
    figsize: Tuple[float, float] = (7.0, 5.0),
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Bolometric/thermal single-curve fit plot.
    data must provide: t_days, y, yerr
    """
    from ..api import lightcurve_bol, predict_bol  # lazy import
    from ..api import _apply_data_filter  # lazy import

    loaded = _to_loaded(res)
    data = _apply_data_filter(data)

    # default model_kwargs: use saved meta if exists
    mk_saved = dict((loaded.get("meta", {}) or {}).get("model_kwargs", {}) or {})
    model_kwargs = dict(mk_saved if model_kwargs is None else model_kwargs)
    # Keep interpolation behavior controlled by `interp_fill_model`.
    model_kwargs.pop("interp_fill", None)

    ctx_dict = loaded.get("ctx", {}) or {}
    if ctx_dict:
        z, _, _, y_kind, _, _ = _forward_inputs_from_ctx_dict(ctx_dict)
    else:
        raise ValueError("Stored forward metadata is required (not found in loaded result).")

    y_kind = str(y_kind).lower()  # usually irrelevant for bol, but keep for label

    model_name = _get_model_name(loaded, fallback=model)

    samples = np.asarray(loaded["samples"], float)
    logp = loaded.get("log_prob", None)
    logp = np.asarray(logp, float) if logp is not None else None
    subset = _best_subset(samples, logp, max_n=max(3000, (n_draws * 20) if n_draws else 3000))

    mode = str(summary).strip().lower()
    if mode == "best":
        p0 = _paramdict_best_sample(loaded, samples, logp)
        model_label = "best-fit model"
    elif mode == "median":
        p0 = _paramdict_from_samples(loaded, subset, use="median")
        model_label = "median model"
    else:
        raise ValueError("summary must be 'best' or 'median'.")
    params_0, t_shift_0 = _params_and_shift_from_paramdict(loaded, p0)

    t_obs = np.asarray(data.t_days, float).reshape(-1)
    t_model_plot, t_plot, model_t_max = _model_zero_time_grid(
        t_obs, t_shift_0, t_pad, n_t
    )
    model_kwargs_eval = _prepare_plot_model_kwargs(model_kwargs, model_t_max)

    y_obs = np.asarray(data.y, float).reshape(-1)
    y_err = np.asarray(data.yerr, float).reshape(-1)

    with plt.rc_context(_journal_rc()):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        lc = lightcurve_bol(
            model=model_name,
            params=params_0,
            z=z,
            **model_kwargs_eval,
        )
        t_line = np.asarray(lc.t_days, float) - t_shift_0
        y_line = np.asarray(lc.Lbol, float)

        ax.errorbar(
                t_obs, y_obs, yerr=y_err,
                fmt="o",
                ms=ms_data,    
                mfc=data_face_color,
                mec=data_edge_color,
                mew=1.2,              
                elinewidth=1.0,
                capsize=capsize,
                alpha=0.9,
                ecolor=data_err_color,
                label="data",
            )

        ax.plot(t_line, y_line, lw=lw_model, color=model_color, label=model_label)

        if show_1sigma:
            rng = np.random.default_rng(123)
            draw_n = int(n_draws) if int(n_draws) > 0 else 300
            draw = min(draw_n, subset.shape[0])
            jj = rng.choice(subset.shape[0], size=draw, replace=False)

            ys = []
            pnames = [str(x) for x in loaded["param_names"]]
            fixed = dict(loaded.get("fixed", {}) or {})

            for j in jj:
                p = dict(fixed)
                for i, name in enumerate(pnames):
                    p[name] = float(subset[j, i])
                params_j, tshift_j = _params_and_shift_from_paramdict(loaded, p)
                t_eval = t_plot + tshift_j
                yj = np.full_like(t_plot, np.nan, dtype=float)
                valid = t_eval >= 0.0
                if np.any(valid):
                    yj[valid] = predict_bol(
                        model=model_name,
                        params=params_j,
                        z=z,
                        t_days=t_eval[valid],
                        interp_fill=interp_fill_model,
                        **model_kwargs_eval,
                    )
                yj = _anchor_model_origin_for_plot(
                    yj,
                    y_kind="bol",
                    y_reference=y_obs,
                    t_plot=t_plot,
                )
                ys.append(yj)

            ys = np.asarray(ys, float)
            valid_col = np.any(np.isfinite(ys), axis=0)
            if np.any(valid_col):
                lo = np.full(t_plot.shape, np.nan, dtype=float)
                hi = np.full(t_plot.shape, np.nan, dtype=float)
                lo[valid_col] = np.nanquantile(ys[:, valid_col], 0.16, axis=0)
                hi[valid_col] = np.nanquantile(ys[:, valid_col], 0.84, axis=0)
                ax.fill_between(
                    t_plot, lo, hi,
                    where=valid_col,
                    color=band_color,
                    alpha=alpha_band,
                    label=r"$1\sigma$ posterior",
                )
        ax.set_yscale("log")
        ax.set_xlabel("Since First Detection (days)")
        ax.set_ylabel("Bolometric Luminosity")
        xlim_model = np.concatenate([t_line, t_plot]) if show_1sigma else t_line
        _set_since_first_detection_xlim(ax, t_obs, xlim_model)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.minorticks_on()
        ax.legend(ncol=1)
        fig.tight_layout()

        plt.close(fig)
        return fig


# -----------------------------------------------------------------------------
# public: fit_multiband
# -----------------------------------------------------------------------------

def fit_multiband(
    res: Union[Dict[str, Any], Any],
    data,
    *,
    model: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    sed=None,
    # plotting controls
    t_pad: float = 50.0,
    n_t: int = 800,
    show_1sigma: bool = False,
    n_draws: int = 0,  # posterior draw count for 1sigma band; <=0 uses internal default
    summary: Literal["best", "median"] = "best",
    interp_fill_model: Literal["edge", "nan", "raise"] = "nan",
    alpha_band: float = 0.18,
    lw_model: float = 2.2,
    ms_data: float = 6.0,
    capsize: float = 2.0,
    figsize: Tuple[float, float] = (7.0, 5.0),
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Multi-band fit plot in a SINGLE panel.
    data must provide: t_days, band, y, yerr. Upper limits from
    ``data.is_upper_limit`` are shown as downward triangles.
    Bands are grouped exactly as they appear (case-sensitive, no normalization).
    """
    from ..api import lightcurve_multiband, predict_multiband  # lazy import
    from ..api import _apply_data_filter  # lazy import

    loaded = _to_loaded(res)
    data = _apply_data_filter(data)

    # default model_kwargs: use saved meta if exists
    mk_saved = dict((loaded.get("meta", {}) or {}).get("model_kwargs", {}) or {})
    model_kwargs = dict(mk_saved if model_kwargs is None else model_kwargs)
    # Keep interpolation behavior controlled by `interp_fill_model`.
    model_kwargs.pop("interp_fill", None)

    ctx_dict = loaded.get("ctx", {}) or {}
    if ctx_dict:
        z, distance_modulus, filters, y_kind, mag_system, extinction = _forward_inputs_from_ctx_dict(ctx_dict)
    else:
        raise ValueError("Stored forward metadata is required (not found in loaded result).")

    y_kind = str(y_kind).lower()
    mag_system = str(mag_system).lower()
    model_name = _get_model_name(loaded, fallback=model)
    plot_sed = sed if sed is not None else _plot_sed_from_meta(loaded)

    samples = np.asarray(loaded["samples"], float)
    logp = loaded.get("log_prob", None)
    logp = np.asarray(logp, float) if logp is not None else None
    subset = _best_subset(samples, logp, max_n=max(3000, (n_draws * 20) if n_draws else 3000))

    mode = str(summary).strip().lower()
    if mode == "best":
        p0 = _paramdict_best_sample(loaded, samples, logp)
        model_tag = "best"
    elif mode == "median":
        p0 = _paramdict_from_samples(loaded, subset, use="median")
        model_tag = "median"
    else:
        raise ValueError("summary must be 'best' or 'median'.")
    params_0, t_shift_0 = _params_and_shift_from_paramdict(loaded, p0)

    t_obs = np.asarray(data.t_days, float).reshape(-1)
    t_model_plot, t_plot, model_t_max = _model_zero_time_grid(
        t_obs, t_shift_0, t_pad, n_t
    )
    model_kwargs_eval = _prepare_plot_model_kwargs(model_kwargs, model_t_max)

    y_obs = np.asarray(data.y, float).reshape(-1)
    y_err = np.asarray(data.yerr, float).reshape(-1)
    band = np.asarray(data.band).reshape(-1)
    upper_raw = getattr(data, "is_upper_limit", None)
    is_upper_limit = (
        np.zeros(t_obs.size, dtype=bool)
        if upper_raw is None
        else np.asarray(upper_raw, bool).reshape(-1)
    )
    if is_upper_limit.size != t_obs.size:
        raise ValueError("data.is_upper_limit must have the same length as data.t_days.")

    bands = _unique_in_order(band.tolist())

    with plt.rc_context(_journal_rc()):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_map = {b: cycle[i % len(cycle)] for i, b in enumerate(bands)}

        fixed = dict(loaded.get("fixed", {}) or {})
        pnames = [str(x) for x in loaded["param_names"]]

        lc = lightcurve_multiband(
            model=model_name,
            params=params_0,
            z=z,
            distance_modulus=distance_modulus,
            filters=filters,
            bands=bands,
            y_kind=y_kind,
            mag_system=mag_system,
            extinction=extinction,
            sed=plot_sed,
            **model_kwargs_eval,
        )
        t_line = np.asarray(lc.t_days, float) - t_shift_0

        for b in bands:
            c = color_map[b]
            m = (band == b)
            m_detection = m & ~is_upper_limit
            m_upper = m & is_upper_limit

            if np.any(m_detection):
                ax.errorbar(
                    t_obs[m_detection], y_obs[m_detection], yerr=y_err[m_detection],
                    fmt=".", ms=ms_data, capsize=capsize,
                    color=c, alpha=0.9, label=f"{b} data",
                )
            if np.any(m_upper):
                ax.plot(
                    t_obs[m_upper], y_obs[m_upper],
                    linestyle="none", marker="v", ms=ms_data + 1.0,
                    markerfacecolor="none", markeredgecolor=c,
                    markeredgewidth=1.2, alpha=0.9,
                    label=f"{b} upper limit",
                )

            # Keep x-axis in observed time and shift the raw model grid by t_shift.
            y_line = np.asarray(lc.y[b], float)
            ax.plot(t_line, y_line, lw=lw_model, color=c, label=f"{b} {model_tag}")

            if show_1sigma:
                rng = np.random.default_rng(1000 + bands.index(b))
                draw_n = int(n_draws) if int(n_draws) > 0 else 300
                draw = min(draw_n, subset.shape[0])
                jj = rng.choice(subset.shape[0], size=draw, replace=False)

                ys = []
                for j in jj:
                    p = dict(fixed)
                    for i, name in enumerate(pnames):
                        p[name] = float(subset[j, i])
                    params_j, tshift_j = _params_and_shift_from_paramdict(loaded, p)
                    t_eval = t_plot + tshift_j
                    yj = np.full_like(t_plot, np.nan, dtype=float)
                    valid = t_eval >= 0.0
                    if np.any(valid):
                        yj[valid] = predict_multiband(
                            model=model_name,
                            params=params_j,
                            z=z,
                            distance_modulus=distance_modulus,
                            filters=filters,
                            t_days=t_eval[valid],
                            band=np.array([b] * int(np.sum(valid)), dtype=object),
                            y_kind=y_kind,
                            mag_system=mag_system,
                            extinction=extinction,
                            sed=plot_sed,
                            interp_fill=interp_fill_model,
                            **model_kwargs_eval,
                        )
                    yj = _anchor_model_origin_for_plot(
                        yj,
                        y_kind=y_kind,
                        y_reference=(
                            y_obs[m_detection]
                            if np.any(m_detection)
                            else y_obs[m]
                        ),
                        t_plot=t_plot,
                    )
                    ys.append(yj)

                ys = np.asarray(ys, float)
                valid_col = np.any(np.isfinite(ys), axis=0)
                if np.any(valid_col):
                    lo = np.full(t_plot.shape, np.nan, dtype=float)
                    hi = np.full(t_plot.shape, np.nan, dtype=float)
                    lo[valid_col] = np.nanquantile(ys[:, valid_col], 0.16, axis=0)
                    hi[valid_col] = np.nanquantile(ys[:, valid_col], 0.84, axis=0)
                    ax.fill_between(t_plot, lo, hi, where=valid_col, color=c, alpha=alpha_band)
        if ylim is None:
            if y_kind == "mag":
                ax.set_ylim(float(np.nanmin(y_obs)) - 2.0, float(np.nanmax(y_obs)) + 2.0)
            else:
                finite = np.isfinite(y_obs)
                if np.any(finite):
                    ymin = float(np.nanmin(y_obs[finite]))
                    ymax = float(np.nanmax(y_obs[finite]))
                    pad = 0.05 * (ymax - ymin) if ymax > ymin else max(abs(ymax) * 0.1, 1e-30)
                    ax.set_ylim(ymin - pad, ymax + pad)
        xlim_model = np.concatenate([t_line, t_plot]) if show_1sigma else t_line
        _set_since_first_detection_xlim(ax, t_obs, xlim_model)
        ax.set_xlabel("Since First Detection (days)")
        ax.set_ylabel(_mag_ylabel(y_kind, mag_system))
        _maybe_invert_mag_axis(ax, y_kind)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.minorticks_on()
        ax.legend(ncol=2, fontsize=10)
        fig.tight_layout()

        plt.close(fig)
        return fig


__all__ = ["fit", "corner", "fit_multiband", "fit_bol"]
