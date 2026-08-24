# transfit/data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class MultiBandData:
    """
    Standard multi-band observation container.

    Parameters
    ----------
    t_days : array
        Observer-frame time in days (already relative to some t0 chosen by user).
    band : array
        Band labels (e.g. "g","r","b","v"...), same length as t_days.
    y : array
        Observed values. If ctx.y_kind == "mag", y is magnitude (AB by default).
        If ctx.y_kind == "flux", y is flux density Fnu (cgs).
    yerr : array
        1-sigma uncertainties, same length as y.
    mask : array, optional
        Boolean mask of same length; if provided, only masked-in points are used.
    is_upper_limit : array, optional
        Boolean flag for non-detection limits. For flagged rows, ``y`` is the
        reported limiting magnitude or flux. ``yerr`` may be NaN when the
        one-sigma noise is unavailable.
    upper_limit_nsigma : array, optional
        Detection significance (for example 3 or 5) for upper-limit rows that
        do not provide ``yerr``. May be a scalar or an array aligned with the
        data. Missing values use TransFit's default 5-sigma interpretation.
    """
    t_days: np.ndarray
    band: np.ndarray
    y: np.ndarray
    yerr: np.ndarray
    mask: Optional[np.ndarray] = None
    is_upper_limit: Optional[np.ndarray] = None
    upper_limit_nsigma: Optional[np.ndarray] = None

    def __post_init__(self):
        t = np.asarray(self.t_days, float)
        b = np.asarray(self.band)
        y = np.asarray(self.y, float)
        e = np.asarray(self.yerr, float)

        if not (t.ndim == b.ndim == y.ndim == e.ndim == 1):
            raise ValueError("t_days/band/y/yerr must be 1D arrays.")
        n = t.size
        if not (b.size == y.size == e.size == n):
            raise ValueError("t_days/band/y/yerr must have the same length.")

        m = None
        if self.mask is not None:
            m = np.asarray(self.mask, bool)
            if m.shape != (n,):
                raise ValueError("mask must have shape (N,).")

        if self.is_upper_limit is None:
            upper = np.zeros(n, dtype=bool)
        else:
            upper_in = np.asarray(self.is_upper_limit, bool)
            if upper_in.ndim == 0:
                upper = np.full(n, bool(upper_in), dtype=bool)
            elif upper_in.shape == (n,):
                upper = upper_in
            else:
                raise ValueError("is_upper_limit must be a scalar or have shape (N,).")

        if self.upper_limit_nsigma is None:
            nsigma = np.full(n, np.nan, dtype=float)
        else:
            nsigma_in = np.asarray(self.upper_limit_nsigma, float)
            if nsigma_in.ndim == 0:
                nsigma = np.full(n, np.nan, dtype=float)
                nsigma[upper] = float(nsigma_in)
            elif nsigma_in.shape == (n,):
                nsigma = nsigma_in
            else:
                raise ValueError(
                    "upper_limit_nsigma must be a scalar or have shape (N,)."
                )

        # store normalized arrays back (frozen dataclass -> use object.__setattr__)
        object.__setattr__(self, "t_days", t)
        object.__setattr__(self, "band", b)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "yerr", e)
        object.__setattr__(self, "mask", m)
        object.__setattr__(self, "is_upper_limit", upper)
        object.__setattr__(self, "upper_limit_nsigma", nsigma)

    def filtered(self) -> "MultiBandData":
        """
        Return a new MultiBandData with the explicit mask applied.

        This method does not silently remove invalid values. Public fitting
        APIs validate remaining data and raise clear errors instead.
        """
        if self.mask is None:
            return self

        good = np.asarray(self.mask, bool)
        upper = getattr(self, "is_upper_limit", None)
        nsigma = getattr(self, "upper_limit_nsigma", None)

        return MultiBandData(
            t_days=self.t_days[good],
            band=self.band[good],
            y=self.y[good],
            yerr=self.yerr[good],
            mask=None,
            is_upper_limit=(
                None if upper is None else np.asarray(upper, bool)[good]
            ),
            upper_limit_nsigma=(
                None if nsigma is None else np.asarray(nsigma, float)[good]
            ),
        )

    @property
    def bands(self):
        """Sorted unique band labels present in the data."""
        return sorted(set(self.band.tolist()))

@dataclass(frozen=True)
class BolometricData:
    """
    Bolometric observation container.

    t_days : observer-frame days
    y     : Lbol (cgs) or other bolometric observable
    yerr  : 1-sigma errors
    """
    t_days: np.ndarray
    y: np.ndarray
    yerr: np.ndarray
    mask: Optional[np.ndarray] = None

    def __post_init__(self):
        t = np.asarray(self.t_days, float)
        y = np.asarray(self.y, float)
        e = np.asarray(self.yerr, float)

        if not (t.ndim == y.ndim == e.ndim == 1):
            raise ValueError("t_days/y/yerr must be 1D arrays.")
        if not (t.size == y.size == e.size):
            raise ValueError("t_days/y/yerr must have the same length.")

        if self.mask is not None:
            m = np.asarray(self.mask, bool)
            if m.shape != (t.size,):
                raise ValueError("mask must have shape (N,).")

        object.__setattr__(self, "t_days", t)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "yerr", e)

    def filtered(self) -> "BolometricData":
        """
        Return a new BolometricData with the explicit mask applied.

        Invalid luminosities or uncertainties are rejected by fitting APIs
        unless the user excludes them with `mask`.
        """
        if self.mask is None:
            return self

        good = np.asarray(self.mask, bool)
        return BolometricData(self.t_days[good], self.y[good], self.yerr[good], None)
