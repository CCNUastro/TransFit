# transfit/priors/common.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class UniformBoundsPrior:
    bounds: np.ndarray
    param_names: Sequence[str]

    def __post_init__(self):
        b = np.asarray(self.bounds, float)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must be shape (ndim, 2)")
        if len(self.param_names) != b.shape[0]:
            raise ValueError("param_names length must match bounds ndim")
        lo = b[:, 0]
        hi = b[:, 1]
        if np.any(~np.isfinite(b)) or np.any(lo >= hi):
            raise ValueError("bounds must be finite and satisfy lo < hi")
        object.__setattr__(self, "bounds", b)

    def lnprior(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, float)
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        ok = np.all((theta > lo) & (theta < hi))
        return 0.0 if ok else -np.inf

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return rng.uniform(lo, hi, size=(int(n), self.bounds.shape[0]))


@dataclass(frozen=True)
class MixedBoundsPrior:
    """
    Independent box priors with optional log-uniform dimensions.

    Parameters
    ----------
    bounds : (ndim, 2)
        Bounds in linear parameter space.
    param_names : Sequence[str]
        Parameter names.
    log_flags : Sequence[bool], optional
        Whether each dimension uses log-uniform prior over bounds.
    """
    bounds: np.ndarray
    param_names: Sequence[str]
    log_flags: Optional[Sequence[bool]] = None
    inclusive_names: Sequence[str] = ()

    def __post_init__(self):
        b = np.asarray(self.bounds, float)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must be shape (ndim, 2)")
        if len(self.param_names) != b.shape[0]:
            raise ValueError("param_names length must match bounds ndim")
        lo = b[:, 0]
        hi = b[:, 1]
        if np.any(~np.isfinite(b)) or np.any(lo >= hi):
            raise ValueError("bounds must be finite and satisfy lo < hi")

        if self.log_flags is None:
            lf = np.zeros(b.shape[0], dtype=bool)
        else:
            lf = np.asarray(self.log_flags, bool).reshape(-1)
            if lf.size != b.shape[0]:
                raise ValueError("log_flags length must match bounds ndim")

        if np.any(lf):
            bad = np.where(lf & ((lo <= 0.0) | (hi <= 0.0)))[0]
            if bad.size > 0:
                names = [str(self.param_names[int(i)]) for i in bad.tolist()]
                raise ValueError(
                    f"log-uniform prior requires positive bounds; invalid params: {names}"
                )

        object.__setattr__(self, "bounds", b)
        object.__setattr__(self, "log_flags", lf)

    def lnprior(self, theta: np.ndarray) -> float:
        x = np.asarray(theta, float)
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        inclusive = np.asarray([name in self.inclusive_names for name in self.param_names], bool)
        ok = np.all(np.where(inclusive, (x >= lo) & (x <= hi), (x > lo) & (x < hi)))
        if not ok:
            return -np.inf

        lf = np.asarray(self.log_flags, bool)
        if not np.any(lf):
            return 0.0

        x_log = x[lf]
        if np.any(x_log <= 0.0):
            return -np.inf

        # For log-uniform p(x) ∝ 1/x over [lo, hi], normalization constants
        # are dropped because samplers only need relative ln-prior.
        return -float(np.sum(np.log(x_log)))

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        n = int(n)
        ndim = self.bounds.shape[0]
        out = np.empty((n, ndim), dtype=float)

        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        lf = np.asarray(self.log_flags, bool)

        if np.any(~lf):
            idx = np.where(~lf)[0]
            out[:, idx] = rng.uniform(lo[idx], hi[idx], size=(n, idx.size))

        if np.any(lf):
            idx = np.where(lf)[0]
            llo = np.log(lo[idx])
            lhi = np.log(hi[idx])
            out[:, idx] = np.exp(rng.uniform(llo, lhi, size=(n, idx.size)))

        return out


def apply_user_bounds(names, bounds: np.ndarray, priors: Optional[Dict[str, Tuple[float, float]]] = None):
    if not priors:
        return np.asarray(bounds, float)

    b = np.asarray(bounds, float).copy()
    idx = {n: i for i, n in enumerate(names)}
    for k, (lo, hi) in priors.items():
        if k not in idx:
            raise KeyError(f"Unknown prior key '{k}'. Allowed: {list(names)}")
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"Invalid bounds for '{k}': ({lo},{hi}); require finite lo < hi")
        b[idx[k], 0] = lo
        b[idx[k], 1] = hi
    return b
