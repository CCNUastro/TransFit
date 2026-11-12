# transfit/priors/__init__.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np

from ..model_registry import canonical_model_name
from .common import UniformBoundsPrior, MixedBoundsPrior, apply_user_bounds


def build_bounds(
    model: str,
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    include_t_shift: bool = True,
):
    m = canonical_model_name(model, warn_legacy=False)

    if m == "nickel":
        from .nickel import default_names_and_bounds
        names, bounds = default_names_and_bounds(include_t_shift=include_t_shift)

    elif m == "magnetar":
        from .magnetar import default_names_and_bounds
        names, bounds = default_names_and_bounds(include_t_shift=include_t_shift)

    elif m == "magnetar_ni":
        from .magnetar_ni import default_names_and_bounds
        names, bounds = default_names_and_bounds(include_t_shift=include_t_shift)

    elif m == "csm":
        from .csm import default_names_and_bounds
        names, bounds = default_names_and_bounds(include_t_shift=include_t_shift)

    elif m == "nickel_csm_staged":
        from .nickel_csm import default_names_and_bounds
        names, bounds = default_names_and_bounds(include_t_shift=include_t_shift)

    else:
        raise ValueError(f"Unknown model='{model}' for priors.build_bounds")

    bounds = apply_user_bounds(names, np.asarray(bounds, float), priors=priors)
    return names, np.asarray(bounds, float)


__all__ = [
    "UniformBoundsPrior",
    "MixedBoundsPrior",
    "build_bounds",
]
