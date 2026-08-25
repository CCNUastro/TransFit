from importlib import import_module
from importlib.metadata import version

__version__ = version("transfit")

from .api import (
    BolometricLC, MultiBandLC,
    BolometricData, MultiBandData,
    model_param_names, param_template,
    lightcurve_bol, lightcurve_multiband,
    predict_bol, predict_multiband,
    fit_bol, fit_multiband,
)
from .modules.io import save, load, default_outpath


def __getattr__(name):
    if name == "plot":
        # Keep plotting optional until the user explicitly requests it.
        return import_module(".modules.plot", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BolometricData", "MultiBandData",
    "model_param_names", "param_template",
    "lightcurve_bol", "lightcurve_multiband",
    "predict_bol", "predict_multiband",
    "fit_bol", "fit_multiband",
    "save", "load", "default_outpath",
    "__version__",
    "plot",
]
