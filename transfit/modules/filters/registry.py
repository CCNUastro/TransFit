from __future__ import annotations

from typing import Dict, List

from .core import FilterProfile


_BUILTIN_FILTERS: Dict[str, Dict[str, object]] = {
    "johnson_cousins.U": {
        "kind": "mono",
        "nu_eff_hz": 8.44e14,
        "zero_points_jy": {"vega": 1438.73},
        "meta": {
            "family": "Johnson-Cousins",
            "svo_fps_id": "Generic/Johnson.U",
        },
    },
    "johnson_cousins.B": {
        "kind": "mono",
        "nu_eff_hz": 6.80e14,
        "zero_points_jy": {"vega": 4260.0},
        "meta": {
            "family": "Johnson-Cousins",
            "svo_fps_id": "Generic/Johnson.B",
        },
    },
    "johnson_cousins.V": {
        "kind": "mono",
        "nu_eff_hz": 5.50e14,
        "zero_points_jy": {"vega": 3640.0},
        "meta": {
            "family": "Johnson-Cousins",
            "svo_fps_id": "Generic/Johnson.V"
        },
    },
    "johnson_cousins.R": {
        "kind": "mono",
        "nu_eff_hz": 4.70e14,
        "zero_points_jy": {"vega": 3080.0},
        "meta": {
            "family": "Johnson-Cousins",
            "svo_fps_id": "Generic/Cousins.R"
        },
    },
    "johnson_cousins.I": {
        "kind": "mono",
        "nu_eff_hz": 3.90e14,
        "zero_points_jy": {"vega": 2550.0},
        "meta": {
            "family": "Johnson-Cousins",
            "svo_fps_id": "Generic/Cousins.I"
        },
    },
    "sdss.u": {
        "kind": "mono",
        "nu_eff_hz": 8.42e+14,
        "zero_points_jy": {},
        "meta": {
            "family": "SDSS",
            "svo_fps_id": "SLOAN/SDSS.u"
        },
    },
    "sdss.g": {
        "kind": "mono",
        "nu_eff_hz": 6.35e+14,
        "zero_points_jy": {},
        "meta": {
            "family": "SDSS",
            "svo_fps_id": "SLOAN/SDSS.g"
        },
    },
    "sdss.r": {
        "kind": "mono",
        "nu_eff_hz": 4.85e+14,
        "zero_points_jy": {},
        "meta": {
            "family": "SDSS",
            "svo_fps_id": "SLOAN/SDSS.r"
        },
    },
    "sdss.i": {
        "kind": "mono",
        "nu_eff_hz": 4.00e+14,
        "zero_points_jy": {},
        "meta": {
            "family": "SDSS",
            "svo_fps_id": "SLOAN/SDSS.i"
        },
    },
    "sdss.z": {
        "kind": "mono",
        "nu_eff_hz": 3.35e+14,
        "zero_points_jy": {},
        "meta": {
            "family": "SDSS",
            "svo_fps_id": "SLOAN/SDSS.z"
        },
    },
    "ztf.g": {
        "kind": "mono",
        "nu_eff_hz": 6.25e14,
        "zero_points_jy": {},
        "meta": {
            "family": "ZTF",
            "svo_fps_id": "Palomar/ZTF.g"
        },
    },
    "ztf.r": {
        "kind": "mono",
        "nu_eff_hz": 4.66e14,
        "zero_points_jy": {},
        "meta": {
            "family": "ZTF",
            "svo_fps_id": "Palomar/ZTF.r"
        },
    },
    "ztf.i": {
        "kind": "mono",
        "nu_eff_hz": 3.80e14,
        "zero_points_jy": {},
        "meta": {
            "family": "ZTF",
            "svo_fps_id": "Palomar/ZTF.i"
        },
    },
        "lsst.u": {
        "kind": "mono",
        "nu_eff_hz": 8.14e+14,
        "zero_points_jy": {},
        "meta": {
            "family": "LSST",
            "svo_fps_id": "LSST/LSST.u"
        }
    },
    "lsst.g": {
        "kind": "mono",
        "nu_eff_hz": 6.24e+14,
        "zero_points_jy": {},
        "meta": {
            "family": "LSST",
            "svo_fps_id": "LSST/LSST.g"
        }
    },
    "lsst.r": {
        "kind": "mono",
        "nu_eff_hz": 4.81e+14,
        "zero_points_jy": {},
        "meta": {
            "family": "LSST",
            "svo_fps_id": "LSST/LSST.r"
        }
    },
    "lsst.i": {
        "kind": "mono",
        "nu_eff_hz": 3.98e+14,
        "zero_points_jy": {},
        "meta": {
            "family": "LSST",
            "svo_fps_id": "LSST/LSST.i"
        }
    },
    "lsst.z": {
        "kind": "mono",
        "nu_eff_hz": 3.45e+14,
        "zero_points_jy": {},
        "meta": {
            "family": "LSST",
            "svo_fps_id": "LSST/LSST.z"
        }
    },
    "lsst.y": {
        "kind": "mono",
        "nu_eff_hz": 3.08e+14,
        "zero_points_jy": {},
        "meta": {
            "family": "LSST",
            "svo_fps_id": "LSST/LSST.y"
        }
    },
}


def list_builtin_filters() -> List[str]:
    return sorted(_BUILTIN_FILTERS.keys())


def describe_builtin_filter(filter_id: str) -> Dict[str, object]:
    key = str(filter_id).strip()
    if key not in _BUILTIN_FILTERS:
        raise KeyError(
            f"Unknown built-in filter {filter_id!r}. Available: {list_builtin_filters()}"
        )
    payload = dict(_BUILTIN_FILTERS[key])
    payload["filter_id"] = key
    return payload


def get_builtin_filter(*, label: str, filter_id: str) -> FilterProfile:
    payload = describe_builtin_filter(filter_id)
    return FilterProfile(
        label=label,
        filter_id=str(payload["filter_id"]),
        kind=str(payload["kind"]),
        source="builtin",
        detector="energy",
        nu_eff_hz=float(payload["nu_eff_hz"]) if payload.get("nu_eff_hz") is not None else None,
        zero_points_jy=dict(payload.get("zero_points_jy", {}) or {}),
        meta=dict(payload.get("meta", {}) or {}),
    )
