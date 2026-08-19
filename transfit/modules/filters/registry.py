from __future__ import annotations

from typing import Dict, List

from .core import FilterProfile


_BUILTIN_FILTERS: Dict[str, Dict[str, object]] = {
    "johnson_cousins.B": {
        "kind": "mono",
        "nu_eff_hz": 6.80e14,
        "zero_points_jy": {"vega": 4260.0},
        "meta": {"family": "Johnson-Cousins"},
    },
    "johnson_cousins.V": {
        "kind": "mono",
        "nu_eff_hz": 5.50e14,
        "zero_points_jy": {"vega": 3640.0},
        "meta": {"family": "Johnson-Cousins"},
    },
    "johnson_cousins.R": {
        "kind": "mono",
        "nu_eff_hz": 4.70e14,
        "zero_points_jy": {"vega": 3080.0},
        "meta": {"family": "Johnson-Cousins"},
    },
    "johnson_cousins.I": {
        "kind": "mono",
        "nu_eff_hz": 3.90e14,
        "zero_points_jy": {"vega": 2550.0},
        "meta": {"family": "Johnson-Cousins"},
    },
    "sdss.u": {
        "kind": "mono",
        "nu_eff_hz": 8.31e14,
        "zero_points_jy": {},
        "meta": {"family": "SDSS"},
    },
    "sdss.g": {
        "kind": "mono",
        "nu_eff_hz": 6.42e14,
        "zero_points_jy": {},
        "meta": {"family": "SDSS"},
    },
    "sdss.r": {
        "kind": "mono",
        "nu_eff_hz": 4.88e14,
        "zero_points_jy": {},
        "meta": {"family": "SDSS"},
    },
    "sdss.i": {
        "kind": "mono",
        "nu_eff_hz": 4.02e14,
        "zero_points_jy": {},
        "meta": {"family": "SDSS"},
    },
    "sdss.z": {
        "kind": "mono",
        "nu_eff_hz": 3.36e14,
        "zero_points_jy": {},
        "meta": {"family": "SDSS"},
    },
    "ztf.g": {
        "kind": "mono",
        "nu_eff_hz": 6.38e14,
        "zero_points_jy": {},
        "meta": {"family": "ZTF"},
    },
    "ztf.r": {
        "kind": "mono",
        "nu_eff_hz": 4.84e14,
        "zero_points_jy": {},
        "meta": {"family": "ZTF"},
    },
    "ztf.i": {
        "kind": "mono",
        "nu_eff_hz": 3.80e14,
        "zero_points_jy": {},
        "meta": {"family": "ZTF"},
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
