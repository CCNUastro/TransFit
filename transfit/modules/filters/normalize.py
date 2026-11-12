from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Iterable, Optional

import numpy as np

from transfit.constants import C_LIGHT

from ..labels import normalize_band_label
from .core import FilterProfile
from .registry import get_builtin_filter, list_builtin_filters


_LAMBDA_EFF_FACTORS_A = {
    "lambda_eff_A": 1.0,
    "wavelength_eff_A": 1.0,
    "lambda_eff_nm": 10.0,
    "wavelength_eff_nm": 10.0,
    "lambda_eff_um": 1.0e4,
    "lambda_eff_micron": 1.0e4,
    "wavelength_eff_um": 1.0e4,
    "wavelength_eff_micron": 1.0e4,
}


def _coerce_zero_points(spec: Mapping[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "zero_points_jy" in spec:
        raw = spec["zero_points_jy"]
        if not isinstance(raw, Mapping):
            raise TypeError("zero_points_jy must be a mapping of system -> Jy.")
        for key, value in raw.items():
            out[str(key).strip().lower()] = float(value)
    if "vega_zero_point_jy" in spec:
        out["vega"] = float(spec["vega_zero_point_jy"])
    return out


def _coerce_lambda_eff_A(spec: Mapping[str, object]) -> Optional[float]:
    found = [key for key in _LAMBDA_EFF_FACTORS_A if key in spec]
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            "Specify only one effective wavelength key. "
            f"Got: {found}"
        )
    key = found[0]
    lambda_eff_A = float(spec[key]) * _LAMBDA_EFF_FACTORS_A[key]
    if not np.isfinite(lambda_eff_A) or lambda_eff_A <= 0.0:
        raise ValueError(f"{key} must be positive and finite.")
    return lambda_eff_A


def _nu_eff_hz_from_lambda_eff_A(lambda_eff_A: float) -> float:
    return C_LIGHT / (float(lambda_eff_A) * 1.0e-8)


def _normalize_one(label: str, spec: object) -> FilterProfile:
    if isinstance(spec, FilterProfile):
        if spec.label != label:
            return FilterProfile(
                label=label,
                filter_id=spec.filter_id,
                kind=spec.kind,
                source=spec.source,
                detector=spec.detector,
                nu_eff_hz=spec.nu_eff_hz,
                wavelength_A=spec.wavelength_A,
                throughput=spec.throughput,
                zero_points_jy=spec.zero_points_jy,
                meta=spec.meta,
            )
        return spec

    if np.isscalar(spec) and not isinstance(spec, (str, bytes)):
        return FilterProfile(
            label=label,
            filter_id=f"legacy:{label}",
            kind="mono",
            source="legacy",
            nu_eff_hz=float(spec),
        )

    if isinstance(spec, str):
        return get_builtin_filter(label=label, filter_id=spec)

    if not isinstance(spec, Mapping):
        raise TypeError(
            f"Unsupported filter spec for band {label!r}: {type(spec).__name__}. "
            "Use a preset id string, mapping with lambda_eff_A, or mapping with nu_eff_hz. "
            "Scalar frequencies are accepted for legacy compatibility."
        )

    lambda_eff_A = _coerce_lambda_eff_A(spec)
    has_custom_mono = lambda_eff_A is not None or "nu_eff_hz" in spec or "nu_eff" in spec
    has_bandpass = "wavelength_A" in spec or "throughput" in spec
    if "preset" in spec and (has_custom_mono or has_bandpass):
        raise ValueError("preset cannot be combined with custom filter definitions.")

    if any(k in spec for k in ("filter_id", "id", "preset")) and not has_custom_mono and not has_bandpass:
        filter_id = spec.get("filter_id", spec.get("id", spec.get("preset")))
        prof = get_builtin_filter(label=label, filter_id=str(filter_id))
        zero_points_jy = dict(prof.zero_points_jy)
        zero_points_jy.update(_coerce_zero_points(spec))
        meta = dict(prof.meta)
        meta.update(dict(spec.get("meta", {}) or {}))
        return FilterProfile(
            label=label,
            filter_id=prof.filter_id,
            kind=prof.kind,
            source=prof.source,
            detector=prof.detector,
            nu_eff_hz=prof.nu_eff_hz,
            wavelength_A=prof.wavelength_A,
            throughput=prof.throughput,
            zero_points_jy=zero_points_jy,
            meta=meta,
        )

    if lambda_eff_A is not None:
        meta = dict(spec.get("meta", {}) or {})
        meta.setdefault("lambda_eff_A", lambda_eff_A)
        return FilterProfile(
            label=label,
            filter_id=str(spec.get("filter_id", spec.get("id", f"user:{label}"))),
            kind="mono",
            source="user",
            detector=str(spec.get("detector", "energy")),
            nu_eff_hz=_nu_eff_hz_from_lambda_eff_A(lambda_eff_A),
            zero_points_jy=_coerce_zero_points(spec),
            meta=meta,
        )

    if "nu_eff_hz" in spec or "nu_eff" in spec:
        return FilterProfile(
            label=label,
            filter_id=str(spec.get("filter_id", spec.get("id", f"user:{label}"))),
            kind="mono",
            source="user",
            detector=str(spec.get("detector", "energy")),
            nu_eff_hz=float(spec.get("nu_eff_hz", spec.get("nu_eff"))),
            zero_points_jy=_coerce_zero_points(spec),
            meta=dict(spec.get("meta", {}) or {}),
        )

    if "wavelength_A" in spec or "throughput" in spec:
        raise NotImplementedError(
            "Bandpass filter curves are not implemented yet. "
            "Phase 1 supports mono-frequency filters only."
        )

    raise ValueError(
        f"Could not interpret filter spec for band {label!r}. "
        "Use a preset id string, mapping with lambda_eff_A, or mapping with nu_eff_hz. "
        "Scalar frequencies are accepted for legacy compatibility."
    )


def _require_used_bands(filters: Mapping[str, FilterProfile], used_bands: Iterable[str]) -> None:
    needed = [normalize_band_label(b) for b in used_bands]
    missing = [b for b in needed if b not in filters]
    if missing:
        raise KeyError(
            f"These bands are missing in filters (case-sensitive): {missing}. "
            f"Available: {sorted(filters.keys())}"
        )


def _require_mag_system(filters: Mapping[str, FilterProfile], used_bands: Iterable[str], mag_system: str) -> None:
    system = str(mag_system).strip().lower()
    if system not in ("ab", "vega"):
        raise ValueError("mag_system must be 'ab' or 'vega'.")
    if system != "vega":
        return
    missing = [
        b for b in [normalize_band_label(x) for x in used_bands]
        if "vega" not in filters[b].zero_points_jy
    ]
    if missing:
        raise ValueError(
            "Vega magnitudes require a Vega zero point for every used band. "
            f"Missing Vega zero points for: {missing}"
        )


def normalize_filters(
    filters: Optional[Mapping[str, object]] = None,
    *,
    used_bands: Optional[Iterable[str]] = None,
    mag_system: str = "ab",
) -> Dict[str, FilterProfile]:
    """Normalize explicit filter specs and auto-resolve built-in band labels.

    When ``used_bands`` is provided, any used band that is not explicitly
    present in ``filters`` is interpreted as a built-in ``filter_id``. This
    lets data labelled with registry ids such as ``"sdss.u"`` work without
    requiring the redundant mapping ``{"sdss.u": {"filter_id": "sdss.u"}}``.

    Explicit entries remain authoritative, so aliases and custom filters still
    use the existing mapping forms.
    """
    if filters is None:
        explicit: Mapping[str, object] = {}
    elif isinstance(filters, Mapping):
        explicit = filters
    else:
        raise TypeError("filters must be a mapping or None.")

    out: Dict[str, FilterProfile] = {}
    for raw_label, spec in explicit.items():
        label = normalize_band_label(raw_label)
        out[label] = _normalize_one(label, spec)

    if used_bands is None:
        if not out:
            raise ValueError("filters must be a non-empty mapping when used_bands is not provided.")
        bands_for_check = list(out.keys())
    else:
        bands_for_check = [normalize_band_label(b) for b in used_bands]
        unresolved = []
        for label in dict.fromkeys(bands_for_check):
            if label in out:
                continue
            try:
                out[label] = get_builtin_filter(label=label, filter_id=label)
            except KeyError:
                unresolved.append(label)

        if unresolved:
            raise KeyError(
                "No filter definition was provided for these bands and their labels are not "
                f"recognized built-in filter_ids: {unresolved}. Pass explicit entries in "
                "filters for aliases/custom filters. Available built-in filter_ids: "
                f"{list_builtin_filters()}"
            )
        if not out:
            raise ValueError("used_bands must contain at least one band when filters is omitted.")

    _require_used_bands(out, bands_for_check)
    _require_mag_system(out, bands_for_check, mag_system)
    return out


def validate_filter_map(
    filters: Optional[Mapping[str, object]],
    *,
    used_bands: Iterable[str],
    mag_system: str = "ab",
) -> Dict[str, FilterProfile]:
    return normalize_filters(filters, used_bands=used_bands, mag_system=mag_system)


def mono_effective_frequency(profile: FilterProfile) -> float:
    if profile.kind != "mono" or profile.nu_eff_hz is None:
        raise NotImplementedError(
            f"Filter {profile.label!r} is not a mono-frequency filter. "
            "Bandpass photometry is not implemented yet."
        )
    return float(profile.nu_eff_hz)


def mono_effective_wavelength_A(profile: FilterProfile) -> float:
    nu_eff_hz = mono_effective_frequency(profile)
    return (C_LIGHT / nu_eff_hz) * 1.0e8
