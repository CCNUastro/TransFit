from __future__ import annotations

from collections.abc import Mapping
from typing import Dict

from .core import FilterProfile
from .normalize import normalize_filters


def filters_to_dict(filters: Mapping[str, FilterProfile]) -> Dict[str, Dict[str, object]]:
    return {str(label): profile.to_dict() for label, profile in dict(filters or {}).items()}


def filters_from_dict(payload: Mapping[str, object]) -> Dict[str, FilterProfile]:
    if not payload:
        return {}
    if all(not isinstance(v, Mapping) for v in payload.values()):
        return normalize_filters(payload)
    return {
        str(label): FilterProfile.from_dict(str(label), dict(spec))
        for label, spec in payload.items()
    }
