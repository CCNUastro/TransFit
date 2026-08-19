import pytest

from transfit.api import _context_from_forward_inputs
from transfit.modules.filters import normalize_filters


def test_builtin_filter_ids_resolve_from_used_band_labels():
    filters = normalize_filters(None, used_bands=["sdss.u", "ztf.r"])

    assert filters["sdss.u"].label == "sdss.u"
    assert filters["sdss.u"].filter_id == "sdss.u"
    assert filters["sdss.u"].source == "builtin"
    assert filters["ztf.r"].filter_id == "ztf.r"


def test_explicit_aliases_and_implicit_builtins_can_be_mixed():
    filters = normalize_filters(
        {"u": {"filter_id": "sdss.u"}},
        used_bands=["u", "sdss.r"],
    )

    assert filters["u"].label == "u"
    assert filters["u"].filter_id == "sdss.u"
    assert filters["sdss.r"].label == "sdss.r"
    assert filters["sdss.r"].filter_id == "sdss.r"


def test_unknown_band_still_requires_explicit_definition():
    with pytest.raises(KeyError, match="not recognized built-in filter_ids"):
        normalize_filters(None, used_bands=["g"])


def test_forward_context_can_omit_filters_for_builtin_ids():
    ctx = _context_from_forward_inputs(
        z=0.1,
        distance_modulus=None,
        filters=None,
        y_kind="mag",
        mag_system="ab",
        extinction=None,
        require_filters=True,
        require_distance=True,
        used_bands=["sdss.g", "sdss.r"],
    )

    assert ctx.filters is not None
    assert ctx.filters["sdss.g"].filter_id == "sdss.g"
    assert ctx.filters["sdss.r"].filter_id == "sdss.r"
