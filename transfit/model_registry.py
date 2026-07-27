from __future__ import annotations

CANONICAL_NICKEL = "nickel"
CANONICAL_MAGNETAR = "magnetar"
CANONICAL_MAGNETAR_NI = "magnetar_ni"
CANONICAL_CSM = "csm"

NICKEL_ALIASES = {"ni", "nickel"}

MAGNETAR_ALIASES = {"magnetar", "mag", "mg"}

MAGNETAR_NI_ALIASES = {"magni", "mag_ni", "mag-ni", "mag+ni", "magnetar+ni", "magnetar_ni", "magnetar-ni"}

CSM_ALIASES = {"csm", "interaction", "csm_interaction", "csm-interaction"}


def canonical_model_name(model: str, *, warn_legacy: bool = False) -> str:
    m = str(model).strip().lower()

    if m in NICKEL_ALIASES:
        return CANONICAL_NICKEL

    if m in MAGNETAR_ALIASES:
        return CANONICAL_MAGNETAR

    if m in MAGNETAR_NI_ALIASES:
        return CANONICAL_MAGNETAR_NI

    if m in CSM_ALIASES:
        return CANONICAL_CSM

    raise ValueError(f"Unknown model='{model}'")


def forward_param_defaults(model: str) -> dict[str, float]:
    m = canonical_model_name(model, warn_legacy=False)
    if m == CANONICAL_NICKEL:
        return {"E_Th_in": 0.0, "R_0": 10.0, "delta": 0.0, "n": 10.0}
    if m == CANONICAL_MAGNETAR:
        return {"E_Th_in": 0.0, "R_0": 1.0, "f_mag": 0.2}
    if m == CANONICAL_MAGNETAR_NI:
        return {"f_mag": 0.2}
    if m == CANONICAL_CSM:
        return {"s": 2.0}
    return {}
