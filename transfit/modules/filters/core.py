from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional

import numpy as np
from astropy import units as u
try:
    from astroquery.svo_fps import SvoFps
except ImportError:
    SvoFps = None


@dataclass(frozen=True)
class FilterProfile:
    label: str
    filter_id: str
    kind: Literal["mono", "bandpass"]
    source: Literal["builtin", "svo_fps", "user", "legacy"]
    detector: Literal["energy", "photon"] = "energy"
    nu_eff_hz: Optional[float] = None
    wavelength_A: Optional[np.ndarray] = None
    throughput: Optional[np.ndarray] = None
    zero_points_jy: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        label = str(self.label).strip()
        if not label:
            raise ValueError("Filter label must be a non-empty string.")
        filter_id = str(self.filter_id).strip()
        if not filter_id:
            raise ValueError("filter_id must be a non-empty string.")

        kind = str(self.kind).strip().lower()
        if kind not in ("mono", "bandpass"):
            raise ValueError(f"Unknown filter kind {self.kind!r}.")

        source = str(self.source).strip().lower()
        if source not in ("builtin", "svo_fps", "user", "legacy"):
            raise ValueError(f"Unknown filter source {self.source!r}.")

        detector = str(self.detector).strip().lower()
        if detector not in ("energy", "photon"):
            raise ValueError(f"Unknown detector type {self.detector!r}.")

        nu_eff_hz = None if self.nu_eff_hz is None else float(self.nu_eff_hz)
        if kind == "mono":
            if nu_eff_hz is None or not np.isfinite(nu_eff_hz) or nu_eff_hz <= 0.0:
                raise ValueError("Mono filters require a positive finite nu_eff_hz.")

        wavelength_A = None
        throughput = None
        if self.wavelength_A is not None or self.throughput is not None:
            wavelength_A = np.asarray(self.wavelength_A, float).reshape(-1)
            throughput = np.asarray(self.throughput, float).reshape(-1)
            if wavelength_A.size == 0 or throughput.size == 0:
                raise ValueError("Bandpass arrays must be non-empty.")
            if wavelength_A.shape != throughput.shape:
                raise ValueError("wavelength_A and throughput must have the same shape.")
            if np.any(~np.isfinite(wavelength_A)) or np.any(~np.isfinite(throughput)):
                raise ValueError("Bandpass arrays must be finite.")
            if np.any(np.diff(wavelength_A) <= 0.0):
                raise ValueError("wavelength_A must be strictly increasing.")
            if np.any(throughput < 0.0) or not np.any(throughput > 0.0):
                raise ValueError("throughput must be non-negative and contain at least one positive value.")

        zero_points_jy = {
            str(k).strip().lower(): float(v)
            for k, v in dict(self.zero_points_jy or {}).items()
        }
        for key, value in zero_points_jy.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"Zero point {key!r} must be positive and finite.")

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "filter_id", filter_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "detector", detector)
        object.__setattr__(self, "nu_eff_hz", nu_eff_hz)
        object.__setattr__(self, "wavelength_A", wavelength_A)
        object.__setattr__(self, "throughput", throughput)
        object.__setattr__(self, "zero_points_jy", zero_points_jy)
        object.__setattr__(self, "meta", dict(self.meta or {}))

    @property
    def lambda_eff_A(self) -> float:
        if self.nu_eff_hz is not None:
            nu_eff = float(self.nu_eff_hz) * u.Hz
            return nu_eff.to_value(u.AA, u.spectral())
        return None

    @classmethod
    def from_svofps(cls, label: str, svo_fps_id: str, **kwargs) -> "FilterProfile":
        if SvoFps is None:
            raise RuntimeError("The astroquery.svo_fps package could not be imported. Is astroquery installed?")

        inst_name = svo_fps_id.split("/")[1].split(".")[0]
        transmission = SvoFps.get_transmission_data(svo_fps_id)
        metadata = SvoFps.get_filter_metadata(svo_fps_id)
        lambda_eff = metadata.get("WavelengthEff")
        if lambda_eff is not None:
            nu_eff_hz = lambda_eff.to_value(u.Hz, u.spectral())
        else:
            nu_eff_hz = None
        return cls(
            label=str(label),
            filter_id=f"svo_fps:{svo_fps_id}",
            kind="bandpass",
            source="svo_fps",
            detector=str(kwargs.get("detector", "energy")),
            nu_eff_hz=nu_eff_hz,
            wavelength_A=transmission["Wavelength"].quantity.to_value(u.AA),
            throughput=transmission["Transmission"].quantity.value,
            zero_points_jy={metadata["MagSys"].lower(): metadata["ZeroPoint"].to_value(u.Jy, u.spectral_density(lambda_eff))},
            meta={
                "family": metadata.get("PhotSystem") or inst_name,
                "svo_fps_id": metadata.get("filterID", svo_fps_id),
                "comment": metadata.get("Comments", ""),
            }
        )

    @classmethod
    def from_dict(cls, label: str, payload: Mapping[str, object]) -> "FilterProfile":
        nu_eff_hz = payload.get("nu_eff_hz")
        if nu_eff_hz is None and payload.get("lambda_eff_A") is not None:
            lambda_eff = u.Quantity(payload["lambda_eff_A"], u.AA)
            nu_eff_hz = lambda_eff.to_value(u.Hz, u.spectral())
        return cls(
            label=str(payload.get("label", label)),
            filter_id=str(payload.get("filter_id", payload.get("id", f"user:{label}"))),
            kind=str(payload.get("kind", "mono")),
            source=str(payload.get("source", "user")),
            detector=str(payload.get("detector", "energy")),
            nu_eff_hz=nu_eff_hz,
            wavelength_A=payload.get("wavelength_A"),
            throughput=payload.get("throughput"),
            zero_points_jy=dict(payload.get("zero_points_jy", {}) or {}),
            meta=dict(payload.get("meta", {}) or {}),
        )

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "label": self.label,
            "filter_id": self.filter_id,
            "kind": self.kind,
            "source": self.source,
            "detector": self.detector,
            "zero_points_jy": dict(self.zero_points_jy),
            "meta": dict(self.meta),
        }
        if self.nu_eff_hz is not None:
            out["nu_eff_hz"] = float(self.nu_eff_hz)
            out["lambda_eff_A"] = self.lambda_eff_A
        if self.wavelength_A is not None:
            out["wavelength_A"] = self.wavelength_A.tolist()
        if self.throughput is not None:
            out["throughput"] = self.throughput.tolist()
        return out