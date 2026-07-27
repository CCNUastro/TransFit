# modules/sed/blackbody.py
# -*- coding: utf-8 -*-
"""
Blackbody SED mapping for gray-radiation models.

Given Teff(t) and Rph(t), produce:
- L_nu(nu_rest, t)  [erg/s/Hz]
- F_nu at observer (requires DL_cm)

This module intentionally contains NO cosmology; DL_cm must be provided by caller.
Magnitude conversion lives in `transfit.modules.magnitudes` and is kept out of the
model core on purpose.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# unified constants (CGS, plain floats; Numba-friendly even though this module is numpy)
from transfit.constants import C_LIGHT, H_PLANCK, K_BOLTZ, PI


@dataclass(frozen=True)
class BlackbodySED:
    """
    Blackbody SED: L_nu = 4*pi^2*R^2*B_nu(T).
    """
    Tmin: float = 1.0   # K, avoid divide-by-zero
    Rmin: float = 1.0   # cm

    def planck_Bnu(self, nu_hz: np.ndarray, T_K: np.ndarray) -> np.ndarray:
        """
        Planck function B_nu [erg/s/cm^2/Hz/sr].
        Vectorized: nu (Nb,) and T (Nt,) -> (Nb, Nt)
        """
        nu = np.asarray(nu_hz, dtype=float).reshape(-1, 1)   # (Nb,1)
        T = np.asarray(T_K, dtype=float).reshape(1, -1)      # (1,Nt)
        T = np.clip(T, self.Tmin, np.inf)

        x = (H_PLANCK * nu) / (K_BOLTZ * T)  # (Nb,Nt)
        # In the Wien tail exp(x) may overflow, which correctly corresponds
        # to B_nu underflowing to zero rather than a numerical failure.
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            denom = np.expm1(x)
            denom = np.where(denom == 0.0, np.inf, denom)
            Bnu = (2.0 * H_PLANCK * nu**3) / (C_LIGHT**2) / denom
        return np.where(np.isfinite(Bnu), Bnu, np.nan)

    def lnu(self, nu_rest_hz: np.ndarray, Teff_K: np.ndarray, R_cm: np.ndarray) -> np.ndarray:
        """
        Luminosity density L_nu [erg/s/Hz], shape (Nb, Nt).
        """
        R = np.asarray(R_cm, dtype=float).reshape(1, -1)
        R = np.clip(R, self.Rmin, np.inf)

        Bnu = self.planck_Bnu(nu_rest_hz, Teff_K)   # (Nb,Nt)
        Lnu = (4.0 * PI * PI) * (R**2) * Bnu
        return np.where(np.isfinite(Lnu), Lnu, np.nan)

    def fnu(
        self,
        nu_obs_hz: np.ndarray,
        Teff_K: np.ndarray,
        R_cm: np.ndarray,
        DL_cm: float,
        z: float = 0.0,
    ) -> np.ndarray:
        """
        Observer-frame flux density F_nu [erg/s/cm^2/Hz], shape (Nb, Nt).

        Uses the luminosity-distance convention:
          nu_rest = nu_obs * (1+z)
          Fnu = (1+z) * Lnu / (4*pi*DL^2)
        """
        zp1 = 1.0 + (z or 0.0)
        nu_rest = np.asarray(nu_obs_hz, dtype=float) * zp1
        Lnu = self.lnu(nu_rest, Teff_K, R_cm)
        Fnu = (zp1 * Lnu) / (4.0 * PI * (DL_cm**2))
        return np.where(Fnu > 0.0, Fnu, np.nan)

    def abmag(
        self,
        nu_obs_hz: np.ndarray,
        Teff_K: np.ndarray,
        R_cm: np.ndarray,
        DL_cm: float,
        z: float = 0.0,
    ) -> np.ndarray:
        """
        Legacy convenience wrapper.

        The main code path now keeps the model core in F_nu and performs
        magnitude conversion in `transfit.modules.magnitudes`.
        """
        Fnu = self.fnu(nu_obs_hz, Teff_K, R_cm, DL_cm=DL_cm, z=z)
        mab = -2.5 * np.log10(Fnu) - 48.6
        return mab
