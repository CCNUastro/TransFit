from __future__ import annotations

import numpy as np

# Canonical public parameter order for CSMModel:
# (M_ej, E_sn, M_csm, R_csm_out, kappa, s, n, delta, eps_sh, T_floor)
#
# Backward compatibility:
# - forward-model helpers allow s and T_floor to be omitted.
# - R_csm_in remains fixed to an internal default in the public API.
# - n and delta are public physical parameters, but fitting fixes them to
#   n=10 and delta=0 unless the user supplies explicit priors.
# - bolometric fitting treats T_floor as an internal numerical floor.
# - tau-photosphere multiband fitting fixes T_floor because that mode derives
#   temperature directly from the tau=2/3 photosphere radius.

CSM_PARAM_NAMES = [
    "M_ej",        # Msun
    "E_sn",        # 1e51 erg
    "M_csm",       # Msun
    "R_csm_out",   # R_sun
    "kappa",       # cm^2/g
    "s",           # CSM density power-law index
    "n",           # ejecta outer density power-law index
    "delta",       # ejecta inner density power-law index
    "eps_sh",      # [0,1]
    "T_floor",     # K
]

CSM_DEFAULT_BOUNDS = np.array([
    [0.3,     50.0],      # M_ej
    [0.1,     50.0],      # E_sn
    [0.01,    10.0],      # M_csm
    [100.0, 100000.0],    # R_csm_out
    [0.01,    0.34],      # kappa
    [0.0,      2.0],      # s
    [5.1,     14.0],      # n
    [0.0,      2.9],      # delta
    [0.01,     1.0],      # eps_sh
    [1000.0, 20000.0],    # T_floor
], dtype=float)

T_SHIFT_NAME = "t_shift"
T_SHIFT_BOUNDS = (0.0, 20.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(CSM_PARAM_NAMES)
    bounds = np.array(CSM_DEFAULT_BOUNDS, float)

    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.array([T_SHIFT_BOUNDS[0], T_SHIFT_BOUNDS[1]], float)])
    return names, bounds
