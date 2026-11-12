from __future__ import annotations

import numpy as np


# A shared-radiation-field ejecta + thin-shell + CSM model.  E_sn and M_ej
# set the ejecta velocity scale; there is deliberately no independent v_ej.
NICKEL_CSM_PARAM_NAMES = [
    "M_ej",          # Msun
    "E_sn",          # 1e51 erg
    "M_ni",          # Msun
    "f_ni",          # outer Ni mixing coordinate in v / v_max
    "M_csm",         # Msun
    "R_csm_in",      # R_sun
    "R_csm_out",     # R_sun
    "kappa_ej",      # cm^2/g
    "kappa_csm",     # cm^2/g
    "kappa_gamma",   # cm^2/g
    "s",             # CSM density power-law index
    "eps_fs",        # forward-shock radiative efficiency
    "eps_rs",        # reverse-shock radiative efficiency
    "T_floor",       # K
]


NICKEL_CSM_DEFAULT_BOUNDS = np.array([
    [0.3,       50.0],      # M_ej
    [0.1,       50.0],      # E_sn
    [1.0e-4,     5.0],      # M_ni
    [0.01,        1.0],      # f_ni
    [0.01,       20.0],      # M_csm
    [1.0,     10000.0],      # R_csm_in
    [100.0,  100000.0],      # R_csm_out
    [0.01,        0.4],      # kappa_ej
    [0.01,        0.4],      # kappa_csm
    [1.0e-3,      0.2],      # kappa_gamma
    [0.0,          2.9],      # s
    [0.0,          1.0],      # eps_fs
    [0.0,          1.0],      # eps_rs
    [1000.0,   20000.0],      # T_floor
], dtype=float)

T_SHIFT_NAME = "t_shift"
T_SHIFT_BOUNDS = (0.0, 20.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(NICKEL_CSM_PARAM_NAMES)
    bounds = np.array(NICKEL_CSM_DEFAULT_BOUNDS, dtype=float)
    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.asarray(T_SHIFT_BOUNDS, dtype=float)])
    return names, bounds
