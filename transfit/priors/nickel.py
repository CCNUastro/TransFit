from __future__ import annotations

import numpy as np

# Canonical parameter order for NickelModel:
# (M_ej, v_ej, E_Th_in, M_ni, R_0, f_ni, kappa, kappa_gamma,
#  T_floor, delta, n)
#
# Backward compatibility:
# - old pure-nickel calls that omit E_Th_in and R_0 are still accepted in
#   forward-model helpers and mapped to E_Th_in=0, R_0=10 R_sun.
# - existing parameter dictionaries that omit delta and n use delta=0, n=10.

NICKEL_PARAM_NAMES = [
    "M_ej",         # Msun
    "v_ej",         # 1e9 cm/s
    "E_Th_in",      # 1e49 erg
    "M_ni",         # Msun
    "R_0",          # R_sun
    "f_ni",         # Lagrangian Ni-mixed ejecta mass fraction, [0,1]
    "kappa",        # cm^2/g
    "kappa_gamma",  # cm^2/g
    "T_floor",      # K
    "delta",        # inner BPL density index
    "n",            # outer BPL density index
]

NICKEL_DEFAULT_BOUNDS = np.array([
    [0.1,    10.0],      # M_ej
    [0.1,     3.0],      # v_ej
    [0.01,   10.0],      # E_Th_in
    [0.001,   1.0],      # M_ni
    [0.1,   500.0],      # R_0
    [0.0,     1.0],      # f_ni
    [0.01,    0.5],      # kappa
    [0.001,   0.5],      # kappa_gamma
    [1000.0, 20000.0],   # T_floor
    [0.0,     2.9],      # delta
    [5.1,    14.0],      # n
], dtype=float)

T_SHIFT_NAME = "t_shift"
T_SHIFT_BOUNDS = (0.0, 20.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(NICKEL_PARAM_NAMES)
    bounds = np.array(NICKEL_DEFAULT_BOUNDS, float)

    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.array([T_SHIFT_BOUNDS[0], T_SHIFT_BOUNDS[1]], float)])
    return names, bounds
