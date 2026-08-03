"""Generate the reproducible density-profile figures used by changelog v0.2."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transfit as tf
from transfit.constants import (
    DAY,
    EPSILON_CO,
    EPSILON_NI,
    M_SUN,
    R_SUN,
)
from transfit.models.nickel import (
    _PROFILE_Q_MAX,
    _PROFILE_Q_MIN,
    _eta_q,
    _finite_profile_scales,
    _radioactive_heating_shape,
    _q_profile_moment,
)


NX = 100
NY = 2000
T_MAX_DAYS = 300.0
OUT_DIR = ROOT / "docs/assets/changelog/v0.2"

# Common parameters isolate the density-profile dependence.  In particular,
# v_ej is derived from the requested ejecta mass and kinetic energy through
# E_K=0.5*M_ej*v_ej**2, matching the NickelModel normalization.
M_EJ_M_SUN = 1.0
E_K_ERG = 1.0e51
PARAMS = {
    "M_ej": M_EJ_M_SUN,
    "v_ej": np.sqrt(2.0 * E_K_ERG / (M_EJ_M_SUN * M_SUN)) / 1.0e9,
    "E_Th_in": 1.0,
    "M_ni": 0.2,
    "R_0": 1.0,
    "f_ni": 0.8,
    "kappa": 0.1,
    "kappa_gamma": 0.03,
    "T_floor": 4500.0,
    "delta": 0.0,
    "n": 10.0,
}

PROFILE_META = {
    "uniform": {
        "label": "Uniform",
        "color": "#4C78A8",
        "linestyle": "-",
    },
    "broken_power_law": {
        "label": r"BPL ($\delta=0$, $n=10$)",
        "color": "#D55E00",
        "linestyle": "--",
    },
    "exponential": {
        "label": "Exponential / Ia",
        "color": "#009E73",
        "linestyle": "-.",
    },
}

FIGURE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _density_profile(profile: str) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``rho/rho_mean`` from the common profile ``eta(q)``."""
    delta = PARAMS["delta"]
    n = PARAMS["n"]
    q = np.linspace(_PROFILE_Q_MIN, _PROFILE_Q_MAX, 1000)
    shape = _eta_q(q, profile, delta, n)
    i_mass = _q_profile_moment(
        _PROFILE_Q_MIN,
        _PROFILE_Q_MAX,
        2.0,
        profile,
        delta,
        n,
    )
    i_kin = _q_profile_moment(
        _PROFILE_Q_MIN,
        _PROFILE_Q_MAX,
        4.0,
        profile,
        delta,
        n,
    )

    mass = PARAMS["M_ej"] * M_SUN
    radius = PARAMS["R_0"] * R_SUN
    kinetic_energy = 0.5 * mass * (PARAMS["v_ej"] * 1.0e9) ** 2
    rho_scale, _, _ = _finite_profile_scales(
        mass,
        kinetic_energy,
        radius,
        _PROFILE_Q_MAX,
        i_mass,
        i_kin,
    )
    rho_mean = 3.0 * mass / (4.0 * np.pi * radius**3)
    rho_over_mean = rho_scale * shape / rho_mean
    return q, rho_over_mean


def plot_density_profiles() -> Path:
    output = OUT_DIR / "nickel-density-profiles.png"
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)

    for profile, meta in PROFILE_META.items():
        q, rho_over_mean = _density_profile(profile)
        style = dict(
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=2.2,
            label=meta["label"],
        )
        ax.plot(q, rho_over_mean, **style)

    ax.set(
        yscale="log",
        xlim=(0.0, 1.0),
        ylim=(1.0e-4, 1.0e3),
        xlabel=r"Normalized initial radius $r/R_0$",
        ylabel=r"Initial density $\rho/\bar{\rho}$",
        title="Finite ejecta density profiles",
    )
    ax.legend(loc="lower left", frameon=False)

    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def _solve_lightcurve(profile: str):
    return tf.lightcurve_bol(
        model="nickel",
        params=PARAMS,
        z=0.0,
        t_max_days=T_MAX_DAYS,
        solver_kwargs={
            "Nx": NX,
            "Ny": NY,
            "density_profile": profile,
        },
    )


def _radioactive_deposition(t_days: np.ndarray) -> np.ndarray:
    """Return the Ni/Co heating deposited by the solver's leakage model."""
    t_s = np.asarray(t_days, dtype=float) * DAY
    t_gamma = np.sqrt(
        3.0
        * PARAMS["kappa_gamma"]
        * PARAMS["M_ej"]
        * M_SUN
        / (4.0 * np.pi * (PARAMS["v_ej"] * 1.0e9) ** 2)
    )

    specific_heating = (
        (EPSILON_NI - EPSILON_CO)
        * _radioactive_heating_shape(t_s, t_gamma)
    )
    return PARAMS["M_ni"] * M_SUN * specific_heating


def plot_lightcurve_effect() -> Path:
    output = OUT_DIR / "nickel-density-profile-lightcurves.png"
    solutions = {
        profile: _solve_lightcurve(profile) for profile in PROFILE_META
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)

    for profile, meta in PROFILE_META.items():
        lc = solutions[profile]
        style = dict(
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=2.2,
            label=meta["label"],
        )
        ax.plot(lc.t_days, lc.Lbol, **style)

    heating_time = np.linspace(0.0, T_MAX_DAYS, NY + 1)
    ax.plot(
        heating_time,
        _radioactive_deposition(heating_time),
        color="0.15",
        linestyle=":",
        linewidth=2.0,
        label=r"Deposited $^{56}$Ni+$^{56}$Co heating",
        zorder=2,
    )

    ax.set(
        yscale="log",
        xlim=(0.0, T_MAX_DAYS),
        ylim=(2.0e38, 7.0e42),
        xlabel="Rest-frame time since explosion (d)",
        ylabel=r"Bolometric luminosity (erg s$^{-1}$)",
        title=rf"Density-profile effect on the nickel light curve ($N_x={NX}$, $N_y={NY}$)",
    )
    ax.legend(loc="upper right", frameon=False)

    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> tuple[Path, Path]:
    mpl.rcParams.update(FIGURE_RC)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return plot_density_profiles(), plot_lightcurve_effect()


if __name__ == "__main__":
    for path in main():
        print(path)
