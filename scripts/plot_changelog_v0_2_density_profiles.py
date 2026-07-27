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
from transfit.constants import M_SUN, R_SUN
from transfit.models.nickel import (
    _BPL_V_MAX_OVER_V_T,
    _BPL_X_MIN,
    _EXPONENTIAL_V_MAX_OVER_V_E,
    _EXPONENTIAL_X_MIN,
    _broken_power_law_integrals,
    _density_mass_integral,
    _exponential_integrals,
    _finite_profile_scales,
    _integral_power_law,
)


NX = 100
NY = 1000
T_MAX_DAYS = 150.0
OUT_DIR = ROOT / "docs/assets/changelog/v0.2"

# Common parameters isolate the density-profile dependence.  In particular,
# E_Th_in=0 removes the initial shock-cooling component.
PARAMS = {
    "M_ej": 3.0,
    "v_ej": 1.0,
    "E_Th_in": 0.0,
    "M_ni": 0.08,
    "R_0": 1.0,
    "f_ni": 0.8,
    "kappa": 0.12,
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


def _profile_grid(profile: str) -> tuple[float, float, float, float]:
    """Return x_min, x_max, dimensionless mass and energy integrals."""
    delta = PARAMS["delta"]
    n = PARAMS["n"]
    if profile == "uniform":
        x_min, x_max = 1.0, 1.0e4
        i_mass = _density_mass_integral(x_min, x_max, profile, delta, n)
        i_kin = _integral_power_law(x_min, x_max, 4.0)
    elif profile == "broken_power_law":
        x_min, x_max = _BPL_X_MIN, _BPL_V_MAX_OVER_V_T
        i_mass, i_kin = _broken_power_law_integrals(
            x_min, x_max, delta, n
        )
    elif profile == "exponential":
        x_min, x_max = _EXPONENTIAL_X_MIN, _EXPONENTIAL_V_MAX_OVER_V_E
        i_mass, i_kin = _exponential_integrals(x_min, x_max)
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return float(x_min), float(x_max), float(i_mass), float(i_kin)


def _density_profile(profile: str) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the initial density on the model's normalized-radius axis."""
    x_min, x_max, i_mass, i_kin = _profile_grid(profile)
    q_min = x_min / x_max
    q = np.linspace(q_min, 1.0, 1000)
    x = q * x_max

    if profile == "uniform":
        shape = np.ones_like(x)
    elif profile == "broken_power_law":
        shape = np.where(
            x < 1.0,
            x ** (-PARAMS["delta"]),
            x ** (-PARAMS["n"]),
        )
    else:
        shape = np.exp(-x)

    mass = PARAMS["M_ej"] * M_SUN
    radius = PARAMS["R_0"] * R_SUN
    kinetic_energy = 0.5 * mass * (PARAMS["v_ej"] * 1.0e9) ** 2
    rho_scale, _, _ = _finite_profile_scales(
        mass,
        kinetic_energy,
        radius,
        x_max,
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

    ax.set(
        yscale="log",
        xlim=(0.0, T_MAX_DAYS),
        ylim=(5.0e39, 2.0e42),
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
