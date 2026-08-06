# TransFit Changelog

<p align="right">
  <strong>Language:</strong> English | <a href="changelog_chinese.md">简体中文</a>
</p>

User-visible changes are recorded here, with the newest version first.

## v0.2

### What's new

- The CSM model now radiates from the photosphere, whose radius follows the
  forward-shock position:

```math
R_{\mathrm{ph}}(t)=R_{\mathrm{CSM,ph}}
\qquad \left(R_{\mathrm{FS}}\lt R_{\mathrm{CSM,ph}}\right)
```

```math
R_{\mathrm{ph}}(t)=R_{\mathrm{FS}}(t)
\qquad \left(R_{\mathrm{CSM,ph}}\le R_{\mathrm{FS}}\lt R_{\mathrm{CSM,out}}\right)
```

```math
R_{\mathrm{ph}}(t)=a(t)R_{\mathrm{CSM,out}}
\qquad \left(R_{\mathrm{FS}}\ge R_{\mathrm{CSM,out}}\right)
```

Here, $a(t)$ is the homologous expansion factor after shock exit.

- The CSM ejecta-density indices default to fixed values `n=10` and `delta=0`.
  They can be sampled through `priors` or set explicitly through `fixed`.
- Optional reverse-shock heating is available for the CSM model. It is disabled
  by default and enters the same diffusion calculation when
  `reverse_shock=True`.

- Time sampling around the CSM transition has been improved so that the rapid
  early cooling evolution remains smooth and stable.
- The nickel model supports `uniform`, `bpl`/`broken_power_law`, and
  `exponential`/`ia` density profiles.
- Nickel automatically selects the radiation strategy from the normalized
  density name: Uniform preserves the historical outer-boundary solve and
  homologous blackbody, while BPL/Ia use the moving $\tau=2/3$ photosphere and
  direct escape outside it.
- BPL/Ia multi-band emission adds a diluted physical-photosphere blackbody
  normalized to `Lphotospheric` and a floor-temperature continuum normalized
  to `Ldirect` in flux space, so exterior power is not compressed onto the
  retreating photosphere.
- `R_0` is the progenitor-star radius and the initial outer radius of the
  homologously expanding ejecta at explosion.
- `f_ni` is the Lagrangian mass coordinate reached by nickel mixing. The model
  rejects combinations with `M_ni > f_ni*M_ej`.
- BPL fits keep `delta=0` and `n=10` fixed by default. Add either parameter to
  `priors` only when it should be fitted.
- Exponential/Ia fits fix `R_0=0.01 R_sun` by default.

### Density profiles

![Uniform, BPL, and exponential density profiles](assets/changelog/v0.2/nickel-density-profiles.png)

### Bolometric light curves

![Bolometric light curves for the three density profiles](assets/changelog/v0.2/nickel-density-profile-lightcurves.png)

### Nickel density and photosphere

The Nickel ejecta start at the progenitor radius `R_0` and expand homologously:

```math
R_{\rm out}(t)=R_0+v_{\max}t,
\qquad
\rho(v,t)=\rho(v,0)\left(\frac{R_0}{R_{\rm out}(t)}\right)^3 .
```

The model uses the following three physical density structures. Uniform is

```math
\rho_{\rm Uniform}(v,t)=\rho_{\rm u}(t),
\qquad 0\le v\le v_{\max}.
```

The inner and outer BPL regions are, respectively,

```math
\rho_{\rm BPL}(v,t)=\rho_t(t)(v/v_t)^{-\delta},
\qquad 0\le v<v_t,
```

```math
\rho_{\rm BPL}(v,t)=\rho_t(t)(v/v_t)^{-n},
\qquad v_t\le v\le v_{\max}.
```

The Ia/exponential profile is

```math
\rho_{\rm Ia}(v,t)=\rho_e(t)\exp(-v/v_e),
\qquad 0\le v\le v_{\max}.
```

The BPL profile is truncated at `v_max=3 v_t`, while the Ia/exponential profile
is truncated at `v_max=12 v_e`. The outward optical depth is obtained directly
from the physical density:

```math
\tau(v,t)=\int_{r(v,t)}^{R_{\rm out}(t)}\kappa\rho(r',t)\,dr'
```

The density profile automatically selects the radiation boundary. Uniform uses
the outer-boundary diffusion luminosity, while BPL/Ia use the physical
photosphere and direct escape outside it:

```math
L_{\rm bol}=L_{\rm out}\quad(\mathrm{Uniform}),\qquad
L_{\rm bol}=L_{\rm photospheric}+L_{\rm direct}
\quad(\mathrm{BPL/Ia}).
```

For BPL and Ia/exponential density, the physical photosphere satisfies

```math
\tau(v_{\rm ph},t)=\frac{2}{3}
```

The photospheric radius and temperature are

```math
R_{\rm ph}=R_{\rm out}\frac{v_{\rm ph}}{v_{\max}},\qquad
T_{\rm ph}=\left(\frac{L_{\rm photospheric}}
{4\pi\sigma R_{\rm ph}^2}\right)^{1/4} .
```

### Nickel multi-band emission

Uniform density preserves the historical homologous blackbody:

```math
R_{\rm hom}=R_0+v_{\max}t,\qquad
T_{\rm try}=\left(\frac{L_{\rm bol}}
{4\pi\sigma R_{\rm hom}^2}\right)^{1/4} .
```

BPL and Ia/exponential use the physical photospheric effective temperature and
a pointwise color floor:

```math
T_{\rm eff,ph}=\left(\frac{L_{\rm photospheric}}
{4\pi\sigma R_{\rm ph}^2}\right)^{1/4},\qquad
T_{\rm col,ph}=\max(T_{\rm eff,ph},T_{\rm floor}).
```

The photospheric dilution factor is

```math
W_{\rm ph}=\left(\frac{T_{\rm eff,ph}}{T_{\rm col,ph}}\right)^4,
```

and the two continua are

```math
L_\nu^{\rm ph}=4\pi^2R_{\rm ph}^2W_{\rm ph}B_\nu(T_{\rm col,ph}),
```

```math
L_\nu^{\rm direct}=L_{\rm direct}
\frac{\pi B_\nu(T_{\rm floor})}{\sigma T_{\rm floor}^4},\qquad
L_\nu=L_\nu^{\rm ph}+L_\nu^{\rm direct}.
```

Thus each component recovers its own bolometric luminosity and their sum
recovers `Lbol`. After complete optical transparency, the photospheric term is
zero and only the direct continuum remains.

### API calls

```python
# Options: "uniform", "bpl" / "broken_power_law", "ia" / "exponential"
# Default Uniform uses the historical outer boundary; BPL/Ia use tau=2/3.
solver = {"Nx": 100, "Ny": 1000, "density_profile": "bpl"}

bol = tf.lightcurve_bol(
    model="nickel", params=params, z=0.001728,
    solver_kwargs=solver,
)

multiband = tf.lightcurve_multiband(
    model="nickel", params={**params, "T_floor": 4500.0},
    z=0.001728,
    filters={"B": "johnson_cousins.B", "V": "johnson_cousins.V"},
    bands=["B", "V"], y_kind="mag", mag_system="ab",
    solver_kwargs=solver,
)

bol_fit = tf.fit_bol(
    data=bol_data, model="nickel", priors=bol_priors, fixed=bol_fixed,
    model_kwargs={"solver_kwargs": solver},
)

multiband_fit = tf.fit_multiband(
    data=multiband_data, model="nickel", z=0.001728,
    filters={"B": "johnson_cousins.B", "V": "johnson_cousins.V"},
    priors={**bol_priors, "T_floor": (3000.0, 10000.0)},
    fixed=multiband_fixed,  # do not also fix T_floor here
    model_kwargs={"solver_kwargs": solver},
)
```

`fit_bol` does not include `T_floor`. `fit_multiband` fixes
`T_floor=4500 K` by default and samples it only when it is explicitly included
in `priors`.

Input aliases are normalized before solving: `bpl` is equivalent to
`broken_power_law`, while `exp` and `ia` are equivalent to `exponential`.
Aliases and canonical names produce identical results.
