# TransFit Changelog

User-visible changes are recorded here, with the newest version first.

[中文版本](changelog_chinese.md)

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
- BPL/Ia multi-band emission converts `Ldirect` into an equivalent emitting
  area at $T_{\rm floor}$ before normalizing the blackbody with the complete
  `Lbol`, so exterior power is not compressed onto the retreating photosphere.
- `R_0` is the finite initial outer radius of the ejecta.
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

The ejecta expand homologously. Define

```math
t_{\rm h}=t+\frac{R_0}{v_{\max}},\qquad
r=v t_{\rm h},\qquad
R_{\rm out}=v_{\max}t_{\rm h}=R_0+v_{\max}t .
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

The BPL profile is truncated at $v_{\max}=3v_t$, while the Ia/exponential
profile is truncated at $v_{\max}=12v_e$. The three density normalizations
decline with homologous expansion as

```math
\rho_{\rm u}(t)\propto t_{\rm h}^{-3},\qquad
\rho_t(t)\propto t_{\rm h}^{-3},\qquad
\rho_e(t)\propto t_{\rm h}^{-3}.
```

Their normalization and $v_{\max}$ are fixed jointly by the ejecta mass and
kinetic energy:

```math
M_{\rm ej}=4\pi t_{\rm h}^{3}
\int_0^{v_{\max}}\rho(v,t)v^2\,dv,
\qquad
E_K=2\pi t_{\rm h}^{3}
\int_0^{v_{\max}}\rho(v,t)v^4\,dv
=\frac{1}{2}M_{\rm ej}v_{\rm ej}^2 .
```

Thus the input $v_{\rm ej}$ is the characteristic velocity defined by the
total kinetic energy, not a common fixed outer-edge velocity for every density
profile. The outward optical depth follows directly from the physical density:

```math
\tau(v,t)=\int_{r(v,t)}^{R_{\rm out}(t)}\kappa\rho(r',t)\,dr'
=\kappa t_{\rm h}\int_v^{v_{\max}}\rho(v',t)\,dv' .
```

The density profile automatically selects the radiation boundary. Uniform
density preserves the historical outer-boundary diffusion luminosity:

```math
L_{\rm bol}=L_{\rm out},\qquad L_{\rm direct}=0.
```

This historical path keeps the original gamma treatment: the Ni term is fully
trapped and leakage is applied only to the Co term.

For BPL and Ia/exponential density, the physical photosphere satisfies

```math
\tau(v_{\rm ph},t)=\frac{2}{3}
```

Deposited power inside the photosphere enters diffusion, while deposited power
outside escapes directly. Therefore

```math
L_{\rm bol}=L_{\rm photospheric}+L_{\rm direct},\qquad
R_{\rm ph}=v_{\rm ph}t_{\rm h},\qquad
T_{\rm ph}=\left(\frac{L_{\rm photospheric}}
{4\pi\sigma R_{\rm ph}^2}\right)^{1/4} .
```

For BPL and Ia/exponential, once the total optical depth is below $2/3$,
`photosphere_valid=False`, `Lphotospheric=0`, `Lbol=Ldirect`, and the physical
`Rph/Teff` values are `NaN`.

### Nickel multi-band emission

Uniform density preserves the historical homologous blackbody:

```math
R_{\rm hom}=R_0+v_{\max}t,\qquad
T_{\rm try}=\left(\frac{L_{\rm bol}}
{4\pi\sigma R_{\rm hom}^2}\right)^{1/4} .
```

BPL and Ia/exponential first convert `Ldirect` to a floor-temperature emitting
radius:

```math
R_{\rm direct}=\left(\frac{L_{\rm direct}}
{4\pi\sigma T_{\rm floor}^4}\right)^{1/2},\qquad
R_{\rm try}=\left(R_{\rm ph}^2+R_{\rm direct}^2\right)^{1/2}.
```

The complete `Lbol` then sets the temperature:

```math
T_{\rm try}=\left(\frac{L_{\rm bol}}
{4\pi\sigma R_{\rm try}^2}\right)^{1/4} .
```

At a valid photospheric node with $T_{\rm try}>T_{\rm floor}$, use

```math
T_{\rm BB}=T_{\rm try},\qquad R_{\rm BB}=R_*.
```

At all other epochs, use

```math
T_{\rm BB}=T_{\rm floor},\qquad
R_{\rm BB}=\sqrt{\frac{L_{\rm bol}}
{4\pi\sigma T_{\rm floor}^4}}.
```

The corresponding blackbody spectrum is

```math
L_\nu=4\pi^2R_{\rm BB}^2B_\nu(T_{\rm BB}) .
```

The emitting radius is

```math
R_*=R_{\rm hom}\quad(\mathrm{Uniform}),\qquad
R_*=R_{\rm try}\quad(\mathrm{BPL/Ia}).
```

Once fully thin, $R_{\rm try}=R_{\rm direct}$ naturally. Cool epochs use a floor
radius normalized to the complete `Lbol`; there is no additional nebular SED
component.

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
