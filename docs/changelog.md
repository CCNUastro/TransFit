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
- `R_0` is the finite initial outer radius of the ejecta.
- `f_ni` is the Lagrangian mass coordinate reached by nickel mixing. The model
  rejects combinations with `M_ni > f_ni*M_ej`.
- BPL fits keep `delta=0` and `n=10` fixed by default. Add either parameter to
  `priors` only when it should be fitted.
- Exponential/Ia fits fix `R_0=0.01 R_sun` by default.

### API

Select the density profile directly for a forward calculation:

```python
lc = tf.lightcurve_bol(
    model="nickel",
    params=params,
    solver_kwargs={
        "Nx": 100,
        "Ny": 1000,
        "density_profile": "bpl",
    },
)
```

For fitting, put the same settings under `model_kwargs`:

```python
result = tf.fit_bol(
    data=data,
    model="nickel",
    model_kwargs={
        "solver_kwargs": {
            "Nx": 100,
            "Ny": 1000,
            "density_profile": "bpl",
        }
    },
)
```

To fit the BPL indices explicitly:

```python
priors = {
    "delta": (0.0, 2.0),
    "n": (6.0, 12.0),
}
```

### Density profiles

![Uniform, BPL, and exponential density profiles](assets/changelog/v0.2/nickel-density-profiles.png)

### Bolometric light curves

![Bolometric light curves for the three density profiles](assets/changelog/v0.2/nickel-density-profile-lightcurves.png)

### Nickel density and photosphere

All density structures are written as

```math
\rho(q,t)=\rho_0\,\eta(q)f_R^{-3},\qquad
q=\frac{r}{R_{\rm out}}=\frac{v}{v_{\max}},\qquad
R_{\rm out}=R_0+v_{\max}t,\qquad
f_R=\frac{R_{\rm out}}{R_0} .
```

with

```math
\eta(q)=
\begin{cases}
1, & \mathrm{Uniform},\\
(q/q_t)^{-\delta}, & \mathrm{BPL},\ q<q_t,\\
(q/q_t)^{-n}, & \mathrm{BPL},\ q\ge q_t,\\
\exp(-q/q_e), & \mathrm{Ia/exponential},
\end{cases}
\qquad q_t=\frac{1}{3},\quad q_e=\frac{1}{12}.
```

The outward optical depth is

```math
\tau(q,t)=\frac{\kappa\rho_0R_0}{f_R^2}
\int_q^1\eta(q')\,dq' .
```

The physical photosphere satisfies

```math
\tau(q_{\rm ph},t)=\frac{2}{3}
```

Deposited power inside the photosphere enters diffusion, while deposited power
outside escapes directly. Therefore

```math
L_{\rm bol}=L_{\rm photospheric}+L_{\rm direct},\qquad
R_{\rm ph}=R_{\rm out}q_{\rm ph},\qquad
T_{\rm ph}=\left(\frac{L_{\rm photospheric}}
{4\pi\sigma R_{\rm ph}^2}\right)^{1/4} .
```

Once the total optical depth is below $2/3$,
`photosphere_valid=False`, `Lphotospheric=0`, `Lbol=Ldirect`, and the physical
`Rph/Teff` values are `NaN`.

### Nickel multi-band emission

The effective-temperature treatment is unchanged from the original Uniform
density model and uses the homologous blackbody:

```math
R_{\rm hom}=R_0+v_{\max}t,\qquad
T_{\rm try}=\left(\frac{L_{\rm bol}}
{4\pi\sigma R_{\rm hom}^2}\right)^{1/4} .
```

The final blackbody state and spectrum are

```math
(T_{\rm BB},R_{\rm BB})=
\begin{cases}
(T_{\rm try},R_{\rm hom}), & T_{\rm try}>T_{\rm floor},\\
\left(T_{\rm floor},\sqrt{L_{\rm bol}/(4\pi\sigma T_{\rm floor}^4)}\right),
& T_{\rm try}\le T_{\rm floor},
\end{cases}
\qquad
L_\nu=4\pi^2R_{\rm BB}^2B_\nu(T_{\rm BB}) .
```

### Calling the model

```python
# Options: "uniform", "bpl" / "broken_power_law", "ia" / "exponential"
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
