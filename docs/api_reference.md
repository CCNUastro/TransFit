# TransFit API and Parameter Reference

This document describes the stable public Python interface of TransFit. The
README and tutorial intentionally show only minimal examples; this file is the
place for argument meanings, model parameters, result fields, and advanced
options.

Chinese version: [中文 API 和参数参考](api_reference_chinese.md).

All public time inputs and outputs use **observer-frame days**. The physical
models are solved internally in rest-frame time and converted back to the
observer frame at the API boundary.

## Public Entry Points

| Category | Entry points |
|---|---|
| Data containers | `BolometricData`, `MultiBandData` |
| Model inspection | `model_param_names(model)`, `param_template(model)` |
| Forward light curves | `lightcurve_bol(...)`, `lightcurve_multiband(...)` |
| Interpolated predictions | `predict_bol(...)`, `predict_multiband(...)` |
| Fitting | `fit_bol(...)`, `fit_multiband(...)` |
| Result I/O | `save(res, path=None)`, `load(path, trusted=False)` |
| Plotting | `transfit.plot.fit_bol`, `transfit.plot.fit_multiband`, `transfit.plot.corner` |

## Model Names and Parameters

Accepted canonical model names are `nickel`, `magnetar`, `magnetar_ni`, and
`csm`. Some aliases are accepted for backward compatibility, but new scripts
should use the canonical names.

### `nickel`

| Parameter | Meaning and unit |
|---|---|
| `M_ej` | ejecta mass, $M_\odot$ |
| `v_ej` | ejecta velocity, $10^9\,{\rm cm\,s^{-1}}$ |
| `E_Th_in` | initial thermal energy, $10^{49}\,{\rm erg}$ |
| `M_ni` | nickel mass, $M_\odot$ |
| `R_0` | initial radius, $R_\odot$ |
| `f_ni` | outer Lagrangian mass coordinate of the Ni-mixed region, $M(<x_{\rm Ni})/M_{\rm ej}$ |
| `kappa` | optical opacity, ${\rm cm^2\,g^{-1}}$ |
| `kappa_gamma` | gamma-ray opacity, ${\rm cm^2\,g^{-1}}$ |
| `T_floor` | temperature floor for the Nickel physical-photosphere multi-band blackbody; defaults to 4500 K |
| `delta` | inner BPL density index, dimensionless |
| `n` | outer BPL density index, dimensionless |

### `magnetar`

| Parameter | Meaning and unit |
|---|---|
| `M_ej` | ejecta mass, $M_\odot$ |
| `v_ej` | ejecta velocity, $10^9\,{\rm cm\,s^{-1}}$ |
| `E_Th_in` | initial thermal energy, $10^{49}\,{\rm erg}$ |
| `P_ms` | magnetar spin period, ms |
| `B14` | magnetar dipole field, $10^{14}\,{\rm G}$ |
| `f_mag` | magnetar heating mixing coordinate, dimensionless |
| `R_0` | initial radius, $R_\odot$ |
| `kappa` | optical opacity, ${\rm cm^2\,g^{-1}}$ |
| `kappa_gamma` | gamma-ray opacity, ${\rm cm^2\,g^{-1}}$ |
| `T_floor` | temperature floor, K |

### `magnetar_ni`

| Parameter | Meaning and unit |
|---|---|
| `M_ej` | ejecta mass, $M_\odot$ |
| `v_ej` | ejecta velocity, $10^9\,{\rm cm\,s^{-1}}$ |
| `P_ms` | magnetar spin period, ms |
| `B14` | magnetar dipole field, $10^{14}\,{\rm G}$ |
| `f_mag` | magnetar heating mixing coordinate, dimensionless |
| `M_ni` | nickel mass, $M_\odot$ |
| `f_ni` | nickel mixing coordinate, dimensionless |
| `kappa` | optical opacity, ${\rm cm^2\,g^{-1}}$ |
| `kappa_gamma` | gamma-ray opacity, ${\rm cm^2\,g^{-1}}$ |
| `T_floor` | temperature floor, K |

### `csm`

| Parameter | Meaning and unit |
|---|---|
| `M_ej` | ejecta mass, $M_\odot$ |
| `E_sn` | explosion energy, $10^{51}\,{\rm erg}$ |
| `M_csm` | circumstellar-material mass, $M_\odot$ |
| `R_csm_out` | outer CSM radius, $R_\odot$ |
| `kappa` | optical opacity, ${\rm cm^2\,g^{-1}}$ |
| `s` | CSM density power-law index |
| `n` | outer ejecta density power-law index |
| `delta` | inner ejecta density power-law index |
| `eps_sh` | shock radiation efficiency |
| `T_floor` | temperature floor, K |

The fitting interface uses the optional parameter `t_shift` to shift the model
time axis. It is constrained to be non-negative; set `t_shift=0.0` in `fixed`
when no time shift should be fitted. In fitting, the model is evaluated at:

```text
t_eval = t_obs + t_shift
```

A positive `t_shift` means that the model start is earlier than the user's
observational zero point.

For `magnetar` and `magnetar_ni`, `f_mag` is part of the public parameter
schema but defaults to a fixed value of `0.2` in fitting when omitted. To fit it,
pass an explicit prior such as `priors={"f_mag": (0.05, 0.5)}`. Forward-model
helpers also use `f_mag=0.2` when it is not provided in `params`.

## Data Containers

### `BolometricData`

```python
tf.BolometricData(t_days, y, yerr, mask=None)
```

`BolometricData` stores bolometric observations.

| Field | Meaning |
|---|---|
| `t_days` | observer-frame time in days |
| `y` | bolometric luminosity, ${\rm erg\,s^{-1}}$ |
| `yerr` | one-sigma uncertainty in the same units as `y` |
| `mask` | optional boolean mask; only masked-in points are used |

Unmasked luminosities and uncertainties must be positive and finite for
fitting.

### `MultiBandData`

```python
tf.MultiBandData(t_days, band, y, yerr, mask=None)
```

`MultiBandData` stores multi-band photometry.

| Field | Meaning |
|---|---|
| `t_days` | observer-frame time in days |
| `band` | band label for each point |
| `y` | magnitude if `y_kind="mag"`, flux density if `y_kind="flux"` |
| `yerr` | one-sigma uncertainty in the same units as `y` |
| `mask` | optional boolean mask; only masked-in points are used |

## Forward and Prediction Calls

Bolometric forward call:

```python
params = {
    **params,
    "delta": 0.0,
    "n": 10.0,
}

tf.lightcurve_bol(
    model="nickel",
    params=params,
    z=0.001728,
    t_max_days=150.0,
    solver_kwargs={"density_profile": "bpl"},
)
```

The backward-compatible default is `density_profile="uniform"`. Use either
`"bpl"`/`"broken_power_law"` for the broken power law or
`"exp"`/`"exponential"`/`"ia"` for a finite exponential Type-Ia-like profile.

Returns `BolometricLC` with:

- `t_days`
- `Lbol`
- `Teff`
- `Rph`
- `Lphotospheric`: Uniform outer-boundary luminosity or BPL/Ia photospheric luminosity
- `Ldirect`: zero for Uniform; deposited power outside the BPL/Ia photosphere
- `photosphere_valid`: all true for the Uniform effective blackbody; physical mask for BPL/Ia

Uniform returns `Lphotospheric=Lbol`, `Ldirect=0`, and the historical effective
blackbody `Teff/Rph`. BPL and Ia/exponential satisfy
`Lbol=Lphotospheric+Ldirect`; after complete optical transparency they return
`Lphotospheric=0`, `photosphere_valid=False`, and physical `Teff/Rph=NaN`.

Multi-band forward call:

```python
tf.lightcurve_multiband(
    model="nickel",
    params=params,
    z=0.001728,
    distance_modulus=None,
    filters=filters,
    bands=["B", "V"],
    y_kind="mag",
    mag_system="ab",
    extinction=None,
    sed=None,
    t_max_days=150.0,
    solver_kwargs=None,
)
```

Returns `MultiBandLC` with:

- `t_days`
- `bands`
- `y[band]`

Uniform uses its historical homologous radius,

\[
R_{\rm hom}=R_0+v_{\max}t,\qquad
T_{\rm try}=\left(\frac{L_{\rm bol}}
{4\pi\sigma R_{\rm hom}^2}\right)^{1/4}.
\]

BPL and Ia/exponential convert `Ldirect` to a floor-temperature radius and add
its area to the physical photospheric area:

\[
R_{\rm direct}=\left(\frac{L_{\rm direct}}
{4\pi\sigma T_{\rm floor}^4}\right)^{1/2},\qquad
R_{\rm try}=\left(R_{\rm ph}^2+R_{\rm direct}^2\right)^{1/2},
\]

\[
T_{\rm try}=\left(\frac{L_{\rm bol}}
{4\pi\sigma R_{\rm try}^2}\right)^{1/4}.
\]

The hot branch uses `(Rhom,T_try)` for Uniform and `(R_try,T_try)` for BPL/Ia.
Otherwise it uses

\[
T_{\rm BB}=T_{\rm floor},\qquad
R_{\rm BB}=\left(\frac{L_{\rm bol}}
{4\pi\sigma T_{\rm floor}^4}\right)^{1/2}.
\]

After complete optical transparency, `R_try=R_direct` and the mapping naturally
stays on the floor. No separate nebular SED mode is introduced.

The observer-frame conversion is

\[
F_{\nu,\rm obs}(\nu_{\rm obs})=
\frac{(1+z)L_\nu[(1+z)\nu_{\rm obs}]}{4\pi D_L^2}.
\]

AB outputs then use

\[
m_{\rm AB}=-2.5\log_{10}\!\left(F_\nu/3631\,{\rm Jy}\right).
\]

The built-in filters in this release are monochromatic effective-frequency
definitions. They evaluate the SED at `nu_eff_hz`; they do not integrate a
throughput curve. The retained Nickel mapping is a continuum approximation and
does not model nebular emission lines or wavelength-dependent photospheres.

### Filter Definitions

`filters` maps the band labels in `MultiBandData.band` to physical filter
definitions. Built-in filters use string IDs:

```python
filters = {
    "B": "johnson_cousins.B",
    "V": "johnson_cousins.V",
}
```

Custom mono filters should use effective wavelength as the public input:

```python
filters = {
    "g": {"lambda_eff_A": 4770.0},
    "r": {"lambda_eff_nm": 623.1},
    "i": {"lambda_eff_um": 0.7625},
}
```

If `mag_system="vega"` is used with a custom filter, provide a Vega zero point:

```python
filters = {
    "B": {"lambda_eff_A": 4400.0, "vega_zero_point_jy": 4260.0},
}
```

`nu_eff_hz` remains accepted for backward compatibility, but new user-facing
examples should prefer `lambda_eff_A`. Full bandpass throughput integration is
not implemented yet.

`predict_bol` and `predict_multiband` evaluate the same models at
user-supplied observer-frame times. `interp_fill` may be `"nan"`, `"raise"`,
or `"edge"` for prediction calls. During fitting, `"edge"` is rejected to avoid
silently extrapolating outside the model grid.

## Fitting Calls

Bolometric fit:

```python
res = tf.fit_bol(
    data=data,
    model="nickel",
    z=0.001728,
    priors=priors,
    fixed=fixed,
    sampler="emcee",
    sampler_kwargs=None,
    model_kwargs=None,
)
```

Multi-band fit:

```python
res = tf.fit_multiband(
    data=data,
    model="nickel",
    z=0.001728,
    distance_modulus=None,
    filters=filters,
    y_kind="mag",
    mag_system="ab",
    extinction=None,
    priors=priors,
    fixed=fixed,
    sampler="emcee",
    sed=None,
    sampler_kwargs=None,
    model_kwargs=None,
)
```

`priors` maps parameter names to bounds. A linear uniform prior uses
`(lo, hi)`. A base-10 log-uniform prior uses `("log10", lo, hi)`, where `lo`
and `hi` are bounds in log10 space.

`fixed` maps parameter names to fixed values. In general, model parameters not
supplied in `fixed` are sampled using their default bounds or the bounds supplied
in `priors`.

There are two intentional exceptions. In `fit_bol`, `T_floor` is not part of
the sampled bolometric fit state. In Nickel multi-band fits, `T_floor` defaults
to a fixed 4500 K and becomes sampled only when it is explicitly supplied in
`priors`. In `magnetar`
and `magnetar_ni`, `f_mag` defaults to a fixed value of `0.2` unless the user
supplies an explicit prior for `f_mag` and does not also fix it.

### `sigma_int`

`sigma_int` is a likelihood nuisance parameter, not a physical model
parameter. It may be fixed or sampled through `fixed` and `priors`.

| Observation space | Meaning |
|---|---|
| `y_kind="mag"` | additional magnitude scatter |
| `y_kind="flux"` | converted to fractional flux scatter using $0.4\ln(10)\sigma_{\rm int}$ |

## Keyword Dictionaries

### `sampler_kwargs`

Common `emcee` and `zeus` keys:

| Key | Meaning |
|---|---|
| `nwalkers` | number of walkers |
| `nsteps` | production chain length |
| `burnin` | burn-in steps before production |
| `thin` | thinning factor |
| `seed` | random seed |
| `init` | initial-position mode or array |
| `pool` | user-supplied parallel pool |
| `progress` | show sampler progress |

Common `dynesty` keys:

| Key | Meaning |
|---|---|
| `nlive` | number of live points |
| `sample` | dynesty sampling method |
| `bound` | dynesty bounding method |
| `dlogz` | stopping tolerance |
| `maxiter` | maximum iterations |
| `maxcall` | maximum likelihood calls |
| `seed` | random seed |
| `progress` | show sampler progress |
| `nsamples` | number of posterior samples returned |
| `add_live` | include live points in posterior |
| `pool` | user-supplied parallel pool |
| `queue_size` | dynesty queue size |

### `model_kwargs`

Fit-time model options are passed through `model_kwargs`.

| Key | Meaning |
|---|---|
| `t_max_days` | observer-frame model duration in days |
| `interp_fill` | interpolation fill policy; `"edge"` is not allowed during fitting |
| `solver_kwargs` | advanced numerical-grid options |

If `t_max_days` is omitted, TransFit chooses a value large enough to cover the
data and the allowed `t_shift` range.

### `solver_kwargs`

`solver_kwargs` is the advanced numerical-grid interface.

| Key | Default | Meaning |
|---|---:|---|
| `Nx` | `100` | spatial/grid resolution parameter |
| `Ny` | `1000` | time/grid resolution parameter |

`Nx` and `Ny` must be positive integers.

Uniform preserves the historical linear nodes `t_i=t_max*i/Ny`. BPL and
Ia/exponential use the nested quadratic grid `t_i=t_max*(i/Ny)^2`.

For the `nickel` model, `solver_kwargs` additionally accepts:

| Key | Default | Meaning |
|---|---|---|
| `density_profile` | `"uniform"` | Density structure: `"uniform"`, `"bpl"`/`"broken_power_law"`, or `"exp"`/`"exponential"`/`"ia"`. |

Uniform density uses the historical fixed outer diffusion boundary and reports
its outer flux as `Lbol`, with `Ldirect=0`. It preserves full trapping for the
Ni term and applies gamma leakage only to the Co term. For BPL and
Ia/exponential, the
transport boundary satisfies

```text
tau(q_ph -> 1) = 2/3.
```

The cell intersected by `q_ph` is integrated as a cut cell. Stored radiation
exposed by the receding surface is released explicitly, internal face fluxes
remain conservative, and the same Marshak flux is used both as the last-cell
sink and reported photospheric luminosity. Radioactive source integrals below and
above that boundary add exactly to the requested deposited power. Once the
complete ejecta column is below `2/3`, `Lbol` equals the deposited heating; no
empirical tail normalization is applied.

The public physical photosphere is

\[
R_{\rm ph}=R_{\rm out}q_{\rm ph},\qquad
T_{\rm ph}=\left(\frac{L_{\rm photospheric}}
{4\pi\sigma R_{\rm ph}^2}\right)^{1/4}.
\]

No temperature floor is applied to the BPL/Ia physical quantity. The Uniform
outer boundary is selected by `density_profile="uniform"`; FLD and late-time
aliases are not public solver modes.

For the `csm` model, `solver_kwargs` additionally accepts:

| Key | Default | Meaning |
|---|---:|---|
| `photosphere_mode` | `"tau"` | `"tau"` uses the optical-depth photosphere; `"outer"` retains the legacy outer-boundary/temperature-floor treatment. |
| `reverse_shock` | `False` | Add reverse-shock heating to the forward-shock source. |

The reverse-shock option is enabled with:

```python
solver_kwargs={
    "Nx": 100,
    "Ny": 1000,
    "reverse_shock": True,
}
```

When enabled, the heating power is

$$
L_{\rm sh}=L_{\rm FS}+L_{\rm RS},\qquad
L_{\rm RS}=2\pi\epsilon_{\rm sh}R_{\rm sh}^{2}\rho_{\rm ej}
\left(v_{\rm ej}-v_{\rm sh}\right)^3.
$$

The reverse shock uses the same active interval, deposition position, numerical
kernel, and diffusion treatment as the forward shock. The switch changes only
the heating source; it does not change the thin-shell dynamics. With
`return_full=True`, `L_forward_shock` and `L_reverse_shock` provide the two
instantaneous heating powers separately.

In the default CSM `"tau"` mode, the PDE grid always covers the complete CSM
from `R_csm_in` to `R_csm_out`; the radius with outward optical depth `tau=2/3`
is an internal photosphere, not the numerical outer boundary. Before the
forward shock reaches it, the reported luminosity is the outward diffusive flux
at that fixed photosphere. The emitting radius and flux-evaluation surface then
follow the forward shock until CSM exit. The numerical shock source is deposited
one `Nx=100` reference cell (1% of the full CSM width) downstream, leaving a
source-free buffer in which the local outward flux is measured at the physical
shock surface. After exit, shock heating is switched off and the complete
full-CSM radiation-energy profile is handed to a separate
source-free cooling solve. Its Rannacher-started Crank--Nicolson equation
includes the homologous adiabatic loss `-4 (d ln a/dy) e` for the dimensionless
energy density `e=E_rad/u0`, applied exactly through the integrating factor
`q=a^4 e`. The cooling luminosity is evaluated at the expanding CSM outer
boundary and is not replaced by an imposed power law. The time grid explicitly
includes both physical transition times.

The CSM ejecta structure uses a broken power law with outer index `n` and inner
index `delta`. Forward calls default to `n=10` and `delta=0` when either value is
omitted. Both parameters remain fixed at those values in `fit_bol` and
`fit_multiband` unless the caller explicitly supplies a prior. Values supplied
through `fixed` are also accepted. For example:

```python
priors = {
    "n": (7.0, 14.0),
    "delta": (0.0, 2.0),
}
```

The physical constraints are `n > 5`, `n > s`, and `0 <= delta < 3`.

An optically thin CSM with total radial optical depth at or below `2/3` is
outside this diffusion model. Forward calls raise a physical-domain error;
fitting treats such a sample as `-inf` rather than terminating the sampler.
`T_floor` is inactive and fixed in `"tau"` multiband fits. To retain and fit
the legacy temperature-floor prescription, select:

```python
model_kwargs={
    "solver_kwargs": {
        "Nx": 100,
        "Ny": 1000,
        "photosphere_mode": "outer",
    }
}
```

For the nickel model, `delta` and `n` are physical model parameters rather than
solver options:

| Parameter | Default | Default prior bounds | Meaning |
|---|---:|---:|---|
| `delta` | `0.0` | `[0.0, 2.9]` | Inner BPL density index; physically requires `0 <= delta < 3`. |
| `n` | `10.0` | `[5.1, 14.0]` | Outer BPL density index; physically requires `n > 5`. |

For forward calculations, omitted values use `delta=0` and `n=10`. The same
values remain fixed by default in `fit_bol` and `fit_multiband`, including when
`density_profile="bpl"` is selected. To sample either BPL index, the caller must
explicitly put that parameter in `priors`; an explicit value in `fixed` keeps
it fixed. Uniform and exponential profiles do not use these parameters and
reject attempts to sample them.

The BPL is normalized using the sampled `M_ej` and `v_ej`, with
`rho/rho_t = x**(-delta)` below the transition `x=1` and `x**(-n)` above it.
The historical Uniform solver uses `1 <= x <= 10^4`, equivalent to
`10^-4 <= q=x/10^4 <= 1`, with Crank--Nicolson. BPL and exponential use the
common `q` coordinate with backward Euler; `q_t=v_t/v_max=1/3` and
`q_e=v_e/v_max=1/12` are encoded inside `eta(q)`.

All three profiles are finite at the initial outer radius `R_0`. Their density
and velocity scales are computed from finite-domain mass and kinetic-energy
integrals, so the represented ejecta recover exactly the input `M_ej` and
`E_K=0.5*M_ej*v_ej**2`; no mass at infinity is assumed. The Ni cutoff is set
by the Lagrangian mass coordinate `f_ni=M(<x_Ni)/M_ej`. The solver derives the
profile-dependent radius/velocity coordinate `x_Ni`, assigns a constant Ni
abundance inside that cutoff, and uses the density-weighted mass integral to
ensure the integrated Ni mass remains `M_ni`. Thus `f_ni=0.8` means 80% of the
represented ejecta mass, not necessarily `x_Ni/x_max=0.8`.

For an exponential forward call:

```python
tf.lightcurve_bol(
    model="nickel",
    params=params,
    z=0.001728,
    solver_kwargs={"density_profile": "ia"},
)
```

For `fit_bol` and `fit_multiband`, the exponential/Ia profile does not sample
`R_0`. It defaults to the fixed white-dwarf radius `R_0=0.01 R_sun`; an
explicit `fixed={"R_0": value}` overrides that default. Supplying `R_0` in
`priors` is rejected. The model time grid starts at the explosion epoch
`t=0`, while `R_0/v_max` remains only an expansion timescale.

For a purely radioactive Ia calculation (`E_Th_in=0`), changing the fixed
radius from `0.01` to `1 R_sun` changes the luminosity after five days by less
than about 0.2% in the regression model. A non-zero initial thermal energy can
make the first few days strongly radius-dependent, so the radius choice is not
generally negligible for shock-cooling calculations.

Selecting BPL through `model_kwargs` keeps `delta=0` and `n=10` fixed by
default:

```python
result = tf.fit_bol(
    data=data,
    model="nickel",
    model_kwargs={"solver_kwargs": {"density_profile": "bpl"}},
)

assert "delta" not in result.param_names
assert "n" not in result.param_names
assert result.fixed["delta"] == 0.0
assert result.fixed["n"] == 10.0
```

To fit the density indices, opt in explicitly with
`priors={"delta": (0.0, 2.0), "n": (6.0, 12.0)}`. Either parameter may be
enabled independently. Both are continuous real values; the physical limits
are `0 <= delta < 3` and `n > 5`.
See the [v0.2.0 changelog](changelog.md#v020--2026-07-31) for the complete
selection table, compatibility behavior, and reproducible figures.

## SED Choices

The default multi-band SED is `BlackbodySED`.

```python
from transfit.modules.sed import BlackbodySED, CutoffBlackbodySED

sed = BlackbodySED()
sed = CutoffBlackbodySED(
    cutoff_wavelength_A=3000.0,
    uv_slope=2.0,
    min_factor=0.0,
)
```

`CutoffBlackbodySED` applies a short-wavelength cutoff:

```text
L_nu_cutoff = C(lambda_rest) * L_nu_blackbody
```

with:

```text
C(lambda) = 1                                      for lambda >= lambda_cut
C(lambda) = max(f_min, (lambda/lambda_cut)^a)      for lambda < lambda_cut
```

where:

| Symbol | API parameter |
|---|---|
| `lambda_cut` | `cutoff_wavelength_A` |
| `a` | `uv_slope` |
| `f_min` | `min_factor` |

Set `min_factor=0` for a pure power-law cutoff.

## FitResult Fields

`fit_bol` and `fit_multiband` return a `FitResult`.

If `t_shift` is sampled during fitting, it appears directly in `res.param_names`
and in the parameter dictionaries such as `res.best_params`,
`res.best_params_raw`, `res.median_params`, and `res.best_fit["params"]`.

| Field/property | Meaning |
|---|---|
| `res.best_params` | rounded best-fit parameter dictionary |
| `res.best_params_raw` | full-precision best-fit parameter dictionary |
| `res.median_params` | posterior median parameter dictionary |
| `res.best_fit` | compact record with parameters, errors, best log probability, and best sample |
| `res.best_index` | index of the best posterior sample |
| `res.best_log_prob` | best log posterior value |
| `res.best_sample` | raw best sample vector in `res.param_names` order |
| `res.samples` | flattened posterior samples |
| `res.log_prob` | log posterior values |
| `res.meta` | sampler, prior, model, SED, and context metadata |

## Citation Rules

All model use should cite the TransFit paper.
The `csm` model should additionally cite the TransFit-CSM paper. See
[model citation guide](https://github.com/YuHaoZhang01/TransFit/blob/main/docs/model_citations.md) for BibTeX entries and
model-specific guidance.
