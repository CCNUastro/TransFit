# TransFit

<p align="right">
  <strong>Language:</strong> English | <a href="docs/README_chinese.md">简体中文</a>
</p>

<p align="center">
  <img src="docs/TransFit_logo.png" width="430" alt="TransFit logo">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <img alt="License" src="https://img.shields.io/badge/License-GPL--3.0-blue">
  <img alt="Inference" src="https://img.shields.io/badge/Inference-MCMC-C0392B">
  <img alt="Models" src="https://img.shields.io/badge/Models-Transient%20Light%20Curves-1F4E79">
  <img alt="Data" src="https://img.shields.io/badge/Data-Bolometric%20%7C%20Multi--band-2E8B57">
</p>

TransFit is a Python package for forward modeling and fitting astronomical
transient light curves. It provides a compact interface for bolometric and
multi-band data, with built-in nickel, magnetar, magnetar-plus-nickel, and CSM
interaction models.

## Features

- Physical light-curve models with bolometric luminosity, effective
  temperature, and photospheric radius outputs.
- Multi-band photometry in flux or magnitude space, including filter,
  extinction, and SED handling.
- Bayesian fitting through a consistent result object, with `emcee` installed
  by default and optional `zeus` and `dynesty` backends.

## Installation

```bash
python -m pip install transfit
```

For local development:

```bash
git clone <your-repo-url>
cd TransFit
python -m pip install -e ".[plot,examples]"
```

Install optional sampler backends with:

```bash
python -m pip install "transfit[all-samplers]"
```

## Quick Start

<details>
<summary><strong>Forward bolometric light curve</strong></summary>

```python
import matplotlib.pyplot as plt
import transfit as tf

params = {
    "M_ej": 3.0,
    "v_ej": 1.0,
    "E_Th_in": 1.5,
    "M_ni": 0.08,
    "R_0": 120.0,
    "f_ni": 0.2,
    "kappa": 0.12,
    "kappa_gamma": 0.03,
    "T_floor": 4500.0,
}

lc = tf.lightcurve_bol(
    model="nickel",
    params=params,
    z=0.001728,
    t_max_days=120.0,
)

plt.plot(lc.t_days, lc.Lbol)
plt.yscale("log")
plt.xlabel("Observer-frame time (days)")
plt.ylabel("Bolometric luminosity (erg s$^{-1}$)")
plt.show()
```

<p align="center">
  <img src="docs/lightcurve_bol.png" alt="Bolometric forward model example">
</p>

</details>

<details>
<summary><strong>Forward multi-band light curve</strong></summary>

```python
import matplotlib.pyplot as plt
import transfit as tf

params = {
    "M_ej": 3.0,
    "v_ej": 1.0,
    "E_Th_in": 1.5,
    "M_ni": 0.08,
    "R_0": 120.0,
    "f_ni": 0.2,
    "kappa": 0.12,
    "kappa_gamma": 0.03,
    "T_floor": 4500.0,
}

filters = {
    "B": "johnson_cousins.B",
    "V": "johnson_cousins.V",
    "R": "johnson_cousins.R",
    "I": "johnson_cousins.I",
}

lc = tf.lightcurve_multiband(
    model="nickel",
    params=params,
    z=0.001728,
    filters=filters,
    bands=["B", "V", "R", "I"],
    y_kind="mag",
    mag_system="vega",
    t_max_days=120.0,
)

for band in lc.bands:
    plt.plot(lc.t_days, lc.y[band], label=band)
plt.gca().invert_yaxis()
plt.xlabel("Observer-frame time (days)")
plt.ylabel("Vega magnitude")
plt.legend()
plt.show()
```

`filters` maps the band labels in your data to filter definitions. Built-in
filters use string IDs. Custom mono filters should use an effective wavelength:

```python
filters = {
    "g": {"lambda_eff_A": 4770.0},
    "r": {"lambda_eff_nm": 623.1},
}
```

For custom Vega magnitudes, also provide a Vega zero point:

```python
filters = {
    "B": {"lambda_eff_A": 4400.0, "vega_zero_point_jy": 4260.0},
}
```

<p align="center">
  <img src="docs/lightcurve_multiband.png" alt="Multi-band forward model example">
</p>

</details>

<details>
<summary><strong>Fit a bolometric light curve</strong></summary>

```python
import numpy as np
import transfit as tf

arr = np.loadtxt("examples/data/sn1993j_lbol.txt")
data = tf.BolometricData(
    t_days=arr[:, 0] - arr[:, 0].min(),
    y=arr[:, 1],
    yerr=arr[:, 2],
)

res = tf.fit_bol(
    data=data,
    model="nickel",
    z=0.001728,
    priors={
        "M_ej": (0.5, 8.0),
        "v_ej": (0.2, 3.0),
        "E_Th_in": (0.05, 8.0),
        "M_ni": ("log10", -3.0, -0.2),
        "R_0": (10.0, 400.0),
        "t_shift": (0.0, 20.0),
    },
    fixed={
        "f_ni": 0.2,
        "kappa": 0.12,
        "kappa_gamma": 0.03,
    },
    sampler_kwargs={"nwalkers": 32, "nsteps": 5000, "burnin": 1000, "thin": 10},
)

print(res.best_params_raw)
tf.save(res, "mcmc_out/sn1993j_bol_nickel.npz")
```

</details>

<details>
<summary><strong>Fit a multi-band light curve</strong></summary>

```python
import numpy as np
import transfit as tf

raw = np.genfromtxt(
    "examples/data/sn2007gr.csv",
    delimiter=",",
    names=True,
    dtype=float,
    encoding="utf-8",
)

bands, t_days, y, yerr = [], [], [], []
t0 = np.nanmin(raw["Phase"])
columns = {
    "B": ("Bmag", "e_Bmag"),
    "V": ("Vmag", "e_Vmag"),
    "R": ("Rmag", "e_Rmag"),
    "I": ("Imag", "e_Imag"),
}

for band, (mag_col, err_col) in columns.items():
    good = (
        np.isfinite(raw["Phase"])
        & np.isfinite(raw[mag_col])
        & np.isfinite(raw[err_col])
        & (raw[err_col] > 0)
    )
    t_days.extend((raw["Phase"][good] - t0).tolist())
    y.extend(raw[mag_col][good].tolist())
    yerr.extend(raw[err_col][good].tolist())
    bands.extend([band] * int(np.sum(good)))

data = tf.MultiBandData(
    t_days=np.asarray(t_days, float),
    band=np.asarray(bands, dtype=object),
    y=np.asarray(y, float),
    yerr=np.asarray(yerr, float),
)

filters = {
    "B": "johnson_cousins.B",
    "V": "johnson_cousins.V",
    "R": "johnson_cousins.R",
    "I": "johnson_cousins.I",
}

res = tf.fit_multiband(
    data=data,
    model="nickel",
    z=0.001728,
    filters=filters,
    y_kind="mag",
    mag_system="vega",
    priors={
        "M_ej": (0.5, 8.0),
        "v_ej": (0.2, 3.0),
        "E_Th_in": (0.05, 8.0),
        "M_ni": ("log10", -3.0, -0.2),
        "R_0": (10.0, 400.0),
        "t_shift": (0.0, 20.0),
    },
    fixed={
        "f_ni": 0.2,
        "kappa": 0.12,
        "kappa_gamma": 0.03,
        "T_floor": 4500.0,
    },
    sampler_kwargs={"nwalkers": 32, "nsteps": 5000, "burnin": 1000, "thin": 10},
)

print(res.best_params_raw)
tf.save(res, "mcmc_out/sn2007gr_multiband_nickel.npz")
```

</details>

## Public API

<details>
<summary><strong>Data containers</strong></summary>

```python
tf.BolometricData(t_days, y, yerr, mask=None)
tf.MultiBandData(t_days, band, y, yerr, mask=None)
```

</details>

<details>
<summary><strong>Model inspection</strong></summary>

```python
tf.model_param_names("nickel")
tf.param_template("csm")
```

Canonical model names are `nickel`, `magnetar`, `magnetar_ni`, and `csm`.

</details>

<details>
<summary><strong>Forward and prediction</strong></summary>

```python
tf.lightcurve_bol(model=..., params=..., z=..., t_max_days=...)
tf.lightcurve_multiband(
    model=...,
    params=...,
    z=...,
    filters=...,
    bands=...,
    y_kind="mag",
)

tf.predict_bol(model=..., params=..., z=..., t_days=...)
tf.predict_multiband(
    model=...,
    params=...,
    z=...,
    filters=...,
    t_days=...,
    band=...,
)
```

</details>

<details>
<summary><strong>Fitting</strong></summary>

```python
tf.fit_bol(
    data=...,
    model=...,
    z=...,
    priors=...,
    fixed=...,
    sampler="emcee",
    sampler_kwargs=None,
    model_kwargs=None,
)

tf.fit_multiband(
    data=...,
    model=...,
    z=...,
    filters=...,
    y_kind="mag",
    priors=...,
    fixed=...,
    sed=None,
    sampler="emcee",
    sampler_kwargs=None,
    model_kwargs=None,
)
```

</details>

<details>
<summary><strong>Results, plotting, and I/O</strong></summary>

```python
res.best_params
res.best_params_raw
res.median_params
res.best_fit
res.best_index
res.best_log_prob
res.best_sample
res.samples
res.log_prob
res.meta

tf.plot.fit_bol(res, data=data)
tf.plot.fit_multiband(res, data=data)
tf.plot.corner(res)

path = tf.save(res, path="mcmc_out/result.npz")
loaded = tf.load(path)
```

</details>

Full details are available in [API and parameter reference](docs/api_reference.md).

## Documentation

- [Tutorial notebook](examples/tutorial.ipynb)
- [API and parameter reference](docs/api_reference.md)

## Contact

For questions about this project, please contact:

- Liangduan Liu ([liuld@ccnu.edu.cn](mailto:liuld@ccnu.edu.cn))
- Yuhao Zhang ([zhangyh2001@foxmail.com](mailto:zhangyh2001@foxmail.com))
- GuangLei Wu ([wuguanglei@mails.ccnu.edu.cn](mailto:wuguanglei@mails.ccnu.edu.cn))
- Jingyao Li ([lijy@mails.ccnu.edu.cn](mailto:lijy@mails.ccnu.edu.cn))

## Citation

If TransFit is helpful for your work, please consider giving the repository a star. This helps other researchers discover the project.

[![GitHub stars](https://img.shields.io/github/stars/YuHaoZhang01/TransFit?style=social&label=Stars)](https://github.com/YuHaoZhang01/TransFit/stargazers)

If you use TransFit in research, please cite the TransFit paper:

```bibtex
@ARTICLE{2025ApJ...992...20L,
       author = {{Liu}, Liang-Duan and {Zhang}, Yu-Hao and {Yu}, Yun-Wei and {Du}, Ze-Xin and {Li}, Jing-Yao and {Wu}, Guang-Lei and {Dai}, Zi-Gao},
        title = "{TransFit: An Efficient Framework for Transient Light-curve Fitting with Time-dependent Radiative Diffusion}",
      journal = {\apj},
     keywords = {Supernovae, Radiative transfer, Core-collapse supernovae, Time domain astronomy, 1668, 1335, 304, 2109, High Energy Astrophysical Phenomena, Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = oct,
       volume = {992},
       number = {1},
          eid = {20},
        pages = {20},
          doi = {10.3847/1538-4357/adfed6},
archivePrefix = {arXiv},
       eprint = {2505.13825},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...992...20L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

For the `csm` model, also cite:

```bibtex
@ARTICLE{2026ApJ...999..186Z,
       author = {{Zhang}, Yu-Hao and {Liu}, Liang-Duan and {Du}, Ze-Xin and {Wu}, Guang-Lei and {Li}, Jing-Yao and {Yu}, Yun-Wei},
        title = "{TransFit-CSM: A Fast, Physically Consistent Framework for Interaction-powered Transients}",
      journal = {\apj},
     keywords = {Core-collapse supernovae, Supernovae, Circumstellar matter, Stellar mass loss, 304, 1668, 241, 1613, High Energy Astrophysical Phenomena},
         year = 2026,
        month = mar,
       volume = {999},
       number = {2},
          eid = {186},
        pages = {186},
          doi = {10.3847/1538-4357/ae434a},
archivePrefix = {arXiv},
       eprint = {2511.13265},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026ApJ...999..186Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Papers Using TransFit

- Yuan et al., *Thermal X-rays breaking out from pre-explosion ejecta of a dying massive star*, arXiv e-print (2026), [arXiv:2606.10014](https://arxiv.org/abs/2606.10014).
- Liu et al., *SN 2024igg: A Super-Chandrasekhar/03fg-like SN exhibiting C II-dominated spectra after explosion*, submitted to A&A (2026), [arXiv:2602.03427](https://arxiv.org/abs/2602.03427).

## AI Assistance

Parts of the code and documentation in this project were generated with
assistance from OpenAI Codex.
