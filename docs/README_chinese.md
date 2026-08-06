# TransFit

<p align="right">
  <strong>语言：</strong><a href="../README.md">English</a> | 简体中文
</p>

<p align="center">
  <img src="TransFit_logo.png" width="430" alt="TransFit logo">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <img alt="License" src="https://img.shields.io/badge/License-GPL--3.0-blue">
  <img alt="Inference" src="https://img.shields.io/badge/Inference-MCMC-C0392B">
  <img alt="Models" src="https://img.shields.io/badge/Models-Transient%20Light%20Curves-1F4E79">
  <img alt="Data" src="https://img.shields.io/badge/Data-Bolometric%20%7C%20Multi--band-2E8B57">
</p>

TransFit 是一个用于暂现源光变曲线正向建模和拟合的 Python 软件包。它提供简洁的
bolometric 和多波段数据接口，并内置 nickel、magnetar、magnetar-plus-nickel
以及 CSM 相互作用模型。

## 主要功能

- 输出 bolometric 光度、有效温度和有效辐射半径的物理光变模型。
- 支持流量或星等空间的多波段测光，包括滤光片、消光和 SED 设置。
- 使用统一结果对象进行贝叶斯拟合，默认安装 `emcee`，也可选装 `zeus` 和
  `dynesty`。

## 安装

```bash
python -m pip install transfit
```

本地开发安装：

```bash
git clone <your-repo-url>
cd TransFit
python -m pip install -e ".[plot,examples]"
```

安装可选采样器后端：

```bash
python -m pip install "transfit[all-samplers]"
```

## 快速开始

<details>
<summary><strong>正向计算 bolometric 光变曲线</strong></summary>

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

Nickel 根据 `solver_kwargs["density_profile"]` 自动选择输运。默认 Uniform
保留历史固定外边界光度，返回 `Lphotospheric=Lbol`、`Ldirect=0`，并使用原始
同模膨胀等效黑体。BPL 和 Ia/exponential 使用移动的真实 `tau=2/3` 光球，满足
`Lbol=Lphotospheric+Ldirect`；完全光学薄后其物理 `Rph/Teff` 变为 `NaN`。

<p align="center">
  <img src="lightcurve_bol.png" alt="Bolometric forward model example">
</p>

</details>

<details>
<summary><strong>正向计算多波段光变曲线</strong></summary>

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

Uniform 多波段保留原始同模膨胀半径与逐时刻 `T_floor` 映射。BPL 和
Ia/exponential 在 flux 空间相加两个连续谱：由 `Lphotospheric` 归一化的稀释
物理光球黑体，以及由 `Ldirect` 归一化的 4500 K 黑体连续谱。两者频率积分之和
严格恢复 `Lbol`；完全光学薄后只剩光球外分量。这仍不是星云发射线模型。内置
滤波器目前只在一个有效频率处计算 SED，尚未实现完整 throughput 积分。

`filters` 会把数据中的 band 标签映射到具体滤波器定义。内置滤波器使用字符串
ID；自定义单点滤波器推荐使用有效波长：

```python
filters = {
    "g": {"lambda_eff_A": 4770.0},
    "r": {"lambda_eff_nm": 623.1},
}
```

如果自定义 Vega 星等滤波器，还需要给 Vega 零点：

```python
filters = {
    "B": {"lambda_eff_A": 4400.0, "vega_zero_point_jy": 4260.0},
}
```

<p align="center">
  <img src="lightcurve_multiband.png" alt="Multi-band forward model example">
</p>

</details>

<details>
<summary><strong>拟合 bolometric 光变曲线</strong></summary>

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
<summary><strong>拟合多波段光变曲线</strong></summary>

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
    },
    sampler_kwargs={"nwalkers": 32, "nsteps": 5000, "burnin": 1000, "thin": 10},
)

print(res.best_params_raw)
tf.save(res, "mcmc_out/sn2007gr_multiband_nickel.npz")
```

</details>

## 公开 API

<details>
<summary><strong>数据容器</strong></summary>

```python
tf.BolometricData(t_days, y, yerr, mask=None)
tf.MultiBandData(t_days, band, y, yerr, mask=None)
```

</details>

<details>
<summary><strong>模型参数查看</strong></summary>

```python
tf.model_param_names("nickel")
tf.param_template("csm")
```

规范模型名包括 `nickel`、`magnetar`、`magnetar_ni` 和 `csm`。

</details>

<details>
<summary><strong>正向计算和插值预测</strong></summary>

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
<summary><strong>拟合</strong></summary>

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
<summary><strong>结果、画图和读写</strong></summary>

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

完整说明见 [API 和参数参考](api_reference_chinese.md)。

## 文档

- [教程 notebook](../examples/tutorial.ipynb)
- [API 和参数参考](api_reference_chinese.md)
- [更新日志](changelog_chinese.md)

## 联系方式

如果对本项目有问题，请联系：

- Liangduan Liu ([liuld@ccnu.edu.cn](mailto:liuld@ccnu.edu.cn))
- Yuhao Zhang ([zhangyh2001@foxmail.com](mailto:zhangyh2001@foxmail.com))
- GuangLei Wu ([wuguanglei@mails.ccnu.edu.cn](mailto:wuguanglei@mails.ccnu.edu.cn))

## 引用

如果您觉得 TransFit 对您的工作有帮助，请考虑给我们打一个星，这可以帮助其他人发现这个项目。

[![GitHub stars](https://img.shields.io/github/stars/YuHaoZhang01/TransFit?style=social&label=Stars)](https://github.com/YuHaoZhang01/TransFit/stargazers)

如果你在科研工作中使用 TransFit，请引用 TransFit 论文：

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

对于 `csm` 模型，还请引用：

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

## 使用 TransFit 的论文

- Ni et al., *Mapping the Dense Circumstellar Environments of SNe Ibn, SNe Icn, and Fast Blue Optical Transients*, arXiv e-print (2026), [arXiv:2607.00453](https://arxiv.org/abs/2607.00453)。
- Yuan et al., *Thermal X-rays breaking out from pre-explosion ejecta of a dying massive star*, arXiv e-print (2026), [arXiv:2606.10014](https://arxiv.org/abs/2606.10014)。
- Liu et al., *SN 2024igg: A Super-Chandrasekhar/03fg-like SN exhibiting C II-dominated spectra after explosion*, submitted to A&A (2026), [arXiv:2602.03427](https://arxiv.org/abs/2602.03427)。

## AI 辅助声明

本项目中的部分代码和文档是在 OpenAI Codex 辅助下生成的。
