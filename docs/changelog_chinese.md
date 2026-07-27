# TransFit 更新日志

这里长期记录面向用户的更新，最新版本写在最前面。

[English version](changelog.md)

## v0.2 — 2026-07-26

### 本次更新

- Nickel 模型支持 `uniform`、`bpl`/`broken_power_law` 和
  `exponential`/`ia` 三种密度轮廓。
- `R_0` 表示抛射物有限的初始外半径。
- `f_ni` 表示 Nickel 混合到的拉格朗日质量坐标；当
  `M_ni > f_ni*M_ej` 时，模型会拒绝该组非物理参数。
- BPL 拟合默认固定 `delta=0`、`n=10`。只有显式放入 `priors` 的参数才参与
  拟合。
- Exponential/Ia 拟合默认固定 `R_0=0.01 R_sun`。

### API

正向计算直接选择密度轮廓：

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

拟合时把相同设置放在 `model_kwargs` 中：

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

需要拟合 BPL 指数时再显式提供先验：

```python
priors = {
    "delta": (0.0, 2.0),
    "n": (6.0, 12.0),
}
```

### 密度轮廓

![Uniform、BPL 和指数密度轮廓](assets/changelog/v0.2/nickel-density-profiles.png)

### 热光变曲线

![三种密度轮廓对应的热光变曲线](assets/changelog/v0.2/nickel-density-profile-lightcurves.png)
