# TransFit 更新日志

这里长期记录面向用户的更新，最新版本写在最前面。

[English version](changelog.md)

## v0.2

### 本次更新

- CSM 模型改为在光球面进行辐射，光球半径随正向激波位置演化：

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

其中 $a(t)$ 为激波穿出后的同模膨胀因子。

- CSM 抛射物密度指数默认固定为 `n=10`、`delta=0`；也可以通过 `priors`
  参与拟合，或通过 `fixed` 显式固定。
- CSM 模型新增可选的反向激波加热；默认关闭，设置
  `reverse_shock=True` 即可与正向激波一起进入扩散计算。

- 改善了 CSM 关键阶段附近的时间采样，使冷却初期的快速光度变化更加
  平滑、稳定。
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

### Nickel 密度与光球计算

三种密度结构统一写为

```math
\rho(q,t)=\rho_0\,\eta(q)f_R^{-3},\qquad
q=\frac{r}{R_{\rm out}}=\frac{v}{v_{\max}},\qquad
R_{\rm out}=R_0+v_{\max}t,\qquad
f_R=\frac{R_{\rm out}}{R_0} .
```

其中

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

向外光深为

```math
\tau(q,t)=\frac{\kappa\rho_0R_0}{f_R^2}
\int_q^1\eta(q')\,dq' .
```

物理光球由

```math
\tau(q_{\rm ph},t)=\frac{2}{3}
```

确定。光球内沉积参与扩散，光球外沉积直接逃逸，因此

```math
L_{\rm bol}=L_{\rm photospheric}+L_{\rm direct},\qquad
R_{\rm ph}=R_{\rm out}q_{\rm ph},\qquad
T_{\rm ph}=\left(\frac{L_{\rm photospheric}}
{4\pi\sigma R_{\rm ph}^2}\right)^{1/4} .
```

总光深低于 $2/3$ 后，`photosphere_valid=False`、
`Lphotospheric=0`、`Lbol=Ldirect`，并令物理 `Rph/Teff=NaN`。

### Nickel 多波段

有效温度的处理与原 Uniform 密度模型相同，使用同模膨胀黑体：

```math
R_{\rm hom}=R_0+v_{\max}t,\qquad
T_{\rm try}=\left(\frac{L_{\rm bol}}
{4\pi\sigma R_{\rm hom}^2}\right)^{1/4} .
```

最终黑体量和频谱为

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

### 调用方式

```python
# 可选："uniform"、"bpl" / "broken_power_law"、"ia" / "exponential"
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
    fixed=multiband_fixed,  # 不要在这里同时固定 T_floor
    model_kwargs={"solver_kwargs": solver},
)
```

`fit_bol` 不包含 `T_floor`；`fit_multiband` 默认固定 `T_floor=4500 K`，只有
显式放入 `priors` 时才采样。
