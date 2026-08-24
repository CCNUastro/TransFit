# TransFit 更新日志

<p align="right">
  <strong>Language:</strong> <a href="changelog.md">English</a> | 简体中文
</p>

## v0.3

### 多波段拟合支持 upper limit

- `MultiBandData` 新增逐行对应的 `is_upper_limit` 和
  `upper_limit_nsigma` 字段。
- detection 保持原有 Gaussian likelihood；非探测数据在 flux 空间使用单边
  Gaussian CDF likelihood：

```math
\ln \mathcal{L}_{\rm UL}
=
\ln \Phi\!\left(\frac{F_{\rm lim}-F_{\rm model}}{\sigma_F}\right).
```

- 如果论文说明 upper limit 是 `3σ`、`5σ` 或其他显著度，用户只需把对应
  数值填入 `upper_limit_nsigma`。如果论文只给出 limit 而没有说明显著度，
  可以不填写该字段，TransFit 会使用默认的 `5σ` 设置。
- `sigma_int` 仍然只作用于 detection。多波段拟合图用向下三角表示 upper
  limit。
- `fit_multiband()` 不新增参数；所有 upper-limit 信息都与对应观测行一起存放
  在 `MultiBandData` 中。

### 公开接口与使用方法

```python
tf.MultiBandData(
    t_days,
    band,
    y,
    yerr,
    mask=None,
    is_upper_limit=None,
    upper_limit_nsigma=None,
)
```

每个 upper-limit 行都把论文给出的 limiting magnitude 或 limiting flux 放在
`y` 中。误差信息按照下表填写：

| 已知信息 | `yerr` | `upper_limit_nsigma` |
|---|---:|---:|
| 已知一倍标准差噪声 | 正数 | `np.nan` |
| 论文给出 `3σ` 或 `5σ` limit | `np.nan` | `3.0` 或 `5.0` |
| 只有 limit | `np.nan` | `np.nan`，默认按 `5σ` |

detection 行必须提供有限且为正的 `yerr`，其 `upper_limit_nsigma` 应为
`np.nan`。同一个 upper-limit 行同时填写 `yerr` 和
`upper_limit_nsigma` 会抛出 `ValueError`。

下面的例子包含 `5σ` 和 `3σ` magnitude limits：

```python
import numpy as np
import transfit as tf

data = tf.MultiBandData(
    t_days=np.array([-8, -4, 0, 5, 12, -6, 1, 8], dtype=float),
    band=np.array(["B", "B", "B", "B", "B", "V", "V", "V"]),
    y=np.array([21.7, 21.5, 20.4, 18.8, 19.5, 21.2, 20.1, 19.0]),
    yerr=np.array([
        np.nan, np.nan,
        0.12, 0.08, 0.10,
        np.nan,
        0.10, 0.08,
    ]),
    is_upper_limit=np.array([
        True, True,
        False, False, False,
        True,
        False, False,
    ]),
    upper_limit_nsigma=np.array([
        5.0, 3.0,
        np.nan, np.nan, np.nan,
        5.0,
        np.nan, np.nan,
    ]),
)

result = tf.fit_multiband(
    data=data,
    model="nickel",
    z=0.01,
    filters=filters,
    y_kind="mag",
    priors=priors,
    fixed=fixed,
)
```

如果全部 upper limit 都具有相同显著度，也可以传入标量；该数值只应用到
`is_upper_limit=True` 的数据行：

```python
data = tf.MultiBandData(
    t_days=t_days,
    band=band,
    y=y,
    yerr=yerr,
    is_upper_limit=is_upper_limit,
    upper_limit_nsigma=5.0,
)
```

拟合结果的 metadata 会分别记录 detection 数量、使用实际误差的 limit 数量、
使用显式 `nσ` 的 limit 数量，以及使用默认 `5σ` 的 limit 数量。

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
- Nickel 根据归一化后的密度名称自动选择辐射策略：Uniform 保留历史外边界
  求解和同模膨胀黑体，BPL/Ia 使用移动 $\tau=2/3$ 光球及光球外直接逃逸。
- BPL/Ia 多波段在 flux 空间相加由 `Lphotospheric` 归一化的稀释物理光球
  黑体，以及由 `Ldirect` 归一化的地板温度连续谱，避免把光球外光度压缩到
  后退的物理光球上。
- `R_0` 表示前身星半径，也是爆炸时同模膨胀抛射物的初始外半径。
- `f_ni` 表示 Nickel 混合到的拉格朗日质量坐标；当
  `M_ni > f_ni*M_ej` 时，模型会拒绝该组非物理参数。
- BPL 拟合默认固定 `delta=0`、`n=10`。只有显式放入 `priors` 的参数才参与
  拟合。
- Exponential/Ia 拟合默认固定 `R_0=0.01 R_sun`。

### 密度轮廓

![Uniform、BPL 和指数密度轮廓](assets/changelog/v0.2/nickel-density-profiles.png)

### 热光变曲线

![三种密度轮廓对应的热光变曲线](assets/changelog/v0.2/nickel-density-profile-lightcurves.png)

### Nickel 密度与光球计算

Nickel 抛射物从前身星半径 `R_0` 开始同模膨胀：

```math
R_{\rm out}(t)=R_0+v_{\max}t,
\qquad
\rho(v,t)=\rho(v,0)\left(\frac{R_0}{R_{\rm out}(t)}\right)^3 .
```

模型采用以下三种物理密度结构。Uniform 为

```math
\rho_{\rm Uniform}(v,t)=\rho_{\rm u}(t),
\qquad 0\le v\le v_{\max}.
```

BPL 的内区和外区分别为

```math
\rho_{\rm BPL}(v,t)=\rho_t(t)(v/v_t)^{-\delta},
\qquad 0\le v<v_t,
```

```math
\rho_{\rm BPL}(v,t)=\rho_t(t)(v/v_t)^{-n},
\qquad v_t\le v\le v_{\max}.
```

Ia/exponential 为

```math
\rho_{\rm Ia}(v,t)=\rho_e(t)\exp(-v/v_e),
\qquad 0\le v\le v_{\max}.
```

BPL 在 `v_max=3 v_t` 处截断，Ia/exponential 在 `v_max=12 v_e` 处截断。
向外光深直接由物理密度积分：

```math
\tau(v,t)=\int_{r(v,t)}^{R_{\rm out}(t)}\kappa\rho(r',t)\,dr'
```

不同密度结构自动选择不同的辐射边界。Uniform 使用外边界扩散光度，BPL/Ia
使用物理光球和光球外直接逃逸：

```math
L_{\rm bol}=L_{\rm out}\quad(\mathrm{Uniform}),\qquad
L_{\rm bol}=L_{\rm photospheric}+L_{\rm direct}
\quad(\mathrm{BPL/Ia}).
```

BPL 和 Ia/exponential 的物理光球由

```math
\tau(v_{\rm ph},t)=\frac{2}{3}
```

确定。光球半径和光球温度为

```math
R_{\rm ph}=R_{\rm out}\frac{v_{\rm ph}}{v_{\max}},\qquad
T_{\rm ph}=\left(\frac{L_{\rm photospheric}}
{4\pi\sigma R_{\rm ph}^2}\right)^{1/4} .
```

### Nickel 多波段

Uniform 保留原始同模膨胀黑体：

```math
R_{\rm hom}=R_0+v_{\max}t,\qquad
T_{\rm try}=\left(\frac{L_{\rm bol}}
{4\pi\sigma R_{\rm hom}^2}\right)^{1/4} .
```

BPL 和 Ia/exponential 先由物理光球计算有效温度和逐时刻颜色温度：

```math
T_{\rm eff,ph}=\left(\frac{L_{\rm photospheric}}
{4\pi\sigma R_{\rm ph}^2}\right)^{1/4},\qquad
T_{\rm col,ph}=\max(T_{\rm eff,ph},T_{\rm floor}).
```

光球稀释因子为

```math
W_{\rm ph}=\left(\frac{T_{\rm eff,ph}}{T_{\rm col,ph}}\right)^4,
```

两个连续谱分别为

```math
L_\nu^{\rm ph}=4\pi^2R_{\rm ph}^2W_{\rm ph}B_\nu(T_{\rm col,ph}),
```

```math
L_\nu^{\rm direct}=L_{\rm direct}
\frac{\pi B_\nu(T_{\rm floor})}{\sigma T_{\rm floor}^4},\qquad
L_\nu=L_\nu^{\rm ph}+L_\nu^{\rm direct}.
```

两个分量分别恢复各自的 bolometric 光度，其和严格恢复 `Lbol`。完全光学薄后
光球分量为零，只保留光球外连续谱。

### API 调用

```python
# 可选："uniform"、"bpl" / "broken_power_law"、"ia" / "exponential"
# 默认 uniform 使用历史外边界；bpl/ia 使用当前 tau=2/3 光球
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

`fit_bol` 不包含 `T_floor`；在 `fit_multiband` 中，`T_floor` 现在遵循普通拟合
参数规则：如果没有放入 `fixed`，就使用默认范围 1000--10000 K 参与采样；也可以
通过 `priors` 指定采样范围。若要固定为 4500 K 等具体值，请放入 `fixed`。

输入别名会在求解前统一转换：`bpl` 等价于 `broken_power_law`，`exp` 和
`ia` 等价于 `exponential`；别名与完整名称给出完全相同的结果。
