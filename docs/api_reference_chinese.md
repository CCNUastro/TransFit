# TransFit API 和参数参考

本文档说明 TransFit 稳定公开 Python 接口。README 和 tutorial 只展示最小用法；
API 参数含义、模型参数、结果字段和高级选项统一放在这里。

所有公开时间输入和输出都使用 **observer-frame days**。内部物理模型仍在
rest-frame time 中求解，并在 API 边界转换回 observer frame。

## 公开入口

| 类别 | 入口 |
|---|---|
| 数据容器 | `BolometricData`, `MultiBandData` |
| 模型查看 | `model_param_names(model)`, `param_template(model)` |
| 正向光变曲线 | `lightcurve_bol(...)`, `lightcurve_multiband(...)` |
| 插值预测 | `predict_bol(...)`, `predict_multiband(...)` |
| 拟合 | `fit_bol(...)`, `fit_multiband(...)` |
| 结果读写 | `save(res, path=None)`, `load(path, trusted=False)` |
| 画图 | `transfit.plot.fit_bol`, `transfit.plot.fit_multiband`, `transfit.plot.corner` |

## 模型名和参数

规范模型名包括 `nickel`、`magnetar`、`magnetar_ni` 和 `csm`。为了兼容旧脚本，
部分别名仍可使用，但新代码建议使用规范模型名。

### `nickel`

| 参数 | 含义和单位 |
|---|---|
| `M_ej` | 抛射物质量，M☉ |
| `v_ej` | 抛射物速度，10^9 cm s^-1 |
| `E_Th_in` | 初始热能，10^49 erg |
| `M_ni` | 镍质量，M☉ |
| `R_0` | 初始半径，R☉ |
| `f_ni` | Ni 混合区外边界的拉格朗日质量坐标，M(<x_Ni)/M_ej |
| `kappa` | optical opacity，cm^2 g^-1 |
| `kappa_gamma` | gamma-ray opacity，cm^2 g^-1 |
| `T_floor` | 温度下限，K |
| `delta` | BPL 内层密度指数，无量纲 |
| `n` | BPL 外层密度指数，无量纲 |

### `magnetar`

| 参数 | 含义和单位 |
|---|---|
| `M_ej` | 抛射物质量，M☉ |
| `v_ej` | 抛射物速度，10^9 cm s^-1 |
| `E_Th_in` | 初始热能，10^49 erg |
| `P_ms` | 磁星自转周期，ms |
| `B14` | 磁星偶极磁场，10^14 G |
| `f_mag` | 磁星加热混合位置，无量纲 |
| `R_0` | 初始半径，R☉ |
| `kappa` | optical opacity，cm^2 g^-1 |
| `kappa_gamma` | gamma-ray opacity，cm^2 g^-1 |
| `T_floor` | 温度下限，K |

### `magnetar_ni`

| 参数 | 含义和单位 |
|---|---|
| `M_ej` | 抛射物质量，M☉ |
| `v_ej` | 抛射物速度，10^9 cm s^-1 |
| `P_ms` | 磁星自转周期，ms |
| `B14` | 磁星偶极磁场，10^14 G |
| `f_mag` | 磁星加热混合位置，无量纲 |
| `M_ni` | 镍质量，M☉ |
| `f_ni` | 镍混合位置，无量纲 |
| `kappa` | optical opacity，cm^2 g^-1 |
| `kappa_gamma` | gamma-ray opacity，cm^2 g^-1 |
| `T_floor` | 温度下限，K |

### `csm`

| 参数 | 含义和单位 |
|---|---|
| `M_ej` | 抛射物质量，M☉ |
| `E_sn` | 爆炸能量，10^51 erg |
| `M_csm` | CSM 质量，M☉ |
| `R_csm_out` | CSM 外半径，R☉ |
| `kappa` | optical opacity，cm^2 g^-1 |
| `s` | CSM 密度幂律指数 |
| `eps_sh` | shock 辐射效率 |
| `T_floor` | 温度下限，K |

拟合接口会使用可选参数 `t_shift` 来平移模型时间轴。`t_shift` 被限制为
非负；如果不想拟合它，可以在 `fixed` 中设为 `0.0`。拟合中模型在以下
时间点计算：

```text
t_eval = t_obs + t_shift
```

因此 `t_shift > 0` 表示模型起点早于用户数据的观测零点。

对于 `magnetar` 和 `magnetar_ni`，`f_mag` 属于公开参数结构，但拟合时如果省略，
会默认固定为 `0.2`。如果需要拟合它，需要显式给先验，例如
`priors={"f_mag": (0.05, 0.5)}`。正向模型的 `params` 省略 `f_mag` 时也会使用
`0.2`。

## 数据容器

### `BolometricData`

```python
tf.BolometricData(t_days, y, yerr, mask=None)
```

| 字段 | 含义 |
|---|---|
| `t_days` | observer-frame days |
| `y` | bolometric luminosity，erg s^-1 |
| `yerr` | 与 `y` 同单位的一倍标准差误差 |
| `mask` | 可选布尔 mask；只有 mask 选中的点会进入拟合 |

未被 mask 排除的 luminosity 和误差必须为正且有限。

### `MultiBandData`

```python
tf.MultiBandData(t_days, band, y, yerr, mask=None)
```

| 字段 | 含义 |
|---|---|
| `t_days` | observer-frame days |
| `band` | 每个数据点对应的 band 标签 |
| `y` | 若 `y_kind="mag"` 则为星等；若 `y_kind="flux"` 则为 flux density |
| `yerr` | 与 `y` 同单位的一倍标准差误差 |
| `mask` | 可选布尔 mask；只有 mask 选中的点会进入拟合 |

## 正向计算和预测

Bolometric 正向计算：

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

为保持向后兼容，默认值是 `density_profile="uniform"`。使用
`"bpl"`/`"broken_power_law"` 选择破幂律，使用
`"exp"`/`"exponential"`/`"ia"` 选择有限半径的 Ia 型指数密度。

返回 `BolometricLC`，包含：

- `t_days`
- `Lbol`
- `Teff`
- `Rph`

多波段正向计算：

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

返回 `MultiBandLC`，包含：

- `t_days`
- `bands`
- `y[band]`

### 滤波器定义

`filters` 会把 `MultiBandData.band` 中的 band 标签映射到物理滤波器定义。
内置滤波器使用字符串 ID：

```python
filters = {
    "B": "johnson_cousins.B",
    "V": "johnson_cousins.V",
}
```

自定义单点滤波器推荐使用有效波长作为公开输入：

```python
filters = {
    "g": {"lambda_eff_A": 4770.0},
    "r": {"lambda_eff_nm": 623.1},
    "i": {"lambda_eff_um": 0.7625},
}
```

如果 `mag_system="vega"` 使用自定义滤波器，需要同时提供 Vega 零点：

```python
filters = {
    "B": {"lambda_eff_A": 4400.0, "vega_zero_point_jy": 4260.0},
}
```

`nu_eff_hz` 仍然保留兼容，但新的用户示例应优先使用 `lambda_eff_A`。完整
bandpass throughput 积分暂未实现。

`predict_bol` 和 `predict_multiband` 用于在用户给定的 observer-frame
时间点上计算模型值。`interp_fill` 可取 `"nan"`、`"raise"` 或 `"edge"`。
拟合接口中禁止 `"edge"`，避免在模型时间范围外静默使用边界值。

## 拟合接口

Bolometric 拟合：

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

多波段拟合：

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

`priors` 是参数名到先验范围的映射。线性均匀先验写作 `(lo, hi)`。
以 10 为底的 log-uniform 先验写作 `("log10", lo, hi)`，其中 `lo` 和
`hi` 是 log10 空间中的边界。

`fixed` 是参数名到固定值的映射。一般情况下，没有放在 `fixed` 里的模型参数会被采样，
其范围来自默认边界或 `priors` 中用户给出的边界。

有两个有意保留的例外。`fit_bol` 中的 `T_floor` 作为内部数值温度下限使用，
不进入 bolometric 拟合采样状态。`magnetar` 和 `magnetar_ni` 中的 `f_mag`
如果用户没有显式给先验，默认固定为 `0.2`；如果需要拟合它，需要给出
`priors={"f_mag": (...)}`，并且不要同时在 `fixed` 中固定它。

### `sigma_int`

`sigma_int` 是 likelihood nuisance parameter，不是物理模型参数。它可以通过
`fixed` 固定，也可以通过 `priors` 采样。

| 观测空间 | 含义 |
|---|---|
| `y_kind="mag"` | 额外星等 scatter |
| `y_kind="flux"` | 用 0.4 ln(10) sigma_int 转换成 fractional flux scatter |

## 关键字参数字典

### `sampler_kwargs`

`emcee` 和 `zeus` 常用键：

| 键 | 含义 |
|---|---|
| `nwalkers` | walker 数量 |
| `nsteps` | production chain 步数 |
| `burnin` | production 前的 burn-in 步数 |
| `thin` | thinning 因子 |
| `seed` | 随机种子 |
| `init` | 初始位置模式或数组 |
| `pool` | 用户传入的并行 pool |
| `progress` | 是否显示采样进度 |

`dynesty` 常用键：

| 键 | 含义 |
|---|---|
| `nlive` | live point 数量 |
| `sample` | dynesty sampling method |
| `bound` | dynesty bounding method |
| `dlogz` | 停止阈值 |
| `maxiter` | 最大迭代数 |
| `maxcall` | 最大 likelihood 调用数 |
| `seed` | 随机种子 |
| `progress` | 是否显示采样进度 |
| `nsamples` | 返回的 posterior sample 数 |
| `add_live` | 是否把 live points 加入 posterior |
| `pool` | 用户传入的并行 pool |
| `queue_size` | dynesty queue size |

### `model_kwargs`

拟合时传给模型计算的选项放在 `model_kwargs` 中。

| 键 | 含义 |
|---|---|
| `t_max_days` | observer-frame 模型计算时长，单位 days |
| `interp_fill` | 插值边界策略；拟合中不允许 `"edge"` |
| `solver_kwargs` | 高级数值网格选项 |

如果省略 `t_max_days`，TransFit 会自动选一个足够覆盖数据和 `t_shift`
允许范围的值。

### `solver_kwargs`

`solver_kwargs` 是高级数值网格接口。

| 键 | 默认值 | 含义 |
|---|---:|---|
| `Nx` | `100` | 空间/网格分辨率参数 |
| `Ny` | `1000` | 时间/网格分辨率参数 |

`Nx` 和 `Ny` 都必须是正整数。

对于 `nickel` 模型，`solver_kwargs` 还支持：

| 键 | 默认值 | 含义 |
|---|---|---|
| `density_profile` | `"uniform"` | 密度结构：`"uniform"`、`"bpl"`/`"broken_power_law"` 或 `"exp"`/`"exponential"`/`"ia"`。 |

对于 `csm` 模型，`solver_kwargs` 还支持：

| 键 | 默认值 | 含义 |
|---|---:|---|
| `photosphere_mode` | `"tau"` | `"tau"` 使用光学深度光球；`"outer"` 保留原来的外边界/温度下限处理。 |

默认的 CSM `"tau"` 模式把向外光学深度 `tau=2/3` 的位置作为扩散边界。
前向激波到达该位置以前，光球固定在扩散边界；随后光球跟随前向激波，离开 CSM
以前只改变辐射半径，光度仍由同一个含激波源的 Crank--Nicolson 扩散方程给出，
不会切换到单独的直接激波光度公式。激波离开 CSM 后关闭能量源，同一个求解器
进入无源同模膨胀冷却；冷却光度不会被人为幂律替代，辐射半径跟随 CSM 外边界
膨胀。时间网格会显式包含两个物理阶段切换时刻。

若 CSM 的总径向光学深度不超过 `2/3`，则超出当前扩散模型的适用范围。正向
计算会给出物理域错误；拟合会把该样本记为 `-inf`，不会终止采样器。
`"tau"` 多波段拟合中 `T_floor` 不活动并默认固定。如果需要原来的温度下限
处理并拟合 `T_floor`，使用：

```python
model_kwargs={
    "solver_kwargs": {
        "Nx": 100,
        "Ny": 1000,
        "photosphere_mode": "outer",
    }
}
```

`delta` 和 `n` 是 nickel 模型的物理参数，不属于求解器选项：

| 参数 | 默认值 | 默认先验范围 | 含义 |
|---|---:|---:|---|
| `delta` | `0.0` | `[0.0, 2.9]` | BPL 内层密度指数，物理要求 `0 <= delta < 3`。 |
| `n` | `10.0` | `[5.1, 14.0]` | BPL 外层密度指数，物理要求 `n > 5`。 |

正向计算中若省略，默认使用 `delta=0`、`n=10`。在 `fit_bol` 和
`fit_multiband` 中，即使选择 `density_profile="bpl"`，这两个参数默认也保持
固定。只有用户显式把某个参数写入 `priors`，该参数才进入采样；写入 `fixed`
则使用指定的固定值。uniform 和 exponential 模式不使用 `delta,n`，并拒绝
对它们进行采样。

BPL 使用拟合中的 `M_ej` 和 `v_ej` 归一化：在转折点 `x=1` 内侧
`rho/rho_t = x**(-delta)`，外侧为 `x**(-n)`。
扩散网格固定覆盖 `10^-3 <= x=v/v_t <= 3`；`v_max/v_t=3` 是内部常量，不能作为
公开拟合选项。指数轮廓固定覆盖 `10^-3 <= x=r/R_e=v/v_e <= 12`，其中
`R_e=R_0/12`。BPL 和指数轮廓采用 backward Euler，均匀密度采用
Crank--Nicolson。

三种轮廓都在初始外半径 `R_0` 截止。密度尺度和速度尺度由有限区间内的质量、
动能积分共同确定，因此计算区域内严格恢复输入的 `M_ej` 和
`E_K=0.5*M_ej*v_ej**2`，不再假设无穷远处还有抛射物质量。

`f_ni` 定义为 Ni 混合区外边界的拉格朗日质量坐标：
`f_ni=M(<x_Ni)/M_ej`。程序会根据 uniform、BPL 和指数密度轮廓分别反解
对应的半径/速度坐标 `x_Ni`，在该截止位置以内采用常数 Ni 质量分数，并按密度
加权的质量积分重新归一化，保证积分 Ni 质量仍为 `M_ni`。因此 `f_ni=0.8`
表示计算区域内80%的抛射物质量，不一定满足 `x_Ni/x_max=0.8`。

指数轮廓调用示例：

```python
tf.lightcurve_bol(
    model="nickel",
    params=params,
    z=0.001728,
    solver_kwargs={"density_profile": "ia"},
)
```

在 `fit_bol` 和 `fit_multiband` 中，exponential/Ia 轮廓不采样 `R_0`，默认固定为
白矮星半径 `R_0=0.01 R_sun`。用户可以用 `fixed={"R_0": value}` 覆盖默认值；
如果把 `R_0` 放进 `priors`，接口会直接报错。模型时间网格从爆炸时刻 `t=0`
开始，`R_0/v_max` 只作为膨胀时间尺度。

对于纯放射性 Ia 模型（`E_Th_in=0`），回归参数下把固定半径从 `0.01` 改为
`1 R_sun`，五天后的光度差异小于约 0.2%。若初始热能非零，最初几天可能对
半径非常敏感，因此在激波冷却问题中不能认为两者总是等价。

拟合时只在 `model_kwargs` 中选择 BPL，`delta,n` 默认仍固定：

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

若要拟合密度指数，必须显式给出
`priors={"delta": (0.0, 2.0), "n": (6.0, 12.0)}`；也可以只启用其中一个。
两者都是连续实数参数，物理限制分别为 `0 <= delta < 3` 和 `n > 5`。
完整的选择表、兼容行为和可复现图片见
[v0.2 更新日志](changelog_chinese.md#v02--2026-07-26)。

## SED 选项

默认多波段 SED 是 `BlackbodySED`。

```python
from transfit.modules.sed import BlackbodySED, CutoffBlackbodySED

sed = BlackbodySED()
sed = CutoffBlackbodySED(
    cutoff_wavelength_A=3000.0,
    uv_slope=2.0,
    min_factor=0.0,
)
```

`CutoffBlackbodySED` 会对短波端施加 cutoff：

```text
L_nu_cutoff = C(lambda_rest) * L_nu_blackbody
```

其中：

```text
C(lambda) = 1                                      for lambda >= lambda_cut
C(lambda) = max(f_min, (lambda/lambda_cut)^a)      for lambda < lambda_cut
```

| 符号 | API 参数 |
|---|---|
| `lambda_cut` | `cutoff_wavelength_A` |
| `a` | `uv_slope` |
| `f_min` | `min_factor` |

设置 `min_factor=0` 时就是纯 power-law cutoff。

## FitResult 字段

`fit_bol` 和 `fit_multiband` 返回 `FitResult`。

如果 `t_shift` 参与拟合采样，它会直接出现在 `res.param_names`，也会出现在
`res.best_params`、`res.best_params_raw`、`res.median_params` 和
`res.best_fit["params"]` 等参数字典里。

| 字段/属性 | 含义 |
|---|---|
| `res.best_params` | 四舍五入后的 best-fit 参数字典 |
| `res.best_params_raw` | 全精度 best-fit 参数字典 |
| `res.median_params` | posterior median 参数字典 |
| `res.best_fit` | 包含参数、误差、best log probability 和 best sample 的紧凑记录 |
| `res.best_index` | best posterior sample 的索引 |
| `res.best_log_prob` | best log posterior 值 |
| `res.best_sample` | `res.param_names` 顺序下的原始 best sample 向量 |
| `res.samples` | 展平后的 posterior samples |
| `res.log_prob` | 每个 sample 的 log posterior |
| `res.meta` | sampler、prior、model、SED 和 context 元数据 |

## 引用规则

所有模型都应引用 TransFit 论文。使用 `csm` 模型时，还应额外引用
TransFit-CSM 论文。BibTeX 和模型引用规则见
[model citation guide](https://github.com/YuHaoZhang01/TransFit/blob/main/docs/model_citations.md)。
