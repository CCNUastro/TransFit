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
- Nickel 根据归一化后的密度名称自动选择辐射策略：Uniform 保留历史外边界
  求解和同模膨胀黑体，BPL/Ia 使用移动 $\tau=2/3$ 光球及光球外直接逃逸。
- BPL/Ia 多波段把 `Ldirect` 转换为 $T_{\rm floor}$ 下的等效发射面积，再由
  完整 `Lbol` 归一化黑体，避免把光球外光度压缩到后退的物理光球上。
- `R_0` 表示抛射物有限的初始外半径。
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

抛射物作同模膨胀。定义

```math
t_{\rm h}=t+\frac{R_0}{v_{\max}},\qquad
r=v t_{\rm h},\qquad
R_{\rm out}=v_{\max}t_{\rm h}=R_0+v_{\max}t .
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

BPL 在 $v_{\max}=3v_t$ 处截断，Ia/exponential 在
$v_{\max}=12v_e$ 处截断。三种密度的归一化因子均随同模膨胀按下式下降：

```math
\rho_{\rm u}(t)\propto t_{\rm h}^{-3},\qquad
\rho_t(t)\propto t_{\rm h}^{-3},\qquad
\rho_e(t)\propto t_{\rm h}^{-3}.
```

密度归一化和 $v_{\max}$ 由抛射物质量与动能共同确定：

```math
M_{\rm ej}=4\pi t_{\rm h}^{3}
\int_0^{v_{\max}}\rho(v,t)v^2\,dv,
\qquad
E_K=2\pi t_{\rm h}^{3}
\int_0^{v_{\max}}\rho(v,t)v^4\,dv
=\frac{1}{2}M_{\rm ej}v_{\rm ej}^2 .
```

因此输入的 $v_{\rm ej}$ 是由总动能定义的特征速度，而不是所有密度结构
共用的固定外边界速度。由物理密度直接积分得到向外光深：

```math
\tau(v,t)=\int_{r(v,t)}^{R_{\rm out}(t)}\kappa\rho(r',t)\,dr'
=\kappa t_{\rm h}\int_v^{v_{\max}}\rho(v',t)\,dv' .
```

不同密度结构自动选择不同的辐射边界。Uniform 保留原始外边界扩散光度：

```math
L_{\rm bol}=L_{\rm out},\qquad L_{\rm direct}=0.
```

该历史路径保留原始 gamma 处理：Ni 项完全俘获，只对 Co 项施加 leakage。

BPL 和 Ia/exponential 的物理光球由

```math
\tau(v_{\rm ph},t)=\frac{2}{3}
```

确定。光球内沉积参与扩散，光球外沉积直接逃逸，因此

```math
L_{\rm bol}=L_{\rm photospheric}+L_{\rm direct},\qquad
R_{\rm ph}=v_{\rm ph}t_{\rm h},\qquad
T_{\rm ph}=\left(\frac{L_{\rm photospheric}}
{4\pi\sigma R_{\rm ph}^2}\right)^{1/4} .
```

对于 BPL 和 Ia/exponential，总光深低于 $2/3$ 后，`photosphere_valid=False`、
`Lphotospheric=0`、`Lbol=Ldirect`，并令物理 `Rph/Teff=NaN`。

### Nickel 多波段

Uniform 保留原始同模膨胀黑体：

```math
R_{\rm hom}=R_0+v_{\max}t,\qquad
T_{\rm try}=\left(\frac{L_{\rm bol}}
{4\pi\sigma R_{\rm hom}^2}\right)^{1/4} .
```

BPL 和 Ia/exponential 先把 `Ldirect` 换算成地板温度等效半径：

```math
R_{\rm direct}=\left(\frac{L_{\rm direct}}
{4\pi\sigma T_{\rm floor}^4}\right)^{1/2},\qquad
R_{\rm try}=\left(R_{\rm ph}^2+R_{\rm direct}^2\right)^{1/2},
```

再由完整 `Lbol` 确定温度：

```math
T_{\rm try}=\left(\frac{L_{\rm bol}}
{4\pi\sigma R_{\rm try}^2}\right)^{1/4} .
```

对于物理光球有效且 $T_{\rm try}>T_{\rm floor}$ 的高温节点，取

```math
T_{\rm BB}=T_{\rm try},\qquad R_{\rm BB}=R_*.
```

其他时刻取

```math
T_{\rm BB}=T_{\rm floor},\qquad
R_{\rm BB}=\sqrt{\frac{L_{\rm bol}}
{4\pi\sigma T_{\rm floor}^4}}.
```

相应黑体频谱为

```math
L_\nu=4\pi^2R_{\rm BB}^2B_\nu(T_{\rm BB}) .
```

其中

```math
R_*=R_{\rm hom}\quad(\mathrm{Uniform}),\qquad
R_*=R_{\rm try}\quad(\mathrm{BPL/Ia}).
```

完全光学薄后自然有 $R_{\rm try}=R_{\rm direct}$ ；判断逐时刻进行，低温阶段
由完整 `Lbol` 反算地板半径，不提供额外星云 SED 分量。

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

`fit_bol` 不包含 `T_floor`；`fit_multiband` 默认固定 `T_floor=4500 K`，只有
显式放入 `priors` 时才采样。

输入别名会在求解前统一转换：`bpl` 等价于 `broken_power_law`，`exp` 和
`ia` 等价于 `exponential`；别名与完整名称给出完全相同的结果。
