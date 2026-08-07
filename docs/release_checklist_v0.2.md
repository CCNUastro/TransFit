# Nickel v0.2 发布检查表

## 生产物理

- Uniform 使用历史固定外边界 Crank--Nicolson 求解，返回
  `Lphotospheric=Lbol`、`Ldirect=0`。
- BPL 与 Ia/exponential 的输运区域为 `q_min <= q <= q_ph(t)`，其中
  `tau(q_ph -> 1)=2/3`；最外活动层是真实 cut cell。
- BPL/Ia 输出严格满足 `Lbol=Lphotospheric+Ldirect`。
- BPL/Ia 物理光球采用 `Rph=Rout*q_ph` 和
  `Tph=(Lphotospheric/(4*pi*sigma*Rph**2))**0.25`，不加温度地板。
- BPL/Ia 完全光学薄后 `Lphotospheric=0`、`Ldirect=Lbol=deposited heating`、
  `photosphere_valid=False`，且 `Rph/Tph=NaN`。
- 使用固定灰 `kappa`；本版本不包含复合 opacity。
- Uniform 保留历史线性时间网格；BPL/Ia 使用嵌套二次时间网格。

## 多波段

- Uniform 使用历史同模半径。BPL/Ia 使用由 `Lphotospheric` 归一化的稀释
  物理光球黑体，以及由 `Ldirect` 归一化的 4500 K 光球外连续谱。
- BPL/Ia 光球颜色温度逐时刻取 `max(Tph, T_floor)`，并通过稀释因子保持
  光球频谱积分等于 `Lphotospheric`；不锁定历史状态。
- 两个分量在 flux 空间相加后统一施加消光，频率积分严格恢复 `Lbol`。
- 完全光学薄后光球分量为零，只保留由 `Ldirect=Lbol` 归一化的连续谱。
- `T_floor` 在 `fit_multiband` 中按普通模型参数参与采样；默认范围为
  1000--10000 K。用户可以通过 `priors={"T_floor": (3000, 10000)}` 指定范围，
  或通过 `fixed` 固定具体值。
  `fit_bol` 不包含 `T_floor`。
- 不提供 `emission_mode` 或 `T_neb`；光球外连续谱使用同一个 `T_floor`，
  不计算星云发射线，也不改变 bolometric 输运。BPL/Ia 仅支持标准
  `BlackbodySED`。

## 自动验证

运行：

```bash
python scripts/validate_nickel_photospheric_release.py
pytest -q
```

科学 gate 检查 Uniform 历史回归，以及 BPL/Ia 的分量闭合、`tau=2/3`、
Stefan--Boltzmann、完全光学薄尾部、
空间/时间收敛、双分量 bolometric 闭合和完全光学薄后 `Fnu/Lbol`。结果写入
`result/tables/nickel-photosphere-floor-release-metrics.json`。
目标 `Nx=100, Ny=1000` 网格要求 5 d 后空间误差和 10 d 后时间误差均小于
0.5%；1.5 d 另以相对峰值的绝对差小于 `5e-4` 放行。更早阶段在局部光度几乎
为零时的一阶 backward-Euler 相对差异仍保留为独立 diagnostic，不用很小的
分母制造虚假的全局不收敛结论。

正式验证参数使用 `M_ej=3 M_sun`、`v_ej=1e9 cm/s`、`M_ni=0.08 M_sun`、
`E_Th_in=0`、`f_ni=0.8`、`kappa=0.12`、`Nx=100`、`Ny=1000`。
