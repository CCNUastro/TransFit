# Nickel v0.2 发布检查表

## 生产物理

- 唯一的 bolometric 输运区域为 `q_min <= q <= q_ph(t)`，其中
  `tau(q_ph -> 1)=2/3`。
- 最外活动层是终止于 `q_ph` 的真实 cut cell；光球外的 Nickel 沉积不进入
  扩散矩阵。
- 输出严格满足 `Lbol=Lphotospheric+Ldirect`。
- 物理光球采用 `Rph=Rout*q_ph` 和
  `Tph=(Lphotospheric/(4*pi*sigma*Rph**2))**0.25`，不加温度地板。
- 完全光学薄后 `Lphotospheric=0`、`Ldirect=Lbol=deposited heating`、
  `photosphere_valid=False`，且 `Rph/Tph=NaN`。
- 使用固定灰 `kappa`；本版本不包含复合 opacity。
- 时间积分采用固定嵌套二次网格 `t_i=t_max*(i/Ny)^2`，每步扩散系数、源项和
  swept-energy 都使用各自的局部步长。

## 多波段

- 只保留完整 `Lbol`、`Rhom=R_0+v_max*t` 与 `T_floor` 的同模膨胀黑体。
- `T_floor` 默认固定 4500 K；只有 `fit_multiband` 可通过
  `priors={"T_floor": (3000, 10000)}` 采样。
- 不再提供 `emission_mode`、`T_neb` 或单独的星云 SED；该映射不改变
  bolometric 输运。

## 自动验证

运行：

```bash
python scripts/validate_nickel_photospheric_release.py
pytest -q
```

科学 gate 检查分量闭合、`tau=2/3`、Stefan--Boltzmann、完全光学薄尾部、
空间/时间收敛和进入 `T_floor` 后的 `Fnu/Lbol`。结果写入
`result/tables/nickel-photosphere-homologous-release-metrics.json`。
目标 `Nx=100, Ny=1000` 网格要求 5 d 后空间误差和 10 d 后时间误差均小于
0.5%；1.5 d 另以相对峰值的绝对差小于 `5e-4` 放行。更早阶段在局部光度几乎
为零时的一阶 backward-Euler 相对差异仍保留为独立 diagnostic，不用很小的
分母制造虚假的全局不收敛结论。

正式验证参数使用 `M_ej=3 M_sun`、`v_ej=1e9 cm/s`、`M_ni=0.08 M_sun`、
`E_Th_in=0`、`f_ni=0.8`、`kappa=0.12`、`Nx=100`、`Ny=1000`。
