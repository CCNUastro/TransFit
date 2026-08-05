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

- Uniform 使用历史同模半径；BPL/Ia 把 `Ldirect` 换算成 4500 K 等效面积，
  并与真实 `tau=2/3` 光球面积相加。
- 高温时使用相应的同模或合并半径；其余时刻固定
  地板温度并由完整 `Lbol` 反算半径。分支逐时刻可逆，不锁定历史状态。
- 完全光学薄后合并半径自然退化为 `Ldirect` 的地板温度等效半径。
- `T_floor` 默认固定 4500 K；只有 `fit_multiband` 可通过
  `priors={"T_floor": (3000, 10000)}` 采样。
- 不提供 `emission_mode`、`T_neb`、
  单独的 `Ldirect` 或星云 SED；该映射不改变 bolometric 输运。

## 自动验证

运行：

```bash
python scripts/validate_nickel_photospheric_release.py
pytest -q
```

科学 gate 检查 Uniform 历史回归，以及 BPL/Ia 的分量闭合、`tau=2/3`、
Stefan--Boltzmann、完全光学薄尾部、
空间/时间收敛、物理光球黑体关系和完全光学薄后 `Fnu/Lbol`。结果写入
`result/tables/nickel-photosphere-floor-release-metrics.json`。
目标 `Nx=100, Ny=1000` 网格要求 5 d 后空间误差和 10 d 后时间误差均小于
0.5%；1.5 d 另以相对峰值的绝对差小于 `5e-4` 放行。更早阶段在局部光度几乎
为零时的一阶 backward-Euler 相对差异仍保留为独立 diagnostic，不用很小的
分母制造虚假的全局不收敛结论。

正式验证参数使用 `M_ej=3 M_sun`、`v_ej=1e9 cm/s`、`M_ni=0.08 M_sun`、
`E_Th_in=0`、`f_ni=0.8`、`kappa=0.12`、`Nx=100`、`Ny=1000`。
