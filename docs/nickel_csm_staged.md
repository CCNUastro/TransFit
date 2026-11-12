# Experimental staged Ni + CSM model

`nickel_csm_staged` uses the existing uniform-density nickel ejecta solver to
calculate the luminosity entering an approximate shell/CSM radiation reservoir.
Prescribed thin-shell dynamics supplies forward- and reverse-shock heating.
The model has a single-zone escape time, not a moving-grid transfer solver or
the steady-shock structure solver of CHIPS. It is an experimental approximation
and requires comparison with radiation-hydrodynamic calculations before use
for scientific inference.

```python
import transfit as tf

params = dict(M_ej=2., E_sn=1., M_ni=0.1, f_ni=0.6, M_csm=0.01,
              R_csm_in=100., R_csm_out=4000., kappa_ej=0.1,
              kappa_csm=0.2, kappa_gamma=0.03, s=2., eps_fs=0.5,
              eps_rs=0., T_floor=5000.)
lc = tf.lightcurve_bol(model="nickel_csm_staged", params=params, z=0.,
                       solver_kwargs={"Nx": 160, "Ny": 2400})
```

Masses are solar masses, radii solar radii, energy in 10^51 erg, opacities
cm^2/g, and temperature K. The nickel model assumes uniform-density ejecta
and uniform nickel abundance inside radius fraction `f_ni`, so
`M_ni <= M_ej * f_ni**3`. The ejecta initial radius is 10 solar radii by
default. The escape time depends on the swept shell width, 0.1 times the shock
radius by default; low-level calls can adjust `shell_width_fraction` and
`nickel_R0`. The public API accepts only `Nx` and `Ny` in `solver_kwargs`.

After the forward shock reaches the CSM outer edge, both shock heating terms
are switched off, while nickel luminosity continues to feed the reservoir and
stored radiation escapes. The ejecta solver does not account for gamma-ray
absorption in the CSM or for nickel swept by the reverse shock. Both shock
dynamics and ejecta diffusion are prescribed rather than radiation-hydrodynamic.
The resulting post-breakout trough can depend strongly on these approximations.

The incident nickel-flux idea follows [Takei et al. (2023), section II.3.3](https://arxiv.org/html/2308.10785).
