"""Experimental staged Ni-ejecta diffusion and interaction reservoir.

This is not the CHIPS shock solver. Ni light is calculated by NickelModel and
enters an approximate one-zone shell/CSM radiation reservoir. The thin-shell
trajectory is prescribed, and swept and unswept optical depths evolve with it.
"""
import numpy as np

from transfit.constants import C_LIGHT, DAY, M_SUN, PI, SIGMA_SB
from transfit.modules.interp import interp_fit
from .csm import (_build_params, _solve_shock_ode, _build_expansion_profile,
                  _rho_csm_of_x, _f_rho_ej, _integral_power_law,
                  DEFAULT_EJECTA_N, DEFAULT_EJECTA_DELTA)
from .nickel import NickelModel


class NickelCSMStagedModel:
    """Approximate sequential transport; theta order is shared with nickel_csm."""
    _warmup_theta = (5., 1., .05, .5, 1., 100., 10000., .1, .2, .03, 2., .5, 0., 5000.)
    _warmup_kwargs = {"Nx": 20, "Ny": 40, "t_max_days": 20.}

    def __init__(self, *, warmup=False):
        if warmup:
            self.warmup()

    def warmup(self, **kwargs):
        options = dict(self._warmup_kwargs)
        options.update(kwargs)
        self.calculate_light_curve(self._warmup_theta, **options)
        return self

    def calculate_light_curve(self, theta, Nx=100, Ny=1000, t_max_days=150.,
                              return_full=False, *, n=DEFAULT_EJECTA_N,
                              delta_inner=DEFAULT_EJECTA_DELTA,
                              shell_width_fraction=0.1, nickel_R0=10.,
                              rtol_ode=1.e-6, atol_ode=1.e-8,
                              shock_max_step=0.8, shock_ode_solver='numba',
                              shock_rk_dx=0.05):
        if len(theta) != 14:
            raise ValueError('Expected 14 nickel_csm parameters.')
        mej, esn, mni, fni, mcsm, rin, rout, kej, kcsm, kg, s, efs, ers, tfloor = map(float, theta)
        if not (0. <= mni <= mej and 0. < fni <= 1. and kej > 0. and kg > 0.
                and 0. <= efs <= 1. and 0. <= ers <= 1. and tfloor > 0.
                and 0. < shell_width_fraction <= 1. and nickel_R0 > 0.
                and np.isfinite(t_max_days) and t_max_days > 0. and int(Ny) >= 2):
            raise ValueError('Invalid nickel, opacity, efficiency, grid or shell parameters.')
        # NickelModel assumes a uniform-density ejecta and uniform Ni mass
        # fraction inside fni of its outer radius.
        if mni > mej * fni**3 * (1. + 1.e-12):
            raise ValueError('M_ni exceeds the uniform-density mass inside f_ni.')
        p = _build_params((mej, esn, mcsm, rout, rin, kcsm, s, n, delta_inner, efs, tfloor),
                          int(Nx), int(Ny), float(t_max_days), s, n, delta_inner, tfloor,
                          rtol_ode, atol_ode, shock_max_step, shock_ode_solver,
                          shock_rk_dx, 2, 0.)
        shock = _solve_shock_ode(p)
        sc = shock['scales']
        tmax = t_max_days * DAY
        # The shell power changes on a shorter time scale than the ejecta
        # diffusion solution, particularly near the density-profile break.
        times = np.linspace(0., tmax, 4 * int(Ny) + 1)
        events = [v for v in (sc['t_in'], shock['t_end_days'] * DAY) if 0. < v < tmax]
        # Resolve rapid shock power changes just after first contact; uniform
        # multi-month grids can otherwise miss most of this short transient.
        refine_span = min(DAY, max(tmax-sc['t_in'], 0.),
                          max(shock['t_end_days']*DAY-sc['t_in'], 0.))
        refined = sc['t_in'] + refine_span * np.geomspace(1.e-6, 1., 160)
        times = np.unique(np.r_[times, events, refined[refined < tmax]])
        # The shock solution starts at contact. Clip the dynamical evaluation
        # before contact while continuing Ni + CSM transport from explosion.
        y = np.maximum(times, sc['t_in']) / sc['t_d']
        expansion, _, active, xsh, zsh, wsh = _build_expansion_profile(y, shock, p, sc)
        before = times < sc['t_in']
        active = active & ~before
        xsh[before] = 1.
        zsh[before] = 0.
        wsh[before] = 0.
        vscale = np.sqrt(2. * esn * 1.e51 / (mej * M_SUN)) / 1.e9
        nt = (mej, vscale, 0., mni, nickel_R0, fni, kej, kg, tfloor)
        ni_t, ni_l, _, _ = NickelModel().calculate_light_curve(
            nt, Nx=int(Nx), Ny=int(Ny), t_max_days=float(t_max_days))
        if np.any(~np.isfinite(ni_l)) or np.any(ni_l < 0.):
            raise RuntimeError('Nickel ejecta diffusion produced invalid luminosity.')
        Lni = np.interp(times, np.r_[0., ni_t], np.r_[0., ni_l])
        Lfs = np.zeros_like(times)
        Lrs = np.zeros_like(times)
        tau_csm = np.zeros_like(times)
        tau_shell = np.zeros_like(times)
        tesc = np.zeros_like(times)
        Rsh = p['R_in'] * xsh
        Rsh[~active & ~before] = p['R_csm'] * expansion[~active & ~before]
        Rsh[before] = p['R_in']
        R_out = p['R_csm'] * expansion
        vsh = sc['v_max'] * wsh
        mass_scale = 4. * PI * sc['rho_0'] * p['R_in']**3
        Msh = np.maximum(zsh * mass_scale, 0.)
        for i, t in enumerate(times):
            if active[i]:
                radius = Rsh[i]
                rho_c = float(_rho_csm_of_x(xsh[i], p, sc))
                Lfs[i] = efs * 2. * PI * radius**2 * rho_c * vsh[i]**3
                rho_e = sc['rho_ej_tr'] * float(_f_rho_ej(xsh[i], t / sc['t_in'], p, sc))
                Lrs[i] = ers * 2. * PI * radius**2 * rho_e * max(radius / t - vsh[i], 0.)**3
            xx = np.clip(xsh[i], 1., p['x_max'])
            if xx < p['x_max']:
                column = sc['rho_0'] * p['R_in'] * _integral_power_law(xx, p['x_max'], -s)
                tau_csm[i] = kcsm * column
                mcsm_swept = mass_scale * _integral_power_law(1., xx, 2.-s) if xx > 1. else 0.
            else:
                mcsm_swept = p['M_csm']
            mcsm_swept = np.clip(mcsm_swept, 0., Msh[i])
            tau_shell[i] = (kcsm * mcsm_swept + kej * (Msh[i]-mcsm_swept)) / (4. * PI * Rsh[i]**2)
            width = max(R_out[i] - Rsh[i], 0.)
            # A one-zone random-walk timescale, not a resolved moving-grid PDE.
            tesc[i] = (width * (1. + 3. * tau_csm[i])
                       + shell_width_fraction * Rsh[i] * 3. * tau_shell[i]) / C_LIGHT
        source = Lni + Lfs + Lrs
        E = np.zeros_like(times)
        L = np.zeros_like(times)
        work = np.zeros_like(times)
        injected = np.zeros_like(times)
        escaped = np.zeros_like(times)
        adiabatic = np.zeros_like(times)
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            # Expansion losses apply in proportion to the shell's share of
            # the modelled diffusion delay; the unshocked CSM is stationary.
            shell_delay = 3. * tau_shell[i] * shell_width_fraction * Rsh[i] / C_LIGHT
            share = shell_delay / max(tesc[i], 1.e-300)
            ad_rate = share * vsh[i] / max(Rsh[i], 1.e-300)
            inv_escape = 1. / max(tesc[i], 1.e-30)
            E[i] = (E[i-1] + dt * source[i]) / (1. + dt * (inv_escape + ad_rate))
            L[i] = E[i] * inv_escape
            work[i] = E[i] * ad_rate
            injected[i] = injected[i-1] + dt * source[i]
            escaped[i] = escaped[i-1] + dt * L[i]
            adiabatic[i] = adiabatic[i-1] + dt * work[i]
        residual = E + escaped + adiabatic - injected
        Lsafe = np.maximum(L[1:], 1.e-300)
        rnom = R_out[1:]
        temperature = (Lsafe / (4.*PI*SIGMA_SB*rnom**2))**.25
        rph = np.where(temperature < tfloor, np.sqrt(Lsafe/(4.*PI*SIGMA_SB*tfloor**4)), rnom)
        teff = np.maximum(temperature, tfloor)
        if return_full:
            return dict(t_s=times[1:], t_day=times[1:]/DAY, L_bol=Lsafe, T_eff=teff,
                        R_ph=rph, R_sh=Rsh[1:], R_out=R_out[1:], v_sh=vsh[1:],
                        shock_active=active[1:], shock_end_day=shock['t_end_days'],
                        L_ni_ejecta=Lni[1:], L_fs=Lfs[1:], L_rs=Lrs[1:],
                        L_sh_heat=(Lfs+Lrs)[1:], tau_csm=tau_csm[1:], tau_shell=tau_shell[1:],
                        t_escape=tesc[1:], E_rad=E[1:], E_injected=injected[1:],
                        E_escaped=escaped[1:], E_ad=adiabatic[1:], energy_residual=residual[1:],
                        relative_energy_residual=(residual/np.maximum(injected, 1.e-300))[1:])
        return times[1:], Lsafe, teff, rph

    def calculate_dynamics(self, theta, **kwargs):
        kwargs.pop('return_full', None)
        r = self.calculate_light_curve(theta, return_full=True, **kwargs)
        return r['t_s'], r['v_sh'], r['L_sh_heat']

    def L_bol(self, t_obs, theta, z=0., **kwargs):
        t, L, _, _ = self.calculate_light_curve(theta, **kwargs)
        return interp_fit(t*(1.+float(z))/DAY, L, np.asarray(t_obs, float), yscale='log10', fill='edge')
