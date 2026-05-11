# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TransFit is a supernova light-curve fitting framework. It provides forward models for nickel-56 powered, magnetar-powered, and combined SNe, and fits observed bolometric or multi-band photometry using Bayesian MCMC (emcee, zeus) or nested sampling (dynesty). The physics core solves radiative diffusion on a Lagrangian mass grid with Numba JIT acceleration.

## Build & Install

```bash
pip install -e .          # Core only (no samplers)
pip install -e .[all]     # Core + all samplers + corner plotting
pip install -e .[fit]     # Core + all samplers (no corner)
pip install -e .[sbi]     # Core + sbi + torch for neural posterior estimation
pip install -e .[emcee]   # Single sampler
```

Python >=3.10 required. Core deps: numpy, scipy, numba, astropy, pandas, matplotlib. Sampler deps (emcee, zeus-mcmc, dynesty) and SBI deps (sbi, torch) are lazily imported — `import transfit` works without them.

## Testing

```bash
pytest tests/                                            # All tests
pytest tests/test_multiband_redesign.py                  # Core + MCMC tests (19)
pytest tests/test_sbi.py                                 # SBI tests (skipped if sbi not installed)
pytest tests/test_multiband_redesign.py -k "test_nickel"  # Subset by name
```

No CI is configured. Sampler-dependent tests use `pytest.importorskip` to skip gracefully when a backend is missing. SBI tests similarly skip when `sbi` or `torch` is not installed.

There is no linting or formatting enforcement (no ruff, black, mypy config).

## Architecture

### Package Layout

```
transfit/
  api.py              -- Public API: forward models + fitting functions
  data.py             -- BolometricData / MultiBandData frozen dataclasses
  model_registry.py   -- Model name canonicalization + engine caching
  constants.py        -- CGS physical constants

  models/             -- Physics engines (Numba-JIT radiative diffusion solvers)
    nickel.py           9 params: M_ej, v_ej, E_Th_in, M_Ni, R_0, x_Ni, kappa, kappa_gamma, T_floor
    magnetar.py         9 params: M_ej, v_ej, E_Th_in, P_ms, B14, R_0, kappa, kappa_gamma, T_floor
    magnetar_ni.py      8 params: M_ej, v_ej, P_ms, B14, M_Ni, kappa, kappa_gamma, T_floor

  priors/             -- Per-model parameter bounds (uniform / mixed linear+log10)
  samplers/           -- MCMC backends: emcee, zeus, dynesty (+ FitResult dataclass)
  sbi/                -- Simulation-Based Inference (neural posterior estimation)
    __init__.py         train_sbi / infer_sbi top-level API
    prior.py            TransFitPrior: wraps MixedBoundsPrior as torch Distribution
    simulator.py        make_bolometric_simulator / make_multiband_simulator: theta -> x callables
    embedding.py        SetSummaryNet (DeepSet, auto-mask via validity column) + MLPEmbeddingNet
    training.py         generate_training_data with caching + parallel execution
    posterior.py        SBIPosterior dataclass: sample, log_prob, map_estimate, median
    diagnostics.py      simulation_based_calibration + posterior_predictive_check
    io.py               save_posterior / load_posterior (.pt files, requires trusted=True)
    README.md           SBI subpackage documentation
  modules/
    sed/blackbody.py   -- Planck SED -> Fnu
    filters/           -- FilterProfile dataclass + built-in presets (Johnson, SDSS, ZTF)
    extinction/        -- CCM89 / O'Donnell94 dust extinction (host + MW components)
    photometry.py      -- Multi-band observer-frame output pipeline
    magnitudes.py      -- AB/Vega magnitude conversion
    interp.py          -- 1D interpolation with configurable fill strategies
    plot.py            -- Fit visualization + corner plots (lazy-loaded)
    io.py              -- Save/load FitResult as compressed NPZ
```

### Data Flow (Multi-band Forward Model)

```
params + z + filters + extinction
  -> api._context_from_forward_inputs()   -- resolve distance, filters, extinction
  -> model_registry -> engine.calculate_light_curve(theta)
      -> Numba-JIT solver: (t_rest, Lbol, Teff, Rph)
  -> BlackbodySED.fnu(nu_obs, Teff, Rph, DL, z)
      -> Fnu grid (N_bands x N_times)
  -> apply_extinction_to_fnu_grid()
  -> fnu_grid_to_observation_output()     -> magnitudes or flux
```

### Data Flow (Fitting)

```
fit_bol() / fit_multiband()
  -> build_bounds(model, priors)          -> (param_names, bounds)
  -> _split_sampling(names, bounds, fixed) -> (free_names, free_bounds, fixed_dict)
  -> MixedBoundsPrior (linear + log10-space bounds)
  -> lnprob(theta):
       prior check -> physical constraints (M_Ni <= M_ej)
       -> predict_bol/predict_multiband()
       -> gaussian_lnlike(y_obs, y_model, y_err)
  -> emcee / zeus / dynesty sampler
  -> FitResult(model, samples, log_prob, ...)
```

### Data Flow (SBI / Neural Posterior Estimation)

```
tf.sbi.train_sbi(model, priors, cadence_templates, ...)
  -> build MixedBoundsPrior -> TransFitPrior (torch Distribution)
  -> generate cadence templates (random or explicit)
  -> for each template: make_simulator -> generate_training_data
       -> encode_batch: observations -> padded (batch, max_n_obs, feature_dim) + mask
  -> SNPE(MAF + SetSummaryNet embedding) trains on (theta, x_encoded)
  -> SBIPosterior(model, param_names, posterior, embedding_net, ...)

tf.sbi.infer_sbi(posterior, y_obs, t_days, band)
  -> SBIPosterior._encode_observation(y_obs, t_days, band) -> embedding
  -> posterior.sample(n, x=embedding) -> samples
```

Key difference from MCMC: SBI trains an amortized neural density estimator on simulated data. After training, inference on any new observation is instant (no sampling loop). Variable-length cadences are handled by padding + masking through the DeepSet embedding.

## Key Design Decisions

- **All public time inputs/outputs are observer-frame days.** Internal model runs in rest-frame, transformed by `(1+z)`.
- **Multi-band pipeline order is fixed**: SED -> Fnu -> extinction -> magnitudes. Never reorder.
- **Model name canonicalization**: aliases like "ni", "mag", "magni" all resolve to canonical names ("nickel", "magnetar", "magnetar_ni"). Removed aliases ("sc_ni", "sc_magnetar") raise ValueError.
- **Backward-compatible theta**: Legacy 7-element parameter vectors are automatically expanded to canonical form.
- **Engine instances are cached** in `_ENGINE_CACHE` dict keyed by `(model_name, Nx, Ny)`.
- **All data containers are frozen dataclasses** (BolometricData, MultiBandData, FitResult, FilterProfile, etc.).
- **FitResult.save()** serializes to compressed NPZ with JSON metadata; `load()` requires `trusted=True` to deserialize non-standard types.
- **SBI module is lazily loaded** via `__getattr__` — `import transfit; transfit.sbi` works but doesn't import torch/sbi until accessed.
- **SBIPosterior.save/load** uses PyTorch `.pt` format; `load()` also requires `trusted=True`.
- **Bolometric simulator output is in log10(L)** space for SBI (more uniform scale for neural net training). Multiband simulator output stays in original units (mag/flux).
- **SetSummaryNet uses a validity indicator column** — the last column of the input tensor is 1.0 for real data and 0.0 for padding. The mask is automatically inferred from this column, allowing sbi's internal pipeline to call `embedding_net(x)` without a separate mask argument.

## Public API Surface

All accessed via `import transfit as tf`:

- `tf.BolometricData`, `tf.MultiBandData` — data containers
- `tf.lightcurve_bol()`, `tf.lightcurve_multiband()` — generate light curves
- `tf.predict_bol()`, `tf.predict_multiband()` — evaluate at observed times
- `tf.fit_bol()`, `tf.fit_multiband()` — Bayesian fitting (MCMC / nested sampling)
- `tf.model_param_names()`, `tf.param_template()` — introspection
- `tf.save()`, `tf.load()` — I/O
- `tf.plot.fit()`, `tf.plot.corner()` — visualization
- `tf.sbi.train_sbi()`, `tf.sbi.infer_sbi()` — amortized neural posterior estimation
- `tf.sbi.SBIPosterior` — trained SBI posterior (`.sample()`, `.map_estimate()`, `.median()`)
- `tf.sbi.save_posterior()`, `tf.sbi.load_posterior()` — SBI model I/O (.pt)

## Conventions

- Chinese is used in some code comments and in `readme_zh.md`, `tutorial_zh.ipynb`.
- The design document `docs/multiband_photometry_design.md` is the authoritative reference for the photometry subsystem.
- Example data lives in `examples/data/` (sn1993j bolometric, sn2007gr multi-band CSV).
