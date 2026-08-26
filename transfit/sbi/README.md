# transfit.sbi — Simulation-Based Inference

Amortized neural posterior estimation for supernova light-curve parameters using the [sbi](https://sbi-dev.github.io/sbi/) library.

Unlike the MCMC samplers (`emcee`, `zeus`, `dynesty`) that run a sampling loop per observation, the SBI module trains a neural density estimator (Masked Autoregressive Flow + DeepSet embedding) on simulated data upfront. After training, posterior inference on any new observation is near-instant.

## Installation

```bash
pip install -e ".[sbi]"
```

This installs `sbi>=0.22.0` and `torch>=2.0` alongside the core TransFit dependencies.

For a dataset-backed walkthrough, see `examples/tutorial_sbi_sn1993j.ipynb`.
That notebook trains a bolometric SBI posterior on `examples/data/sn1993j_lbol.txt`
and demonstrates how to recover physical parameters from the observed light curve.

## Quick Start

### Training

```python
import transfit as tf

posterior = tf.sbi.train_sbi(
    model="nickel",
    mode="multiband",
    z=0.01,
    distance_modulus=30.0,
    filters={"B": "johnson_cousins.B", "V": "johnson_cousins.V"},
    priors={"M_ej": (1.0, 8.0), "v_ej": (0.3, 2.0), "M_ni": (0.01, 0.5)},
    fixed={"kappa": 0.2, "kappa_gamma": 0.03, "R_0": 10.0, "T_floor": 5000.0,
           "f_ni": 0.5, "E_Th_in": 0.0, "t_shift": 0.0, "delta": 0.0, "n": 10.0},
    bands_pool=["B", "V"],
    n_simulations=5000,
    max_num_epochs=100,
    device="cuda",  # optional: defaults to CUDA when available, else CPU
)
```

For bolometric light curves, use `mode="bolometric"` and omit `filters`, `bands_pool`, and `distance_modulus`.
At inference time, bolometric observations should be passed in `log10(L)` space because the bolometric simulator is trained on log-luminosity targets.

If you request `device="cuda"`, TransFit passes that through to `sbi.SNPE` and keeps inference tensors on the same device. CUDA still depends on your local PyTorch build and NVIDIA driver being compatible.

### Inference

```python
result = tf.sbi.infer_sbi(
    posterior,
    y_obs=magnitudes,       # np.ndarray of observed values
    t_days=times,           # np.ndarray of observation times
    band=band_labels,       # np.ndarray of band names (multiband only)
    n_samples=5000,
)
print(result["median"])     # dict: param_name -> value
print(result["map"])        # dict: param_name -> MAP estimate
```

### Save / Load

```python
tf.sbi.save_posterior(posterior, "nickel_posterior.pt")

# Later:
posterior = tf.sbi.load_posterior("nickel_posterior.pt", trusted=True)
```

### Diagnostics

```python
from transfit.sbi import simulation_based_calibration, posterior_predictive_check

# Simulation-Based Calibration
sbc_result = simulation_based_calibration(
    posterior, simulator, prior,
    n_tests=100, t_days=times, band=band_labels,
)

# Posterior Predictive Check
ppc_result = posterior_predictive_check(
    posterior, simulator, y_obs,
    t_days=times, band=band_labels, n=100,
)
```

## Running Tests

All SBI tests are in `tests/test_sbi.py`. They require `sbi` and `torch` to be installed.

```bash
# Install SBI dependencies first
pip install -e ".[sbi]"

# Run all SBI tests
pytest tests/test_sbi.py -v

# Run a specific test class
pytest tests/test_sbi.py -k "TestPriorAdapter" -v
pytest tests/test_sbi.py -k "TestSimulator" -v
pytest tests/test_sbi.py -k "TestEmbedding" -v
pytest tests/test_sbi.py -k "TestTrainingData" -v
pytest tests/test_sbi.py -k "TestNPETraining" -v
pytest tests/test_sbi.py -k "TestPosteriorBounds" -v
pytest tests/test_sbi.py -k "TestTrainSBIE2E" -v
pytest tests/test_sbi.py -k "TestIO" -v
```

If `sbi` or `torch` is not installed, all tests are **automatically skipped** via `pytest.importorskip`.

### Test Descriptions

| Test Class | What It Tests |
|---|---|
| `TestPriorAdapter` | `TransFitPrior` sample shape, bounds, log_prob correctness |
| `TestSimulator` | Bolometric and multiband simulator output shapes, NaN handling |
| `TestEmbedding` | `SetSummaryNet`, `MLPEmbeddingNet`, `encode_observations`, `encode_batch` padding |
| `TestTrainingData` | Data generation, NaN filtering, cache round-trip |
| `TestNPETraining` | Low-level: simulate → train NPE → sample (uses `MLPEmbeddingNet`) |
| `TestPosteriorBounds` | Posterior samples fall within prior bounds (>80%) |
| `TestTrainSBIE2E` | **Full `train_sbi` → `infer_sbi` pipeline** (uses default `SetSummaryNet`) |
| `TestIO` | Save/load round-trip, `trusted=True` gate |

### Note on slow tests

`TestNPETraining`, `TestPosteriorBounds`, and `TestTrainSBIE2E` train small neural networks (200 simulations, 5 epochs). They are the slowest tests in the suite. For a quick smoke test:

```bash
pytest tests/test_sbi.py -v -k "not (NPE or PosteriorBounds or TrainSBIE2E)"
```

## Architecture

```
transfit/sbi/
  __init__.py     train_sbi(), infer_sbi()
  prior.py        TransFitPrior — wraps MixedBoundsPrior as torch Distribution
  simulator.py    make_bolometric_simulator / make_multiband_simulator
  embedding.py    SetSummaryNet (DeepSet, auto-mask via validity column) + MLPEmbeddingNet
  training.py     generate_training_data with caching + parallel execution
  posterior.py    SBIPosterior dataclass: sample, log_prob, map_estimate, median
  diagnostics.py  simulation_based_calibration, posterior_predictive_check
  io.py           save_posterior / load_posterior
```

## Logic Flow

### Training Phase (`train_sbi`)

```
用户输入: model, mode, priors, fixed, cadence_templates, ...
│
│  1. 构建 Prior
│     priors ──> _split_prior_specs() ──> (priors_lin, priors_log10)
│                    │
│                    ├──> build_bounds(model, priors_lin) ──> (names_all, bounds_all)
│                    │
│                    └──> _apply_log10_priors(names_all, bounds_all, priors_log10)
│                             ──> (bounds_all, log_set_all)
│
│     _split_sampling(names_all, bounds_all, fixed)
│         ──> (names_samp, bounds_samp, fixed_dict)
│
│     MixedBoundsPrior(bounds_samp, names_samp, log_flags)
│         ──> TransFitPrior(mixed_prior)     ← torch Distribution, 供 sbi 使用
│
│  2. 生成观测 cadence 模板
│     cadence_templates (用户提供)
│         或 _generate_random_cadences()
│             ──> [{"t_days": [...], "band": [...]}, ...]
│
│  3. 对每个 cadence 模板循环生成训练数据
│     ┌──────────────────────────────────────────────────────────────┐
│     │  for tmpl in cadence_templates:                             │
│     │      │                                                      │
│     │      ├── make_bolometric_simulator / make_multiband_simulator
│     │      │       闭包捕获: model, z, t_days, filters, fixed, ... │
│     │      │                                                      │
│     │      ├── generate_training_data(simulator, prior, n_sims)   │
│     │      │       │                                              │
│     │      │       ├── prior.sample((n_sims,)) ──> theta (tensor) │
│     │      │       │                                              │
│     │      │       └── for each theta_i:                          │
│     │      │            simulator(theta_i):                       │
│     │      │              theta_i ──> _param_values_from_sample() │
│     │      │                  ──> _physical_constraints_lnprior() │
│     │      │                      不通过? → NaN, continue          │
│     │      │                  ──> _assemble_model_params_from_values()   │
│     │      │                      ──> (theta_model, t_shift)      │
│     │      │                  ──> predict_bol / predict_multiband │
│     │      │                      ──> y (光变曲线)                 │
│     │      │                  ──> log10(y) [bolometric] / y [multi]│
│     │      │                  ──> 添加噪声 (noise_sigma / model)  │
│     │      │              ──> x_batch (tensor, 可能含 NaN)        │
│     │      │                                                      │
│     │      │       └── _filter_nans() ──> (theta, x, stats)      │
│     │      │                                                      │
│     │      └── encode_batch(x_batch, t_days, band, ...)           │
│     │              │                                              │
│     │              ├── for each observation:                      │
│     │              │     encode_observations(y, t_days, band)     │
│     │              │       ──> [t_norm, onehot?, y_value]         │
│     │              │                                              │
│     │              ├── pad 到统一长度 max_n_obs                    │
│     │              │                                              │
│     │              └── 追加 validity 列 (1.0=有效, 0.0=padding)  │
│     │                  ──> (x_enc: [batch, max_n_obs, feat_dim+1])│
│     └──────────────────────────────────────────────────────────────┘
│
│  4. 合并并过滤
│     torch.cat(all_theta), torch.cat(all_x_encoded)
│         ──> 过滤含 NaN/Inf 的行
│         ──> (theta_train, x_train)
│
│  5. 构建 Embedding Net
│     SetSummaryNet(feature_dim, hidden_features, output_dim)
│         phi: Linear(feat_dim-1 → hidden) → ReLU → Linear → ReLU
│         rho: Linear(hidden → hidden) → ReLU → Linear → output_dim
│         forward 时自动从输入最后一列推断 mask, 忽略 padding
│
│  6. 训练 NPE (Neural Posterior Estimation)
│     density_estimator = posterior_nn("maf", embedding_net, ...)
│     SNPE(prior, density_estimator)
│         .append_simulations(theta_train, x_train)
│         .train(max_num_epochs, ...)
│         .build_posterior()
│
│  7. 返回
│     SBIPosterior(model, param_names, sbi_posterior, embedding_net, meta, ...)
```

### Inference Phase (`infer_sbi` / `SBIPosterior.sample`)

```
用户输入: posterior, y_obs, t_days, band, n_samples
│
│  SBIPosterior._encode_observation(y_obs, t_days, band)
│      │
│      ├── encode_observations(y_obs, t_days, band, band_vocabulary, t_range)
│      │       ──> features: [t_norm, onehot?, y_value]  shape (n_obs, feat_dim)
│      │
│      ├── 追加 validity 列 (全 1.0, 无 padding)
│      │       ──> x_encoded: [t_norm, onehot?, y_value, 1.0]  shape (1, n_obs, feat_dim+1)
│      │
│      └── 注意: 此处返回原始特征, 不经过 embedding_net
│          (embedding_net 由 sbi DirectPosterior 在内部调用)
│
│  sbi_posterior.sample((n_samples,), x=x_encoded)
│      │
│      └── sbi 内部:
│          x_emb = embedding_net(x_encoded)
│              SetSummaryNet.forward(x_encoded):
│                mask = (x_encoded[..., -1] != 0.0)    ← 全 True
│                features = x_encoded[..., :-1]
│                h = phi(features) → masked_sum / n_valid → rho(h)
│              ──> (1, output_dim)
│          MAF flow: sample from p(theta | x_emb)
│              ──> samples: shape (n_samples, ndim)
│
│  计算统计量:
│      median  = np.median(samples, axis=0)
│      map     = argmax(log_prob(samples))
│
│  返回 {"samples", "median", "map", "param_names"}
```

### Diagnostics

```
Simulation-Based Calibration (SBC):
  for i in range(n_tests):
      theta_true ← prior.sample()
      x_sim      ← simulator(theta_true)
      post_samples ← posterior.sample(n, x_sim)
      ranks[d] = sum(post_samples[:, d] < theta_true[d])
  ──> 若后验校准良好, ranks 应服从均匀分布

Posterior Predictive Check (PPC):
  theta_samples ← posterior.sample(n, y_obs)
  y_rep         ← simulator(theta_samples)
  ──> 对比 y_rep 与 y_obs (可视化 / 统计检验)
```

### Variable-Length Cadence Handling

The key design challenge is that different training simulations (and different real observations) may have different numbers of data points at different times in different bands. This is handled by:

1. `encode_batch()` pads variable-length observations to a common length and appends a **validity indicator column** (1.0 = real data, 0.0 = padding).
2. `SetSummaryNet` (DeepSet architecture) automatically infers the mask from this column and ignores padded entries via masked aggregation.
3. This allows `sbi`'s internal training pipeline to call `embedding_net(x)` as a standard single-argument forward pass, without requiring a separate mask argument.
