# transfit/sbi/diagnostics.py
"""SBI diagnostic tools: Simulation-Based Calibration and posterior predictive checks."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .posterior import SBIPosterior
from .prior import TransFitPrior


def simulation_based_calibration(
    posterior: SBIPosterior,
    simulator: Callable,
    prior: TransFitPrior,
    *,
    n_tests: int = 100,
    n_posterior_samples: int = 1000,
    seed: int = 42,
    t_days: Optional[np.ndarray] = None,
    band: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run Simulation-Based Calibration (SBC) to check posterior calibration.

    For each test case:
    1. Draw theta_true from prior
    2. Simulate x_obs
    3. Draw posterior samples given x_obs
    4. Compute rank statistic: how many posterior samples < theta_true

    If the posterior is well-calibrated, rank statistics should be
    uniformly distributed across [0, n_posterior_samples].

    Parameters
    ----------
    posterior : SBIPosterior
        Trained posterior.
    simulator : callable
        theta_tensor -> x_tensor simulator.
    prior : TransFitPrior
        Prior distribution.
    n_tests : int
        Number of SBC test cases.
    n_posterior_samples : int
        Number of posterior samples per test case.
    seed : int
        Random seed.
    t_days, band : optional
        Observation pattern for the simulator.

    Returns
    -------
    dict with keys:
        ranks: np.ndarray, shape (n_tests, ndim)
        theta_true: np.ndarray, shape (n_tests, ndim)
        uniform_expected: expected uniform distribution
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    ndim = prior.event_shape[0]
    ranks = np.zeros((n_tests, ndim), dtype=int)
    theta_true_all = np.zeros((n_tests, ndim), dtype=float)
    n_valid = 0

    for i in range(n_tests):
        # Draw true parameters
        theta_true = prior.sample((1,))  # (1, ndim)

        # Simulate observation
        x_sim = simulator(theta_true)  # (1, n_obs)
        x_np = np.asarray(x_sim[0].detach().numpy(), float)

        # Skip if simulation produced NaN
        if not np.all(np.isfinite(x_np)):
            continue

        # Draw posterior samples
        try:
            post_samples = posterior.sample(
                n_posterior_samples, x_np,
                t_days=t_days, band=band,
            )
        except Exception:
            continue

        if not np.all(np.isfinite(post_samples)):
            continue

        theta_true_np = np.asarray(theta_true[0].detach().numpy(), float)

        # Compute ranks
        for d in range(ndim):
            ranks[n_valid, d] = int(np.sum(post_samples[:, d] < theta_true_np[d]))

        theta_true_all[n_valid] = theta_true_np
        n_valid += 1

    ranks = ranks[:n_valid]
    theta_true_all = theta_true_all[:n_valid]

    return {
        "ranks": ranks,
        "theta_true": theta_true_all,
        "n_valid": n_valid,
        "n_posterior_samples": n_posterior_samples,
        "param_names": prior.param_names,
    }


def posterior_predictive_check(
    posterior: SBIPosterior,
    simulator: Callable,
    y_obs: np.ndarray,
    *,
    t_days: Optional[np.ndarray] = None,
    band: Optional[np.ndarray] = None,
    n: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Posterior predictive check: sample from posterior, simulate, compare.

    Parameters
    ----------
    posterior : SBIPosterior
        Trained posterior.
    simulator : callable
        theta_tensor -> x_tensor.
    y_obs : np.ndarray
        Observed values.
    t_days, band : optional
        Observation pattern.
    n : int
        Number of posterior predictive samples.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict with keys:
        y_rep: np.ndarray, shape (n, n_obs) -- replicated observations
        y_obs: np.ndarray -- original observation
        theta_samples: np.ndarray, shape (n, ndim) -- posterior samples used
    """
    y_obs = np.asarray(y_obs, float)

    # Draw posterior samples
    theta_samples = posterior.sample(
        n, y_obs, t_days=t_days, band=band, seed=seed
    )

    # Simulate replicated observations
    theta_tensor = torch.as_tensor(theta_samples, dtype=torch.float32)
    y_rep_tensor = simulator(theta_tensor)
    y_rep = np.asarray(y_rep_tensor.detach().numpy(), float)

    return {
        "y_rep": y_rep,
        "y_obs": y_obs,
        "theta_samples": theta_samples,
    }
