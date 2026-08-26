# transfit/sbi/training.py
"""Training data generation for SBI with parallel execution and caching."""
from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from .prior import TransFitPrior


def _filter_nans(
    theta: torch.Tensor, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Remove rows where any element of x is NaN or Inf."""
    x_np = np.asarray(x, float)
    valid = np.all(np.isfinite(x_np), axis=1)
    n_total = x_np.shape[0]
    n_valid = int(valid.sum())
    stats = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_total - n_valid,
    }
    return theta[valid], x[valid], stats


def _warmup_simulator(simulator: Callable, ndim: int):
    """Run one simulation to trigger Numba JIT compilation."""
    dummy = torch.zeros(1, ndim, dtype=torch.float32)
    try:
        simulator(dummy)
    except Exception:
        pass


def generate_training_data(
    *,
    simulator: Callable[[torch.Tensor], torch.Tensor],
    prior: TransFitPrior,
    n_simulations: int,
    n_workers: int = 1,
    seed: int = 42,
    cache_path: Optional[str] = None,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Generate (theta, x) training pairs for NPE training.

    Parameters
    ----------
    simulator : callable
        theta_tensor -> x_tensor.
    prior : TransFitPrior
        Prior distribution to draw theta from.
    n_simulations : int
        Number of simulations to run.
    n_workers : int
        Number of parallel workers (1 = sequential).
    seed : int
        Random seed for reproducibility.
    cache_path : str, optional
        If provided, cache results to this NPZ path. Loads from cache if file exists.
    show_progress : bool
        Show progress bar.

    Returns
    -------
    theta : torch.Tensor, shape (n_valid, ndim)
    x : torch.Tensor, shape (n_valid, n_obs)
    stats : dict with generation statistics
    """
    # Check cache
    if cache_path is not None and os.path.exists(cache_path):
        data = np.load(cache_path)
        theta = torch.as_tensor(data["theta"], dtype=torch.float32)
        x = torch.as_tensor(data["x"], dtype=torch.float32)
        stats = {
            "n_total": int(data.get("n_total", theta.shape[0])),
            "n_valid": theta.shape[0],
            "n_invalid": int(data.get("n_invalid", 0)),
            "from_cache": True,
        }
        return theta, x, stats

    ndim = prior.event_shape[0]

    # Numba warmup: run 1 simulation to trigger JIT
    _warmup_simulator(simulator, ndim)

    if n_workers == 1:
        theta_all, x_all = _generate_sequential(
            simulator, prior, n_simulations, seed, show_progress
        )
    else:
        theta_all, x_all = _generate_parallel(
            simulator, prior, n_simulations, n_workers, seed
        )

    theta_all, x_all, stats = _filter_nans(theta_all, x_all)
    stats["from_cache"] = False

    # Cache results
    if cache_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(
            cache_path,
            theta=np.asarray(theta_all, float),
            x=np.asarray(x_all, float),
            n_total=stats["n_total"],
            n_invalid=stats["n_invalid"],
        )

    return theta_all, x_all, stats


def _generate_sequential(
    simulator: Callable,
    prior: TransFitPrior,
    n_simulations: int,
    seed: int,
    show_progress: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    # Draw all theta at once
    theta_all = prior.sample_with_rng((n_simulations,), rng=rng)

    # Simulate in chunks for memory efficiency and progress
    chunk_size = min(500, n_simulations)
    x_parts = []

    iterator = range(0, n_simulations, chunk_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Simulating", unit="chunk")
        except ImportError:
            pass

    for start in iterator:
        end = min(start + chunk_size, n_simulations)
        chunk = theta_all[start:end]
        x_chunk = simulator(chunk)
        x_parts.append(x_chunk)

    x_all = torch.cat(x_parts, dim=0)
    return theta_all, x_all


def _generate_parallel(
    simulator: Callable,
    prior: TransFitPrior,
    n_simulations: int,
    n_workers: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    theta_all = prior.sample_with_rng((n_simulations,), rng=rng)

    chunk_size = max(1, n_simulations // n_workers)
    chunks = []
    for start in range(0, n_simulations, chunk_size):
        end = min(start + chunk_size, n_simulations)
        chunks.append(theta_all[start:end])

    results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
        delayed(simulator)(chunk) for chunk in chunks
    )
    x_all = torch.cat(results, dim=0)
    return theta_all, x_all
