"""Inference latency and throughput benchmarking."""

import time
import numpy as np
import torch


def _sync_device(device: torch.device):
    """Synchronize GPU/MPS execution for accurate timing."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def benchmark_model(model: torch.nn.Module, scaler, X_unscaled_pool: np.ndarray,
                    batch_size: int, n_warmup: int, n_runs: int,
                    device: torch.device, seed: int = 42):
    """End-to-end latency benchmark: scaler.transform + tensor build + model forward.

    Args:
        model: Trained PyTorch model
        scaler: Fitted RobustScaler
        X_unscaled_pool: Unscaled feature pool to sample from
        batch_size: Batch size for inference
        n_warmup: Warmup runs (discarded)
        n_runs: Timed runs
        device: torch.device
        seed: RNG seed

    Returns:
        dict with p50_ms, p95_ms, p99_ms, mean_ms, throughput_flows_per_sec
    """
    model.eval()
    pool_size = X_unscaled_pool.shape[0]
    rng = np.random.default_rng(seed)

    def one_pass():
        idx = rng.integers(0, pool_size, size=batch_size)
        raw = X_unscaled_pool[idx]
        scaled = scaler.transform(raw).astype(np.float32)
        x = torch.from_numpy(scaled).to(device, non_blocking=True)
        _sync_device(device)
        with torch.no_grad():
            _ = model(x)
        _sync_device(device)

    for _ in range(n_warmup):
        one_pass()

    samples = np.empty(n_runs, dtype=np.float64)
    for i in range(n_runs):
        _sync_device(device)
        t0 = time.perf_counter()
        one_pass()
        samples[i] = time.perf_counter() - t0

    return {
        'p50_ms': float(np.percentile(samples, 50) * 1000),
        'p95_ms': float(np.percentile(samples, 95) * 1000),
        'p99_ms': float(np.percentile(samples, 99) * 1000),
        'mean_ms': float(samples.mean() * 1000),
        'throughput_flows_per_sec': float(batch_size / samples.mean()),
    }
