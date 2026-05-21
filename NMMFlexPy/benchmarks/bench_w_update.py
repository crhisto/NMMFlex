"""Benchmark the three implementations of the W multiplicative update.

Run from the repo root with the venv activated:

    .venv/bin/python NMMFlexPy/benchmarks/bench_w_update.py

The triple-loop reference is so slow that we keep its problem size tiny.
The vectorized numpy and torch backends are timed at a realistic
omics-deconvolution shape.
"""

from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np

# Allow running directly without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from NMMFlex.factorization import factorization  # noqa: E402
from NMMFlex import torch_backend as tb  # noqa: E402


def _time(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def bench(I: int, J: int, K: int, run_loop: bool, label: str) -> None:
    rng = np.random.default_rng(0)
    w = rng.uniform(0.1, 1.0, size=(I, K))
    h = rng.uniform(0.1, 1.0, size=(K, J))
    x = rng.uniform(0.1, 1.0, size=(I, J))
    x_hat = w @ h

    print(f"\n[{label}] shape I={I}, J={J}, K={K}")

    if run_loop:
        f = factorization()
        t_loop = _time(lambda: f._calculate_w_new_extended(x, x_hat, w, h), repeat=1)
        print(f"  triple-loop reference : {t_loop*1000:9.2f} ms")
    else:
        t_loop = None
        print(f"  triple-loop reference :   skipped (would take too long)")

    t_np = _time(lambda: tb.calculate_w_new_np(x, x_hat, w, h))
    print(f"  vectorized numpy       : {t_np*1000:9.2f} ms")

    if tb.has_torch():
        import torch
        for dtype, name in [(torch.float64, "torch f64 cpu"),
                            (torch.float32, "torch f32 cpu")]:
            tx = torch.from_numpy(x).to(dtype)
            txh = torch.from_numpy(x_hat).to(dtype)
            tw = torch.from_numpy(w).to(dtype)
            th = torch.from_numpy(h).to(dtype)
            # Warm-up
            _ = tb.calculate_w_new_torch(tx, txh, tw, th)
            t_t = _time(lambda: tb.calculate_w_new_torch(tx, txh, tw, th))
            print(f"  {name:22s}: {t_t*1000:9.2f} ms")

        if torch.backends.mps.is_available():
            dev = torch.device("mps")
            tx = torch.from_numpy(x).float().to(dev)
            txh = torch.from_numpy(x_hat).float().to(dev)
            tw = torch.from_numpy(w).float().to(dev)
            th = torch.from_numpy(h).float().to(dev)
            _ = tb.calculate_w_new_torch(tx, txh, tw, th); torch.mps.synchronize()

            def _run():
                _ = tb.calculate_w_new_torch(tx, txh, tw, th)
                torch.mps.synchronize()
            t_mps = _time(_run)
            print(f"  torch f32 mps         : {t_mps*1000:9.2f} ms")

        if torch.cuda.is_available():
            dev = torch.device("cuda")
            tx = torch.from_numpy(x).float().to(dev)
            txh = torch.from_numpy(x_hat).float().to(dev)
            tw = torch.from_numpy(w).float().to(dev)
            th = torch.from_numpy(h).float().to(dev)
            _ = tb.calculate_w_new_torch(tx, txh, tw, th); torch.cuda.synchronize()

            def _run():
                _ = tb.calculate_w_new_torch(tx, txh, tw, th)
                torch.cuda.synchronize()
            t_cuda = _time(_run)
            print(f"  torch f32 cuda        : {t_cuda*1000:9.2f} ms")

    if t_loop is not None:
        print(f"  speedup numpy vs loop  : {t_loop / t_np:9.1f}x")


if __name__ == "__main__":
    # Small enough that the triple-loop finishes in seconds.
    bench(I=60, J=40, K=5, run_loop=True, label="small (loop included)")
    # Realistic deconvolution shape.
    bench(I=5000, J=200, K=10, run_loop=False, label="omics-ish")
    # Larger shape to give the GPU room to shine.
    bench(I=20000, J=500, K=20, run_loop=False, label="large")
