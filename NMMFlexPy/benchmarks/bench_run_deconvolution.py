"""End-to-end benchmark for ``run_deconvolution_multiple``.

Times the full deconvolution pipeline (input standardization,
checks, init, iteration loop, result wrap-up) for each backend at
several realistic shapes. Prints a markdown table to stdout, fit to
paste into a PR description.

Usage from the repo root:

    .venv/bin/python NMMFlexPy/benchmarks/bench_run_deconvolution.py

Skip the large shape by passing ``--small``; helpful when iterating
on the script. On Apple Silicon the MPS row will populate; on a
CUDA-equipped Linux box the CUDA row will populate; both are
skipped on environments where the device isn't available.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from NMMFlex.factorization import factorization  # noqa: E402
from NMMFlex import _backend as B  # noqa: E402


def _make_problem(I: int, J: int, K: int, seed: int = 0):
    """Random non-negative X plus matching initial W and H."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.uniform(0.1, 1.0, size=(I, J)))
    w0 = rng.uniform(0.1, 1.0, size=(I, K))
    h0 = rng.uniform(0.1, 1.0, size=(K, J))
    return x, w0, h0


def _time_run(backend: str, device: str, dtype, x, w0, h0,
              max_iterations: int, k: int) -> float:
    """Wall-clock seconds for one full run_deconvolution_multiple
    call. Returns the minimum of 3 runs to factor out warmup
    noise."""
    f = factorization(backend=backend, device=device, dtype=dtype)

    def _init(size_rows, size_columns, **_):
        if (size_rows, size_columns) == w0.shape:
            return w0.copy()
        if (size_rows, size_columns) == h0.shape:
            return h0.copy()
        raise AssertionError(
            f"unexpected init shape {(size_rows, size_columns)}"
        )

    f._initialize_matrix = _init  # noqa: SLF001

    best = float("inf")
    devnull = open(os.devnull, "w")
    try:
        for _ in range(3):
            # Fresh instance per run so timing isn't polluted by state
            # left behind on the previous self.w / self.h assignments.
            f = factorization(backend=backend, device=device, dtype=dtype)
            f._initialize_matrix = _init  # noqa: SLF001

            t0 = time.perf_counter()
            with contextlib.redirect_stdout(devnull):
                f.run_deconvolution_multiple(
                    x_matrix=x, y_matrix=None, z_matrix=None, k=k,
                    gamma=1.0, alpha=0.0, beta=0.0,
                    delta_threshold=1e-30,    # force max_iterations
                    max_iterations=max_iterations,
                    proportion_constraint_h=True,
                    verbose=False, print_limit=10**9,
                )
            if device.startswith("cuda"):
                import torch
                torch.cuda.synchronize()
            elif device == "mps":
                import torch
                torch.mps.synchronize()
            dt = time.perf_counter() - t0
            best = min(best, dt)
    finally:
        devnull.close()
    return best


def _row(label: str, shape, max_iter, fn_results: dict) -> str:
    """Format one markdown table row."""
    I, J, K = shape
    cells = [f"{label}", f"{I}×{J}×{K}", str(max_iter)]
    baseline = fn_results.get("numpy")
    for name in ("numpy", "torch-f64-cpu", "torch-f32-cpu",
                  "torch-f32-mps", "torch-f32-cuda"):
        t = fn_results.get(name)
        if t is None:
            cells.append("—")
        else:
            label_ms = f"{t * 1000:.0f} ms"
            if baseline and name != "numpy":
                label_ms += f" ({baseline / t:.1f}×)"
            cells.append(label_ms)
    return "| " + " | ".join(cells) + " |"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true",
                        help="skip the largest shape")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="iterations per run (default 50)")
    args = parser.parse_args()

    shapes = [("small", 200, 50, 5, args.max_iter),
              ("omics", 5000, 200, 10, args.max_iter)]
    if not args.small:
        shapes.append(("large", 20000, 500, 20, args.max_iter))

    has_torch = B.has_torch()
    import_cuda = False
    import_mps = False
    if has_torch:
        import torch
        import_cuda = torch.cuda.is_available()
        import_mps = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

    print("# run_deconvolution_multiple backend benchmark\n")
    print(
        f"PyTorch available: {has_torch}; CUDA: {import_cuda}; "
        f"MPS: {import_mps}\n"
    )
    print(
        "| Shape label | I×J×K | iters | numpy | torch f64 CPU | "
        "torch f32 CPU | torch f32 MPS | torch f32 CUDA |"
    )
    print(
        "|---|---|---|---|---|---|---|---|"
    )

    for label, I, J, K, max_iter in shapes:
        x, w0, h0 = _make_problem(I, J, K)

        results: dict = {}

        # numpy baseline
        results["numpy"] = _time_run("numpy", "cpu", None,
                                      x, w0, h0, max_iter, K)

        if has_torch:
            import torch
            results["torch-f64-cpu"] = _time_run(
                "torch", "cpu", torch.float64,
                x, w0, h0, max_iter, K,
            )
            results["torch-f32-cpu"] = _time_run(
                "torch", "cpu", torch.float32,
                x, w0, h0, max_iter, K,
            )
            if import_mps:
                results["torch-f32-mps"] = _time_run(
                    "torch", "mps", torch.float32,
                    x, w0, h0, max_iter, K,
                )
            if import_cuda:
                results["torch-f32-cuda"] = _time_run(
                    "torch", "cuda", torch.float32,
                    x, w0, h0, max_iter, K,
                )

        print(_row(label, (I, J, K), max_iter, results))


if __name__ == "__main__":
    main()
