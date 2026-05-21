"""Vectorized numpy + optional PyTorch implementations of the NMMF
multiplicative update rules.

This module is a proof-of-concept showing that the slow triple-nested
Python loops in factorization.py (e.g. _calculate_w_new_extended,
_calculate_h_new_extended, _calculate_x_hat_extended,
_calculate_divergence_extended) can be replaced with pure linear-algebra
calls and then optionally executed on GPU via PyTorch with the same code
shape.

Three implementations are provided for each operation:
  - The original loop versions live in factorization.py and are used as
    the reference for parity tests.
  - ``*_np`` functions here implement the same update with vectorized
    numpy. This alone gives a large speedup over the loop version.
  - ``*_torch`` functions accept torch tensors. They mirror the numpy
    code line-for-line; just create the inputs on a GPU device to get
    GPU execution.

Conventions match the original code:
  X has shape (I, J)         observed matrix
  W has shape (I, K)         factor (signatures)
  H has shape (K, J)         factor (proportions)
  X_hat = W @ H              reconstruction
"""

from __future__ import annotations

import numpy as np

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _torch = None
    _HAS_TORCH = False


def has_torch() -> bool:
    return _HAS_TORCH


# ---------------------------------------------------------------------------
# numpy implementations (vectorized)
# ---------------------------------------------------------------------------

def _safe_divide_np(num: np.ndarray, den: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Elementwise num/den, returning ``default`` wherever den == 0.

    Matches the semantics of factorization.division (which returns 0 on
    zero-denominator) but operates on whole arrays at once.
    """
    out = np.full_like(num, default, dtype=np.result_type(num, den, np.float64))
    mask = den != 0
    np.divide(num, den, out=out, where=mask)
    return out


def calculate_x_hat_np(w: np.ndarray, h: np.ndarray) -> np.ndarray:
    return w @ h


def calculate_divergence_np(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Generalized KL divergence elementwise: x * log(x/x_hat) - x + x_hat.

    Mirrors _calculate_divergence_extended. The original calls
    division(x, x_hat) which returns 0 when x_hat == 0; np.log(0) is then
    -inf. We reproduce that exact behaviour for parity.
    """
    ratio = _safe_divide_np(x, x_hat, default=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(ratio)
    return x * log_ratio - x + x_hat


def calculate_w_new_np(
    x: np.ndarray,
    x_hat: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Vectorized form of _calculate_w_new_extended.

    Multiplicative update:
        w_new[i, k] = w[i, k] * sum_j (x[i, j] / x_hat[i, j]) * h[k, j]
                              / sum_j h[k, j]

    In matrix form:
        ratio = X / X_hat                  shape (I, J)
        num   = ratio @ H.T                shape (I, K)
        den   = H.sum(axis=1)              shape (K,)
        W_new = W * (num / den[None, :])
    """
    ratio = _safe_divide_np(x, x_hat)
    num = ratio @ h.T
    den = h.sum(axis=1)
    factor = _safe_divide_np(num, den[None, :])
    return w * factor


def calculate_h_new_np(
    x: np.ndarray,
    x_hat: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Vectorized form of _calculate_h_new_extended (without the
    proportion constraint; apply that separately if needed).

        h_new[k, j] = h[k, j] * sum_i (x[i, j] / x_hat[i, j]) * w[i, k]
                              / sum_i w[i, k]
    """
    ratio = _safe_divide_np(x, x_hat)
    num = w.T @ ratio
    den = w.sum(axis=0)
    factor = _safe_divide_np(num, den[:, None])
    return h * factor


# ---------------------------------------------------------------------------
# torch implementations (mirror the numpy code; device-agnostic)
# ---------------------------------------------------------------------------

def _require_torch():
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is not installed. Install with `pip install torch` "
            "to use the torch backend."
        )


def _safe_divide_torch(num, den, default: float = 0.0):
    out = _torch.full_like(num, default)
    mask = den != 0
    out = _torch.where(mask, num / _torch.where(mask, den, _torch.ones_like(den)), out)
    return out


def calculate_x_hat_torch(w, h):
    _require_torch()
    return w @ h


def calculate_divergence_torch(x, x_hat):
    _require_torch()
    ratio = _safe_divide_torch(x, x_hat, default=0.0)
    log_ratio = _torch.log(ratio)
    return x * log_ratio - x + x_hat


def calculate_w_new_torch(x, x_hat, w, h):
    _require_torch()
    ratio = _safe_divide_torch(x, x_hat)
    num = ratio @ h.T
    den = h.sum(dim=1)
    factor = _safe_divide_torch(num, den.unsqueeze(0))
    return w * factor


def calculate_h_new_torch(x, x_hat, w, h):
    _require_torch()
    ratio = _safe_divide_torch(x, x_hat)
    num = w.T @ ratio
    den = w.sum(dim=0)
    factor = _safe_divide_torch(num, den.unsqueeze(1))
    return h * factor
