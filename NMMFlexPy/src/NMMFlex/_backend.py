"""Backend-agnostic tensor ops.

Most of the NMMF update rules can be written with a small set of
primitives: ``matmul``, ``sum``, elementwise multiply/divide, ``log``,
and a safe-divide that handles zero denominators. This module exposes
those primitives in a single API that works on either numpy arrays or
PyTorch tensors, dispatching by input type.

Why a shim instead of two parallel implementations:
  - The math is identical -- duplicating numpy/torch code is bug bait.
  - Adding a new operation in one place is cheap; in two places it
    rots.
  - End users get a single set of functions to call; the backend
    follows their data.

Conventions:
  - ``axis=`` is used everywhere; we translate to torch's ``dim=``.
  - Functions return the same backend they received (numpy in, numpy
    out; torch in, torch out). Mixing in a single call is unsupported.
  - PyTorch is optional. ``is_torch()`` returns False if it isn't
    installed, and every shim function falls through to numpy in that
    case.
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


def is_torch(x) -> bool:
    return _HAS_TORCH and isinstance(x, _torch.Tensor)


# ---------------------------------------------------------------------------
# Reductions / elementwise
# ---------------------------------------------------------------------------

def sum(a, axis=None, keepdims: bool = False):
    if is_torch(a):
        if axis is None:
            return a.sum()
        return a.sum(dim=axis, keepdim=keepdims)
    return np.sum(a, axis=axis, keepdims=keepdims)


def log(x):
    if is_torch(x):
        return _torch.log(x)
    # errstate so log(0) returns -inf without warnings; matches the
    # legacy factorization.division -> log(0) behaviour.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x)


def where(cond, x, y):
    if is_torch(cond):
        return _torch.where(cond, x, y)
    return np.where(cond, x, y)


def zeros_like(x):
    if is_torch(x):
        return _torch.zeros_like(x)
    return np.zeros_like(x)


def ones_like(x):
    if is_torch(x):
        return _torch.ones_like(x)
    return np.ones_like(x)


def full_like(x, fill_value):
    if is_torch(x):
        return _torch.full_like(x, fill_value)
    return np.full_like(x, fill_value, dtype=np.result_type(x, np.float64))


def abs(x):
    if is_torch(x):
        return _torch.abs(x)
    return np.abs(x)


def to_python_float(x) -> float:
    """Coerce a 0-D numpy array or torch scalar tensor to a Python
    float, so iteration-loop tracking can use plain Python arithmetic
    on convergence metrics without dragging tensors around.
    """
    if is_torch(x):
        return float(x.item() if x.ndim == 0 else x.detach().cpu().numpy())
    return float(x)


def expand_dims(x, axis: int):
    """Add a singleton dimension at ``axis``. Matches np.expand_dims /
    torch.unsqueeze. Convenience wrapper because they look different
    in code that needs to broadcast a 1-D denominator into a matrix.
    """
    if is_torch(x):
        return x.unsqueeze(axis)
    return np.expand_dims(x, axis)


# ---------------------------------------------------------------------------
# Safe divide -- matches factorization.division semantics (0 on zero
# denominator) but vectorized.
# ---------------------------------------------------------------------------

def safe_divide(num, den, default: float = 0.0):
    """Elementwise ``num / den``, returning ``default`` wherever
    ``den == 0``. Mirrors ``factorization.division`` but operates on
    arrays/tensors at once.
    """
    if is_torch(num) or is_torch(den):
        mask = den != 0
        # Replace zeros in the denominator with ones inside the masked
        # region so the division never sees a zero, then patch the
        # masked-out positions with the default value via `where`.
        safe_den = _torch.where(mask, den, _torch.ones_like(den))
        return _torch.where(
            mask,
            num / safe_den,
            _torch.full_like(num, default),
        )
    out = np.full_like(
        num, default, dtype=np.result_type(num, den, np.float64)
    )
    mask = den != 0
    np.divide(num, den, out=out, where=mask)
    return out
