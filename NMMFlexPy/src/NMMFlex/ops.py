"""Backend-agnostic NMMF update rules.

The functions here use the ``_backend`` shim and work transparently
on numpy arrays or PyTorch tensors. Pass numpy in, get numpy out;
pass torch in (any device), get torch out.

This is the unified replacement for the parallel ``*_np`` / ``*_torch``
functions in ``torch_backend.py``. The older module is kept for
backwards compatibility with the original parity tests, but new code
should import from here.

Conventions (same as ``factorization.py``):
    X has shape (I, J)         observed matrix
    W has shape (I, K)         factor (signatures)
    H has shape (K, J)         factor (proportions)
    X_hat = W @ H              reconstruction
"""

from __future__ import annotations

from . import _backend as B


def calculate_x_hat(w, h):
    """Reconstruction X̂ = W @ H. The @ operator works on both
    numpy arrays and torch tensors."""
    return w @ h


def calculate_divergence(x, x_hat):
    """Elementwise generalized KL divergence:
        D(x || x̂) = x * log(x / x̂) - x + x̂
    Where x̂ = 0 the safe-divide returns 0, and log(0) is -inf -- this
    matches the legacy ``_calculate_divergence_extended`` behaviour
    exactly so parity tests stay tight.
    """
    ratio = B.safe_divide(x, x_hat, default=0.0)
    log_ratio = B.log(ratio)
    return x * log_ratio - x + x_hat


def calculate_w_new(x, x_hat, w, h):
    """Multiplicative update for W (Lee & Seung KL variant), vectorized:

        ratio = X / X̂                  shape (I, J)
        num   = ratio @ H.T             shape (I, K)
        den   = H.sum(axis=1)           shape (K,)
        W_new = W * (num / den)         broadcasting den as a row

    Replaces ``factorization._calculate_w_new_extended`` triple loop.
    """
    ratio = B.safe_divide(x, x_hat)
    num = ratio @ h.T
    den = B.sum(h, axis=1)
    factor = B.safe_divide(num, B.expand_dims(den, 0))
    return w * factor


def calculate_h_new(x, x_hat, w, h):
    """Multiplicative update for H (no proportion constraint here --
    apply it as a post-step if needed):

        ratio = X / X̂                  shape (I, J)
        num   = W.T @ ratio             shape (K, J)
        den   = W.sum(axis=0)           shape (K,)
        H_new = H * (num / den)         broadcasting den as a column
    """
    ratio = B.safe_divide(x, x_hat)
    num = w.T @ ratio
    den = B.sum(w, axis=0)
    factor = B.safe_divide(num, B.expand_dims(den, 1))
    return h * factor
