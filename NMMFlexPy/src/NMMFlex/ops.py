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

import numpy as np

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


# ---------------------------------------------------------------------------
# Coupled (alpha/beta) update rules.
#
# These are the methods that run inside ``run_deconvolution_multiple``.
# Same multiplicative-update family as ``calculate_w_new`` /
# ``calculate_h_new`` above, but with additional cross-matrix terms:
#   - W is coupled to Z via the b factor (weighted by ``beta``).
#   - H is coupled to Y via the a factor (weighted by ``alpha``).
# When alpha == 0 / beta == 0 the extra terms vanish and the update
# collapses to the basic ``calculate_w_new`` / ``calculate_h_new`` form.
# ---------------------------------------------------------------------------

def calculate_w_new_alpha_beta(x, x_hat, w, h, beta=0.0,
                                z=None, z_hat=None, b=None,
                                alpha_regularizer_w=0.0):
    """Vectorized form of ``factorization._calculate_w_new_extended_alpha_beta``.

        ratio_x       = X / X̂                  shape (I, J)
        up_first      = ratio_x @ H.T           shape (I, K)
        up_second     = (Z / Ẑ) @ B.T           shape (I, K)   (if beta != 0)
        up            = up_first + beta * up_second
        down_first    = H.sum(axis=1)           shape (K,)
        down_second   = B.sum(axis=1)           shape (K,)     (if beta != 0)
        regularizer   = alpha_reg * W * (1 - W) shape (I, K)   (medecom_soft_binary)
        down          = down_first[None, :] + beta * down_second[None, :]
                          + regularizer
        W_new         = W * safe_divide(up, down)

    Only the ``medecom_soft_binary`` regularizer is implemented because
    that is the only branch the original loop body could ever reach
    (``regularizer_function_type`` was hard-coded to that value).
    """
    ratio_x = B.safe_divide(x, x_hat)
    up_first = ratio_x @ h.T
    up = up_first

    down_first = B.sum(h, axis=1)
    down = B.expand_dims(down_first, 0)

    if beta != 0 and z is not None and z_hat is not None and b is not None:
        ratio_z = B.safe_divide(z, z_hat)
        up = up_first + beta * (ratio_z @ b.T)
        down_second = B.sum(b, axis=1)
        down = down + beta * B.expand_dims(down_second, 0)

    if alpha_regularizer_w != 0:
        down = down + alpha_regularizer_w * (w * (1 - w))

    factor = B.safe_divide(up, down)
    return w * factor


def calculate_h_new_alpha_beta(x, x_hat, w, h, alpha=0.0,
                                y=None, y_hat=None, a=None):
    """Vectorized form of ``factorization._calculate_h_new_extended_alpha_beta``.

        ratio_x   = X / X̂                  shape (I, J)
        up_first  = W.T @ ratio_x           shape (K, J)
        up_second = A.T @ (Y / Ŷ)           shape (K, J)   (if alpha != 0)
        up        = up_first + alpha * up_second
        down_first  = W.sum(axis=0)         shape (K,)
        down_second = A.sum(axis=0)         shape (K,)     (if alpha != 0)
        down      = down_first[:, None] + alpha * down_second[:, None]
        H_new     = H * safe_divide(up, down)
    """
    ratio_x = B.safe_divide(x, x_hat)
    up = w.T @ ratio_x

    down_first = B.sum(w, axis=0)
    down = B.expand_dims(down_first, 1)

    if alpha != 0 and y is not None and y_hat is not None and a is not None:
        ratio_y = B.safe_divide(y, y_hat)
        up = up + alpha * (a.T @ ratio_y)
        down_second = B.sum(a, axis=0)
        down = down + alpha * B.expand_dims(down_second, 1)

    factor = B.safe_divide(up, down)
    return h * factor


def calculate_a_new(y, y_hat, a, h):
    """Vectorized form of ``factorization._calculate_a_new_extended``.

        ratio_y = Y / Ŷ                    shape (N, J)
        num     = ratio_y @ H.T            shape (N, K)
        den     = H.sum(axis=1)            shape (K,)
        A_new   = A * safe_divide(num, den)  broadcasting den as a row
    """
    ratio = B.safe_divide(y, y_hat)
    num = ratio @ h.T
    den = B.sum(h, axis=1)
    factor = B.safe_divide(num, B.expand_dims(den, 0))
    return a * factor


def calculate_b_new(z, z_hat, b, w):
    """Vectorized form of ``factorization._calculate_b_new_extended``.

        ratio_z = Z / Ẑ                    shape (I, M)
        num     = W.T @ ratio_z            shape (K, M)
        den     = W.sum(axis=0)            shape (K,)
        B_new   = B * safe_divide(num, den)  broadcasting den as a column
    """
    ratio = B.safe_divide(z, z_hat)
    num = w.T @ ratio
    den = B.sum(w, axis=0)
    factor = B.safe_divide(num, B.expand_dims(den, 1))
    return b * factor


# ---------------------------------------------------------------------------
# Sparse / NaN-tolerant variants
#
# The legacy ``_calculate_*_extended_*_sparse`` methods do two things on
# top of the dense path:
#   1. Accept scipy.sparse (csr_matrix) inputs for the observed
#      matrices X / Y / Z and call ``.toarray()``-equivalent indexing.
#   2. Skip NaN values in any of the inputs (numerator term, h[k,j],
#      w[i,k], b[k,m], etc.) instead of letting them propagate.
#
# The cleanest vectorized form is to:
#   - Materialize sparse inputs to dense via .toarray() (the original
#     loop iterates every (i,j) position anyway, so the memory profile
#     is unchanged).
#   - Replace NaN with 0 in every input.
#   - Reuse the dense ops above.
#
# NaN-as-0 in a *sum* gives the same answer as skipping NaN terms
# (because NaN + non-NaN-contribution becomes 0 + non-NaN). For the
# final assignment ``W_new = W * factor`` the NaN positions in W stay
# 0 in the output, matching the loop version's behaviour (which guards
# the write with ``if not math.isnan(w[i][k])`` and leaves the
# pre-zeroed value in place).
# ---------------------------------------------------------------------------

def _prep_sparse(*arrays):
    """Coerce each input to a dense numpy array with NaN → 0.
    ``None`` passes through unchanged so callers can pass optional
    arguments straight in."""
    out = []
    for arr in arrays:
        if arr is None:
            out.append(None)
            continue
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        arr = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0)
        out.append(arr)
    return out


def calculate_w_new_alpha_beta_sparse(x, x_hat, w, h, beta=0.0,
                                       z=None, z_hat=None, b=None,
                                       alpha_regularizer_w=0.0):
    """Sparse / NaN-tolerant variant of
    ``calculate_w_new_alpha_beta``."""
    x, x_hat, w, h = _prep_sparse(x, x_hat, w, h)
    z, z_hat, b = _prep_sparse(z, z_hat, b)
    return calculate_w_new_alpha_beta(
        x, x_hat, w, h,
        beta=beta, z=z, z_hat=z_hat, b=b,
        alpha_regularizer_w=alpha_regularizer_w,
    )


def calculate_h_new_alpha_beta_sparse(x, x_hat, w, h, alpha=0.0,
                                       y=None, y_hat=None, a=None):
    """Sparse / NaN-tolerant variant of
    ``calculate_h_new_alpha_beta``."""
    x, x_hat, w, h = _prep_sparse(x, x_hat, w, h)
    y, y_hat, a = _prep_sparse(y, y_hat, a)
    return calculate_h_new_alpha_beta(
        x, x_hat, w, h,
        alpha=alpha, y=y, y_hat=y_hat, a=a,
    )


def calculate_a_new_sparse(y, y_hat, a, h):
    """Sparse / NaN-tolerant variant of ``calculate_a_new``."""
    y, y_hat, a, h = _prep_sparse(y, y_hat, a, h)
    return calculate_a_new(y, y_hat, a, h)


def calculate_b_new_sparse(z, z_hat, b, w):
    """Sparse / NaN-tolerant variant of ``calculate_b_new``."""
    z, z_hat, b, w = _prep_sparse(z, z_hat, b, w)
    return calculate_b_new(z, z_hat, b, w)
