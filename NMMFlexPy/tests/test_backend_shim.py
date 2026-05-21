"""Tests for the backend-agnostic shim and the unified ops module.

Verifies that:
  - The shim dispatches correctly on numpy vs torch input.
  - The unified ops in NMMFlex.ops produce the same numerical result
    as the legacy triple-loop methods on factorization.py AND as the
    parallel numpy/torch functions in NMMFlex.torch_backend.
  - safe_divide handles zero denominators identically on both
    backends.
"""

from __future__ import annotations

import numpy as np
import pytest

from NMMFlex import _backend as B
from NMMFlex import ops
from NMMFlex import torch_backend as tb  # legacy parallel impl
from NMMFlex.factorization import factorization

torch = pytest.importorskip("torch") if B.has_torch() else None

ATOL_F64 = 1e-10
ATOL_F32 = 1e-4


def _random_nonneg(shape, rng, low=0.1, high=1.0):
    return rng.uniform(low, high, size=shape)


@pytest.fixture
def small_problem():
    rng = np.random.default_rng(0)
    I, J, K = 12, 9, 3
    w = _random_nonneg((I, K), rng)
    h = _random_nonneg((K, J), rng)
    x = _random_nonneg((I, J), rng)
    x_hat = w @ h
    return x, x_hat, w, h


# ---------------------------------------------------------------------------
# Shim dispatch
# ---------------------------------------------------------------------------

def test_is_torch_on_numpy_returns_false():
    assert not B.is_torch(np.zeros(3))


@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_is_torch_on_torch_returns_true():
    assert B.is_torch(torch.zeros(3))


def test_safe_divide_numpy_returns_zero_on_zero_denominator():
    num = np.array([1.0, 2.0, 3.0])
    den = np.array([1.0, 0.0, 2.0])
    out = B.safe_divide(num, den, default=0.0)
    np.testing.assert_allclose(out, [1.0, 0.0, 1.5])


@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_safe_divide_torch_returns_zero_on_zero_denominator():
    num = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    den = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)
    out = B.safe_divide(num, den, default=0.0)
    np.testing.assert_allclose(out.numpy(), [1.0, 0.0, 1.5])


def test_sum_axis_numpy_matches_native():
    a = np.arange(12.0).reshape(3, 4)
    np.testing.assert_allclose(B.sum(a, axis=0), a.sum(axis=0))
    np.testing.assert_allclose(B.sum(a, axis=1), a.sum(axis=1))
    np.testing.assert_allclose(B.sum(a), a.sum())


@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_sum_axis_torch_translates_to_dim():
    a = torch.arange(12.0, dtype=torch.float64).reshape(3, 4)
    np.testing.assert_allclose(B.sum(a, axis=0).numpy(), a.sum(dim=0).numpy())
    np.testing.assert_allclose(B.sum(a, axis=1).numpy(), a.sum(dim=1).numpy())


def test_expand_dims_numpy():
    a = np.arange(4.0)
    assert B.expand_dims(a, 0).shape == (1, 4)
    assert B.expand_dims(a, 1).shape == (4, 1)


@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_expand_dims_torch():
    a = torch.arange(4.0, dtype=torch.float64)
    assert B.expand_dims(a, 0).shape == (1, 4)
    assert B.expand_dims(a, 1).shape == (4, 1)


# ---------------------------------------------------------------------------
# Unified ops vs legacy implementations (numpy)
# ---------------------------------------------------------------------------

def test_ops_w_new_matches_loop_reference(small_problem):
    x, x_hat, w, h = small_problem
    ref = factorization()._calculate_w_new_extended(x, x_hat, w, h)
    np.testing.assert_allclose(ops.calculate_w_new(x, x_hat, w, h), ref,
                                atol=ATOL_F64)


def test_ops_h_new_matches_loop_reference(small_problem):
    x, x_hat, w, h = small_problem
    ref = factorization()._calculate_h_new_extended(
        x, x_hat, w, h, proportion_constraint=False
    )
    np.testing.assert_allclose(ops.calculate_h_new(x, x_hat, w, h), ref,
                                atol=ATOL_F64)


def test_ops_x_hat_matches_legacy(small_problem):
    _, _, w, h = small_problem
    np.testing.assert_allclose(
        ops.calculate_x_hat(w, h),
        factorization()._calculate_x_hat_extended(w, h),
        atol=ATOL_F64,
    )


def test_ops_divergence_matches_legacy(small_problem):
    x, x_hat, _, _ = small_problem
    np.testing.assert_allclose(
        ops.calculate_divergence(x, x_hat),
        factorization()._calculate_divergence_extended(x, x_hat),
        atol=ATOL_F64,
    )


def test_ops_w_new_matches_torch_backend_np(small_problem):
    """The unified op should match the older PoC ``*_np`` function."""
    x, x_hat, w, h = small_problem
    np.testing.assert_allclose(
        ops.calculate_w_new(x, x_hat, w, h),
        tb.calculate_w_new_np(x, x_hat, w, h),
        atol=ATOL_F64,
    )


# ---------------------------------------------------------------------------
# Unified ops on torch tensors
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_ops_w_new_torch_matches_numpy(small_problem):
    x, x_hat, w, h = small_problem
    ref = ops.calculate_w_new(x, x_hat, w, h)
    got = ops.calculate_w_new(
        torch.from_numpy(x).double(),
        torch.from_numpy(x_hat).double(),
        torch.from_numpy(w).double(),
        torch.from_numpy(h).double(),
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_ops_h_new_torch_matches_numpy(small_problem):
    x, x_hat, w, h = small_problem
    ref = ops.calculate_h_new(x, x_hat, w, h)
    got = ops.calculate_h_new(
        torch.from_numpy(x).double(),
        torch.from_numpy(x_hat).double(),
        torch.from_numpy(w).double(),
        torch.from_numpy(h).double(),
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_ops_divergence_torch_matches_numpy(small_problem):
    x, x_hat, _, _ = small_problem
    ref = ops.calculate_divergence(x, x_hat)
    got = ops.calculate_divergence(
        torch.from_numpy(x).double(), torch.from_numpy(x_hat).double()
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_ops_w_new_torch_float32_within_tolerance(small_problem):
    x, x_hat, w, h = small_problem
    ref = ops.calculate_w_new(x, x_hat, w, h)
    got = ops.calculate_w_new(
        torch.from_numpy(x).float(),
        torch.from_numpy(x_hat).float(),
        torch.from_numpy(w).float(),
        torch.from_numpy(h).float(),
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F32)


# ---------------------------------------------------------------------------
# End-to-end iteration parity (multiple multiplicative updates in a row)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_e2e_iteration_parity_numpy_vs_torch():
    """Run several multiplicative-update iterations on both backends
    and assert every intermediate W, H, and X̂ match. This is the
    real-deal safety net for a future ``backend=`` switch on
    ``factorization``.
    """
    rng = np.random.default_rng(42)
    I, J, K = 30, 20, 4
    n_iter = 25

    x = _random_nonneg((I, J), rng)
    w0 = _random_nonneg((I, K), rng)
    h0 = _random_nonneg((K, J), rng)

    # numpy track
    w_np, h_np = w0.copy(), h0.copy()
    # torch track
    w_t = torch.from_numpy(w0).double()
    h_t = torch.from_numpy(h0).double()
    x_t = torch.from_numpy(x).double()

    for _ in range(n_iter):
        xh_np = ops.calculate_x_hat(w_np, h_np)
        xh_t = ops.calculate_x_hat(w_t, h_t)
        np.testing.assert_allclose(xh_t.numpy(), xh_np, atol=ATOL_F64)

        # Update W on both tracks.
        w_np = ops.calculate_w_new(x, xh_np, w_np, h_np)
        w_t = ops.calculate_w_new(x_t, xh_t, w_t, h_t)
        np.testing.assert_allclose(w_t.numpy(), w_np, atol=ATOL_F64)

        # Recompute X̂ with the new W, then update H.
        xh_np = ops.calculate_x_hat(w_np, h_np)
        xh_t = ops.calculate_x_hat(w_t, h_t)
        h_np = ops.calculate_h_new(x, xh_np, w_np, h_np)
        h_t = ops.calculate_h_new(x_t, xh_t, w_t, h_t)
        np.testing.assert_allclose(h_t.numpy(), h_np, atol=ATOL_F64)
