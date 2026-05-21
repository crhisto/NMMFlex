"""Parity tests for the vectorized numpy and PyTorch implementations of
the NMMF multiplicative update rules.

Each test compares three implementations of the same operation:
  1. The original triple-loop reference in factorization.py
  2. The vectorized numpy port in NMMFlex.torch_backend
  3. The torch port in NMMFlex.torch_backend (skipped if torch missing)

They should all agree to within float64 precision on well-conditioned
inputs, and to within float32 precision when the torch path uses float32.
"""

from __future__ import annotations

import numpy as np
import pytest

from NMMFlex.factorization import factorization
from NMMFlex import torch_backend as tb

torch = pytest.importorskip("torch") if tb.has_torch() else None

ATOL_F64 = 1e-10
ATOL_F32 = 1e-4


def _random_nonneg(shape, rng, low=0.1, high=1.0):
    """Random positive matrix (no zeros) so KL divergence is finite."""
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
# x_hat (matrix product)
# ---------------------------------------------------------------------------

def test_x_hat_loop_vs_vectorized(small_problem):
    _, _, w, h = small_problem
    f = factorization()
    ref = f._calculate_x_hat_extended(w, h)
    got = tb.calculate_x_hat_np(w, h)
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


@pytest.mark.skipif(not tb.has_torch(), reason="torch not installed")
def test_x_hat_torch_matches_numpy(small_problem):
    _, _, w, h = small_problem
    ref = tb.calculate_x_hat_np(w, h)
    got = tb.calculate_x_hat_torch(
        torch.from_numpy(w).double(), torch.from_numpy(h).double()
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# W update
# ---------------------------------------------------------------------------

def test_w_update_loop_vs_vectorized(small_problem):
    x, x_hat, w, h = small_problem
    f = factorization()
    ref = f._calculate_w_new_extended(x, x_hat, w, h)
    got = tb.calculate_w_new_np(x, x_hat, w, h)
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


@pytest.mark.skipif(not tb.has_torch(), reason="torch not installed")
def test_w_update_torch_matches_numpy(small_problem):
    x, x_hat, w, h = small_problem
    ref = tb.calculate_w_new_np(x, x_hat, w, h)
    got = tb.calculate_w_new_torch(
        torch.from_numpy(x).double(),
        torch.from_numpy(x_hat).double(),
        torch.from_numpy(w).double(),
        torch.from_numpy(h).double(),
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


@pytest.mark.skipif(not tb.has_torch(), reason="torch not installed")
def test_w_update_torch_float32_within_tolerance(small_problem):
    x, x_hat, w, h = small_problem
    ref = tb.calculate_w_new_np(x, x_hat, w, h)
    got = tb.calculate_w_new_torch(
        torch.from_numpy(x).float(),
        torch.from_numpy(x_hat).float(),
        torch.from_numpy(w).float(),
        torch.from_numpy(h).float(),
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F32)


# ---------------------------------------------------------------------------
# H update
# ---------------------------------------------------------------------------

def test_h_update_loop_vs_vectorized(small_problem):
    x, x_hat, w, h = small_problem
    f = factorization()
    # proportion_constraint=False so we compare the raw multiplicative
    # update without the post-hoc renormalization step.
    ref = f._calculate_h_new_extended(x, x_hat, w, h, proportion_constraint=False)
    got = tb.calculate_h_new_np(x, x_hat, w, h)
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


@pytest.mark.skipif(not tb.has_torch(), reason="torch not installed")
def test_h_update_torch_matches_numpy(small_problem):
    x, x_hat, w, h = small_problem
    ref = tb.calculate_h_new_np(x, x_hat, w, h)
    got = tb.calculate_h_new_torch(
        torch.from_numpy(x).double(),
        torch.from_numpy(x_hat).double(),
        torch.from_numpy(w).double(),
        torch.from_numpy(h).double(),
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def test_divergence_loop_vs_vectorized(small_problem):
    x, x_hat, _, _ = small_problem
    f = factorization()
    ref = f._calculate_divergence_extended(x, x_hat)
    got = tb.calculate_divergence_np(x, x_hat)
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_safe_divide_handles_zero_denominator():
    """factorization.division returns 0 on zero-denominator; the
    vectorized versions must do the same."""
    rng = np.random.default_rng(1)
    I, J, K = 5, 4, 2
    w = _random_nonneg((I, K), rng)
    h = _random_nonneg((K, J), rng)
    x = _random_nonneg((I, J), rng)
    x_hat = w @ h
    # Inject zeros into x_hat to trigger the safe-divide path.
    x_hat[0, 0] = 0.0
    x_hat[2, 3] = 0.0

    f = factorization()
    ref = f._calculate_w_new_extended(x, x_hat, w, h)
    got = tb.calculate_w_new_np(x, x_hat, w, h)
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)

    assert np.isfinite(got).all(), "vectorized W update produced non-finite values"
