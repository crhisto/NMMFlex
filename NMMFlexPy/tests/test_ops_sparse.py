"""Parity tests for the sparse / NaN-tolerant update rules.

The vectorized sparse wrappers in ``NMMFlex.ops`` should produce
exactly the same results as the legacy
``_calculate_*_extended_*_sparse`` loops on:
  - scipy.sparse (csr_matrix) inputs
  - dense inputs that contain NaN values (which the loop versions
    explicitly skip)

We cover three families of inputs per op:
  1. Plain dense, no NaNs                 -> should also match the
     dense ops above; NaN-handling is a no-op.
  2. Dense with NaNs scattered in h/w/b   -> tests the NaN-skip
     semantics.
  3. csr_matrix for the observed matrix   -> tests the sparse-input
     coercion.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from NMMFlex import ops
from NMMFlex.factorization import factorization

ATOL_F64 = 1e-10


def _rng_nonneg(shape, rng, low=0.1, high=1.0):
    return rng.uniform(low, high, size=shape)


@pytest.fixture
def problem():
    rng = np.random.default_rng(7)
    I, J, K = 14, 11, 4
    N = 6
    M = 8
    w = _rng_nonneg((I, K), rng)
    h = _rng_nonneg((K, J), rng)
    a = _rng_nonneg((N, K), rng)
    b = _rng_nonneg((K, M), rng)
    x = _rng_nonneg((I, J), rng)
    y = _rng_nonneg((N, J), rng)
    z = _rng_nonneg((I, M), rng)
    return {
        'x': x, 'y': y, 'z': z,
        'x_hat': w @ h, 'y_hat': a @ h, 'z_hat': w @ b,
        'w': w, 'h': h, 'a': a, 'b': b,
    }


def _with_nans(arr, indices):
    arr = arr.copy()
    for idx in indices:
        arr[idx] = np.nan
    return arr


# ---------------------------------------------------------------------------
# W sparse alpha/beta -- 3 input families
# ---------------------------------------------------------------------------

def test_w_alpha_beta_sparse_plain_dense_matches_loop(problem):
    p = problem
    f = factorization()
    ref = f._calculate_w_new_extended_alpha_beta_sparse(
        p['x'], p['x_hat'], p['w'], p['h'],
        beta=0.3, z=p['z'], z_hat=p['z_hat'], b=p['b'],
        regularize_w=None, alpha_regularizer_w=0.05,
    )
    got = ops.calculate_w_new_alpha_beta_sparse(
        p['x'], p['x_hat'], p['w'], p['h'],
        beta=0.3, z=p['z'], z_hat=p['z_hat'], b=p['b'],
        alpha_regularizer_w=0.05,
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_w_alpha_beta_sparse_with_nans(problem):
    p = problem
    # Sprinkle NaN into h, w, b -- the legacy code skips these terms.
    h = _with_nans(p['h'], [(0, 1), (2, 5)])
    w = _with_nans(p['w'], [(0, 0), (3, 2)])
    b = _with_nans(p['b'], [(1, 4)])

    f = factorization()
    ref = f._calculate_w_new_extended_alpha_beta_sparse(
        p['x'], p['x_hat'], w, h,
        beta=0.3, z=p['z'], z_hat=p['z_hat'], b=b,
        regularize_w=None, alpha_regularizer_w=0.05,
    )
    got = ops.calculate_w_new_alpha_beta_sparse(
        p['x'], p['x_hat'], w, h,
        beta=0.3, z=p['z'], z_hat=p['z_hat'], b=b,
        alpha_regularizer_w=0.05,
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_w_alpha_beta_sparse_with_csr_input(problem):
    p = problem
    # Force x and z into csr_matrix the same way the legacy method
    # does at the top of its body.
    x_csr = csr_matrix(p['x'])
    z_csr = csr_matrix(p['z'])

    f = factorization()
    ref = f._calculate_w_new_extended_alpha_beta_sparse(
        x_csr, p['x_hat'], p['w'], p['h'],
        beta=0.4, z=z_csr, z_hat=p['z_hat'], b=p['b'],
        regularize_w=None, alpha_regularizer_w=0.0,
    )
    got = ops.calculate_w_new_alpha_beta_sparse(
        x_csr, p['x_hat'], p['w'], p['h'],
        beta=0.4, z=z_csr, z_hat=p['z_hat'], b=p['b'],
        alpha_regularizer_w=0.0,
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# H sparse alpha/beta
# ---------------------------------------------------------------------------

def test_h_alpha_beta_sparse_plain_dense(problem):
    p = problem
    f = factorization()
    ref = f._calculate_h_new_extended_alpha_beta_sparse(
        p['x'], p['x_hat'], p['w'], p['h'],
        alpha=0.2, y=p['y'], y_hat=p['y_hat'], a=p['a'],
    )
    got = ops.calculate_h_new_alpha_beta_sparse(
        p['x'], p['x_hat'], p['w'], p['h'],
        alpha=0.2, y=p['y'], y_hat=p['y_hat'], a=p['a'],
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_h_alpha_beta_sparse_with_nans(problem):
    p = problem
    h = _with_nans(p['h'], [(1, 0), (3, 7)])
    w = _with_nans(p['w'], [(5, 1)])
    a = _with_nans(p['a'], [(0, 2)])

    f = factorization()
    ref = f._calculate_h_new_extended_alpha_beta_sparse(
        p['x'], p['x_hat'], w, h,
        alpha=0.2, y=p['y'], y_hat=p['y_hat'], a=a,
    )
    got = ops.calculate_h_new_alpha_beta_sparse(
        p['x'], p['x_hat'], w, h,
        alpha=0.2, y=p['y'], y_hat=p['y_hat'], a=a,
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_h_alpha_beta_sparse_with_csr_input(problem):
    p = problem
    x_csr = csr_matrix(p['x'])
    y_csr = csr_matrix(p['y'])
    f = factorization()
    ref = f._calculate_h_new_extended_alpha_beta_sparse(
        x_csr, p['x_hat'], p['w'], p['h'],
        alpha=0.3, y=y_csr, y_hat=p['y_hat'], a=p['a'],
    )
    got = ops.calculate_h_new_alpha_beta_sparse(
        x_csr, p['x_hat'], p['w'], p['h'],
        alpha=0.3, y=y_csr, y_hat=p['y_hat'], a=p['a'],
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# A sparse / B sparse
# ---------------------------------------------------------------------------

def test_a_sparse_plain_dense(problem):
    p = problem
    f = factorization()
    ref = f._calculate_a_new_extended_sparse(
        p['y'], p['y_hat'], p['a'], p['h']
    )
    got = ops.calculate_a_new_sparse(
        p['y'], p['y_hat'], p['a'], p['h']
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_a_sparse_with_csr_and_nans(problem):
    p = problem
    y_csr = csr_matrix(p['y'])
    h = _with_nans(p['h'], [(0, 0), (2, 3)])
    a = _with_nans(p['a'], [(1, 1)])

    f = factorization()
    ref = f._calculate_a_new_extended_sparse(y_csr, p['y_hat'], a, h)
    got = ops.calculate_a_new_sparse(y_csr, p['y_hat'], a, h)
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_b_sparse_plain_dense(problem):
    p = problem
    f = factorization()
    ref = f._calculate_b_new_extended_sparse(
        p['z'], p['z_hat'], p['b'], p['w']
    )
    got = ops.calculate_b_new_sparse(
        p['z'], p['z_hat'], p['b'], p['w']
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_b_sparse_with_csr_and_nans(problem):
    p = problem
    z_csr = csr_matrix(p['z'])
    w = _with_nans(p['w'], [(2, 1)])
    b = _with_nans(p['b'], [(3, 0)])

    f = factorization()
    ref = f._calculate_b_new_extended_sparse(z_csr, p['z_hat'], b, w)
    got = ops.calculate_b_new_sparse(z_csr, p['z_hat'], b, w)
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)
