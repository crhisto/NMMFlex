"""Parity tests for the vectorized alpha/beta and A/B update rules
in ``NMMFlex.ops``.

Each test compares the new vectorized op against the original
triple-loop implementation in ``factorization`` on the same fixed
random inputs. The vectorized form must match to float64 precision
(or float32 tolerance for the torch f32 path).

Coverage:
  - W update with beta == 0 (no Z coupling) and alpha_regularizer_w == 0
  - W update with beta != 0 (Z coupling active)
  - W update with alpha_regularizer_w != 0 (medecom_soft_binary)
  - W update with beta != 0 AND alpha_regularizer_w != 0
  - H update with alpha == 0
  - H update with alpha != 0 (Y coupling active)
  - A update
  - B update
  - Torch backend equivalence for each
"""

from __future__ import annotations

import numpy as np
import pytest

from NMMFlex import _backend as B
from NMMFlex import ops
from NMMFlex.factorization import factorization

torch = pytest.importorskip("torch") if B.has_torch() else None

ATOL_F64 = 1e-10
ATOL_F32 = 1e-4


def _rng_nonneg(shape, rng, low=0.1, high=1.0):
    return rng.uniform(low, high, size=shape)


@pytest.fixture
def alpha_beta_problem():
    """A consistent random multi-matrix problem for parity tests."""
    rng = np.random.default_rng(7)
    I, J, K = 14, 11, 4
    N = 6   # Y has shape (N, J)
    M = 8   # Z has shape (I, M)

    w = _rng_nonneg((I, K), rng)
    h = _rng_nonneg((K, J), rng)
    a = _rng_nonneg((N, K), rng)
    b = _rng_nonneg((K, M), rng)

    x = _rng_nonneg((I, J), rng)
    y = _rng_nonneg((N, J), rng)
    z = _rng_nonneg((I, M), rng)
    x_hat = w @ h
    y_hat = a @ h
    z_hat = w @ b

    return {
        'x': x, 'y': y, 'z': z,
        'x_hat': x_hat, 'y_hat': y_hat, 'z_hat': z_hat,
        'w': w, 'h': h, 'a': a, 'b': b,
    }


# ---------------------------------------------------------------------------
# W alpha/beta update
# ---------------------------------------------------------------------------

def test_w_alpha_beta_no_coupling_no_reg(alpha_beta_problem):
    p = alpha_beta_problem
    f = factorization()
    ref = f._calculate_w_new_extended_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        beta=0.0, z=p['z'], z_hat=p['z_hat'], b=p['b'],
        regularize_w=None, alpha_regularizer_w=0.0,
    )
    got = ops.calculate_w_new_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        beta=0.0, z=p['z'], z_hat=p['z_hat'], b=p['b'],
        alpha_regularizer_w=0.0,
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_w_alpha_beta_with_z_coupling(alpha_beta_problem):
    p = alpha_beta_problem
    f = factorization()
    ref = f._calculate_w_new_extended_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        beta=0.3, z=p['z'], z_hat=p['z_hat'], b=p['b'],
        regularize_w=None, alpha_regularizer_w=0.0,
    )
    got = ops.calculate_w_new_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        beta=0.3, z=p['z'], z_hat=p['z_hat'], b=p['b'],
        alpha_regularizer_w=0.0,
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_w_alpha_beta_with_regularizer(alpha_beta_problem):
    p = alpha_beta_problem
    f = factorization()
    ref = f._calculate_w_new_extended_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        beta=0.0, z=p['z'], z_hat=p['z_hat'], b=p['b'],
        regularize_w=None, alpha_regularizer_w=0.05,
    )
    got = ops.calculate_w_new_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        beta=0.0, z=p['z'], z_hat=p['z_hat'], b=p['b'],
        alpha_regularizer_w=0.05,
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_w_alpha_beta_full(alpha_beta_problem):
    p = alpha_beta_problem
    f = factorization()
    ref = f._calculate_w_new_extended_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        beta=0.4, z=p['z'], z_hat=p['z_hat'], b=p['b'],
        regularize_w=None, alpha_regularizer_w=0.05,
    )
    got = ops.calculate_w_new_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        beta=0.4, z=p['z'], z_hat=p['z_hat'], b=p['b'],
        alpha_regularizer_w=0.05,
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# H alpha/beta update
# ---------------------------------------------------------------------------

def test_h_alpha_beta_no_coupling(alpha_beta_problem):
    p = alpha_beta_problem
    f = factorization()
    ref = f._calculate_h_new_extended_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        alpha=0.0, y=p['y'], y_hat=p['y_hat'], a=p['a'],
    )
    got = ops.calculate_h_new_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        alpha=0.0, y=p['y'], y_hat=p['y_hat'], a=p['a'],
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_h_alpha_beta_with_y_coupling(alpha_beta_problem):
    p = alpha_beta_problem
    f = factorization()
    ref = f._calculate_h_new_extended_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        alpha=0.25, y=p['y'], y_hat=p['y_hat'], a=p['a'],
    )
    got = ops.calculate_h_new_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        alpha=0.25, y=p['y'], y_hat=p['y_hat'], a=p['a'],
    )
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# A and B updates
# ---------------------------------------------------------------------------

def test_a_new_matches_loop_reference(alpha_beta_problem):
    p = alpha_beta_problem
    f = factorization()
    ref = f._calculate_a_new_extended(p['y'], p['y_hat'], p['a'], p['h'])
    got = ops.calculate_a_new(p['y'], p['y_hat'], p['a'], p['h'])
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


def test_b_new_matches_loop_reference(alpha_beta_problem):
    p = alpha_beta_problem
    f = factorization()
    ref = f._calculate_b_new_extended(p['z'], p['z_hat'], p['b'], p['w'])
    got = ops.calculate_b_new(p['z'], p['z_hat'], p['b'], p['w'])
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# Torch parity for the alpha/beta family
# ---------------------------------------------------------------------------

def _to_t(arr, dtype=None):
    t = torch.from_numpy(arr)
    if dtype is None:
        dtype = torch.float64
    return t.to(dtype=dtype)


@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_w_alpha_beta_torch_matches_numpy(alpha_beta_problem):
    p = alpha_beta_problem
    ref = ops.calculate_w_new_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        beta=0.4, z=p['z'], z_hat=p['z_hat'], b=p['b'],
        alpha_regularizer_w=0.05,
    )
    got = ops.calculate_w_new_alpha_beta(
        _to_t(p['x']), _to_t(p['x_hat']),
        _to_t(p['w']), _to_t(p['h']),
        beta=0.4,
        z=_to_t(p['z']), z_hat=_to_t(p['z_hat']),
        b=_to_t(p['b']),
        alpha_regularizer_w=0.05,
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_h_alpha_beta_torch_matches_numpy(alpha_beta_problem):
    p = alpha_beta_problem
    ref = ops.calculate_h_new_alpha_beta(
        p['x'], p['x_hat'], p['w'], p['h'],
        alpha=0.25, y=p['y'], y_hat=p['y_hat'], a=p['a'],
    )
    got = ops.calculate_h_new_alpha_beta(
        _to_t(p['x']), _to_t(p['x_hat']),
        _to_t(p['w']), _to_t(p['h']),
        alpha=0.25,
        y=_to_t(p['y']), y_hat=_to_t(p['y_hat']),
        a=_to_t(p['a']),
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_a_new_torch_matches_numpy(alpha_beta_problem):
    p = alpha_beta_problem
    ref = ops.calculate_a_new(p['y'], p['y_hat'], p['a'], p['h'])
    got = ops.calculate_a_new(
        _to_t(p['y']), _to_t(p['y_hat']), _to_t(p['a']), _to_t(p['h'])
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)


@pytest.mark.skipif(not B.has_torch(), reason="torch not installed")
def test_b_new_torch_matches_numpy(alpha_beta_problem):
    p = alpha_beta_problem
    ref = ops.calculate_b_new(p['z'], p['z_hat'], p['b'], p['w'])
    got = ops.calculate_b_new(
        _to_t(p['z']), _to_t(p['z_hat']), _to_t(p['b']), _to_t(p['w'])
    ).numpy()
    np.testing.assert_allclose(got, ref, atol=ATOL_F64)
