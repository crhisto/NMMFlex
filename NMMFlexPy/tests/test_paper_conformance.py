"""Conformance tests against Takeuchi et al. 2013 (NMMF paper).

These tests aren't about parity with the legacy loop implementation
(``test_ops_alpha_beta.py`` covers that). They check that the
*algorithm* honours what the paper proves:

  - Multiplicative-update theorem: D is monotonically non-increasing
    over iterations.
  - Eq. 6: alpha = beta = 0 reduces NMMF to plain NMF.
  - The alpha / beta coupling actually does something.
  - With noise-free synthetic data the reconstruction can be
    recovered to high accuracy.

Tests are ordered fastest-narrowest first, slowest-end-to-end last
so a regression gives the cheapest signal first.

Paper reference:
    Takeuchi, Ishiguro, Kimura, Sawada. "Non-Negative Multiple
    Matrix Factorization". IJCAI 2013.
    https://www.ijcai.org/Proceedings/13/Papers/254.pdf
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from NMMFlex import ops
from NMMFlex.factorization import factorization


def _rng_nonneg(shape, rng, low=0.1, high=1.0):
    return rng.uniform(low, high, size=shape)


# ---------------------------------------------------------------------------
# Test 1 -- unit, fastest: with all coupling disabled, the alpha/beta
# ops must reduce to the plain ops (paper's Eq. 6 at the function level).
# ---------------------------------------------------------------------------

def test_alpha_beta_zero_couplings_match_basic_ops():
    """When beta = 0, no Z/B, no regularizer: calculate_w_new_alpha_beta
    must equal calculate_w_new. Likewise for H with alpha = 0.
    Pinned at the ops-module level so a future refactor of the W
    update doesn't accidentally leak a coupling-on-by-default bug
    into the simple-NMF path.
    """
    rng = np.random.default_rng(0)
    I, J, K = 12, 9, 3
    w = _rng_nonneg((I, K), rng)
    h = _rng_nonneg((K, J), rng)
    x = _rng_nonneg((I, J), rng)
    x_hat = w @ h

    w_basic = ops.calculate_w_new(x, x_hat, w, h)
    w_alpha_beta = ops.calculate_w_new_alpha_beta(
        x, x_hat, w, h,
        beta=0.0, z=None, z_hat=None, b=None,
        alpha_regularizer_w=0.0,
    )
    np.testing.assert_allclose(w_alpha_beta, w_basic, atol=1e-12)

    h_basic = ops.calculate_h_new(x, x_hat, w, h)
    h_alpha_beta = ops.calculate_h_new_alpha_beta(
        x, x_hat, w, h,
        alpha=0.0, y=None, y_hat=None, a=None,
    )
    np.testing.assert_allclose(h_alpha_beta, h_basic, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 2 -- integration: multiplicative-update theorem -- divergence
# must be monotonically non-increasing across iterations. Paper Eq. 25
# proves F is biconvex w.r.t. each factor matrix, so this is a hard
# guarantee, not a heuristic. The strictest sanity check on the whole
# iteration loop.
# ---------------------------------------------------------------------------

def test_divergence_non_increasing_during_iterations():
    rng = np.random.default_rng(1)
    I, J, K = 30, 15, 4
    x = pd.DataFrame(_rng_nonneg((I, J), rng))

    f = factorization()
    f.run_deconvolution_multiple(
        x_matrix=x, y_matrix=None, z_matrix=None, k=K,
        gamma=1.0, alpha=0.0, beta=0.0,
        delta_threshold=1e-30,           # force max_iterations
        max_iterations=40,
        proportion_constraint_h=True,
        verbose=False, print_limit=10**9,
    )

    divs = f.running_info['divergence_value'].values.astype(float)
    diffs = np.diff(divs)
    # Tiny float wobble can let an "equal" iteration drift up by an
    # eps; insist on no rise larger than 1e-9 in absolute terms.
    assert (diffs <= 1e-9).all(), (
        "Divergence increased between consecutive iterations; "
        f"max upward step = {diffs.max():.3e}, sequence = {divs.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 3 -- integration: paper Eq. 6. alpha = beta = 0, no Y/Z =>
# NMMF degenerates to NMF. With identical initial W/H, the legacy
# ``run_deconvolution`` (which IS the simple NMF path) and the
# multi-matrix entry point must converge to the same W and H.
# ---------------------------------------------------------------------------

def test_alpha_beta_zero_matches_simple_run_deconvolution():
    rng = np.random.default_rng(2)
    I, J, K = 15, 8, 3
    x_np = _rng_nonneg((I, J), rng)
    w0 = _rng_nonneg((I, K), rng)
    h0 = _rng_nonneg((K, J), rng)

    # Simple-NMF run via run_deconvolution; injects our fixed seeds
    # in place of the random init so the two paths start identically.
    f_simple = factorization()

    def _init_simple(size_rows, size_columns, **_):
        if (size_rows, size_columns) == w0.shape:
            return w0.copy()
        if (size_rows, size_columns) == h0.shape:
            return h0.copy()
        raise AssertionError(
            f"unexpected init shape {(size_rows, size_columns)}"
        )

    # Simple path initialises W/H via factorization._random_like;
    # override that on this instance with our fixed seeds so the two
    # paths start from identical W/H.

    def _stub_random_like(shape):
        if shape == w0.shape:
            return w0.copy()
        if shape == h0.shape:
            return h0.copy()
        raise AssertionError(f"unexpected random_like shape {shape}")

    f_simple._random_like = _stub_random_like  # noqa: SLF001
    f_simple.run_deconvolution(
        x_matrix=x_np, k=K,
        delta_threshold=1e-30, max_iterations=25, print_limit=10**9,
    )

    # NMMF multi-matrix run with alpha = beta = 0 and the same
    # initial W/H. proportion_constraint_h is on in both paths.
    f_multi = factorization()
    f_multi._initialize_matrix = _init_simple  # noqa: SLF001
    f_multi.run_deconvolution_multiple(
        x_matrix=pd.DataFrame(x_np),
        y_matrix=None, z_matrix=None, k=K,
        gamma=1.0, alpha=0.0, beta=0.0,
        delta_threshold=1e-30, max_iterations=25,
        proportion_constraint_h=True,
        verbose=False, print_limit=10**9,
    )

    np.testing.assert_allclose(
        np.asarray(f_multi.w.values), f_simple.w, atol=1e-9,
        err_msg="NMMF with alpha=beta=0 must match plain NMF on W"
    )
    np.testing.assert_allclose(
        np.asarray(f_multi.h.values), f_simple.h, atol=1e-9,
        err_msg="NMMF with alpha=beta=0 must match plain NMF on H"
    )


# ---------------------------------------------------------------------------
# Test 4 -- integration: alpha-coupling actually moves the solution.
# If we hand the multi-matrix path an informative Y that genuinely
# shares H with X, turning the coupling on (alpha > 0) must produce
# H values different from the alpha = 0 baseline. Otherwise the
# coupling term is silently a no-op.
# ---------------------------------------------------------------------------

def test_alpha_coupling_changes_solution():
    rng = np.random.default_rng(3)
    I, J, K, N = 20, 10, 3, 8
    # Ground-truth factorization so X and Y genuinely share H.
    w_true = _rng_nonneg((I, K), rng)
    h_true = _rng_nonneg((K, J), rng)
    a_true = _rng_nonneg((N, K), rng)

    x = pd.DataFrame(w_true @ h_true)
    y = pd.DataFrame(a_true @ h_true)

    # Same initial W/H/A so the only difference between the two runs
    # is the alpha coupling.
    w0 = _rng_nonneg((I, K), rng)
    h0 = _rng_nonneg((K, J), rng)
    a0 = _rng_nonneg((N, K), rng)

    def _init(size_rows, size_columns, **_):
        if (size_rows, size_columns) == w0.shape:
            return w0.copy()
        if (size_rows, size_columns) == h0.shape:
            return h0.copy()
        if (size_rows, size_columns) == a0.shape:
            return a0.copy()
        raise AssertionError(
            f"unexpected init shape {(size_rows, size_columns)}"
        )

    common = dict(
        x_matrix=x, y_matrix=y, z_matrix=None, k=K,
        gamma=1.0, beta=0.0,
        delta_threshold=1e-30, max_iterations=30,
        proportion_constraint_h=True,
        verbose=False, print_limit=10**9,
    )

    f_no = factorization()
    f_no._initialize_matrix = _init  # noqa: SLF001
    f_no.run_deconvolution_multiple(alpha=0.0, **common)

    f_yes = factorization()
    f_yes._initialize_matrix = _init  # noqa: SLF001
    f_yes.run_deconvolution_multiple(alpha=0.5, **common)

    h_no = np.asarray(f_no.h.values)
    h_yes = np.asarray(f_yes.h.values)

    # The two H matrices must NOT be numerically identical -- if they
    # are, the alpha coupling has been silently dropped.
    diff = np.linalg.norm(h_yes - h_no) / np.linalg.norm(h_no)
    assert diff > 1e-3, (
        f"alpha=0 and alpha=0.5 produced effectively identical H "
        f"(relative diff {diff:.3e}); the Y-coupling term is not "
        f"contributing to the update."
    )


# ---------------------------------------------------------------------------
# Test 5 -- integration, slowest: noise-free recovery. With X = W H
# exactly, NMMF with k = K should drive the reconstruction error
# down toward zero. Validates that the update rules actually solve
# the optimisation, not just that they shuffle numbers around.
# Identifiability up to permutation/scale means we check the
# *reconstruction*, not the factors themselves.
# ---------------------------------------------------------------------------

def test_recovery_on_noise_free_synthetic_data():
    # Multiplicative-update NMF with KL divergence converges slowly
    # and is heavily seed-dependent. To make this test reliable
    # we pin the initialisation (so the result is deterministic)
    # and run for many iterations. The threshold (10%) is loose
    # enough to tolerate a single-init local minimum, tight enough
    # that a structural bug in the update rules would blow past it.
    rng = np.random.default_rng(4)
    I, J, K = 25, 12, 3
    w_true = _rng_nonneg((I, K), rng)
    h_true = _rng_nonneg((K, J), rng)
    x_np = w_true @ h_true
    x = pd.DataFrame(x_np)

    # Pinned init: random but seeded separately so this test is
    # deterministic across CI runs.
    init_rng = np.random.default_rng(42)
    w0 = _rng_nonneg((I, K), init_rng)
    h0 = _rng_nonneg((K, J), init_rng)

    f = factorization()

    def _fixed_init(size_rows, size_columns, **_):
        if (size_rows, size_columns) == w0.shape:
            return w0.copy()
        if (size_rows, size_columns) == h0.shape:
            return h0.copy()
        raise AssertionError(
            f"unexpected init shape {(size_rows, size_columns)}"
        )

    f._initialize_matrix = _fixed_init  # noqa: SLF001
    f.run_deconvolution_multiple(
        x_matrix=x, y_matrix=None, z_matrix=None, k=K,
        gamma=1.0, alpha=0.0, beta=0.0,
        delta_threshold=1e-30,           # let max_iterations cap it
        max_iterations=20000,
        proportion_constraint_h=False,   # don't rescale H; we are
                                          # measuring reconstruction only
        verbose=False, print_limit=10**9,
    )

    w_hat = np.asarray(f.w.values)
    h_hat = np.asarray(f.h.values)
    x_reconstruction = w_hat @ h_hat
    rel_err = (
        np.linalg.norm(x_np - x_reconstruction) / np.linalg.norm(x_np)
    )
    assert rel_err < 0.10, (
        f"reconstruction error {rel_err:.3e} is too high for "
        f"noise-free synthetic data (k = K = {K}). NMMF should "
        f"recover X = W H to within ~10 percent at this seed."
    )
