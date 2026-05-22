"""DecoFlex compatibility smoke test.

DecoFlex (https://github.com/crhisto/DecoFlex) is the downstream R
consumer of this package. It talks to us through ``reticulate``:

    NMMFlex <- reticulate::import('NMMFlex')
    NMMFlex_factorization <- NMMFlex$factorization()
    deco_result <- NMMFlex_factorization$run_deconvolution_multiple(
        x_matrix = ..., y_matrix = NULL, z_matrix = NULL,
        k = ..., alpha = ..., beta = ...,
        delta_threshold = ..., max_iterations = ...,
        proportion_constraint_h = TRUE,
        fixed_w = ..., verbose = ...
    )
    # then reads many attributes off deco_result via $ accessor

These tests freeze the surface DecoFlex relies on. If anything here
fails, DecoFlex is broken too. The contract:

  1. ``import NMMFlex`` works (matches reticulate::import).
  2. ``NMMFlex.factorization`` and ``NMMFlex.grid_search`` are public.
  3. ``factorization()`` constructs with no positional args.
  4. ``run_deconvolution_multiple`` accepts the exact kwargs DecoFlex
     passes.
  5. After the call, the instance exposes the 27 attributes DecoFlex
     reads in ``trans_mult_deco_to_R``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


# The exact attribute set DecoFlex's trans_mult_deco_to_R() reads off
# the returned object. Captured from DecoFlexR/R/NMMFlex_wrapper.R as
# of the DecoFlex commit referenced in the repo README.
DECOFLEX_RESULT_ATTRIBUTES = (
    'x', 'y', 'z',
    'x_hat', 'y_hat', 'z_hat',
    'w', 'h',
    'a', 'b',
    'is_x_sparse', 'is_y_sparse', 'is_z_sparse', 'is_model_sparse',
    'initialized_w', 'initialized_h', 'initialized_a', 'initialized_b',
    'iterations', 'divergence_value', 'delta_divergence_value',
    'running_info',
    'alpha', 'beta', 'alpha_regularizer_w',
)


# ---------------------------------------------------------------------------
# Import-surface contract
# ---------------------------------------------------------------------------

def test_package_imports_as_NMMFlex():
    """reticulate::import('NMMFlex') is what DecoFlex calls. The
    package must be importable under that exact name."""
    pkg = importlib.import_module('NMMFlex')
    assert pkg is not None


def test_factorization_class_available_on_package():
    """DecoFlex accesses NMMFlex$factorization() — i.e. it reaches in
    via the package namespace, not via the submodule."""
    pkg = importlib.import_module('NMMFlex')
    assert hasattr(pkg, 'factorization'), \
        "NMMFlex.factorization must be exported (DecoFlex calls NMMFlex$factorization)"


def test_grid_search_class_available_on_package():
    pkg = importlib.import_module('NMMFlex')
    assert hasattr(pkg, 'grid_search'), \
        "NMMFlex.grid_search must be exported (DecoFlex calls NMMFlex$grid_search)"


# ---------------------------------------------------------------------------
# Construction contract
# ---------------------------------------------------------------------------

def test_factorization_constructs_with_no_args():
    """DecoFlex constructs factorization() with no positional args.
    The new backend/device/dtype kwargs added in Tier 2 must all
    have defaults so this keeps working."""
    pkg = importlib.import_module('NMMFlex')
    f = pkg.factorization()
    assert f is not None


def test_factorization_default_backend_is_numpy():
    """DecoFlex doesn't know about the backend kwarg; its calls must
    keep landing on the numpy backend so behaviour matches what users
    have today."""
    pkg = importlib.import_module('NMMFlex')
    f = pkg.factorization()
    assert f.backend == 'numpy'


# ---------------------------------------------------------------------------
# run_deconvolution_multiple call signature contract
# ---------------------------------------------------------------------------

def test_run_deconvolution_multiple_accepts_decoflex_kwargs():
    """Mirrors the exact call shape DecoFlex makes in
    NMMFlex_wrapper.R::run_complete_deconvolution -- specifically the
    DecoFlex 'simple' path (alpha=beta=0, no Y/Z, fixed W reference).

    The contract here is purely about the keyword names and their
    accepted shapes; we are not validating the numerical output (the
    factorization tests cover that). We just need the call to return
    a populated factorization instance.
    """
    pkg = importlib.import_module('NMMFlex')
    f = pkg.factorization()

    rng = np.random.default_rng(0)
    I, J, K = 12, 5, 3
    # DecoFlex always wraps its inputs with data.frame(...) before
    # handing them to reticulate, so the Python side sees pd.DataFrame.
    bulk_x = pd.DataFrame(rng.uniform(0.1, 1.0, size=(I, J)))
    references_w = pd.DataFrame(rng.uniform(0.1, 1.0, size=(I, K)))

    result = f.run_deconvolution_multiple(
        x_matrix=bulk_x,
        y_matrix=None,
        z_matrix=None,
        k=K,
        alpha=0.0,
        beta=0.0,
        delta_threshold=1e-6,
        max_iterations=10,
        proportion_constraint_h=True,
        fixed_w=references_w,
        verbose=False,
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Result-attribute contract
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_factorization():
    """One small fit shared by all attribute-access tests so we don't
    re-run the loop per test."""
    pkg = importlib.import_module('NMMFlex')
    f = pkg.factorization()

    rng = np.random.default_rng(1)
    I, J, K = 10, 4, 2
    bulk_x = pd.DataFrame(rng.uniform(0.1, 1.0, size=(I, J)))
    references_w = pd.DataFrame(rng.uniform(0.1, 1.0, size=(I, K)))

    f.run_deconvolution_multiple(
        x_matrix=bulk_x,
        y_matrix=None,
        z_matrix=None,
        k=K,
        alpha=0.0,
        beta=0.0,
        delta_threshold=1e-6,
        max_iterations=5,
        proportion_constraint_h=True,
        fixed_w=references_w,
        verbose=False,
    )
    return f


@pytest.mark.parametrize("attr", DECOFLEX_RESULT_ATTRIBUTES)
def test_decoflex_reads_attribute_after_fit(fitted_factorization, attr):
    """Each attribute DecoFlex's ``trans_mult_deco_to_R`` reads must
    exist on the instance. The value can be None (DecoFlex tolerates
    None for unused matrices like y/z); what must not happen is an
    AttributeError, which would crash the R wrapper.
    """
    assert hasattr(fitted_factorization, attr), (
        f"DecoFlex reads `{attr}` off the result of "
        f"run_deconvolution_multiple. The attribute is missing, which "
        f"would break the R wrapper's trans_mult_deco_to_R."
    )
