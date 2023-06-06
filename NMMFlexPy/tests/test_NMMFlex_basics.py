import unittest
from os import getcwd

import numpy as np
import pandas as pd
import pandas.core.frame

from NMMFlex.NMMFlex import NMMFlex


class test_NMMFlex_basics(unittest.TestCase):
    """
    The test_NMMFlex_basics class is a TestCase subclass for testing the basics
    of NMMFlex.

    Attributes:

    bulk_methylation_matrix (numpy.ndarray or None): The input bulk methylation
        matrix for testing. Default is None.
    expected_proportions (numpy.ndarray or None): The expected proportions for
        the tests cases. Default is None.
    bulk_expression_matrix (numpy.ndarray or None): The input bulk expression
        matrix for testing. Default is None.
    data_expression_auxiliary_matrix_random (numpy.ndarray or None): The random
        data expression auxiliary matrix for testing. Default is None.
    dec (Deconvolution or None): The Deconvolution instance to be used for
        testing. Default is None.

    Notes:

    - All tests should be implemented as methods with names starting with
    'tests'.
    - Each tests method should use the assert methods from the unittest.
    TestCase class to check for various conditions.
    - The setUp method can be used to set up any state that is common to all
    tests methods.
    - The tearDown method can be used to clean up any resources after tests
    are run.
    """

    relative_path = ''
    bulk_methylation_matrix = None
    expected_proportions = None
    bulk_expression_matrix = None
    data_expression_auxiliary_matrix_random = None
    dec = None

    def __init__(self, relative_path):
        super(test_NMMFlex_basics, self).__init__(relative_path)
        # To allow the test cases to run from different configurations of main
        # path without problems
        if 'tests' in getcwd():
            self.relative_path = getcwd() + '/'
        else:
            self.relative_path = getcwd() + "/tests/"

    # This function will be called for each test.
    def setUp(self):
        # Let's import the data
        self._import_test_data()
        self.dec = NMMFlex()

    def tearDown(self):
        del self.bulk_methylation_matrix
        del self.expected_proportions
        del self.bulk_expression_matrix
        del self.data_expression_auxiliary_matrix_random
        del self.dec

    def _import_test_data(self):
        # Let's import the data
        self.bulk_methylation_matrix = pd.read_csv(
            self.relative_path +
            'data/meth_bulk_samples.filtered.markers.CpGs.500.csv',
            index_col=0)
        self.expected_proportions = pd.read_csv(
            self.relative_path +
            'data/expected.proportions.edec.markers.GpGs.500.csv',
            index_col=0)
        self.bulk_expression_matrix = pd.read_csv(
            self.relative_path +
            'data/gene_exp_mixtures.csv',
            index_col=0)
        self.data_expression_auxiliary_matrix_random = pd.DataFrame(
            data=np.random.rand(
                len(self.bulk_methylation_matrix.index),
                len(self.bulk_methylation_matrix.columns)),
            index=self.bulk_methylation_matrix.index.values,
            columns=self.bulk_methylation_matrix.columns.values)

    #  Multiplication of two matrices W and H
    def test_calculate_x_hat(self):
        w_test = np.array([[3, 6, 7, 3],
                           [5, 3, 0, 2],
                           [5, 2, 1, 2],
                           [8, 1, 0, 10]])  # k = 4
        h_test = np.array([[3, 6], [1, 3], [9, 1], [1, 1]])
        x_hat_expected = np.array([[81, 46], [20, 41], [28, 39], [35, 61]])

        x_hat = self.dec._calculate_x_hat(w_test, h_test)
        print(x_hat)
        np.testing.assert_array_equal(
            x_hat_expected, x_hat,
            'The X^ (x hat) matrix is not the one expected.')

    def test_calculate_x_hat_extended(self):
        w_test = np.array([[3, 6, 7, 3],
                           [5, 3, 0, 2],
                           [5, 2, 1, 2],
                           [8, 1, 0, 10]])  # k = 4
        h_test = np.array([[3, 6], [1, 3], [9, 1], [1, 1]])
        x_hat_expected = np.array([[81, 46], [20, 41], [28, 39], [35, 61]])

        x_hat = self.dec._calculate_x_hat_extended(w_test,
                                                   h_test).astype(float)
        print('X^ (x hat) matrix calculated with extended code:')
        print(x_hat)
        print('X^ (x hat) matrix calculated with dot function:')
        x_hat_dot_product = self.dec._calculate_x_hat(w_test,
                                                      h_test).astype(float)
        print(x_hat_dot_product)
        np.testing.assert_array_equal(
            x_hat_dot_product, x_hat,
            'The X^ (x hat) matrix is not equal to the dot prod".')
        np.testing.assert_array_equal(
            x_hat_expected, x_hat,
            'The X^ (x hat) matrix is not the one expected.')

    def test_calculate_w_new_extended(self):
        x_test = np.array([[3, 6], [5, 3], [5, 3], [5, 3]])
        x_hat_test = np.array([[3, 7], [6, 3], [8, 3], [3, 3]])

        w = np.array([[43, 4, 3, 3],
                      [1, 9, 2, 1],
                      [2, 9, 2, 1],
                      [3, 11, 7, 2]])
        w_new_expected = np.array([[38.9, 3.8, 2.8, 2.8],
                                   [0.9, 8.1, 1.8, 0.9],
                                   [1.8, 6.9, 1.5, 0.8],
                                   [3.7, 15.6, 9.9, 2.8]])

        h_test = np.array([[3, 6], [5, 3], [5, 3], [5, 3]])

        w_new = self.dec._calculate_w_new_extended(x_test, x_hat_test,
                                                   w, h_test)
        print("W_new matrix: ")
        print(w_new)
        # I want to see if I can get the matrix with a decimal.
        np.testing.assert_almost_equal(
            w_new_expected,
            w_new, 1,
            'The new w matrix is not the one expected.', )

    def test_proportion_constraint_h(self):
        h = np.array([[0.11692266, 0.49023124, 0.03157466],
                      [0.38518708, 0.08108722, 0.34163091],
                      [0.11876603, 0.26641129, 0.75638277],
                      [0.42219656, 0.17418068, 0.06863269]]).astype(float)
        h_expected = np.array(
            [[0.112094489, 0.4844611, 0.026351282],
             [0.369281275, 0.080132804, 0.285115101],
             [0.11386174, 0.263275565, 0.631254794],
             [0.404762496, 0.172130531, 0.057278823]]).astype(float)

        h_new = self.dec._proportion_constraint_h(h)
        h_new_sum = h_new.sum(axis=0).sum().astype(int)
        np.testing.assert_allclose(
            h_expected, h_new, 1e-7, 0,
            'The matrix H normalize is not the expected.')
        self.assertEqual(
            3, h_new_sum,
            'Each column sums 1 in the H matrix (proportion matrix)')

    def test_calculate_divergence_equal_matrix(self):
        k = 4
        w = np.array(np.random.rand(10, k))
        h = np.array(np.random.rand(k, 6))
        x = w.dot(h)
        x_hat = x * 1

        divergence_matrix = self.dec._calculate_divergence_generic(x,
                                                                   x_hat,
                                                                   False)
        print(divergence_matrix)
        divergence_value = np.sum(divergence_matrix)
        self.assertEqual(0.0, divergence_value,
                         'The divergence sum is not the expected.')

    def test_divergence_extended(self):
        k = 4
        w = np.array(np.random.rand(10, k))
        h = np.array(np.random.rand(k, 6))
        x = w.dot(h)
        x_hat = x * 1.25

        divergence_matrix_dot_method = \
            self.dec._calculate_divergence_generic(x, x_hat,
                                                   False).astype(float)
        divergence_matrix_extended_method = \
            self.dec._calculate_divergence_extended(x, x_hat).astype(float)

        np.testing.assert_array_equal(
            divergence_matrix_dot_method,
            divergence_matrix_extended_method,
            'The matrix created with dot method is different to the one ' +
            'created with the extended method')

    def test_normalization(self):
        matrix_quantile_normalization = self.dec.normalization(
            matrix=self.bulk_methylation_matrix,
            normalization_type='quantile_norm')

        matrix_min_max = self.dec.normalization(
            matrix=matrix_quantile_normalization,
            normalization_type='norm_zero_min_max')

        matrix_quantile_normalization_min_max = self.dec.normalization(
            matrix=self.bulk_methylation_matrix,
            normalization_type='quantile_norm_min_max')

        print(type(matrix_quantile_normalization))
        print("pandas.core.frame.DataFrame" ==
              type(matrix_quantile_normalization))
        print(isinstance(matrix_quantile_normalization,
                         pandas.core.frame.DataFrame))


if __name__ == '__main__':
    unittest.main()
