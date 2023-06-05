import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from NMMFlex.NMMFlex import NMMFlex


class test_NMMFlex_sparseness_core(unittest.TestCase):
    w = None
    h = None
    x = None
    y = None
    z = None

    a = None

    x_sparse = None
    y_sparse = None
    z_sparse = None

    def setUp(self):
        self.dec = NMMFlex()
        self._create_sparce_matrix_setup()

    def tearDown(self):
        del self.dec

    def _create_sparce_matrix_setup(self):
        # Just in case I need it in future tests.
        # k = 4
        w_sparse = np.array([[np.nan, 6, 0, 3], [0, 3, np.nan, 2], [5, 0, 1, 2], [8, 1, 0, np.nan], [8, 1, 0, np.nan],
                             [np.nan, 6, 0, 3], [0, 3, np.nan, 2], [5, 0, 1, 2], [8, 1, 0, np.nan], [8, 1, 0, np.nan]])
        h_sparse = np.array(
            [[np.nan, 6, 0, 3, 5, 5], [0, 3, np.nan, 2, 8, 9], [5, 0, 1, 2, 9, 0], [8, 1, 0, np.nan, 3, 2]])

        # k = 4 and 6 samples
        self.w = np.array([[1, 6, 0, 3], [0, 8, 1, 2], [5, 0, 1, 2], [8, 1, 0, 7], [8, 1, 0, 9],
                           [2, 6, 0, 9], [0, 2, 5, 2], [2, 0, 9, 9], [2, 0, 0, 1], [5, 1, 6, 10]])
        self.h = np.array([[3, 6, 0, 3, 5, 5], [0, 3, 6, 2, 8, 9], [5, 0, 1, 2, 9, 0], [8, 1, 0, 11, 3, 2]])

        self.a = np.array([[2, 3, 6, 7], [0, 0, 6, 2], [2, 0, 1, 4], [6, 1, 0, 2],
                           [5, 6, 0, 5], [0, 1, 1, 2], [1, 0, 2, 3], [7, 1, 0, 5]])

        self.b = np.array([[2, 3, 6, 7, 0, 1], [0, 0, 6, 2, 1, 4], [2, 0, 1, 4, 3, 6], [6, 1, 0, 2, 0, 6]])

        # Let's do a dot product with sparse matrices
        self.x = self.w.dot(self.h)
        self.y = self.a.dot(self.h)
        self.z = self.w.dot(self.b)

        # Now I have to convert the matrix to csr_matrix.
        self.x_sparse = csr_matrix(self.x.astype(float))
        self.y_sparse = csr_matrix(self.y.astype(float))
        self.z_sparse = csr_matrix(self.z.astype(float))

        # Now let's put zeros and nulls in the reads
        rest_x = self.x_sparse[0, 1] + self.x_sparse[1, 3] + self.x_sparse[5, 5]
        print('In X sparse the sum of values missing is: ', rest_x)

        # Initialization of x to be sparse with nulls because it is already sparce in zeros.
        self.x_sparse[0, 1] = np.nan
        self.x_sparse[1, 3] = np.nan
        self.x_sparse[5, 5] = np.nan

        # Now let's put zeros and nulls in the reads
        rest_y = self.y_sparse[0, 2] + self.y_sparse[1, 2] + self.y_sparse[5, 2]
        print('In Y sparse the sum of values missing is: ', rest_y)

        # Initialization of x to be sparse with nulls because it is already sparce in zeros.
        self.y_sparse[0, 2] = np.nan
        self.y_sparse[1, 2] = np.nan
        self.y_sparse[5, 2] = np.nan

        # Now let's put zeros and nulls in the reads
        rest_z = self.z_sparse[0, 3] + self.z_sparse[1, 2] + self.z_sparse[2, 0]
        print('In Z sparse the sum of values missing is: ', rest_z)

        # Initialization of x to be sparse with nulls because it is already sparce in zeros.
        self.z_sparse[0, 3] = np.nan
        self.z_sparse[1, 2] = np.nan
        self.z_sparse[2, 0] = np.nan

    def test_calculate_calculate_divergence_sparse(self):
        # Let's take a matrix with different values based on the original one.
        x_hat = self.x * 1.25

        divergence_matrix_sparse = self.dec._calculate_divergence_generic(self.x_sparse, x_hat, True).astype(float)

        # As in the core function, I calculate the divergence_sum with np.sum function.
        sum_divergence = np.sum(divergence_matrix_sparse)
        print('Divergence sum: ', sum_divergence)
        divergence_matrix_normal = pd.DataFrame.sparse.from_spmatrix(divergence_matrix_sparse)

        self.assertEqual(True, divergence_matrix_normal is not None,
                         'The divergence_matrix_sparsity is not calculated.')
        self.assertGreater(sum_divergence, 0, 'Unexpected negative value for the sum of the divergence.')

    def test_calculate_calculate_divergence_not_sparse_equal_matrices(self):
        # Let's take a matrix with different values based on the original one. In this case the matrices are not sparse.
        x_hat = self.x * 1

        divergence_matrix_sparse = self.dec._calculate_divergence_generic(self.x, x_hat, True).astype(float)
        divergence_matrix_not_sparse = self.dec._calculate_divergence_generic(self.x, x_hat, True).astype(float)

        # As in the core function, I calculate the divergence_sum with np.sum function.
        sum_divergence_sparse = np.sum(divergence_matrix_sparse)
        sum_divergence_not_sparse = np.sum(divergence_matrix_not_sparse)
        print('Divergence sum sparse: ', sum_divergence_sparse)
        print('Divergence sum: not sparse', sum_divergence_not_sparse)

        divergence_matrix_normal = pd.DataFrame.sparse.from_spmatrix(divergence_matrix_sparse)

        self.assertEqual(True, divergence_matrix_normal is not None,
                         'The divergence_matrix_sparsity is not calculated.')
        self.assertEqual(sum_divergence_sparse, 0, 'Different values for the sum of the sparse divergence. '
                                                   'It should be zero.')
        self.assertEqual(sum_divergence_not_sparse, 0, 'Different values for the sum of the not sparse divergence. '
                                                       'It should be zero.')

    def test_calculate_h_new_extended_alpha_beta_sparse_only_x(self):
        # Let's take a matrix with different values based on the original one.
        x_hat = self.x * 1.25

        # Parameters for an extended version of the model.
        alpha = 0
        y = None
        y_hat = None
        a = None
        proportion_constraint = True
        is_model_sparse = True

        h_new = self.dec._calculate_h_new_extended_alpha_beta_generic(self.x_sparse, x_hat, self.w, self.h,
                                                                      alpha, y, y_hat, a,
                                                                      proportion_constraint, is_model_sparse)
        print(h_new)

        sparsity_nulls = self.dec.sparsity_calculation(h_new, type_analysis='nulls')

        self.assertEqual(True, h_new is not None, 'The matrix H is calculate based on a X sparse matrix.')
        self.assertEqual(0.0, sparsity_nulls, 'The H matrix has nulls, therefore some went wrong')

    def test_calculate_h_new_extended_alpha_beta_sparse_x_y(self):
        # Let's take a matrix with different values based on the original one.
        x_hat = self.x * 1.25
        y_hat = self.y * 2.23

        # Parameters for an extended version of the model.
        alpha = 0.05
        proportion_constraint = True
        is_model_sparse = True

        h_new = self.dec._calculate_h_new_extended_alpha_beta_generic(self.x_sparse, x_hat, self.w, self.h,
                                                                      alpha, self.y_sparse, y_hat, self.a,
                                                                      proportion_constraint, is_model_sparse)
        print(h_new)

        sparsity_nulls = self.dec.sparsity_calculation(h_new, type_analysis='nulls')

        self.assertEqual(True, h_new is not None, 'The matrix H is calculate based on a X and Y sparse matrix.')
        self.assertEqual(0.0, sparsity_nulls, 'The H matrix does not have nulls')

    def test_calculate_a_new_extended_sparse(self):
        # Let's take a matrix with different values based on the original one.
        y_hat = self.y * 2.23

        # Parameters for an extended version of the model.
        is_model_sparse = True

        a_new = self.dec._calculate_a_new_extended_generic(self.y_sparse, y_hat, self.a, self.h, is_model_sparse)
        print(a_new)

        sparsity_nulls = self.dec.sparsity_calculation(a_new, type_analysis='nulls')

        self.assertEqual(True, a_new is not None, 'The matrix A is calculate based on a Y sparse matrix.')
        self.assertEqual(0.0, sparsity_nulls, 'The A matrix does not have nulls.')

    def test_calculate_b_new_extended_sparse(self):
        # Let's take a matrix with different values based on the original one.
        z_hat = self.z * 3.2

        # Parameters for an extended version of the model.
        is_model_sparse = True

        b_new = self.dec._calculate_b_new_extended_generic(self.z_sparse, z_hat, self.b, self.w, is_model_sparse)
        print(b_new)

        sparsity_nulls = self.dec.sparsity_calculation(b_new, type_analysis='nulls')

        self.assertEqual(True, b_new is not None, 'The matrix B is calculate based on a Z sparse matrix.')
        self.assertEqual(0.0, sparsity_nulls, 'The B matrix does not have nulls.')

    def test_calculate_w_new_extended_alpha_beta_sparse_only_x(self):
        # Let's take a matrix with different values based on the original one.
        x_hat = self.x * 0.75

        # Parameters for an extended version of the model.
        beta = 0
        z = None
        z_hat = None
        b = None
        regularize_w = None
        alpha_regularizer_w = 0
        is_model_sparse = True

        w_new = self.dec._calculate_w_new_extended_alpha_beta_generic(self.x_sparse, x_hat, self.w, self.h, beta, z,
                                                                      z_hat, b, regularize_w, alpha_regularizer_w,
                                                                      is_model_sparse)
        print(w_new)

        sparsity_nulls = self.dec.sparsity_calculation(w_new, type_analysis='nulls')

        self.assertEqual(True, w_new is not None, 'The matrix W has not been calculate based on a X sparse matrix.')
        self.assertEqual(0.0, sparsity_nulls, 'The W matrix has nulls')

    def test_calculate_w_new_extended_alpha_beta_sparse_x_z(self):
        # Let's take a matrix with different values based on the original one.
        x_hat = self.x * 0.75
        z_hat = self.z * 2.62

        # Parameters for an extended version of the model.
        beta = 0.5
        # TODO: I should check the unit tests using regularizer parameters.
        regularize_w = None
        alpha_regularizer_w = 0
        is_model_sparse = True

        w_new = self.dec._calculate_w_new_extended_alpha_beta_generic(self.x_sparse, x_hat, self.w, self.h, beta,
                                                                      self.z_sparse, z_hat, self.b, regularize_w,
                                                                      alpha_regularizer_w, is_model_sparse)
        print(w_new)

        sparsity_nulls = self.dec.sparsity_calculation(w_new, type_analysis='nulls')
        sparsity_zeros = self.dec.sparsity_calculation(w_new, type_analysis='zeros')

        self.assertEqual(w_new is not None, True, 'The matrix W is calculate based on a X and Z sparse matrix.')
        self.assertEqual(sparsity_nulls, 0.0, 'The W matrix does not have nulls')


if __name__ == '__main__':
    unittest.main()
