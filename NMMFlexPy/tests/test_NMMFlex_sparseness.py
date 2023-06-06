import unittest

import numpy as np
import pandas as pd

from NMMFlex.NMMFlex import NMMFlex


class test_NMMFlex_sparseness(unittest.TestCase):

    # This function will be called for each tests.
    def setUp(self):
        self.dec = NMMFlex()

    def tearDown(self):
        del self.dec

    def test_sparsity_calculation_non_zeros_or_null(self):
        non_zeros_or_null_matrix = np.random.rand(5, 5)
        sparsity_index = self.dec.sparsity_calculation(
            non_zeros_or_null_matrix)
        sparsity_percentage = self.dec.sparsity_calculation(
            non_zeros_or_null_matrix,
            type_return='percentage')
        sparsity_boolean = self.dec.sparsity_calculation(
            non_zeros_or_null_matrix,
            type_return='boolean')

        self.assertEqual(0.0,
                         sparsity_index,
                         'The sparsity for a matrix with 0.0 values zero.')
        self.assertEqual(0,
                         sparsity_percentage,
                         'The sparsity for a matrix with 0% values zero.')
        self.assertEqual(0, sparsity_boolean,
                         'The sparsity for a matrix with FALSE values zero.')

    def test_sparsity_calculation_all_zeros(self):
        zeros_matrix = np.zeros((7, 7))

        sparsity_index = self.dec.sparsity_calculation(zeros_matrix)
        sparsity_percentage = self.dec.sparsity_calculation(
            zeros_matrix,
            type_return='percentage')
        sparsity_boolean = self.dec.sparsity_calculation(
            zeros_matrix,
            type_return='boolean')

        self.assertEqual(1.0,
                         sparsity_index,
                         'The sparsity for a matrix with 1.0 values zero.')
        self.assertEqual(100,
                         sparsity_percentage,
                         'The sparsity for a matrix with 100% values zero.')
        self.assertEqual(True,
                         sparsity_boolean,
                         'The sparsity for a matrix with TRUE values zero.')

    def test_sparsity_calculation_some_zeros(self):
        # 6 zero values over 16 equal to: 7/16 = 0.4375 sparseness
        matrix_test = np.array([[3, 0, 0, 3],
                                [0, 3, 0, 2],
                                [5, 0, 1, 2],
                                [8, 1, 0, 0]])

        sparsity_index = self.dec.sparsity_calculation(
            matrix_test,
            verbose=True)
        sparsity_percentage = self.dec.sparsity_calculation(
            matrix_test,
            type_return='percentage')
        sparsity_boolean = self.dec.sparsity_calculation(
            matrix_test,
            type_return='boolean')

        self.assertAlmostEqual(
            first=0.4375,
            second=sparsity_index,
            places=None,
            msg='The sparsity for a matrix with some zero values '
                '(7/16 = 0.4375) was not detected.',
            delta=0.00)
        self.assertEqual(43.75, sparsity_percentage,
                         'The sparsity for a matrix with some zero values: '
                         '43.75% sparseness (7/16 = 0.4375) was not detected.')
        self.assertEqual(True, sparsity_boolean,
                         'The sparsity for a matrix with some zero values, '
                         'TRUE sparse (7/16 = 0.4375) was not detected.')

    def test_sparsity_calculation_some_zeros_and_null(self):
        # 6 zero and NaN values over 16 equal to: 6/16 = 0.375 sparseness
        matrix_test = np.array([[3, 6, 0, 3],
                                [0, 3, np.nan, 2],
                                [5, 0, 1, 2],
                                [8, 1, 0, np.nan]])

        sparsity_index = self.dec.sparsity_calculation(
            matrix_test,
            verbose=True)
        sparsity_percentage = self.dec.sparsity_calculation(
            matrix_test,
            type_return='percentage')
        sparsity_boolean = self.dec.sparsity_calculation(
            matrix_test,
            type_return='boolean')

        self.assertAlmostEqual(first=0.375,
                               second=sparsity_index,
                               places=None,
                               msg='The sparsity for a matrix with some zero '
                                   'and null or NaN values (7/16 = 0.4375)'
                                   ' was not detected.',
                               delta=0.00)
        self.assertEqual(37.5,
                         sparsity_percentage,
                         'The sparsity for a matrix with some zero values: '
                         '37.5% sparseness (6/16 = 0.375) was not detected.')
        self.assertEqual(True, sparsity_boolean,
                         'The sparsity for a matrix with some zero values, '
                         'TRUE sparse (6/16 = 0.375) was not detected.')

    def test_sparsity_calculation_zero_sparsity(self):
        # 6 zero and NaN values over 16 equal, 4 zeros: 4/16 = 0.25 sparseness
        matrix_test = np.array([[3, 6, 0, 3],
                                [0, 3, np.nan, 2],
                                [5, 0, 1, 2],
                                [8, 1, 0, np.nan]])

        sparsity_index = self.dec.sparsity_calculation(matrix_test,
                                                       verbose=True,
                                                       type_analysis='zeros')
        sparsity_percentage = self.dec.sparsity_calculation(
            matrix_test,
            type_return='percentage',
            type_analysis='zeros')
        sparsity_boolean = self.dec.sparsity_calculation(
            matrix_test,
            type_return='boolean',
            type_analysis='zeros')

        self.assertAlmostEqual(first=0.25,
                               second=sparsity_index,
                               places=None,
                               msg='The sparsity for a matrix with some zero '
                                   'values (4/16 = 0.25) was not detected.',
                               delta=0.00)
        self.assertEqual(25,
                         sparsity_percentage,
                         'The sparsity for a matrix with some zero values: '
                         '25% sparseness (4/16 = 0.25) was not detected.')
        self.assertEqual(True, sparsity_boolean,
                         'The sparsity for a matrix with some zero values, '
                         'TRUE sparse (4/16 = 0.25) was not detected.')

    def test_sparsity_calculation_null_sparsity(self):
        # 6 zero and NaN values over 16 equal, 2 zeros: 2/16 = 0.125 sparseness
        matrix_test = np.array([[3, 6, 0, 3],
                                [0, 3, np.nan, 2],
                                [5, 0, 1, 2],
                                [8, 1, 0, np.nan]])

        sparsity_index = self.dec.sparsity_calculation(
            matrix_test,
            verbose=True,
            type_analysis='nulls')
        sparsity_percentage = self.dec.sparsity_calculation(
            matrix_test,
            type_return='percentage',
            type_analysis='nulls')
        sparsity_boolean = self.dec.sparsity_calculation(
            matrix_test,
            type_return='boolean',
            type_analysis='nulls')

        self.assertAlmostEqual(first=0.125,
                               second=sparsity_index,
                               places=None,
                               msg='The sparsity for a matrix with some zero '
                                   'values (2/16 = 0.125) was not detected.',
                               delta=0.00)
        self.assertEqual(12.5,
                         sparsity_percentage,
                         'The sparsity for a matrix with some zero values: '
                         '25% sparseness (2/16 = 0.125) was not detected.')
        self.assertEqual(True, sparsity_boolean,
                         'The sparsity for a matrix with some zero values, '
                         'TRUE sparse (2/16 = 0.125) was not detected.')

    def test_analyse_sparsity_matrix(self):
        # 6 zero and NaN values over 16 equal, 4 zeros: 4/16 = 0.25 sparseness
        matrix_test = np.array([[3, 6, 0, 3],
                                [0, 3, np.nan, 2],
                                [5, 0, 1, 2],
                                [8, 1, 0, np.nan]])

        description_text = self.dec._analyse_sparsity_matrix(matrix_test, 'X')
        self.assertEqual('The matrix X. Sparsity:  37.5% - zeros: 25.0% and '
                         'nulls: 12.5%', description_text,
                         'The description of the matrix is not correct.')

    def test_analyse_sparsity_matrix_empty(self):
        # null matrix
        matrix_test = None

        description_text = self.dec._analyse_sparsity_matrix(matrix_test, 'X')
        self.assertEqual('The matrix X is empty.', description_text,
                         'The description of the empty matrix is not correct.')

    def test_standardize_sparse_matrices_empty_strings(self):
        matrix_test = np.array([[3, 6, 0, 3],
                                [0, 3, '', 2],
                                [5, 0, 1, 2],
                                [8, 1, 0, ""]])
        matrix_test_df = pd.DataFrame(matrix_test)
        matrix_expected = np.array([[3., 6., 0., 3.],
                                    [0., 3., np.nan, 2.],
                                    [5., 0., 1., 2.],
                                    [8., 1., 0., np.nan]])

        standardized_matrix = self.dec._standardize_sparse_matrix(
            matrix_test_df)

        np.testing.assert_array_equal(
            matrix_expected,
            standardized_matrix,
            "Matrix with empty spaces transformed to np.nan")

    def test_standardize_sparse_matrices_null_na_string(self):
        matrix_test = np.array([['NA', 6, 0, 3],
                                [0, 3, 'na', 2],
                                [5, 0, 1, 2],
                                [8, 1, 0, "null"]])
        matrix_test_df = pd.DataFrame(matrix_test)
        matrix_expected = np.array([[np.nan, 6., 0., 3.],
                                    [0., 3., np.nan, 2.],
                                    [5., 0., 1., 2.],
                                    [8., 1., 0., np.nan]])
        matrix_expected_df = pd.DataFrame(matrix_expected)

        standardized_matrix = self.dec._standardize_sparse_matrix(
            matrix_test_df)

        print("Matrix with null and na string spaces transformed to np.nan")
        pd.testing.assert_frame_equal(standardized_matrix, matrix_expected_df)


if __name__ == '__main__':
    unittest.main()
