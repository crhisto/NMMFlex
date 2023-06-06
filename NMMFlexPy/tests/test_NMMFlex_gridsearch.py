import unittest
from os import getcwd

import numpy
import numpy as np
import pandas as pd

from NMMFlex.NMMFlex import grid_search_parallelized_async


class test_NMMFlex_gridsearch(unittest.TestCase):
    relative_path = ''
    bulk_methylation_matrix = None
    expected_proportions = None
    bulk_expression_matrix = None
    data_expression_auxiliary_matrix_random = None
    grid_search_deco = None

    def __init__(self, relative_path):
        super(test_NMMFlex_gridsearch, self).__init__(relative_path)
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
        self.grid_search_deco = grid_search_parallelized_async()

    def tearDown(self):
        del self.bulk_methylation_matrix
        del self.expected_proportions
        del self.bulk_expression_matrix
        del self.data_expression_auxiliary_matrix_random
        del self.grid_search_deco

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
            'data/gene_exp_mixtures.csv', index_col=0)
        self.data_expression_auxiliary_matrix_random = pd.DataFrame(
            data=np.random.rand(len(self.bulk_methylation_matrix.index),
                                len(self.bulk_methylation_matrix.columns)),
            index=self.bulk_methylation_matrix.index.values,
            columns=self.bulk_methylation_matrix.columns.values)

    def test_calculate_pair_parameters_alpha_one(self):
        alpha_list = np.array([0.0, 0.1, 0.2]).astype(float)
        beta_list = np.array([0.0]).astype(float)
        expected_result = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])

        results = self.grid_search_deco._calculate_pair_parameters(alpha_list,
                                                                   beta_list)

        numpy.testing.assert_array_equal(
            expected_result, results, 'The array is not correct.')

    def test_calculate_pair_parameters_beta_one(self):
        alpha_list = np.array([0.0]).astype(float)
        beta_list = np.array([0.0, 0.1, 0.2]).astype(float)
        expected_result = np.array([[0.0, 0.0], [0.0, 0.1], [0.0, 0.2]])

        results = self.grid_search_deco._calculate_pair_parameters(alpha_list,
                                                                   beta_list)

        numpy.testing.assert_array_equal(expected_result,
                                         results,
                                         'The array is not correct.')

    def test_run_deconvolution_multiple_alpha_variation_sync(self):

        alpha_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).astype(float)

        results = self.grid_search_deco.grid_search_parallelized_alpha_beta(
            bulk_data_methylation=self.bulk_methylation_matrix.iloc[:, 1:2],
            bulk_data_expression=self.bulk_expression_matrix.iloc[:, 1:4],
            data_expression_auxiliary=None,
            k=4,
            alpha_list=alpha_list,
            beta_list=None,
            delta_threshold=0.005,
            max_iterations=2, print_limit=100,
            threads=6)
        print('Number of models in the object: ', len(results))
        self.assertEqual(len(alpha_list), len(results),
                         'The model number is not the expected: 3')

    def test_run_deconvolution_multiple_alpha_variation_async(self):

        alpha_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).astype(float)

        results = self.grid_search_deco.grid_search_parallelized_alpha_beta(
            bulk_data_methylation=self.bulk_methylation_matrix.iloc[:, 1:2],
            bulk_data_expression=self.bulk_expression_matrix.iloc[:, 1:4],
            data_expression_auxiliary=None,
            k=4,
            alpha_list=alpha_list,
            beta_list=None,
            delta_threshold=0.005,
            max_iterations=2, print_limit=100,
            threads=6)
        print('Number of models in the object: ', len(results))
        self.assertEqual(len(alpha_list),
                         len(results),
                         'The model number is not the expected: 3')

    def test_run_deconvolution_multiple_alpha_beta_async_subset(self):

        alpha_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                               0.6, 0.7, 0.8, 0.9, 1.0]).astype(float)
        beta_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]).astype(float)

        results = self.grid_search_deco.grid_search_parallelized_alpha_beta(
            bulk_data_methylation=self.bulk_methylation_matrix.iloc[:, 1:2],
            bulk_data_expression=self.bulk_expression_matrix.iloc[:, 1:2],
            data_expression_auxiliary=
            self.data_expression_auxiliary_matrix_random.iloc[:, 1:2],
            k=4,
            alpha_list=alpha_list,
            beta_list=beta_list,
            delta_threshold=0.005,
            max_iterations=3, print_limit=100,
            threads=0)
        print('Number of models in the object: ', len(results))
        self.assertEqual((len(alpha_list) * len(beta_list)), len(results),
                         'The model number is not the expected: 66')

    def test_run_deconvolution_multiple_alpha_beta_async_complete_set(self):

        alpha_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4,
                               0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).astype(float)
        beta_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]).astype(float)

        results = self.grid_search_deco.grid_search_parallelized_alpha_beta(
            bulk_data_methylation=self.bulk_methylation_matrix,
            bulk_data_expression=self.bulk_expression_matrix,
            data_expression_auxiliary=
            self.data_expression_auxiliary_matrix_random,
            k=4,
            alpha_list=alpha_list,
            beta_list=beta_list,
            delta_threshold=0.005,
            max_iterations=2, print_limit=100,
            threads=0)
        print('Number of models in the object: ', len(results))
        self.assertEqual((len(alpha_list) * len(beta_list)), len(results),
                         'The model number is not the expected: 66')


if __name__ == '__main__':
    unittest.main()
