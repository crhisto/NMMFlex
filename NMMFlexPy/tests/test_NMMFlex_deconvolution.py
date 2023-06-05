import unittest
from os import getcwd

import numpy as np
import numpy.testing
import pandas as pd

from NMMFlex.NMMFlex import NMMFlex


# TODO improve the tests cases dividing it in three: general, gridsearch and sparseness
class test_NMMFlex_deconvolution(unittest.TestCase):
    relative_path = ''
    bulk_methylation_matrix = None
    expected_proportions = None
    bulk_expression_matrix = None
    data_expression_auxiliary_matrix_random = None
    dec = None

    def __init__(self, relative_path):
        super(test_NMMFlex_deconvolution, self).__init__(relative_path)
        # To allow the test cases to run from different configurations of main path without problems
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
        self.bulk_methylation_matrix = pd.read_csv(self.relative_path + 'data/meth_bulk_samples.filtered.markers.CpGs'
                                                                        '.500.csv', index_col=0)
        self.expected_proportions = pd.read_csv(self.relative_path + 'data/expected.proportions.edec.markers.GpGs.500'
                                                                     '.csv', index_col=0)
        self.bulk_expression_matrix = pd.read_csv(self.relative_path + 'data/gene_exp_mixtures.csv', index_col=0)
        self.data_expression_auxiliary_matrix_random = pd.DataFrame(
            data=np.random.rand(len(self.bulk_methylation_matrix.index), len(self.bulk_methylation_matrix.columns)),
            index=self.bulk_methylation_matrix.index.values,
            columns=self.bulk_methylation_matrix.columns.values)

    def test_run_deconvolution(self):
        k = 4
        w = np.array(np.random.rand(10, k))
        h = np.array(np.random.rand(k, 6))
        x_derived = w.dot(h)

        # Basic deconvolution with built data
        results = self.dec.run_deconvolution(x_matrix=x_derived, k=k, delta_threshold=0.0000000001,
                                             max_iterations=100000)

        np.testing.assert_equal(np.all(results.h == 0), False, 'The h matrix has been calculated and it is all '
                                                               'zeros.')
        np.testing.assert_equal(np.all(results.w == 0), False, 'The w matrix has been calculated and it is all '
                                                               'zeros.')

    def test_run_deconvolution_methylation_data(self):
        # print(bulk_methylation_matrix)
        # print(expected_proportions)

        # Initial run with bulk methylation data.
        results = self.dec.run_deconvolution(x_matrix=self.bulk_methylation_matrix.iloc[:, 0:2],
                                             k=4, delta_threshold=1e-20,
                                             max_iterations=1000)

        np.testing.assert_equal(np.all(results.h == 0), False, 'The h matrix has been calculated and it is all '
                                                               'zeros.')
        np.testing.assert_equal(np.all(results.w == 0), False, 'The w matrix has been calculated and it is all '
                                                               'zeros.')

    def test_run_deconvolution_multiple_alpha(self):
        # print(bulk_methylation_matrix)
        # print(expected_proportions)

        results = self.dec.run_deconvolution_multiple(x_matrix=self.bulk_methylation_matrix.iloc[:, 0:2],
                                                      y_matrix=self.bulk_expression_matrix.iloc[:, 0:2],
                                                      z_matrix=None,
                                                      k=4,
                                                      alpha=0.5, beta=0.0, delta_threshold=0.005, max_iterations=3,
                                                      print_limit=100)

        np.testing.assert_equal(np.all(results.h == 0), False, 'The h matrix has been calculated and it is all '
                                                               'zeros.')
        np.testing.assert_equal(np.all(results.w == 0), False, 'The w matrix has been calculated and it is all '
                                                               'zeros.')
        np.testing.assert_equal(np.all(results.a == 0), False, 'The a matrix has been calculated and it is all '
                                                               'zeros.')

    def test_run_deconvolution_multiple_alpha_beta_one_model(self):
        results = self.dec.run_deconvolution_multiple(x_matrix=self.bulk_methylation_matrix.iloc[:, 0:2],
                                                      y_matrix=self.bulk_expression_matrix.iloc[:, 0:2],
                                                      z_matrix=self.data_expression_auxiliary_matrix_random.iloc[:,
                                                               0:2],
                                                      k=4, alpha=0.5, beta=0.5, delta_threshold=0.005, max_iterations=2,
                                                      print_limit=100)

        np.testing.assert_equal(np.all(results.h == 0), False, 'The h matrix has been calculated and it is all '
                                                               'zeros.')
        np.testing.assert_equal(np.all(results.w == 0), False, 'The w matrix has been calculated and it is all '
                                                               'zeros.')
        np.testing.assert_equal(np.all(results.a == 0), False, 'The a matrix has been calculated and it is all '
                                                               'zeros.')
        np.testing.assert_equal(np.all(results.b == 0), False, 'The b matrix has been calculated and it is all '
                                                               'zeros.')


if __name__ == '__main__':
    unittest.main()
