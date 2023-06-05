import unittest
import numpy as np
import numpy.testing
import pandas as pd
import pytest

from NMMFlex.NMMFlex import NMMFlex


class test_NMMFlex_sparseness_deconvolution(unittest.TestCase):
    bulk_methylation_matrix = None
    expected_proportions = None
    bulk_expression_matrix = None
    data_expression_auxiliary_matrix_random = None
    dec = None

    # This function will be called for each tests.
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
        self.bulk_methylation_matrix = pd.read_csv('data/meth_bulk_samples.filtered.markers.CpGs.500.csv', index_col=0)
        self.expected_proportions = pd.read_csv('data/expected.proportions.edec.markers.GpGs.500.csv', index_col=0)
        self.bulk_expression_matrix = pd.read_csv('data/gene_exp_mixtures.csv', index_col=0)
        self.data_expression_auxiliary_matrix_random = pd.DataFrame(
            data=np.random.rand(len(self.bulk_methylation_matrix.index), len(self.bulk_methylation_matrix.columns)),
            index=self.bulk_methylation_matrix.index.values,
            columns=self.bulk_methylation_matrix.columns.values)

    @pytest.mark.skip(reason="This unit tests takes too long.  Run this manually")
    def test_simple_NMF_only_X(self):
        # Initialization of x to be sparse with nulls and zeros
        bulk_methylation_matrix_sparse = self.bulk_methylation_matrix.copy()
        # Because we expected a dataframe.
        bulk_methylation_matrix_sparse.iloc[0, 1] = np.NaN
        bulk_methylation_matrix_sparse.iloc[30, 1] = 0.0

        # I will try with the first model with only Tumor samples and three cell types, since I know the accuracy will
        # be good and I can compare that with the expected proportions.
        results_normal = self.dec.run_deconvolution_multiple(x_matrix=self.bulk_methylation_matrix,
                                                             y_matrix=None, z_matrix=None,
                                                             k=4, max_iterations=10, print_limit=1)

        results_sparse = self.dec.run_deconvolution_multiple(x_matrix=bulk_methylation_matrix_sparse,
                                                             y_matrix=None, z_matrix=None,
                                                             k=4, max_iterations=10, print_limit=1)
        # 1e-07
        np.testing.assert_allclose(results_normal.w, results_sparse.w, rtol=1e-2, atol=1e-2,
                                   err_msg='The proportion matrices are different.', verbose=True)

        numpy.testing.assert_array_almost_equal(x=results_normal.w, y=results_sparse.w,
                                                decimal=13, err_msg='The proportion matrices are different.')

        numpy.testing.assert_array_equal(results_normal.w, results_sparse.w,
                                         'The proportion matrices are different.')

    @pytest.mark.skip(reason="This unit tests takes too long: 37 min approx. Run this manually")
    def test_simple_NMF_X_Y(self):
        # Initialization of x to be sparse with nulls and zeros
        bulk_methylation_matrix_sparse = self.bulk_methylation_matrix.copy()
        # Because we expected a dataframe.
        bulk_methylation_matrix_sparse.iloc[0, 1] = np.NaN
        bulk_methylation_matrix_sparse.iloc[30, 1] = 0.0

        # I will try with the first model with only Tumor samples and three cell types, since I know the accuracy will
        # be good and I can compare that with the expected proportions.
        results_normal = self.dec.run_deconvolution_multiple(x_matrix=self.bulk_methylation_matrix,
                                                             y_matrix=self.bulk_expression_matrix, z_matrix=None,
                                                             k=4, alpha=0.0001, max_iterations=2, print_limit=1)

        # This takes ages, so I need to improve the code to run faster: TODO: improve code in terms of performance.
        results_sparse = self.dec.run_deconvolution_multiple(x_matrix=bulk_methylation_matrix_sparse,
                                                             y_matrix=self.bulk_expression_matrix, z_matrix=None,
                                                             k=4, alpha=0.0001, max_iterations=2, print_limit=1)

        np.testing.assert_allclose(results_normal.w, results_sparse.w, rtol=1e-2, atol=1e-2,
                                   err_msg='The proportion matrices are different.', verbose=True)

        numpy.testing.assert_array_almost_equal(x=results_normal.w, y=results_sparse.w,
                                                decimal=13, err_msg='The proportion matrices are different.')

        numpy.testing.assert_array_equal(results_normal.w, results_sparse.w,
                                         'The proportion matrices are different.')


if __name__ == '__main__':
    unittest.main()
