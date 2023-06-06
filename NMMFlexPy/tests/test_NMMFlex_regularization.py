import unittest
from os import getcwd

import numpy as np
import numpy.testing
import pandas as pd

from NMMFlex.NMMFlex import NMMFlex


class test_NMMFlex_regularization(unittest.TestCase):
    relative_path = ''
    bulk_methylation_matrix = None
    expected_proportions = None
    bulk_expression_matrix = None
    data_expression_auxiliary_matrix_random = None
    dec = None

    def __init__(self, relative_path):
        super(test_NMMFlex_regularization, self).__init__(relative_path)
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
            data=np.random.rand(len(self.bulk_methylation_matrix.index),
                                len(self.bulk_methylation_matrix.columns)),
            index=self.bulk_methylation_matrix.index.values,
            columns=self.bulk_methylation_matrix.columns.values)

    def test_quantile_normalize(self):
        results = self.dec.quantile_normalize(self.bulk_methylation_matrix)
        check_results = np.all((results >= 0) & (results <= 1))

        self.assertTrue(check_results,
                        'The matrix was not quantile normalized.')


if __name__ == '__main__':
    unittest.main()
