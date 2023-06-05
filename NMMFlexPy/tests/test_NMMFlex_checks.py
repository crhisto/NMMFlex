import unittest
import numpy as np
import pandas as pd

from NMMFlex.NMMFlex import NMMFlex


# It's true that we shouldn't create unit tests based on specific strings, but in this case create a
# static list of the messages is too much for the functionality, maybe in the future I can improve this.
class test_NMMFlex_checks(unittest.TestCase):
    """
    The test_NMMFlex_checks class is a TestCase subclass for testing the NMMFlex checks.

    Attributes:
    dec (NMMFlex): The NMMFlex instance to be used for testing.

    Methods:
    setUp: Sets up a NMMFlex instance before each tests.
    tearDown: Cleans up after each tests.

    Notes:
    - All tests should be implemented as methods with names starting with 'tests'.
    - Each tests method should use the assert methods from the unittest.TestCase class to check for various conditions.
    - The setUp method is run before each individual tests, and the tearDown method is run after each tests.
    - The dec attribute holds an instance of the NMMFlex class that can be used in each tests.
    """

    # This function will be called for each tests.
    def setUp(self):
        self.dec = NMMFlex()

    def tearDown(self):
        del self.dec

    def test_check_parameters_latent_components_zero(self):
        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=None, y_matrix=None, z_matrix=None, k=0, alpha=0, beta=0,
                                       delta_threshold=0,
                                       max_iterations=1, print_limit=1, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w=None, init_method_h=None,
                                       init_method_a=None, init_method_b=None)
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('K') >= 0, 'Message is unrelated with the error.')

    def test_check_parameters_latent_components_exceeded(self):
        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=None, y_matrix=None, z_matrix=None, k=21, alpha=0, beta=0,
                                       delta_threshold=0,
                                       max_iterations=1, print_limit=1, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w=None, init_method_h=None,
                                       init_method_a=None, init_method_b=None)
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('K') >= 0, 'Message is unrelated with the error.')

    def test_check_parameters_alpha(self):
        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=None, y_matrix=None, z_matrix=None, k=2, alpha=-1, beta=0,
                                       delta_threshold=0,
                                       max_iterations=1, print_limit=1, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w=None, init_method_h=None,
                                       init_method_a=None, init_method_b=None)
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('Alpha') >= 0, 'Message is unrelated with the error.')

    def test_check_parameters_beta(self):
        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=None, y_matrix=None, z_matrix=None, k=2, alpha=0, beta=-1,
                                       delta_threshold=0,
                                       max_iterations=1, print_limit=1, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w=None, init_method_h=None,
                                       init_method_a=None, init_method_b=None)
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('Beta') >= 0, 'Message is unrelated with the error.')

    def test_check_parameters_delta_threshold(self):
        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=None, y_matrix=None, z_matrix=None, k=2, alpha=0, beta=0,
                                       delta_threshold=0,
                                       max_iterations=1, print_limit=1, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w=None, init_method_h=None,
                                       init_method_a=None, init_method_b=None)
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('Delta threshold') >= 0, 'Message is unrelated with the error.')

    def test_check_parameters_max_iterations(self):
        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=None, y_matrix=None, z_matrix=None, k=2, alpha=0, beta=0,
                                       delta_threshold=0.005,
                                       max_iterations=0, print_limit=1, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w=None, init_method_h=None,
                                       init_method_a=None, init_method_b=None)
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('Maximum number of iterations') >= 0,
                        'Message is unrelated with the error.')

    def test_check_parameters_print_limit(self):
        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=None, y_matrix=None, z_matrix=None, k=2, alpha=0, beta=0,
                                       delta_threshold=0.005,
                                       max_iterations=2, print_limit=0, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w=None, init_method_h=None,
                                       init_method_a=None, init_method_b=None)
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('Number of prints') >= 0, 'Message is unrelated with the error.')

    def test_check_parameters_proportion_constraint_h(self):
        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=None, y_matrix=None, z_matrix=None, k=2, alpha=0, beta=0,
                                       delta_threshold=0.005,
                                       max_iterations=2, print_limit=100, proportion_constraint_h='True',
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w=None, init_method_h=None,
                                       init_method_a=None, init_method_b=None)
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('proportion_constraint_h') >= 0,
                        'Message is unrelated with the error.')

    def test_check_parameters_check_init_methods(self):
        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=None, y_matrix=None, z_matrix=None, k=2, alpha=0, beta=0,
                                       delta_threshold=0.005,
                                       max_iterations=2, print_limit=100, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w='random_based.uniform_wrong', init_method_h=None,
                                       init_method_a=None, init_method_b=None)
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('init_method_w') >= 0, 'Message is unrelated with the error.')

    def test_check_parameters_None_matrix_x(self):
        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=None, y_matrix=None, z_matrix=None, k=2, alpha=0, beta=0,
                                       delta_threshold=0.005,
                                       max_iterations=2, print_limit=100, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w='random_based.uniform', init_method_h='random_based.uniform',
                                       init_method_a='random_based.uniform', init_method_b='random_based.uniform')
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('X') >= 0, 'Message is unrelated with the error.')

    def test_check_parameters_non_negative_values_matrices(self):
        matrix_test = np.array([[-1, 0, 0, 3], [0, 3, 0, 2], [5, 0, 1, 2], [8, 1, 0, 0]])
        # Since the original pipeline send always a dataframe, I will convert this
        matrix_test_df = pd.DataFrame(matrix_test)

        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=matrix_test_df, y_matrix=None, z_matrix=None, k=2, alpha=0, beta=0,
                                       delta_threshold=0.005,
                                       max_iterations=2, print_limit=100, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w='random_based.uniform', init_method_h='random_based.uniform',
                                       init_method_a='random_based.uniform', init_method_b='random_based.uniform')
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('x_matrix') >= 0, 'Message is unrelated with the error.')

    def test_check_parameters_check_complete_zero_matrix(self):
        # Sometimes I like to see actual values!!
        matrix_test = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        # Since the original pipeline send always a dataframe, I will convert this
        matrix_test_df = pd.DataFrame(matrix_test)

        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=matrix_test_df, y_matrix=None, z_matrix=None, k=2, alpha=0, beta=0,
                                       delta_threshold=0.005,
                                       max_iterations=2, print_limit=100, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w='random_based.uniform', init_method_h='random_based.uniform',
                                       init_method_a='random_based.uniform', init_method_b='random_based.uniform')
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('x_matrix') >= 0, 'Message is unrelated with the error.')

    def test_check_parameters_check_complete_null_matrix(self):
        # Sometimes I like to see actual values!!
        matrix_test = np.array([[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]])
        # Since the original pipeline send always a dataframe, I will convert this
        matrix_test_df = pd.DataFrame(matrix_test)

        with self.assertRaises(Exception) as error:
            self.dec._check_parameters(x_matrix=matrix_test_df, y_matrix=None, z_matrix=None, k=2, alpha=0, beta=0,
                                       delta_threshold=0.005,
                                       max_iterations=2, print_limit=100, proportion_constraint_h=True,
                                       regularize_w=None, alpha_regularizer_w=None,
                                       fixed_w=None, fixed_h=None, fixed_a=None, fixed_b=None,
                                       initialized_w=None, initialized_h=None, initialized_a=None, initialized_b=None,
                                       init_method_w='random_based.uniform', init_method_h='random_based.uniform',
                                       init_method_a='random_based.uniform', init_method_b='random_based.uniform')
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('x_matrix') >= 0, 'Message is unrelated with the error.')

    def test_check_fixed_matrices_w_shape(self):
        matrix_test_similar_w = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        matrix_test_x = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        with self.assertRaises(Exception) as error:
            self.dec._check_fixed_matrices(x_matrix=matrix_test_x,
                                           y_matrix=None,
                                           z_matrix=None,
                                           similar_w=matrix_test_similar_w,
                                           similar_a=None,
                                           similar_b=None,
                                           similar_h=None,
                                           k=0,
                                           type_matrix='fixed')
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('_w is not compatible with the matrix X and the k value.') >= 0,
                        'Message is unrelated with the error.')

    def test_check_fixed_matrices_h_shape(self):
        matrix_test_similar_h = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        matrix_test_x = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

        with self.assertRaises(Exception) as error:
            self.dec._check_fixed_matrices(x_matrix=matrix_test_x,
                                           y_matrix=None,
                                           z_matrix=None,
                                           similar_w=None,
                                           similar_a=None,
                                           similar_b=None,
                                           similar_h=matrix_test_similar_h,
                                           k=0,
                                           type_matrix='fixed')
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('_h is not compatible with the matrix X and the K value.') >= 0,
                        'Message is unrelated with the error.')

    def test_check_fixed_matrices_a_shape(self):
        matrix_test_similar_a = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        matrix_test_y = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        with self.assertRaises(Exception) as error:
            self.dec._check_fixed_matrices(x_matrix=None,
                                           y_matrix=matrix_test_y,
                                           z_matrix=None,
                                           similar_w=None,
                                           similar_a=matrix_test_similar_a,
                                           similar_b=None,
                                           similar_h=None,
                                           k=0,
                                           type_matrix='fixed')
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('_a is not compatible with the matrix Y and the K value.') >= 0,
                        'Message is unrelated with the error.')

    def test_check_fixed_matrices_a_y_existence(self):
        matrix_test_similar_a = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        with self.assertRaises(Exception) as error:
            self.dec._check_fixed_matrices(x_matrix=None,
                                           y_matrix=None,
                                           z_matrix=None,
                                           similar_w=None,
                                           similar_a=matrix_test_similar_a,
                                           similar_b=None,
                                           similar_h=None,
                                           k=0,
                                           type_matrix='fixed')
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('The matrix Y is not given as a input, therefore the') >= 0,
                        'Message is unrelated with the error.')

    def test_check_fixed_matrices_b_z_existence(self):
        matrix_test_similar_b = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        with self.assertRaises(Exception) as error:
            self.dec._check_fixed_matrices(x_matrix=None,
                                           y_matrix=None,
                                           z_matrix=None,
                                           similar_w=None,
                                           similar_a=None,
                                           similar_b=matrix_test_similar_b,
                                           similar_h=None,
                                           k=0,
                                           type_matrix='fixed')
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('The matrix Z is not given as a input, therefore the ') >= 0,
                        'Message is unrelated with the error.')

    def test_check_fixed_matrices_b_shape(self):
        matrix_test_similar_b = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        matrix_test_z = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        with self.assertRaises(Exception) as error:
            self.dec._check_fixed_matrices(x_matrix=None,
                                           y_matrix=None,
                                           z_matrix=matrix_test_z,
                                           similar_w=None,
                                           similar_a=None,
                                           similar_b=matrix_test_similar_b,
                                           similar_h=None,
                                           k=0,
                                           type_matrix='fixed')
        print(str(error.exception))
        # I need to check if the message is related somehow with the parameter.
        self.assertTrue(str(error.exception).rfind('_b is not compatible with the matrix Y and the K value.') >= 0,
                        'Message is unrelated with the error.')


if __name__ == '__main__':
    unittest.main()
