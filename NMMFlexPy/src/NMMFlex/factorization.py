""" src: Non-negative Multiple Matrix Factorization.
    Comments follow the https://peps.python.org/pep-0257/
"""
# Author: Crhistian Cardona <crhisto@gmail.com>
#         Original paper: Non-Negative Multiple Matrix Factorization
#         Original paper authors: Koh Takeuchi, Katsuhiko Ishiguro, Akisato
#         Kimura, and Hiroshi Sawada
# License: Released under GNU Public License (GPL)

__version__ = '0.1.10'
__author__ = 'Crhistian Cardona <crhisto@gmail.com>'

import math
import numbers

import deprecation
import numpy
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn import preprocessing
from sklearn.preprocessing import quantile_transform, StandardScaler
from sklearn.utils.validation import check_non_negative

# This is related with the warning: "FutureWarning: elementwise comparison
# failed; returning scalar, but in the future will perform elementwise
# comparison". You can check the original issue:
# https://stackoverflow.com/questions/40659212/. The function affected is:
# _standardize_sparse_matrix. Since this is a clear disagreement between Numpy
# and native python on what should happen when you compare a strings to
# numpy's numeric types, I will follow the advice of skyping the message
# because it seems that is not going to change the behaviour in the future and
# my code is going to run anyway correctly.
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Code created based on the pep-0008: https://www.python.org/dev/peps/pep-0008/
# TODO: Check constraints for the entire set of matrices in the multiple
#  factorization method.
# TODO: Implement the tree method for both simple and complex deconvolution.
# TODO: Implement correction for calculating global and relative proportions
#  accurately.
# TODO: Verify how Python and the library handle NA values, testing it with
#  the RStudio interface.
# TODO: Create a deconvolution method optimized for expression data with
#  references.
# TODO: Create a deconvolution method optimized for methylation data with
#  references.
# TODO: Try deconvolution by fixing the expression profile based on methylation
#  and filtering genes using DEG.
# TODO: Define the reference based on both bulk data and single-cell data for
#  expression and methylation.
# TODO: Implement noise treatment for src based on the approach described
#  in this article:
#       [Noise Treatment Article]
#       (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3673742/)
# TODO: Configure all the verbose messages to provide comprehensive progress
#  information.
class factorization:
    """
    A flexible and modular implementation of Non-negative Multiple Matrix
    Factorization (NMMF) as src

    This class provides multiple variations and configurations of the NMF
    algorithm, making it adaptable to a wide range of problems.

    The main feature is the incorporation of additional constraints, beyond the
    basic non-negativity, which can improve the interpretability and usefulness
    of the results.

    Key configurations and features include:
      - Regularization on factor matrices
      - Proportion constraints
      - Option to fix certain factor matrices
      - Grid search functionality with parallel execution
      - Preprocessing methods for handling sparse data

    """

    x = None
    y = None
    z = None

    x_hat = None
    y_hat = None
    z_hat = None

    w = None
    h = None

    a = None
    b = None

    # Variables to be initialized at the beginning of the process and will
    # help to use the correct matrix operation function
    is_x_sparse = False
    is_y_sparse = False
    is_z_sparse = False
    is_model_sparse = False

    # Variables to save the initialization for the matrices.
    initialized_w = None
    initialized_h = None
    initialized_a = None
    initialized_b = None

    iterations = None
    divergence_value = None
    delta_divergence_value = None
    running_info = None

    alpha = None
    beta = None
    alpha_regularizer_w = None

    def __init__(self):
        pass

    @deprecation.deprecated("Use the function run_deconvolution_multiple with "
                            "delta_threshold and beta parameters set to zero.")
    def run_deconvolution(self, x_matrix, k, delta_threshold=0.005,
                          max_iterations=200, print_limit=100):
        """
        Run deconvolution on the input matrix using a basic decomposition setup

        Args:
            x_matrix (numpy.ndarray): The input matrix to be deconvoluted.
            k (int): Number of k components or ranks
            delta_threshold (float, optional): The convergence threshold for
            stopping the deconvolution iterations.
            Default is 0.005.
            max_iterations (int, optional): The maximum number of iterations to
            perform during deconvolution.
            Default is 200.
            print_limit (int, optional): The iteration interval at which to
            print progress messages during
            deconvolution.
            Default is 100.

        Returns:
            dec: The deconvoluted matrix.

        Raises:
            ValueError: If the input matrix `x_matrix`.
            ValueError: If the shape of the input matrix `x_matrix` is not
            compatible with the Rank `k`.
            ValueError: If `delta_threshold` is not a positive float.
            ValueError: If `max_iterations` is not a positive integer.
            ValueError: If `print_limit` is not a positive integer.

        Notes:

            - The input matrix `x_matrix` and the rank `k` should have
                compatible shapes for deconvolution to work properly.
            - The `delta_threshold` specifies the convergence threshold for
                stopping the iterations. Smaller values result  in higher
                precision but may require more iterations.
            - The `max_iterations` parameter limits the number of iterations
                to prevent infinite loops.
            - The `print_limit` parameter controls the interval at which
                progress messages are printed during the deconvolution.
        """

        print("Non-Negative multiple matrix factorization: Simple setup")

        x = np.array(x_matrix)
        # I assign the number of columns and rows
        w = np.random.rand(np.shape(x)[0], k)
        h = np.random.rand(k, np.shape(x)[1])

        # Creating the dynamic variables with the correct sizing.
        x_hat = np.zeros(shape=(np.shape(x)))
        w_new = np.zeros(shape=(np.shape(w)))
        h_new = np.zeros(shape=(np.shape(h)))

        iterations = 0
        divergence_value = 1
        delta_divergence_value = 1
        running_info = np.empty((0, 3), float)
        # The optimization is going to end until we reach a good fit or a max
        # number of iterations.
        while np.abs(delta_divergence_value) > delta_threshold and \
                iterations <= max_iterations:
            # Calculation of each main variable in the order expected
            x_hat = self._calculate_x_hat(w, h)

            divergence_matrix = self._calculate_divergence_generic(x, x_hat,
                                                                   False)
            divergence_actual_value = np.sum(divergence_matrix)
            delta_divergence_value = np.abs(divergence_actual_value -
                                            divergence_value)
            divergence_value = divergence_actual_value

            # Add an object with the values of iterations, divergence and
            # delta to plot it and see performance
            current_loop = np.array([[iterations, divergence_value,
                                      delta_divergence_value]])
            running_info = np.append(running_info, current_loop, axis=0)

            # To print the procedure status each 100 iterations
            if iterations % print_limit == 0:
                print("Iteration: ", iterations, " Divergence: ",
                      divergence_value, "Delta divergence: ",
                      delta_divergence_value)

            # Calculate the new values for the w and h matrices
            w_new = self._calculate_w_new_extended(x, x_hat, w, h)
            # I apply the constraint where the H matrix is a proportion matrix
            # for each column,
            h_new = self._calculate_h_new_extended(x, x_hat, w, h, True)

            # Assign the new values to the original variables w and h.
            w = w_new
            h = h_new

            iterations += 1

        print("The process finished!")
        print('iterations: ', iterations, ' Divergence: ', divergence_value,
              ' Delta divergence: ', delta_divergence_value)
        self.w = w_new
        self.h = h_new
        self.x_hat = x_hat
        self.iterations = iterations
        self.divergence_value = divergence_value
        self.delta_divergence_value = delta_divergence_value
        self.running_info = running_info

        return self

    # TODO Create a grid search using a alpha beta optimization for the
    #  parameters based on the divergence value
    # TODO Optimize the operations with parallelization and matrices operators.
    # TODO Add the w matrix constraint: [0,1] MeDeCoM paper.
    # TODO add verbose mode through the whole library.
    def run_deconvolution_multiple(self, x_matrix, y_matrix, z_matrix, k,
                                   gamma=1, alpha=0.0, beta=0.0,
                                   delta_threshold=1e-10, max_iterations=200,
                                   print_limit=100,
                                   proportion_constraint_h=True,
                                   regularize_w=None, alpha_regularizer_w=0,
                                   fixed_w=None, fixed_h=None, fixed_a=None,
                                   fixed_b=None, initialized_w=None,
                                   initialized_h=None, initialized_a=None,
                                   initialized_b=None,
                                   init_method_w='random_based.uniform',
                                   init_method_h='random_based.uniform',
                                   init_method_a='random_based.uniform',
                                   init_method_b='random_based.uniform',
                                   partial_w_fixed=None,
                                   partial_h_fixed=None,
                                   w_mask_fixed=None,
                                   h_mask_fixed=None,
                                   scale_w_unfixed_col=True,
                                   batches_partial_fixed=1,
                                   constraint_type_w=None,
                                   constraint_value_w=None,
                                   constraint_type_a=None,
                                   constraint_value_a=None,
                                   verbose=True):
        """
        Function that runs the src with three matrices as input. You can
        define if the model is complete or partial in terms of the input and
        therefore the output. For example, if some matrices are fixed
        (e.g., WH), and Y is provided, then the variable Gamma = 0, resulting
        in Divergence(X|WG) being zero, while the rest can still be calculated.

        Args:

            x_matrix (numpy.ndarray): Input matrix corresponding to a dataframe
            y_matrix (numpy.ndarray, optional): Input matrix.
            z_matrix (numpy.ndarray, optional): Input matrix.
            k (int): The rank used for deconvolution.
            gamma (float, optional): The gamma parameter value. Default is 1.
            alpha (float, optional): The alpha parameter value. Default is 0.0.
            beta (float, optional): The beta parameter value. Default is 0.0.
            delta_threshold (float, optional): The convergence threshold for
                stopping the deconvolution iterations.
            Default is 1e-10.
            max_iterations (int, optional): The maximum number of iterations to
                perform during deconvolution.
            Default is 200.
            print_limit (int, optional): The iteration interval at which to
                print progress messages during deconvolution
            Default is 100.
            proportion_constraint_h (bool, optional): Whether to apply a
                proportion constraint to matrix H.
            Default is True.
            regularize_w (numpy.ndarray or None, optional): The regularization
                matrix for W. Default is None.
            alpha_regularizer_w (float, optional): The alpha regularization
                parameter for W. Default is 0.
            fixed_w (numpy.ndarray or None, optional): The fixed matrix for W.
                Default is None.
            fixed_h (numpy.ndarray or None, optional): The fixed matrix for H.
                Default is None.
            fixed_a (numpy.ndarray or None, optional): The fixed matrix for A.
                Default is None.
            fixed_b (numpy.ndarray or None, optional): The fixed matrix for B.
                Default is None.
            initialized_w (numpy.ndarray or None, optional): The initial value
                for W. Default is None.
            initialized_h (numpy.ndarray or None, optional): The initial value
                for H. Default is None.
            initialized_a (numpy.ndarray or None, optional): The initial value
                for A. Default is None.
            initialized_b (numpy.ndarray or None, optional): The initial value
                for B. Default is None.
            init_method_w (str, optional): The initialization method for W.
                Default is 'random_based.uniform'.
            init_method_h (str, optional): The initialization method for H.
                Default is 'random_based.uniform'.
            init_method_a (str, optional): The initialization method for A.
                Default is 'random_based.uniform'.
            init_method_b (str, optional): The initialization method for B.
                Default
            partial_w_fixed (np.ndarray, optional): The partial fixed matrix
                for W. If None, no fixed matrix for W is used. Defaults to
                None.
            partial_h_fixed (np.ndarray, optional): The partial fixed matrix
                for H. If None, no fixed matrix for H is used. Defaults to
                None.
            w_mask_fixed (np.ndarray, optional): A mask matrix for W defining
                the positions that are fixed with TRUE values. If None, no
                mask is applied. Defaults to None.
            h_mask_fixed (np.ndarray, optional): A mask matrix for H defining
                the positions that are fixed with TRUE values. If None, no
                mask is applied. Defaults to None.
            scale_w_unfixed_col (bool, optional): Whether to apply a
                scale constraint to the unfixed columns in matrix W.
            Default is True.
            batches_partial_fixed (int, optional): Defines the number of
                batches where the procedure will inject again the partially
                fixed matrices. Defaults to 1.
            constraint_type_w : str, optional
                Type of constraint to apply to the w Matrix during
                deconvolution depending on the data used.
                Possible values are: ' ', 'power', 'exp', 'zero_one', or
                None. If None, no constraint is applied.
            constraint_value_w : float or None, optional
                The value to use for the constraint on the w Matrix.
                If None and constraint_type_w is not None, a default value will
                 be used.
            constraint_type_a : str, optional
                Type of constraint to apply to the A Matrix during
                deconvolution depending on the data used.
                Possible values are: 'sigmoid', 'power', 'exp', 'zero_one', or
                None.
                If None, no constraint is applied.
            constraint_value_a : float or None, optional
                The value to use for the constraint on the A Matrix.
                If None and constraint_type_a is not None, a default value
                will be used.
            verbose (bool, optional): Whether to display progress messages
                during deconvolution. Default is True.

        Returns:

            The deconvoluted matrix.

        Raises:

            ValueError: If any of the input matrices.
            ValueError: If the shape of the input matrices and the rank k is
                not compatible.
            ValueError: If any of the parameter values are invalid or not in
                the expected range.

        Notes:

            - The input matrices `x_matrix`, `y_matrix`, and `z_matrix` and the
                rank `k` should have compatible shapes
                for deconvolution to work properly.
            - The `gamma`, `alpha`, and `beta` parameters control the balance
                between different parts of the deconvolution process.
            - The `delta_threshold` specifies the convergence threshold for
                stopping the iterations. Smaller values result in higher
                precision but may require more iterations.
            - The `max_iterations` parameter limits the number of iterations to
                prevent infinite loops.
            - The `print_limit` parameter controls the interval at which
                progress messages are printed during deconvolution.
            - The `proportion_constraint_h` parameter determines whether to
                apply a proportion constraint to matrix H.
            - The `regularize_w` and `alpha_regularizer_w` parameters are used
                for regularization of matrix W.
            - The `fixed_w`, `fixed_h`, `fixed_a`, and `fixed_b` parameters
                allow fixing specific matrices during deconvolution.
            - The `initialized_w`, `initialized_h`, `initialized_a`, and
                `initialized_b` parameters allow providing initial values for
                the corresponding matrices.
            - The `init_method_w`, `init_method_h`, `init_method_a`, and
                `init_method_b` parameters specify the initialization method
                for the corresponding matrices.
            - The `verbose` parameter controls whether progress messages are
                displayed during deconvolution.

        """
        print(__version__)
        # 1. Now It's time to convert the matrices in a format where Zero and
        # null values are treated differently.
        if x_matrix is not None:
            x_matrix = self._standardize_sparse_matrix(x_matrix)
        if y_matrix is not None:
            y_matrix = self._standardize_sparse_matrix(y_matrix)
        if z_matrix is not None:
            z_matrix = self._standardize_sparse_matrix(z_matrix)

        # 2. Lets tests the basic constrains of the input matrices
        self._check_parameters(x_matrix=x_matrix, y_matrix=y_matrix,
                               z_matrix=z_matrix, k=k, alpha=alpha, beta=beta,
                               delta_threshold=delta_threshold,
                               max_iterations=max_iterations,
                               print_limit=print_limit,
                               proportion_constraint_h=proportion_constraint_h,
                               regularize_w=regularize_w,
                               alpha_regularizer_w=alpha_regularizer_w,
                               fixed_w=fixed_w,
                               fixed_h=fixed_h,
                               fixed_a=fixed_a,
                               fixed_b=fixed_b,
                               initialized_w=initialized_w,
                               initialized_h=initialized_h,
                               initialized_a=initialized_a,
                               initialized_b=initialized_b,
                               init_method_w=init_method_w,
                               init_method_h=init_method_h,
                               init_method_a=init_method_a,
                               init_method_b=init_method_b)

        # 3. We need to inform about the current configuration of the method in
        # terms of sparsity
        if verbose:
            print("Non-Negative multiple matrix factorization: Complex setup")
            print(self._analyse_sparsity_matrix(x_matrix, 'X'))
            print(self._analyse_sparsity_matrix(y_matrix, 'Y'))
            print(self._analyse_sparsity_matrix(z_matrix, 'Z'))

        # 4. Initialization of sparsity flag variables for matrices x, y and z
        if x_matrix is not None:
            self.is_x_sparse = self.sparsity_calculation(x_matrix,
                                                         type_return='boolean')
        if y_matrix is not None:
            self.is_y_sparse = self.sparsity_calculation(y_matrix,
                                                         type_return='boolean')
        if z_matrix is not None:
            self.is_z_sparse = self.sparsity_calculation(z_matrix,
                                                         type_return='boolean')

        # If at least one of the input matrices is sparse, the complete model
        # is sparse.
        self.is_model_sparse = self.is_x_sparse or self.is_y_sparse or \
                               self.is_z_sparse

        # Assign main matrices as the input of the model.
        x = np.array(x_matrix)

        y = None
        y_hat = None
        a = None
        a_new = None
        if alpha != 0:
            y = np.array(y_matrix)
            y_hat = np.zeros(shape=(np.shape(y)))
            a_new = np.zeros(shape=(np.shape(a)))

            # In case that I have a fixed A matrix
            if fixed_a is None:
                a = self._initialize_matrix(size_rows=np.shape(y)[0],
                                            size_columns=k,
                                            initialized_matrix=initialized_a,
                                            method=init_method_a)
                # I save the initial value of the matrix
                # (static or method-based) to help the replication of results.
                self.initialized_a = a
            else:
                a = np.array(fixed_a)

        z = None
        z_hat = None
        b = None
        b_new = None
        if beta != 0:
            z = np.array(z_matrix)
            z_hat = np.zeros(shape=(np.shape(z)))
            b_new = np.zeros(shape=(np.shape(b)))

            # In case that I have a fixed B matrix
            if fixed_b is None:
                b = self._initialize_matrix(size_rows=k,
                                            size_columns=np.shape(x)[1],
                                            initialized_matrix=initialized_b,
                                            method=init_method_b)
            else:
                b = np.array(fixed_b)
                # I save the initial value of the matrix
                # (static or method-based) to help the replication of results.
                self.initialized_b = b

        # I assign the number of columns and rows if the there is none W and H
        # matrix fixed
        if fixed_w is None:
            w = self._initialize_matrix(size_rows=np.shape(x)[0],
                                        size_columns=k,
                                        initialized_matrix=initialized_w,
                                        method=init_method_w)
            # I save the initial value of the matrix (static or method-based)
            # to help the replication of results.
            self.initialized_w = w
            w_new = np.zeros(shape=(np.shape(w)))
        else:
            w = np.array(fixed_w)
            w_new = w

        if fixed_h is None:
            h = self._initialize_matrix(size_rows=k,
                                        size_columns=np.shape(x)[1],
                                        initialized_matrix=initialized_h,
                                        method=init_method_h)
            # I save the initial value of the matrix (static or method-based)
            # to help the replication of results.
            self.initialized_h = h
            h_new = np.zeros(shape=(np.shape(h)))
        else:
            h = np.array(fixed_h)
            h_new = h

        # Partial fixed matrices and replace at the beginning through a mask
        if partial_h_fixed is not None and h_mask_fixed is not None:
            # We know that the matrix h is now all zeros.
            # Since I received the parameter with the mask, I apply it
            np.putmask(h, h_mask_fixed, partial_h_fixed)
            h_new = h

        if partial_w_fixed is not None and w_mask_fixed is not None:

            # Scale of w
            # if scale_w_unfixed_col and False:
            #     # First I will scale w_mask_fixed
            #     partial_w_fixed_scaled = self._scale(
            #         matrix=partial_w_fixed)
            #
            #     # Convert to df
            #     partial_w_fixed_scaled_df = pd.DataFrame(
            #         partial_w_fixed_scaled,
            #         index=partial_w_fixed.index,
            #         columns=partial_w_fixed.columns)
            #     partial_w_fixed = partial_w_fixed_scaled_df
            #
            #     # assign to the original array
            #     partial_w_fixed = partial_w_fixed_scaled
            #
            #     print('Fixing w with scale function')

            # We know that the matrix h is now all zeros.
            # Since I received the parameter with the mask, I apply it
            np.putmask(w, w_mask_fixed, partial_w_fixed)
            w_new = w

        # TODO: check this validations: for a grid_search is not working!
        # In case of some combinations of fixed matrices that get the
        # (gamma, alpha, beta) = 0
        # gamma, alpha, beta = \
        #     self.check_fixed_matrices_gamma_alpha_beta(fixed_w=fixed_w,
        #                                                fixed_h=fixed_h,
        #                                                fixed_a=fixed_a,
        #                                                fixed_b=fixed_b,
        #                                                gamma=gamma,
        #                                                alpha=alpha,
        #                                                beta=beta)

        print("Values of parameters: gamma: ", str(gamma), ', alpha: ',
              str(alpha), ', beta: ', str(beta),
              ', constraint_type_w: ', constraint_type_w,
              ', constraint_value_w: ', constraint_value_w,
              ', constraint_type_a: ', constraint_type_a,
              ', constraint_value_a: ', constraint_value_a)

        # Creating the dynamic variables with the correct sizing.
        x_hat = np.zeros(shape=(np.shape(x)))

        iterations = 0
        divergence_value = 1
        delta_divergence_value = 1
        running_info = np.empty((0, 6), float)
        # The optimization is going to end until we reach a good fit or a max
        # number of iterations.
        while np.abs(delta_divergence_value) > delta_threshold and \
                iterations < max_iterations:

            # Calculation of all related with X. In this case, if gamma is 0,
            # the matrices WH are fixed,
            # therefore it could be a optimization over A and B
            divergence_matrix_x = None
            divergence_actual_value_x = -1
            if gamma != 0:
                x_hat = self._calculate_x_hat(w, h)
                divergence_matrix_x = \
                    self._calculate_divergence_generic(x, x_hat,
                                                       self.is_x_sparse)
                divergence_actual_value_x = np.sum(divergence_matrix_x)

            divergence_matrix_y = None
            divergence_actual_value_y = -1
            if alpha != 0:
                y_hat = self._calculate_y_hat(a, h)
                divergence_matrix_y = \
                    self._calculate_divergence_generic(y, y_hat,
                                                       self.is_y_sparse)
                divergence_actual_value_y = np.sum(divergence_matrix_y)

            divergence_matrix_z = None
            divergence_actual_value_z = -1
            if beta != 0:
                z_hat = self._calculate_z_hat(w, b)
                divergence_matrix_z = \
                    self._calculate_divergence_generic(z, z_hat,
                                                       self.is_z_sparse)
                divergence_actual_value_z = np.sum(divergence_matrix_z)

            # Calculation of total divergence with the multiple matrices
            # including the gamma variable.
            divergence_actual_value = (gamma * divergence_actual_value_x) + \
                                      (alpha * divergence_actual_value_y) + \
                                      (beta * divergence_actual_value_z)

            delta_divergence_value = np.abs(divergence_actual_value -
                                            divergence_value)
            divergence_value = divergence_actual_value

            # Add an object with the values of iterations,
            # divergence (x,y,z and total) and delta to plot it and see
            # performance
            current_loop = np.array([[iterations,
                                      divergence_actual_value_x,
                                      divergence_actual_value_y,
                                      divergence_actual_value_z,
                                      divergence_value,
                                      delta_divergence_value]]).astype(float)

            running_info = np.append(running_info, current_loop, axis=0)

            # To print the procedure status each 100 iterations
            if iterations % print_limit == 0:
                if verbose:
                    print("Iteration: ", iterations, " Divergence: ",
                          divergence_value,
                          "Delta divergence: ", delta_divergence_value)

            if alpha != 0:
                # I apply the constraint where the H matrix is a proportion
                # matrix for each column,
                if fixed_h is None:
                    h_new = self._calculate_h_new_extended_alpha_beta_generic(
                        x, x_hat, w, h, alpha,
                        y, y_hat, a, proportion_constraint_h,
                        self.is_model_sparse,
                        h_mask_fixed)
                # I will calculate the new A matrix if there is not fixed
                # matrix.
                if fixed_a is None:
                    a_new = self._calculate_a_new_extended_generic(
                        y, y_hat, a, h, self.is_model_sparse,
                        constraint_type_a, constraint_value_a)
                    a = a_new
            elif fixed_h is None:
                h_new = self._calculate_h_new_extended_alpha_beta_generic(
                    x, x_hat, w, h, alpha, None, None, None,
                    proportion_constraint_h, self.is_model_sparse,
                    h_mask_fixed)

            if beta != 0:
                # Calculate the new values for the w and h matrices
                if fixed_w is None:
                    w_new = self._calculate_w_new_extended_alpha_beta_generic(
                        x, x_hat, w, h, beta, z, z_hat, b, regularize_w,
                        alpha_regularizer_w, self.is_model_sparse,
                        constraint_type_w, constraint_value_w,
                        scale_w_unfixed_col=scale_w_unfixed_col,
                        w_mask_fixed=w_mask_fixed
                    )

                if fixed_b is None:
                    b_new = self._calculate_b_new_extended_generic(
                        z, z_hat, b, w, self.is_model_sparse)
                    b = b_new
            elif fixed_w is None:
                w_new = self._calculate_w_new_extended_alpha_beta_generic(
                    x, x_hat, w, h, beta, None, None, None, regularize_w,
                    alpha_regularizer_w, self.is_model_sparse,
                    constraint_type_w, constraint_value_w,
                    scale_w_unfixed_col=scale_w_unfixed_col,
                    w_mask_fixed=w_mask_fixed
                )

            # Assign the new values to the original variables w and h if there
            # is none fixed matrix
            if fixed_w is None:
                w = w_new

            if fixed_h is None:
                h = h_new

            # To apply the partial replacement with the partial_references
            # using the mask provided as parameter
            if iterations % batches_partial_fixed == 0:
                # Addition of compatibility with partial fixed references.
                if partial_h_fixed is not None and h_mask_fixed is not None:
                    # We know that the matrix h is now all zeros.
                    # Since I received the parameter with the mask, I apply it
                    np.putmask(h, h_mask_fixed, partial_h_fixed)

                if partial_w_fixed is not None and w_mask_fixed is not None:
                    # We know that the matrix w is now all zeros.
                    # Since I received the parameter with the mask, I apply it
                    np.putmask(w, w_mask_fixed, partial_w_fixed)

            # I iterate to the next step.
            iterations += 1

        print("The process finished!")
        # Let's print the reason because the process finishes.
        if np.abs(delta_divergence_value) < delta_threshold:
            if verbose:
                print("delta_threshold stop. Current limit: ", delta_threshold,
                      ", Actual value: ", np.abs(delta_divergence_value))
        else:
            if verbose:
                print("Iterations stop. Current limit: ", max_iterations,
                      ", Actual value: ", iterations)

        if verbose:
            print('Final number of iterations: ', iterations, ' Divergence: ',
                  divergence_value, ' Delta divergence: ',
                  delta_divergence_value)

        # Call the function to save the  variables globally.
        self._assign_global_variables \
            (fixed_w=fixed_w, fixed_h=fixed_h,
             fixed_a=fixed_a, fixed_b=fixed_b,
             partial_w_fixed=partial_w_fixed,
             partial_h_fixed=partial_h_fixed,
             x=x_matrix, y=y_matrix, z=z_matrix,
             x_hat=x_hat, y_hat=y_hat, z_hat=z_hat,
             w_new=w_new, h_new=h_new, a_new=a_new,
             b_new=b_new, running_info=running_info,
             alpha=alpha, beta=beta, k=k,
             iterations=iterations,
             divergence_value=divergence_value,
             delta_divergence_value=delta_divergence_value,
             alpha_regularizer_w=alpha_regularizer_w)

        return self

    def _assign_global_variables(self, fixed_w, fixed_h, fixed_a, fixed_b,
                                 partial_w_fixed, partial_h_fixed,
                                 x, y, z, x_hat, y_hat, z_hat, w_new, h_new,
                                 a_new, b_new, running_info, alpha, beta, k,
                                 iterations, divergence_value,
                                 delta_divergence_value, alpha_regularizer_w):
        """
        Assigns global variables in the class instance using the given
        parameters. It handles NaN values by replacing them with -1 to avoid
        errors, especially for compatibility with R.

        Args:

            fixed_w, fixed_h, fixed_a, fixed_b (type): These are flags
                indicating whether the respective matrices are fixed.
            partial_w_fixed, partial_w_fixed (np.ndarray, optional): The
                partial fixed matrix for H and W. If None, no fixed matrix for
                 H and W is used. Defaults to None.
            x, y, z (DataFrame): Input data matrices.
            x_hat, y_hat, z_hat (DataFrame): Estimated data matrices.
            w_new, h_new, a_new, b_new (DataFrame): Newly calculated matrices
                W, H, A, B.
            running_info (np.ndarray): Information collected during the
                algorithm's run such as iteration number and divergence values.
            alpha, beta (float): Parameters of the deconvolution model.
            k (int): Number of clusters/groups (rank).
            iterations (int): Number of iterations performed.
            divergence_value (float): Value of divergence.
            delta_divergence_value (float): Change in divergence value.
            alpha_regularizer_w (float): Regularization parameter for W matrix.

        Returns:
            None. The method updates class instance attributes.
        """

        # Calculating column names depending on the fixed matrices.
        k_values = self._calculate_k_values_labels(
            fixed_w=fixed_w,
            fixed_h=fixed_h,
            fixed_a=fixed_a,
            fixed_b=fixed_b,
            partial_w_fixed=partial_w_fixed,
            partial_h_fixed=partial_h_fixed,
            k=k)

        self.x = pd.DataFrame(x)
        self.y = pd.DataFrame(y)
        self.z = pd.DataFrame(z)

        self.alpha = alpha
        self.beta = beta
        self.alpha_regularizer_w = alpha_regularizer_w
        # TODO: Apply the same treatment to the other additional variables of
        #  matrix type, as they lack
        #  column and row names.
        self.x_hat = pd.DataFrame(data=np.nan_to_num(x_hat, nan=-1),
                                  index=self.x.index.values,
                                  columns=self.x.columns.values)
        self.w = pd.DataFrame(data=np.nan_to_num(w_new, nan=-1),
                              index=self.x.index.values,
                              columns=k_values)

        self.h = pd.DataFrame(data=np.nan_to_num(h_new, nan=-1),
                              index=k_values,
                              columns=self.x.columns.values)

        if alpha != 0:
            self.y_hat = pd.DataFrame(data=np.nan_to_num(y_hat, nan=-1),
                                      index=self.y.index.values,
                                      columns=self.y.columns.values)
            self.a = pd.DataFrame(data=np.nan_to_num(a_new, nan=-1),
                                  index=self.y.index.values,
                                  columns=k_values)

        if beta != 0:
            self.z_hat = pd.DataFrame(data=np.nan_to_num(z_hat, nan=-1),
                                      index=self.z.index.values,
                                      columns=self.z.columns.values)
            self.b = pd.DataFrame(data=np.nan_to_num(b_new, nan=-1),
                                  index=k_values,
                                  columns=self.z.columns.values)

        # I replace the nan values by -1 in order to avoid error in R.
        self.running_info = pd.DataFrame(
            data=np.nan_to_num(running_info, nan=-1),
            index=running_info[:, 0],
            columns=np.array(['iteration', 'divergence_value_x',
                              'divergence_value_y', 'divergence_value_z',
                              'divergence_value', 'delta_divergence_value']))

        self.iterations = iterations
        self.divergence_value = np.nan_to_num(divergence_value, nan=-1)
        self.delta_divergence_value = np.nan_to_num(delta_divergence_value,
                                                    nan=-1)

    # TODO: Implement unit tests for the _calculate_k_values_labels function.
    def _calculate_k_values_labels(self, fixed_w, fixed_h, fixed_a, fixed_b,
                                   partial_w_fixed, partial_h_fixed,
                                   k):
        """
        This function calculates the labels for the k-values, which are based
        on the columns of input matrices, namely W, A, H, and B. These labels
        are required for ranking and denoting the k-values.

        The function behavior depends on the existence of the input matrices:
        1. If matrix W is provided, the labels will be derived from the column
        names of matrix W. Additionally, the rows of matrices H and B and the
        columns of matrix A will be assigned these column names.
        2. If only matrix A is provided, the labels will be the column names
        of matrix A. The column names of matrix W, as well as the row names for
        matrices H and B, will be assigned these column names.
        3. If both matrices W and A are provided, their column names should
        match. In this case, the labels will be derived from the column names
        of matrix W, and the rows of matrices H and B will be assigned these
        column names.

        If none of the matrices are provided, a default set of labels will be
        generated. Each label will be in the format 'rank_k', where k
        represents the rank.

        Please note that the function will also return an error message if none
        of the matrices are fixed, or a notification if at least one matrix is
        fixed.

        Parameters:

            fixed_w (pd.DataFrame): Fixed W matrix with labels.
            fixed_h (pd.DataFrame): Fixed H matrix with labels.
            fixed_a (pd.DataFrame): Fixed A matrix with labels.
            fixed_b (pd.DataFrame): Fixed B matrix with labels.
            partial_w_fixed (np.ndarray, optional): The partial fixed matrix
                for W. If None, no fixed matrix for W is used. Defaults to
                None.
            partial_h_fixed (np.ndarray, optional): The partial fixed matrix
                for H. If None, no fixed matrix for H is used. Defaults to
                None.
            k (int): The number of ranks for which labels need to be generated
                when no fixed matrix is provided.

        Returns:
            k_values (np.array): An array of strings representing labels for
            the k-values.

        Note: This function needs to be unit tested.
        """

        k_values = []
        if fixed_w is None and fixed_h is None and fixed_h is None and \
                fixed_a is None and fixed_b is None and \
                partial_w_fixed is None and partial_h_fixed is None:
            print("There aren't fixed matrices...")
            # Creation of vector for the k ranks
            for k in range(k):
                current_string = 'rank_' + str(k)
                k_values = np.append(k_values, np.array([current_string]),
                                     axis=0)
        else:
            print("There is at least one fixed matrix...")
            if fixed_w is not None:
                k_values = fixed_w.columns.values
            elif fixed_a is not None:
                k_values = fixed_a.columns.values
            elif fixed_h is not None:
                k_values = fixed_h.index.values
            elif fixed_b is not None:
                k_values = fixed_b.index.values
            elif partial_w_fixed is not None:
                k_values = partial_w_fixed.columns.values
            elif partial_h_fixed is not None:
                k_values = partial_h_fixed.index.values

        return k_values

    def _proportion_constraint_h(self, h, h_mask_fixed=None,
                                 known_proportions_constrain=False):
        """
        This function calculates the proportion of each element in the input
        matrix 'h' along the columns. Each value in 'h' is divided by the sum
        of its corresponding column, generating a new matrix where the sum of
        each column is equal to 1. This results in a proportional distribution
        of the column's values.

        Parameters:

            h (pd.DataFrame or np.ndarray): Input matrix. Elements should be
                numeric.
            h_mask_fixed (np.ndarray, optional): A mask matrix for H defining
                the positions that are fixed with TRUE values. If None, no
                mask is applied. Defaults to None.

        Returns:

            h_proportions (pd.DataFrame or np.ndarray): Proportional matrix
                derived from 'h', where each value is a proportion of the total
                sum of its column.
        """

        h_proportions = None

        # This is temp, since I think always that I activate the constrain
        # all cell-types must sum up to 1, however I leave open the
        # possibility of summing up to 1 just the known cell-types which at
        # the first glance doesn't make sense. In the future I will delete
        # the extended part.
        if not known_proportions_constrain:
            h_proportions = h / h.sum(axis=0)
        else:

            # 0. Check which rows are the fixed ones (TRUE rows)
            unknown_cell_type_name = []
            known_rows_index = []
            for rows_mask in h_mask_fixed.index:
                index = h_mask_fixed.index.get_loc(rows_mask)
                if np.all(h_mask_fixed.iloc[index, :]):
                    unknown_cell_type_name.append(rows_mask)
                else:
                    known_rows_index.append(index)

            # Since the h is a np.array, I will convert it to df
            h_df = pd.DataFrame(data=h,
                                index=h_mask_fixed.index,
                                columns=h_mask_fixed.columns)

            # 1. get the limited part of the data in H that I want to re-scale.
            h_temp = h_df.drop(unknown_cell_type_name)

            # 2. Scale the data and put it in a matrix with the same size of
            # the target matrix, otherwise can be wrong assignments.
            h_proportions_temp = h_temp / h_temp.sum(axis=0)
            h_proportions_complete = np.zeros(np.shape(h), dtype=float)
            h_proportions_complete[known_rows_index, :] = h_proportions_temp

            # 3. Create the mask with all true
            mask_without_unknown = np.ones(np.shape(h), dtype=bool)

            # 4. Now I will take the index for each unknown row
            unknown_index = []
            for unknown_counter in unknown_cell_type_name:
                unknown_index.append(h_df.index.get_loc(unknown_counter))

            mask_without_unknown[unknown_index, :] = False
            h_proportions = h.copy()

            # 5. Reassign the data to the H matrix
            np.putmask(h_proportions, mask_without_unknown,
                       h_proportions_complete)

        return h_proportions

    def _calculate_x_hat(self, w, h):
        """
        This function calculates the matrix product of the input matrices 'w'
        and 'h'.

        The operation performed is a dot product (matrix multiplication), where
        the product of 'w' and 'h' results in a new matrix 'x_hat'.

        Parameters:

            w (pd.DataFrame or np.ndarray): First input matrix. It should be
            numeric and compatible for multiplication with 'h'.
            h (pd.DataFrame or np.ndarray): Second input matrix. It should be
                numeric and compatible for multiplication with 'w'.

        Returns:

            x_hat (pd.DataFrame or np.ndarray): The product matrix resulting
                from the dot product of 'w' and 'h'.
        """

        x_hat = w.dot(h)
        return x_hat

    def _calculate_y_hat(self, a, h):
        """
        This function calculates the matrix product of the input matrices 'a'
        and 'h'.

        The operation performed is a dot product (matrix multiplication),
        where the product of 'a' and 'h' results in a new matrix 'y_hat'.

        Parameters:

            a (pd.DataFrame or np.ndarray): First input matrix. It should be
                numeric and compatible for multiplication with 'h'.
            h (pd.DataFrame or np.ndarray): Second input matrix. It should be
                numeric and compatible for multiplication with 'a'.

        Returns:

            y_hat (pd.DataFrame or np.ndarray): The product matrix resulting
                from the dot product of 'a' and 'h'.
        """

        y_hat = a.dot(h)
        return y_hat

    def _calculate_z_hat(self, w, b):
        """
        This function calculates the matrix product of the input matrices 'w'
        and 'b'.

        The operation performed is a dot product (matrix multiplication), where
        the product of 'w' and 'b' results in a new matrix 'z_hat'.

        Parameters:

            w (pd.DataFrame or np.ndarray): First input matrix. It should be
                numeric and compatible for multiplication with 'b'.
            b (pd.DataFrame or np.ndarray): Second input matrix. It should be
                numeric and compatible for multiplication with 'w'.

        Returns:

            z_hat (pd.DataFrame or np.ndarray): The product matrix resulting
                from the dot product of 'w' and 'b'.
        """

        z_hat = w.dot(b)
        return z_hat

    def _calculate_x_hat_extended(self, w, h):
        """
        This function calculates the matrix product of the input matrices 'w'
        and 'h' using explicit loops instead of direct matrix multiplication.
        The main idea was to create it in the more basic form to be sure that
        results are correct. After make sure that everything is correct, an
        optimization will be generated.

        The function first initializes an all-zeros matrix 'x_hat' of size
        (rows of 'w' x columns of 'h'). It then iterates through each element
        of 'x_hat', and for each, computes the dot product of the corresponding
        row of 'w' and column of 'h', which is then assigned to the respective
        element in 'x_hat'.

        Parameters:

            w (np.ndarray): First input matrix. It should be numeric and its
                number of columns should match the number of rows in 'h'.
            h (np.ndarray): Second input matrix. It should be numeric and its
                number of rows should match the number of columns in 'w'.

        Returns:

            x_hat (np.ndarray): The product matrix resulting from the explicit
                calculation of the matrix multiplication
                of 'w' and 'h'.
        """

        x_hat = np.zeros(shape=(np.shape(w)[0], np.shape(h)[1]))

        for i in range(np.shape(w)[0]):
            for j in range(np.shape(h)[1]):

                value_temp = 0
                for k in range(np.shape(w)[1]):
                    value_temp = value_temp + w[i][k] * h[k][j]

                x_hat[i][j] = value_temp

        return x_hat

    def _calculate_w_new_extended(self, x, x_hat, w, h):
        """
        This function calculates a new 'w' matrix using an extended version of
        the update rule often used in algorithms such as Non-negative Matrix
        Factorization. The rule involves division, multiplication, and summing
        operations on the matrices 'w', 'x', 'x_hat', and 'h'.

        The function first initializes an all-zeros matrix 'w_new' of the same
        size as 'w'. Then, for each element of 'w_new', it computes a fraction
        where the numerator is a weighted sum of elements from 'x' and 'h' and
        the denominator is the sum of elements from 'h'. This fraction is then
        multiplied by the corresponding element from 'w' to get the updated
        value for 'w_new'.

        Please note, this function does not currently account for sparseness
        (zero values) in its computation. This should be considered when
        enhancing the function.

        Parameters:

            x (np.ndarray): The 'x' matrix in the current context. It should
                be numeric and compatible with 'x_hat' for
                division.
            x_hat (np.ndarray): The 'x_hat' matrix as calculated in the current
                context. It should be numeric and compatible with 'x' for
                division.
            w (np.ndarray): The initial 'w' matrix. It should be numeric.
            h (np.ndarray): The 'h' matrix in the current context. It should be
                numeric.

        Returns:

            w_new (np.ndarray): The updated 'w' matrix as calculated by the
                function's rule.

        Note: This function needs to be updated to handle sparseness
            (zero values).
        """

        w_new = np.zeros(shape=(np.shape(w)))

        for i in range(np.shape(w)[0]):
            for k in range(np.shape(w)[1]):

                # TODO: Implement sparsity constraint (zero values) for the
                #  data. Add the necessary constraints.
                up_temp = 0
                for j in range(np.shape(h)[1]):
                    up_temp = up_temp + self.division(x[i][j],
                                                      x_hat[i][j]) * h[k][j]

                down_temp = 0
                for j in range(np.shape(h)[1]):
                    down_temp = down_temp + h[k][j]

                w_new[i][k] = w[i][k] * self.division(up_temp, down_temp)

        return w_new

    def _calculate_w_new_extended_alpha_beta_generic(self, x, x_hat, w, h,
                                                     beta, z, z_hat, b,
                                                     regularize_w,
                                                     alpha_regularizer_w=0,
                                                     is_sparse=False,
                                                     constraint_type_w=None,
                                                     constraint_value_w=None,
                                                     scale_w_unfixed_col=True,
                                                     w_mask_fixed=None):
        """
        This function calculates a new 'w' matrix using a generic variant of an
        extended update rule that could be applied in algorithms such as
        Non-negative Matrix Factorization. The function supports both regular
        and sparse computations based on the provided input matrices and
        parameters.

        The function determines whether to use the sparse or regular
        computation based on the 'is_sparse' boolean flag.
        If 'is_sparse' is True, the
        '_calculate_w_new_extended_alpha_beta_sparse' method is invoked, and if
        it's False, the '_calculate_w_new_extended_alpha_beta' method is
        called. Both methods are expected to take the same arguments as this
        function.

        Please note, the methods '_calculate_w_new_extended_alpha_beta_sparse'
        and '_calculate_w_new_extended_alpha_beta' are not included in this
        function and need to be implemented separately.

        Parameters:

            x (Any): The 'x' matrix in the current context. It should be
                numeric and compatible with 'x_hat' for division.
            x_hat (Any): The 'x_hat' matrix as calculated in the current
                context. It should be numeric and compatible with 'x' for
                division.
            w (Any): The initial 'w' matrix. It should be numeric.
            h (Any): The 'h' matrix in the current context. It should be
                numeric.
            beta (float): The beta regularization parameter for the update rule
            z (Any): The 'z' matrix in the current context. It should be
                numeric and compatible with 'z_hat' for division.
            z_hat (Any): The 'z_hat' matrix as calculated in the current
                context. It should be numeric and compatible with 'z' for
                division.
            b (Any): The 'b' matrix in the current context. It should be
                numeric.
            regularize_w (Any): The regularization parameter for the 'w' matrix
            alpha_regularizer_w (float, optional): The alpha regularization
                parameter for the 'w' matrix. Default is 0.
            constraint_type_w : str, optional
                Type of constraint to apply to the W Matrix during
                deconvolution depending on the data used.
                Possible values are: 'sigmoid', 'power', 'exp', 'zero_one', or
                None. If None, no constraint is applied.
            constraint_value_w : float or None, optional
                The value to use for the constraint on the W Matrix.
                If None and constraint_type_w is not None, a default value will
                be used.
            is_sparse (bool, optional): A flag to determine whether to perform
                the sparse or regular computation. Default is False.
            scale_w_unfixed_col (bool, optional): Whether to apply a
                scale constraint to the unfixed columns in matrix W.
                Default is True.
            w_mask_fixed (np.ndarray, optional): A mask matrix for W defining
                the positions that are fixed with TRUE values. If None, no
                mask is applied. Defaults to None.

        Returns:

            w_new (np.ndarray): The updated 'w' matrix as calculated by the
                selected method.

        Note: The methods '_calculate_w_new_extended_alpha_beta_sparse' and
            '_calculate_w_new_extended_alpha_beta' need to be implemented
            separately.
        """

        w_new = np.zeros(shape=(np.shape(w)))

        if is_sparse:
            w_new = self._calculate_w_new_extended_alpha_beta_sparse(
                x, x_hat, w, h, beta, z, z_hat,
                b, regularize_w, alpha_regularizer_w)
        else:
            w_new = self._calculate_w_new_extended_alpha_beta(
                x, x_hat, w, h, beta, z, z_hat,
                b, regularize_w, alpha_regularizer_w)

        # Checking if there is a constraint for the W
        if constraint_type_w is not None:
            w_new_scaled = self.scale_matrix(matrix=w_new,
                                             scale_type=constraint_type_w,
                                             value=constraint_value_w)
            w_new = w_new_scaled

        # The main idea is to scale the unknown columns without centering the
        # the values.
        if scale_w_unfixed_col and w_mask_fixed is not None:
            print('Class of w_new:', type(w_new).__name__)
            # Assign the new scale reference to the w_new variable
            w_new = self._reference_scale_w(w=w, w_mask_fixed=w_mask_fixed)

        return w_new

    def _reference_scale_w(self, w, w_mask_fixed):
        # In this part there are two possibilities:
        # 1. Complete columns for known and unknown cell-types. In this case
        #    the algorithm works fine.
        # 2. For known cell-types, I have extra markers that are not part of my
        #    reference, therefore have to be opened, however Do I need to
        #    scale those as well??? this is an open question.

        print('Class of w:', type(w).__name__)
        print('Class of w_mask_fixed:', type(w_mask_fixed).__name__)

        # 0. Check which rows are the fixed ones (TRUE rows)
        unknown_cell_type_name = []
        unknown_column_index = []
        known_cell_type_name = []
        known_column_index = []
        for columns_mask in w_mask_fixed.columns:
            index = w_mask_fixed.columns.get_loc(columns_mask)
            if np.all(w_mask_fixed.iloc[:, index]):
                known_cell_type_name.append(columns_mask)
                known_column_index.append(index)
            else:
                unknown_cell_type_name.append(columns_mask)
                unknown_column_index.append(index)

        # Since the w is a np.array, I will convert it to df
        w_df = pd.DataFrame(data=w,
                            index=w_mask_fixed.index,
                            columns=w_mask_fixed.columns)

        # 1. get the limited part of the data in W that I want to
        # re-scale: unknown
        print('known_cell_type_name: ', known_cell_type_name)
        w_unfixed_temp = w_df.drop(known_cell_type_name, axis=1)
        print('w_unfixed_temp: ', w_unfixed_temp)

        # 2. Scale the data  and put it in a matrix with the same size of
        # the target matrix, otherwise can be wrong assignments.
        scaled_data_w = self._scale(matrix=w_unfixed_temp.to_numpy())
        print('scaled_data_w: ', scaled_data_w)

        # 3. Assignment of the scaled matrix
        scaled_data_w_complete = np.zeros(np.shape(w), dtype=float)
        scaled_data_w_complete[:, unknown_column_index] = scaled_data_w

        # 4. Create the mask with all true
        mask_with_unknown = np.ones(np.shape(w), dtype=bool)

        # 5. Now I will take the index for each known column
        known_columns = []
        for known_counter in known_cell_type_name:
            known_columns.append(w_df.columns.get_loc(known_counter))

        mask_with_unknown[:, known_columns] = False

        # 6. I add the w reference to replace in it the scaled values
        w_reference = w

        # 5. Reassign the data to the H matrix
        np.putmask(w_reference,
                   mask_with_unknown,
                   scaled_data_w_complete)

        # 6. Finally I will convert to df
        # w_reference_df = pd.DataFrame(data=w_reference,
        #                               index=w.index,
        #                               columns=w.columns)

        return w_reference

    def _scale(self, matrix, version_scale='3'):

        print('Matrix to scale:', np.shape(matrix))
        print(matrix)

        scaled_matrix = None
        if version_scale == '1':
            scaler = StandardScaler(with_mean=False, with_std=True)
            scaled_matrix = scaler.fit_transform(matrix)
        elif version_scale == '2':
            scaled_matrix = preprocessing.scale(matrix, axis=0,
                                                with_mean=False, with_std=True)
        elif version_scale == '3':
            scaled_matrix = matrix
            for counter_columns in range(np.shape(matrix)[1]):
                rms = np.sqrt(np.mean(matrix[:, counter_columns] ** 2))
                scaled_matrix[:, counter_columns] = scaled_matrix[:,
                                                    counter_columns] / rms

        return scaled_matrix

    def _calculate_w_new_extended_alpha_beta(self, x, x_hat, w, h, beta,
                                             z, z_hat, b, regularize_w,
                                             alpha_regularizer_w=0):
        """
        This function calculates a new 'w' matrix using a comprehensive
        extended update rule that could be applied in algorithms such as
        Non-negative Matrix Factorization. The rule involves complex
        computations based on the input matrices, regularization parameters,
        and an optional sparsity constraint.

        The function first initializes an all-zeros matrix 'w_new' of the same
        size as 'w'. Then, for each element of 'w_new', it computes a complex
        fraction where the numerator is a sum of two terms - one based on 'x'
        and 'h', and another based on 'z', 'b', and 'beta'. The denominator is
        also a sum of two terms - one based on 'h' and another based on 'b',
        'beta', and optional regularization based on 'alpha_regularizer_w'.

        In the computation, the function supports three types of regularizer
        functions for 'w' - 'lasso_sparcity', 'medecom_soft_binary', and
        'medecom_soft_binary_derived'. After the computation, the function may
        also apply additional regularization to 'w_new' based on
        'regularize_w'.

        Parameters:

            x (np.ndarray): The 'x' matrix in the current context. It should be
                numeric and compatible with 'x_hat' for division.
            x_hat (np.ndarray): The 'x_hat' matrix as calculated in the current
                context. It should be numeric and compatible with 'x' for
                division.
            w (np.ndarray): The initial 'w' matrix. It should be numeric.
            h (np.ndarray): The 'h' matrix in the current context. It should be
                numeric.
            beta (float): The beta regularization parameter for the update
                rule.
            z (np.ndarray): The 'z' matrix in the current context. It should be
                numeric and compatible with 'z_hat' for division.
            z_hat (np.ndarray): The 'z_hat' matrix as calculated in the current
                context. It should be numeric and compatible with 'z' for
                division.
            b (np.ndarray): The 'b' matrix in the current context. It should be
                numeric.
            regularize_w (float): The regularization parameter for the 'w'
                matrix.
            alpha_regularizer_w (float, optional): The alpha regularization
                parameter for the 'w' matrix. Default is 0.

        Returns:

            w_new (np.ndarray): The updated 'w' matrix as calculated by the
                function's rule.

        Note: The function does not currently enforce a constraint on the
            'beta' parameter being between 0 and 1. This should be considered
            when enhancing the function.
        """

        # TODO: Apply a constraint to ensure that the value of beta remains
        #  between 0 and 1.
        w_new = np.zeros(shape=(np.shape(w)))

        for i in range(np.shape(w)[0]):
            for k in range(np.shape(w)[1]):
                up_temp = 0
                up_temp_first = 0
                for j in range(np.shape(h)[1]):
                    up_temp_first = up_temp_first + \
                                    self.division(x[i][j],
                                                  x_hat[i][j]) * h[k][j]

                # If running a simple NMF with beta=0, skip executing this
                # code.
                up_temp_second = 0
                if beta != 0:
                    for m in range(np.shape(b)[1]):
                        # Verify the equation in this part. The original paper
                        # (Takeuchi et al, 2013) has an issue where i and k are
                        # reversed, which seems incorrect.
                        up_temp_second = up_temp_second + \
                                         self.division(z[i][m],
                                                       z_hat[i][m]) * b[k][m]

                up_temp = up_temp_first + (beta * up_temp_second)

                down_temp = 0
                down_temp_first = 0
                for j in range(np.shape(h)[1]):
                    down_temp_first = down_temp_first + h[k][j]

                down_temp_second = 0
                if beta != 0:
                    for m in range(np.shape(b)[1]):
                        down_temp_second = down_temp_second + b[k][m]

                regularization_part = 0
                if alpha_regularizer_w != 0:
                    regularizer_function_type = 'medecom_soft_binary'

                    if regularizer_function_type == 'lasso_sparcity':
                        # Sparcity
                        regularization_part = alpha_regularizer_w * 1
                    elif regularizer_function_type == 'medecom_soft_binary':
                        # Direct regularizator from MeDeCom
                        regularization_part = alpha_regularizer_w * \
                                              (w[i][k] * (1 - w[i][k]))
                    elif regularizer_function_type == \
                            'medecom_soft_binary_derived':
                        # Derived regularizator from MeDeCom: based on
                        # the function x(1-x)
                        regularization_part = alpha_regularizer_w * \
                                              (1 - 2 * w[i][k])

                down_temp = down_temp_first + \
                            (beta * down_temp_second) + \
                            regularization_part

                w_new[i][k] = w[i][k] * self.division(up_temp, down_temp)

        if regularize_w is not None:
            w_new = self._regularize_w(w_new, regularization_type=regularize_w)

        return w_new

    def _calculate_w_new_extended_alpha_beta_sparse(self, x, x_hat, w, h,
                                                    beta, z, z_hat, b,
                                                    regularize_w,
                                                    alpha_regularizer_w=0):
        """
        This function calculates a new 'w' matrix using a comprehensive
        extended update rule designed for sparse matrices. The rule involves
        complex computations based on the input matrices, regularization
        parameters, and an optional sparsity constraint.

        This function performs the same calculations as
        `_calculate_w_new_extended_alpha_beta` but specifically handles
        sparse input matrices 'x' and 'z'. If these inputs are not in the
        Compressed Sparse Row (csr_matrix) format, they are converted to it.
        The function then checks if each computed term is a non-NaN number
        before including it in the sum for the numerator or denominator of the
        complex fraction.

        Parameters:

            x (np.ndarray or csr_matrix): The 'x' matrix in the current
                context. It should be numeric and compatible with 'x_hat' for
                division.
            x_hat (np.ndarray): The 'x_hat' matrix as calculated in the current
                context. It should be numeric and compatible with 'x' for
                division.
            w (np.ndarray): The initial 'w' matrix. It should be numeric.
            h (np.ndarray): The 'h' matrix in the current context. It should be
                numeric.
            beta (float): The beta regularization parameter for the update
                rule.
            z (np.ndarray or csr_matrix): The 'z' matrix in the current
                context. It should be numeric and compatible with 'z_hat' for
                division.
            z_hat (np.ndarray): The 'z_hat' matrix as calculated in the current
                context. It should be numeric and compatible with 'z' for
                division.
            b (np.ndarray): The 'b' matrix in the current context. It should
                be numeric.
            regularize_w (float): The regularization parameter for the 'w'
                matrix.
            alpha_regularizer_w (float, optional): The alpha regularization
                parameter for the 'w' matrix. Default is 0.

        Returns:

            w_new (np.ndarray): The updated 'w' matrix as calculated by the
                function's rule.

        Note:
        - The function does not currently enforce a constraint on the 'beta'
        parameter being between 0 and 1. This
        should be considered when enhancing the function.
        - There is a potential issue with applying the `_regularize_w` function
        to sparse data, which may need to be resolved.
        """

        # TODO: Apply a constraint to ensure that the value of beta is within
        #  the range of 0 and 1.
        w_new = np.zeros(shape=(np.shape(w)))

        # The input X and Y must be sparse always
        if type(x) is not csr_matrix:
            x = csr_matrix(x)
        if type(z) is not csr_matrix and z is not None:
            z = csr_matrix(z)

        for i in range(np.shape(w)[0]):
            for k in range(np.shape(w)[1]):
                up_temp = 0
                up_temp_first = 0
                for j in range(np.shape(h)[1]):
                    up_temp_first_complement = (x[i, j] /
                                                x_hat[i][j]) * h[k][j]
                    if not math.isnan(up_temp_first_complement):
                        up_temp_first = up_temp_first + \
                                        up_temp_first_complement

                # In case that we are running a simple NMF with beta=0
                # I shouldn't run the code.
                up_temp_second = 0
                if beta != 0:
                    for m in range(np.shape(b)[1]):
                        # Verify the equation in this part. The original paper
                        # (Takeuchi et al, 2013) has an issue where i and k are
                        # reversed, which seems incorrect.
                        up_temp_second_complement = self.division(
                            z[i, m], z_hat[i][m]) * b[k][m]

                        if not math.isnan(up_temp_second_complement):
                            up_temp_second = up_temp_second + \
                                             up_temp_second_complement

                up_temp = up_temp_first + (beta * up_temp_second)

                down_temp = 0
                down_temp_first = 0
                for j in range(np.shape(h)[1]):
                    if not math.isnan(h[k][j]):
                        down_temp_first = down_temp_first + h[k][j]

                down_temp_second = 0
                if beta != 0:
                    for m in range(np.shape(b)[1]):
                        if not math.isnan(b[k][m]):
                            down_temp_second = down_temp_second + b[k][m]

                regularization_part = 0
                if alpha_regularizer_w != 0:
                    regularizer_function_type = 'medecom_soft_binary'

                    if regularizer_function_type == 'lasso_sparcity':
                        # Sparcity
                        regularization_part = alpha_regularizer_w * 1
                    elif regularizer_function_type == 'medecom_soft_binary':
                        # Direct regularizator from MeDeCom
                        if not math.isnan(w[i][k]):
                            regularization_part = alpha_regularizer_w * \
                                                  (w[i][k] * (1 - w[i][k]))
                    elif regularizer_function_type == \
                            'medecom_soft_binary_derived':
                        # Derived regularizator from MeDeCom: based on the
                        # function x(1-x)
                        if not math.isnan(w[i][k]):
                            regularization_part = alpha_regularizer_w * \
                                                  (1 - 2 * w[i][k])

                down_temp = down_temp_first + \
                            (beta * down_temp_second) + \
                            regularization_part

                if not math.isnan(w[i][k]):
                    w_new[i][k] = w[i][k] * self.division(up_temp, down_temp)

        if regularize_w is not None:
            # TODO: Test the _regularize_w function with sparse values to
            #  ensure its correct behavior.
            w_new = self._regularize_w(w_new, regularization_type=regularize_w)

        return w_new

    def _regularize_w(self, w, regularization_type):
        """
        This function applies a specified normalization to the 'w' matrix. It
        aims to normalize each column of 'w' so that the values fall within the
        range [0, 1]. The normalization type is flexible and can be specified
        by the 'regularization_type' parameter.

        Parameters:

            w (np.ndarray): The 'w' matrix that needs to be normalized. It
                should be numeric.regularization_type (str): Specifies the type
                of normalization to be applied. It should match one of the
                supported types in the 'normalization' function.

        Returns:
            regularized_w (np.ndarray): The normalized 'w' matrix.

        Note: The normalization is performed by the 'normalization' function
        (not included in this docstring), so the normalization options and the
        behavior of this function depends on the 'normalization' function.
        """

        # The goal is to normalize each column within the range of [0, 1].
        regularized_w = self.normalization(
            matrix=w, normalization_type=regularization_type)
        return regularized_w

    def normalization(self, matrix, normalization_type):
        """
        This function normalizes the provided matrix based on a specified
        normalization type. It supports various types of normalization,
        including column_max, global_max, centered, norm_zero_min_max,
        centered_norm_zero_min_max, quantile_norm, quantile_norm_min_max, and
        quantile_transform.

        Parameters:

            matrix (np.ndarray): The input matrix that needs to be normalized.
                    It should be numeric.
            normalization_type (str): Specifies the type of normalization to be
                applied. Supported types include 'column_max', 'global_max',
                'centered', 'norm_zero_min_max', 'centered_norm_zero_min_max',
                'quantile_norm', 'quantile_norm_min_max',
                'quantile_transform' and, 'standard_scaler'.

        Returns:

            normalized_matrix (np.ndarray): The normalized version of the input
                matrix.

        Supported Normalization Types:

            - 'column_max': Each value in a column is divided by the maximum
                value of that column.
            - 'global_max': Each value in the matrix is divided by the maximum
                value of the entire matrix.
            - 'centered': The mean is subtracted from each value, and then
                divided by the standard deviation.
            - 'norm_zero_min_max': Each value is subtracted by the minimum
                value of the matrix, and then divided by the range (max - min)
                of the matrix.
            - 'centered_norm_zero_min_max': The matrix is first centered,
                then normalized between 0 and 1.
            - 'quantile_norm': Quantile normalization is applied to the matrix
                (see 'quantile_normalize_v2' function for details).
            - 'quantile_norm_min_max': The matrix is first transformed using a
                quantile transform, then normalized between 0 and 1.
            - 'quantile_transform': A quantile transform is applied to the
                matrix with n_quantiles=4. Random state is fixed to 0.
            - 'standard_scaler': Standard scaler using StandardScaler from
                sklearn.preprocessing. This is created initially for unknown
                references, however can be used for the entire matrix.

        Note: The actual implementation of 'quantile_normalize_v2' and
        'quantile_transform' is not included in this docstring. It depends on
        the implementation of these functions.
        """

        # Create a variable to store the normalized matrix.
        normalized_matrix = None

        if normalization_type == "column_max":
            normalized_matrix = matrix / matrix.max(axis=0)
        elif normalization_type == "global_max":
            normalized_matrix = matrix / matrix.max()
        elif normalization_type == "centered":
            normalized_matrix = (matrix - matrix.mean()) / matrix.std()
        elif normalization_type == "norm_zero_min_max":
            normalized_matrix = (matrix - matrix.min()) / \
                                (matrix.max() - matrix.min())
        elif normalization_type == "centered_norm_zero_min_max":
            normalized_matrix_temp = self.normalization(matrix, "centered")
            normalized_matrix = self.normalization(normalized_matrix_temp,
                                                   "norm_zero_min_max")
        elif normalization_type == "quantile_norm":
            normalized_matrix = self.quantile_normalize_v2(matrix)
        elif normalization_type == "quantile_norm_min_max":
            normalized_matrix_temp = self.normalization(matrix,
                                                        "quantile_transform")
            normalized_matrix = self.normalization(normalized_matrix_temp,
                                                   "norm_zero_min_max")
        elif normalization_type == "quantile_transform":
            normalized_matrix = quantile_transform(matrix, n_quantiles=4,
                                                   random_state=0, copy=True)
        elif normalization_type == "standard_scaler":
            scaler = StandardScaler(with_mean=False, with_std=True)
            normalized_matrix = scaler.fit_transform(matrix)

        return normalized_matrix

    # From https://github.com/ShawnLYU/Quantile_Normalize/blob/master/quantile_norm.py # noqa
    # https://stackoverflow.com/questions/37935920/quantile-normalization-on-pandas-dataframe# noqa
    def quantile_normalize(self, matrix):
        """
        This function applies quantile normalization to a given matrix.
        The implementation is adapted from the code available at
        https://github.com/ShawnLYU/Quantile_Normalize and
        https://stackoverflow.com/questions/37935920/
        quantile-normalization-on-pandas-dataframe. The input matrix can be
        either a numpy ndarray or a pandas DataFrame.

        Parameters:

            matrix (np.ndarray or pd.DataFrame): The input matrix that needs to
                be quantile normalized. The matrix should be numeric.

        Returns:

            df (pd.DataFrame): The quantile normalized version of the input
                matrix.

        Procedure:
            1. Converts the input matrix into a pandas DataFrame if it isn't
            one already.
            2. Creates a copy of the input DataFrame to avoid modifying the
            original data.
            3. Computes the rank (average of each row when sorted by columns)
            of the DataFrame.
            4. For each column, sorts the column, matches the original values
            with their rank, and replaces the original values with the
            corresponding rank values. This results in columns having the same
            distribution.

        Note:
            The quantile normalization procedure is often used in
            bioinformatics and statistics to make the distribution of values in
            different columns (usually representing different samples) the
            same, or quantiles the same. This is particularly useful when
            different columns have different scales or distribution of values,
            and we want to make them comparable across columns.
        """

        # Converting matrix array to dataframe
        if isinstance(matrix, pd.core.frame.DataFrame):
            df_input = matrix
        else:
            df_input = pd.DataFrame(matrix)

        df = df_input.copy()
        # compute rank
        dic = {}
        for col in df:
            dic.update({col: sorted(df[col])})
        sorted_df = pd.DataFrame(dic)
        rank = sorted_df.mean(axis=1).tolist()
        # sort
        for col in df:
            t = np.searchsorted(np.sort(df[col]), df[col])
            df[col] = [rank[i] for i in t]
        return df

    # https://cmdlinetips.com/2020/06/computing-quantile-normalization-in-python/ #noqa
    def quantile_normalize_v2(self, matrix):
        """
        This function implements a version of quantile normalization on a given
        matrix. The code is adapted from
        url{https://cmdlinetips.com/2020/06/
        computing-quantile-normalization-in-python/}
        The input matrix can be either a numpy ndarray or a pandas DataFrame.

        Parameters:

            matrix (np.ndarray or pd.DataFrame): The input matrix to be
                quantile normalized. The matrix should be numeric.

        Returns:

            df_qn (pd.DataFrame): The quantile normalized version of the input
                matrix.

        Procedure:
            1. Converts the input matrix into a pandas DataFrame if it isn't
            one already.
            2. Sorts the values of the DataFrame along each column, creating a
            new DataFrame.
            3. Calculates the mean of each row in the sorted DataFrame and
            assigns it to a new pandas Series, df_mean.
            4. Ranks each value in the original DataFrame, and replaces each
            rank with the corresponding mean value from df_mean.

        Note:
            Quantile normalization is a technique often used in bioinformatics
            and statistics to make the distribution of values in different
            columns (usually representing different samples) the same, or
            quantiles the same. This is particularly useful when different
            columns have different scales or distribution of values, and we
            want to make them comparable across columns.
        """

        # Converting matrix array to dataframe
        if isinstance(matrix, pd.core.frame.DataFrame):
            df = matrix
        else:
            df = pd.DataFrame(matrix)

        """
        input: dataframe with numerical columns
        output: dataframe with quantile normalized values
        """
        df_sorted = pd.DataFrame(np.sort(df.values,
                                         axis=0),
                                 index=df.index,
                                 columns=df.columns)
        df_mean = df_sorted.mean(axis=1)
        df_mean.index = np.arange(1, len(df_mean) + 1)
        df_qn = df.rank(method="min").stack().astype(int). \
            map(df_mean).unstack()
        return df_qn

    def _calculate_h_new_extended(self, x, x_hat, w, h, proportion_constraint):
        """
        This function calculates a new version of the matrix 'h' by applying
        certain transformations and checks. This function is typically used in
        the context of matrix factorization techniques such as non-negative
        matrix factorization (NMF), where 'w' and 'h' are the two factor
        matrices.

        Parameters:

            x (np.ndarray): The original matrix that is being factorized.
            x_hat (np.ndarray): The current estimate of the original matrix 'x'
                from the product of 'w' and 'h'.
            w (np.ndarray): The current estimate of the factor matrix 'w'.
            h (np.ndarray): The current estimate of the factor matrix 'h'.
            proportion_constraint (bool): A flag indicating whether a
                proportion constraint should be applied to the new 'h'.

        Returns:
            h_new (np.ndarray): The updated version of the 'h' matrix.

        Procedure:
            1. Initializes a new zero matrix 'h_new' of the same shape as 'h'.
            2. For each element in 'h', it computes a new value based on the
                corresponding elements in 'x', 'x_hat', and 'w'.
            3. If the 'proportion_constraint' flag is set to True, it applies a
                proportion constraint to 'h_new'.
            4. Returns 'h_new'.

        Note:
            The updates are based on a form of multiplicative update rule that
            is common in NMF. The goal is to make 'x_hat' (the product of 'w'
            and 'h') as close to 'x' as possible.
            The 'proportion_constraint' refers to any additional constraints
            that might need to be applied on 'h' to ensure that the
            factorization meets certain criteria.
        """

        h_new = np.zeros(shape=(np.shape(h)))

        for k in range(np.shape(h)[0]):
            for j in range(np.shape(h)[1]):

                up_temp = 0
                for i in range(np.shape(x)[0]):
                    up_temp = up_temp + self.division(x[i][j],
                                                      x_hat[i][j]) * w[i][k]

                down_temp = 0
                for i in range(np.shape(w)[0]):
                    down_temp = down_temp + w[i][k]

                h_new[k][j] = h[k][j] * self.division(up_temp, down_temp)

        if proportion_constraint:
            # I will send None for h_mask_fixed to have compatibility with
            # the old version.
            h_new = self._proportion_constraint_h(h=h_new, h_mask_fixed=None)

        return h_new

    def _calculate_h_new_extended_alpha_beta_generic(self, x, x_hat, w, h,
                                                     alpha, y, y_hat, a,
                                                     proportion_constraint,
                                                     is_sparse, h_mask_fixed):
        """
        This function calculates an updated version of the matrix 'h'
        considering an additional constraint on 'alpha' and potentially dealing
        with sparse matrices. This function is typically used in the context of
        matrix factorization techniques, where 'w' and 'h' are the two factor
        matrices.

        Parameters:

            x (Any): The original matrix that is being factorized.
            x_hat (Any): The current estimate of the original matrix 'x' from
                the product of 'w' and 'h'.
            w (Any): The current estimate of the factor matrix 'w'.
            h (Any): The current estimate of the factor matrix 'h'.
            alpha (float): A constraint parameter on the transformation.
            y (Any): An additional original matrix that is being factorized.
            y_hat (Any): The current estimate of the original matrix 'y'.
            a (Any): The current estimate of an additional factor matrix 'a'.
            proportion_constraint (bool): A flag indicating whether a
                proportion constraint should be applied to the new 'h'.
            is_sparse (bool): A flag indicating whether the input matrices are
                sparse.
            h_mask_fixed (np.ndarray, optional): A mask matrix for H defining
                the positions that are fixed with TRUE values. If None, no
                mask is applied. Defaults to None.

        Returns:
            h_new (np.ndarray): The updated version of the 'h' matrix.

        Procedure:
            1. Initializes a new zero matrix 'h_new' of the same shape as 'h'.
            2. If the 'is_sparse' flag is True, it calls a separate method to
                compute 'h_new' specifically for sparse matrices, else it calls
                the general method for the same.
            3. If the 'proportion_constraint' flag is set to True, it applies a
                proportion constraint to 'h_new'.
            4. Returns 'h_new'.

        Note:
            The updates are based on a form of multiplicative update rule that
            is common in NMF. The goal is to make
            'x_hat' (the product of 'w' and 'h') and 'y_hat' (the product of
            'a' and 'h') as close to 'x' and 'y' as possible.
            The 'proportion_constraint' refers to any additional constraints
            that might need to be applied on 'h' to ensure that the
            factorization meets certain criteria.
        """

        h_new = np.zeros(shape=(np.shape(h)))

        if is_sparse:
            h_new = self._calculate_h_new_extended_alpha_beta_sparse(
                x, x_hat, w, h,
                alpha, y, y_hat, a)
        else:
            h_new = self._calculate_h_new_extended_alpha_beta(
                x, x_hat, w, h,
                alpha, y, y_hat, a)

        # If I need to calculate proportions I will do with the final H matrix
        if proportion_constraint:
            h_new = self._proportion_constraint_h(h=h_new,
                                                  h_mask_fixed=h_mask_fixed)

        return h_new

    # TODO: Check again the regularization model for H.
    def _calculate_h_new_extended_alpha_beta(self, x, x_hat, w, h, alpha,
                                             y, y_hat, a):
        """
        This function calculates an updated version of the matrix 'h' given the
        constraints of the 'alpha' parameter. It is a specialized version of
        the previous method that works specifically for non-sparse data. This
        function is used in matrix factorization techniques, where 'w' and 'h'
        are the factor matrices.

        Parameters:

            x (np.ndarray): The original matrix that is being factorized.
            x_hat (np.ndarray): The current estimate of the original matrix 'x'
                from the product of 'w' and 'h'.
            w (np.ndarray): The current estimate of the factor matrix 'w'.
            h (np.ndarray): The current estimate of the factor matrix 'h'.
            alpha (float): A constraint parameter on the transformation.
            y (np.ndarray): An additional original matrix that is being
                factorized.
            y_hat (np.ndarray): The current estimate of the original matrix
                'y'.
            a (np.ndarray): The current estimate of an additional factor matrix
                'a'.

        Returns:
            h_new (np.ndarray): The updated version of the 'h' matrix.

        Procedure:
            1. Initializes a new zero matrix 'h_new' of the same shape as 'h'.
            2. For each element in 'h_new', it computes the corresponding
                element in 'h_new' using a multiplicative update rule that is
                based on 'x', 'x_hat', 'w', 'h', 'alpha', 'y', 'y_hat', and
                'a'.
            3. Returns 'h_new'.

        Note:
        The updates are based on a form of multiplicative update rule that is
        common in NMF. The goal is to make 'x_hat' (the product of 'w' and 'h')
        and 'y_hat' (the product of 'a' and 'h') as close to 'x' and 'y' as
        possible.
        """

        h_new = np.zeros(shape=(np.shape(h)))

        for k in range(np.shape(h)[0]):
            for j in range(np.shape(h)[1]):

                up_temp = 0
                up_temp_first = 0
                for i in range(np.shape(x)[0]):
                    up_temp_first = up_temp_first + \
                                    (self.division(x[i][j],
                                                   x_hat[i][j])) * w[i][k]

                # Avoid the calculation if running a simple model.
                up_temp_second = 0
                if alpha != 0:
                    for n in range(np.shape(y)[0]):
                        up_temp_second = up_temp_second + \
                                         self.division(y[n][j],
                                                       y_hat[n][j]) * a[n][k]

                up_temp = up_temp_first + (alpha * up_temp_second)

                down_temp = 0
                down_temp_first = 0
                for i in range(np.shape(w)[0]):
                    down_temp_first = down_temp_first + w[i][k]

                down_temp_second = 0
                if alpha != 0:
                    for n in range(np.shape(y)[0]):
                        down_temp_second = down_temp_second + a[n][k]

                down_temp = down_temp_first + (alpha * down_temp_second)

                h_new[k][j] = h[k][j] * (self.division(up_temp, down_temp))

        return h_new

    def _calculate_h_new_extended_alpha_beta_sparse(self, x, x_hat, w, h,
                                                    alpha, y, y_hat, a):
        """
        Calculates the new values for matrix H based on X, X_hat, W, H, Y,
        Y_hat, A, and alpha for sparse inputs. This is part of the update rules
        in Non-negative Matrix Factorization models.

        The function works with sparse matrices and processes them efficiently.
        In case the input matrices X and Y are not sparse, they will be
        converted to sparse format.

        Parameters:

        x : csr_matrix
            Input matrix X. If not in csr_matrix format, will be converted to
            it.

        x_hat : array-like
            Estimated matrix X.

        w : array-like
            Matrix W.

        h : array-like
            Matrix H, values of which will be updated.

        alpha : float
            A parameter in the NMF model which weights the divergence of Y and
            AH in the loss function.

        y : csr_matrix
            Input matrix Y. If not in csr_matrix format, will be converted to
            it.

        y_hat : array-like
            Estimated matrix Y.

        a : array-like
            Matrix A.

        Returns:
        h_new : array-like
        Updated matrix H based on X, X_hat, W, H, Y, Y_hat, A, and alpha.
        """

        h_new = np.zeros(shape=(np.shape(h)))

        # The input X and Y must be sparse always and at least X must be
        # filled.
        if type(x) is not csr_matrix:
            x = csr_matrix(x)
        if type(y) is not csr_matrix and y is not None:
            y = csr_matrix(y)

        for k in range(np.shape(h)[0]):
            for j in range(np.shape(h)[1]):

                up_temp = 0
                up_temp_first = 0
                for i in range(np.shape(x)[0]):
                    # If the complementary part is not NaN
                    up_temp_first_complement = (x[i, j] /
                                                x_hat[i][j]) * w[i][k]
                    if not math.isnan(up_temp_first_complement):
                        up_temp_first = up_temp_first + \
                                        up_temp_first_complement

                # In case that we are running a simple model, we should avoid
                # the calculation.
                up_temp_second = 0
                if alpha != 0:
                    for n in range(np.shape(y)[0]):
                        # If the complementary part is not NaN
                        up_temp_second_complement = self.division(
                            y[n, j], y_hat[n][j]) * a[n][k]

                        if not math.isnan(up_temp_second_complement):
                            up_temp_second = up_temp_second + \
                                             up_temp_second_complement

                up_temp = up_temp_first + (alpha * up_temp_second)

                down_temp = 0
                down_temp_first = 0
                for i in range(np.shape(w)[0]):
                    # I sum up if the value is not null
                    if not math.isnan(w[i][k]):
                        down_temp_first = down_temp_first + w[i][k]

                down_temp_second = 0
                if alpha != 0:
                    for n in range(np.shape(y)[0]):
                        # I sum up if the value is not null
                        if not math.isnan(a[n][k]):
                            down_temp_second = down_temp_second + a[n][k]

                down_temp = down_temp_first + (alpha * down_temp_second)

                if not math.isnan(h[k][j]):
                    # I sum up if the value is not null
                    h_new[k][j] = h[k][j] * self.division(up_temp, down_temp)

        return h_new

    def _calculate_a_new_extended_generic(self, y, y_hat, a, h, is_sparse,
                                          constraint_type_a=None,
                                          constraint_value_a=None):
        """
        This function calculates an updated version of the matrix 'a' for
        sparse and not sparse data matrices.
        This function is used in matrix factorization techniques, where 'a'
        and 'h' are the factor matrices.

        Parameters:

            y (np.ndarray or csr_matrix): The original matrix that is being
                factorized.
            y_hat (np.ndarray): The current estimate of the original matrix 'y'
                from the product of 'a' and 'h'.
            a (np.ndarray): The current estimate of the factor matrix 'a'.
            h (np.ndarray): The current estimate of the factor matrix 'h'.
            is_sparse : boolean
                A flag indicating whether the data structure is sparse or
                standard. If True, the sparse method is used.
            constraint_type_a : str, optional
                Type of constraint to apply to the A Matrix during
                deconvolution depending on the data used.
                Possible values are: 'sigmoid', 'power', 'exp', 'zero_one', or
                None. If None, no constraint is applied.
            constraint_value_a : float or None, optional
                The value to use for the constraint on the A Matrix.
                If None and constraint_type_w is not None, a default value will
                be used.

        Returns:
            h_new (np.ndarray): The updated version of the 'h' matrix.

        Procedure:
            1. Initializes a new zero matrix 'a_new' of the same shape as 'a'.
            2. Ensures that 'y' and 'y_hat' are in the csr_matrix format for
                sparse matrices.
            3. For each element in 'a_new', it computes the corresponding
                element in 'a_new' using a multiplicative update rule that is
                based on 'y', 'y_hat', 'a', and 'h'
            4. Returns 'a_new'.

        Note:
            The updates are based on a form of multiplicative update rule that
            is common in NMF. The goal is to make 'x_hat' (the product of 'w'
            and 'h') and 'y_hat' (the product of 'a' and 'h') as close to 'x'
            and 'y' as possible.
        """

        a_new = np.zeros(shape=(np.shape(a)))

        if is_sparse:
            a_new = self._calculate_a_new_extended_sparse(y, y_hat, a, h)
        else:
            a_new = self._calculate_a_new_extended(y, y_hat, a, h)

        # Checking if there is a constraint for the A
        if constraint_type_a is not None:
            a_new_scaled = self.scale_matrix(matrix=a_new,
                                             scale_type=constraint_type_a,
                                             value=constraint_value_a)
            a_new = a_new_scaled

        return a_new

    def _calculate_a_new_extended(self, y, y_hat, a, h):
        """
        This function is a part of an extended non-negative matrix
        factorization (NMF) model. It updates the 'a' matrix which is one of
        the factor matrices in this model. The updating process takes into
        account the original matrix 'y', the current estimate of this matrix
        'y_hat', the current version of the factor matrix 'a', and the other
        factor matrix 'h'.

        In this process, each element of the matrix 'a' is updated based on a
        ratio: the numerator is the sum of a series of products - each product
        involves an element of 'y', its corresponding element in 'y_hat', and a
        corresponding element in 'h'; the denominator is simply the sum of a
        series of elements in 'h'. The element in
        'a' is then multiplied by this ratio.

        The function returns the updated version of the 'a' matrix.

        Parameters:

        y : array-like
            The original matrix that is being factorized.

        y_hat : array-like
            The current estimate of the original matrix 'y'.

        a : array-like
            The current version of the factor matrix 'a' that needs to be
            updated.

        h : array-like
            The other factor matrix involved in the NMF model.

        Returns:
        a_new : array-like
        The updated version of the 'a' matrix.
        """

        a_new = np.zeros(shape=(np.shape(a)))

        for n in range(np.shape(y)[0]):
            for k in range(np.shape(a)[1]):

                up_temp = 0
                for j in range(np.shape(y)[1]):
                    up_temp = up_temp + self.division(y[n][j],
                                                      y_hat[n][j]) * h[k][j]

                down_temp = 0
                for j in range(np.shape(y)[1]):
                    down_temp = down_temp + h[k][j]

                a_new[n][k] = a[n][k] * self.division(up_temp, down_temp)

        return a_new

    def _calculate_a_new_extended_sparse(self, y, y_hat, a, h):
        """
        This function is an enhanced version of the one that updates the 'a'
        matrix in an extended non-negative matrix factorization (NMF) model,
        specifically designed for sparse matrices. It works similarly to the
        original function, but with special handling for sparse matrix formats
        and NaN values.

        The original matrix 'y' and the current estimate 'y_hat' of it should
        be in sparse format for efficiency. If 'y' is not sparse, it will be
        converted to a sparse format. Then, each element of 'a' is updated
        using a ratio similar to that in the original function, but with a
        consideration for potential NaN values in the 'y', 'y_hat', 'h'
        matrices.

        The function returns the updated 'a' matrix.

        Parameters:

        y : csr_matrix or array-like
            The original matrix that is being factorized. It should be a sparse
            csr_matrix, if not, it will be converted.

        y_hat : array-like
            The current estimate of the original matrix 'y'.

        a : array-like
            The current version of the factor matrix 'a' that needs to be
            updated.

        h : array-like
            The other factor matrix involved in the NMF model.

        Returns:
        a_new : array-like
        The updated version of the 'a' matrix.
        """

        a_new = np.zeros(shape=(np.shape(a)))

        # The input X and Y must be sparse always
        if type(y) is not csr_matrix:
            y = csr_matrix(y)

        for n in range(np.shape(y)[0]):
            for k in range(np.shape(a)[1]):

                up_temp = 0
                for j in range(np.shape(y)[1]):
                    up_temp_complement = self.division(y[n, j],
                                                       y_hat[n][j]) * h[k][j]
                    # If the complementary part is not NaN
                    if not math.isnan(up_temp_complement):
                        up_temp = up_temp + up_temp_complement

                down_temp = 0
                for j in range(np.shape(y)[1]):
                    # I sum up if the value is not null
                    if not math.isnan(h[k][j]):
                        down_temp = down_temp + h[k][j]

                # I sum up if the value is not null
                if not math.isnan(a[n][k]):
                    a_new[n][k] = a[n][k] * self.division(up_temp, down_temp)

        return a_new

    def _calculate_b_new_extended_generic(self, z, z_hat, b, w, is_sparse):
        """
        This function updates the 'b' matrix in an extended non-negative matrix
        factorization (NMF) model. It supports both standard and sparse data
        structures, by choosing the appropriate calculation function based on
        the 'is_sparse' parameter.

        In particular, if the 'is_sparse' flag is set to True, the function
        uses the method designed for sparse matrices, otherwise, it uses the
        standard method.

        Parameters:

        z : csr_matrix or array-like
            The original matrix that is being factorized. It should be a sparse
            csr_matrix for the sparse method.

        z_hat : array-like
            The current estimate of the original matrix 'z'.

        b : array-like
            The current version of the factor matrix 'b' that needs to be
            updated.

        w : array-like
            The other factor matrix involved in the NMF model.

        is_sparse : boolean
            A flag indicating whether the data structure is sparse or standard.
            If True, the sparse method is used.

        Returns:
        b_new : array-like
        The updated version of the 'b' matrix.
        """

        b_new = np.zeros(shape=(np.shape(b)))

        if is_sparse:
            b_new = self._calculate_b_new_extended_sparse(z, z_hat, b, w)
        else:
            b_new = self._calculate_b_new_extended(z, z_hat, b, w)

        return b_new

    def _calculate_b_new_extended(self, z, z_hat, b, w):
        """
        This function calculates the updated version of the 'b' matrix in an
        extended non-negative matrix factorization  (NMF) model. It follows a
        standard multiplicative update rule, where each element of 'b' is
        updated by multiplying its current value by a ratio of two terms.
        The numerator of the ratio is a sum over all rows of the original
        matrix 'z', where each term in the sum is the product of an element of
        'z' divided by the corresponding element of the estimated matrix
        'z_hat', and an element of the other factor matrix 'w'. The denominator
        of the ratio is the sum of the corresponding elements of 'w'.

        This function is meant for dense or standard data structures. For
        sparse data, use the corresponding sparse function.

        Parameters:

        z : array-like
            The original matrix that is being factorized.

        z_hat : array-like
            The current estimate of the original matrix 'z'.

        b : array-like
            The current version of the factor matrix 'b' that needs to be
            updated.

        w : array-like
            The other factor matrix involved in the NMF model.

        Returns:
        b_new : array-like
        The updated version of the 'b' matrix.
        """

        b_new = np.zeros(shape=(np.shape(b)))

        for k in range(np.shape(b)[0]):
            for m in range(np.shape(b)[1]):

                up_temp = 0
                for i in range(np.shape(z)[0]):
                    up_temp = up_temp + self.division(z[i][m],
                                                      z_hat[i][m]) * w[i][k]

                down_temp = 0
                for i in range(np.shape(z)[0]):
                    down_temp = down_temp + w[i][k]

                b_new[k][m] = b[k][m] * self.division(up_temp, down_temp)

        return b_new

    def _calculate_b_new_extended_sparse(self, z, z_hat, b, w):
        """
        This function calculates the updated version of the 'b' matrix in an
        extended non-negative matrix factorization(NMF) model, optimized for
        sparse matrices. It follows a standard multiplicative update rule,
        where each element of 'b' is updated by multiplying its current value
        by a ratio of two terms. The numerator of the ratio is a sum over all
        rows of the original matrix 'z', where each term in the sum is the
        product of an element of 'z' divided by the corresponding element of
        the estimated matrix 'z_hat', and an element of the other factor matrix
        'w'. The denominator of the ratio is the sum of the corresponding
        elements of 'w'.

        This function is meant for sparse data structures. For dense data,
        use the corresponding dense function.

        Parameters:

        z : sparse matrix
            The original matrix that is being factorized. It must be a sparse
            matrix (preferably in CSR format).

        z_hat : array-like
            The current estimate of the original matrix 'z'.

        b : array-like
            The current version of the factor matrix 'b' that needs to be
            updated.

        w : array-like
            The other factor matrix involved in the NMF model.

        Returns:
        b_new : array-like
        The updated version of the 'b' matrix.
        """

        b_new = np.zeros(shape=(np.shape(b)))

        # The input Z must be sparse always
        if type(z) is not csr_matrix:
            z = csr_matrix(z)

        for k in range(np.shape(b)[0]):
            for m in range(np.shape(b)[1]):

                up_temp = 0
                for i in range(np.shape(z)[0]):
                    up_temp_complement = self.division(z[i, m],
                                                       z_hat[i][m]) * w[i][k]
                    # If the complementary part is not NaN
                    if not math.isnan(up_temp_complement):
                        up_temp = up_temp + up_temp_complement

                down_temp = 0
                for i in range(np.shape(z)[0]):
                    # I sum up if the value is not null
                    if not math.isnan(w[i][k]):
                        down_temp = down_temp + w[i][k]

                # I sum up if the value is not null
                if not math.isnan(b[k][m]):
                    b_new[k][m] = b[k][m] * self.division(up_temp, down_temp)

        return b_new

    def _calculate_divergence_generic(self, x, x_hat, is_sparse):
        """
        This function computes the divergence between the original matrix 'x'
        and the estimated matrix 'x_hat'. The divergence is a measure of how
        much the estimated matrix differs from the original matrix. The
        function uses a general form of Kullback-Leibler divergence formula,
        which is a measure of how one probability distribution diverges from a
        second, expected probability distribution.

        The function supports both sparse and dense matrices. For sparse
        matrices, a specific function optimized for sparse computations is
        called.

        Parameters:

        x : array-like or sparse matrix
            The original matrix that is being factorized.

        x_hat : array-like
            The current estimate of the original matrix 'x'.

        is_sparse : bool
            Whether the input matrix is sparse (True) or dense (False).

        Returns:
        divergence : float
        The divergence between the original matrix and its estimate.
        """

        divergence = None

        if is_sparse:
            divergence = self._calculate_divergence_sparse(x, x_hat)
        else:
            divergence = x * np.log(x / x_hat) - x + x_hat

        return divergence

    def _calculate_divergence_extended(self, x, x_hat):
        """
        This function calculates the divergence matrix for the original matrix
        'x' and the estimated matrix 'x_hat'. The divergence matrix is a
        measure of the difference between each corresponding element of 'x'
        and 'x_hat'.

        The divergence is computed using the general form of Kullback-Leibler
        divergence formula for each element in the matrices, which measures
        how one probability distribution diverges from a second, expected
        probability distribution.

        Parameters:
        x : array-like
        The original matrix that is being factorized.

        x_hat : array-like
            The current estimate of the original matrix 'x'.

        Returns:

        divergence_matrix : array-like
            The matrix where each element is the divergence between the
            corresponding elements of 'x' and 'x_hat'.
        """

        divergence_matrix = np.zeros(shape=(np.shape(x)))

        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                divergence_matrix[i][j] = x[i][j] * \
                                          (np.log(
                                              self.division(x[i][j],
                                                            x_hat[i][j]))) - \
                                          x[i][j] + \
                                          x_hat[i][j]

        return divergence_matrix

    def _calculate_divergence_sparse(self, x, x_hat):
        """
        This function calculates the Kullback-Leibler (KL) divergence between
        two matrices 'x' and 'x_hat', specifically designed for the case when
        'x' is a sparse matrix and 'x_hat' is its dense approximation. The
        KL-divergence is used as a measure of how one probability distribution
        is different from a second, expected probability distribution.

        Due to the sparsity of 'x', the computation may encounter NaN (Not a
        Number) or inf (infinity) values. This function addresses these
        scenarios, replacing NaN or inf with zeros for a valid divergence
        computation.

        This function employs the 'lil_matrix' format for efficient computation
        due to the sparsity structure of 'x'. When 'x' is not in 'csr_matrix'
        format, it is converted into 'csr_matrix' before computation.

        Parameters:

        x : array-like, sparse matrix
            The original matrix that is being factorized. Must be a sparse
            matrix.

        x_hat : array-like, dense matrix
            The current estimate of the original matrix 'x'.

        Returns:

        divergence_matrix : array-like, sparse matrix
            The sparse matrix where each element is the divergence between the
            corresponding elements of 'x' and 'x_hat'.
        """

        # A Sparse Efficiency Warning suggests that changing the sparsity
        # structure of a csr_matrix is expensive. To address this, I replaced
        # it with lil_matrix, which offers better efficiency in this context.
        divergence_matrix = lil_matrix(np.zeros(shape=(np.shape(x))))
        if type(x) is not csr_matrix:
            x = csr_matrix(x)

        for i in range(np.shape(x)[0]):
            for j in range(np.shape(x)[1]):
                x_ij = x[i, j]
                x_hat_ij = x_hat[i][j]

                # To avoid errors with log(0)
                temp_internal_value = self.division(x_ij, x_hat_ij)
                if temp_internal_value == 0:
                    divergence_complement = 0
                else:
                    divergence_complement = (np.log(temp_internal_value))

                # I convert all possible nan into zero.
                if math.isnan(x_ij):
                    x_ij = 0
                if math.isnan(x_hat_ij):
                    x_hat_ij = 0
                # Note that for the complement, there is a possibility of the
                # divergence being infinite.
                if math.isnan(divergence_complement) or \
                        math.isinf(divergence_complement):
                    divergence_complement = 0

                divergence_value_ij = x_ij * \
                                      divergence_complement - \
                                      x_ij + \
                                      x_hat_ij

                if math.isnan(divergence_value_ij):
                    divergence_matrix[i, j] = 0
                else:
                    divergence_matrix[i, j] = divergence_value_ij

        return divergence_matrix

    def _initialize_matrix(self, size_rows, size_columns,
                           initialized_matrix=None,
                           method='random_based.uniform'):
        """
        This function initializes a non-negative matrix of given size using
        several methods. This is a crucial step for Non-negative Matrix
        Factorization (NMF) algorithms since the choice of initialization can
        impact the algorithm's speed and ability to find the global optimum.
        These approaches are inspired by several sources including the
        scikit-learn library and the reference by C. Boutsidis and
        E. Gallopoulos on "Non-negative Double Singular Value Decomposition
        (NNDSVD) based initialization: A head start for nonnegative
        matrix factorization", Pattern Recognition, 2008
        (http://tinyurl.com/nndsvd).

        Different methods are implemented including:

        - Uniform random initialization, 'random_based.uniform': generates a
            matrix populated with random samples from a uniform distribution
            over the interval [0, 1).
        - Power distribution random initialization, 'random_based.power':
            generates a matrix with samples drawn from a power distribution
            with positive exponent a - 1, where 'a' is equivalent to the number
            of rows in the matrix.
        - Non-negative Double Singular Value Decomposition (NNDSVD), 'nndsvd',
            'nndsvda' and 'nndsvdar': these methods are described in the
            reference below and they are particularly good when the factorized
            matrix is sparse, however, these are currently marked as
            TODO and not implemented in this function.

        Available methods for matrix initialization include:

        - 'random_based.uniform': Generates a matrix populated with random
            samples from a uniform distribution over the interval [0, 1).
            scaled with: sqrt(X.mean() / n_components)
        - 'random_based.power': Generates a matrix with samples drawn from a
            power distribution with positive exponent a - 1, where 'a' is
            equivalent to the number of rows in the matrix.
        - 'nndsvd': This method applies Nonnegative Double Singular Value
            Decomposition for initialization which is beneficial for sparse
            matrices (currently not implemented).
        - 'nndsvda': Similar to 'nndsvd' but fills zero entries with the
            average of the matrix, preferred when sparsity is not desired
            (currently not implemented).
        - 'nndsvdar': Like 'nndsvda' but fills zeros with small random values
            for a faster but less accurate alternative
            (currently not implemented).
        - 'None': If no method is specified or if the chosen method is not
            recognized, a simple uniform random initialization is performed by
            default. 'nndsvda' if n_components <= min(n_samples, n_features),
            otherwise 'random'

        Parameters:

        size_rows : int
            Number of rows for the initialized matrix.
        size_columns : int
            Number of columns for the initialized matrix.
        initialized_matrix : array-like, optional
            If provided, this matrix is used as the initial matrix.
        method : str, optional
            Method to use for initialization. Default is
            'random_based.uniform'.

        Returns:

        new_matrix : array-like
            The initialized matrix.

        References:
        - C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start
        for non-negative matrix factorization -
        Pattern Recognition, 2008. http://tinyurl.com/nndsvd
        """

        new_matrix = None

        # If there is a matrix with initialization values.
        if initialized_matrix is not None:
            new_matrix = initialized_matrix
        else:
            if method == 'random_based.uniform':
                # The python function rand:
                # url{https://numpy.org/doc/stable/reference/random/generated/
                # numpy.random.rand.html}
                # random samples from a uniform distribution
                # over [0, 1).
                new_matrix = np.random.rand(size_rows, size_columns)
            elif method == 'random_based.power':
                # Draw samples in [0, 1] from a power distribution with
                # positive exponent a - 1.
                # url{https://numpy.org/doc/stable/reference/random/generated/
                # numpy.random.power.html}
                # Also known as the power function distribution
                new_matrix = np.random.power(a=size_rows, size=size_columns)
            elif method == 'nndsvd':
                # TODO: needs to be implemented
                print('nndsvd')
            elif method == 'nndsvda':
                # TODO: needs to be implemented
                print('nndsvda')
            elif method == 'nndsvdar':
                # TODO: needs to be implemented
                print('nndsvdar')
            else:
                # by default is a simple random initialization
                new_matrix = np.random.rand(size_rows, size_columns)

        return new_matrix

    def sparsity_calculation(self, matrix, type_analysis='zeros_and_nulls',
                             type_return='index', verbose=False):
        """
        This function calculates the degree of sparsity in a given matrix.
        A matrix's sparsity is measured by the
        proportion of its cells that contain zero or NaN/null values. For
        example, a matrix with 10% of its cells filled with non-zero values has
        a density of 10% and hence, is 90% sparse.

        The function allows for different types of sparsity analysis:
        - 'zeros_and_nulls': considers both zero and NaN/null values when
        calculating sparsity.
        - 'zeros': considers only zero values, and
        - 'nulls': considers only NaN/null values.

        The sparsity measure can be returned in three forms:
        - 'boolean': returns 'False' if the matrix is not sparse (sparsity = 0)
        and 'True' otherwise,
        - 'percentage': returns the sparsity as a percentage of the total cells
        in the matrix, and
        - 'index': returns the sparsity as a float in the range of [0, 1].

        The function also supports verbose mode which, when enabled, prints
        additional information on the computed values.

        Parameters:

        matrix : array-like
            The matrix to analyze for sparsity. If not an ndarray, the matrix
            is converted to one.
        type_analysis : str, optional
            The type of analysis to conduct. It can be either
            'zeros_and_nulls','zeros', or 'nulls'. Default is
            'zeros_and_nulls'.
        type_return : str, optional
            The type of value to return. It can be either 'boolean',
            'percentage', or 'index'. Default is 'index'.
        verbose : bool, optional
            If 'True', the function prints additional information. Default is
            'False'.

        Returns:

        sparseness_return : bool or float
            The sparsity measure of the matrix. The type of the returned value
            depends on 'type_return'.

        """

        # Just to convert to array
        if matrix is not None and type(matrix) is not np.ndarray:
            matrix = matrix.to_numpy()

        sparseness_return = 0.0

        total_values = np.prod(matrix.shape)
        non_zero_values = np.count_nonzero(matrix)

        zero_values = total_values - non_zero_values
        null_values = np.isnan(matrix).sum()
        sparse_values = zero_values + null_values

        # Possible values
        # I need to take in account not just null but also zeros, in this case
        # null will mean NaN or np.nan
        sparseness_total = sparse_values / total_values
        sparseness_zeros = zero_values / total_values
        sparseness_nulls = null_values / total_values

        if verbose:
            print("Total values:", total_values)
            print("non-zeros values:", non_zero_values)

            print("-Zeros values: ", zero_values)
            print("-null_values values: ", null_values)

            print("-->Sparse values (Zeros + Null)", sparseness_total)
            print("-->Sparse values (Zeros)", sparseness_zeros)
            print("-->Sparse values (Null)", sparseness_nulls)

        # Now we can check what sparsity analysis it needed
        if type_analysis == 'zeros_and_nulls':
            sparseness_return = sparseness_total
        elif type_analysis == 'zeros':
            sparseness_return = sparseness_zeros
        elif type_analysis == 'nulls':
            sparseness_return = sparseness_nulls

        # Finally, we should see what kind of value is needed
        if type_return == 'boolean':
            if sparseness_return == 0.0:
                sparseness_return = False
            else:
                sparseness_return = True
        elif type_return == 'percentage':
            sparseness_return = sparseness_return * 100
        elif type_return == 'index':
            sparseness_return = sparseness_return

        return sparseness_return

    def _analyse_sparsity_matrix(self, matrix, matrix_name):
        """
        Calculates and returns the sparsity information of a given matrix in a
        textual format. The function calculates three types of sparsity: total
        sparsity (includes both zeros and nulls/Nan), zero sparsity
        (only zeros), and null sparsity (only nulls/Nan).

        If the matrix is empty, a message indicating that is returned.

        Parameters:
        matrix : array-like
        The matrix to analyze for sparsity.
        matrix_name : str
        The name of the matrix. Used in the returned description text.

        Returns:
        description_text : str
        Textual information about the matrix's sparsity. If the matrix is
        empty, a message stating that is returned.
        """

        description_text = None

        if matrix is not None:

            sparsity_percentage = self.sparsity_calculation(
                matrix, type_return='percentage')
            sparsity_zero_percentage = self.sparsity_calculation(
                matrix, type_analysis='zeros', type_return='percentage')
            sparsity_null_percentage = self.sparsity_calculation(
                matrix, type_analysis='nulls', type_return='percentage')

            description_text = 'The matrix ' + matrix_name + '. Sparsity:  ' + \
                               str(sparsity_percentage) \
                               + '% - ' + 'zeros: ' + \
                               str(sparsity_zero_percentage) \
                               + '% and ' + 'nulls: ' + \
                               str(sparsity_null_percentage) + '%'
        else:
            description_text = 'The matrix ' + matrix_name + ' is empty.'

        return description_text

    def _check_parameters(self, x_matrix, y_matrix, z_matrix, k, alpha, beta,
                          delta_threshold, max_iterations, print_limit,
                          proportion_constraint_h,
                          regularize_w, alpha_regularizer_w,
                          fixed_w, fixed_h, fixed_a, fixed_b,
                          initialized_w, initialized_h,
                          initialized_a, initialized_b,
                          init_method_w, init_method_h,
                          init_method_a, init_method_b):
        """
        Checks the parameters provided to ensure they meet the necessary
        constraints, and raises an exception if any constraints are violated.
        This function checks parameters for several matrices and factors
        related to Non-negative Matrix Factorization (NMF), as well as other
        parameters associated with the optimization model and stopping
        criteria.

        Parameters:

        x_matrix, y_matrix, z_matrix : array-like
            The matrices for which to check the constraints. These matrices
            will be converted to numpy arrays for the purpose of validation.

        k : int
            The number of latent components.

        alpha, beta : float
            The alpha and beta values for the optimization model.

        delta_threshold : float
            The threshold for the stopping criteria.

        max_iterations : int
            The maximum number of iterations for the optimization.

        print_limit : int
            The number of iterations between each print.

        proportion_constraint_h : bool
            The proportion constraint for the H matrix.

        regularize_w : bool, optional
            Whether to apply regularization on W matrix. Default is None.

        alpha_regularizer_w : float, optional
            Alpha regularizer value. Default is 0.

        fixed_w, fixed_h, fixed_a, fixed_b : array-like, optional
            Fixed matrices. If None, it implies that the matrix is not fixed.
            Default is None.

        initialized_w, initialized_h, initialized_a, initialized_b :
            array-like, optional
                Initialized matrices. If None, it implies that the matrix is
                not initialized. Default is None.

        init_method_w, init_method_h, init_method_a, init_method_b : str
            Methods for initializing the matrices. Allowed methods include
            'random_based.uniform', 'random_based.power', 'nndsvd', 'nndsvda',
            'nndsvdar'.

        Raises:
        ValueError
        If any parameters fail to meet the necessary constraints.

        Returns:
        None
        """

        # Convert to array to run the verifications.
        if x_matrix is not None:
            x_matrix = x_matrix.to_numpy()
        if y_matrix is not None:
            y_matrix = y_matrix.to_numpy()
        if z_matrix is not None:
            z_matrix = z_matrix.to_numpy()

        print("Checking parameters...")

        # 1. Latent components
        if k is None or k < 1 or not isinstance(k, int):
            raise ValueError(
                "Number of latent component K must be defined and it should be"
                " a positive integer number "
                "greater than 1. Current value: " + str(k)
            )
        elif k > 20:
            raise ValueError(
                "Number of latent component K must be small since the method is"
                " finding a low rank decomposition. "
                "Current value: " + str(k)
            )

        # 2. alpha
        if not isinstance(alpha, numbers.Real) or alpha < 0:
            raise ValueError(
                "Alpha value for the optimization model for (𝒀│𝑨,𝑯) must be "
                "positive and greater or equal to 0. " +
                "If the value is zero, the Y and A matrices are not taken in "
                "account for the optimization. " +
                "Current value: " + str(alpha)
            )

        # 3. beta
        if not isinstance(beta, numbers.Number) or beta < 0:
            raise ValueError(
                "Beta value for the optimization model for (𝒁│𝑾,𝑩) must be "
                "positive and greater or equal to 0. " +
                "If the value is zero, the Z and B matrices are not taken in "
                "account for the optimization." +
                "Current value: " + str(beta)
            )

        # 4. delta_threshold
        if not isinstance(delta_threshold, numbers.Number) or \
                delta_threshold <= 0:
            raise ValueError(
                "Delta threshold for stopping criteria must be positive and "
                "greater than 0. "
                "The default number is 1e-10 recommended for final "
                "convergence. The current value is: " +
                str(delta_threshold)
            )

        # 5. Maximum iterations
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError(
                "Maximum number of iterations must be a positive greater than "
                "1. Current value: " + str(max_iterations)
            )

        # 6. print_limit
        if not isinstance(print_limit, int) or print_limit < 1:
            raise ValueError(
                "Number of prints for the iterations must be at least 1. "
                "Current value: " + str(print_limit)
            )

        # 7. proportion_constraint_h
        if not isinstance(proportion_constraint_h, bool):
            raise ValueError(
                "The proportion_constraint_h parameter must be Boolean with "
                "True or False value. Current value: " +
                str(proportion_constraint_h)
            )

        # TODO 8. regularize_w
        # TODO 9. alpha_regularizer_w

        # TODO 10. fixed_w, fixed_h, fixed_a, fixed_b
        # TODO 11. initialized_w, initialized_h, initialized_a, initialized_b,

        # 12. init_method_w, init_method_h, init_method_a, init_method_b

        # First let's create an inner or nested function to check this init
        # methods
        def check_init_methods(parameter_value, parameter_name):
            allowed_init_methods = ('random_based.uniform',
                                    'random_based.power',
                                    'nndsvd',
                                    'nndsvda',
                                    'nndsvdar')
            if init_method_w not in allowed_init_methods:
                raise ValueError(
                    "Invalid " + str(parameter_name) +
                    " parameter. Current value: '" +
                    str(parameter_value) +
                    "' instead of one of " +
                    f"{allowed_init_methods}"
                )

        # Now we can check the four parameters with the same function
        check_init_methods(init_method_w, "init_method_w")
        check_init_methods(init_method_h, "init_method_h")
        check_init_methods(init_method_a, "init_method_a")
        check_init_methods(init_method_b, "init_method_b")

        # 13. At least X matrix and then Y and Z.
        if x_matrix is None:
            raise ValueError(
                "The matrix X must be passed as input parameter at least to "
                "create a standard NMF with W and H outputs."
                " Current value: None"
            )

        # 14. Compatibility between matrices.
        #   a - When X and Y is given. ok
        #   b - When X, Y and Z is given. ok
        #   c - When the initialization matrices are given.ok
        #   d - When the fixed matrices are given. ok
        self._check_main_matrices(x_matrix, y_matrix, z_matrix)
        self._check_fixed_matrices(x_matrix, y_matrix, z_matrix,
                                   fixed_w, fixed_h,
                                   fixed_a, fixed_b,
                                   k, 'fixed')
        self._check_fixed_matrices(x_matrix, y_matrix, z_matrix,
                                   initialized_w, initialized_h,
                                   initialized_a, initialized_b,
                                   k, 'initialized')

        # 15. Check Non-negative values for the matrices using
        # sklearn.utils.validation
        if x_matrix is not None:
            check_non_negative(x_matrix, "x_matrix")
        if y_matrix is not None:
            check_non_negative(y_matrix, "y_matrix")
        if z_matrix is not None:
            check_non_negative(z_matrix, "z_matrix")

        # TODO 16. 100% zero or null values are prohibited

        # First I will create the function to check this for a given matrix.
        def check_complete_zero_null_matrix(matrix, matrix_name):
            null_perc = self.sparsity_calculation(x_matrix,
                                                  type_return='percentage',
                                                  type_analysis='nulls')
            zero_perc = self.sparsity_calculation(x_matrix,
                                                  type_return='percentage',
                                                  type_analysis='zeros')
            if null_perc == 100 or zero_perc == 100:
                raise ValueError(
                    "The matrix " + str(matrix_name) +
                    " can not be fill with 100% zero or Null values." +
                    " Current null percentages: " + str(null_perc) +
                    "%, Current zero values: " + str(zero_perc) + "%"
                )

        # Let's check all matrices.
        if x_matrix is not None:
            check_complete_zero_null_matrix(x_matrix, "x_matrix")
        if y_matrix is not None:
            check_complete_zero_null_matrix(y_matrix, "y_matrix")
        if z_matrix is not None:
            check_complete_zero_null_matrix(z_matrix, "z_matrix")

        # TODO 17. Just numeric values, not letters or another symbol.

    # TODO: Unit tests for the function _check_main_matrices
    def _check_main_matrices(self, x_matrix, y_matrix, z_matrix):
        if x_matrix is not None:
            if y_matrix is not None:
                if z_matrix is not None:
                    # Columns and rows for X,Y,Z make senses.
                    expression_x_y_z_matrices = np.shape(x_matrix)[1] == \
                                                np.shape(y_matrix)[1] and \
                                                np.shape(x_matrix)[0] == \
                                                np.shape(z_matrix)[0]
                    if not expression_x_y_z_matrices:
                        raise ValueError(
                            "The dimensions of X, Y, Z matrices are not "
                            "compatible. The actual sizes are:" +
                            "X[", str(np.shape(x_matrix)[0]), ',',
                            str(np.shape(x_matrix)[1]), "], " +
                            "Y[", str(np.shape(y_matrix)[0]), ',',
                            str(np.shape(y_matrix)[1]), "], " +
                            "Z[", str(np.shape(z_matrix)[0]), ',',
                            str(np.shape(z_matrix)[1]), "]"
                        )
                else:
                    expression_x_y_matrices = \
                        np.shape(x_matrix)[1] == np.shape(y_matrix)[1]

                    if not expression_x_y_matrices:
                        raise ValueError(
                            "The dimensions of X, Y matrices are not "
                            "compatible. The actual sizes are:" +
                            "X[", str(np.shape(x_matrix)[0]), ',',
                            str(np.shape(x_matrix)[1]), "], " +
                            "Y[", str(np.shape(y_matrix)[0]), ',',
                            str(np.shape(y_matrix)[1]), "]"
                        )

    # TODO: Unit tests for the function _check_fixed_matrices
    def _check_fixed_matrices(self, x_matrix, y_matrix, z_matrix, similar_w,
                              similar_h, similar_a, similar_b,
                              k, type_matrix):
        if similar_w is not None and x_matrix is not None:
            # The K value is missing.
            check_w = np.shape(similar_w)[0] == np.shape(x_matrix)[0] and \
                      np.shape(similar_w)[1] == k
            if not check_w:
                raise ValueError(
                    "The dimensions of " + str(type_matrix) +
                    "_w is not compatible with the matrix X and the k value." +
                    "The actual sizes are:" +
                    str(type_matrix) +
                    "_w[", str(np.shape(similar_w)[0]), ',',
                    str(np.shape(similar_w)[1]), "], " +
                    "X[", str(np.shape(x_matrix)[0]), ',',
                    str(np.shape(x_matrix)[1]), "]" +
                    "K=" + str(k)
                )
        if similar_h is not None and x_matrix is not None:
            check_h = np.shape(similar_h)[0] == k and \
                      np.shape(similar_h)[1] == np.shape(x_matrix)[1]
            if not check_h:
                raise ValueError(
                    "The dimensions of " + str(type_matrix) +
                    "_h is not compatible with the matrix X and the K value." +
                    " The actual sizes are:" +
                    str(type_matrix) +
                    "_h[", str(np.shape(similar_h)[0]), ',',
                    str(np.shape(similar_h)[1]), "], " +
                    "X[", str(np.shape(x_matrix)[0]), ',',
                    str(np.shape(x_matrix)[1]), "]" +
                    "K=" + str(k)
                )

        # In case that we have the fixed matrix A but not the Y input matrix.
        if similar_a is not None and y_matrix is None:
            raise ValueError(
                "The matrix Y is not given as a input, therefore the " +
                str(type_matrix) + "_a parameter is useless."
            )

        if similar_a is not None and y_matrix is not None:
            check_fixed_a = np.shape(similar_a)[0] == \
                            np.shape(y_matrix)[0] and \
                            np.shape(similar_a)[1] == k

            if not check_fixed_a:
                raise ValueError(
                    "The dimensions of " + str(type_matrix) +
                    "_a is not compatible with the matrix Y and the K value." +
                    " The actual sizes are:" +
                    str(type_matrix) +
                    "_a[", str(np.shape(similar_a)[0]), ',',
                    str(np.shape(similar_a)[1]), "], " +
                    "Y[", str(np.shape(y_matrix)[0]), ',',
                    str(np.shape(y_matrix)[1]), "]" +
                    "K=" + str(k)
                )

        # In case that we have the fixed matrix B but not the Z input matrix.
        if similar_b is not None and z_matrix is None:
            raise ValueError(
                "The matrix Z is not given as a input, therefore the " +
                str(type_matrix) + "_b parameter is useless."
            )

        if similar_b is not None and z_matrix is not None:
            check_fixed_z = np.shape(similar_b)[0] == k and \
                            np.shape(similar_b)[1] == np.shape(z_matrix)[1]

            if not check_fixed_z:
                raise ValueError(
                    "The dimensions of " + str(type_matrix) +
                    "_b is not compatible with the matrix Y and the K value." +
                    " The actual sizes are:" +
                    str(type_matrix) +
                    "_b[", str(np.shape(similar_b)[0]), ',',
                    str(np.shape(similar_b)[1]), "], " +
                    "Z[", str(np.shape(z_matrix)[0]), ',',
                    str(np.shape(z_matrix)[1]), "]" +
                    "K=" + str(k)
                )

    def _standardize_sparse_matrix(self, matrix, verbose=False):
        """
        Standardizes a sparse matrix by replacing empty strings, "null", "na",
        and "NA" values with numpy's nan, and converts the matrix to a float
        type. The original index and column names are retained.

        Parameters:
        matrix : array-like
            The sparse matrix to be standardized. The matrix is converted to a
            numpy array for processing.
        verbose (bool, optional): Whether to display progress messages
            during the process of standardize the sparse matrix.
            Default is True.

        Returns:
        standardized_matrix_df : DataFrame
        The standardized matrix, returned as a pandas DataFrame. All replaced
        values are represented as np.nan, and all data is converted to float
        type.

        Raises:
        TypeError
        If the input matrix cannot be converted to float.
        """

        if verbose:
            print("Standardizing sparse matrix...", 'pd.DataFrame: ',
                  isinstance(matrix, pd.DataFrame))
        else:
            print("Standardizing sparse matrix")

        # Let's take the original row and column names
        column_names = matrix.columns.values
        row_names = matrix.index.values

        # Now we should convert to array to do the replacement
        # matrix = np.array(matrix).astype(float)

        # Check for the null values and set them up as np.nan.

        # 1. replace "" values for np.nan
        # warnings.simplefilter(action='ignore', category=FutureWarning)
        standardized_matrix = np.where(matrix != '', matrix, np.nan)

        # 2. replace "null" values for np.nan
        standardized_matrix = np.where(standardized_matrix != 'null',
                                       standardized_matrix, np.nan)

        # 3. replace "na" or 'NA' values for np.nan
        standardized_matrix = np.where(standardized_matrix != 'na',
                                       standardized_matrix, np.nan)
        standardized_matrix = np.where(standardized_matrix != 'NA',
                                       standardized_matrix, np.nan)

        # 4. Convert to float type
        standardized_matrix = standardized_matrix.astype(float)

        # 5. Convert to Dataframe with names again.
        standardized_matrix_df = pd.DataFrame(data=standardized_matrix,
                                              index=row_names,
                                              columns=column_names)

        return standardized_matrix_df.astype(float)

    @staticmethod
    def check_fixed_matrices_gamma_alpha_beta(fixed_w, fixed_h,
                                              fixed_a, fixed_b,
                                              gamma, alpha, beta):
        """
        Modifies gamma, alpha, and beta based on the given fixed matrices to 
        prevent unnecessary divergence calculations. If all parameters become 
        zero, a ValueError is raised to prevent model execution.

        Parameters:

        fixed_w : array-like or None
            Fixed matrix W. If None, it implies that W is not fixed.

        fixed_h : array-like or None
            Fixed matrix H. If None, it implies that H is not fixed.

        fixed_a : array-like or None
            Fixed matrix A. If None, it implies that A is not fixed.

        fixed_b : array-like or None
            Fixed matrix B. If None, it implies that B is not fixed.

        gamma : float
            A scalar weighting the divergence of X and WH in the loss function.

        alpha : float
            A scalar weighting the divergence of Y and AH in the loss function.

        beta : float
            A scalar weighting the divergence of Z and WB in the loss function.

        Returns:
        gamma : float
        Possibly modified gamma value. It's set to 0 if W and H are both fixed.

        alpha : float
            Possibly modified alpha value. It's set to 0 if A and H are both
            fixed.

        beta : float
            Possibly modified beta value. It's set to 0 if W and B are both
            fixed.

        Raises:

        ValueError
            If gamma, alpha, and beta are all 0, indicating that the model
            cannot be executed.
        """

        # In case that the Gamma=0 or some combinations of fixed matrices
        if fixed_w is not None and fixed_h is not None:
            # 1. Since the W and H are fixed, Gamma must be 0 to avoid the
            # calculation of the divergence(X|WH)
            gamma = 0  # TODO Fix this validation
            print('The Gamma variable has been changed to zero because W and H'
                  ' are fixed. The divergence(X|WH) will not'
                  ' be calculated')

        if fixed_a is not None and fixed_h is not None:
            # 2. Since the A and H are fixed, alpha must be 0 to avoid the
            # calculation of the divergence(Y|AH)
            alpha = 0  # TODO Fix this validation
            print('The alpha variable has been changed to zero because A and '
                  'H are fixed. The divergence(Y|AH) will not be calculated')

        if fixed_w is not None and fixed_b is not None:
            # 3. Since the W and B are fixed, Beta must be 0 to avoid the
            # calculation of the divergence(Z|WB)
            beta = 0
            print('The beta variable has been changed to zero because W and B '
                  'are fixed. The divergence(Z|WB) will not'
                  ' be calculated')

        # Just in case the three parameters become zero.
        if gamma == 0 and alpha == 0 and beta == 0:
            raise ValueError("Gamma, alpha and beta are zero, therefore the "
                             "model can not be executed.")

        return gamma, alpha, beta

    @staticmethod
    def division(x, y, default=0):
        """
        Perform division of two numbers with error handling for zero division.

        Parameters:
        - x (float): The numerator.
        - y (float): The denominator.
        - default (float): The default value that in case of error will
            return. Default 0

        Returns:
        - float: The result of the division, or 0.0 if there is a zero
            division error.
        """
        try:
            if y == 0.0:
                return default
            else:
                return x / y
        except ZeroDivisionError:
            return default

    @staticmethod
    def scale_matrix(matrix, scale_type, value=0.5, verbose=False):
        transformed_matrix = None

        # Find the minimum and maximum of each column
        min_vals = np.min(matrix, axis=0)
        max_vals = np.max(matrix, axis=0)

        # Subtract the minimum and divide by the range
        scaled_matrix = (matrix - min_vals) / (max_vals - min_vals)

        if scale_type == 'sigmoid':
            # Apply sigmoid function
            sigmoid_matrix = 1 / (1 + np.exp(-scaled_matrix))
            transformed_matrix = sigmoid_matrix
        elif scale_type == 'power':
            # Apply power transformation
            power_matrix = np.power(scaled_matrix, value)
            transformed_matrix = power_matrix
        elif scale_type == 'exp':
            # Subtract 0.5 to center the values around 0
            centered_matrix = scaled_matrix - 0.5

            # Apply exponential transformation
            exp_matrix = np.exp(centered_matrix)

            # Rescale the result back to [0, 1]
            exp_matrix = (exp_matrix - np.min(exp_matrix, axis=0)) / (
                    np.max(exp_matrix, axis=0) - np.min(exp_matrix,
                                                        axis=0))

            transformed_matrix = exp_matrix
        elif scale_type == 'zero_one':
            # Subtract the minimum and divide by the range
            transformed_matrix = scaled_matrix

        if verbose:
            print('scale_type: ', scale_type, ', value: ', value)

        return transformed_matrix
