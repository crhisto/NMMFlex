from . import factorization

import multiprocessing as mp
import numpy as np


# https://www.machinelearningplus.com/python/parallel-processing-python/
# TODO Create the grid search for alpha and beta where I can have a table with
# the general behaviour of the grid (parallelized_async)
class grid_search:
    """
    Class for parallelized grid search to optimize hyperparameters
    asynchronously.

    The class implements methods for conducting a grid search in an efficient
    way by leveraging the power of multiple CPUs. The grid search is executed
    asynchronously, which means that the search does not wait for all processes
    to finish before it continues, thereby potentially speeding up the
    execution time.

    Key features include:
      - Asynchronous execution of grid search
      - Utilization of multiple CPUs for faster processing
      - Adaptability to various models and hyperparameters
    """

    alpha_beta_combinations = None

    def __init__(self):
        pass

    def grid_search_parallelized_alpha_beta(self,
                                            bulk_data_methylation,
                                            bulk_data_expression,
                                            data_expression_auxiliary,
                                            k,
                                            alpha_list=None, beta_list=None,
                                            delta_threshold=1e-20,
                                            max_iterations=200,
                                            print_limit=100,
                                            threads=0,
                                            proportion_constraint_h=True,
                                            regularize_w=None,
                                            alpha_regularizer_w_list=None,
                                            fixed_w=None, fixed_h=None,
                                            fixed_a=None, fixed_b=None):
        """
        Performs a grid search in a parallelized manner over different values
        of alpha and beta in the Non-negative matrix factorization. The search
        is conducted over the combinations of given alpha and beta values.

        Parameters:
        bulk_data_methylation : pd.DataFrame
        Bulk data methylation matrix.

        bulk_data_expression : pd.DataFrame
            Bulk data expression matrix.

        data_expression_auxiliary : pd.DataFrame
            Auxiliary expression data matrix.

        k : int
            Number of clusters.

        alpha_list : list, optional
            List of alpha values to be considered in the grid search. Default
            is None.

        beta_list : list, optional
            List of beta values to be considered in the grid search. Default is
            None.

        delta_threshold : float, optional
            The threshold value for convergence. Default is 1e-20.

        max_iterations : int, optional
            Maximum number of iterations for convergence. Default is 200.

        print_limit : int, optional
            Limit for print statements. Default is 100.

        threads : int, optional
            Number of CPU threads to be used. If 0, then it uses the total
            number of CPUs minus one. Default is 0.

        proportion_constraint_h : bool, optional
            Whether to apply proportion constraint on H matrix. Default is
            True.

        regularize_w : bool, optional
            Whether to apply regularization on W matrix. Default is None.

        alpha_regularizer_w_list : list, optional
            List of alpha regularizer values to be considered in the grid
            search. Default is None.

        fixed_w : array-like, optional
            Fixed matrix W. If None, it implies that W is not fixed. Default is
            None.

        fixed_h : array-like, optional
            Fixed matrix H. If None, it implies that H is not fixed. Default is
            None.

        fixed_a : array-like, optional
            Fixed matrix A. If None, it implies that A is not fixed. Default is
            None.

        fixed_b : array-like, optional
            Fixed matrix B. If None, it implies that B is not fixed. Default is
            None.

        Returns:

        result_objects : list
            List of AsyncResult objects representing the results of the grid
            search.

        Raises : ValueError
            If gamma, alpha and beta are all 0, indicating that the model
            cannot be executed.
            If any two of gamma, alpha, beta are zero, it suggests that the
            user should switch to the more direct function
            run_deconvolution_multiple().
        """

        # let's initialize the values of alpha and beta. The idea is to have 1
        # if there are values on the list to reuse the static function on
        # src.
        gamma, alpha, beta = 1, 0, 0
        if alpha_list is not None:
            alpha = 1
        if beta_list is not None:
            beta = 1

        # In case of some combinations of fixed matrices that get the
        # (gamma, alpha, beta) = 0
        gamma, alpha, beta = \
            factorization.check_fixed_matrices_gamma_alpha_beta(
                fixed_w=fixed_w, fixed_h=fixed_h,
                fixed_a=fixed_a, fixed_b=fixed_b,
                gamma=gamma, alpha=alpha, beta=beta)

        print("Values of parameters: gamma: ", str(gamma), ', alpha: ',
              str(alpha), ', beta: ', str(beta))

        # TODO Check this validation
        # Double check of the gamma, alpha and beta parameters to avoid wrong
        # interpretations of results when a list of values is given. Also, the
        # idea is not running expensive resources without necessity.
        # if gamma == 0 and alpha == 0:
        #     raise ValueError("Gamma and alpha are zero, therefore the model "
        #                    "is a standard NMF such argmin D(Z|WB). Switch to"
        #                    " the more direct function: "
        #                    "run_deconvolution_multiple().")
        # if gamma == 0 and beta == 0:
        #   raise ValueError("Gamma and beta are zero, therefore the model is "
        #                      "a standard NMF such argmin D(Y|AH). Switch to "
        #                      "the more direct function: "
        #                      "run_deconvolution_multiple().")
        # if alpha == 0 and beta == 0:
        #   raise ValueError("Alpha and beta are zero, therefore the model is "
        #                      "a standard NMF such argmin D(X|WH). "
        #                      "Switch to the more direct function: "
        #                      "run_deconvolution_multiple().")

        # By default, I will use the total number of CPU's minus one.
        cpu_count_selected = threads
        if threads == 0:
            print("Number of processors: ", mp.cpu_count())
            cpu_count_selected = mp.cpu_count() - 1

        # Step 1: Init multiprocessing.Pool()
        pool = mp.Pool(cpu_count_selected)

        # The NMF cases is one of the alpha and beta list is empty, meaning
        # that it's zero!
        if alpha_list is None:
            alpha_list = np.array([0.0]).astype(float)
        else:
            alpha_list = np.array(alpha_list).astype(float)

        if beta_list is None:
            beta_list = np.array([0.0]).astype(float)
        else:
            beta_list = np.array(beta_list).astype(float)

        # I have to create the combinations of beta and alpha values.
        alpha_beta_combinations = self._calculate_pair_parameters(alpha_list,
                                                                  beta_list)
        self.alpha_beta_combinations = alpha_beta_combinations

        print('Combinations(alpha, beta):')
        print(alpha_beta_combinations)

        # Currently, I am using only the regularizer with alpha and beta set to
        # 0 temporarily. If this approach proves successful, I will proceed to
        # implement a comprehensive grid search involving all three parameters.
        result_objects = None
        if alpha_regularizer_w_list is None:
            # call apply_async() without callback
            result_objects = [pool.apply_async(
                self.run_deconvolution_async,
                args=(bulk_data_methylation,
                      bulk_data_expression,
                      data_expression_auxiliary,
                      k, alpha_beta_value_iter[0], alpha_beta_value_iter[1],
                      # beta=0
                      delta_threshold,
                      max_iterations,
                      print_limit,
                      proportion_constraint_h,
                      regularize_w, 0,
                      fixed_w, fixed_h, fixed_a, fixed_b))
                for alpha_beta_value_iter in alpha_beta_combinations]
        else:
            # call apply_async() without callback
            result_objects = [pool.apply_async(
                self.run_deconvolution_async,
                args=(bulk_data_methylation,
                      bulk_data_expression,
                      data_expression_auxiliary,
                      k, 0, 0,  # beta=0
                      delta_threshold,
                      max_iterations,
                      print_limit,
                      proportion_constraint_h,
                      regularize_w, alpha_regularizer_w_value_iter,
                      fixed_w, fixed_h, fixed_a, fixed_b))
                for alpha_regularizer_w_value_iter in alpha_regularizer_w_list]

        pool.close()
        pool.join()

        return result_objects

    def run_deconvolution_async(self,
                                bulk_data_methylation,
                                bulk_data_expression,
                                data_expression_auxiliary,
                                k, alpha, beta,
                                delta_threshold, max_iterations, print_limit,
                                proportion_constraint_h, regularize_w=None,
                                alpha_regularizer_w=0, fixed_w=None,
                                fixed_h=None, fixed_a=None, fixed_b=None):
        """
        Executes the deconvolution process for Non-negative Multiple Matrix
        Factorization (NMMF). The deconvolution process for Non-negative
        Multiple Matrix Factorization (NMMF) is executed in this
        implementation. It is specifically designed for the analysis of Omic
        data, such as methylation and expression data.

        Parameters:
        bulk_data_methylation : pd.DataFrame
        Bulk data methylation matrix.

        bulk_data_expression : pd.DataFrame
            Bulk data expression matrix.

        data_expression_auxiliary : pd.DataFrame
            Auxiliary expression data matrix.

        k : int
            Number of clusters (Rank)

        alpha : float
            Parameter alpha for the NMF model.

        beta : float
            Parameter beta for the NMF model.

        delta_threshold : float
            The threshold value for convergence.

        max_iterations : int
            Maximum number of iterations for convergence.

        print_limit : int
            Limit for print statements.

        proportion_constraint_h : bool
            Whether to apply proportion constraint on H matrix.

        regularize_w : bool, optional
            Whether to apply regularization on W matrix. Default is None.

        alpha_regularizer_w : float, optional
            Alpha regularizer value. Default is 0.

        fixed_w : array-like, optional
            Fixed matrix W. If None, it implies that W is not fixed. Default is
            None.

        fixed_h : array-like, optional
            Fixed matrix H. If None, it implies that H is not fixed. Default is
            None.

        fixed_a : array-like, optional
            Fixed matrix A. If None, it implies that A is not fixed. Default is
            None.

        fixed_b : array-like, optional
            Fixed matrix B. If None, it implies that B is not fixed. Default is
            None.

        Returns:
        result : src object
        src object containing the results of the deconvolution process.
        """

        dec_object = factorization()
        result = dec_object.run_deconvolution_multiple(
            x_matrix=bulk_data_methylation, y_matrix=bulk_data_expression,
            z_matrix=data_expression_auxiliary, k=k, alpha=alpha, beta=beta,
            delta_threshold=delta_threshold,
            max_iterations=max_iterations, print_limit=print_limit,
            proportion_constraint_h=proportion_constraint_h,
            regularize_w=regularize_w,
            alpha_regularizer_w=alpha_regularizer_w,
            fixed_w=fixed_w, fixed_h=fixed_h, fixed_a=fixed_a, fixed_b=fixed_b)

        return result

    def _calculate_pair_parameters(self, alpha_list, beta_list):
        """
        Calculates the combinations of two parameter lists, alpha and beta.

        Parameters:
        alpha_list : list
        A list of alpha values.

        beta_list : list
            A list of beta values.

        Returns:

        alpha_beta_combinations : list of lists
            A list of [alpha, beta] pairs, where alpha is from alpha_list and
            beta is from beta_list.
        """

        alpha_beta_combinations = []
        for alpha_val in alpha_list:
            for beta_val in beta_list:
                alpha_beta_combinations.append([alpha_val, beta_val])

        return alpha_beta_combinations
