import unittest
from os import getcwd

import numpy as np
import pandas as pd
import pandas.core.frame

from src.NMMFlex.factorization import factorization


class test_NMMFlex_basics(unittest.TestCase):
    """
    The test_NMMFlex_basics class is a TestCase subclass for testing the basics
    of src.

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

    # Variables for scale test functions
    row_names = None
    vector_w = None
    vector_expected_w = None

    def __init__(self, relative_path):
        super(test_NMMFlex_basics, self).__init__(relative_path)
        # To allow the test cases to run from different configurations of main
        # path without problems
        print('Current root path where everything is ran:', getcwd())
        if 'tests' in getcwd():
            self.relative_path = getcwd() + '/'
        elif 'NMMFlexPy' in getcwd():
            self.relative_path = getcwd() + "/tests/"
        else:
            self.relative_path = getcwd() + "/NMMFlexPy/tests/"

    # This function will be called for each test.
    def setUp(self):
        # Let's import the data
        self._import_test_data()
        self._create_scale_objects()
        self.dec = factorization()

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

    def _create_scale_objects(self):
        self.row_names = ['row_1', 'row_2', 'row_3', 'row_4', 'row_5', 'row_6',
                          'row_7', 'row_8', 'row_9', 'row_10', 'row_11',
                          'row_12',
                          'row_13', 'row_14', 'row_15', 'row_16', 'row_17',
                          'row_18', 'row_19', 'row_20', 'row_21', 'row_22',
                          'row_23', 'row_24', 'row_25', 'row_26', 'row_27',
                          'row_28', 'row_29', 'row_30', 'row_31', 'row_32',
                          'row_33', 'row_34', 'row_35', 'row_36', 'row_37',
                          'row_38', 'row_39', 'row_40', 'row_41', 'row_42',
                          'row_43', 'row_44', 'row_45', 'row_46', 'row_47',
                          'row_48', 'row_49', 'row_50', 'row_51', 'row_52',
                          'row_53', 'row_54', 'row_55', 'row_56', 'row_57',
                          'row_58', 'row_59', 'row_60', 'row_61', 'row_62',
                          'row_63', 'row_64', 'row_65', 'row_66', 'row_67',
                          'row_68', 'row_69', 'row_70', 'row_71', 'row_72',
                          'row_73', 'row_74', 'row_75', 'row_76', 'row_77',
                          'row_78', 'row_79', 'row_80', 'row_81', 'row_82',
                          'row_83', 'row_84', 'row_85', 'row_86', 'row_87',
                          'row_88', 'row_89', 'row_90', 'row_91', 'row_92',
                          'row_93', 'row_94', 'row_95', 'row_96', 'row_97',
                          'row_98', 'row_99', 'row_100', 'row_101', 'row_102',
                          'row_103', 'row_104', 'row_105', 'row_106',
                          'row_107',
                          'row_108', 'row_109', 'row_110', 'row_111',
                          'row_112',
                          'row_113', 'row_114', 'row_115', 'row_116',
                          'row_117',
                          'row_118', 'row_119', 'row_120', 'row_121',
                          'row_122',
                          'row_123', 'row_124', 'row_125', 'row_126',
                          'row_127',
                          'row_128', 'row_129', 'row_130', 'row_131',
                          'row_132',
                          'row_133', 'row_134', 'row_135', 'row_136',
                          'row_137',
                          'row_138', 'row_139']

        self.vector_w = [2144.04512514046, 1234.12313778294, 527.721477088469,
                         521.684336889795, 313.776371601745, 3762.60997803322,
                         824.225753864234, 1289.8145205145, 11631.6044706936,
                         16037.2136626412, 237.534295066678, 34125.2914241053,
                         385.783436232189, 2112.36821552533, 835.060674509659,
                         1762.6941754088, 59.1180153204123, 854.267821967888,
                         1318.77307917526, 31730.0132528068, 15040.7766839414,
                         983.867113190048, 110.863512361166, 14906.3325820439,
                         223.263866206162, 1277.94093791392, 1041.04337375048,
                         1039.27347484291, 75982.4672124783, 207.113910870699,
                         1463.8219213037, 638.468245200753, 381.28406272268,
                         142.066489881025, 15999.5927718095, 422.050568881101,
                         75.0106763149074, 938.291031568208, 1126.71011801524,
                         1770.72707189943, 2114.01761827404, 171.808934633457,
                         762.106760130491, 1560.93135147135, 81.6749130704166,
                         157.143455509349, 15019.6906365253, 21459.2767437355,
                         11499.4166967976, 14726.4991203779, 1227.6422461364,
                         5621.18213965833, 483.967243617198, 752.772720504604,
                         3648.78075740076, 4604.73820976071, 2727.56789880113,
                         124.182462144608, 45155.3800224986, 115.20533380018,
                         3878.34297282387, 45130.3921539691, 2314.00137104076,
                         93.8985406946534, 646.793651471288, 4274.48661223249,
                         1920.75857779506, 1745.54785654911, 1053.95333938062,
                         16762.306026555, 13300.6688103201, 466.571461356833,
                         722.260297953265, 245.645138469081, 51873.2887854523,
                         11027.9224622221, 468.970972498434, 3493.98760576313,
                         2517.3456996009, 2512.8292565017, 2895.73396550778,
                         4756.35638534189, 2412.19457457424, 1921.33301300431,
                         902.317840666097, 106.410504112754, 1191.17159122783,
                         120.219255853645, 18454.1595689252, 2452.88275958334,
                         1775.79556454523, 215.57560844213, 3371.29338949873,
                         4191.87065600182, 846.348656118452, 1150.94047558275,
                         618.970193659083, 664.536113114267, 179.920153187018,
                         1704.5140908221, 41174.7575945275, 1441.89281126052,
                         315.506580906101, 131.295382734179, 816.52653542382,
                         3945.50538801564, 479.882551330492, 643.37513250604,
                         269.626640564534, 43337.6068863126, 111.928611355041,
                         39047.165377238, 304.40029779196, 28611.5243177786,
                         274.588362360545, 815.770649724661, 135.306853805709,
                         35808.0784352, 578.406635337261, 343.495390145685,
                         1772.30724796426, 11101.0359336359, 792.544192976084,
                         1633.07402260012, 2976.03772248096, 30255.7735878829,
                         316.060946281093, 455.86643568064, 2998.86930670701,
                         613.510873498364, 19326.1859104222, 765.913564361909,
                         114.768689540617, 721.108175869882, 719.66048896514,
                         502.711880395261, 121.993211564522, 88.0838136927713,
                         591.487211916918]

        self.vector_expected_w = [0.154662023326754, 0.0890242370768047,
                                  0.038067515670473, 0.0376320228222649,
                                  0.0226344529483138, 0.271418201682638,
                                  0.0594560353585423, 0.0930415694706664,
                                  0.83905299421165, 1.15685434252465,
                                  0.0171346835258889, 2.46164903731748,
                                  0.0278287271634309, 0.152376989827986,
                                  0.0602376190714767, 0.12715303632294,
                                  0.00426451465843938, 0.0616231385521035,
                                  0.0951305130393355, 2.28886416256856,
                                  1.08497574377864, 0.0709718637104974,
                                  0.00799720814353906, 1.07527753520085,
                                  0.0161052772995795, 0.092185061234151,
                                  0.0750963086864185, 0.0749686359322779,
                                  5.48104234311335, 0.0149402902666429,
                                  0.10559370112327, 0.0460563023952965,
                                  0.0275041620679842, 0.0102480542570155,
                                  1.15414053625856, 0.0304448792443593,
                                  0.0054109416046977, 0.0676842048285617,
                                  0.0812759323540571, 0.12773249428763,
                                  0.152495970517061, 0.0123935439345212,
                                  0.0549750432631668, 0.112598750027198,
                                  0.00589167045151618, 0.0113356405127161,
                                  1.08345468868559, 1.54797822181912,
                                  0.829517547252025, 1.06230513703768,
                                  0.0885567339430028, 0.405487455937204,
                                  0.034911276933504, 0.0543017265323647,
                                  0.263207060335708, 0.332165643372338,
                                  0.196754800094074, 0.00895797883718743,
                                  3.25731131144481, 0.00831040812277799,
                                  0.279766672957847, 3.25550879606858,
                                  0.166921922411778, 0.00677342940266392,
                                  0.0466568607341673, 0.308342739795502,
                                  0.138555109909145, 0.125916175989942,
                                  0.0760275770548053, 1.2091593293865,
                                  0.959451984327235, 0.0336564214035564,
                                  0.0521006940293377, 0.0177197642393463,
                                  3.74191182176983, 0.795506018552708,
                                  0.0338295116262404, 0.25204096044878,
                                  0.181590291523212, 0.181264495102339,
                                  0.208885563493957, 0.343102714394553,
                                  0.174005149978842, 0.138596547148824,
                                  0.0650892564176342, 0.0076759876349245,
                                  0.085925900657793, 0.00867209049620319,
                                  1.3312022327598, 0.176940217410591,
                                  0.128098113144539, 0.0155506800629355,
                                  0.243190337150121, 0.302383186612649,
                                  0.0610518846176726, 0.0830238042077696,
                                  0.0446497983683971, 0.0479367241961457,
                                  0.0129786517097398, 0.122956179879021,
                                  2.97016664662832, 0.104011831185337,
                                  0.0227592626683391, 0.00947107377033327,
                                  0.0589006474667174, 0.284611475384471,
                                  0.0346166251249569, 0.0464102637508544,
                                  0.019449684749417, 3.12618512016507,
                                  0.00807403971928547, 2.81669146400189,
                                  0.0219581040556162, 2.0639100313527,
                                  0.0198076016249329, 0.0588461211835414,
                                  0.00976044372116345, 2.58303792084055,
                                  0.0417237209599402, 0.0247782527617575,
                                  0.127846481266977, 0.800780104106551,
                                  0.0571706663373142, 0.11780280630108,
                                  0.214678324750976, 2.18251897105551,
                                  0.0227992521580906, 0.0328842077446968,
                                  0.216325295223168, 0.0442559869265834,
                                  1.39410639311944, 0.0552496494446955,
                                  0.00827891051861563, 0.0520175849891192,
                                  0.0519131551974112, 0.036263432919694,
                                  0.00880005589036364, 0.00635398046819017,
                                  0.0426672964548626]

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

    def test_proportion_constraint_h_partial_fixed(self):

        h = np.array([[0.11692266, 0.49023124, 0.03157466],
                      [0.38518708, 0.08108722, 0.34163091],
                      [0.11876603, 0.26641129, 0.75638277],
                      [0.42219656, 0.17418068, 0.06863269]]).astype(float)

        # Row and column names
        column_names = ['column_1', 'column_2', 'column_3']
        row_names = ['row_1', 'row_2', 'row_3', 'cluster_unknown']

        # We define and np array initially
        h_mask_fixed = np.ones(shape=np.shape(h), dtype=bool)
        h_mask_fixed[[0, 1, 2], :] = False
        # Then we generate the DataFrame that must have the names
        # (index, columns)
        h_mask_fixed_df = pd.DataFrame(data=np.nan_to_num(h_mask_fixed,
                                                          nan=-1),
                                       index=row_names,
                                       columns=column_names)

        h_new = self.dec._proportion_constraint_h(h=h,
                                                  h_mask_fixed=h_mask_fixed_df)

        # Now we test if the first rows are 1 deleting the unknown one
        prop_columns = h_new[[0, 1, 2], :].sum(axis=0)
        self.assertTrue(np.all(np.round(prop_columns, 3) == 1.0),
                        'All known variables sum up 1.')

    def test_proportion_constraint_h_partial_fixed_multiple_k(self):

        h = np.array([[0.11692266, 0.49023124, 0.03157466],
                      [0.38518708, 0.08108722, 0.34163091],
                      [0.11876603, 0.26641129, 0.75638277],
                      [0.42219656, 0.17418068, 0.06863269]]).astype(float)

        # Row and column names
        column_names = ['column_1', 'column_2', 'column_3']
        row_names = ['row_1', 'row_2', 'cluster_unknown_01',
                     'cluster_unknown_02']

        # We define and np array initially
        h_mask_fixed = np.ones(shape=np.shape(h), dtype=bool)
        h_mask_fixed[[0, 1], :] = False
        # Then we generate the DataFrame that must have the names
        # (index, columns)
        h_mask_fixed_df = pd.DataFrame(data=np.nan_to_num(h_mask_fixed,
                                                          nan=-1),
                                       index=row_names,
                                       columns=column_names)

        h_new = self.dec._proportion_constraint_h(h=h,
                                                  h_mask_fixed=h_mask_fixed_df)

        # Now we test if the first rows are 1 deleting the unknown one
        prop_columns = h_new[[0, 1], :].sum(axis=0)
        self.assertTrue(np.all(np.round(prop_columns, 3) == 1.0),
                        'All known variables sum up 1 for k=2.')

    def test_proportion_constraint_h_partial_fixed_multiple_k_semi_defined(
            self):

        h = np.array([[0.11692266, 0.49023124, 0.03157466],
                      [0.38518708, 0.08108722, 0.34163091],
                      [0.11876603, 0.26641129, 0.75638277],
                      [0.42219656, 0.17418068, 0.06863269]]).astype(float)

        # Row and column names
        column_names = ['column_1', 'column_2', 'column_3']
        row_names = ['row_1', 'row_2', 'cluster_unknown_01',
                     'cluster_unknown_02']

        # We define and np array initially making sure that some are the one
        # that will change or not
        h_mask_fixed = np.ones(shape=np.shape(h), dtype=bool)
        h_mask_fixed[0, [0, 1]] = False
        h_mask_fixed[1, [0, 1]] = False

        # Then we generate the DataFrame that must have the names
        # (index, columns)
        h_mask_fixed_df = pd.DataFrame(data=np.nan_to_num(h_mask_fixed,
                                                          nan=-1),
                                       index=row_names,
                                       columns=column_names)

        h_new = self.dec._proportion_constraint_h(h=h,
                                                  h_mask_fixed=h_mask_fixed_df)

        # Now we test if the first rows are 1 deleting the unknown one
        prop_columns = h_new[[0, 1], :].sum(axis=0)
        self.assertTrue(np.all(np.round(prop_columns, 3) == 1.0),
                        'All known variables sum up 1 for k=2.')

    def test_scale_vector(self):
        # Row and column names
        column_names = ['column_1']
        w = np.array(np.transpose([self.vector_w])).astype(float)
        expected_w = np.array(np.transpose([self.vector_expected_w])).astype(
            float)

        w_df = pd.DataFrame(data=w,
                            index=self.row_names,
                            columns=column_names)
        w_new = self.dec._scale(matrix=w)

        np.testing.assert_allclose(
            expected_w, w_new, 1e-7, 0,
            'The scaled version of w is not correct.')

    def test_scale_matrix(self):
        # Row and column names
        column_names = ['column_1', 'column_2']
        w = np.array(np.transpose([self.vector_w, self.vector_w])).astype(
            float)

        expected_w = np.array(np.transpose([self.vector_expected_w,
                              self.vector_expected_w])).astype(float)

        w_df = pd.DataFrame(data=w,
                            index=self.row_names,
                            columns=column_names)
        w_new = self.dec._scale(matrix=w)

        np.testing.assert_allclose(
            expected_w, w_new, 1e-7, 0,
            'The scaled version of the matrix w is not correct.')

    def test_reference_scale_w_partial_fixed(self):

        # Row and column names
        column_names = ['column_1', 'column_2', 'unknown_1']
        row_names = ['row_1', 'row_2', 'row_3', 'row_4']

        # The idea is to fix the first two columns
        w = np.array([[0.11692266, 0.49023124, 0.03157466],
                      [0.38518708, 0.08108722, 0.34163091],
                      [0.11876603, 0.26641129, 0.75638277],
                      [0.42219656, 0.17418068, 0.06863269]]).astype(float)

        w_df = pd.DataFrame(data=np.nan_to_num(w, nan=-1),
                            index=row_names,
                            columns=column_names)

        # We define and np array initially
        w_mask_fixed = np.ones(shape=np.shape(w_df), dtype=bool)
        w_mask_fixed[:, [2]] = False
        # Then we generate the DataFrame that must have the names
        # (index, columns)
        w_mask_fixed_df = pd.DataFrame(data=np.nan_to_num(w_mask_fixed,
                                                          nan=-1),
                                       index=row_names,
                                       columns=column_names)

        w_new = self.dec._reference_scale_w(w=w,
                                            w_mask_fixed=w_mask_fixed_df)

        # Now we test if the first rows are 1 deleting the unknown one
        mean_columns = w_new[:, [2]].mean(axis=0)
        std_columns = np.std(w_new[:, [2]])
        self.assertTrue(np.all(w_new[:, [2]] > 0) and
                        np.all(std_columns > 0.0),
                        'All unknown variables are greater than zero and... ')

    def test_reference_scale_w_partial_fixed_multiple(self):
        # Row and column names
        column_names = ['column_1', 'unknown_2', 'unknown_1']
        row_names = ['row_1', 'row_2', 'row_3', 'row_4']

        # The idea is to fix the first two columns
        w = np.array([[0.11692266, 0.49023124, 0.03157466],
                      [0.38518708, 0.08108722, 0.34163091],
                      [0.11876603, 0.26641129, 0.75638277],
                      [0.42219656, 0.17418068, 0.06863269]]).astype(float)

        w_df = pd.DataFrame(data=np.nan_to_num(w, nan=-1),
                            index=row_names,
                            columns=column_names)

        # We define and np array initially
        w_mask_fixed = np.ones(shape=np.shape(w_df), dtype=bool)
        w_mask_fixed[:, [1, 2]] = False
        # Then we generate the DataFrame that must have the names
        # (index, columns)
        w_mask_fixed_df = pd.DataFrame(data=np.nan_to_num(w_mask_fixed,
                                                          nan=-1),
                                       index=row_names,
                                       columns=column_names)

        w_new = self.dec._reference_scale_w(w=w,
                                            w_mask_fixed=w_mask_fixed_df)

        # Now we test if the first rows are 1 deleting the unknown one
        mean_columns = w_new[:, [1, 2]].mean(axis=0)
        std_first_column = np.std(w_new[:, 1])
        std_second_column = np.std(w_new[:, 2])
        self.assertTrue(np.all(w_new[:, [1, 2]] > 0) and
                        std_first_column > 0.0 and
                        std_second_column > 0.0,
                        'All unknown variables are greater than zero and... ')

    def test_reference_scale_w_partial_fixed_multiple_rows_unfixed_fixed(self):
        # Row and column names
        column_names = ['column_1', 'unknown_2', 'unknown_1']
        row_names = ['row_1', 'row_2', 'row_3', 'row_4']

        # The idea is to fix the first two columns
        w = np.array([[0.11692266, 0.49023124, 0.03157466],
                      [0.38518708, 0.08108722, 0.34163091],
                      [0.11876603, 0.26641129, 0.75638277],
                      [0.42219656, 0.17418068, 0.06863269]]).astype(float)

        w_df = pd.DataFrame(data=np.nan_to_num(w, nan=-1),
                            index=row_names,
                            columns=column_names)

        # We define and np array initially
        w_mask_fixed = np.ones(shape=np.shape(w_df), dtype=bool)
        w_mask_fixed[:, [1, 2]] = False
        # We fixed just the first rows of column 1 to simulate a real scenario
        # where some genes are not in the reference.
        w_mask_fixed[[2, 3], 0] = False
        # Then we generate the DataFrame that must have the names
        # (index, columns)
        w_mask_fixed_df = pd.DataFrame(data=np.nan_to_num(w_mask_fixed,
                                                          nan=-1),
                                       index=row_names,
                                       columns=column_names)

        w_new = self.dec._reference_scale_w(w=w,
                                            w_mask_fixed=w_mask_fixed_df)

        # Now we test if the first rows are 1 deleting the unknown one
        mean_columns = w_new[:, [1, 2]].mean(axis=0)
        std_first_column = np.std(w_new[:, 1])
        std_second_column = np.std(w_new[:, 2])
        self.assertTrue(np.all(w_new[:, [1, 2]] > 0) and
                        std_first_column > 0.0 and
                        std_second_column > 0.0,
                        'All unknown variables are greater than zero and... ')

    def test_reference_scale_w_partial_fixed_multiple_rows_active(self):
        # Row and column names
        column_names = ['column_1', 'unknown_2', 'unknown_1']
        row_names = ['row_1', 'row_2', 'row_3', 'row_4']

        # The idea is to fix the first two columns
        w = np.array([[0.11692266, 0.49023124, 0.03157466],
                      [0.38518708, 0.08108722, 0.34163091],
                      [0.11876603, 0.26641129, 0.75638277],
                      [0.42219656, 0.17418068, 0.06863269]]).astype(float)

        w_df = pd.DataFrame(data=np.nan_to_num(w, nan=-1),
                            index=row_names,
                            columns=column_names)

        # We define and np array initially
        w_mask_fixed = np.ones(shape=np.shape(w_df), dtype=bool)
        w_mask_fixed[:, [1, 2]] = False
        # We fixed just the first rows of column 1 to simulate a real scenario
        # where some genes are not in the reference.
        w_mask_fixed[[2, 3], 0] = False
        # Then we generate the DataFrame that must have the names
        # (index, columns)
        w_mask_fixed_df = pd.DataFrame(data=np.nan_to_num(w_mask_fixed,
                                                          nan=-1),
                                       index=row_names,
                                       columns=column_names)

        w_new = self.dec._reference_scale_w(w=w,
                                            w_mask_fixed=w_mask_fixed_df,
                                            known_scaled=True)

        # Now we test if the first rows are 1 deleting the unknown one
        mean_columns = w_new[:, [1, 2]].mean(axis=0)
        std_first_column = np.std(w_new[:, 1])
        std_second_column = np.std(w_new[:, 2])
        self.assertTrue(np.all(w_new[:, [1, 2]] > 0) and
                        std_first_column > 0.0 and
                        std_second_column > 0.0 and
                        round(w_new[2, 0], 3) == 0.383 and
                        round(w_new[3, 0], 3) ==  1.361,
                        'All unknown variables are greater than zero and... ')

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

    def test_normalization_quantile(self):
        matrix_quantile_normalization = self.dec.normalization(
            matrix=self.bulk_methylation_matrix,
            normalization_type='quantile_norm')

        print(type(matrix_quantile_normalization))
        print("pandas.core.frame.DataFrame" ==
              type(matrix_quantile_normalization))
        print(isinstance(matrix_quantile_normalization,
                         pandas.core.frame.DataFrame))

        # quantile_normalization
        check_results_quantile_normalization = \
            np.all((matrix_quantile_normalization >= 0) &
                   (matrix_quantile_normalization <= 1))

        self.assertTrue(check_results_quantile_normalization,
                        'The matrix was not quantile normalized.')

    def test_normalization_zero_min_max(self):

        matrix_zero_min_max = self.dec.normalization(
            matrix=self.bulk_methylation_matrix,
            normalization_type='norm_zero_min_max')

        # quantile_normalization_min_max
        check_results_min_max = np.all(
            (matrix_zero_min_max >= 0) &
            (matrix_zero_min_max <= 1))

        self.assertTrue(check_results_min_max,
                        'The matrix was not min_max '
                        'normalized.')

    def test_normalization_quantile_norm_min_max(self):

        matrix_quantile_normalization_min_max = self.dec.normalization(
            matrix=self.bulk_methylation_matrix,
            normalization_type='quantile_norm_min_max')

        # quantile_normalization_min_max
        check_results_quantile_normalization_min_max = np.all(
            (matrix_quantile_normalization_min_max >= 0 &
             (matrix_quantile_normalization_min_max <= 1)))

        self.assertTrue(check_results_quantile_normalization_min_max,
                        'The matrix was not quantile_normalization_min_max '
                        'normalized.')

    def test_normalization_standard_scale(self):

        matrix_standard_scale = self.dec.normalization(
            matrix=self.bulk_methylation_matrix.values,
            normalization_type='standard_scaler')
        scaled_features_df = pd.DataFrame(
            matrix_standard_scale,
            index=self.bulk_methylation_matrix.index,
            columns=self.bulk_methylation_matrix.columns)

        # standard scale
        check_results_standard_scale = np.all(
            (scaled_features_df >= 0))

        print(self.bulk_methylation_matrix)
        print(matrix_standard_scale)

        self.assertTrue(check_results_standard_scale,
                        'The matrix was not standard scale '
                        'normalized.')

    def test_division_zero(self):
        result = self.dec.division(x=1, y=0)

        self.assertEqual(result, 0, 'Division by zero result in zero.')

    def test_scale_multiple_contrasts(self):
        # Suppose this is your matrix
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        normal_zero_one_matrix = self.dec.scale_matrix(matrix, 'zero_one')
        sigmoid_matrix = self.dec.scale_matrix(matrix, 'sigmoid')
        power_matrix_75 = self.dec.scale_matrix(matrix, 'power', value=0.75)
        power_matrix_50 = self.dec.scale_matrix(matrix, 'power', value=0.50)
        power_matrix_25 = self.dec.scale_matrix(matrix, 'power', value=0.25)
        exp_matrix = self.dec.scale_matrix(matrix, 'exp')

        self.assertTrue((np.all(np.round(normal_zero_one_matrix, 3) <= 1.0) and
                         np.all(np.round(normal_zero_one_matrix, 3) >= 0.0)),
                        'Scaled matrix between 0 and 1')

        self.assertTrue((np.all(np.round(sigmoid_matrix, 3) <= 1.0) and
                         np.all(np.round(sigmoid_matrix, 3) >= 0.0)),
                        'Scaled sigmoid matrix between 0 and 1')

        self.assertTrue((np.all(np.round(power_matrix_75, 3) <= 1.0) and
                         np.all(np.round(power_matrix_75, 3) >= 0.0)),
                        'Scaled power .75 matrix between 0 and 1')

        self.assertTrue((np.all(np.round(exp_matrix, 3) <= 1.0) and
                         np.all(np.round(exp_matrix, 3) >= 0.0)),
                        'Scaled exponential matrix between 0 and 1')


if __name__ == '__main__':
    unittest.main()
