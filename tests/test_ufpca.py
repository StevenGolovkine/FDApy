#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class UFPCA in the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
import warnings

from FDApy.representation.argvals import DenseArgvals
from FDApy.representation.values import DenseValues
from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import (
    UFPCA
)


class UFPCATest(unittest.TestCase):
    def test_init(self):
        # Test default initialization
        fpca = UFPCA()
        self.assertEqual(fpca.method, 'covariance')
        self.assertIsNone(fpca.n_components)
        self.assertFalse(fpca.normalize)
        self.assertEqual(fpca.weights, 1)

        # Test custom initialization
        fpca = UFPCA(method='inner-product', n_components=3, normalize=True)
        self.assertEqual(fpca.method, 'inner-product')
        self.assertEqual(fpca.n_components, 3)
        self.assertTrue(fpca.normalize)

    def test_method(self):
        ufpc = UFPCA()
        ufpc.method = 'inner-product'
        self.assertEqual(ufpc.method, 'inner-product')

    def test_n_components(self):
        ufpc = UFPCA()
        ufpc.n_components = 4
        self.assertEqual(ufpc.n_components, 4)

    def test_normalize(self):
        ufpc = UFPCA()
        ufpc.normalize = True
        self.assertTrue(ufpc.normalize)


class TestFit(unittest.TestCase):
    def setUp(self):
        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)

        self.fdata_uni = kl.data
        self.fdata_sparse = kl.sparse_data

    def test_fit_covariance(self):
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.fdata_uni)

        np.testing.assert_almost_equal(self.fdata_uni.argvals['input_dim_0'], uf.mean.argvals['input_dim_0'])

        expected_mean = np.array([ 0.25140901,  0.24368371,  0.23613364,  0.22875992,   0.22156365,  0.21454593,  0.20770789,  0.20105062,   0.19457524,  0.18828286,  0.18217458,  0.17625151,   0.17051474,  0.16496535,  0.15960444,  0.15443302,   0.14945205,  0.1446623 ,  0.14006437,  0.13565859,   0.13144498,  0.12742319,  0.12359247,  0.11995162,   0.11649893,  0.11323215,  0.11014846,  0.1072444 ,   0.10451587,  0.10195804,  0.09956531,  0.09733124,   0.09524849,  0.09330869,  0.0915023 ,  0.08981849,   0.08824486,  0.08676714,  0.0853688 ,  0.08403038,   0.08276789,  0.08167392,  0.0807398 ,  0.0799563 ,   0.07931366,  0.07880153,  0.07840899,  0.07812456,   0.07793619,  0.07783122,  0.07779644,  0.07781805,   0.07788167,  0.07797232,  0.07807447,  0.07817198,   0.07824816,  0.07828572,  0.07826684,  0.0781731 ,   0.07798553,  0.07768462,  0.07739751,  0.07715871,   0.07695264,  0.07676479,  0.07658136,  0.07638902,   0.07617468,  0.0759254 ,  0.07562824,  0.07527024,   0.0748383 ,  0.07431915,  0.07369935,  0.07296523,   0.07210287,  0.07109813,  0.06993659,  0.06860361,   0.06708432,  0.06536363,  0.06342625,  0.06125675,   0.05883955,  0.056159  ,  0.0531994 ,  0.04994505,   0.0463803 ,  0.04248959,  0.03825745,  0.03366854,   0.02870763,  0.02335957,  0.01760933,  0.01144195,   0.00484258, -0.00220354, -0.0097111 , -0.01769469,  -0.02616881])
        np.testing.assert_array_almost_equal(uf.mean.values[0], expected_mean)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf.covariance, DenseFunctionalData)

    def test_fit_inner_product(self):
        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(self.fdata_sparse)

        np.testing.assert_almost_equal(self.fdata_sparse.argvals.to_dense()['input_dim_0'], uf.mean.argvals['input_dim_0'])

        expected_mean = np.array([ 0.28376655,  0.27226067,  0.26151169,  0.25122347,0.24151566,  0.23249356,  0.2237986 ,  0.21527625,0.20701826,  0.19913702,  0.19159585,  0.1844392 ,0.17766052,  0.17118638,  0.16501919,  0.15916035,0.1536228 ,  0.14839666,  0.14346798,  0.13882251,0.13443622,  0.13029322,  0.12639942,  0.12274674,0.11929506,  0.11601014,  0.1129029 ,  0.10998983,0.10724221,  0.10463146,  0.10217847,  0.09990654,0.0978084 ,  0.09587624,  0.09409314,  0.09244486,0.09090257,  0.08941525,  0.08794176,  0.08649046,0.08508513,  0.08368808,  0.08249575,  0.08169721,0.08115533,  0.08075489,  0.08040056,  0.08009846,0.07992895,  0.07981205,  0.07967847,  0.07956885,0.07946866,  0.07935828,  0.07921175,  0.07907922,0.07889339,  0.07851242,  0.07798815,  0.07735139,0.07669992,  0.07609985,  0.07550173,  0.07487512,0.0742502 ,  0.07360217,  0.07290657,  0.07218689,0.07140302,  0.07060384,  0.06979056,  0.06888127,0.06785737,  0.06669848,  0.06537975,  0.06389539,0.06221831,  0.06034092,  0.05828097,  0.05602698,0.0535707 ,  0.05089959,  0.04800349,  0.04486199,0.04145401,  0.03778645,  0.03383956,  0.02959964,0.02513171,  0.02041768,  0.01539678,  0.00996362,0.00408526, -0.00220905, -0.00887934, -0.01581486,-0.02315817, -0.03100882, -0.03959173, -0.04895527,-0.05899069])
        np.testing.assert_array_almost_equal(uf.mean.values[0], expected_mean)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf.covariance, DenseFunctionalData)

    # def test_fit_normalize(self):
    #     uf = UFPCA(n_components=2, method='covariance', normalize=True)
    #     uf.fit(self.fdata_uni)

    #     np.testing.assert_almost_equal(self.fdata_uni.argvals['input_dim_0'], uf.mean.argvals['input_dim_0'])

    #     expected_mean = np.array([ 1.18436009,  1.14796703,  1.11239951,  1.07766273,   1.04376189,  1.0107022 ,  0.97848891,  0.94712725,   0.91662246,  0.88697975,  0.85820433,  0.8303014 ,   0.80327609,  0.77713356,  0.75187888,  0.72751691,   0.70405208,  0.68148811,  0.65982777,  0.63907263,   0.61922277,  0.60027657,  0.5822305 ,  0.56507886,   0.5488136 ,  0.53342416,  0.51889721,  0.50521652,   0.49236272,  0.48031306,  0.46904117,  0.45851674,   0.44870513,  0.43956692,  0.43105723,  0.42312499,   0.41571178,  0.40875044,  0.40216299,  0.39585782,   0.38991038,  0.38475681,  0.38035627,  0.37666532,   0.37363791,  0.37122531,  0.3693761 ,  0.36803619,   0.36714877,  0.36665428,  0.36649045,  0.36659226,   0.36689194,  0.367319  ,  0.3678002 ,  0.36825956,   0.36861843,  0.36879541,  0.36870645,  0.36826483,   0.36738122,  0.36596366,  0.36461114,  0.36348616,   0.36251537,  0.36163044,  0.36076635,  0.35986024,   0.35885051,  0.35767615,  0.3562763 ,  0.35458979,   0.35255495,  0.35010932,  0.34718952,  0.34373115,   0.33966868,  0.33493542,  0.32946354,  0.32318403,   0.31602683,  0.30792083,  0.29879407,  0.28857378,   0.27718662,  0.26455884,  0.25061649,  0.23528562,   0.21849249,  0.20016377,  0.18022663,  0.15860879,   0.13523847,  0.11004436,  0.08295559,  0.05390175,   0.02281286, -0.01038065, -0.04574793, -0.08335772,  -0.12327839])
    #     np.testing.assert_array_almost_equal(uf.mean.values[0], expected_mean)

    #     self.assertIsInstance(uf.eigenvalues, np.ndarray)
    #     self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
    #     self.assertIsInstance(uf.covariance, DenseFunctionalData)

    def test_error_method(self):
        uf = UFPCA(n_components=2, method='error')
        with self.assertRaises(NotImplementedError):
            uf.fit(self.fdata_uni)

    def test_error_covariance_2d(self):
        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, dimension='2D'
        )
        kl.new(n_obs=10)

        uf = UFPCA(n_components=2, method='covariance')
        with self.assertRaises(ValueError):
            uf.fit(kl.data)


# class TestPace(unittest.TestCase):
#     def setUp(self):
#         argvals = np.linspace(0, 1, 10)
#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals, 
#             n_functions=5, random_state=42
#         )
#         kl.new(n_obs=50)
#         self.data = kl.data

#     def test_pace(self):
#         self.data.covariance()
        
#         uf = UFPCA(n_components=2, method='covariance')
#         uf.fit(self.data)

#         scores = uf._pace(self.data)

#         expected_scores = np.array([
#             [ 1.46015886e+00,  2.04695739e+00],
#             [ 4.94950452e-01,  1.78515078e-01],
#             [ 2.15517571e-01, -1.99545738e-01],
#             [ 4.73664501e-01, -1.56381155e-01],
#             [ 7.73468093e-01,  2.56786248e-01]
#         ])
#         np.testing.assert_array_almost_equal(
#             np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#         )


# class TestNumericalIntegration(unittest.TestCase):
#     def setUp(self):
#         warnings.simplefilter('ignore', category=UserWarning)

#         argvals = np.linspace(0, 1, 10)
#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals, 
#             n_functions=5, random_state=42
#         )
#         kl.new(n_obs=50)
#         self.data = kl.data

#     def test_numerical_integration(self):
#         self.data.covariance()
        
#         uf = UFPCA(n_components=2, method='covariance')
#         uf.fit(self.data)

#         scores = uf._numerical_integration(self.data)

#         expected_scores = np.array([
#             [ 1.42086765,  2.00210923],
#             [ 0.64501025,  0.34982201],
#             [ 0.11092837, -0.31895034],
#             [ 0.49936318, -0.12704974],
#             [ 0.78610985,  0.27121312]
#         ])
#         np.testing.assert_array_almost_equal(
#             np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#         )

#     def test_numerical_integration_2d(self):
#         argvals = np.linspace(0, 1, 10)
#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals, n_functions=5,
#             dimension='2D', random_state=42
#         )
#         kl.new(n_obs=50)
#         data = kl.data

#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(data)

#         scores = uf._numerical_integration(data)

#         expected_scores = np.array([
#             [-1.42086765, -2.00210923],
#             [-0.64501025, -0.34982201],
#             [-0.11092837,  0.31895034],
#             [-0.49936318,  0.12704974],
#             [-0.78610985, -0.27121312]
#         ])
#         np.testing.assert_array_almost_equal(
#             np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#         )

#     def test_value_error(self):
#         argvals = {
#             'input_dim_0': np.array([3, 4, 3]),
#             'input_dim_1': np.array([5, 6]),
#             'input_dim_2': np.array([1, 2, 4])
#         }
#         values = np.array([
#             [
#                 [[1, 2, 3], [1, 2, 3]],
#                 [[5, 6, 7], [5, 6, 7]],
#                 [[3, 4, 5], [3, 4, 5]]
#             ]
#         ])
#         data = DenseFunctionalData(DenseArgvals(argvals), DenseValues(values))

#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data)

#         with self.assertRaises(ValueError):
#             uf._numerical_integration(data)


class TestTransform(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)

        self.fdata_uni = kl.data
        self.fdata_sparse = kl.sparse_data

    # def test_error_innpro(self):
    #     uf = UFPCA(n_components=2, method='covariance')
    #     uf.fit(self.fdata_uni)
    #     with self.assertRaises(ValueError):
    #         uf.transform(None, method='InnPro')

    #     uf = UFPCA(n_components=2, method='inner-product')
    #     uf.fit(self.fdata_uni)
    #     with self.assertRaises(ValueError):
    #         uf.transform(self.fdata_uni, method='InnPro')

    def test_error_unkown_method(self):
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.fdata_uni)

        with self.assertRaises(ValueError):
            uf.transform(self.fdata_uni, method='error')

#     def test_pace(self):
#         self.data.covariance()

#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data)

#         scores = uf.transform(self.data, method='PACE')
#         expected_scores = np.array([
#             [-1.35951225, -1.83425839],
#             [-0.39430398,  0.03418355],
#             [-0.11487114,  0.41224429],
#             [-0.37301803,  0.36907972],
#             [-0.67282157, -0.0440876 ]
#         ])
#         np.testing.assert_array_almost_equal(
#             np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#         )

#     def test_numint(self):
#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data)

#         scores = uf.transform(self.data, method='NumInt')
#         expected_scores = np.array([
#             [-1.32124942, -1.7905831 ],
#             [-0.54539202, -0.13829588],
#             [-0.01131015,  0.53047647],
#             [-0.39974495,  0.33857587],
#             [-0.68649162, -0.05968698]
#         ])
#         np.testing.assert_array_almost_equal(
#             np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#         )

    # def test_innpro(self):
    #     uf = UFPCA(n_components=2, method='inner-product')
    #     uf.fit(self.data_uni)

    #     scores = uf.transform(self.data, method='InnPro')
    #     expected_scores = np.array([
    #         [-1.32124942, -1.7905831 ],
    #         [-0.54539202, -0.13829588],
    #         [-0.01131015,  0.53047647],
    #         [-0.39974495,  0.33857587],
    #         [-0.68649162, -0.05968698]
    #     ])
    #     np.testing.assert_array_almost_equal(
    #         np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
    #     )

#     # def test_normalize(self):
#     #     uf = UFPCA(n_components=2, method='inner-product', normalize=True)
#     #     uf.fit(self.data)

#     #     scores = uf.transform(self.data, method='InnPro')
#     #     expected_scores = np.array([
#     #         [-0.30298673, -0.41061355],
#     #         [-0.1250684 , -0.03171378],
#     #         [-0.00259362,  0.12164799],
#     #         [-0.09166885,  0.07764166],
#     #         [-0.15742512, -0.01368732]
#     #     ])
#     #     np.testing.assert_array_almost_equal(
#     #         np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#     #     )


# class TestInverseTranform(unittest.TestCase):
#     def setUp(self):
#         warnings.simplefilter('ignore', category=UserWarning)

#         argvals = np.linspace(0, 1, 10)
#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals,
#             n_functions=5, random_state=42
#         )
#         kl.new(n_obs=50)
#         self.data_1d = kl.data

#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals,
#             n_functions=5, dimension='2D', random_state=42
#         )
#         kl.new(n_obs=50)
#         self.data_2d = kl.data

#     def test_inverse_tranform_1D(self):
#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data_1d)
#         scores = uf.transform(self.data_1d)

#         expected_data = uf.weights * np.dot(scores, uf.eigenfunctions.values)
#         data = uf.inverse_transform(scores)

#         np.testing.assert_array_almost_equal(
#             data.values,
#             expected_data + uf.mean.values
#         )
    
#     def test_inverse_tranform_2D(self):
#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data_2d)
#         scores = uf.transform(self.data_2d)

#         expected_data = uf.weights * np.einsum(
#             'ij,jkl->ikl',
#             scores,
#             uf.eigenfunctions.values
#         )
#         data = uf.inverse_transform(scores)

#         np.testing.assert_array_almost_equal(
#             data.values,
#             expected_data + uf.mean.values
#         )

#     def test_error(self):
#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data_1d)
#         scores = uf.transform(self.data_1d)

#         argvals = {
#             'input_dim_0': np.array([3, 4, 3]),
#             'input_dim_1': np.array([5, 6]),
#             'input_dim_2': np.array([1, 2, 4])
#         }
#         values = np.array([
#             [
#                 [[1, 2, 3], [1, 2, 3]],
#                 [[5, 6, 7], [5, 6, 7]],
#                 [[3, 4, 5], [3, 4, 5]]
#             ]
#         ])
#         uf._eigenfunctions = DenseFunctionalData(DenseArgvals(argvals), DenseValues(values))

#         with self.assertRaises(ValueError):
#             uf.inverse_transform(scores)
