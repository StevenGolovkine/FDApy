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
        kl.new(n_obs=100)
        kl.sparsify(0.8, 0.05)

        self.fdata_uni = kl.data
        self.fdata_sparse = kl.sparse_data

#     def test_fit_covariance(self):
#         # Initialize a UFPCA object
#         uf = UFPCA(n_components=2)

#         # Compute UFPCA covariance
#         uf._fit_covariance(self.data)

#         # Expected output
#         expected_eigenvalues = np.array([1.24653269, 1.0329227])
#         expected_eigenfunctions = np.array([[ 0.68114728, -0.88313991, -2.11738635, -1.42791831, -0.05059818,  0.03678524, -0.75646213, -0.56949506,  0.57736795,  0.68114728],[ 0.7776058 , -0.40973431,  0.18979468,  1.41280427,  0.87919815, -0.90633626, -1.21192797,  0.50901826,  1.69502368,  0.7776058 ]])

#         # Test that eigenvalues and eigenfunctions are computed correctly
#         np.testing.assert_array_almost_equal(
#             uf.eigenvalues, expected_eigenvalues
#         )
#         np.testing.assert_array_almost_equal(
#             np.abs(uf.eigenfunctions.values),
#             np.abs(expected_eigenfunctions),
#             decimal=5
#         )
        
#     def test_with_known_covariance(self):
#         # Compute empirical covariance
#         covariance = self.data.covariance()

#         # Initialize a UFPCA object
#         uf = UFPCA(n_components=2)

#         # Compute UFPCA covariance
#         uf._fit_covariance(self.data, covariance=covariance)

#         # Expected output
#         expected_eigenvalues = np.array([1.24653269, 1.0329227])
#         expected_eigenfunctions = np.array([[ 0.68114728, -0.88313991, -2.11738635, -1.42791831, -0.05059818,  0.03678524, -0.75646213, -0.56949506,  0.57736795,  0.68114728],[ 0.7776058 , -0.40973431,  0.18979468,  1.41280427,  0.87919815, -0.90633626, -1.21192797,  0.50901826,  1.69502368,  0.7776058 ]])

#         # Test that eigenvalues and eigenfunctions are computed correctly
#         np.testing.assert_array_almost_equal(
#             uf.eigenvalues, expected_eigenvalues
#         )
#         np.testing.assert_array_almost_equal(
#             np.abs(uf.eigenfunctions.values),
#             np.abs(expected_eigenfunctions),
#             decimal=5
#         )


class TestFitCovariance(unittest.TestCase):
    def setUp(self) -> None:
        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=100)
        kl.sparsify(0.8, 0.05)

        self.fdata_uni = kl.data
        self.fdata_sparse = kl.sparse_data

    def test_fit_covariance_dense(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 31)})
        mean = self.fdata_uni.mean()

        ufpca = UFPCA(n_components=3)
        ufpca._fit_covariance(data=self.fdata_uni, points=points, mean=mean)

        expected_eigenvalues = np.array([0.249208  , 0.11510566, 0.05382122])
        np.testing.assert_array_almost_equal(ufpca.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions = np.array([[-2.03050399e-01, -1.28816058e-02,  1.58461253e-01,   3.12084925e-01,  4.48608916e-01,  5.67813128e-01,   6.71393644e-01,  7.61050939e-01,  8.38270483e-01,   9.04603116e-01,  9.61523904e-01,  1.01051016e+00,   1.05313115e+00,  1.09096996e+00,  1.12560006e+00,   1.15825499e+00,  1.18934515e+00,  1.21840081e+00,   1.24416558e+00,  1.26500126e+00,  1.27926904e+00,   1.28536490e+00,  1.28175573e+00,  1.26691694e+00,   1.23924699e+00,  1.19723682e+00,  1.13896279e+00,   1.06237725e+00,  9.69200082e-01,  8.60231831e-01,   7.35341051e-01], [-1.87465891e+00, -1.78530124e+00, -1.70093552e+00,  -1.61868936e+00, -1.53396349e+00, -1.44138716e+00,  -1.34249113e+00, -1.23838079e+00, -1.12938083e+00,  -1.01595434e+00, -8.98522110e-01, -7.77518446e-01,  -6.53351887e-01, -5.26421711e-01, -3.97158188e-01,  -2.66068751e-01, -1.33996886e-01, -2.09931192e-03,   1.28241691e-01,  2.55449294e-01,  3.77955334e-01,   4.94221661e-01,  6.02713920e-01,  7.01919576e-01,   7.90260404e-01,  8.66282936e-01,  9.27798870e-01,   9.72197049e-01,  1.00370793e+00,  1.02566344e+00,   1.03976094e+00], [ 1.75152846e+00,  1.48624714e+00,  1.23319139e+00,   9.90945002e-01,  7.57849945e-01,  5.33284961e-01,   3.20364665e-01,  1.21078648e-01, -6.34034473e-02,  -2.31694338e-01, -3.82397003e-01, -5.14149040e-01,  -6.25545611e-01, -7.15138244e-01, -7.81363353e-01,  -8.22434472e-01, -8.36519252e-01, -8.21777519e-01,  -7.76141442e-01, -6.97478613e-01, -5.83564682e-01,  -4.31959834e-01, -2.40317262e-01, -6.36913765e-03,   2.72342877e-01,  5.98631433e-01,  9.74050370e-01,   1.39697189e+00,  1.86981071e+00,  2.39976001e+00,   2.99455896e+00]])
        np.testing.assert_array_almost_equal(np.abs(ufpca.eigenfunctions.values), np.abs(expected_eigenfunctions))

        expected_covariance = np.array([ 0.52599863,  0.48580502,  0.44767562,  0.41090333,  0.37440838,  0.33703812,  0.2993508 ,  0.26172727,  0.22432305,  0.1873393 ,  0.1509695 ,  0.11540714,  0.08084351,  0.04747144,  0.01549939, -0.01482944, -0.04319645, -0.06921523, -0.09243229, -0.11234775, -0.12845617, -0.14024115, -0.14719473, -0.14881979, -0.14459023, -0.13397751, -0.11640148, -0.09144919, -0.05980236, -0.02157285,  0.02351   ])
        np.testing.assert_almost_equal(ufpca._covariance.values[0, 1], expected_covariance)

    def test_fit_covariance_irregular(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 31)})
        mean = self.fdata_sparse.mean()

        ufpca = UFPCA(n_components=3)
        ufpca._fit_covariance(data=self.fdata_sparse, points=points, mean=mean)

        expected_eigenvalues = np.array([0.25039568, 0.11548423, 0.05415895])
        np.testing.assert_array_almost_equal(ufpca.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions = np.array([[-0.22004529, -0.03219429,  0.13773589,  0.30001494,   0.46207611,  0.60961836,  0.73063454,  0.81031355,   0.85748449,  0.89850535,  0.94699771,  1.01080893,   1.06807786,  1.10111334,  1.12061952,  1.14621192,   1.18029694,  1.2156647 ,  1.24588578,  1.2699721 ,   1.28388024,  1.29036547,  1.28494763,  1.25210825,   1.20593077,  1.16052985,  1.10085329,  1.02966196,   0.95531412,  0.88651111,  0.85587579], [-1.98403232, -1.77704242, -1.66479685, -1.58372222,  -1.51019657, -1.43646396, -1.34599018, -1.23768187,  -1.10293329, -0.9648364 , -0.83998325, -0.73885463,  -0.65867079, -0.56506005, -0.44591305, -0.29288539,  -0.15283413, -0.02262808,  0.10788183,  0.23879689,   0.37646003,  0.50975427,  0.62639165,  0.72027683,   0.79483645,  0.86458321,  0.93778743,  0.99963232,   1.04445634,  1.07390297,  1.06067932], [ 1.73045808,  1.35721894,  1.1459679 ,  0.96243124,   0.78403516,  0.59366248,  0.36960302,  0.16265274,  -0.01558696, -0.18087274, -0.34024768, -0.47811362,  -0.57875294, -0.65694524, -0.70426965, -0.75096739,  -0.78398543, -0.79080569, -0.76423205, -0.70816079,  -0.64538609, -0.56468348, -0.41711984, -0.16919133,   0.15823433,  0.53777091,  0.95793989,  1.41576884,   1.911816  ,  2.49024978,  3.07602527]])
        np.testing.assert_array_almost_equal(np.abs(ufpca.eigenfunctions.values), np.abs(expected_eigenfunctions))

        expected_covariance = np.array([ 0.53613597,  0.46470795,  0.42477489,  0.39333758,  0.36382909,  0.33351483,  0.2975026 ,  0.25942122,  0.2182862 ,  0.1774658 ,  0.13973754,  0.10833557,  0.084021  ,  0.0587963 ,  0.03070905, -0.00433419, -0.03577734, -0.06328471, -0.08835832, -0.11129737, -0.13504648, -0.15652137, -0.16956731, -0.17034561, -0.16120691, -0.14725633, -0.13091351, -0.10937848, -0.08151572, -0.04448603,  0.00153244])
        np.testing.assert_almost_equal(ufpca._covariance.values[0, 1], expected_covariance)


# class TestFitInnerProduct(unittest.TestCase):
#     def setUp(self):
#         argvals = np.linspace(0, 1, 10)
#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals, 
#             n_functions=5, random_state=42
#         )
#         kl.new(n_obs=50)
#         self.data = kl.data

#     def test_fit_covariance(self):
#         # Initialize a UFPCA object
#         uf = UFPCA(n_components=2)

#         # Compute UFPCA covariance
#         uf._fit_inner_product(self.data)

#         # Expected output
#         expected_eigenvalues = np.array([1.23451254, 1.05652506])
#         expected_eigenfunctions = np.array([[-0.78057529,  0.90183037,  2.06354079,  1.23567777, -0.06523557,  0.08696559,  0.94945447,  0.56796216, -0.74594942, -0.78057529],[-0.71805317,  0.21150077, -0.49982238, -1.57957827, -0.85686903,  0.96901378,  1.25263279, -0.3996304 , -1.51419005, -0.71805317]])

#         # Test that eigenvalues and eigenfunctions are computed correctly
#         np.testing.assert_array_almost_equal(
#             uf.eigenvalues, expected_eigenvalues
#         )
#         np.testing.assert_array_almost_equal(
#             np.abs(uf.eigenfunctions.values),
#             np.abs(expected_eigenfunctions),
#             decimal=5
#         )

#     def test_warnings_2d(self):
#         kl = KarhunenLoeve(
#             basis_name='fourier', n_functions=5, dimension='2D'
#         )
#         kl.new(n_obs=50)
#         data = kl.data
#         uf = UFPCA(n_components=2, method='inner-product')

#         with self.assertWarns(UserWarning):
#             uf.fit(data)


# class TestFit(unittest.TestCase):
#     def setUp(self):
#         argvals = np.linspace(0, 1, 10)
#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals, 
#             n_functions=5, random_state=42
#         )
#         kl.new(n_obs=50)
#         self.data = kl.data

#     def test_raise_type_error(self):
#         # Initialize a UFPCA object
#         uf = UFPCA(n_components=2)

#         with self.assertRaises(TypeError):
#             uf.fit(data=np.array([1, 2, 3]))

#     def test_raise_value_error(self):
#         kl = KarhunenLoeve(
#             basis_name='fourier', n_functions=5, dimension='2D'
#         )
#         kl.new(n_obs=50)
#         data = kl.data

#         # Initialize a UFPCA object
#         uf = UFPCA(n_components=2, method='covariance')
#         with self.assertRaises(ValueError):
#             uf.fit(data)

#     def test_raise_not_implemented_error(self):
#         # Initialize a UFPCA object
#         uf = UFPCA(n_components=2, method='error')

#         with self.assertRaises(NotImplementedError):
#             uf.fit(data=self.data)

#     def test_fit_covariance(self):
#         uf = UFPCA(n_components=2, method='covariance')
#         uf.fit(self.data)

#         expected_eigenvalues = np.array([1.24653269, 1.0329227 ])
#         np.testing.assert_array_almost_equal(
#             uf.eigenvalues, expected_eigenvalues
#         )
    
#     def test_fit_inner_product(self):
#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data)

#         expected_eigenvalues = np.array([1.22160203, 1.01226424])
#         np.testing.assert_array_almost_equal(
#             uf.eigenvalues, expected_eigenvalues
#         )

#     # def test_fit_normalize(self):
#     #     uf = UFPCA(n_components=2, normalize=True)
#     #     uf.fit(self.data)

#     #     expected_eigenvalues = np.array([0.06555129, 0.05431821])
#     #     np.testing.assert_array_almost_equal(
#     #         uf.eigenvalues, expected_eigenvalues
#     #     )


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


# class TestTransform(unittest.TestCase):
#     def setUp(self):
#         warnings.simplefilter('ignore', category=UserWarning)

#         argvals = np.linspace(0, 1, 10)
#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals, 
#             n_functions=5, random_state=42
#         )
#         kl.new(n_obs=50)
#         self.data = kl.data

#     def test_error_innpro(self):
#         uf = UFPCA(n_components=2, method='covariance')
#         uf.fit(self.data)

#         with self.assertRaises(ValueError):
#             uf.transform(self.data, method='InnPro')

#     def test_error_unkown_method(self):
#         uf = UFPCA(n_components=2, method='covariance')
#         uf.fit(self.data)

#         with self.assertRaises(ValueError):
#             uf.transform(self.data, method='error')

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

#     def test_innpro(self):
#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data)

#         scores = uf.transform(self.data, method='InnPro')
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
