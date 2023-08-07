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


# class UFPCATest(unittest.TestCase):
#     def test_init(self):
#         # Test default initialization
#         ufpc = UFPCA()
#         self.assertEqual(ufpc.method, 'covariance')
#         self.assertIsNone(ufpc.n_components)
#         self.assertFalse(ufpc.normalize)
#         self.assertEqual(ufpc.weights, 1)

#         # Test custom initialization
#         ufpc = UFPCA(method='inner-product', n_components=3, normalize=True)
#         self.assertEqual(ufpc.method, 'inner-product')
#         self.assertEqual(ufpc.n_components, 3)
#         self.assertTrue(ufpc.normalize)

#     def test_method(self):
#         ufpc = UFPCA()
#         ufpc.method = 'inner-product'
#         self.assertEqual(ufpc.method, 'inner-product')

#     def test_n_components(self):
#         ufpc = UFPCA()
#         ufpc.n_components = 4
#         self.assertEqual(ufpc.n_components, 4)

#     def test_normalize(self):
#         ufpc = UFPCA()
#         ufpc.normalize = True
#         self.assertTrue(ufpc.normalize)


# class TestFitCovariance(unittest.TestCase):
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
