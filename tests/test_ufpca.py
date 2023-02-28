#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class UFPCA in the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
import warnings

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import (
    UFPCA
)


class TestFitCovariance(unittest.TestCase):
    def setUp(self):
        argvals = {'input_dim_0': np.linspace(0, 1, 10)}
        kl = KarhunenLoeve(
            basis_name='fourier', argvals=argvals, 
            n_functions=5, random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

    def test_fit_covariance(self):
        # Initialize a UFPCA object
        uf = UFPCA(n_components=2)

        # Compute UFPCA covariance
        uf._fit_covariance(self.data)

        # Expected output
        expected_eigenvalues = np.array([2.12083689, 0.30121072])
        expected_eigenfunctions = np.array([
            [-0.81090817, -0.89598764, -0.96671352, -1.02049924, -1.055401,
             -1.0702047 , -1.06447946, -1.03859515, -0.99370311, -0.931681],
            [1.6177605 , 1.35775569, 1.0347427 , 0.66299649, 0.25913981,
             -0.15865381, -0.57153304, -0.96087493, -1.30918003, -1.60091268]
        ])

        # Test that eigenvalues and eigenfunctions are computed correctly
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues
        )
        np.testing.assert_array_almost_equal(
            np.abs(uf.eigenfunctions.values),
            np.abs(expected_eigenfunctions),
            decimal=5
        )
        
    def test_with_known_covariance(self):
        # Compute empirical covariance
        covariance = self.data.covariance()

        # Initialize a UFPCA object
        uf = UFPCA(n_components=2)

        # Compute UFPCA covariance
        uf._fit_covariance(self.data, covariance=covariance)

        # Expected output
        expected_eigenvalues = np.array([2.12083689, 0.30121072])
        expected_eigenfunctions = np.array([
            [-0.81090817, -0.89598764, -0.96671352, -1.02049924, -1.055401,
             -1.0702047 , -1.06447946, -1.03859515, -0.99370311, -0.931681],
            [1.6177605 , 1.35775569, 1.0347427 , 0.66299649, 0.25913981,
             -0.15865381, -0.57153304, -0.96087493, -1.30918003, -1.60091268]
        ])

        # Test that eigenvalues and eigenfunctions are computed correctly
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues
        )
        np.testing.assert_array_almost_equal(
            np.abs(uf.eigenfunctions.values),
            np.abs(expected_eigenfunctions),
            decimal=5
        )


class TestFitInnerProduct(unittest.TestCase):
    def setUp(self):
        argvals = {'input_dim_0': np.linspace(0, 1, 10)}
        kl = KarhunenLoeve(
            basis_name='fourier', argvals=argvals, 
            n_functions=5, random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

    def test_fit_covariance(self):
        # Initialize a UFPCA object
        uf = UFPCA(n_components=2)

        # Compute UFPCA covariance
        uf._fit_inner_product(self.data)

        # Expected output
        expected_eigenvalues = np.array([2.09295274, 0.31180837])
        expected_eigenfunctions = np.array([
            [-0.82550138, -0.90805814, -0.97573227, -1.02608867, -1.05735198,
             -1.0684853, -1.05923499, -1.0301395, -0.9825019, -0.91832733],
            [-1.61444622, -1.35183415, -1.0264989, -0.65293216, -0.24794233,
             0.17015282, 0.5824073, 0.97015873, 1.31592627, 1.60425072]
        ])

        # Test that eigenvalues and eigenfunctions are computed correctly
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues
        )
        np.testing.assert_array_almost_equal(
            np.abs(uf.eigenfunctions.values),
            np.abs(expected_eigenfunctions),
            decimal=5
        )

    def test_warnings_2d(self):
        kl = KarhunenLoeve(
            basis_name='fourier', n_functions=5, dimension='2D'
        )
        kl.new(n_obs=50)
        data = kl.data
        uf = UFPCA(n_components=2, method='inner-product')

        with self.assertWarns(UserWarning):
            uf.fit(data)


class TestFit(unittest.TestCase):
    def setUp(self):
        argvals = {'input_dim_0': np.linspace(0, 1, 10)}
        kl = KarhunenLoeve(
            basis_name='fourier', argvals=argvals, 
            n_functions=5, random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

    def test_raise_type_error(self):
        # Initialize a UFPCA object
        uf = UFPCA(n_components=2)

        with self.assertRaises(TypeError):
            uf.fit(data=np.array([1, 2, 3]))

    def test_raise_value_error(self):
        kl = KarhunenLoeve(
            basis_name='fourier', n_functions=5, dimension='2D'
        )
        kl.new(n_obs=50)
        data = kl.data

        # Initialize a UFPCA object
        uf = UFPCA(n_components=2, method='covariance')
        with self.assertRaises(ValueError):
            uf.fit(data)

    def test_raise_not_implemented_error(self):
        # Initialize a UFPCA object
        uf = UFPCA(n_components=2, method='error')

        with self.assertRaises(NotImplementedError):
            uf.fit(data=self.data)

    def test_fit_covariance(self):
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.data)

        expected_eigenvalues = np.array([2.12083689, 0.30121072])
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues
        )
    
    def test_fit_inner_product(self):
        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(self.data)

        expected_eigenvalues = np.array([2.07842015, 0.29518651])
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues
        )

    def test_fit_normalize(self):
        uf = UFPCA(n_components=2, normalize=True)
        uf.fit(self.data)

        expected_eigenvalues = np.array([0.37250134, 0.0529043])
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues
        )


class TestPace(unittest.TestCase):
    def setUp(self):
        argvals = {'input_dim_0': np.linspace(0, 1, 10)}
        kl = KarhunenLoeve(
            basis_name='fourier', argvals=argvals, 
            n_functions=5, random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

    def test_pace(self):
        self.data.covariance()
        
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.data)

        scores = uf._pace(self.data)

        expected_scores = np.array([
            [-2.70324475e-01,  -7.54032250e-01],
            [ 1.89943408e+00,  -5.17586244e-01],
            [-2.30573992e+00,  -2.14846678e-01],
            [ 8.99858737e-01,  -5.28052839e-01],
            [-2.25425896e-01, 2.02702135e-01]
        ])
        np.testing.assert_array_almost_equal(
            np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
        )


class TestNumericalIntegration(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        argvals = {'input_dim_0': np.linspace(0, 1, 10)}
        kl = KarhunenLoeve(
            basis_name='fourier', argvals=argvals, 
            n_functions=5, random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

    def test_numerical_integration(self):
        self.data.covariance()
        
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.data)

        scores = uf._numerical_integration(self.data)

        expected_scores = np.array([
            [-0.26337429, -0.75976051],
            [ 1.89087774, -0.51679176],
            [-2.31549977, -0.21327057],
            [ 0.87391109, -0.52526401],
            [-0.21814701, 0.19876454]
        ])
        np.testing.assert_array_almost_equal(
            np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
        )

    def test_numerical_integration_2d(self):
        argvals = {'input_dim_0': np.linspace(0, 1, 10)}
        kl = KarhunenLoeve(
            basis_name='fourier', argvals=argvals, n_functions=5,
            dimension='2D', random_state=42
        )
        kl.new(n_obs=50)
        data = kl.data

        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(data)

        scores = uf._numerical_integration(data)

        expected_scores = np.array([
            [-0.88091993, 1.23604672],
            [-2.36805008, -0.0315452],
            [2.23914369, -1.25243888],
            [0.60748215, -0.7046951],
            [0.49341521, 0.39777882]
        ])
        np.testing.assert_array_almost_equal(
            np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
        )

    def test_value_error(self):
        argvals = {
            'input_dim_0': np.array([3, 4, 3]),
            'input_dim_1': np.array([5, 6]),
            'input_dim_2': np.array([1, 2, 4])
        }
        values = np.array([
            [
                [[1, 2, 3], [1, 2, 3]],
                [[5, 6, 7], [5, 6, 7]],
                [[3, 4, 5], [3, 4, 5]]
            ]
        ])
        data = DenseFunctionalData(argvals, values)

        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(self.data)

        with self.assertRaises(ValueError):
            uf._numerical_integration(data)


class TestTransform(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        argvals = {'input_dim_0': np.linspace(0, 1, 10)}
        kl = KarhunenLoeve(
            basis_name='fourier', argvals=argvals, 
            n_functions=5, random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

    def test_error_innpro(self):
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.data)

        with self.assertRaises(ValueError):
            uf.transform(self.data, method='InnPro')

    def test_error_unkown_method(self):
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.data)

        with self.assertRaises(ValueError):
            uf.transform(self.data, method='error')

    def test_pace(self):
        self.data.covariance()

        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(self.data)

        scores = uf.transform(self.data, method='PACE')
        expected_scores = np.array([
            [-0.39121016,  0.62482911],
            [ 1.77859707,  0.38835183],
            [-2.42667221,  0.085571  ],
            [ 0.77899918,  0.39881965],
            [-0.34631175, -0.33203334]
        ])
        np.testing.assert_array_almost_equal(
            np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
        )

    def test_numint(self):
        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(self.data)

        scores = uf.transform(self.data, method='NumInt')
        expected_scores = np.array([
            [-0.38336168,  0.63031539],
            [ 1.77089036,  0.38734664],
            [-2.43548716,  0.08382546],
            [ 0.75392371,  0.39581889],
            [-0.33813439, -0.32820965]
        ])
        np.testing.assert_array_almost_equal(
            np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
        )

    def test_innpro(self):
        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(self.data)

        scores = uf.transform(self.data, method='InnPro')
        expected_scores = np.array([
            [-0.38336168,  0.63031539],
            [ 1.77089036,  0.38734664],
            [-2.43548716,  0.08382546],
            [ 0.75392371,  0.39581889],
            [-0.33813439, -0.32820965]
        ])
        np.testing.assert_array_almost_equal(
            np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
        )

    def test_normalize(self):
        uf = UFPCA(n_components=2, method='inner-product', normalize=True)
        uf.fit(self.data)

        scores = uf.transform(self.data, method='InnPro')
        expected_scores = np.array([
            [-0.16066415,  0.2641607 ],
            [ 0.74216755,  0.16233422],
            [-1.02069534,  0.03513065],
            [ 0.31596406,  0.16588488],
            [-0.14170972, -0.13755033]
        ])
        np.testing.assert_array_almost_equal(
            np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
        )


class TestInverseTranform(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        argvals = {'input_dim_0': np.linspace(0, 1, 10)}
        kl = KarhunenLoeve(
            basis_name='fourier', argvals=argvals,
            n_functions=5, random_state=42
        )
        kl.new(n_obs=50)
        self.data_1d = kl.data

        kl = KarhunenLoeve(
            basis_name='fourier', argvals=argvals,
            n_functions=5, dimension='2D', random_state=42
        )
        kl.new(n_obs=50)
        self.data_2d = kl.data

    def test_inverse_tranform_1D(self):
        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(self.data_1d)
        scores = uf.transform(self.data_1d)

        expected_data = uf.weights * np.dot(scores, uf.eigenfunctions.values)
        data = uf.inverse_transform(scores)

        np.testing.assert_array_almost_equal(
            data.values,
            expected_data + uf.mean.values
        )
    
    def test_inverse_tranform_2D(self):
        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(self.data_2d)
        scores = uf.transform(self.data_2d)

        expected_data = uf.weights * np.einsum(
            'ij,jkl->ikl',
            scores,
            uf.eigenfunctions.values
        )
        data = uf.inverse_transform(scores)

        np.testing.assert_array_almost_equal(
            data.values,
            expected_data + uf.mean.values
        )

    def test_error(self):
        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(self.data_1d)
        scores = uf.transform(self.data_1d)

        argvals = {
            'input_dim_0': np.array([3, 4, 3]),
            'input_dim_1': np.array([5, 6]),
            'input_dim_2': np.array([1, 2, 4])
        }
        values = np.array([
            [
                [[1, 2, 3], [1, 2, 3]],
                [[5, 6, 7], [5, 6, 7]],
                [[3, 4, 5], [3, 4, 5]]
            ]
        ])
        uf.eigenfunctions = DenseFunctionalData(argvals, values)

        with self.assertRaises(ValueError):
            uf.inverse_transform(scores)
