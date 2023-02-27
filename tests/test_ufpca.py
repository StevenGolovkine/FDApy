#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class UFPCA in the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

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
            uf.eigenfunctions.values, expected_eigenfunctions, decimal=5
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
            uf.eigenfunctions.values, expected_eigenfunctions, decimal=5
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
            uf.eigenfunctions.values, expected_eigenfunctions, decimal=5
        )

    def test_wrnings_2d(self):
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
