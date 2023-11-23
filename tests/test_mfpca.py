#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class MFPCA in the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
import warnings

from FDApy.representation.functional_data import MultivariateFunctionalData
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import (
    MFPCA
)


class MFPCATest(unittest.TestCase):
    def test_init(self):
        # Test default initialization
        fpca = MFPCA()
        self.assertEqual(fpca.method, 'covariance')
        self.assertIsNone(fpca.n_components)
        self.assertFalse(fpca.normalize)
        self.assertEqual(fpca.weights, None)

        # Test custom initialization
        fpca = MFPCA(method='inner-product', n_components=3, normalize=True)
        self.assertEqual(fpca.method, 'inner-product')
        self.assertEqual(fpca.n_components, 3)
        self.assertTrue(fpca.normalize)

    def test_method(self):
        ufpc = MFPCA()
        ufpc.method = 'inner-product'
        self.assertEqual(ufpc.method, 'inner-product')

    def test_n_components(self):
        ufpc = MFPCA()
        ufpc.n_components = [4, 3]
        self.assertEqual(ufpc.n_components, [4, 3])

    def test_normalize(self):
        ufpc = MFPCA()
        ufpc.normalize = True
        self.assertTrue(ufpc.normalize)


class TestFitCovariance(unittest.TestCase):
    def setUp(self):
        argvals = {'input_dim_0': np.linspace(0, 1, 10)}
        kl = KarhunenLoeve(
            basis_name=['fourier', 'legendre'],
            dimension=['1D', '1D'],
            argvals=argvals,
            n_functions=5,
            random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

    # def test_fit_covariance(self):
    #     # Initialize a UFPCA object
    #     uf = MFPCA(n_components=2)

    #     # Compute UFPCA covariance
    #     uf._fit_covariance(self.data)

    #     # Expected output
    #     expected_eigenvalues = np.array([2.12083689, 0.30121072])
    #     expected_eigenfunctions = np.array([
    #         [-0.81090817, -0.89598764, -0.96671352, -1.02049924, -1.055401,
    #          -1.0702047 , -1.06447946, -1.03859515, -0.99370311, -0.931681],
    #         [1.6177605 , 1.35775569, 1.0347427 , 0.66299649, 0.25913981,
    #          -0.15865381, -0.57153304, -0.96087493, -1.30918003, -1.60091268]
    #     ])

    #     # Test that eigenvalues and eigenfunctions are computed correctly
    #     np.testing.assert_array_almost_equal(
    #         uf.eigenvalues, expected_eigenvalues
    #     )
    #     np.testing.assert_array_almost_equal(
    #         np.abs(uf.eigenfunctions.values),
    #         np.abs(expected_eigenfunctions),
    #         decimal=5
    #     )
        
    # def test_with_known_covariance(self):
    #     # Compute empirical covariance
    #     covariance = self.data.covariance()

    #     # Initialize a UFPCA object
    #     uf = MFPCA(n_components=2)

    #     # Compute UFPCA covariance
    #     uf._fit_covariance(self.data, covariance=covariance)

    #     # Expected output
    #     expected_eigenvalues = np.array([2.12083689, 0.30121072])
    #     expected_eigenfunctions = np.array([
    #         [-0.81090817, -0.89598764, -0.96671352, -1.02049924, -1.055401,
    #          -1.0702047 , -1.06447946, -1.03859515, -0.99370311, -0.931681],
    #         [1.6177605 , 1.35775569, 1.0347427 , 0.66299649, 0.25913981,
    #          -0.15865381, -0.57153304, -0.96087493, -1.30918003, -1.60091268]
    #     ])

    #     # Test that eigenvalues and eigenfunctions are computed correctly
    #     np.testing.assert_array_almost_equal(
    #         uf.eigenvalues, expected_eigenvalues
    #     )
    #     np.testing.assert_array_almost_equal(
    #         np.abs(uf.eigenfunctions.values),
    #         np.abs(expected_eigenfunctions),
    #         decimal=5
    #     )
