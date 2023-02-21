#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the brownian.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.simulation.brownian import (
    _init_brownian,
    _standard_brownian,
    _geometric_brownian,
    _fractional_brownian,
    _simulate_brownian,
    Brownian
)


class TestInitBrownian(unittest.TestCase):
    def test_delta(self):
        # Test if delta is calculated correctly
        argvals = np.array([1, 2, 3, 4, 5])
        delta, _ = _init_brownian(argvals)
        np.testing.assert_almost_equal(delta, 0.8)
    
    def test_argvals(self):
        # Test if the function returns the correct argvals
        argvals = np.array([1, 2, 3, 4, 5])
        _, returned_argvals = _init_brownian(argvals)
        np.testing.assert_array_equal(returned_argvals, argvals)


class TestStandardBrownian(unittest.TestCase):
    def test_shape(self):
        # Test if the output array has the correct shape
        argvals = np.arange(0, 1, 0.01)
        output = _standard_brownian(argvals)
        self.assertEqual(output.shape, argvals.shape)

    def test_starting_point(self):
        # Test if the starting point is correct
        argvals = np.arange(0, 1, 0.01)
        init_point = 1.0
        output = _standard_brownian(argvals, init_point=init_point)
        self.assertAlmostEqual(output[0], init_point)

    def test_reproducibility(self):
        # Test if the function is reproducible with the same rnorm
        argvals = np.arange(0, 1, 0.01)
        rnorm = np.random.default_rng(42).normal
        output1 = _standard_brownian(argvals, rnorm=rnorm)

        rnorm = np.random.default_rng(42).normal
        output2 = _standard_brownian(argvals, rnorm=rnorm)
        np.testing.assert_array_equal(output1, output2)


class TestGeometricBrownian(unittest.TestCase):
    def setUp(self):
        self.argvals = np.arange(0, 1, 0.01)

    def test_init_point(self):
        with self.assertRaises(ValueError):
            _geometric_brownian(self.argvals, init_point=0.0)

    def test_output_shape(self):
        output = _geometric_brownian(self.argvals)
        self.assertEqual(output.shape, self.argvals.shape)

    def test_positive_values(self):
        output = _geometric_brownian(self.argvals)
        self.assertTrue(np.all(output > 0))

    def test_reproducibility(self):
        output1 = _geometric_brownian(
            self.argvals, rnorm=np.random.default_rng(42).normal
        )
        output2 = _geometric_brownian(
            self.argvals, rnorm=np.random.default_rng(42).normal
        )
        np.testing.assert_allclose(output1, output2)


class TestFractionalBrownian(unittest.TestCase):
    def setUp(self):
        self.argvals = np.arange(0, 1, 0.01)

    def test_output_shape(self):
        
        fbm = _fractional_brownian(
            self.argvals, hurst=0.7, rnorm=np.random.default_rng(42).normal
        )
        self.assertEqual(fbm.shape, (100,))

    def test_reproducibility(self):
        output1 = _fractional_brownian(
            self.argvals, hurst=0.7, rnorm=np.random.default_rng(42).normal
        )
        output2 = _fractional_brownian(
            self.argvals, hurst=0.7, rnorm=np.random.default_rng(42).normal
        )
        np.testing.assert_array_almost_equal(output1, output2)

    def test_negative_hurst(self):
        with self.assertRaises(ValueError):
            _fractional_brownian(self.argvals, hurst=-0.2)

    def test_zero_hurst(self):
        with self.assertRaises(ValueError):
            _fractional_brownian(self.argvals, hurst=0)


class TestSimulateBrownian(unittest.TestCase):
    def setUp(self):
        self.argvals = np.arange(0, 1, 0.01)

    def test_error(self):
        with self.assertRaises(NotImplementedError):
            _simulate_brownian(name='error', argvals=self.argvals)

    def test_standard_brownian(self):
        """Test if _simulate_brownian returns a standard brownian motion."""
        brownian_type = 'standard'
        brownian = _simulate_brownian(brownian_type, self.argvals)
        self.assertEqual(brownian[0], 0)

    def test_geometric_brownian(self):
        """Test if _simulate_brownian returns a geometric brownian motion."""
        brownian_type = 'geometric'
        mu, sigma, init_point = 0.1, 0.5, 1.0
        brownian = _simulate_brownian(
            brownian_type, self.argvals,
            mu=mu, sigma=sigma, init_point=init_point
        )
        self.assertTrue(np.all(brownian > 0))

    def test_fractional_brownian(self):
        """Test if _simulate_brownian returns a fractional brownian motion."""
        brownian_type = 'fractional'
        hurst = 0.6
        brownian = _simulate_brownian(
            brownian_type, self.argvals, hurst=hurst
        )
        self.assertEqual(brownian.shape, (100,))


class TestBrownian(unittest.TestCase):
    def test_standard_brownian(self):
        # Test standard Brownian motion simulation
        brownian = Brownian(name='standard')
        brownian.new(n_obs=1)
        self.assertIsInstance(brownian.data, DenseFunctionalData)
        self.assertEqual(brownian.data.n_obs, 1)

    def test_geometric_brownian(self):
        # Test geometric Brownian motion simulation
        brownian = Brownian(name='geometric', random_state=42)
        brownian.new(n_obs=1, mu=0.05, sigma=0.1, init_point=100)
        self.assertIsInstance(brownian.data, DenseFunctionalData)
        self.assertEqual(brownian.data.n_obs, 1)

    def test_fractional_brownian(self):
        # Test fractional Brownian motion simulation
        brownian = Brownian(name='fractional', random_state=42)
        brownian.new(n_obs=1, hurst=0.4)
        self.assertIsInstance(brownian.data, DenseFunctionalData)
        self.assertEqual(brownian.data.n_obs, 1)
