#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the simulation.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
from unittest.mock import patch

from FDApy.representation.argvals import DenseArgvals
from FDApy.representation.values import DenseValues
from FDApy.representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData,
    MultivariateFunctionalData
)
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.simulation.simulation import (
    Simulation,
    _add_noise_univariate_data,
    _sparsify_univariate_data
)


class TestAddNoiseUnivariateData(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 1, num=1001)
        self.y = np.sin(2 * np.pi * self.x)
        self.data = DenseFunctionalData(
            DenseArgvals({'input_dim_0': self.x}),
            DenseValues(self.y[np.newaxis])
        )

    def test_output_shape(self):
        # Test if the output shape is the same as the input shape
        noisy_data = _add_noise_univariate_data(
            self.data,
            noise_variance=1.0,
            rnorm=np.random.default_rng(42).normal
        )
        self.assertEqual(noisy_data.values.shape, self.data.values.shape)

    def test_output_variance(self):
        # Test if the output variance is close to the input variance plus noise variance
        noise_variance = 1.0
        noisy_data = _add_noise_univariate_data(
            self.data,
            noise_variance=noise_variance,
            rnorm=np.random.default_rng(42).normal
        )

        diff = self.data - noisy_data
        output = np.var(diff.values)
        np.testing.assert_almost_equal(output, noise_variance, decimal=1)

    def test_output_mean(self):
        # Test if the output mean is close to the input mean
        noisy_data = _add_noise_univariate_data(
            self.data,
            noise_variance=1.0,
            rnorm=np.random.default_rng(42).normal
        )
        diff = self.data - noisy_data
        output = np.mean(diff.values)
        np.testing.assert_almost_equal(output, 0, decimal=1)


class TestSparsifyUnivariateData(unittest.TestCase):
    def setUp(self):
        self.rchoice = np.random.default_rng(42).choice
        self.runif = np.random.default_rng(42).uniform
        self.x = np.linspace(0, 1, num=101)
        self.y = np.sin(2 * np.pi * self.x)
        self.data = DenseFunctionalData(
            DenseArgvals({'input_dim_0': self.x}), DenseValues(self.y[np.newaxis])
        )

    def test_sparsify(self):
        sparse_data = _sparsify_univariate_data(
            self.data,
            percentage=0.8,
            epsilon=0.05,
            runif=self.runif, rchoice=self.rchoice
        )

        # Check if number of observations is between expected values
        expected_percentage = 0.8
        expected_epsilon = 0.05
        expected_min_perc = max(0, expected_percentage - expected_epsilon)
        expected_max_perc = min(1, expected_percentage + expected_epsilon)

        actual_percentage = (
            sparse_data.argvals.n_points[0][0] / 
            self.data.n_points[0]
        )
        self.assertGreaterEqual(actual_percentage, expected_min_perc)
        self.assertLessEqual(actual_percentage, expected_max_perc)

        # Check if the sparse data is a subclass of IrregularFunctionalData
        self.assertIsInstance(sparse_data, IrregularFunctionalData)


class TestCheckData(unittest.TestCase):
    def setUp(self):
        self.simulation = KarhunenLoeve('fourier', random_state=1)

    def test_check_data(self):
        with self.assertRaises(ValueError):
            self.simulation._check_data()


class TestCheckDimension(unittest.TestCase):
    def setUp(self):
        self.simulation = KarhunenLoeve('fourier', dimension='2D')
        self.simulation.new(n_obs=1, n_clusters=1)
        self.simulation_multi = KarhunenLoeve(
            ['fourier', 'bsplines'], dimension=['2D', '2D']
        )
        self.simulation_multi.new(n_obs=1, n_clusters=1)

    def test_dimension(self):
        with self.assertRaises(ValueError):
            self.simulation._check_dimension()

    def test_dimension_multi(self):
        with self.assertRaises(ValueError):
            self.simulation_multi._check_dimension()


class TestSimulationUnivariate(unittest.TestCase):
    def setUp(self):
        self.simulation = KarhunenLoeve('fourier', random_state=1)
        self.simulation.new(
            n_obs=50, n_clusters=1, argvals=np.linspace(0, 1, 10)
        )

    def test_new(self):
        self.assertEqual(self.simulation.data.n_obs, 50)

    def test_add_noise(self):
        self.simulation.add_noise(noise_variance=0.5)
        self.assertIsInstance(
            self.simulation.noisy_data, DenseFunctionalData
        )

    def test_sparsify(self):
        self.simulation.sparsify(percentage=0.5, epsilon=0.1)
        self.assertIsInstance(
            self.simulation.sparse_data, IrregularFunctionalData
        )

    def test_add_noise_and_sparsify(self):
        self.simulation.add_noise_and_sparsify(noise_variance=0.1, percentage=0.5, epsilon=0.1)
        self.assertIsInstance(
            self.simulation.noisy_data, DenseFunctionalData
        )
        self.assertIsInstance(
            self.simulation.sparse_data, IrregularFunctionalData
        )

class TestSimulationMultivariate(unittest.TestCase):
    def setUp(self):
        self.simulation = KarhunenLoeve(
            ['fourier', 'bsplines'], random_state=1
        )
        self.simulation.new(
            n_obs=50, n_clusters=1, argvals=np.linspace(0, 1, 10)
        )

    def test_new(self):
        self.assertEqual(self.simulation.data.n_obs, 50)

    def test_add_noise(self):
        self.simulation.add_noise(noise_variance=0.5)
        self.assertIsInstance(
            self.simulation.noisy_data, MultivariateFunctionalData
        )

    def test_sparsify(self):
        self.simulation.sparsify(percentage=0.5, epsilon=0.1)
        self.assertIsInstance(
            self.simulation.sparse_data, MultivariateFunctionalData
        )

    def test_add_noise_and_sparsify(self):
        self.simulation.add_noise_and_sparsify(noise_variance=0.1, percentage=0.5, epsilon=0.1)
        self.assertIsInstance(
            self.simulation.noisy_data, MultivariateFunctionalData
        )
        self.assertIsInstance(
            self.simulation.noisy_data.data[0], DenseFunctionalData
        )
        self.assertIsInstance(
            self.simulation.sparse_data, MultivariateFunctionalData
        )
        self.assertIsInstance(
            self.simulation.sparse_data.data[0], IrregularFunctionalData
        )


class TestSimulationRandom(unittest.TestCase):
    def setUp(self) -> None:
        self.simulation = KarhunenLoeve(['fourier', 'bsplines'])
        self.simulation.new(n_obs=50, n_clusters=1)

    def test_add_noise(self):
        self.simulation.add_noise(noise_variance=0.5)
        self.assertTrue(True)

    def test_sparsify(self):
        self.simulation.sparsify(percentage=0.5, epsilon=0.1)
        self.assertTrue(True)


class MyAbcClassTest(unittest.TestCase):
    @patch.multiple(Simulation, __abstractmethods__=set())
    def test(self):
        self.instance = Simulation('fourier')
        self.instance.new(1)
