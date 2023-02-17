#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the karhunen.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.simulation.karhunen import (
    _eigenvalues_linear,
    _eigenvalues_exponential,
    _eigenvalues_wiener,
    _simulate_eigenvalues,
    _make_coef,
    _initialize_centers,
    _initialize_cluster_std
)


class TestEigenvaluesLinear(unittest.TestCase):
    def test_eigenvalues_exponential_default(self):
        expected_output = np.array(
            [1.0, 0.6666666666666666, 0.3333333333333333]
        )
        result = _eigenvalues_linear()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_linear_n4(self):
        # Test that the function returns the correct values for n=4
        expected = np.array([1., 0.75, 0.5, 0.25])
        result = _eigenvalues_linear(n=4)
        np.testing.assert_allclose(result, expected)
    
    def test_eigenvalues_linear_n1(self):
        # Test that the function returns the correct values for n=1
        expected = np.array([1.0])
        result = _eigenvalues_linear(n=1)
        np.testing.assert_allclose(result, expected)


class TestEigenvaluesExponential(unittest.TestCase):
    def test_eigenvalues_exponential_default(self):
        expected_output = np.array(
            [0.36787944117144233, 0.22313016014842982, 0.1353352832366127]
        )
        result = _eigenvalues_exponential()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_exponential_n4(self):
        expected_output = np.array(
            [0.36787944, 0.22313016, 0.13533528, 0.082085]
        )
        result = _eigenvalues_exponential(n=4)
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_exponential_n1(self):
        expected_output = np.array([0.36787944117144233])
        result = _eigenvalues_exponential(n=1)
        np.testing.assert_allclose(result, expected_output)


class TestEigenvaluesWiener(unittest.TestCase):
    def test_eigenvalues_wiener_default(self):
        expected_output = np.array(
            [0.4052847345693511, 0.04503163717437235, 0.016211389382774045]
        )
        result = _eigenvalues_wiener()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_wiener_n4(self):
        expected_output = np.array(
            [
                0.4052847345693511, 0.04503163717437235,
                0.016211389382774045, 0.008271117032027573
            ]
        )
        result = _eigenvalues_wiener(n=4)
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_wiener_n1(self):
        expected_output = np.array([0.4052847345693511])
        result = _eigenvalues_wiener(n=1)
        np.testing.assert_allclose(result, expected_output)


class TestSimulateEigenvalues(unittest.TestCase):

    def test_linear_eigenvalues(self):
        expected_output = np.array(
            [1.0, 0.6666666666666666, 0.3333333333333333]
        )
        output = _simulate_eigenvalues('linear', n=3)
        np.testing.assert_allclose(output, expected_output)

    def test_exponential_eigenvalues(self):
        expected_output = np.array(
            [0.36787944117144233, 0.22313016014842982, 0.1353352832366127]
        )
        output = _simulate_eigenvalues('exponential', n=3)
        np.testing.assert_allclose(output, expected_output)

    def test_wiener_eigenvalues(self):
        expected_output = np.array(
            [0.4052847345693511, 0.04503163717437235, 0.016211389382774045]
        )
        output = _simulate_eigenvalues('wiener', n=3)
        np.testing.assert_allclose(output, expected_output)

    def test_raises_value_error_for_negative_n(self):
        with self.assertRaises(ValueError):
            _simulate_eigenvalues('linear', n=-1)

    def test_raises_not_implemented_error_for_unknown_method(self):
        with self.assertRaises(NotImplementedError):
            _simulate_eigenvalues('unknown', n=3)


class TestInitializeCenters(unittest.TestCase):
    def test_centers_initialized_to_zero(self):
        n_features = 3
        n_clusters = 4
        centers = None
        expected_centers = np.zeros((n_features, n_clusters))
        output = _initialize_centers(n_features, n_clusters, centers)
        np.testing.assert_array_equal(output, expected_centers)
    
    def test_centers_initialized_to_provided_value(self):
        n_features = 3
        n_clusters = 4
        centers = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        expected_centers = centers
        output = _initialize_centers(n_features, n_clusters, centers)
        np.testing.assert_array_equal(output, expected_centers)


class TestInitializeClusterStd(unittest.TestCase):
    def test_default(self):
        n_features = 2
        n_clusters = 3
        expected = np.ones((n_features, n_clusters))
        result = _initialize_cluster_std(n_features, n_clusters)
        np.testing.assert_array_equal(result, expected)

    def test_linear(self):
        n_features = 2
        n_clusters = 3
        expected = np.array([
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5]
        ])
        result = _initialize_cluster_std(
            n_features, n_clusters, cluster_std='linear'
        )
        np.testing.assert_allclose(result, expected)

    def test_exponential(self):
        n_features = 2
        n_clusters = 3
        expected = np.array([
            [0.36787944, 0.36787944, 0.36787944],
            [0.22313016, 0.22313016, 0.22313016]
        ])
        result = _initialize_cluster_std(
            n_features, n_clusters, cluster_std='exponential'
        )
        np.testing.assert_allclose(result, expected)

    def test_wiener(self):
        n_features = 2
        n_clusters = 3
        expected = np.array([
            [0.40528473, 0.40528473, 0.40528473],
            [0.04503164, 0.04503164, 0.04503164]
        ])
        result = _initialize_cluster_std(
            n_features, n_clusters, cluster_std='wiener'
        )
        np.testing.assert_allclose(result, expected)

    def test_custom(self):
        n_features = 2
        n_clusters = 3
        cluster_std = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]
        ])
        expected = cluster_std
        result = _initialize_cluster_std(
            n_features, n_clusters, cluster_std=cluster_std
        )
        np.testing.assert_array_equal(result, expected)


class TestMakeCoef(unittest.TestCase):
    def setUp(self):
        self.centers = np.array([[1, 2, 3], [0, 4, 6]])
        self.cluster_std = np.array([[0.5, 0.25, 1],[1, 0.1, 0.5]])
        self.n_obs = 4
        self.n_features = 2
        self.rnorm = np.random.default_rng(42).multivariate_normal

    def test_output_shape(self):
        coefs, labels = _make_coef(
            self.n_obs, self.n_features, self.centers, self.cluster_std
        )
        np.testing.assert_equal(coefs.shape, (self.n_obs, self.n_features))
        np.testing.assert_equal(labels.shape, (self.n_obs,))

    def test_output_labels(self):
        _, labels = _make_coef(
            self.n_obs, self.n_features,
            self.centers, self.cluster_std,
            self.rnorm
        )
        unique_labels = np.unique(labels)
        output = np.array([0, 1, 2])
        np.testing.assert_array_almost_equal(output, unique_labels)

    def test_output_coefs(self):
        coefs, _ = _make_coef(
            self.n_obs, self.n_features, self.centers, self.cluster_std, self.rnorm
        )
        output = np.array([
            [0.26462019, 0.30471708],
            [1.66507969, 0.7504512 ],
            [1.02448241, 3.58821468],
            [3.1278404 , 5.77638272]
        ])
        np.testing.assert_array_almost_equal(output, coefs)
