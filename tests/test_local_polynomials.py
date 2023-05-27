#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class LocalPolynomial in the
local_polynomial.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from sklearn.preprocessing import PolynomialFeatures

from FDApy.preprocessing.smoothing.local_polynomial import (
    _gaussian,
    _epanechnikov,
    _tri_cube,
    _bi_square,
    _kernel,
    LocalPolynomial
)


class TestGaussian(unittest.TestCase):
    def test_gaussian(self):
        x = np.array([0, 1, 2, 3])
        expected_result = np.array([0.39894228, 0.24197072, 0.05399097, 0.00443185])
        result = _gaussian(x)
        np.testing.assert_allclose(result, expected_result, rtol=10e-7)

    def test_gaussian_with_negative_values(self):
        x = np.array([-1, -2, -3])
        expected_result = np.array([0.24197072, 0.05399097, 0.00443185])
        result = _gaussian(x)
        np.testing.assert_allclose(result, expected_result, rtol=10e-7)


class TestEpanechnikov(unittest.TestCase):
    def test_epanechnikov(self):
        x = np.array([-1, -0.5, 0, 0.5, 1])
        expected_result = np.array([0.    , 0.5625, 0.75  , 0.5625, 0.    ])
        result = _epanechnikov(x)
        np.testing.assert_allclose(result, expected_result)

    def test_epanechnikov_with_negative_values(self):
        x = np.array([-3, -2, -1])
        expected_result = np.array([0, 0, 0])
        result = _epanechnikov(x)
        np.testing.assert_allclose(result, expected_result)

    def test_epanechnikov_with_large_values(self):
        x = np.array([-10, 10])
        expected_result = np.array([0, 0])
        result = _epanechnikov(x)
        np.testing.assert_allclose(result, expected_result)


class TestTriCube(unittest.TestCase):
    def test_tri_cube(self):
        x = np.array([-1, -0.5, 0, 0.5, 1])
        expected_result = np.array([0.        , 0.66992188, 1.        , 0.66992188, 0.        ])
        result = _tri_cube(x)
        np.testing.assert_allclose(result, expected_result)

    def test_tri_cube_with_negative_values(self):
        x = np.array([-3, -2, -1])
        expected_result = np.array([0, 0, 0])
        result = _tri_cube(x)
        np.testing.assert_allclose(result, expected_result)

    def test_tri_cube_with_large_values(self):
        x = np.array([-10, 10])
        expected_result = np.array([0, 0])
        result = _tri_cube(x)
        np.testing.assert_allclose(result, expected_result)


class TestBiSquare(unittest.TestCase):
    def test_bi_square(self):
        x = np.array([-1, -0.5, 0, 0.5, 1])
        expected_result = np.array([0.    , 0.5625, 1.    , 0.5625, 0.    ])
        result = _bi_square(x)
        np.testing.assert_allclose(result, expected_result)

    def test_bi_square_with_negative_values(self):
        x = np.array([-3, -2, -1])
        expected_result = np.array([0, 0, 0])
        result = _bi_square(x)
        np.testing.assert_allclose(result, expected_result)

    def test_bi_square_with_large_values(self):
        x = np.array([-10, 10])
        expected_result = np.array([0, 0])
        result = _bi_square(x)
        np.testing.assert_allclose(result, expected_result)


class TestKernel(unittest.TestCase):
    def test_gaussian_kernel(self):
        kernel_func = _kernel('gaussian')
        self.assertEqual(kernel_func, _gaussian)

    def test_epanechnikov_kernel(self):
        kernel_func = _kernel('epanechnikov')
        self.assertEqual(kernel_func, _epanechnikov)

    def test_tricube_kernel(self):
        kernel_func = _kernel('tricube')
        self.assertEqual(kernel_func, _tri_cube)

    def test_bisquare_kernel(self):
        kernel_func = _kernel('bisquare')
        self.assertEqual(kernel_func, _bi_square)

    def test_unknown_kernel(self):
        with self.assertRaises(NotImplementedError):
            _kernel('unknown_kernel')


class LocalPolynomialTest(unittest.TestCase):
    def test_init(self):
        # Test default initialization
        lp = LocalPolynomial()
        self.assertEqual(lp.kernel_name, "gaussian")
        self.assertEqual(lp.bandwidth, 0.05)
        self.assertEqual(lp.degree, 1)
        self.assertTrue(callable(lp.kernel))
        self.assertIsInstance(lp.poly_features, PolynomialFeatures)
        self.assertFalse(lp.robust)

        # Test custom initialization
        lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.1, degree=2, robust=True)
        self.assertEqual(lp.kernel_name, "epanechnikov")
        self.assertEqual(lp.bandwidth, 0.1)
        self.assertEqual(lp.degree, 2)
        self.assertTrue(callable(lp.kernel))
        self.assertIsInstance(lp.poly_features, PolynomialFeatures)
        self.assertTrue(lp.robust)

    def test_kernel_name(self):
        lp = LocalPolynomial()
        lp.kernel_name = "tricube"
        self.assertEqual(lp.kernel_name, "tricube")

        with self.assertRaises(NotImplementedError):
            lp.kernel_name = "unknown"

    def test_bandwidth(self):
        lp = LocalPolynomial()
        lp.bandwidth = 0.2
        self.assertEqual(lp.bandwidth, 0.2)

        with self.assertRaises(ValueError):
            lp.bandwidth = -0.1  # Bandwidth must be strictly positive

        with self.assertRaises(ValueError):
            lp.bandwidth = 0  # Bandwidth must be strictly positive

    def test_degree(self):
        lp = LocalPolynomial()
        lp.degree = 3
        self.assertEqual(lp.degree, 3)
        self.assertIsInstance(lp.poly_features, PolynomialFeatures)

        with self.assertRaises(ValueError):
            lp.degree = -1  # Degree must be positive

    def test_kernel(self):
        lp = LocalPolynomial()
        self.assertTrue(callable(lp.kernel))

        with self.assertRaises(AttributeError):
            lp.kernel = 1  # Can't set attributes

    def test_poly_features(self):
        lp = LocalPolynomial()
        self.assertIsInstance(lp.poly_features, PolynomialFeatures)

        with self.assertRaises(AttributeError):
            lp.poly_features = 1  # Can't set attributes
