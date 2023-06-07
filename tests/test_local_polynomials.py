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
    _compute_kernel,
    _local_regression,
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


class TestComputeKernel(unittest.TestCase):
    def test_kernel_one_dimensional(self):
        x = np.linspace(0, 1, 101)
        x0 = 0.3
        bandwidth = 0.2
        kernel = _epanechnikov

        output = _compute_kernel(x, x0, bandwidth, kernel)
        expected_output = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.66533454e-16, 7.31250000e-02, 1.42500000e-01, 2.08125000e-01, 2.70000000e-01, 3.28125000e-01, 3.82500000e-01, 4.33125000e-01, 4.80000000e-01, 5.23125000e-01, 5.62500000e-01, 5.98125000e-01, 6.30000000e-01, 6.58125000e-01, 6.82500000e-01, 7.03125000e-01, 7.20000000e-01, 7.33125000e-01, 7.42500000e-01, 7.48125000e-01, 7.50000000e-01, 7.48125000e-01, 7.42500000e-01, 7.33125000e-01, 7.20000000e-01, 7.03125000e-01, 6.82500000e-01, 6.58125000e-01, 6.30000000e-01, 5.98125000e-01, 5.62500000e-01, 5.23125000e-01, 4.80000000e-01, 4.33125000e-01, 3.82500000e-01, 3.28125000e-01, 2.70000000e-01, 2.08125000e-01, 1.42500000e-01, 7.31250000e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_kernel_two_dimensional(self):
        x = np.linspace(0, 1, 11)
        xx, yy = np.meshgrid(x, x)
        x = np.column_stack([xx.flatten(), yy.flatten()])
        x0 = np.array([0.3, 0.1])
        bandwidth = 0.2
        kernel = _epanechnikov

        output = _compute_kernel(x, x0, bandwidth, kernel)
        expected_output = np.array([0.00000000e+00, 0.00000000e+00, 3.75000000e-01, 5.62500000e-01, 3.75000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.66533454e-16, 5.62500000e-01, 7.50000000e-01, 5.62500000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.75000000e-01, 5.62500000e-01, 3.75000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_kernel_three_dimensional(self):
        x = np.linspace(0, 1, 5)
        xx, yy, zz = np.meshgrid(x, x, x)
        x = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        x0 = np.array([0.3, 0.1, 0.5])
        bandwidth = 0.2
        kernel = _epanechnikov

        output = _compute_kernel(x, x0, bandwidth, kernel)
        expected_output = np.array([0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.515625, 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.28125 , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      , 0.      ])
        np.testing.assert_array_almost_equal(output, expected_output)


class TestLocalregression(unittest.TestCase):
    def test_regression_one_dimensional(self):
        x = np.linspace(0, 1, 11)
        y = np.sin(x)
        x0 = np.array([0.5])
        bandwidth = 0.8
        kernel = _epanechnikov

        design_matrix = PolynomialFeatures(degree=2).fit_transform(x.reshape(-1, 1))
        design_matrix_x0 = PolynomialFeatures(degree=2).fit_transform(x0.reshape(-1, 1))

        output = _local_regression(y, x, x0, design_matrix, design_matrix_x0, bandwidth, kernel)
        expected_output = 0.47930025640766993
        np.testing.assert_allclose(output, expected_output)
    
    def test_regression_two_dimensional(self):
        x = np.linspace(0, 1, 11)
        xx, yy = np.meshgrid(x, x)
        x = np.column_stack([xx.flatten(), yy.flatten()])
        y = np.sin(x[:, 0]) * np.cos(x[:, 1])
        x0 = np.array([0.3, 0.1])
        bandwidth = 0.2
        kernel = _epanechnikov

        design_matrix = PolynomialFeatures(degree=2).fit_transform(x)
        design_matrix_x0 = PolynomialFeatures(degree=2).fit_transform(x0.reshape(1, 2))

        output = _local_regression(y, x, x0, design_matrix, design_matrix_x0, bandwidth, kernel)
        expected_output = 0.29404124636834406
        np.testing.assert_allclose(output, expected_output)

    def test_regression_three_dimensional(self):
        x = np.linspace(0, 1, 5)
        xx, yy, zz = np.meshgrid(x, x, x)
        x = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        y = np.sin(x[:, 0]) * np.cos(x[:, 1]) * x[:, 2] #+ np.random.normal(0, 0.05, len(x))
        x0 = np.array([0.3, 0.1, 0.5])
        bandwidth = 0.2
        kernel = _epanechnikov

        design_matrix = PolynomialFeatures(degree=2).fit_transform(x)
        design_matrix_x0 = PolynomialFeatures(degree=2).fit_transform(x0.reshape(1, 3))

        output = _local_regression(y, x, x0, design_matrix, design_matrix_x0, bandwidth, kernel)
        expected_output = 0.12373018800232748
        np.testing.assert_allclose(output, expected_output)


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


class TestLocalPolynomial(unittest.TestCase):
    def test_predict_one_dimensional_data(self):
        n_points = 11
        x = np.linspace(0, 1, n_points)
        y = np.sin(x) + np.random.default_rng(42).normal(0, 0.05, n_points)

        lp = LocalPolynomial(kernel_name='epanechnikov', bandwidth=0.3, degree=1)
        y_pred = lp.predict(y=y, x=x)
        expected_output = np.array([-0.00610104,  0.09962946,  0.19383716,  0.2748045 ,  0.36336718, 0.44248079,  0.52922859,  0.62088602,  0.70524752,  0.7807435 , 0.86891931])
        
        np.testing.assert_array_almost_equal(y_pred, expected_output)

    def test_predict_two_dimensional_data(self):
        n_points = 5
        pts = np.linspace(0, 1, n_points)
        xx, yy = np.meshgrid(pts, pts, indexing='ij')
        x = np.column_stack([xx.flatten(), yy.flatten()])
        x_new = np.array([[0.3, 0.1], [0.5, 0.5]])
        eps = np.random.default_rng(42).normal(0, 0.05, len(x))
        y = np.sin(x[:, 0]) * np.cos(x[:, 1]) + eps

        lp = LocalPolynomial(kernel_name='gaussian', bandwidth=0.8, degree=2)
        y_pred = lp.predict(y=y, x=x, x_new=x_new)
        expected_output = np.array([0.293509, 0.4417991])

        np.testing.assert_array_almost_equal(y_pred, expected_output)

    def test_predict_three_dimensional_data(self):
        n_points = 11
        x = np.linspace(0, 1, n_points)
        xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
        x = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        y = np.sin(x[:, 0]) * np.cos(x[:, 1]) * x[:, 2] + np.random.default_rng(42).normal(0, 0.05, len(x))
        x_new = np.array([[0.3, 0.1, 0.5], [0.3, 0.1, 0.8], [0.3, 0.2, 0.5]])

        lp = LocalPolynomial(kernel_name='epanechnikov', bandwidth=0.3, degree=1)
        y_pred = lp.predict(y=y, x=x, x_new=x_new)
        expected_output = np.array([0.14659093, 0.23125241, 0.13948904])

        np.testing.assert_array_almost_equal(y_pred, expected_output)
