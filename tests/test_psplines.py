#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class PSplines in the
psplines.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.preprocessing.smoothing.psplines import (
    _row_tensor,
    _h_transform,
    _rotate,
    _rotated_h_transform,
    _create_permutation,
    _tensor_product_penalties,
    _fit_n_dimensional,
    _fit_one_dimensional,
    PSplines,
)


class TestRowTensor(unittest.TestCase):
    def test_row_tensor_with_y(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6, 7], [7, 8, 9]])
        expected_result = np.array(
            [[5.0, 6.0, 7.0, 10.0, 12.0, 14.0], [21.0, 24.0, 27.0, 28.0, 32.0, 36.0]]
        )
        result = _row_tensor(x, y)
        np.testing.assert_array_equal(result, expected_result)

    def test_row_tensor_without_y(self):
        x = np.array([[1, 2], [3, 4]])
        expected_result = np.array([[1.0, 2.0, 2.0, 4.0], [9.0, 12.0, 12.0, 16.0]])
        result = _row_tensor(x)
        np.testing.assert_array_equal(result, expected_result)

    def test_row_tensor_with_mismatched_shapes(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 2]])
        with self.assertRaises(ValueError):
            _row_tensor(x, y)


class TestHTransform(unittest.TestCase):
    def test_h_transform(self):
        x = np.array([[1, 2, 3]])
        y = np.array([[1, 2], [3, 4], [5, 6]])
        expected_result = np.array([[22, 28]])
        result = _h_transform(x, y)
        np.testing.assert_array_equal(result, expected_result)

    def test_h_transform_with_invalid_dimensions(self):
        x = np.array([[1, 2, 3]])
        y = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            _h_transform(x, y)


class TestRotate(unittest.TestCase):
    def test_rotate(self):
        x = np.array([[[1, 2], [3, 4], [5, 6]], [[5, 6], [7, 8], [9, 0]]])
        expected_result = np.array(
            [[[1, 5], [2, 6]], [[3, 7], [4, 8]], [[5, 9], [6, 0]]]
        )
        result = _rotate(x)
        np.testing.assert_array_equal(result, expected_result)


class TestRotatedHTransform(unittest.TestCase):
    def test_rotated_h_transform(self):
        x = np.array([[1, 2, 3]])
        y = np.array([[1, 2], [3, 4], [5, 6]])
        expected_result = np.array([[22], [28]])
        result = _rotated_h_transform(x, y)
        np.testing.assert_array_equal(result, expected_result)


class TestCreatePermutation(unittest.TestCase):
    def test_create_permutation(self):
        result = _create_permutation(3, 2)
        expected_result = np.array([0, 3, 1, 4, 2, 5])
        np.testing.assert_array_equal(result, expected_result)

        result = _create_permutation(2, 3)
        expected_result = np.array([0, 2, 4, 1, 3, 5])
        np.testing.assert_array_equal(result, expected_result)


class TestTensorProductPenalties(unittest.TestCase):
    def test_tensor_product_penalties_one(self):
        penalties = [np.array([[1.0, -1.0], [-1.0, 2.0]])]
        result = _tensor_product_penalties(penalties)
        expected_result = np.array([[1.0, -1.0], [-1.0, 2.0]])
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_tensor_product_penalties(self):
        penalties = [
            np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]),
            np.array([[1.0, -1.0], [-1.0, 2.0]]),
        ]
        result = _tensor_product_penalties(penalties)

        expected_result = [
            np.array(
                [
                    [1.0, 0.0, -1.0, -0.0, 0.0, 0.0],
                    [0.0, 1.0, -0.0, -1.0, 0.0, 0.0],
                    [-1.0, -0.0, 2.0, 0.0, -1.0, -0.0],
                    [-0.0, -1.0, 0.0, 2.0, -0.0, -1.0],
                    [0.0, 0.0, -1.0, -0.0, 2.0, 0.0],
                    [0.0, 0.0, -0.0, -1.0, 0.0, 2.0],
                ]
            ),
            np.array(
                [
                    [1.0, -1.0, 0.0, -0.0, 0.0, -0.0],
                    [-1.0, 2.0, -0.0, 0.0, -0.0, 0.0],
                    [0.0, -0.0, 1.0, -1.0, 0.0, -0.0],
                    [-0.0, 0.0, -1.0, 2.0, -0.0, 0.0],
                    [0.0, -0.0, 0.0, -0.0, 1.0, -1.0],
                    [-0.0, 0.0, -0.0, 0.0, -1.0, 2.0],
                ]
            ),
        ]

        for res, exp in zip(result, expected_result):
            np.testing.assert_array_almost_equal(res, exp)


class TestFitOneDimensional(unittest.TestCase):
    def test_fit_one_dimensional(self):
        data = np.array([1, 2, 3, 4, 5])
        basis = np.array([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [1, 4, 9, 16, 25]])

        result = _fit_one_dimensional(data, basis)

        expected_y_hat = np.array(
            [1.25143869, 1.95484728, 2.83532537, 3.89287295, 5.12749004]
        )
        expected_beta_hat = np.array([0.7250996, 0.43780434, 0.08853475])
        expected_hat_matrix = np.array(
            [0.3756529, 0.3549800, 0.2669322, 0.2788401, 0.7545816]
        )

        np.testing.assert_allclose(result["y_hat"], expected_y_hat, atol=1e-6)
        np.testing.assert_allclose(result["beta_hat"], expected_beta_hat, atol=1e-6)
        np.testing.assert_allclose(result["hat_matrix"], expected_hat_matrix, atol=1e-6)


class TestFitNDimensional(unittest.TestCase):
    def test_fit_n_dimensional(self):
        data = np.array([[1, 2], [3, 4]])
        basis_list = [np.array([[1, 1], [1, 2]]), np.array([[1, 1], [2, 3]])]

        result = _fit_n_dimensional(data, basis_list)

        expected_y_hat = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected_beta_hat = np.array([[-3.0, 1.0], [2.0, 0.0]])
        expected_hat_matrix = np.array([[1.0, 1.0], [1.0, 1.0]])

        np.testing.assert_allclose(result["y_hat"], expected_y_hat, atol=1e-10)
        np.testing.assert_allclose(result["beta_hat"], expected_beta_hat, atol=1e-10)
        np.testing.assert_allclose(
            result["hat_matrix"], expected_hat_matrix, atol=1e-10
        )


class TestPSplines(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3, 4, 5])
        self.y = np.array([1, 2, 3, 4, 5])

    def test_getter(self):
        ps = PSplines(order_penalty=2, order_derivative=2)

        ps.order_penalty = 3
        np.testing.assert_equal(ps.order_penalty, 3)

        ps.order_derivative = 3
        np.testing.assert_equal(ps.order_derivative, 3)

    def test_getter_error(self):
        ps = PSplines(order_penalty=2, order_derivative=2)

        with self.assertRaises(ValueError):
            ps.order_derivative = -1
        with self.assertRaises(ValueError):
            ps.order_penalty = -1

    def test_fit(self):
        ps = PSplines(n_segments=3, degree=2)
        ps.fit(self.y, self.x)

        exp_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        exp_beta = np.array([0.33333333, 1.66666667, 3.0, 4.33333333, 5.66666667])
        exp_hat_matrix = np.array(
            [0.68604385, 0.334076, 0.30348758, 0.334076, 0.68604385]
        )

        np.testing.assert_allclose(ps.y_hat, exp_y)
        np.testing.assert_allclose(ps.beta_hat, exp_beta)
        np.testing.assert_allclose(ps.diagnostics["hat_matrix"], exp_hat_matrix)

    def test_predict(self):
        ps = PSplines(n_segments=3, degree=2)
        ps.fit(self.y, self.x)
        y_pred = ps.predict(self.x)
        np.testing.assert_allclose(y_pred, self.y)

        y_pred2 = ps.predict()
        np.testing.assert_allclose(y_pred2, self.y)

    def test_fit_and_predict_with_weights(self):
        ps = PSplines(n_segments=3, degree=2)
        sample_weights = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        ps.fit(self.y, self.x, sample_weights=sample_weights)
        y_pred = ps.predict(self.x)

        np.testing.assert_allclose(y_pred, self.y)

    def test_fit_and_predict_with_penalty(self):
        ps = PSplines(n_segments=3, degree=2)
        ps.fit(self.y, self.x, penalty=(0.5,))
        y_pred = ps.predict(self.x)

        np.testing.assert_allclose(y_pred, self.y)

    def test_fit_with_multiple_dimensions(self):
        x = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3])]
        y = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 1]])

        ps = PSplines(n_segments=(3, 3), degree=(2, 2))
        ps.fit(y, x)

        exp_y = np.array(
            [
                [0.7231239, 2.45512079, 4.29236073],
                [2.96571433, 4.105447, 5.13557615],
                [5.22050526, 5.67014021, 5.70659025],
                [7.50950298, 7.04259125, 5.75080022],
                [9.83917741, 8.210653, 5.37269653],
            ]
        )
        exp_beta = np.array(
            [
                [-1.458321, -0.0818564, 1.32241776, 2.84544532, 4.48771706],
                [1.75937483, 2.67329817, 3.57286394, 4.47335536, 5.36292519],
                [5.01901586, 5.40621621, 5.74062147, 5.86693485, 5.79368057],
                [8.44412798, 8.09989072, 7.57601514, 6.44787524, 5.05759567],
                [12.06151528, 10.75117564, 9.03080549, 6.43058263, 3.55473257],
            ]
        )
        exp_hat_matrix = np.array(
            [
                [0.56273035, 0.30308193, 0.56273035],
                [0.27986477, 0.16897093, 0.27986477],
                [0.23096001, 0.15414815, 0.23096001],
                [0.27986477, 0.16897093, 0.27986477],
                [0.56273035, 0.30308193, 0.56273035],
            ]
        )

        np.testing.assert_allclose(ps.y_hat, exp_y)
        np.testing.assert_allclose(ps.beta_hat, exp_beta)
        np.testing.assert_allclose(ps.diagnostics["hat_matrix"], exp_hat_matrix)

    def test_predict_with_multiple_dimensions(self):
        x = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3])]
        y = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 1]])

        ps = PSplines(n_segments=(3, 3), degree=(2, 2))
        ps.fit(y, x)
        y_pred = ps.predict(x)

        exp_y = np.array(
            [
                [0.7231239, 2.45512079, 4.29236073],
                [2.96571433, 4.105447, 5.13557615],
                [5.22050526, 5.67014021, 5.70659025],
                [7.50950298, 7.04259125, 5.75080022],
                [9.83917741, 8.210653, 5.37269653],
            ]
        )

        np.testing.assert_allclose(y_pred, exp_y)
