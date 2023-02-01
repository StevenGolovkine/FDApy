#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the utils.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from numpy import AxisError

from FDApy.misc.utils import (
    _col_mean,
    _col_var,
    _get_axis_dimension,
    _get_dict_dimension,
    _get_obs_shape,
    _inner_product,
    _inner_product_2d,
    _integrate,
    _integration_weights,
    _outer,
    _normalization,
    _row_mean,
    _row_var,
    _shift,
    _standardization
)


class TestNormalization(unittest.TestCase):
    def test_normalization_default(self):
        x = np.array([0, 5, 10])
        expected = np.array([0., 0.5, 1.])
        result = _normalization(x)
        self.assertTrue(np.allclose(result, expected))

    def test_normalization_with_min_max(self):
        x = np.array([0, 5, 10])
        max_x = 10
        min_x = 0
        expected = np.array([0., 0.5, 1.])
        result = _normalization(x, max_x=max_x, min_x=min_x)
        self.assertTrue(np.allclose(result, expected))

    def test_normalization_with_nan_min_max(self):
        x = np.array([0, 5, 10])
        max_x = np.nan
        min_x = np.nan
        expected = np.array([0., 0.5, 1.])
        result = _normalization(x, max_x=max_x, min_x=min_x)
        self.assertTrue(np.allclose(result, expected))

    def test_normalization_with_zero_range(self):
        x = np.array([5, 5, 5])
        expected = np.array([0., 0., 0.])
        result = _normalization(x)
        self.assertTrue(np.allclose(result, expected))


class TestStandardization(unittest.TestCase):
    def test_standardization(self):
        x = np.array([0, 5, 10])
        expected = np.array([-1.22474487, 0., 1.22474487])
        result = _standardization(x)
        self.assertTrue(np.allclose(result, expected))

    def test_standardization_with_zero_mean_std(self):
        x = np.array([5, 5, 5])
        expected = np.array([0., 0., 0.])
        result = _standardization(x)
        self.assertTrue(np.allclose(result, expected))

    def test_standardization_with_negative_value(self):
        x = np.array([-5, 5, 10])
        expected = np.array([-1.33630621, 0.26726124, 1.06904497])
        result = _standardization(x)
        self.assertTrue(np.allclose(result, expected))


class TestRowMean(unittest.TestCase):
    def test_row_mean(self):
        x = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        expected = np.array([1., 2., 3.])
        result = _row_mean(x)
        self.assertTrue(np.allclose(result, expected))

    def test_row_mean_with_negative_values(self):
        x = np.array(
            [
                [-1., 2., 3.],
                [1., -2., 3.],
                [1., 2., -3.]
            ]
        )
        expected = np.array([0.33333333, 0.66666667, 1.])
        result = _row_mean(x)
        self.assertTrue(np.allclose(result, expected))

    def test_row_mean_with_different_shapes(self):
        x = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        y = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        expected_x = np.mean(x, axis=0)
        expected_y = np.mean(y, axis=0)
        result_x = _row_mean(x)
        result_y = _row_mean(y)
        self.assertTrue(np.allclose(result_x, expected_x))
        self.assertTrue(np.allclose(result_y, expected_y))


class TestRowVar(unittest.TestCase):
    def test_row_var(self):
        x = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        expected = np.array([0., 0., 0.])
        result = _row_var(x)
        self.assertTrue(np.array_equal(expected, result))

    def test_row_var_unequal_values(self):
        x = np.array(
            [
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [10., 11., 12.]
            ]
        )
        expected = np.array([11.25, 11.25, 11.25])
        result = _row_var(x)
        self.assertTrue(np.array_equal(expected, result))

    def test_row_var_input_is_1d_array(self):
        x = np.array([1, 2, 3])
        expected = 0.6666666666666666
        result = _row_var(x)
        self.assertTrue(np.array_equal(expected, result))


class TestColMean(unittest.TestCase):
    def test_col_mean(self):
        x = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        expected = np.array([2., 2., 2., 2.])
        result = _col_mean(x)
        self.assertTrue(np.array_equal(expected, result))

    def test_col_mean_unequal_values(self):
        x = np.array(
            [
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [10., 11., 12.]
            ]
        )
        expected = np.array([2., 5., 8., 11.])
        result = _col_mean(x)
        self.assertTrue(np.array_equal(expected, result))

    def test_col_mean_input_is_1d_array(self):
        x = np.array([1, 2, 3])
        self.assertRaises(AxisError, _col_mean, x)


class TestColVar(unittest.TestCase):
    def test_col_var(self):
        x = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        expected = np.array([0.66666667, 0.66666667, 0.66666667, 0.66666667])
        result = _col_var(x)
        self.assertTrue(np.allclose(expected, result, rtol=1e-6))

    def test_col_var_unequal_values(self):
        x = np.array(
            [
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [10., 11., 12.]
            ]
        )
        expected = np.array([0.66666667, 0.66666667, 0.66666667, 0.66666667])
        result = _col_var(x)
        self.assertTrue(np.allclose(expected, result, rtol=1e-6))

    def test_col_var_input_is_1d_array(self):
        x = np.array([1, 2, 3])
        self.assertRaises(AxisError, _col_var, x)


class TestGetAxisDimension(unittest.TestCase):
    def test_default_axis(self):
        x = np.array([[1, 2], [4, 5], [7, 8]])
        result = _get_axis_dimension(x)
        self.assertEqual(result, 3)

    def test_specified_axis(self):
        x = np.array([[1, 2], [4, 5], [7, 8]])
        result = _get_axis_dimension(x, axis=1)
        self.assertEqual(result, 2)

    def test_empty_array(self):
        x = np.array([])
        result = _get_axis_dimension(x)
        self.assertEqual(result, 0)


class TestGetDictDimension(unittest.TestCase):
    def test_get_dict_dimension(self):
        x = {
            'a': np.array([1, 2, 3]),
            'b': np.array([4, 5])
        }
        self.assertEqual(_get_dict_dimension(x), (3, 2))

    def test_get_dict_dimension_unequal_size(self):
        x = {
            'a': np.array([1, 2, 3]),
            'b': np.array([[4, 5], [6, 7]])
        }
        self.assertEqual(_get_dict_dimension(x), (3, 2))

    def test_get_dict_dimension_same_size(self):
        x = {
            'a': np.array([1, 2, 3]),
            'b': np.array([[4, 5], [6, 7], [8, 9]])
        }
        self.assertEqual(_get_dict_dimension(x), (3, 3))

    def test_get_dict_dimension_multiple_obs(self):
        x = {
            'a': np.array([1, 2, 3]),
            'b': np.array([4, 5]),
            'c': np.array([[1, 2, 3], [4, 5, 6]])
        }
        self.assertEqual(_get_dict_dimension(x), (3, 2, 2))


class TestGetObsShape(unittest.TestCase):
    def test_get_obs_shape(self):
        x = {
            'a': {0: np.array([1, 2, 3]), 1: np.array([4, 5])},
            'b': {0: np.array([1, 2]), 1: np.array([3, 4])}
        }
        obs_0 = _get_obs_shape(x, 0)
        obs_1 = _get_obs_shape(x, 1)

        self.assertEqual(obs_0, (3, 2))
        self.assertEqual(obs_1, (2, 2))


class TestShift(unittest.TestCase):
    def test_shift_positive_num(self):
        x = np.array([1, 2, 3, 4, 5])
        num = 2
        fill_value = np.nan
        expected = np.array([np.nan, np.nan, 1., 2., 3.])
        result = _shift(x, num, fill_value)
        self.assertTrue(np.array_equal(result, expected, equal_nan=True))

    def test_shift_negative_num(self):
        x = np.array([1, 2, 3, 4, 5])
        num = -2
        fill_value = np.nan
        expected = np.array([3., 4., 5., np.nan, np.nan])
        result = _shift(x, num, fill_value)
        self.assertTrue(np.array_equal(result, expected, equal_nan=True))

    def test_shift_zero_num(self):
        x = np.array([1, 2, 3, 4, 5])
        num = 0
        fill_value = np.nan
        expected = np.array([1, 2, 3, 4, 5])
        result = _shift(x, num, fill_value)
        self.assertTrue(np.array_equal(result, expected))


class TestInnerProduct(unittest.TestCase):
    def test_inner_product(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        t = np.linspace(0, 1, 3)
        expected_output = 10.5
        self.assertAlmostEqual(expected_output, _inner_product(x, y, t))

    def test_inner_product_unequal_shape(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6, 7])
        self.assertRaises(ValueError, _inner_product, x, y)

    def test_inner_product_no_t(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        expected_output = 10.5
        self.assertAlmostEqual(expected_output, _inner_product(x, y))


class TestInnerProduct2D(unittest.TestCase):
    def test_inner_product_2d(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        y = np.array([[4, 5, 6], [1, 2, 3], [4, 5, 6]])
        primary_axis = np.linspace(0, 1, x.shape[0])
        secondary_axis = np.linspace(0, 1, x.shape[1])
        expected_result = 10.5
        result = _inner_product_2d(x, y, primary_axis, secondary_axis)
        self.assertAlmostEqual(result, expected_result, delta=1e-5)

    def test_inner_product_2d_unequal_shapes(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        y = np.array([[4, 5, 6], [1, 2, 3]])
        with self.assertRaises(ValueError):
            _inner_product_2d(x, y)

    def test_inner_product_2d_no_t(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        y = np.array([[4, 5, 6], [1, 2, 3], [4, 5, 6]])
        expected_output = 10.5
        self.assertAlmostEqual(expected_output, _inner_product_2d(x, y))


class TestOuterFunction(unittest.TestCase):
    def test_outer(self):
        x = np.array([1, 2, 3])
        y = np.array([-1, 2])
        expected_output = np.array([[-1, 2], [-2, 4], [-3, 6]])
        np.testing.assert_array_equal(_outer(x, y), expected_output)

    def test_outer_with_negative_input(self):
        x = np.array([-1, -2, -3])
        y = np.array([-1, -2])
        expected_output = np.array([[1, 2], [2, 4], [3, 6]])
        np.testing.assert_array_equal(_outer(x, y), expected_output)

    def test_outer_with_zero_input(self):
        x = np.array([0, 0, 0])
        y = np.array([0, 0])
        expected_output = np.array([[0, 0], [0, 0], [0, 0]])
        np.testing.assert_array_equal(_outer(x, y), expected_output)


class TestIntegrate(unittest.TestCase):
    def test_integrate_simpson(self):
        X = np.array([1, 2, 4])
        Y = np.array([1, 4, 16])
        result = _integrate(X, Y)
        self.assertEqual(result, 21.0)

    def test_integrate_trapeze(self):
        X = np.array([1, 2, 4])
        Y = np.array([1, 4, 16])
        result = _integrate(X, Y, method='trapz')
        self.assertEqual(result, 22.5)

    def test_integrate_method_error(self):
        X = np.array([1, 2, 4])
        Y = np.array([1, 4, 16])
        with self.assertRaises(ValueError):
            _integrate(X, Y, method='error')


class TestIntegrationWeights(unittest.TestCase):

    def test_trapz(self):
        x = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([0.5, 1., 1., 1., 0.5])
        np.testing.assert_array_equal(
            _integration_weights(x, method='trapz'),
            expected_output
        )

    def test_simpson(self):
        x = np.array([1, 2, 3, 4, 5])
        expected_output = np.array(
            [0.33333333, 1.33333333, 0.66666667, 1.33333333, 0.33333333]
        )
        np.testing.assert_array_almost_equal(
            _integration_weights(x, method='simpson'),
            expected_output
        )

    def test_callable_method(self):
        def custom_weights(x: np.ndarray) -> np.ndarray:
            return x

        x = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(
            _integration_weights(x, method=custom_weights),
            expected_output
        )

    def test_invalid_method(self):
        x = np.array([1, 2, 3, 4, 5])
        with self.assertRaises(NotImplementedError) as cm:
            _integration_weights(x, method='invalid_method')
        self.assertEqual(
            str(cm.exception), 'invalid_method not implemented!',
            "NotImplementedError should be raised for invalid method"
        )
