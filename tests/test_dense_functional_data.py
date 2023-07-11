#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the functional_data.py
file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import (
    FunctionalData,
    DenseFunctionalData,
    IrregularFunctionalData
)
from FDApy.representation._argvals import DenseArgvals
from FDApy.representation._values import DenseValues


class TestDenseFunctionalData(unittest.TestCase):
    def setUp(self):
        self.argvals = {'input_dim_0': np.array([1, 2, 3, 4, 5])}
        self.values = np.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10],[11, 12, 13, 14, 15]])
        self.func_data = DenseFunctionalData(self.argvals, self.values)

        argvals = {'input_dim_0': {0: np.array([1, 2, 3, 4]),1: np.array([2, 4])}}
        values = {0: np.array([1, 6, 9, 4]),1: np.array([2, 3])}
        self.irreg_data = IrregularFunctionalData(argvals, values)

    def test_getitem_dense_functional_data(self):
        data = self.func_data[1]
        expected_argvals = DenseArgvals(self.argvals)
        expected_values = DenseValues(np.array([[6, 7, 8, 9, 10]]))
        np.testing.assert_array_equal(data.argvals, expected_argvals)
        np.testing.assert_array_equal(data.values, expected_values)

    def test_argvals_property(self):
        argvals = self.func_data.argvals
        self.assertEqual(argvals, DenseArgvals(self.argvals))

    def test_argvals_setter(self):
        new_argvals = {'x': np.linspace(0, 5, 5)}
        self.func_data.argvals = new_argvals
        self.assertEqual(self.func_data._argvals, DenseArgvals(new_argvals))

        expected_argvals_stand = {"x": np.linspace(0, 1, 5),}
        np.testing.assert_array_almost_equal(self.func_data._argvals_stand['x'], expected_argvals_stand['x'])

    def test_values_property(self):
        dense_values = self.func_data.values
        np.testing.assert_array_equal(dense_values, self.values)

    def test_values_setter(self):
        new_values = np.array([[11, 12, 13, 14, 15]])
        self.func_data.values = new_values
        np.testing.assert_array_equal(self.func_data.values, new_values)

    def test_range_obs(self):
        expected_result = (1, 15)
        result = self.func_data.range_obs
        self.assertEqual(result, expected_result)

    def test_n_points(self):
        expected_result = {"input_dim_0": 5}
        result = self.func_data.n_points
        self.assertDictEqual(result, expected_result)

    def test_range_dim(self):
        expected_range = {"input_dim_0": (1, 5)}
        result = self.func_data.range_dim
        self.assertDictEqual(result, expected_range)

    def test_shape(self):
        expected_output = {'input_dim_0': 5}
        result = self.func_data.shape
        self.assertDictEqual(result, expected_output)

    def test_is_compatible(self):
        DenseFunctionalData._is_compatible(self.func_data, self.func_data)
        self.assertTrue(True)

    def test_non_compatible_type(self):
        with self.assertRaises(TypeError):
            DenseFunctionalData._is_compatible(self.func_data, self.irreg_data)

    def test_non_compatible_nobs(self):
        argvals = {'input_dim_0': np.array([1, 2, 3, 4, 5])}
        values = np.array([[1, 2, 3, 4, 5]])
        func_data = DenseFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            DenseFunctionalData._is_compatible(self.func_data, func_data)

    def test_non_compatible_ndim(self):
        argvals = {'input_dim_0': np.array([1, 2, 3, 4]), 'input_dim_1': np.array([5, 6, 7])}
        values = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],[[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]],[[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]])
        func_data = DenseFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            DenseFunctionalData._is_compatible(self.func_data, func_data)

    def test_non_compatible_argvals_equality(self):
        argvals = {'input_dim_0': np.array([1, 2, 3, 4, 6])}
        values = np.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10],[11, 12, 13, 14, 15]])
        func_data = DenseFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            DenseFunctionalData._is_compatible(self.func_data, func_data)

    def test_inner_product(self):
        result = self.func_data.inner_product()
        expected = np.array([[42., 102., 162.],[102., 262., 422.],[162., 422., 682.]])
        np.testing.assert_array_almost_equal(result, expected)


class TestPerformComputation(unittest.TestCase):
    def setUp(self):
        self.argvals1 = {'input_dim_0': np.array([1, 2, 3, 4, 5])}
        self.values1 = np.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10],[11, 12, 13, 14, 15]])
        self.func_data1 = DenseFunctionalData(self.argvals1, self.values1)

        self.argvals2 = {'input_dim_0': np.array([1, 2, 3, 4, 5])}
        self.values2 = np.array([[6, 7, 8, 9, 10],[11, 12, 13, 14, 15],[1, 2, 3, 4, 5]])
        self.func_data2 = DenseFunctionalData(self.argvals2, self.values2)

    def test_addition(self):
        result = self.func_data1 + self.func_data2

        expected_values = np.array([[7, 9, 11, 13, 15],[17, 19, 21, 23, 25],[12, 14, 16, 18, 20]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_substraction(self):
        result = self.func_data1 - self.func_data2

        expected_values = np.array([[-5, -5, -5, -5, -5],[-5, -5, -5, -5, -5],[10, 10, 10, 10, 10]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_multiplication(self):
        result = self.func_data1 * self.func_data2

        expected_values = np.array([[6, 14, 24, 36, 50],[66, 84, 104, 126, 150],[11, 24, 39, 56, 75]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_right_multiplication(self):
        result = FunctionalData.__rmul__(self.func_data1, self.func_data2)

        expected_values = np.array([[6, 14, 24, 36, 50],[66, 84, 104, 126, 150],[11, 24, 39, 56, 75]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_true_divide(self):
        result = self.func_data1 / self.func_data2

        expected_values = np.array([[0.16666667, 0.28571429, 0.375, 0.44444444, 0.5],[0.54545455, 0.58333333, 0.61538462, 0.64285714, 0.66666667],[11., 6., 4.33333333, 3.5, 3.]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_almost_equal(result.values, expected_values)

    def test_floor_divide(self):
        result = self.func_data1 // self.func_data2

        expected_values = np.array([ [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [11, 6, 4, 3, 3]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_almost_equal(result.values, expected_values)


class TestDenseFunctionalData2D(unittest.TestCase):
    """Test class for the class DenseFunctionalData in two dimension."""

    def setUp(self):
        argvals = {'input_dim_0': np.array([1, 2, 3, 4]), 'input_dim_1': np.array([5, 6, 7])}

        values = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]], [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]], [[3, 4, 6], [3, 4, 5], [3, 4, 5], [3, 4, 5]], [[3, 4, 7], [3, 4, 5], [3, 4, 5], [3, 4, 5]]])
        self.dense_fd = DenseFunctionalData(argvals, values)

    def test_argvals_stand(self):
        is_equal_dim0 = np.allclose(self.dense_fd.argvals_stand['input_dim_0'], np.array([0., 0.33333333, 0.66666667, 1.]))
        is_equal_dim1 = np.allclose(self.dense_fd.argvals_stand['input_dim_1'], np.array([0., 0.5, 1.]))
        self.assertTrue(is_equal_dim0 and is_equal_dim1)

    def test_n_obs(self):
        self.assertEqual(self.dense_fd.n_obs, 5)

    def test_n_dim(self):
        self.assertEqual(self.dense_fd.n_dim, 2)

    def test_range_obs(self):
        self.assertEqual(self.dense_fd.range_obs, (1, 7))

    def test_range_dim(self):
        self.assertEqual(self.dense_fd.range_dim, {'input_dim_0': (1, 4), 'input_dim_1': (5, 7)})

    def test_shape(self):
        self.assertEqual(self.dense_fd.shape, {'input_dim_0': 4, 'input_dim_1': 3})

    def test_subset(self):
        new_dense_fd = self.dense_fd[2]
        self.assertIsInstance(new_dense_fd, DenseFunctionalData)
        self.assertEqual(new_dense_fd.n_obs, 1)
        new_dense_fd = self.dense_fd[1:4]
        self.assertIsInstance(new_dense_fd, DenseFunctionalData)
        self.assertEqual(new_dense_fd.n_obs, 3)

    def test_is_compatible(self):
        DenseFunctionalData._is_compatible(self.dense_fd, self.dense_fd)
        self.assertTrue(True)

    def test_mean(self):
        mean_fd = self.dense_fd.mean()
        is_equal = np.allclose(mean_fd.values, np.array([[[3., 4., 5.6], [3., 4., 5.], [3., 4., 5.], [3., 4., 5.]]]))
        self.assertTrue(is_equal)
