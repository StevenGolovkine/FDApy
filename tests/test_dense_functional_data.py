#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the functional_data.py
file.

Written with the help of ChatGPT.

"""
import numpy as np
import pandas as pd
import pickle
import unittest

from pathlib import Path

from FDApy.representation.functional_data import (
    GridFunctionalData,
    DenseFunctionalData,
    IrregularFunctionalData,
)
from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
from FDApy.representation.values import DenseValues, IrregularValues

THIS_DIR = Path(__file__)


class TestDenseFunctionalData(unittest.TestCase):
    def setUp(self):
        self.argvals = DenseArgvals({"input_dim_0": np.array([1, 2, 3, 4, 5])})
        self.values = DenseValues(
            np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        )
        self.func_data = DenseFunctionalData(self.argvals, self.values)

        argvals = IrregularArgvals(
            {
                0: DenseArgvals({"input_dim_0": np.array([1, 2, 3, 4])}),
                1: DenseArgvals({"input_dim_0": np.array([2, 4])}),
            }
        )
        values = IrregularValues({0: np.array([1, 6, 9, 4]), 1: np.array([2, 3])})
        self.irreg_data = IrregularFunctionalData(argvals, values)

    def test_repr(self):
        expected_repr = (
            "Functional data object with 3 observations on a 1-dimensional support."
        )
        actual_repr = repr(self.func_data)
        self.assertEqual(actual_repr, expected_repr)

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
        new_argvals = DenseArgvals({"x": np.linspace(0, 5, 5)})
        self.func_data.argvals = new_argvals
        self.assertEqual(self.func_data._argvals, DenseArgvals(new_argvals))

        expected_argvals_stand = {"x": np.linspace(0, 1, 5)}
        np.testing.assert_array_almost_equal(
            self.func_data._argvals_stand["x"], expected_argvals_stand["x"]
        )

        with self.assertRaises(TypeError):
            self.func_data.argvals = 0

    def test_argvals_stand_setter(self):
        new_argvals = DenseArgvals({"x": np.linspace(0, 1, 5)})
        self.func_data.argvals_stand = new_argvals
        self.assertEqual(self.func_data._argvals_stand, new_argvals)

        with self.assertRaises(TypeError):
            self.func_data.argvals_stand = 0

    def test_values_property(self):
        dense_values = self.func_data.values
        np.testing.assert_array_equal(dense_values, self.values)

    def test_values_setter(self):
        new_values = DenseValues(np.array([[11, 12, 13, 14, 15]]))
        self.func_data.values = new_values
        np.testing.assert_array_equal(self.func_data.values, new_values)

        with self.assertRaises(TypeError):
            self.func_data.values = 0

    def test_n_points(self):
        expected_result = (5,)
        result = self.func_data.n_points
        np.testing.assert_equal(result, expected_result)

    def test_is_compatible(self):
        DenseFunctionalData._is_compatible(self.func_data, self.func_data)
        self.assertTrue(True)

    def test_non_compatible_type(self):
        with self.assertRaises(TypeError):
            DenseFunctionalData._is_compatible(self.func_data, self.irreg_data)

    def test_non_compatible_nobs(self):
        argvals = DenseArgvals({"input_dim_0": np.array([1, 2, 3, 4, 5])})
        values = DenseValues(np.array([[1, 2, 3, 4, 5]]))
        func_data = DenseFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            DenseFunctionalData._is_compatible(self.func_data, func_data)

    def test_non_compatible_ndim(self):
        argvals = DenseArgvals(
            {"input_dim_0": np.array([1, 2, 3, 4]), "input_dim_1": np.array([5, 6, 7])}
        )
        values = DenseValues(
            np.array(
                [
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]],
                    [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]],
                ]
            )
        )
        func_data = DenseFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            DenseFunctionalData._is_compatible(self.func_data, func_data)

    def test_non_compatible_argvals_equality(self):
        argvals = DenseArgvals({"input_dim_0": np.array([1, 2, 3, 4, 6])})
        values = DenseValues(
            np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        )
        func_data = DenseFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            DenseFunctionalData._is_compatible(self.func_data, func_data)

    def test_concatenate(self):
        fdata = DenseFunctionalData.concatenate(self.func_data, self.func_data)

        expected_argvals = DenseArgvals({"input_dim_0": np.array([1, 2, 3, 4, 5])})
        expected_values = DenseValues(
            np.array(
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                ]
            )
        )

        self.assertIsInstance(fdata, DenseFunctionalData)
        self.assertEqual(fdata.argvals, expected_argvals)
        np.testing.assert_allclose(fdata.values, expected_values)

    def test_to_long(self):
        result = self.func_data.to_long()

        expected_dim = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        expected_id = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        expected_values = DenseValues(
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        )

        self.assertTrue(isinstance(result, pd.DataFrame))
        np.testing.assert_array_equal(result["input_dim_0"].values, expected_dim)
        np.testing.assert_array_equal(result["id"].values, expected_id)
        np.testing.assert_array_equal(result["values"].values, expected_values)


class TestPerformComputation(unittest.TestCase):
    def setUp(self):
        self.argvals1 = DenseArgvals({"input_dim_0": np.array([1, 2, 3, 4, 5])})
        self.values1 = DenseValues(
            np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        )
        self.func_data1 = DenseFunctionalData(self.argvals1, self.values1)

        self.argvals2 = DenseArgvals({"input_dim_0": np.array([1, 2, 3, 4, 5])})
        self.values2 = DenseValues(
            np.array([[6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [1, 2, 3, 4, 5]])
        )
        self.func_data2 = DenseFunctionalData(self.argvals2, self.values2)

    def test_equality(self):
        self.assertTrue(self.func_data1 == self.func_data1)
        self.assertFalse(self.func_data1 == self.func_data2)

    def test_addition(self):
        result = self.func_data1 + self.func_data2

        expected_values = np.array(
            [[7, 9, 11, 13, 15], [17, 19, 21, 23, 25], [12, 14, 16, 18, 20]]
        )
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_addition_number(self):
        result = self.func_data1 + 2

        expected_values = np.array(
            [[3, 4, 5, 6, 7], [8, 9, 10, 11, 12], [13, 14, 15, 16, 17]]
        )
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_addition_error(self):
        with self.assertRaises(TypeError):
            self.func_data1 + [1, 2, 3]

    def test_substraction(self):
        result = self.func_data1 - self.func_data2

        expected_values = np.array(
            [[-5, -5, -5, -5, -5], [-5, -5, -5, -5, -5], [10, 10, 10, 10, 10]]
        )
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_substraction_number(self):
        result = self.func_data1 - 2

        expected_values = np.array(
            [[-1, 0, 1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12, 13]]
        )
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_substraction_error(self):
        with self.assertRaises(TypeError):
            self.func_data1 - [1, 2, 3]

    def test_multiplication(self):
        result = self.func_data1 * self.func_data2

        expected_values = np.array(
            [[6, 14, 24, 36, 50], [66, 84, 104, 126, 150], [11, 24, 39, 56, 75]]
        )
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_multiplication_number(self):
        result = self.func_data1 * 2

        expected_values = np.array(
            [[2, 4, 6, 8, 10], [12, 14, 16, 18, 20], [22, 24, 26, 28, 30]]
        )
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_multiplication_error(self):
        with self.assertRaises(TypeError):
            self.func_data1 * [1, 2, 3]

    def test_right_multiplication(self):
        result = GridFunctionalData.__rmul__(self.func_data1, self.func_data2)

        expected_values = np.array(
            [[6, 14, 24, 36, 50], [66, 84, 104, 126, 150], [11, 24, 39, 56, 75]]
        )
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_true_divide(self):
        result = self.func_data1 / self.func_data2

        expected_values = np.array(
            [
                [0.16666667, 0.28571429, 0.375, 0.44444444, 0.5],
                [0.54545455, 0.58333333, 0.61538462, 0.64285714, 0.66666667],
                [11.0, 6.0, 4.33333333, 3.5, 3.0],
            ]
        )
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_almost_equal(result.values, expected_values)

    def test_true_divide_number(self):
        result = self.func_data1 / 2

        expected_values = np.array(
            [[0.5, 1, 1.5, 2, 2.5], [3, 3.5, 4, 4.5, 5], [5.5, 6, 6.5, 7, 7.5]]
        )
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_true_divide_error(self):
        with self.assertRaises(TypeError):
            self.func_data1 / [1, 2, 3]

    def test_floor_divide(self):
        result = self.func_data1 // self.func_data2

        expected_values = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [11, 6, 4, 3, 3]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_almost_equal(result.values, expected_values)

    def test_floor_divide_number(self):
        result = self.func_data1 // 2

        expected_values = np.array([[0, 1, 1, 2, 2], [3, 3, 4, 4, 5], [5, 6, 6, 7, 7]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_floor_divide_error(self):
        with self.assertRaises(TypeError):
            self.func_data1 // [1, 2, 3]


class TestDenseFunctionalData2D(unittest.TestCase):
    """Test class for the class DenseFunctionalData in two dimension."""

    def setUp(self):
        argvals = DenseArgvals(
            {"input_dim_0": np.array([1, 2, 3, 4]), "input_dim_1": np.array([5, 6, 7])}
        )

        values = DenseValues(
            np.array(
                [
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]],
                    [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]],
                ]
            )
        )
        self.dense_fd = DenseFunctionalData(argvals, values)

    def test_argvals_stand(self):
        is_equal_dim0 = np.allclose(
            self.dense_fd.argvals_stand["input_dim_0"],
            np.array([0.0, 0.33333333, 0.66666667, 1.0]),
        )
        is_equal_dim1 = np.allclose(
            self.dense_fd.argvals_stand["input_dim_1"], np.array([0.0, 0.5, 1.0])
        )
        self.assertTrue(is_equal_dim0 and is_equal_dim1)

    def test_n_obs(self):
        self.assertEqual(self.dense_fd.n_obs, 3)

    def test_n_dimension(self):
        self.assertEqual(self.dense_fd.n_dimension, 2)

    def test_subset(self):
        new_dense_fd = self.dense_fd[2]
        self.assertIsInstance(new_dense_fd, DenseFunctionalData)
        self.assertEqual(new_dense_fd.n_obs, 1)
        new_dense_fd = self.dense_fd[1:3]
        self.assertIsInstance(new_dense_fd, DenseFunctionalData)
        self.assertEqual(new_dense_fd.n_obs, 2)

    def test_is_compatible(self):
        DenseFunctionalData._is_compatible(self.dense_fd, self.dense_fd)
        self.assertTrue(True)

    def test_to_long(self):
        result = self.dense_fd.to_long()

        expected_dim_0 = np.array(
            [
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
                4,
                4,
                4,
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
                4,
                4,
                4,
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
                4,
                4,
                4,
            ]
        )
        expected_dim_1 = np.array(
            [
                5,
                6,
                7,
                5,
                6,
                7,
                5,
                6,
                7,
                5,
                6,
                7,
                5,
                6,
                7,
                5,
                6,
                7,
                5,
                6,
                7,
                5,
                6,
                7,
                5,
                6,
                7,
                5,
                6,
                7,
                5,
                6,
                7,
                5,
                6,
                7,
            ]
        )
        expected_id = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ]
        )
        expected_values = DenseValues(
            np.array(
                [
                    1,
                    2,
                    3,
                    1,
                    2,
                    3,
                    1,
                    2,
                    3,
                    1,
                    2,
                    3,
                    5,
                    6,
                    7,
                    5,
                    6,
                    7,
                    5,
                    6,
                    7,
                    5,
                    6,
                    7,
                    3,
                    4,
                    5,
                    3,
                    4,
                    5,
                    3,
                    4,
                    5,
                    3,
                    4,
                    5,
                ]
            )
        )

        self.assertTrue(isinstance(result, pd.DataFrame))
        np.testing.assert_array_equal(result["input_dim_0"].values, expected_dim_0)
        np.testing.assert_array_equal(result["input_dim_1"].values, expected_dim_1)
        np.testing.assert_array_equal(result["id"].values, expected_id)
        np.testing.assert_array_equal(result["values"].values, expected_values)

    def test_mean(self):
        mean_fd = self.dense_fd.mean()
        is_equal = np.allclose(
            mean_fd.values,
            np.array(
                [[[3.0, 4.0, 5.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]]
            ),
        )
        self.assertTrue(is_equal)


class TestNoisevariance(unittest.TestCase):
    def setUp(self) -> None:
        fname = THIS_DIR.parent / "data/data_noisy_5_100_005.pickle"
        with open(fname, "rb") as handle:
            self.fdata = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_noisy_5_100_005_2D.pickle"
        with open(fname, "rb") as handle:
            self.fdata_2D = pickle.load(handle)

    def test_noise_variance(self):
        res = self.fdata.noise_variance(order=2)
        expected_res = 0.05073993078636138
        np.testing.assert_almost_equal(res, expected_res)

    def test_noise_variance_error(self):
        with self.assertWarns(UserWarning):
            self.fdata_2D.noise_variance(2)


class TestSmoothDense(unittest.TestCase):
    def setUp(self):
        fname = THIS_DIR.parent / "data/data_noisy_5_1_005.pickle"
        with open(fname, "rb") as handle:
            self.fdata = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_noisy_5_1_001_2D.pickle"
        with open(fname, "rb") as handle:
            self.fdata_2D = pickle.load(handle)

    def test_smooth_1d(self):
        fdata_smooth = self.fdata.smooth(method="LP", bandwidth=0.39)

        expected_values = DenseValues(
            [
                [
                    -0.5974539,
                    -0.5770563,
                    -0.5567013,
                    -0.53681577,
                    -0.51666073,
                    -0.49650265,
                    -0.47590542,
                    -0.45504107,
                    -0.43403896,
                    -0.41221324,
                    -0.39022177,
                    -0.36811323,
                    -0.34594561,
                    -0.32384773,
                    -0.30197577,
                    -0.28002337,
                    -0.25812816,
                    -0.23621977,
                    -0.21441236,
                    -0.1926956,
                    -0.17101794,
                    -0.14932908,
                    -0.1275875,
                    -0.10590862,
                    -0.08434261,
                    -0.06272827,
                    -0.04121643,
                    -0.02001235,
                    0.0008623,
                    0.02141482,
                    0.04189588,
                    0.0622418,
                    0.08256068,
                    0.10262253,
                    0.12254019,
                    0.14240816,
                    0.16202086,
                    0.18153975,
                    0.20071818,
                    0.21962596,
                    0.23799934,
                    0.25594531,
                    0.27370902,
                    0.29111553,
                    0.30805695,
                    0.32498924,
                    0.3418569,
                    0.35852336,
                    0.37502934,
                    0.39113758,
                    0.40660591,
                    0.4213291,
                    0.4350499,
                    0.44818447,
                    0.46058513,
                    0.47250812,
                    0.48353252,
                    0.49417749,
                    0.50448802,
                    0.51401314,
                    0.52298692,
                    0.53121671,
                    0.53881234,
                    0.54623076,
                    0.55341543,
                    0.56069049,
                    0.56745812,
                    0.57366049,
                    0.57920933,
                    0.58437755,
                    0.5892376,
                    0.59351972,
                    0.59707963,
                    0.59992868,
                    0.60231936,
                    0.6042433,
                    0.60564393,
                    0.60635084,
                    0.60648809,
                    0.60603723,
                    0.60500314,
                    0.60340517,
                    0.60116442,
                    0.59825346,
                    0.59461609,
                    0.59020948,
                    0.58493499,
                    0.57896381,
                    0.57226515,
                    0.56490677,
                    0.55701339,
                    0.54859825,
                    0.53901691,
                    0.52902455,
                    0.51781731,
                    0.50679843,
                    0.49555999,
                    0.48387272,
                    0.4713239,
                    0.45753037,
                    0.44197021,
                ]
            ]
        )
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)

    def test_smooth_default(self):
        fdata_smooth = self.fdata.smooth(method="LP")

        expected_values = DenseValues(
            [
                [
                    -5.97946849e-01,
                    -5.77628823e-01,
                    -5.57720092e-01,
                    -5.37722708e-01,
                    -5.17677102e-01,
                    -4.97285943e-01,
                    -4.76582381e-01,
                    -4.55720181e-01,
                    -4.34128897e-01,
                    -4.12189879e-01,
                    -3.90102165e-01,
                    -3.67919432e-01,
                    -3.45768869e-01,
                    -3.23855778e-01,
                    -3.01910480e-01,
                    -2.80014570e-01,
                    -2.58073901e-01,
                    -2.36198219e-01,
                    -2.14422621e-01,
                    -1.92714565e-01,
                    -1.71023123e-01,
                    -1.49302917e-01,
                    -1.27602800e-01,
                    -1.05987972e-01,
                    -8.43776390e-02,
                    -6.28222725e-02,
                    -4.14945411e-02,
                    -2.04688315e-02,
                    2.47664488e-04,
                    2.08191964e-02,
                    4.12680822e-02,
                    6.16516810e-02,
                    8.18423834e-02,
                    1.01857027e-01,
                    1.21788439e-01,
                    1.41518221e-01,
                    1.61103612e-01,
                    1.80410992e-01,
                    1.99420678e-01,
                    2.18148201e-01,
                    2.36404834e-01,
                    2.54289826e-01,
                    2.71915643e-01,
                    2.89222351e-01,
                    3.06279238e-01,
                    3.23043019e-01,
                    3.39781191e-01,
                    3.56305794e-01,
                    3.72448502e-01,
                    3.88351517e-01,
                    4.03586715e-01,
                    4.17944352e-01,
                    4.31555572e-01,
                    4.44356113e-01,
                    4.56795964e-01,
                    4.68577805e-01,
                    4.79640169e-01,
                    4.90280287e-01,
                    5.00531421e-01,
                    5.10219392e-01,
                    5.19084636e-01,
                    5.27188830e-01,
                    5.35025616e-01,
                    5.42640708e-01,
                    5.49991723e-01,
                    5.57134061e-01,
                    5.64146870e-01,
                    5.70607270e-01,
                    5.76459863e-01,
                    5.81719515e-01,
                    5.86592500e-01,
                    5.91058250e-01,
                    5.94900017e-01,
                    5.98028875e-01,
                    6.00521191e-01,
                    6.02539336e-01,
                    6.04060374e-01,
                    6.05002586e-01,
                    6.05295287e-01,
                    6.05009738e-01,
                    6.04136203e-01,
                    6.02680622e-01,
                    6.00620616e-01,
                    5.97903797e-01,
                    5.94481796e-01,
                    5.90315982e-01,
                    5.85341373e-01,
                    5.79514509e-01,
                    5.73022424e-01,
                    5.65827367e-01,
                    5.58019421e-01,
                    5.49705999e-01,
                    5.40725764e-01,
                    5.30707366e-01,
                    5.20114467e-01,
                    5.08608958e-01,
                    4.97274948e-01,
                    4.85694093e-01,
                    4.73567880e-01,
                    4.60462547e-01,
                    4.45942893e-01,
                ]
            ]
        )
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)

    def test_smooth_2d(self):
        fdata_smooth = self.fdata_2D.smooth(method="LP", bandwidth=0.38321537573562725)

        expected_values = DenseValues(
            [
                [
                    [
                        0.14901552,
                        0.22200477,
                        0.27836061,
                        0.32016144,
                        0.34278684,
                        0.34304577,
                        0.32214149,
                        0.279505,
                        0.2239588,
                        0.14181915,
                        0.05396124,
                    ],
                    [
                        -0.03042153,
                        0.07012503,
                        0.14474165,
                        0.20521169,
                        0.24770292,
                        0.27070299,
                        0.26925612,
                        0.24555191,
                        0.20669493,
                        0.14364319,
                        0.06338992,
                    ],
                    [
                        -0.23031995,
                        -0.11439407,
                        -0.02166728,
                        0.06005878,
                        0.12748385,
                        0.17547295,
                        0.19914352,
                        0.19693752,
                        0.17637224,
                        0.13073192,
                        0.0591867,
                    ],
                    [
                        -0.42612638,
                        -0.29990003,
                        -0.18618753,
                        -0.08388081,
                        0.00668385,
                        0.07843129,
                        0.12345954,
                        0.14099084,
                        0.13612521,
                        0.10533424,
                        0.04524713,
                    ],
                    [
                        -0.58197391,
                        -0.45733577,
                        -0.3314915,
                        -0.21256728,
                        -0.10435914,
                        -0.01222493,
                        0.05171364,
                        0.08753309,
                        0.09529844,
                        0.07848976,
                        0.03176578,
                    ],
                    [
                        -0.68871294,
                        -0.57145294,
                        -0.44371455,
                        -0.31602447,
                        -0.19468324,
                        -0.08980949,
                        -0.01039747,
                        0.03837056,
                        0.05908849,
                        0.05357526,
                        0.01712355,
                    ],
                    [
                        -0.72542753,
                        -0.61917603,
                        -0.49766031,
                        -0.37236096,
                        -0.25024835,
                        -0.13998483,
                        -0.05267548,
                        -0.002512,
                        0.02346868,
                        0.02745365,
                        0.00357735,
                    ],
                    [
                        -0.68023751,
                        -0.59266825,
                        -0.48305519,
                        -0.36974936,
                        -0.25979277,
                        -0.15721989,
                        -0.07682903,
                        -0.0257981,
                        0.00443158,
                        0.01147921,
                        -0.00650524,
                    ],
                    [
                        -0.58422828,
                        -0.51386561,
                        -0.42108341,
                        -0.32785859,
                        -0.23717952,
                        -0.15015676,
                        -0.08259833,
                        -0.03453249,
                        -0.00176507,
                        0.00825392,
                        -0.00343421,
                    ],
                    [
                        -0.43404635,
                        -0.38360271,
                        -0.31831431,
                        -0.25143576,
                        -0.18702277,
                        -0.12248644,
                        -0.07055354,
                        -0.03272841,
                        -0.00112642,
                        0.0155741,
                        0.01228484,
                    ],
                    [
                        -0.25615377,
                        -0.20998924,
                        -0.16693782,
                        -0.12934043,
                        -0.09847777,
                        -0.06665242,
                        -0.03928355,
                        -0.01945646,
                        0.01217356,
                        0.03223529,
                        0.0342955,
                    ],
                ]
            ]
        )
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)


class TestMeanDense(unittest.TestCase):
    def setUp(self):
        fname = THIS_DIR.parent / "data/data_noisy_5_100_005.pickle"
        with open(fname, "rb") as handle:
            self.fdata = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_noisy_5_100_005_2D.pickle"
        with open(fname, "rb") as handle:
            self.fdata_2D = pickle.load(handle)

    def test_mean_1d(self):
        mean_estim = self.fdata.mean(method_smoothing="LP")

        expected_values = DenseValues(
            [
                [
                    -1.22780428e-02,
                    -1.21995436e-02,
                    -1.21118890e-02,
                    -1.20417494e-02,
                    -1.19568636e-02,
                    -1.19141539e-02,
                    -1.18820759e-02,
                    -1.18849138e-02,
                    -1.18923259e-02,
                    -1.18817029e-02,
                    -1.18235561e-02,
                    -1.17072378e-02,
                    -1.15773573e-02,
                    -1.14007817e-02,
                    -1.11819461e-02,
                    -1.09302828e-02,
                    -1.06590759e-02,
                    -1.03696641e-02,
                    -1.00600628e-02,
                    -9.73525437e-03,
                    -9.39623183e-03,
                    -9.04039540e-03,
                    -8.66791885e-03,
                    -8.27643887e-03,
                    -7.87341806e-03,
                    -7.45648750e-03,
                    -7.03304245e-03,
                    -6.59815117e-03,
                    -6.14008768e-03,
                    -5.65860699e-03,
                    -5.15661708e-03,
                    -4.65511256e-03,
                    -4.14575344e-03,
                    -3.62910208e-03,
                    -3.11753598e-03,
                    -2.60744507e-03,
                    -2.09199587e-03,
                    -1.58531920e-03,
                    -1.07696658e-03,
                    -5.73012902e-04,
                    -5.71941834e-05,
                    4.46331208e-04,
                    9.42526719e-04,
                    1.40759432e-03,
                    1.87482887e-03,
                    2.35687114e-03,
                    2.80087940e-03,
                    3.21199474e-03,
                    3.58544489e-03,
                    3.96787204e-03,
                    4.35139983e-03,
                    4.73159858e-03,
                    5.09984177e-03,
                    5.45754231e-03,
                    5.77433538e-03,
                    6.07710296e-03,
                    6.38669906e-03,
                    6.71424178e-03,
                    7.02943709e-03,
                    7.34321437e-03,
                    7.62354689e-03,
                    7.89566449e-03,
                    8.18065866e-03,
                    8.46998157e-03,
                    8.77941034e-03,
                    9.06913895e-03,
                    9.33745762e-03,
                    9.61252853e-03,
                    9.87193463e-03,
                    1.01017119e-02,
                    1.02944802e-02,
                    1.04614739e-02,
                    1.05958142e-02,
                    1.06963103e-02,
                    1.07731884e-02,
                    1.08278610e-02,
                    1.08654935e-02,
                    1.08622834e-02,
                    1.08389955e-02,
                    1.07922165e-02,
                    1.07198347e-02,
                    1.06221894e-02,
                    1.05036446e-02,
                    1.03684946e-02,
                    1.02225404e-02,
                    1.00668562e-02,
                    9.90297216e-03,
                    9.71867844e-03,
                    9.50852854e-03,
                    9.26698800e-03,
                    9.00992758e-03,
                    8.73682743e-03,
                    8.39724610e-03,
                    8.05732681e-03,
                    7.72689006e-03,
                    7.43063659e-03,
                    7.12414165e-03,
                    6.80568964e-03,
                    6.56070779e-03,
                    6.39236977e-03,
                    6.27015618e-03,
                ]
            ]
        )
        np.testing.assert_array_almost_equal(mean_estim.values, expected_values)

    def test_smooth_2d(self):
        new_argvals = DenseArgvals(
            {"input_dim_0": np.linspace(0, 1, 21), "input_dim_1": np.linspace(0, 1, 21)}
        )
        fdata_smooth = self.fdata_2D.mean(points=new_argvals, method_smoothing="LP")

        expected_values = DenseValues(
            [
                [
                    [
                        -1.50908426e-03,
                        3.33369860e-03,
                        4.30436936e-03,
                        3.64125585e-03,
                        2.69726593e-03,
                        9.35993214e-04,
                        -1.12216112e-03,
                        -1.75416878e-03,
                        -1.26692812e-04,
                        1.55586020e-03,
                        2.45388203e-03,
                        4.27665860e-03,
                        3.95982743e-03,
                        2.95210115e-03,
                        1.42930488e-03,
                        3.35959244e-04,
                        1.14485936e-04,
                        -3.70252772e-04,
                        -4.35098185e-04,
                        3.27550162e-04,
                        -2.34344593e-03,
                    ],
                    [
                        6.92605392e-04,
                        2.16365331e-03,
                        2.07115332e-03,
                        1.19439346e-03,
                        5.22595698e-04,
                        1.56886078e-04,
                        -1.41211905e-04,
                        3.99358393e-05,
                        1.24082994e-03,
                        2.09145788e-03,
                        2.52497972e-03,
                        2.92669079e-03,
                        1.96450407e-03,
                        8.91778678e-04,
                        1.27185393e-04,
                        1.36124154e-04,
                        5.81955039e-04,
                        1.01833423e-03,
                        1.01879631e-03,
                        1.11797608e-03,
                        -5.07770372e-04,
                    ],
                    [
                        -4.72408047e-04,
                        1.07291755e-03,
                        7.30415410e-04,
                        -1.20664204e-04,
                        -1.06787815e-03,
                        -1.11486980e-03,
                        -4.30457519e-04,
                        5.27758180e-04,
                        1.72075907e-03,
                        2.77029098e-03,
                        3.36001693e-03,
                        3.22188937e-03,
                        2.18343980e-03,
                        1.30062939e-03,
                        1.09964556e-03,
                        8.73658827e-04,
                        1.21508293e-03,
                        1.97975600e-03,
                        2.18239233e-03,
                        2.55144723e-03,
                        2.20429229e-03,
                    ],
                    [
                        -2.37625768e-03,
                        -5.81758253e-04,
                        -6.96593831e-04,
                        -1.62056221e-03,
                        -2.40093323e-03,
                        -2.22405292e-03,
                        -1.06818862e-03,
                        4.34967333e-04,
                        1.57897446e-03,
                        2.82230237e-03,
                        3.70197729e-03,
                        3.52508328e-03,
                        2.45048847e-03,
                        1.81328865e-03,
                        1.80984247e-03,
                        2.06320155e-03,
                        2.54992930e-03,
                        3.08952674e-03,
                        3.13433553e-03,
                        3.21364323e-03,
                        3.96740156e-03,
                    ],
                    [
                        -5.03845221e-03,
                        -2.69896349e-03,
                        -2.44752635e-03,
                        -2.63492683e-03,
                        -3.06378346e-03,
                        -3.15476314e-03,
                        -2.15880198e-03,
                        -6.08368973e-04,
                        6.75613717e-04,
                        2.15179640e-03,
                        3.02342144e-03,
                        2.97177385e-03,
                        2.55947770e-03,
                        2.33566037e-03,
                        2.11342166e-03,
                        2.65469666e-03,
                        3.19566608e-03,
                        3.59494879e-03,
                        3.48195847e-03,
                        3.45402272e-03,
                        3.82075419e-03,
                    ],
                    [
                        -9.13666863e-03,
                        -5.78388477e-03,
                        -4.34389389e-03,
                        -3.98704632e-03,
                        -4.12937899e-03,
                        -4.16594850e-03,
                        -3.74954521e-03,
                        -2.20888920e-03,
                        -8.05660604e-04,
                        5.68607171e-04,
                        1.47599583e-03,
                        1.87229015e-03,
                        1.85672036e-03,
                        1.97842526e-03,
                        2.47441390e-03,
                        2.96664915e-03,
                        3.33637170e-03,
                        3.58569738e-03,
                        3.47239742e-03,
                        3.27563065e-03,
                        3.77628796e-03,
                    ],
                    [
                        -1.35558820e-02,
                        -9.79763030e-03,
                        -7.69728910e-03,
                        -6.74480140e-03,
                        -6.01137589e-03,
                        -5.39744527e-03,
                        -4.87269712e-03,
                        -3.48745587e-03,
                        -2.27086758e-03,
                        -1.10383902e-03,
                        -1.29695345e-04,
                        7.18870653e-04,
                        1.00728962e-03,
                        1.39440972e-03,
                        1.81334863e-03,
                        2.55948043e-03,
                        3.14594102e-03,
                        3.02163989e-03,
                        2.82037581e-03,
                        2.38986735e-03,
                        2.95151145e-03,
                    ],
                    [
                        -1.61860494e-02,
                        -1.34190783e-02,
                        -1.11343497e-02,
                        -9.66566829e-03,
                        -8.22554973e-03,
                        -7.03676785e-03,
                        -5.89301558e-03,
                        -4.56258605e-03,
                        -3.28653830e-03,
                        -2.09948356e-03,
                        -7.72236022e-04,
                        1.52961480e-04,
                        7.27385129e-04,
                        1.07134744e-03,
                        8.85531916e-04,
                        1.69507087e-03,
                        2.14707764e-03,
                        2.06276576e-03,
                        1.83173905e-03,
                        1.06904378e-03,
                        1.06226714e-03,
                    ],
                    [
                        -1.73162497e-02,
                        -1.55449653e-02,
                        -1.35122136e-02,
                        -1.20854893e-02,
                        -1.02508963e-02,
                        -8.25832152e-03,
                        -6.82237982e-03,
                        -5.40691158e-03,
                        -3.87562815e-03,
                        -2.45316248e-03,
                        -9.54365696e-04,
                        7.53262588e-05,
                        9.52196008e-04,
                        1.16152636e-03,
                        8.84080174e-04,
                        8.29677840e-04,
                        4.80116615e-04,
                        4.45774717e-04,
                        3.09412065e-04,
                        4.60305327e-05,
                        6.13253265e-05,
                    ],
                    [
                        -1.85758663e-02,
                        -1.66860008e-02,
                        -1.50617368e-02,
                        -1.35218832e-02,
                        -1.15664823e-02,
                        -9.52816240e-03,
                        -7.71561916e-03,
                        -5.71165434e-03,
                        -3.96033976e-03,
                        -2.44383901e-03,
                        -1.09058412e-03,
                        1.91209872e-04,
                        1.07640980e-03,
                        1.45286955e-03,
                        1.21807714e-03,
                        7.71109798e-04,
                        1.49392597e-04,
                        -4.20680347e-04,
                        -5.77199667e-04,
                        -9.10201454e-04,
                        -3.39461576e-04,
                    ],
                    [
                        -1.68871819e-02,
                        -1.57252350e-02,
                        -1.49512683e-02,
                        -1.38746290e-02,
                        -1.22375729e-02,
                        -1.03762445e-02,
                        -8.31864080e-03,
                        -6.17215310e-03,
                        -4.51496950e-03,
                        -3.29606068e-03,
                        -1.82081318e-03,
                        -1.08267695e-04,
                        9.92803631e-04,
                        1.49552101e-03,
                        1.48038480e-03,
                        7.11999667e-04,
                        2.31454587e-06,
                        -5.06812617e-04,
                        -9.40558029e-04,
                        -1.69681530e-03,
                        -1.73658832e-03,
                    ],
                    [
                        -1.45926284e-02,
                        -1.44153818e-02,
                        -1.45990702e-02,
                        -1.39252040e-02,
                        -1.27159199e-02,
                        -1.10907025e-02,
                        -9.06592846e-03,
                        -7.16515191e-03,
                        -5.60046886e-03,
                        -4.63092603e-03,
                        -2.85782188e-03,
                        -1.21684196e-03,
                        2.78165088e-04,
                        1.18507579e-03,
                        1.18420257e-03,
                        6.73106044e-04,
                        1.75680703e-05,
                        -2.09722007e-04,
                        -7.77769882e-04,
                        -1.42418017e-03,
                        -2.55222064e-03,
                    ],
                    [
                        -1.31608481e-02,
                        -1.43837125e-02,
                        -1.47149980e-02,
                        -1.40801972e-02,
                        -1.28230848e-02,
                        -1.11872319e-02,
                        -9.43686914e-03,
                        -7.80724545e-03,
                        -7.06986982e-03,
                        -5.78169774e-03,
                        -3.96623755e-03,
                        -2.42219303e-03,
                        -9.69159215e-04,
                        4.55815933e-04,
                        7.16566421e-04,
                        2.62257793e-04,
                        -9.33590107e-05,
                        -1.70298304e-05,
                        -4.34456679e-04,
                        -8.36041925e-04,
                        -2.61118280e-03,
                    ],
                    [
                        -1.20633301e-02,
                        -1.42617528e-02,
                        -1.46898038e-02,
                        -1.40549993e-02,
                        -1.26157568e-02,
                        -1.11175373e-02,
                        -9.49186152e-03,
                        -8.11731656e-03,
                        -7.37456679e-03,
                        -6.13385006e-03,
                        -4.75369168e-03,
                        -3.49330734e-03,
                        -2.27824428e-03,
                        -7.84261664e-04,
                        -3.42270515e-04,
                        -2.22781815e-04,
                        -1.47301033e-04,
                        -6.31937477e-05,
                        -1.15483112e-04,
                        -3.16666597e-04,
                        -2.33933628e-03,
                    ],
                    [
                        -1.11228086e-02,
                        -1.36836527e-02,
                        -1.39275144e-02,
                        -1.33394437e-02,
                        -1.19676439e-02,
                        -1.04402335e-02,
                        -9.34496529e-03,
                        -8.00147810e-03,
                        -7.11588550e-03,
                        -5.72285730e-03,
                        -4.67534814e-03,
                        -3.23899260e-03,
                        -2.34470847e-03,
                        -1.67095903e-03,
                        -1.29402002e-03,
                        -1.06471794e-03,
                        -6.28645703e-04,
                        -2.61524856e-04,
                        -2.60006648e-04,
                        -5.63333739e-04,
                        -2.32383358e-03,
                    ],
                    [
                        -1.13482756e-02,
                        -1.28339628e-02,
                        -1.27782551e-02,
                        -1.20653700e-02,
                        -1.09125464e-02,
                        -9.44871055e-03,
                        -8.55508066e-03,
                        -7.34269150e-03,
                        -6.16152142e-03,
                        -4.94569225e-03,
                        -3.91585445e-03,
                        -2.66035240e-03,
                        -2.27897474e-03,
                        -2.04596445e-03,
                        -1.70039479e-03,
                        -1.44139236e-03,
                        -1.32101306e-03,
                        -7.75926841e-04,
                        -6.98458372e-04,
                        -9.66018732e-04,
                        -1.79796164e-03,
                    ],
                    [
                        -1.08030064e-02,
                        -1.11448852e-02,
                        -1.09258777e-02,
                        -1.06646113e-02,
                        -9.55188493e-03,
                        -8.36087622e-03,
                        -7.61633093e-03,
                        -6.22883838e-03,
                        -4.96230295e-03,
                        -3.90578715e-03,
                        -2.64478221e-03,
                        -1.70370393e-03,
                        -1.59390269e-03,
                        -1.78136654e-03,
                        -1.47209048e-03,
                        -7.37840843e-04,
                        -8.57328874e-04,
                        -8.60936509e-04,
                        -1.11858386e-03,
                        -1.28822324e-03,
                        -8.45311089e-04,
                    ],
                    [
                        -9.66622682e-03,
                        -9.54741668e-03,
                        -9.36612301e-03,
                        -9.08656896e-03,
                        -8.32710912e-03,
                        -7.50841419e-03,
                        -6.25539664e-03,
                        -4.88967402e-03,
                        -3.50216336e-03,
                        -2.67455959e-03,
                        -1.81977561e-03,
                        -1.35314797e-03,
                        -1.35881842e-03,
                        -1.33471402e-03,
                        -8.59496109e-04,
                        -2.28025411e-04,
                        -2.69421242e-04,
                        -3.72340176e-04,
                        -1.17206654e-03,
                        -1.52542208e-03,
                        -4.33722355e-04,
                    ],
                    [
                        -8.80790464e-03,
                        -8.61393097e-03,
                        -8.36431934e-03,
                        -7.88273207e-03,
                        -6.57738987e-03,
                        -5.77418523e-03,
                        -4.47452320e-03,
                        -3.45148963e-03,
                        -2.61225093e-03,
                        -2.00831971e-03,
                        -1.33110552e-03,
                        -1.27271515e-03,
                        -1.19390126e-03,
                        -8.01566736e-04,
                        -6.30881218e-04,
                        -4.32224142e-06,
                        3.51588229e-04,
                        -4.64785481e-05,
                        -5.51511782e-04,
                        -8.44234251e-04,
                        1.10517393e-03,
                    ],
                    [
                        -7.95700760e-03,
                        -8.00267223e-03,
                        -7.37507098e-03,
                        -6.60973251e-03,
                        -4.88589192e-03,
                        -3.78032957e-03,
                        -2.51668012e-03,
                        -1.62813076e-03,
                        -1.61895840e-03,
                        -1.54992041e-03,
                        -1.28121871e-03,
                        -1.85755952e-03,
                        -1.84385299e-03,
                        -1.05165692e-03,
                        -6.87260863e-04,
                        4.18552389e-04,
                        1.18247162e-03,
                        1.02796333e-03,
                        8.55077312e-04,
                        7.75799157e-04,
                        3.27535487e-03,
                    ],
                    [
                        -7.04802957e-03,
                        -7.53780336e-03,
                        -6.66537953e-03,
                        -4.83142485e-03,
                        -2.49413495e-03,
                        -1.93384800e-03,
                        -1.36880515e-03,
                        -4.07888543e-04,
                        -1.06725384e-03,
                        -1.71795958e-03,
                        -1.94102366e-03,
                        -3.43561706e-03,
                        -3.81986394e-03,
                        -3.03174151e-03,
                        -2.92728380e-03,
                        -1.52390035e-03,
                        1.73363511e-04,
                        1.68586466e-03,
                        2.73118005e-03,
                        3.89724117e-03,
                        6.00061236e-03,
                    ],
                ]
            ]
        )
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)


class TestCenterDense(unittest.TestCase):
    def setUp(self):
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_noisy_5_10_001_2D.pickle"
        with open(fname, "rb") as handle:
            self.fdata_2D = pickle.load(handle)

    def test_center_1d(self):
        fdata_center = self.fdata.center(method_smoothing="LP")

        expected_values = DenseValues(
            [
                [
                    -0.56230509,
                    -0.39856828,
                    -0.40495846,
                    -0.05037069,
                    0.17990767,
                    0.37865577,
                    0.56929082,
                    0.8191031,
                    0.56213625,
                    0.62339897,
                    0.13118983,
                ],
                [
                    -0.29175426,
                    -0.14883092,
                    -0.08193504,
                    -0.0892208,
                    -0.12018125,
                    -0.26838171,
                    -0.29165542,
                    -0.16241996,
                    -0.27645003,
                    -0.41104266,
                    -0.44687958,
                ],
                [
                    0.5103985,
                    0.5902552,
                    0.47304842,
                    0.44859187,
                    0.28309914,
                    0.33619261,
                    0.43883173,
                    0.43864607,
                    0.61760414,
                    0.59345249,
                    0.67664309,
                ],
                [
                    -0.16912479,
                    -0.27542225,
                    -0.20307774,
                    -0.41660176,
                    -0.45847555,
                    -0.4245009,
                    -0.3313725,
                    -0.13937749,
                    -0.01992546,
                    0.12600243,
                    0.28205796,
                ],
                [
                    -0.52349565,
                    -0.28975803,
                    -0.03164793,
                    0.24041336,
                    0.57204582,
                    0.54404914,
                    0.57984328,
                    0.62890013,
                    0.27760965,
                    0.17354387,
                    -0.19148151,
                ],
                [
                    0.26279178,
                    0.26930995,
                    0.36486935,
                    0.48420599,
                    0.20991916,
                    0.40339136,
                    0.36591174,
                    0.26593811,
                    0.16265634,
                    0.29741741,
                    0.22744764,
                ],
                [
                    -0.04986392,
                    -0.26245018,
                    -0.23396295,
                    -0.5033525,
                    -0.63990914,
                    -0.56811956,
                    -0.61593972,
                    -0.55191869,
                    -0.6371991,
                    -0.68116872,
                    -0.50636442,
                ],
                [
                    -0.22046098,
                    -0.32328053,
                    -0.50180134,
                    -0.58494495,
                    -0.80162547,
                    -0.69418135,
                    -0.84498948,
                    -1.00509669,
                    -0.95964561,
                    -0.78160458,
                    -0.62490444,
                ],
                [
                    0.50097767,
                    0.46435409,
                    0.05606118,
                    -0.2444384,
                    -0.4916243,
                    -0.32887007,
                    -0.439879,
                    -0.3493603,
                    -0.28991064,
                    -0.15596293,
                    0.04243258,
                ],
                [
                    0.59661572,
                    0.57128931,
                    0.44244871,
                    0.31259069,
                    0.36389429,
                    0.36088849,
                    0.5331664,
                    0.38068158,
                    0.4806396,
                    0.33750602,
                    0.25517476,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(fdata_center.values, expected_values)

    def test_center_2d(self):
        fdata_center = self.fdata_2D.center(method_smoothing="LP")

        expected_values = DenseValues(
            [
                [
                    2.45557025e-01,
                    3.05664675e-01,
                    1.12914227e-01,
                    2.34869435e-01,
                    2.15561764e-01,
                    1.79145566e-01,
                    1.77314417e-01,
                    3.10662567e-01,
                    2.78347815e-02,
                    1.71971734e-01,
                    -1.13138189e-01,
                ],
                [
                    2.93090705e-02,
                    1.18182931e-01,
                    2.08856115e-01,
                    2.68198775e-01,
                    3.10280233e-01,
                    2.06905913e-01,
                    1.76668290e-01,
                    2.71905026e-01,
                    1.20196224e-01,
                    -3.36986461e-02,
                    -5.37825658e-02,
                ],
                [
                    -2.53756185e-01,
                    -5.85752370e-02,
                    -1.48032817e-02,
                    1.26366198e-01,
                    9.12970760e-02,
                    1.98023166e-01,
                    2.53991761e-01,
                    1.40215585e-01,
                    1.80635434e-01,
                    2.96111340e-02,
                    2.62000969e-02,
                ],
                [
                    -4.40044913e-01,
                    -4.58903924e-01,
                    -1.87008595e-01,
                    -1.62021008e-01,
                    3.78961405e-04,
                    1.32670132e-01,
                    1.69932572e-01,
                    1.95011250e-01,
                    9.93068951e-02,
                    3.99228088e-02,
                    4.74697805e-02,
                ],
                [
                    -7.75212073e-01,
                    -6.86708921e-01,
                    -5.56288520e-01,
                    -3.79172350e-01,
                    -9.69515733e-02,
                    -1.12152311e-01,
                    9.39773295e-03,
                    2.05041252e-01,
                    3.87747683e-02,
                    1.33181069e-01,
                    -5.32877695e-02,
                ],
                [
                    -7.50335701e-01,
                    -7.64304822e-01,
                    -5.81260164e-01,
                    -3.08216095e-01,
                    -4.08609191e-01,
                    -6.13815430e-02,
                    5.01759533e-03,
                    -3.34404428e-02,
                    -1.03837177e-01,
                    4.45735063e-02,
                    -2.66445390e-02,
                ],
                [
                    -7.15380213e-01,
                    -6.83817824e-01,
                    -4.12863500e-01,
                    -4.50159219e-01,
                    -3.75607388e-01,
                    -1.27078736e-01,
                    -3.51944656e-02,
                    1.29409754e-01,
                    9.96633689e-02,
                    5.50365225e-02,
                    1.57569174e-01,
                ],
                [
                    -7.71026479e-01,
                    -6.74177569e-01,
                    -6.08820491e-01,
                    -4.30274915e-01,
                    -3.98404623e-01,
                    -8.53521381e-02,
                    -9.24067786e-02,
                    -1.70292383e-01,
                    -1.00743579e-01,
                    3.96160071e-02,
                    8.55338771e-02,
                ],
                [
                    -3.06536404e-01,
                    -1.84798138e-01,
                    -3.64679345e-01,
                    -4.12989423e-01,
                    -4.27296726e-01,
                    -9.90043341e-02,
                    -1.45238763e-01,
                    -6.72427266e-02,
                    -6.66905151e-02,
                    -1.24653665e-02,
                    1.02005431e-01,
                ],
                [
                    -3.22853617e-01,
                    -3.37892909e-01,
                    -3.79923183e-01,
                    -3.83658960e-01,
                    -1.98326500e-01,
                    -9.42961247e-02,
                    1.29745716e-01,
                    -8.45456783e-03,
                    9.23725642e-02,
                    -5.10328041e-02,
                    -1.24277695e-01,
                ],
                [
                    -2.81611387e-01,
                    -2.53135105e-01,
                    5.49517102e-02,
                    -2.07185656e-01,
                    -4.11082665e-03,
                    -1.43426658e-01,
                    6.29664120e-02,
                    2.19157764e-02,
                    -1.98124998e-02,
                    -1.17574864e-03,
                    -6.02423789e-02,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(fdata_center.values[0], expected_values)

        expected_values = DenseValues(
            [
                [
                    0.22162085,
                    0.43678014,
                    0.07074014,
                    0.13223938,
                    0.29094615,
                    0.33571937,
                    0.36225347,
                    0.2850687,
                    0.13545034,
                    0.21775149,
                    0.18219078,
                ],
                [
                    0.2348745,
                    0.34330786,
                    0.30113061,
                    0.23568153,
                    0.3648156,
                    0.29006538,
                    0.2415451,
                    0.34069074,
                    -0.10408768,
                    -0.04812962,
                    -0.05947879,
                ],
                [
                    0.15661129,
                    0.19282329,
                    0.08666129,
                    0.14807794,
                    0.1579576,
                    0.47151625,
                    0.37306234,
                    0.11092585,
                    0.07455722,
                    -0.03877241,
                    -0.14032954,
                ],
                [
                    0.15040466,
                    0.0833909,
                    0.18682116,
                    0.03735645,
                    0.11271137,
                    0.18538062,
                    0.18039966,
                    0.15400526,
                    -0.05017119,
                    0.14397286,
                    -0.0392645,
                ],
                [
                    0.1942609,
                    0.14564326,
                    0.01421009,
                    0.11656478,
                    0.17229732,
                    0.17942338,
                    0.08399586,
                    0.23029452,
                    0.15752996,
                    0.01507716,
                    0.10318426,
                ],
                [
                    0.29494169,
                    0.30775594,
                    -0.01063297,
                    0.01323244,
                    0.24068817,
                    -0.03006877,
                    -0.07116886,
                    -0.00084971,
                    0.07812624,
                    0.01647145,
                    -0.04601972,
                ],
                [
                    0.00895258,
                    0.0109828,
                    -0.00870657,
                    0.29917061,
                    0.08955921,
                    0.13347029,
                    -0.01538404,
                    0.00708033,
                    -0.05136176,
                    -0.25049519,
                    -0.1302833,
                ],
                [
                    -0.12251968,
                    -0.16404324,
                    -0.06409681,
                    0.17915483,
                    0.03087905,
                    0.15197508,
                    0.05614279,
                    0.08843235,
                    -0.0817851,
                    0.05942168,
                    0.07302274,
                ],
                [
                    -0.01005121,
                    0.10333735,
                    0.09720602,
                    -0.02572702,
                    -0.09311091,
                    -0.1074769,
                    0.01389463,
                    0.03542216,
                    -0.07702212,
                    0.00459193,
                    0.05199595,
                ],
                [
                    0.10313078,
                    0.19247217,
                    0.1581777,
                    -0.08019231,
                    0.1795325,
                    -0.04726602,
                    0.10585081,
                    0.00564485,
                    -0.04118214,
                    -0.18121169,
                    -0.02299791,
                ],
                [
                    -0.11674756,
                    0.12829621,
                    0.1750107,
                    -0.06522941,
                    0.03733649,
                    -0.06960047,
                    -0.06433018,
                    0.07298184,
                    -0.00441389,
                    0.17731389,
                    -0.04298686,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(fdata_center.values[9], expected_values)

    def test_center_1d_with_given_mean(self):
        precomputed_mean = self.fdata.mean(method_smoothing="LP")

        bb = 1 / np.prod(self.fdata.n_points)
        fdata_center = self.fdata.center(
            mean=precomputed_mean, method_smoothing="LP", bandwidth=bb
        )
        expected_values = DenseValues(
            [
                [
                    -0.56230509,
                    -0.39856828,
                    -0.40495846,
                    -0.05037069,
                    0.17990767,
                    0.37865577,
                    0.56929082,
                    0.8191031,
                    0.56213625,
                    0.62339897,
                    0.13118983,
                ],
                [
                    -0.29175426,
                    -0.14883092,
                    -0.08193504,
                    -0.0892208,
                    -0.12018125,
                    -0.26838171,
                    -0.29165542,
                    -0.16241996,
                    -0.27645003,
                    -0.41104266,
                    -0.44687958,
                ],
                [
                    0.5103985,
                    0.5902552,
                    0.47304842,
                    0.44859187,
                    0.28309914,
                    0.33619261,
                    0.43883173,
                    0.43864607,
                    0.61760414,
                    0.59345249,
                    0.67664309,
                ],
                [
                    -0.16912479,
                    -0.27542225,
                    -0.20307774,
                    -0.41660176,
                    -0.45847555,
                    -0.4245009,
                    -0.3313725,
                    -0.13937749,
                    -0.01992546,
                    0.12600243,
                    0.28205796,
                ],
                [
                    -0.52349565,
                    -0.28975803,
                    -0.03164793,
                    0.24041336,
                    0.57204582,
                    0.54404914,
                    0.57984328,
                    0.62890013,
                    0.27760965,
                    0.17354387,
                    -0.19148151,
                ],
                [
                    0.26279178,
                    0.26930995,
                    0.36486935,
                    0.48420599,
                    0.20991916,
                    0.40339136,
                    0.36591174,
                    0.26593811,
                    0.16265634,
                    0.29741741,
                    0.22744764,
                ],
                [
                    -0.04986392,
                    -0.26245018,
                    -0.23396295,
                    -0.5033525,
                    -0.63990914,
                    -0.56811956,
                    -0.61593972,
                    -0.55191869,
                    -0.6371991,
                    -0.68116872,
                    -0.50636442,
                ],
                [
                    -0.22046098,
                    -0.32328053,
                    -0.50180134,
                    -0.58494495,
                    -0.80162547,
                    -0.69418135,
                    -0.84498948,
                    -1.00509669,
                    -0.95964561,
                    -0.78160458,
                    -0.62490444,
                ],
                [
                    0.50097767,
                    0.46435409,
                    0.05606118,
                    -0.2444384,
                    -0.4916243,
                    -0.32887007,
                    -0.439879,
                    -0.3493603,
                    -0.28991064,
                    -0.15596293,
                    0.04243258,
                ],
                [
                    0.59661572,
                    0.57128931,
                    0.44244871,
                    0.31259069,
                    0.36389429,
                    0.36088849,
                    0.5331664,
                    0.38068158,
                    0.4806396,
                    0.33750602,
                    0.25517476,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(fdata_center.values, expected_values)

    def test_center_2d_with_given_mean(self):
        precomputed_mean = self.fdata_2D.mean(method_smoothing="LP")

        bb = 1 / np.prod(self.fdata.n_points)
        fdata_center = self.fdata_2D.center(
            mean=precomputed_mean, method_smoothing="LP", bandwidth=bb
        )
        expected_values = DenseValues(
            [
                [
                    2.45557025e-01,
                    3.05664675e-01,
                    1.12914227e-01,
                    2.34869435e-01,
                    2.15561764e-01,
                    1.79145566e-01,
                    1.77314417e-01,
                    3.10662567e-01,
                    2.78347815e-02,
                    1.71971734e-01,
                    -1.13138189e-01,
                ],
                [
                    2.93090705e-02,
                    1.18182931e-01,
                    2.08856115e-01,
                    2.68198775e-01,
                    3.10280233e-01,
                    2.06905913e-01,
                    1.76668290e-01,
                    2.71905026e-01,
                    1.20196224e-01,
                    -3.36986461e-02,
                    -5.37825658e-02,
                ],
                [
                    -2.53756185e-01,
                    -5.85752370e-02,
                    -1.48032817e-02,
                    1.26366198e-01,
                    9.12970760e-02,
                    1.98023166e-01,
                    2.53991761e-01,
                    1.40215585e-01,
                    1.80635434e-01,
                    2.96111340e-02,
                    2.62000969e-02,
                ],
                [
                    -4.40044913e-01,
                    -4.58903924e-01,
                    -1.87008595e-01,
                    -1.62021008e-01,
                    3.78961405e-04,
                    1.32670132e-01,
                    1.69932572e-01,
                    1.95011250e-01,
                    9.93068951e-02,
                    3.99228088e-02,
                    4.74697805e-02,
                ],
                [
                    -7.75212073e-01,
                    -6.86708921e-01,
                    -5.56288520e-01,
                    -3.79172350e-01,
                    -9.69515733e-02,
                    -1.12152311e-01,
                    9.39773295e-03,
                    2.05041252e-01,
                    3.87747683e-02,
                    1.33181069e-01,
                    -5.32877695e-02,
                ],
                [
                    -7.50335701e-01,
                    -7.64304822e-01,
                    -5.81260164e-01,
                    -3.08216095e-01,
                    -4.08609191e-01,
                    -6.13815430e-02,
                    5.01759533e-03,
                    -3.34404428e-02,
                    -1.03837177e-01,
                    4.45735063e-02,
                    -2.66445390e-02,
                ],
                [
                    -7.15380213e-01,
                    -6.83817824e-01,
                    -4.12863500e-01,
                    -4.50159219e-01,
                    -3.75607388e-01,
                    -1.27078736e-01,
                    -3.51944656e-02,
                    1.29409754e-01,
                    9.96633689e-02,
                    5.50365225e-02,
                    1.57569174e-01,
                ],
                [
                    -7.71026479e-01,
                    -6.74177569e-01,
                    -6.08820491e-01,
                    -4.30274915e-01,
                    -3.98404623e-01,
                    -8.53521381e-02,
                    -9.24067786e-02,
                    -1.70292383e-01,
                    -1.00743579e-01,
                    3.96160071e-02,
                    8.55338771e-02,
                ],
                [
                    -3.06536404e-01,
                    -1.84798138e-01,
                    -3.64679345e-01,
                    -4.12989423e-01,
                    -4.27296726e-01,
                    -9.90043341e-02,
                    -1.45238763e-01,
                    -6.72427266e-02,
                    -6.66905151e-02,
                    -1.24653665e-02,
                    1.02005431e-01,
                ],
                [
                    -3.22853617e-01,
                    -3.37892909e-01,
                    -3.79923183e-01,
                    -3.83658960e-01,
                    -1.98326500e-01,
                    -9.42961247e-02,
                    1.29745716e-01,
                    -8.45456783e-03,
                    9.23725642e-02,
                    -5.10328041e-02,
                    -1.24277695e-01,
                ],
                [
                    -2.81611387e-01,
                    -2.53135105e-01,
                    5.49517102e-02,
                    -2.07185656e-01,
                    -4.11082665e-03,
                    -1.43426658e-01,
                    6.29664120e-02,
                    2.19157764e-02,
                    -1.98124998e-02,
                    -1.17574864e-03,
                    -6.02423789e-02,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(fdata_center.values[0], expected_values)

        expected_values = DenseValues(
            [
                [
                    0.22162085,
                    0.43678014,
                    0.07074014,
                    0.13223938,
                    0.29094615,
                    0.33571937,
                    0.36225347,
                    0.2850687,
                    0.13545034,
                    0.21775149,
                    0.18219078,
                ],
                [
                    0.2348745,
                    0.34330786,
                    0.30113061,
                    0.23568153,
                    0.3648156,
                    0.29006538,
                    0.2415451,
                    0.34069074,
                    -0.10408768,
                    -0.04812962,
                    -0.05947879,
                ],
                [
                    0.15661129,
                    0.19282329,
                    0.08666129,
                    0.14807794,
                    0.1579576,
                    0.47151625,
                    0.37306234,
                    0.11092585,
                    0.07455722,
                    -0.03877241,
                    -0.14032954,
                ],
                [
                    0.15040466,
                    0.0833909,
                    0.18682116,
                    0.03735645,
                    0.11271137,
                    0.18538062,
                    0.18039966,
                    0.15400526,
                    -0.05017119,
                    0.14397286,
                    -0.0392645,
                ],
                [
                    0.1942609,
                    0.14564326,
                    0.01421009,
                    0.11656478,
                    0.17229732,
                    0.17942338,
                    0.08399586,
                    0.23029452,
                    0.15752996,
                    0.01507716,
                    0.10318426,
                ],
                [
                    0.29494169,
                    0.30775594,
                    -0.01063297,
                    0.01323244,
                    0.24068817,
                    -0.03006877,
                    -0.07116886,
                    -0.00084971,
                    0.07812624,
                    0.01647145,
                    -0.04601972,
                ],
                [
                    0.00895258,
                    0.0109828,
                    -0.00870657,
                    0.29917061,
                    0.08955921,
                    0.13347029,
                    -0.01538404,
                    0.00708033,
                    -0.05136176,
                    -0.25049519,
                    -0.1302833,
                ],
                [
                    -0.12251968,
                    -0.16404324,
                    -0.06409681,
                    0.17915483,
                    0.03087905,
                    0.15197508,
                    0.05614279,
                    0.08843235,
                    -0.0817851,
                    0.05942168,
                    0.07302274,
                ],
                [
                    -0.01005121,
                    0.10333735,
                    0.09720602,
                    -0.02572702,
                    -0.09311091,
                    -0.1074769,
                    0.01389463,
                    0.03542216,
                    -0.07702212,
                    0.00459193,
                    0.05199595,
                ],
                [
                    0.10313078,
                    0.19247217,
                    0.1581777,
                    -0.08019231,
                    0.1795325,
                    -0.04726602,
                    0.10585081,
                    0.00564485,
                    -0.04118214,
                    -0.18121169,
                    -0.02299791,
                ],
                [
                    -0.11674756,
                    0.12829621,
                    0.1750107,
                    -0.06522941,
                    0.03733649,
                    -0.06960047,
                    -0.06433018,
                    0.07298184,
                    -0.00441389,
                    0.17731389,
                    -0.04298686,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(fdata_center.values[9], expected_values)


class TestNormDense(unittest.TestCase):
    def setUp(self) -> None:
        argvals = np.array([0, 1, 2])
        X = np.array([[0, 1, 4], [0, 1, np.sqrt(2)]])
        self.fdata = DenseFunctionalData(
            DenseArgvals({"input_dim_0": argvals}), DenseValues(X)
        )

        argvals = np.array([0, 1])
        X = np.array([[[0, 1], [1, 2]]])
        self.fdata_2D = DenseFunctionalData(
            DenseArgvals({"input_dim_0": argvals, "input_dim_1": argvals}),
            DenseValues(X),
        )

    def test_norm_1d(self):
        results = self.fdata.norm()
        expected_results = np.array([3, np.sqrt(2)])
        np.testing.assert_almost_equal(results, expected_results)

    def test_norm_1d_stand(self):
        results = self.fdata.norm(use_argvals_stand=True)
        expected_results = np.array([np.sqrt(4.5), 1])
        np.testing.assert_almost_equal(results, expected_results)

    def test_norm_1d_squared(self):
        results = self.fdata.norm(squared=True)
        expected_results = np.array([9, 2])
        np.testing.assert_almost_equal(results, expected_results)

    def test_norm_2d(self):
        results = self.fdata_2D.norm()
        expected_results = np.array([np.sqrt(1.5)])
        np.testing.assert_almost_equal(results, expected_results)


class TestNormalizeDense(unittest.TestCase):
    def setUp(self) -> None:
        argvals = np.array([0, 1, 2])
        X = np.array([[0, 1, 4], [0, 1, np.sqrt(2)]])
        self.fdata = DenseFunctionalData(
            DenseArgvals({"input_dim_0": argvals}), DenseValues(X)
        )

        argvals = np.array([0, 1])
        X = np.array([[[0, 1], [1, 2]]])
        self.fdata_2D = DenseFunctionalData(
            DenseArgvals({"input_dim_0": argvals, "input_dim_1": argvals}),
            DenseValues(X),
        )

    def test_normalize_1d(self):
        results = self.fdata.normalize()

        expected_results = np.array([[0, 1 / 3, 4 / 3], [0, 1 / np.sqrt(2), 1]])
        np.testing.assert_almost_equal(results.values, expected_results)

    def test_normalize_2d(self):
        results = self.fdata_2D.normalize()

        expected_results = np.array(
            [[[0, 1 / np.sqrt(1.5)], [1 / np.sqrt(1.5), 2 / np.sqrt(1.5)]]]
        )
        np.testing.assert_almost_equal(results.values, expected_results)


class TestStandardizeDense(unittest.TestCase):
    def setUp(self) -> None:
        argvals = np.array([0, 1, 2])
        X = np.array([[0, 1, 4], [0, 1, np.sqrt(2)]])
        self.fdata = DenseFunctionalData(
            DenseArgvals({"input_dim_0": argvals}), DenseValues(X)
        )

        argvals = np.array([0, 1])
        X = np.array([[[0, 1], [1, 2]], [[-1, 3], [6, 2]]])
        self.fdata_2D = DenseFunctionalData(
            DenseArgvals({"input_dim_0": argvals, "input_dim_1": argvals}),
            DenseValues(X),
        )

    def test_standardize_1d(self):
        results = self.fdata.standardize()

        expected_results = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        np.testing.assert_almost_equal(results.values, expected_results)

    def test_standardize_2d(self):
        results = self.fdata_2D.standardize()

        expected_results = np.array(
            [[[1.0, -1.0], [-1.0, 0.0]], [[-1.0, 1.0], [1.0, 0.0]]]
        )
        np.testing.assert_almost_equal(results.values, expected_results)


class TestRescaleDense(unittest.TestCase):
    def setUp(self) -> None:
        argvals = np.array([0, 1, 2])
        X = np.array([[0, 1, 4], [0, 1, np.sqrt(2)]])
        self.fdata = DenseFunctionalData(
            DenseArgvals({"input_dim_0": argvals}), DenseValues(X)
        )

        argvals = np.array([0, 1])
        X = np.array([[[0, 1], [1, 2]], [[-1, 3], [6, 2]]])
        self.fdata_2D = DenseFunctionalData(
            DenseArgvals({"input_dim_0": argvals, "input_dim_1": argvals}),
            DenseValues(X),
        )

    def test_rescale_1d(self):
        results, _ = self.fdata.rescale()

        expected_results = np.array(
            [[0.0, 1.09383632, 4.37534529], [0.0, 1.09383632, 1.54691816]]
        )
        np.testing.assert_almost_equal(results.values, expected_results)

    def test_rescale_1d_given_weights(self):
        results, weights = self.fdata.rescale(weights=1)

        expected_results = np.array([[0, 1, 4], [0, 1, np.sqrt(2)]])
        np.testing.assert_almost_equal(results.values, expected_results)

        expected_weights = 1
        np.testing.assert_equal(weights, expected_weights)

    def test_rescale_1d_use_argvals_stand(self):
        results, _ = self.fdata.rescale(use_argvals_stand=True)

        expected_results = np.array(
            [[0.0, 1.54691816, 6.18767264], [0.0, 1.54691816, 2.18767264]]
        )
        np.testing.assert_almost_equal(results.values, expected_results)

    def test_rescale_2d(self):
        results, _ = self.fdata_2D.rescale()

        expected_results = np.array(
            [
                [[0.0, 0.73029674], [0.73029674, 1.46059349]],
                [[-0.73029674, 2.19089023], [4.38178046, 1.46059349]],
            ]
        )
        np.testing.assert_almost_equal(results.values, expected_results)


class TestInnerProductDense(unittest.TestCase):
    def setUp(self) -> None:
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_noisy_5_10_001_2D.pickle"
        with open(fname, "rb") as handle:
            self.fdata_2D = pickle.load(handle)

    def test_inner_product_1d(self):
        inn_pro = self.fdata.inner_product(method_smoothing="LP", noise_variance=0)
        expected_inn_pro = np.array(
            [
                [
                    0.23674765,
                    -0.06842593,
                    0.09559542,
                    -0.01996373,
                    0.16692144,
                    0.05550479,
                    -0.17102978,
                    -0.23555852,
                    -0.13432876,
                    0.07244315,
                ],
                [
                    -0.06842593,
                    0.06225314,
                    -0.11304303,
                    0.02984773,
                    -0.04906827,
                    -0.06529572,
                    0.11783618,
                    0.15921136,
                    0.03421792,
                    -0.09227384,
                ],
                [
                    0.09559542,
                    -0.11304303,
                    0.24443619,
                    -0.08098453,
                    0.08730794,
                    0.14420199,
                    -0.23577517,
                    -0.32868103,
                    -0.05344681,
                    0.20477586,
                ],
                [
                    -0.01996373,
                    0.02984773,
                    -0.08098453,
                    0.08806433,
                    -0.07533418,
                    -0.07317048,
                    0.10047474,
                    0.13680832,
                    0.04717606,
                    -0.0908707,
                ],
                [
                    0.16692144,
                    -0.04906827,
                    0.08730794,
                    -0.07533418,
                    0.17602604,
                    0.07520043,
                    -0.16505275,
                    -0.22738922,
                    -0.13727583,
                    0.08600905,
                ],
                [
                    0.05550479,
                    -0.06529572,
                    0.14420199,
                    -0.07317048,
                    0.07520043,
                    0.10268282,
                    -0.15058057,
                    -0.20667663,
                    -0.04854681,
                    0.1270917,
                ],
                [
                    -0.17102978,
                    0.11783618,
                    -0.23577517,
                    0.10047474,
                    -0.16505275,
                    -0.15058057,
                    0.27926812,
                    0.37868223,
                    0.12209768,
                    -0.20028278,
                ],
                [
                    -0.23555852,
                    0.15921136,
                    -0.32868103,
                    0.13680832,
                    -0.22738922,
                    -0.20667663,
                    0.37868223,
                    0.52985729,
                    0.16415945,
                    -0.28354623,
                ],
                [
                    -0.13432876,
                    0.03421792,
                    -0.05344681,
                    0.04717606,
                    -0.13727583,
                    -0.04854681,
                    0.12209768,
                    0.16415945,
                    0.1178676,
                    -0.04885544,
                ],
                [
                    0.07244315,
                    -0.09227384,
                    0.20477586,
                    -0.0908707,
                    0.08600905,
                    0.1270917,
                    -0.20028278,
                    -0.28354623,
                    -0.04885544,
                    0.18671467,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(inn_pro, expected_inn_pro)

    def test_inner_product_1d_unknow_noise_variance(self):
        inn_pro = self.fdata.inner_product(method_smoothing="LP")
        expected_inn_pro = np.array(
            [
                [
                    0.21413749,
                    -0.06842593,
                    0.09559542,
                    -0.01996373,
                    0.16692144,
                    0.05550479,
                    -0.17102978,
                    -0.23555852,
                    -0.13432876,
                    0.07244315,
                ],
                [
                    -0.06842593,
                    0.03964299,
                    -0.11304303,
                    0.02984773,
                    -0.04906827,
                    -0.06529572,
                    0.11783618,
                    0.15921136,
                    0.03421792,
                    -0.09227384,
                ],
                [
                    0.09559542,
                    -0.11304303,
                    0.22182604,
                    -0.08098453,
                    0.08730794,
                    0.14420199,
                    -0.23577517,
                    -0.32868103,
                    -0.05344681,
                    0.20477586,
                ],
                [
                    -0.01996373,
                    0.02984773,
                    -0.08098453,
                    0.06545417,
                    -0.07533418,
                    -0.07317048,
                    0.10047474,
                    0.13680832,
                    0.04717606,
                    -0.0908707,
                ],
                [
                    0.16692144,
                    -0.04906827,
                    0.08730794,
                    -0.07533418,
                    0.15341589,
                    0.07520043,
                    -0.16505275,
                    -0.22738922,
                    -0.13727583,
                    0.08600905,
                ],
                [
                    0.05550479,
                    -0.06529572,
                    0.14420199,
                    -0.07317048,
                    0.07520043,
                    0.08007267,
                    -0.15058057,
                    -0.20667663,
                    -0.04854681,
                    0.1270917,
                ],
                [
                    -0.17102978,
                    0.11783618,
                    -0.23577517,
                    0.10047474,
                    -0.16505275,
                    -0.15058057,
                    0.25665797,
                    0.37868223,
                    0.12209768,
                    -0.20028278,
                ],
                [
                    -0.23555852,
                    0.15921136,
                    -0.32868103,
                    0.13680832,
                    -0.22738922,
                    -0.20667663,
                    0.37868223,
                    0.50724714,
                    0.16415945,
                    -0.28354623,
                ],
                [
                    -0.13432876,
                    0.03421792,
                    -0.05344681,
                    0.04717606,
                    -0.13727583,
                    -0.04854681,
                    0.12209768,
                    0.16415945,
                    0.09525745,
                    -0.04885544,
                ],
                [
                    0.07244315,
                    -0.09227384,
                    0.20477586,
                    -0.0908707,
                    0.08600905,
                    0.1270917,
                    -0.20028278,
                    -0.28354623,
                    -0.04885544,
                    0.16410452,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(inn_pro, expected_inn_pro)

    def test_inner_prodct_2d(self):
        inn_pro = self.fdata_2D.inner_product(method_smoothing="LP", noise_variance=0)
        expected_inn_pro = np.array(
            [
                [
                    8.15168322e-02,
                    2.69330725e-02,
                    -1.42132586e-02,
                    7.26957364e-03,
                    8.40540169e-03,
                    -2.17780622e-02,
                    -2.99854971e-02,
                    -2.93794856e-02,
                    -6.51995424e-03,
                    1.73903591e-04,
                ],
                [
                    2.69330725e-02,
                    4.36032428e-02,
                    -3.88194217e-02,
                    2.35860887e-03,
                    -5.99674479e-04,
                    -2.49371650e-02,
                    8.97887278e-03,
                    1.33863404e-02,
                    2.51167781e-04,
                    -1.84988342e-02,
                ],
                [
                    -1.42132586e-02,
                    -3.88194217e-02,
                    7.00983409e-02,
                    7.31016356e-03,
                    -4.35745786e-03,
                    2.79137838e-02,
                    -2.70363732e-02,
                    -3.26577576e-02,
                    -1.40530235e-03,
                    2.96616587e-02,
                ],
                [
                    7.26957364e-03,
                    2.35860887e-03,
                    7.31016356e-03,
                    2.06870941e-02,
                    -8.84782435e-03,
                    -4.26203513e-03,
                    -7.76930760e-03,
                    -4.26472156e-03,
                    1.04252440e-04,
                    2.94744510e-03,
                ],
                [
                    8.40540169e-03,
                    -5.99674479e-04,
                    -4.35745786e-03,
                    -8.84782435e-03,
                    1.88821085e-02,
                    1.54926265e-03,
                    -5.04607567e-04,
                    -2.45345772e-03,
                    -2.31852805e-03,
                    -9.28485256e-04,
                ],
                [
                    -2.17780622e-02,
                    -2.49371650e-02,
                    2.79137838e-02,
                    -4.26203513e-03,
                    1.54926265e-03,
                    2.88503235e-02,
                    -6.60574486e-03,
                    -9.81152745e-03,
                    -6.67735375e-06,
                    1.39361114e-02,
                ],
                [
                    -2.99854971e-02,
                    8.97887278e-03,
                    -2.70363732e-02,
                    -7.76930760e-03,
                    -5.04607567e-04,
                    -6.60574486e-03,
                    4.25071775e-02,
                    3.36597018e-02,
                    2.80910401e-03,
                    -1.67740124e-02,
                ],
                [
                    -2.93794856e-02,
                    1.33863404e-02,
                    -3.26577576e-02,
                    -4.26472156e-03,
                    -2.45345772e-03,
                    -9.81152745e-03,
                    3.36597018e-02,
                    4.68471700e-02,
                    3.28966705e-03,
                    -2.03961484e-02,
                ],
                [
                    -6.51995424e-03,
                    2.51167781e-04,
                    -1.40530235e-03,
                    1.04252440e-04,
                    -2.31852805e-03,
                    -6.67735375e-06,
                    2.80910401e-03,
                    3.28966705e-03,
                    9.07435882e-03,
                    -2.88593091e-03,
                ],
                [
                    1.73903591e-04,
                    -1.84988342e-02,
                    2.96616587e-02,
                    2.94744510e-03,
                    -9.28485256e-04,
                    1.39361114e-02,
                    -1.67740124e-02,
                    -2.03961484e-02,
                    -2.88593091e-03,
                    2.72125884e-02,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(inn_pro, expected_inn_pro)


class TestCovarianceDense(unittest.TestCase):
    def setUp(self) -> None:
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_noisy_5_10_001_2D.pickle"
        with open(fname, "rb") as handle:
            self.fdata_2D = pickle.load(handle)

    def test_default(self):
        self.fdata.covariance(method_smoothing="LP")
        results = self.fdata._covariance
        expected_results = DenseValues(
            [
                [
                    [
                        0.20710407,
                        0.16515663,
                        0.11524821,
                        0.0696014,
                        0.03624571,
                        0.01495635,
                        0.01182869,
                        0.01785731,
                        0.03101928,
                        0.05678239,
                        0.0913355,
                    ],
                    [
                        0.16515663,
                        0.1462883,
                        0.11637048,
                        0.08890965,
                        0.07124684,
                        0.05672914,
                        0.0534345,
                        0.05765955,
                        0.06480845,
                        0.07809725,
                        0.09294409,
                    ],
                    [
                        0.11524821,
                        0.11637048,
                        0.1170728,
                        0.11298844,
                        0.11337194,
                        0.11434201,
                        0.11492798,
                        0.11488718,
                        0.11401878,
                        0.10961003,
                        0.09938631,
                    ],
                    [
                        0.0696014,
                        0.08890965,
                        0.11298844,
                        0.13697923,
                        0.15743868,
                        0.17170976,
                        0.17747295,
                        0.17485549,
                        0.16211574,
                        0.14184908,
                        0.10864598,
                    ],
                    [
                        0.03624571,
                        0.07124684,
                        0.11337194,
                        0.15743868,
                        0.1963617,
                        0.22238668,
                        0.23110294,
                        0.22941677,
                        0.20933535,
                        0.17299238,
                        0.11990722,
                    ],
                    [
                        0.01495635,
                        0.05672914,
                        0.11434201,
                        0.17170976,
                        0.22238668,
                        0.25738889,
                        0.26976492,
                        0.27017385,
                        0.24924838,
                        0.20390656,
                        0.13284343,
                    ],
                    [
                        0.01182869,
                        0.0534345,
                        0.11492798,
                        0.17747295,
                        0.23110294,
                        0.26976492,
                        0.28640531,
                        0.29171597,
                        0.27385794,
                        0.22882114,
                        0.15000381,
                    ],
                    [
                        0.01785731,
                        0.05765955,
                        0.11488718,
                        0.17485549,
                        0.22941677,
                        0.27017385,
                        0.29171597,
                        0.30206706,
                        0.29015017,
                        0.24759385,
                        0.17381735,
                    ],
                    [
                        0.03101928,
                        0.06480845,
                        0.11401878,
                        0.16211574,
                        0.20933535,
                        0.24924838,
                        0.27385794,
                        0.29015017,
                        0.28690024,
                        0.25040383,
                        0.18355512,
                    ],
                    [
                        0.05678239,
                        0.07809725,
                        0.10961003,
                        0.14184908,
                        0.17299238,
                        0.20390656,
                        0.22882114,
                        0.24759385,
                        0.25040383,
                        0.22677207,
                        0.17602776,
                    ],
                    [
                        0.0913355,
                        0.09294409,
                        0.09938631,
                        0.10864598,
                        0.11990722,
                        0.13284343,
                        0.15000381,
                        0.17381735,
                        0.18355512,
                        0.17602776,
                        0.14552852,
                    ],
                ]
            ]
        )
        expected_noise = 0

        np.testing.assert_almost_equal(results.values, expected_results)
        np.testing.assert_almost_equal(self.fdata._noise_variance_cov, expected_noise)

    def test_points(self):
        points = DenseArgvals({"input_dim_0": np.linspace(0, 1, 6)})
        self.fdata.covariance(points=points, method_smoothing="LP")

        results = self.fdata._covariance
        expected_results = DenseValues(
            [
                [
                    [
                        0.20710407,
                        0.11524821,
                        0.03624571,
                        0.01182869,
                        0.03101928,
                        0.0913355,
                    ],
                    [
                        0.11524821,
                        0.1170728,
                        0.11337194,
                        0.11492798,
                        0.11401878,
                        0.09938631,
                    ],
                    [
                        0.03624571,
                        0.11337194,
                        0.1963617,
                        0.23110294,
                        0.20933535,
                        0.11990722,
                    ],
                    [
                        0.01182869,
                        0.11492798,
                        0.23110294,
                        0.28640531,
                        0.27385794,
                        0.15000381,
                    ],
                    [
                        0.03101928,
                        0.11401878,
                        0.20933535,
                        0.27385794,
                        0.28690024,
                        0.18355512,
                    ],
                    [
                        0.0913355,
                        0.09938631,
                        0.11990722,
                        0.15000381,
                        0.18355512,
                        0.14552852,
                    ],
                ]
            ]
        )
        expected_noise = 0

        np.testing.assert_almost_equal(results.values, expected_results)
        np.testing.assert_almost_equal(self.fdata._noise_variance_cov, expected_noise)

    def test_data2d(self):
        with self.assertRaises(ValueError):
            self.fdata_2D.covariance()
