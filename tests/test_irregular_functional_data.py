#!/usr/bin/python3
# -*-coding:utf8 -*

import numpy as np
import pandas as pd
import pickle
import unittest
import pytest

from pathlib import Path

from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
from FDApy.representation.values import DenseValues, IrregularValues
from FDApy.representation.functional_data import (
    GridFunctionalData,
    DenseFunctionalData,
    IrregularFunctionalData,
)
from FDApy.simulation import KarhunenLoeve

THIS_DIR = Path(__file__)


class TestIrregularFunctionalData(unittest.TestCase):
    def setUp(self):
        self.argvals = IrregularArgvals(
            {
                0: DenseArgvals({"input_dim_0": np.array([0, 1, 2, 3, 4])}),
                1: DenseArgvals({"input_dim_0": np.array([0, 2, 4])}),
                2: DenseArgvals({"input_dim_0": np.array([2, 4])}),
            }
        )
        self.values = IrregularValues(
            {
                0: np.array([1, 2, 3, 4, 5]),
                1: np.array([2, 5, 6]),
                2: np.array([4, 7]),
            }
        )
        self.fdata = IrregularFunctionalData(self.argvals, self.values)

        self.dense_argvals = DenseArgvals({"input_dim_0": np.array([1, 2, 3, 4, 5])})
        self.dense_values = DenseValues(
            np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        )
        self.dense_data = DenseFunctionalData(self.dense_argvals, self.dense_values)

    def test_get_item_slice(self):
        fdata = self.fdata[1:3]
        self.assertIsInstance(fdata, IrregularFunctionalData)
        self.assertEqual(fdata.n_obs, 2)
        self.assertEqual(fdata.n_dimension, 1)
        np.testing.assert_equal(
            fdata.argvals[1]["input_dim_0"], self.argvals[1]["input_dim_0"]
        )
        np.testing.assert_equal(
            fdata.argvals[2]["input_dim_0"], self.argvals[2]["input_dim_0"]
        )
        np.testing.assert_equal(fdata.values[1], self.values[1])

    def test_get_item_index(self):
        fdata = self.fdata[1]
        self.assertIsInstance(fdata, IrregularFunctionalData)
        self.assertEqual(fdata.n_obs, 1)
        self.assertEqual(fdata.n_dimension, 1)
        np.testing.assert_equal(
            fdata.argvals[1]["input_dim_0"], self.argvals[1]["input_dim_0"]
        )
        np.testing.assert_equal(fdata.values[1], self.values[1])

    def test_argvals_getter(self):
        argvals = self.fdata.argvals
        np.testing.assert_equal(argvals, self.argvals)

    def test_argvals_setter(self):
        new_argvals = IrregularArgvals(
            {
                0: DenseArgvals({"input_dim_0": np.array([5, 6, 7, 8, 9])}),
                1: DenseArgvals({"input_dim_0": np.array([6, 8, 10])}),
                2: DenseArgvals({"input_dim_0": np.array([6, 8])}),
            }
        )
        self.fdata.argvals = new_argvals
        np.testing.assert_equal(self.fdata._argvals, new_argvals)

        expected_argvals_stand = IrregularArgvals(
            {
                0: DenseArgvals({"input_dim_0": np.array([0, 0.2, 0.4, 0.6, 0.8])}),
                1: DenseArgvals({"input_dim_0": np.array([0.2, 0.6, 1])}),
                2: DenseArgvals({"input_dim_0": np.array([0.2, 0.6])}),
            }
        )
        np.testing.assert_equal(self.fdata.argvals_stand, expected_argvals_stand)

        with self.assertRaises(TypeError):
            self.fdata.argvals = 0

    def test_values_property(self):
        values = self.fdata.values
        np.testing.assert_array_equal(values, self.values)

    def test_values_setter(self):
        new_values = IrregularValues(
            {
                0: np.array([1, 4, 3, 4, 9]),
                1: np.array([1, 5, 3]),
                2: np.array([7, 7]),
            }
        )
        self.fdata.values = new_values
        np.testing.assert_array_equal(self.fdata.values, new_values)

        with self.assertRaises(TypeError):
            self.fdata.values = 0

    def test_n_points(self):
        expected_n_points = {0: (5,), 1: (3,), 2: (2,)}
        self.assertDictEqual(self.fdata.n_points, expected_n_points)

    def test_is_compatible(self):
        IrregularFunctionalData._is_compatible(self.fdata, self.fdata)
        self.assertTrue(True)

    def test_non_compatible_type(self):
        with self.assertRaises(TypeError):
            IrregularFunctionalData._is_compatible(self.fdata, self.dense_data)

    def test_non_compatible_nobs(self):
        argvals = IrregularArgvals(
            {
                0: DenseArgvals({"input_dim_0": np.array([0, 1, 2, 3, 4])}),
                1: DenseArgvals({"input_dim_0": np.array([0, 2, 4])}),
            }
        )
        values = IrregularValues(
            {
                0: np.array([1, 2, 3, 4, 5]),
                1: np.array([2, 5, 6]),
            }
        )
        func_data = IrregularFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            IrregularFunctionalData._is_compatible(self.fdata, func_data)

    def test_non_compatible_ndim(self):
        argvals = IrregularArgvals(
            {
                0: DenseArgvals(
                    {
                        "input_dim_0": np.array([0, 1, 2, 3, 4]),
                        "input_dim_1": np.array([5, 6, 7]),
                    }
                ),
                1: DenseArgvals(
                    {
                        "input_dim_0": np.array([0, 2, 4]),
                        "input_dim_1": np.array([1, 2, 3]),
                    }
                ),
                2: DenseArgvals(
                    {"input_dim_0": np.array([2, 4]), "input_dim_1": np.array([1, 2])}
                ),
            }
        )
        values = IrregularValues(
            {
                0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4], [1, 2, 4]]),
                1: np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
                2: np.array([[1, 2], [3, 4]]),
            }
        )
        func_data = IrregularFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            IrregularFunctionalData._is_compatible(self.fdata, func_data)

    def test_concatenate(self):
        fdata = IrregularFunctionalData.concatenate(self.fdata, self.fdata)

        expected_argvals = IrregularArgvals(
            {
                0: DenseArgvals({"input_dim_0": np.array([0, 1, 2, 3, 4])}),
                1: DenseArgvals({"input_dim_0": np.array([0, 2, 4])}),
                2: DenseArgvals({"input_dim_0": np.array([2, 4])}),
                3: DenseArgvals({"input_dim_0": np.array([0, 1, 2, 3, 4])}),
                4: DenseArgvals({"input_dim_0": np.array([0, 2, 4])}),
                5: DenseArgvals({"input_dim_0": np.array([2, 4])}),
            }
        )
        expected_values = IrregularValues(
            {
                0: np.array([1, 2, 3, 4, 5]),
                1: np.array([2, 5, 6]),
                2: np.array([4, 7]),
                3: np.array([1, 2, 3, 4, 5]),
                4: np.array([2, 5, 6]),
                5: np.array([4, 7]),
            }
        )

        self.assertIsInstance(fdata, IrregularFunctionalData)
        np.testing.assert_allclose(fdata.argvals, expected_argvals)
        np.testing.assert_allclose(fdata.values, expected_values)

    def test_to_long(self):
        result = self.fdata.to_long()

        expected_dim = np.array([0, 1, 2, 3, 4, 0, 2, 4, 2, 4])
        expected_id = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
        expected_values = DenseValues(np.array([1, 2, 3, 4, 5, 2, 5, 6, 4, 7]))

        self.assertTrue(isinstance(result, pd.DataFrame))
        np.testing.assert_array_equal(result["input_dim_0"].values, expected_dim)
        np.testing.assert_array_equal(result["id"].values, expected_id)
        np.testing.assert_array_equal(result["values"].values, expected_values)


class TestIrregularFunctionalData1D(unittest.TestCase):
    """Test class for the class IrregularFunctionalData in one dimension."""

    def setUp(self):
        self.argvals = {
            0: DenseArgvals({"input_dim_0": np.array([1, 2, 3, 4])}),
            1: DenseArgvals({"input_dim_0": np.array([2, 4])}),
            2: DenseArgvals({"input_dim_0": np.array([0, 2, 3])}),
        }
        self.values = {
            0: np.array([1, 2, 3, 4]),
            1: np.array([5, 6]),
            2: np.array([8, 9, 7]),
        }
        self.irregu_fd = IrregularFunctionalData(
            IrregularArgvals(self.argvals), IrregularValues(self.values)
        )

    def test_argvals_stand(self):
        is_equal = [
            np.allclose(
                self.irregu_fd.argvals_stand[0]["input_dim_0"],
                np.array([0.25, 0.5, 0.75, 1.0]),
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[1]["input_dim_0"], np.array([0.5, 1.0])
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[2]["input_dim_0"],
                np.array([0.0, 0.5, 0.75]),
            ),
        ]
        self.assertTrue(np.all(is_equal))

    def test_n_obs(self):
        self.assertEqual(self.irregu_fd.n_obs, 3)

    def test_n_dimension(self):
        self.assertEqual(self.irregu_fd.n_dimension, 1)

    def test_subset(self):
        new_irregu_fd = self.irregu_fd[2]
        self.assertIsInstance(new_irregu_fd, IrregularFunctionalData)
        self.assertEqual(new_irregu_fd.n_obs, 1)
        new_irregu_fd = self.irregu_fd[:2]
        self.assertIsInstance(new_irregu_fd, IrregularFunctionalData)
        self.assertEqual(new_irregu_fd.n_obs, 2)

    def test_is_compatible(self):
        IrregularFunctionalData._is_compatible(self.irregu_fd, self.irregu_fd)
        self.assertTrue(True)

    def test_mean(self):
        mean_fd = self.irregu_fd.mean()
        is_equal = np.allclose(
            mean_fd.values, np.array([[8.0, 1.0, 5.33333333, 5.0, 5.0]])
        )
        self.assertTrue(is_equal)


class TestIrregularFunctionalData2D(unittest.TestCase):
    """Test class for the class IrregularFunctionalData in two dimension."""

    def setUp(self):
        argvals = {
            0: DenseArgvals(
                {
                    "input_dim_0": np.array([1, 2, 3, 4]),
                    "input_dim_1": np.array([5, 6, 7]),
                }
            ),
            1: DenseArgvals(
                {"input_dim_0": np.array([2, 4]), "input_dim_1": np.array([1, 2, 3])}
            ),
            2: DenseArgvals(
                {"input_dim_0": np.array([4, 5, 6]), "input_dim_1": np.array([8, 9])}
            ),
        }
        values = {
            0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
            1: np.array([[1, 2, 3], [1, 2, 3]]),
            2: np.array([[8, 9], [8, 9], [8, 9]]),
        }
        self.irregu_fd = IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(values)
        )

    def test_argvals_stand(self):
        is_equal = [
            np.allclose(
                self.irregu_fd.argvals_stand[0]["input_dim_0"],
                np.array([0.0, 0.2, 0.4, 0.6]),
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[1]["input_dim_0"], np.array([0.2, 0.6])
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[2]["input_dim_0"],
                np.array([0.6, 0.8, 1.0]),
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[0]["input_dim_1"],
                np.array([0.5, 0.625, 0.75]),
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[1]["input_dim_1"],
                np.array([0.0, 0.125, 0.25]),
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[2]["input_dim_1"], np.array([0.875, 1.0])
            ),
        ]
        self.assertTrue(np.all(is_equal))

    def test_n_obs(self):
        self.assertEqual(self.irregu_fd.n_obs, 3)

    def test_n_dimension(self):
        self.assertEqual(self.irregu_fd.n_dimension, 2)

    def test_subset(self):
        new_irregu_fd = self.irregu_fd[2]
        self.assertIsInstance(new_irregu_fd, IrregularFunctionalData)
        self.assertEqual(new_irregu_fd.n_obs, 1)
        new_irregu_fd = self.irregu_fd[:2]
        self.assertIsInstance(new_irregu_fd, IrregularFunctionalData)
        self.assertEqual(new_irregu_fd.n_obs, 2)

    def test_is_compatible(self):
        IrregularFunctionalData._is_compatible(self.irregu_fd, self.irregu_fd)
        self.assertTrue(True)

    def test_to_long(self):
        result = self.irregu_fd.to_long()

        expected_dim_0 = np.array(
            [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 5, 5, 6, 6]
        )
        expected_dim_1 = np.array(
            [5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 1, 2, 3, 1, 2, 3, 8, 9, 8, 9, 8, 9]
        )
        expected_id = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        )
        expected_values = np.array(
            [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 8, 9, 8, 9, 8, 9]
        )

        self.assertTrue(isinstance(result, pd.DataFrame))
        np.testing.assert_array_equal(result["input_dim_0"].values, expected_dim_0)
        np.testing.assert_array_equal(result["input_dim_1"].values, expected_dim_1)
        np.testing.assert_array_equal(result["id"].values, expected_id)
        np.testing.assert_array_equal(result["values"].values, expected_values)

    def test_mean(self):
        mean_fd = self.irregu_fd.mean()
        expected_mean = DenseValues(
            np.array(
                [
                    [
                        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 3.0, 4.0, 1.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 8.0, 9.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0],
                    ]
                ]
            )
        )
        np.testing.assert_allclose(mean_fd.values, expected_mean)


class TestPerformComputation(unittest.TestCase):
    def setUp(self):
        self.argvals = {
            0: DenseArgvals({"input_dim_0": np.array([0, 1, 2, 3, 4])}),
            1: DenseArgvals({"input_dim_0": np.array([0, 2, 4])}),
            2: DenseArgvals({"input_dim_0": np.array([2, 4])}),
        }
        self.values1 = {
            0: np.array([1, 2, 3, 4, 5]),
            1: np.array([2, 5, 6]),
            2: np.array([4, 7]),
        }
        self.func_data1 = IrregularFunctionalData(
            IrregularArgvals(self.argvals), IrregularValues(self.values1)
        )

        self.values2 = {
            0: np.array([5, 4, 3, 2, 1]),
            1: np.array([5, 3, 1]),
            2: np.array([5, 3]),
        }
        self.func_data2 = IrregularFunctionalData(
            IrregularArgvals(self.argvals), IrregularValues(self.values2)
        )

    def test_addition(self):
        result = self.func_data1 + self.func_data2

        expected_values = IrregularValues(
            {0: np.array([6, 6, 6, 6, 6]), 1: np.array([7, 8, 7]), 2: np.array([9, 10])}
        )
        np.testing.assert_equal(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_addition_number(self):
        result = self.func_data1 + 2

        expected_values = IrregularValues(
            {0: np.array([3, 4, 5, 6, 7]), 1: np.array([4, 7, 8]), 2: np.array([6, 8])}
        )
        np.testing.assert_equal(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_addition_error(self):
        with self.assertRaises(TypeError):
            self.func_data1 + [1, 2, 3]

    def test_substraction(self):
        result = self.func_data1 - self.func_data2

        expected_values = IrregularValues(
            {
                0: np.array([-4, -2, 0, 2, 4]),
                1: np.array([-3, 2, 5]),
                2: np.array([-1, 4]),
            }
        )
        self.assertEqual(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_substraction_number(self):
        result = self.func_data1 - 2

        expected_values = IrregularValues(
            {0: np.array([-1, 0, 1, 2, 3]), 1: np.array([0, 3, 4]), 2: np.array([2, 5])}
        )
        np.testing.assert_equal(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_substraction_error(self):
        with self.assertRaises(TypeError):
            self.func_data1 - [1, 2, 3]

    def test_multiplication(self):
        result = self.func_data1 * self.func_data2

        expected_values = IrregularValues(
            {
                0: np.array([5, 8, 9, 8, 5]),
                1: np.array([10, 15, 6]),
                2: np.array([20, 21]),
            }
        )
        self.assertEqual(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_multiplication_number(self):
        result = self.func_data1 - 2

        expected_values = IrregularValues(
            {
                0: np.array([2, 4, 6, 8, 10]),
                1: np.array([4, 10, 12]),
                2: np.array([8, 14]),
            }
        )
        np.testing.assert_equal(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_multiplication_error(self):
        with self.assertRaises(TypeError):
            self.func_data1 * [1, 2, 3]

    def test_right_multiplication(self):
        result = GridFunctionalData.__rmul__(self.func_data1, self.func_data2)

        expected_values = IrregularValues(
            {
                0: np.array([5, 8, 9, 8, 5]),
                1: np.array([10, 15, 6]),
                2: np.array([20, 21]),
            }
        )
        self.assertEqual(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_true_divide(self):
        result = self.func_data1 / self.func_data2

        expected_values = IrregularValues(
            {
                0: np.array([0.2, 0.5, 1.0, 2.0, 5.0]),
                1: np.array([0.4, 1.66666667, 6.0]),
                2: np.array([0.8, 2.33333333]),
            }
        )
        self.assertEqual(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_almost_equal(result.values, expected_values)

    def test_true_divide_number(self):
        result = self.func_data1 / 2

        expected_values = IrregularValues(
            {
                0: np.array([0.5, 1, 1.5, 2, 2.5]),
                1: np.array([1, 2.5, 3]),
                2: np.array([2, 3.5]),
            }
        )
        np.testing.assert_equal(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_true_divide_error(self):
        with self.assertRaises(TypeError):
            self.func_data1 / [1, 2, 3]

    def test_floor_divide(self):
        result = self.func_data1 // self.func_data2

        expected_values = IrregularValues(
            {0: np.array([0, 0, 1, 2, 5]), 1: np.array([0, 1, 6]), 2: np.array([0, 2])}
        )
        self.assertEqual(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_almost_equal(result.values, expected_values)

    def test_floor_divide_number(self):
        result = self.func_data1 // 2

        expected_values = IrregularValues(
            {0: np.array([0, 1, 1, 2, 2]), 1: np.array([1, 2, 3]), 2: np.array([2, 3])}
        )
        np.testing.assert_equal(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_floor_divide_error(self):
        with self.assertRaises(TypeError):
            self.func_data1 // [1, 2, 3]


class TestNoisevariance(unittest.TestCase):
    def setUp(self) -> None:
        fname = THIS_DIR.parent / "data/data_sparse_5_100_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)

    def test_noise_variance(self):
        res = self.fdata_sparse.noise_variance(order=2)
        expected_res = 0.0005090234464867475
        np.testing.assert_almost_equal(res, expected_res)

    def test_noise_variance_error(self):
        argvals = {
            0: DenseArgvals(
                {
                    "input_dim_0": np.array([1, 2, 3, 4]),
                    "input_dim_1": np.array([5, 6, 7]),
                }
            ),
            1: DenseArgvals(
                {"input_dim_0": np.array([2, 4]), "input_dim_1": np.array([1, 2, 3])}
            ),
            2: DenseArgvals(
                {"input_dim_0": np.array([4, 5, 6]), "input_dim_1": np.array([8, 9])}
            ),
        }
        values = {
            0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
            1: np.array([[1, 2, 3], [1, 2, 3]]),
            2: np.array([[8, 9], [8, 9], [8, 9]]),
        }
        irregu_fd = IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(values)
        )

        with self.assertWarns(UserWarning):
            irregu_fd.noise_variance(2)


class TestSmoothIrregular(unittest.TestCase):
    def setUp(self):
        fname = THIS_DIR.parent / "data/data_sparse_5_1_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)

    def test_smooth_1d(self):
        fdata_smooth = self.fdata_sparse.smooth(method="LP", degree=1)
        expected_values = DenseValues(
            [
                [
                    -0.56326329,
                    -0.54494478,
                    -0.52641978,
                    -0.50764072,
                    -0.46922041,
                    -0.44956399,
                    -0.42961677,
                    -0.40938993,
                    -0.38884672,
                    -0.36810284,
                    -0.34716776,
                    -0.32605597,
                    -0.30478539,
                    -0.28337458,
                    -0.26185082,
                    -0.24023413,
                    -0.19680983,
                    -0.17503878,
                    -0.15325514,
                    -0.13148416,
                    -0.1097493,
                    -0.0880272,
                    -0.06637831,
                    -0.04482937,
                    -0.02340539,
                    -0.00212955,
                    0.01898122,
                    0.04004811,
                    0.06090899,
                    0.08154169,
                    0.10192679,
                    0.12204756,
                    0.16144216,
                    0.18069549,
                    0.21828335,
                    0.23661482,
                    0.27275286,
                    0.29054899,
                    0.30839357,
                    0.32583092,
                    0.34283442,
                    0.35946872,
                    0.37610682,
                    0.3922227,
                    0.42281379,
                    0.43725521,
                    0.45110589,
                    0.46435006,
                    0.48895389,
                    0.5002791,
                    0.51092835,
                    0.52089975,
                    0.54112228,
                    0.55092278,
                    0.56042424,
                    0.57831341,
                    0.58660713,
                    0.59441184,
                    0.60169115,
                    0.60841245,
                    0.62007098,
                    0.62496515,
                    0.6292169,
                    0.63282094,
                    0.63578081,
                    0.63811047,
                    0.64004341,
                    0.64134317,
                    0.64202213,
                    0.64220011,
                    0.64172507,
                    0.63883052,
                    0.63306105,
                    0.62907168,
                    0.62436033,
                    0.61288891,
                    0.60622996,
                    0.59905879,
                    0.59080173,
                    0.58186235,
                    0.57232521,
                    0.56228659,
                    0.5518756,
                    0.54009588,
                    0.52769149,
                    0.51479854,
                ]
            ]
        )
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)

    def test_smooth_2d(self):
        argvals = IrregularArgvals(
            {
                0: DenseArgvals(
                    {
                        "input_dim_0": np.array([1, 2, 3, 4]),
                        "input_dim_1": np.array([5, 6, 7]),
                    }
                ),
                1: DenseArgvals(
                    {
                        "input_dim_0": np.array([2, 4]),
                        "input_dim_1": np.array([1, 2, 3]),
                    }
                ),
                2: DenseArgvals(
                    {
                        "input_dim_0": np.array([4, 5, 6]),
                        "input_dim_1": np.array([8, 9]),
                    }
                ),
            }
        )
        values = IrregularValues(
            {
                0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
                1: np.array([[1, 2, 3], [1, 2, 3]]),
                2: np.array([[8, 9], [8, 9], [8, 9]]),
            }
        )
        data = IrregularFunctionalData(argvals, values)
        fdata_smooth = data.smooth(method="LP")

        expected_values = DenseValues(
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0, 1.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0, 4.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0],
                ],
            ]
        )
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)


class TestInnerProductIrregular(unittest.TestCase):
    def setUp(self):
        kl = KarhunenLoeve(
            basis_name="bsplines",
            n_functions=5,
            argvals=DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)}),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)
        self.fdata_sparse = kl.sparse_data

    def test_inner_product(self):
        res = self.fdata_sparse.inner_product()
        expected_res = np.array(
            [
                [
                    0.23035739,
                    -0.0621319,
                    0.10556175,
                    -0.03698105,
                    0.15467418,
                    0.0733234,
                    -0.18983942,
                    -0.23764374,
                    -0.10367822,
                    0.06607494,
                ],
                [
                    -0.0621319,
                    0.05180212,
                    -0.10963497,
                    0.03631035,
                    -0.05850593,
                    -0.07394127,
                    0.12552066,
                    0.14873259,
                    0.03535411,
                    -0.09378843,
                ],
                [
                    0.10556175,
                    -0.10963497,
                    0.23670954,
                    -0.07630333,
                    0.10922808,
                    0.1596596,
                    -0.2612113,
                    -0.30625829,
                    -0.06474301,
                    0.20670926,
                ],
                [
                    -0.03698105,
                    0.03631035,
                    -0.07630333,
                    0.0851861,
                    -0.09668021,
                    -0.08066304,
                    0.09862734,
                    0.12282804,
                    0.04126052,
                    -0.0938674,
                ],
                [
                    0.15467418,
                    -0.05850593,
                    0.10922808,
                    -0.09668021,
                    0.17140669,
                    0.10464569,
                    -0.17737578,
                    -0.22283667,
                    -0.09272966,
                    0.10789093,
                ],
                [
                    0.0733234,
                    -0.07394127,
                    0.1596596,
                    -0.08066304,
                    0.10464569,
                    0.12220188,
                    -0.18450459,
                    -0.21927603,
                    -0.05435964,
                    0.15263131,
                ],
                [
                    -0.18983942,
                    0.12552066,
                    -0.2612113,
                    0.09862734,
                    -0.17737578,
                    -0.18450459,
                    0.32190416,
                    0.38440204,
                    0.10635025,
                    -0.22415604,
                ],
                [
                    -0.23764374,
                    0.14873259,
                    -0.30625829,
                    0.12282804,
                    -0.22283667,
                    -0.21927603,
                    0.38440204,
                    0.46065588,
                    0.13265149,
                    -0.26353797,
                ],
                [
                    -0.10367822,
                    0.03535411,
                    -0.06474301,
                    0.04126052,
                    -0.09272966,
                    -0.05435964,
                    0.10635025,
                    0.13265149,
                    0.05444484,
                    -0.05483334,
                ],
                [
                    0.06607494,
                    -0.09378843,
                    0.20670926,
                    -0.0938674,
                    0.10789093,
                    0.15263131,
                    -0.22415604,
                    -0.26353797,
                    -0.05483334,
                    0.19659407,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(res, expected_res)

    def test_inner_product_with_unkwon_noise_variance(self):
        res = self.fdata_sparse.inner_product(noise_variance=0)
        expected_res = np.array(
            [
                [
                    0.23064007,
                    -0.0621319,
                    0.10556175,
                    -0.03698105,
                    0.15467418,
                    0.0733234,
                    -0.18983942,
                    -0.23764374,
                    -0.10367822,
                    0.06607494,
                ],
                [
                    -0.0621319,
                    0.05208479,
                    -0.10963497,
                    0.03631035,
                    -0.05850593,
                    -0.07394127,
                    0.12552066,
                    0.14873259,
                    0.03535411,
                    -0.09378843,
                ],
                [
                    0.10556175,
                    -0.10963497,
                    0.23699222,
                    -0.07630333,
                    0.10922808,
                    0.1596596,
                    -0.2612113,
                    -0.30625829,
                    -0.06474301,
                    0.20670926,
                ],
                [
                    -0.03698105,
                    0.03631035,
                    -0.07630333,
                    0.08546878,
                    -0.09668021,
                    -0.08066304,
                    0.09862734,
                    0.12282804,
                    0.04126052,
                    -0.0938674,
                ],
                [
                    0.15467418,
                    -0.05850593,
                    0.10922808,
                    -0.09668021,
                    0.17168937,
                    0.10464569,
                    -0.17737578,
                    -0.22283667,
                    -0.09272966,
                    0.10789093,
                ],
                [
                    0.0733234,
                    -0.07394127,
                    0.1596596,
                    -0.08066304,
                    0.10464569,
                    0.12248455,
                    -0.18450459,
                    -0.21927603,
                    -0.05435964,
                    0.15263131,
                ],
                [
                    -0.18983942,
                    0.12552066,
                    -0.2612113,
                    0.09862734,
                    -0.17737578,
                    -0.18450459,
                    0.32218684,
                    0.38440204,
                    0.10635025,
                    -0.22415604,
                ],
                [
                    -0.23764374,
                    0.14873259,
                    -0.30625829,
                    0.12282804,
                    -0.22283667,
                    -0.21927603,
                    0.38440204,
                    0.46093856,
                    0.13265149,
                    -0.26353797,
                ],
                [
                    -0.10367822,
                    0.03535411,
                    -0.06474301,
                    0.04126052,
                    -0.09272966,
                    -0.05435964,
                    0.10635025,
                    0.13265149,
                    0.05472751,
                    -0.05483334,
                ],
                [
                    0.06607494,
                    -0.09378843,
                    0.20670926,
                    -0.0938674,
                    0.10789093,
                    0.15263131,
                    -0.22415604,
                    -0.26353797,
                    -0.05483334,
                    0.19687675,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(res, expected_res)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_inner_product_2d(self):
        argvals = IrregularArgvals(
            {
                0: DenseArgvals(
                    {
                        "input_dim_0": np.array([1, 2, 3, 4]),
                        "input_dim_1": np.array([5, 6, 7]),
                    }
                ),
                1: DenseArgvals(
                    {
                        "input_dim_0": np.array([1, 2, 3, 4]),
                        "input_dim_1": np.array([5, 6, 7]),
                    }
                ),
                2: DenseArgvals(
                    {
                        "input_dim_0": np.array([1, 2, 3, 4]),
                        "input_dim_1": np.array([5, 6, 7]),
                    }
                ),
            }
        )
        values = IrregularValues(
            {
                0: np.array(
                    [
                        [np.nan, 2, np.nan],
                        [4, np.nan, np.nan],
                        [2, np.nan, 3],
                        [np.nan, 1, 2],
                    ]
                ),
                1: np.array(
                    [
                        [1, 2, np.nan],
                        [4, np.nan, np.nan],
                        [2, np.nan, 3],
                        [4, np.nan, np.nan],
                    ]
                ),
                2: np.array(
                    [[1, np.nan, 3], [4, 4, np.nan], [2, 3, np.nan], [4, np.nan, 2]]
                ),
            }
        )
        data = IrregularFunctionalData(argvals, values)

        res = data.inner_product(method_smoothing="PS")
        expected_res = np.array(
            [
                [0.39776689, 0.01796032, -0.41572722],
                [0.01796032, 0.26960847, -0.2875688],
                [-0.41572722, -0.2875688, 0.70329601],
            ]
        )
        np.testing.assert_array_almost_equal(res, expected_res)


class TestCenterIrregular(unittest.TestCase):
    def setUp(self):
        fname = THIS_DIR.parent / "data/data_sparse_5_10_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)

    def test_center_1d(self):
        fdata_center = self.fdata_sparse.center(method_smoothing="LP")

        self.assertIsInstance(fdata_center, IrregularFunctionalData)

        expected_values = DenseValues(
            [
                -0.64181022,
                -0.63146583,
                -0.62030816,
                -0.58099734,
                -0.5661533,
                -0.55023599,
                -0.53335645,
                -0.51568341,
                -0.49725285,
                -0.47817533,
                -0.4584701,
                -0.43808277,
                -0.41707468,
                -0.39550758,
                -0.37344991,
                -0.32797614,
                -0.30460999,
                -0.28085438,
                -0.23229342,
                -0.20755938,
                -0.18253189,
                -0.15721873,
                -0.13167712,
                -0.08011008,
                -0.02812959,
                -0.00209584,
                0.02391934,
                0.04985519,
                0.07569925,
                0.10142703,
                0.12699214,
                0.15241893,
                0.17770677,
                0.20280282,
                0.22762609,
                0.25217087,
                0.27630286,
                0.29990463,
                0.32303646,
                0.34559645,
                0.36748365,
                0.38869127,
                0.40925719,
                0.4484441,
                0.46704345,
                0.4849944,
                0.50221202,
                0.51858717,
                0.53414597,
                0.56261616,
                0.57543936,
                0.59844251,
                0.60873782,
                0.61824492,
                0.62695713,
                0.63479877,
                0.64783359,
                0.65298429,
                0.66061823,
                0.66304938,
                0.66518197,
                0.66484485,
                0.66357081,
                0.66133763,
                0.65814578,
                0.65395601,
                0.64244274,
                0.62658196,
                0.60625719,
                0.5944127,
                0.56724056,
                0.55188592,
                0.53523524,
                0.51726329,
                0.49803127,
                0.47757039,
                0.45583332,
                0.43281069,
                0.4083933,
                0.38243443,
                0.32665841,
                0.2970316,
                0.2661739,
                0.23397908,
            ]
        )
        np.testing.assert_array_almost_equal(fdata_center.values[0], expected_values)

        expected_values = DenseValues(
            [
                0.53031768,
                0.53537618,
                0.53953078,
                0.54308738,
                0.54601671,
                0.54810052,
                0.54964395,
                0.55082904,
                0.55185083,
                0.55159705,
                0.55078707,
                0.54945877,
                0.54772366,
                0.54557748,
                0.54301563,
                0.54002675,
                0.53663784,
                0.51483315,
                0.50439313,
                0.49897162,
                0.49343211,
                0.48777439,
                0.47621104,
                0.47033297,
                0.45846482,
                0.4524936,
                0.44063979,
                0.43480054,
                0.42910479,
                0.423609,
                0.41831743,
                0.41320623,
                0.4083268,
                0.40360195,
                0.39897135,
                0.39455242,
                0.38221562,
                0.37852829,
                0.37517351,
                0.37216729,
                0.36957577,
                0.36741174,
                0.36558475,
                0.36398017,
                0.36261859,
                0.36147656,
                0.36040407,
                0.35864729,
                0.35801088,
                0.35762631,
                0.35747435,
                0.3575428,
                0.35775047,
                0.35807932,
                0.35903777,
                0.35965966,
                0.36109454,
                0.36190745,
                0.36275365,
                0.36362288,
                0.36450991,
                0.36538702,
                0.36624917,
                0.36705159,
                0.3677494,
                0.36831528,
                0.36871124,
                0.36891816,
                0.36869972,
                0.36826062,
                0.36657616,
                0.36529282,
                0.36359104,
                0.36144011,
                0.35889571,
                0.35598355,
                0.34888256,
                0.34456414,
                0.33954331,
                0.32193393,
                0.30817141,
            ]
        )
        np.testing.assert_array_almost_equal(fdata_center.values[9], expected_values)

    def test_center_1d_with_given_mean(self):
        precomputed_mean = self.fdata_sparse.mean()

        new_argvals = self.fdata_sparse.argvals.to_dense()
        bb = 1 / np.prod(new_argvals.n_points)
        fdata_center = self.fdata_sparse.center(mean=precomputed_mean, bandwidth=bb)
        self.assertIsInstance(fdata_center, IrregularFunctionalData)

        expected_values = DenseValues(
            [
                -0.64181022,
                -0.63146583,
                -0.62030816,
                -0.58099734,
                -0.5661533,
                -0.55023599,
                -0.53335645,
                -0.51568341,
                -0.49725285,
                -0.47817533,
                -0.4584701,
                -0.43808277,
                -0.41707468,
                -0.39550758,
                -0.37344991,
                -0.32797614,
                -0.30460999,
                -0.28085438,
                -0.23229342,
                -0.20755938,
                -0.18253189,
                -0.15721873,
                -0.13167712,
                -0.08011008,
                -0.02812959,
                -0.00209584,
                0.02391934,
                0.04985519,
                0.07569925,
                0.10142703,
                0.12699214,
                0.15241893,
                0.17770677,
                0.20280282,
                0.22762609,
                0.25217087,
                0.27630286,
                0.29990463,
                0.32303646,
                0.34559645,
                0.36748365,
                0.38869127,
                0.40925719,
                0.4484441,
                0.46704345,
                0.4849944,
                0.50221202,
                0.51858717,
                0.53414597,
                0.56261616,
                0.57543936,
                0.59844251,
                0.60873782,
                0.61824492,
                0.62695713,
                0.63479877,
                0.64783359,
                0.65298429,
                0.66061823,
                0.66304938,
                0.66518197,
                0.66484485,
                0.66357081,
                0.66133763,
                0.65814578,
                0.65395601,
                0.64244274,
                0.62658196,
                0.60625719,
                0.5944127,
                0.56724056,
                0.55188592,
                0.53523524,
                0.51726329,
                0.49803127,
                0.47757039,
                0.45583332,
                0.43281069,
                0.4083933,
                0.38243443,
                0.32665841,
                0.2970316,
                0.2661739,
                0.23397908,
            ]
        )
        np.testing.assert_array_almost_equal(fdata_center.values[0], expected_values)

        expected_values = DenseValues(
            [
                0.53031768,
                0.53537618,
                0.53953078,
                0.54308738,
                0.54601671,
                0.54810052,
                0.54964395,
                0.55082904,
                0.55185083,
                0.55159705,
                0.55078707,
                0.54945877,
                0.54772366,
                0.54557748,
                0.54301563,
                0.54002675,
                0.53663784,
                0.51483315,
                0.50439313,
                0.49897162,
                0.49343211,
                0.48777439,
                0.47621104,
                0.47033297,
                0.45846482,
                0.4524936,
                0.44063979,
                0.43480054,
                0.42910479,
                0.423609,
                0.41831743,
                0.41320623,
                0.4083268,
                0.40360195,
                0.39897135,
                0.39455242,
                0.38221562,
                0.37852829,
                0.37517351,
                0.37216729,
                0.36957577,
                0.36741174,
                0.36558475,
                0.36398017,
                0.36261859,
                0.36147656,
                0.36040407,
                0.35864729,
                0.35801088,
                0.35762631,
                0.35747435,
                0.3575428,
                0.35775047,
                0.35807932,
                0.35903777,
                0.35965966,
                0.36109454,
                0.36190745,
                0.36275365,
                0.36362288,
                0.36450991,
                0.36538702,
                0.36624917,
                0.36705159,
                0.3677494,
                0.36831528,
                0.36871124,
                0.36891816,
                0.36869972,
                0.36826062,
                0.36657616,
                0.36529282,
                0.36359104,
                0.36144011,
                0.35889571,
                0.35598355,
                0.34888256,
                0.34456414,
                0.33954331,
                0.32193393,
                0.30817141,
            ]
        )
        np.testing.assert_array_almost_equal(fdata_center.values[9], expected_values)


class TestNormIrregular(unittest.TestCase):
    def setUp(self):
        argvals = {
            0: DenseArgvals({"input_dim_0": np.array([0, 1, 2])}),
            1: DenseArgvals({"input_dim_0": np.array([0, 1, 2])}),
        }
        X = {0: np.array([0, 1, 4]), 1: np.array([0, 1, np.sqrt(2)])}
        self.fdata_sparse = IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(X)
        )

        argvals = {
            0: DenseArgvals(
                {"input_dim_0": np.array([0, 1]), "input_dim_1": np.array([0, 1])}
            )
        }
        X = {0: np.array([[0, 1], [1, 2]])}
        self.fdata_sparse_2D = IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(X)
        )

    def test_norm_1d(self):
        res = self.fdata_sparse.norm()
        expected_results = np.array([3, np.sqrt(2)])
        np.testing.assert_array_almost_equal(res, expected_results)

    def test_norm_1d_stand(self):
        results = self.fdata_sparse.norm(use_argvals_stand=True)
        expected_results = np.array([np.sqrt(4.5), 1])
        np.testing.assert_almost_equal(results, expected_results)

    def test_norm_1d_squared(self):
        results = self.fdata_sparse.norm(squared=True)
        expected_results = np.array([9, 2])
        np.testing.assert_almost_equal(results, expected_results)

    def test_norm_2d(self):
        results = self.fdata_sparse_2D.norm()
        expected_results = np.array([np.sqrt(1.5)])
        np.testing.assert_almost_equal(results, expected_results)


class TestNormalizeIrregular(unittest.TestCase):
    def setUp(self):
        argvals = {
            0: DenseArgvals({"input_dim_0": np.array([0, 1, 2])}),
            1: DenseArgvals({"input_dim_0": np.array([0, 1, 2])}),
        }
        X = {0: np.array([0, 1, 4]), 1: np.array([0, 1, np.sqrt(2)])}
        self.fdata_sparse = IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(X)
        )

        argvals = {
            0: DenseArgvals(
                {"input_dim_0": np.array([0, 1]), "input_dim_1": np.array([0, 1])}
            )
        }
        X = {0: np.array([[0, 1], [1, 2]])}
        self.fdata_sparse_2D = IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(X)
        )

    def test_normalize_1d(self):
        results = self.fdata_sparse.normalize()

        expected_results = IrregularValues(
            {0: np.array([0, 1 / 3, 4 / 3]), 1: np.array([0, 1 / np.sqrt(2), 1])}
        )
        np.testing.assert_allclose(results.values, expected_results)

    def test_normalize_2d(self):
        results = self.fdata_sparse_2D.normalize()

        expected_results = IrregularValues(
            {0: np.array([[0, 1 / np.sqrt(1.5)], [1 / np.sqrt(1.5), 2 / np.sqrt(1.5)]])}
        )
        np.testing.assert_allclose(results.values, expected_results)


class TestStandardizeIrregular(unittest.TestCase):
    def setUp(self):
        argvals = {
            0: DenseArgvals({"input_dim_0": np.array([0, 1, 2])}),
            1: DenseArgvals({"input_dim_0": np.array([0, 1, 2])}),
        }
        X = {0: np.array([0, 1, 4]), 1: np.array([0, 1, np.sqrt(2)])}
        self.fdata_sparse = IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(X)
        )

    def test_standardize_1d(self):
        results = self.fdata_sparse.standardize(remove_diagonal=False)

        expected_results = IrregularValues(
            {0: np.array([0, 0, 1]), 1: np.array([0, 0, -1])}
        )
        np.testing.assert_allclose(results.values, expected_results)


class TestRescaleIrregular(unittest.TestCase):
    def setUp(self):
        argvals = {
            0: DenseArgvals({"input_dim_0": np.array([0, 1, 2])}),
            1: DenseArgvals({"input_dim_0": np.array([0, 1, 2])}),
        }
        X = {0: np.array([0, 1, 4]), 1: np.array([0, 1, np.sqrt(2)])}
        self.fdata_sparse = IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(X)
        )

        argvals = {
            0: DenseArgvals(
                {"input_dim_0": np.array([0, 1]), "input_dim_1": np.array([0, 1])}
            ),
            1: DenseArgvals(
                {"input_dim_0": np.array([0, 1]), "input_dim_1": np.array([0, 1])}
            ),
        }
        X = {0: np.array([[0, 1], [1, 2]]), 1: np.array([[-1, 3], [6, 2]])}
        self.fdata_sparse_2D = IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(X)
        )

    def test_rescale_1d(self):
        results, _ = self.fdata_sparse.rescale()

        expected_results = IrregularValues(
            {
                0: np.array([0.0, 1.09383632, 4.37534529]),
                1: np.array([0.0, 1.09383632, 1.54691816]),
            }
        )
        np.testing.assert_allclose(results.values, expected_results)

    def test_rescale_1d_given_weights(self):
        results, weights = self.fdata_sparse.rescale(weights=1)

        expected_results = expected_results = IrregularValues(
            {0: np.array([0, 1, 4]), 1: np.array([0, 1, np.sqrt(2)])}
        )
        np.testing.assert_allclose(results.values, expected_results)

        expected_weights = 1
        np.testing.assert_equal(weights, expected_weights)

    def test_rescale_1d_use_argvals_stand(self):
        results, _ = self.fdata_sparse.rescale(use_argvals_stand=True)

        expected_results = IrregularValues(
            {
                0: DenseValues([0.0, 1.54691816, 6.18767264]),
                1: DenseValues([0.0, 1.54691816, 2.18767264]),
            }
        )
        np.testing.assert_allclose(results.values, expected_results)

    def test_rescale_2d(self):
        results, _ = self.fdata_sparse_2D.rescale(method_smoothing="LP")

        expected_results = IrregularValues(
            {
                0: np.array([[0.0, 0.73029674], [0.73029674, 1.46059349]]),
                1: np.array([[-0.73029674, 2.19089023], [4.38178046, 1.46059349]]),
            }
        )
        np.testing.assert_allclose(results.values, expected_results)


class TestCovariance(unittest.TestCase):
    def setUp(self):
        fname = THIS_DIR.parent / "data/data_sparse_5_10_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)

    def test_covariance_1d(self):
        self.fdata_sparse.covariance()

        expected_cov = np.array(
            [
                0.15472411,
                0.14855848,
                0.14454352,
                0.14209262,
                0.14081867,
                0.14034563,
                0.14044741,
                0.14034013,
                0.13951657,
                0.13789322,
                0.13548868,
                0.13251866,
                0.12893286,
                0.12500313,
                0.12101261,
                0.11667988,
                0.11198565,
                0.10703057,
                0.10208075,
                0.0975719,
                0.09360273,
                0.08994467,
                0.08667802,
                0.08362184,
                0.08077714,
                0.07843426,
                0.07657831,
                0.07482117,
                0.07291144,
                0.07111405,
                0.0694995,
                0.06761433,
                0.06512602,
                0.062037,
                0.0584792,
                0.05452101,
                0.05044692,
                0.04647898,
                0.04252055,
                0.03854402,
                0.03455831,
                0.030547,
                0.02653045,
                0.02278495,
                0.01977571,
                0.01759656,
                0.01629781,
                0.01568603,
                0.01537593,
                0.01534242,
                0.01569544,
                0.01656671,
                0.017817,
                0.01895484,
                0.01990127,
                0.02071034,
                0.02116685,
                0.02107946,
                0.02070206,
                0.02007342,
                0.01925561,
                0.01868937,
                0.01840727,
                0.01824926,
                0.01840311,
                0.01893859,
                0.01984927,
                0.02123337,
                0.02314564,
                0.02542681,
                0.02801345,
                0.03065551,
                0.03299675,
                0.03504692,
                0.03696051,
                0.03897061,
                0.04105642,
                0.04323314,
                0.04541497,
                0.04720705,
                0.04856793,
                0.04933011,
                0.04969371,
                0.05008922,
                0.05043764,
                0.0509771,
                0.05164393,
                0.05219735,
                0.05266391,
                0.05313121,
                0.05371643,
                0.05442826,
                0.05535529,
                0.05640362,
                0.0574211,
                0.05865464,
                0.06027855,
                0.06191592,
                0.06363343,
                0.06521477,
                0.06644915,
            ]
        )
        np.testing.assert_array_almost_equal(
            self.fdata_sparse._covariance.values[0, 1], expected_cov
        )

        expected_noise = 0.0
        np.testing.assert_almost_equal(
            self.fdata_sparse._noise_variance_cov, expected_noise
        )

    def test_covariance_1d_points(self):
        points = DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)})
        self.fdata_sparse.covariance(points=points)

        expected_cov = np.array(
            [
                0.1385776,
                0.13040424,
                0.10086206,
                0.08711904,
                0.06104903,
                0.04737855,
                0.04940163,
                0.05046211,
                0.06321718,
                0.05957814,
                0.0524317,
            ]
        )
        np.testing.assert_array_almost_equal(
            self.fdata_sparse._covariance.values[0, 1], expected_cov
        )

        expected_noise = 0.0
        np.testing.assert_almost_equal(
            self.fdata_sparse._noise_variance_cov, expected_noise
        )

    def test_covariance_2d(self):
        argvals = IrregularArgvals(
            {
                0: DenseArgvals(
                    {
                        "input_dim_0": np.array([1, 2, 3, 4]),
                        "input_dim_1": np.array([5, 6, 7]),
                    }
                ),
                1: DenseArgvals(
                    {
                        "input_dim_0": np.array([2, 4]),
                        "input_dim_1": np.array([1, 2, 3]),
                    }
                ),
                2: DenseArgvals(
                    {
                        "input_dim_0": np.array([4, 5, 6]),
                        "input_dim_1": np.array([8, 9]),
                    }
                ),
            }
        )
        values = IrregularValues(
            {
                0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
                1: np.array([[1, 2, 3], [1, 2, 3]]),
                2: np.array([[8, 9], [8, 9], [8, 9]]),
            }
        )
        data = IrregularFunctionalData(argvals, values)

        with self.assertRaises(NotImplementedError):
            data.covariance()
