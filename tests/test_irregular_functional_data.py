#!/usr/bin/python3
# -*-coding:utf8 -*

import numpy as np
import pandas as pd
import unittest

from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
from FDApy.representation.values import DenseValues, IrregularValues
from FDApy.representation.functional_data import (
    FunctionalData,
    DenseFunctionalData,
    IrregularFunctionalData
)
from FDApy.simulation.karhunen import KarhunenLoeve


class TestIrregularFunctionalData(unittest.TestCase):
    def setUp(self):
        self.argvals = IrregularArgvals({
            0: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4])}),
            1: DenseArgvals({'input_dim_0': np.array([0, 2, 4])}),
            2: DenseArgvals({'input_dim_0': np.array([2, 4])}),
        })
        self.values = IrregularValues({
            0: np.array([1, 2, 3, 4, 5]),
            1: np.array([2, 5, 6]),
            2: np.array([4, 7]),
        })
        self.fdata = IrregularFunctionalData(self.argvals, self.values)

        self.dense_argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 5])})
        self.dense_values = DenseValues(np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ]))
        self.dense_data = DenseFunctionalData(self.dense_argvals, self.dense_values)

    def test_get_item_slice(self):
        fdata = self.fdata[1:3]
        self.assertIsInstance(fdata, IrregularFunctionalData)
        self.assertEqual(fdata.n_obs, 2)
        self.assertEqual(fdata.n_dimension, 1)
        np.testing.assert_equal(fdata.argvals[1]['input_dim_0'], self.argvals[1]['input_dim_0'])
        np.testing.assert_equal(fdata.argvals[2]['input_dim_0'], self.argvals[2]['input_dim_0'])
        np.testing.assert_equal(fdata.values[1], self.values[1])

    def test_get_item_index(self):
        fdata = self.fdata[1]
        self.assertIsInstance(fdata, IrregularFunctionalData)
        self.assertEqual(fdata.n_obs, 1)
        self.assertEqual(fdata.n_dimension, 1)
        np.testing.assert_equal(fdata.argvals[1]['input_dim_0'], self.argvals[1]['input_dim_0'])
        np.testing.assert_equal(fdata.values[1], self.values[1])

    def test_argvals_getter(self):
        argvals = self.fdata.argvals
        np.testing.assert_equal(argvals, self.argvals)

    def test_argvals_setter(self):
        new_argvals = IrregularArgvals({
            0: DenseArgvals({'input_dim_0': np.array([5, 6, 7, 8, 9])}),
            1: DenseArgvals({'input_dim_0': np.array([6, 8, 10])}),
            2: DenseArgvals({'input_dim_0': np.array([6, 8])}),
        })
        self.fdata.argvals = new_argvals
        np.testing.assert_equal(self.fdata._argvals, new_argvals)

        expected_argvals_stand = IrregularArgvals({
            0: DenseArgvals({'input_dim_0': np.array([0, 0.2, 0.4, 0.6, 0.8])}),
            1: DenseArgvals({'input_dim_0': np.array([0.2, 0.6, 1])}),
            2: DenseArgvals({'input_dim_0': np.array([0.2, 0.6])}),
        })
        np.testing.assert_equal(self.fdata.argvals_stand, expected_argvals_stand)

        with self.assertRaises(TypeError):
            self.fdata.argvals = 0

    def test_values_property(self):
        values = self.fdata.values
        np.testing.assert_array_equal(values, self.values)

    def test_values_setter(self):
        new_values = IrregularValues({
            0: np.array([1, 4, 3, 4, 9]),
            1: np.array([1, 5, 3]),
            2: np.array([7, 7]),
        })
        self.fdata.values = new_values
        np.testing.assert_array_equal(self.fdata.values, new_values)

        with self.assertRaises(TypeError):
            self.fdata.values = 0

    def test_n_points(self):
        expected_n_points = {0: (5,), 1: (3, ), 2: (2, )}
        self.assertDictEqual(self.fdata.n_points, expected_n_points)

    def test_as_dense(self):
        dense_fdata = self.fdata.as_dense()
        expected_argvals = {'input_dim_0': np.arange(5)}
        expected_values = np.array([
            [1, 2, 3, 4, 5],
            [2, np.nan, 5, np.nan, 6],
            [np.nan, np.nan, 4, np.nan, 7]
        ])
        np.testing.assert_array_equal(
            dense_fdata.argvals['input_dim_0'],
            expected_argvals['input_dim_0']
        )
        np.testing.assert_array_equal(
            dense_fdata.values,
            expected_values
        )

    def test_is_compatible(self):
        IrregularFunctionalData._is_compatible(self.fdata, self.fdata)
        self.assertTrue(True)

    def test_non_compatible_type(self):
        with self.assertRaises(TypeError):
            IrregularFunctionalData._is_compatible(self.fdata, self.dense_data)

    def test_non_compatible_nobs(self):
        argvals = IrregularArgvals({
            0: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4])}),
            1: DenseArgvals({'input_dim_0': np.array([0, 2, 4])}),
        })
        values = IrregularValues({
            0: np.array([1, 2, 3, 4, 5]),
            1: np.array([2, 5, 6]),
        })
        func_data = IrregularFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            IrregularFunctionalData._is_compatible(self.fdata, func_data)

    def test_non_compatible_ndim(self):
        argvals = IrregularArgvals({
            0: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4]), 'input_dim_1': np.array([5, 6, 7])}),
            1: DenseArgvals({'input_dim_0': np.array([0, 2, 4]), 'input_dim_1': np.array([1, 2, 3])}),
            2: DenseArgvals({'input_dim_0': np.array([2, 4]), 'input_dim_1': np.array([1, 2])})
        })
        values = IrregularValues({
            0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4], [1, 2, 4]]),
            1: np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            2: np.array([[1, 2], [3, 4]])
        })
        func_data = IrregularFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            IrregularFunctionalData._is_compatible(self.fdata, func_data)

    def test_concatenate(self):
        fdata = IrregularFunctionalData.concatenate(self.fdata, self.fdata)

        expected_argvals = IrregularArgvals({
            0: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4])}),
            1: DenseArgvals({'input_dim_0': np.array([0, 2, 4])}),
            2: DenseArgvals({'input_dim_0': np.array([2, 4])}),
            3: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4])}),
            4: DenseArgvals({'input_dim_0': np.array([0, 2, 4])}),
            5: DenseArgvals({'input_dim_0': np.array([2, 4])}),
        })
        expected_values = IrregularValues({
            0: np.array([1, 2, 3, 4, 5]),
            1: np.array([2, 5, 6]),
            2: np.array([4, 7]),
            3: np.array([1, 2, 3, 4, 5]),
            4: np.array([2, 5, 6]),
            5: np.array([4, 7]),
        })

        self.assertIsInstance(fdata, IrregularFunctionalData)
        np.testing.assert_allclose(fdata.argvals, expected_argvals)
        np.testing.assert_allclose(fdata.values, expected_values)

    def test_to_long(self):
        result = self.fdata.to_long()

        expected_dim = np.array([0, 1, 2, 3, 4, 0, 2, 4, 2, 4])
        expected_id = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
        expected_values = DenseValues(np.array([1, 2, 3, 4, 5, 2, 5, 6, 4, 7]))

        self.assertTrue(isinstance(result, pd.DataFrame))
        np.testing.assert_array_equal(result['input_dim_0'].values, expected_dim)
        np.testing.assert_array_equal(result['id'].values, expected_id)
        np.testing.assert_array_equal(result['values'].values, expected_values)


class TestIrregularFunctionalData1D(unittest.TestCase):
    """Test class for the class IrregularFunctionalData in one dimension."""

    def setUp(self):
        self.argvals = {
            0: DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4])}),
            1: DenseArgvals({'input_dim_0': np.array([2, 4])}),
            2: DenseArgvals({'input_dim_0': np.array([0, 2, 3])}),
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
                self.irregu_fd.argvals_stand[0]['input_dim_0'],
                np.array([0.25, 0.5, 0.75, 1.])
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[1]['input_dim_0'],
                np.array([0.5, 1.])
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[2]['input_dim_0'],
                np.array([0., 0.5, 0.75])
            )
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

    def test_as_dense(self):
        dense_fd = self.irregu_fd.as_dense()
        self.assertIsInstance(dense_fd, DenseFunctionalData)
        self.assertEqual(dense_fd.n_obs, 3)

    def test_is_compatible(self):
        IrregularFunctionalData._is_compatible(self.irregu_fd, self.irregu_fd)
        self.assertTrue(True)

    def test_mean(self):
        mean_fd = self.irregu_fd.mean()
        is_equal = np.allclose(mean_fd.values,
                               np.array([[8., 1., 5.33333333, 5., 5.]]))
        self.assertTrue(is_equal)


class TestIrregularFunctionalData2D(unittest.TestCase):
    """Test class for the class IrregularFunctionalData in two dimension."""

    def setUp(self):
        argvals = {
            0: DenseArgvals({
                'input_dim_0': np.array([1, 2, 3, 4]),
                'input_dim_1': np.array([5, 6, 7])
            }),
            1: DenseArgvals({
                'input_dim_0': np.array([2, 4]),
                'input_dim_1': np.array([1, 2, 3])
            }),
            2: DenseArgvals({
                'input_dim_0': np.array([4, 5, 6]),
                'input_dim_1': np.array([8, 9])
            })
        }
        values = {
            0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
            1: np.array([[1, 2, 3], [1, 2, 3]]),
            2: np.array([[8, 9], [8, 9], [8, 9]])
        }
        self.irregu_fd = IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(values)
        )

    def test_argvals_stand(self):
        is_equal = [
            np.allclose(
                self.irregu_fd.argvals_stand[0]['input_dim_0'],
                np.array([0., 0.2, 0.4, 0.6])
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[1]['input_dim_0'],
                np.array([0.2, 0.6])
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[2]['input_dim_0'],
                np.array([0.6, 0.8, 1.])
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[0]['input_dim_1'],
                np.array([0.5, 0.625, 0.75])
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[1]['input_dim_1'],
                np.array([0., 0.125, 0.25])
            ),
            np.allclose(
                self.irregu_fd.argvals_stand[2]['input_dim_1'],
                np.array([0.875, 1.])
            )
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

    def test_as_dense(self):
        dense_fd = self.irregu_fd.as_dense()
        self.assertIsInstance(dense_fd, DenseFunctionalData)
        self.assertEqual(dense_fd.n_obs, 3)

    def test_is_compatible(self):
        IrregularFunctionalData._is_compatible(self.irregu_fd, self.irregu_fd)
        self.assertTrue(True)

    def test_to_long(self):
        result = self.irregu_fd.to_long()

        expected_dim_0 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 5, 5, 6, 6])
        expected_dim_1 = np.array([5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 1, 2, 3, 1, 2, 3, 8, 9, 8, 9, 8, 9])
        expected_id = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        expected_values = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 8, 9, 8, 9, 8, 9])

        self.assertTrue(isinstance(result, pd.DataFrame))
        np.testing.assert_array_equal(result['input_dim_0'].values, expected_dim_0)
        np.testing.assert_array_equal(result['input_dim_1'].values, expected_dim_1)
        np.testing.assert_array_equal(result['id'].values, expected_id)
        np.testing.assert_array_equal(result['values'].values, expected_values)

    def test_mean(self):
        N = np.nan
        mean_fd = self.irregu_fd.mean()
        is_equal = np.allclose(mean_fd.values,
                               np.array([[[N, N, N, 1., 2., 3., N, N],
                                          [1., 2., 3., 4., 1., 2., N, N],
                                          [N, N, N, 3., 4., 1., N, N],
                                          [1., 2., 3., 2., 3., 4., 8., 9.],
                                          [N, N, N, N, N, N, 8., 9.],
                                          [N, N, N, N, N, N, 8., 9.]]]),
                               equal_nan=True)
        self.assertTrue(is_equal)


class TestPerformComputation(unittest.TestCase):
    def setUp(self):
        self.argvals = {
            0: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4])}),
            1: DenseArgvals({'input_dim_0': np.array([0, 2, 4])}),
            2: DenseArgvals({'input_dim_0': np.array([2, 4])})
        }
        self.values1 = {
            0: np.array([1, 2, 3, 4, 5]),
            1: np.array([2, 5, 6]),
            2: np.array([4, 7])
        }
        self.func_data1 = IrregularFunctionalData(IrregularArgvals(self.argvals), IrregularValues(self.values1))

        self.values2 = {
            0: np.array([5, 4, 3, 2, 1]),
            1: np.array([5, 3, 1]),
            2: np.array([5, 3])
        }
        self.func_data2 = IrregularFunctionalData(IrregularArgvals(self.argvals), IrregularValues(self.values2))

    def test_addition(self):
        result = self.func_data1 + self.func_data2

        expected_values = IrregularValues({0: np.array([6, 6, 6, 6, 6]), 1: np.array([7, 8, 7]), 2: np.array([ 9, 10])})
        np.testing.assert_equal(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_substraction(self):
        result = self.func_data1 - self.func_data2

        expected_values = IrregularValues({0: np.array([-4, -2,  0,  2,  4]), 1: np.array([-3,  2,  5]), 2: np.array([-1,  4])})
        self.assertEqual(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_multiplication(self):
        result = self.func_data1 * self.func_data2

        expected_values = IrregularValues({0: np.array([5, 8, 9, 8, 5]), 1: np.array([10, 15,  6]), 2: np.array([20, 21])})
        self.assertEqual(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_right_multiplication(self):
        result = FunctionalData.__rmul__(self.func_data1, self.func_data2)

        expected_values = IrregularValues({0: np.array([5, 8, 9, 8, 5]), 1: np.array([10, 15,  6]), 2: np.array([20, 21])})
        self.assertEqual(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_true_divide(self):
        result = self.func_data1 / self.func_data2

        expected_values = IrregularValues({0: np.array([0.2, 0.5, 1. , 2. , 5. ]), 1: np.array([0.4, 1.66666667, 6.]), 2: np.array([0.8, 2.33333333])})
        self.assertEqual(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_almost_equal(result.values, expected_values)

    def test_floor_divide(self):
        result = self.func_data1 // self.func_data2

        expected_values = IrregularValues({0: np.array([0, 0, 1, 2, 5]), 1: np.array([0, 1, 6]), 2: np.array([0, 2])})
        self.assertEqual(result.argvals, IrregularArgvals(self.argvals))
        np.testing.assert_array_almost_equal(result.values, expected_values)


class TestSmoothIrregular(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5

        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=1)
        kl.add_noise_and_sparsify(0.05, 0.5)
        self.fdata_1d = kl.sparse_data

    def test_smooth_1d(self):
        fdata_smooth = self.fdata_1d.smooth()
        print(self.fdata_1d.values)

        expected_values = DenseValues([[-0.19645506, -0.21794753, -0.23989429, -0.24048419,  -0.23790629, -0.23440484, -0.22994697, -0.22468187,  -0.21201513, -0.20394813, -0.1747512 , -0.12601596,  -0.09895502, -0.08492093, -0.04109609, -0.02596334,  -0.01082828,  0.00421961,  0.04567071,  0.05745157,   0.06811439,  0.11375887,  0.12281441,  0.13186628,   0.14994229,  0.16348497,  0.16780846,  0.17302711,   0.17348835,  0.17186662,  0.14886201,  0.13550609,   0.12792604,  0.10153141,  0.05514138, -0.12873871,  -0.18394248, -0.21406013, -0.24600976, -0.28000348,  -0.31583867, -0.35312089, -0.81082022, -0.95808598,  -1.21479802, -1.40218008]])
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)
