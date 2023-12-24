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
        expected_mean = DenseValues(np.array([[
            [0., 0., 0., 1., 2., 3., 0., 0.],
            [1., 2., 3., 4., 1., 2., 0., 0.],
            [0., 0., 0., 3., 4., 1., 0., 0.],
            [1., 2., 3., 2., 3., 4., 8., 9.],
            [0., 0., 0., 0., 0., 0., 8., 9.],
            [0., 0., 0., 0., 0., 0., 8., 9.]
        ]]))
        np.testing.assert_allclose(mean_fd.values, expected_mean)


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


class TestNoisevariance(unittest.TestCase):
    def setUp(self) -> None:
        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=100)
        kl.sparsify(0.5)
        self.fdata = kl.sparse_data

    def test_noise_variance(self):
        res = self.fdata.noise_variance(order=2)
        expected_res = 0.006805102395353494
        np.testing.assert_almost_equal(res, expected_res)

    def test_noise_variance_error(self):
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
        irregu_fd = IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(values)
        )

        with self.assertWarns(UserWarning):
            irregu_fd.noise_variance(2)


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

    def test_smooth_2d(self):
        argvals = IrregularArgvals({
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
        })
        values = IrregularValues({
            0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
            1: np.array([[1, 2, 3], [1, 2, 3]]),
            2: np.array([[8, 9], [8, 9], [8, 9]])
        })
        data = IrregularFunctionalData(argvals, values)
        fdata_smooth = data.smooth()

        expected_values = DenseValues([[[0., 0., 0., 1., 2., 3., 0., 0.],[0., 0., 0., 4., 1., 2., 0., 0.],[0., 0., 0., 3., 4., 1., 0., 0.],[0., 0., 0., 2., 3., 4., 0., 0.],[0., 0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0., 0.]],[[0., 0., 0., 0., 0., 0., 0., 0.],[1., 2., 3., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0., 0.],[1., 2., 3., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0., 0.]],[[0., 0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 8., 9.],[0., 0., 0., 0., 0., 0., 8., 9.],[0., 0., 0., 0., 0., 0., 8., 9.]]])
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)


class TestInnerProductIrregular(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=10)
        kl.sparsify(percentage=0.8, epsilon=0.05)
        self.data = kl.sparse_data

    def test_inner_product(self):
        res = self.data.inner_product()
        expected_res = np.array([[ 2.87431056e-01,  1.27349521e-01, -7.86126924e-02,  2.93881793e-02,  1.25888272e-01, -2.45177301e-02, -1.47752236e-01, -1.35725306e-01, -7.24520965e-02, -2.91464666e-02],[ 1.27349521e-01,  2.68157146e-01, -2.48690884e-01,  1.09780523e-01,  1.01197590e-01, -4.35876576e-02, -1.09921910e-01, -1.31249011e-04, -7.26042877e-02, -1.29449187e-01],[-7.86126924e-02, -2.48690884e-01,  3.29198255e-01, -6.45841037e-02, -5.52733779e-02,  1.33100461e-01, -8.61587305e-02, -1.66483011e-01,  3.02852376e-02,  2.13139470e-01],[ 2.93881793e-02,  1.09780523e-01, -6.45841037e-02,  1.10719121e-01,  3.86106394e-02,  2.26313667e-02, -9.44364209e-02, -2.62613470e-02, -3.97532199e-02, -4.35904516e-02],[ 1.25888272e-01,  1.01197590e-01, -5.52733779e-02,  3.86106394e-02,  8.99205832e-02,  2.04656058e-02, -1.38489153e-01, -1.01082742e-01, -6.33965118e-02, -1.15509009e-02],[-2.45177301e-02, -4.35876576e-02,  1.33100461e-01,  2.26313667e-02,  2.04656058e-02,  1.05952957e-01, -1.60566463e-01, -1.48235426e-01, -2.76878543e-02,  1.06839268e-01],[-1.47752236e-01, -1.09921910e-01, -8.61587305e-02, -9.44364209e-02, -1.38489153e-01, -1.60566463e-01,  4.10497228e-01,  3.32971770e-01,  1.13386861e-01, -1.25727486e-01],[-1.35725306e-01, -1.31249011e-04, -1.66483011e-01, -2.62613470e-02, -1.01082742e-01, -1.48235426e-01,  3.32971770e-01,  3.14640310e-01,  7.89510687e-02, -1.62498483e-01],[-7.24520965e-02, -7.26042877e-02,  3.02852376e-02, -3.97532199e-02, -6.33965118e-02, -2.76878543e-02,  1.13386861e-01,  7.89510687e-02,  4.49509294e-02,  1.92015305e-03],[-2.91464666e-02, -1.29449187e-01,  2.13139470e-01, -4.35904516e-02, -1.15509009e-02,  1.06839268e-01, -1.25727486e-01, -1.62498483e-01,  1.92015305e-03,  1.58875881e-01]])
        np.testing.assert_array_almost_equal(res, expected_res)

    def test_inner_product_with_unkwon_noise_variance(self):
        res = self.data.inner_product(noise_variance=0)
        expected_res = np.array([[ 2.89643085e-01,  1.27349521e-01, -7.86126924e-02,  2.93881793e-02,  1.25888272e-01, -2.45177301e-02, -1.47752236e-01, -1.35725306e-01, -7.24520965e-02, -2.91464666e-02],[ 1.27349521e-01,  2.70369175e-01, -2.48690884e-01,  1.09780523e-01,  1.01197590e-01, -4.35876576e-02, -1.09921910e-01, -1.31249011e-04, -7.26042877e-02, -1.29449187e-01],[-7.86126924e-02, -2.48690884e-01,  3.31410284e-01, -6.45841037e-02, -5.52733779e-02,  1.33100461e-01, -8.61587305e-02, -1.66483011e-01,  3.02852376e-02,  2.13139470e-01],[ 2.93881793e-02,  1.09780523e-01, -6.45841037e-02,  1.12931149e-01,  3.86106394e-02,  2.26313667e-02, -9.44364209e-02, -2.62613470e-02, -3.97532199e-02, -4.35904516e-02],[ 1.25888272e-01,  1.01197590e-01, -5.52733779e-02,  3.86106394e-02,  9.21326118e-02,  2.04656058e-02, -1.38489153e-01, -1.01082742e-01, -6.33965118e-02, -1.15509009e-02],[-2.45177301e-02, -4.35876576e-02,  1.33100461e-01,  2.26313667e-02,  2.04656058e-02,  1.08164986e-01, -1.60566463e-01, -1.48235426e-01, -2.76878543e-02,  1.06839268e-01],[-1.47752236e-01, -1.09921910e-01, -8.61587305e-02, -9.44364209e-02, -1.38489153e-01, -1.60566463e-01,  4.12709257e-01,  3.32971770e-01,  1.13386861e-01, -1.25727486e-01],[-1.35725306e-01, -1.31249011e-04, -1.66483011e-01, -2.62613470e-02, -1.01082742e-01, -1.48235426e-01,  3.32971770e-01,  3.16852339e-01,  7.89510687e-02, -1.62498483e-01],[-7.24520965e-02, -7.26042877e-02,  3.02852376e-02, -3.97532199e-02, -6.33965118e-02, -2.76878543e-02,  1.13386861e-01,  7.89510687e-02,  4.71629580e-02,  1.92015305e-03],[-2.91464666e-02, -1.29449187e-01,  2.13139470e-01, -4.35904516e-02, -1.15509009e-02,  1.06839268e-01, -1.25727486e-01, -1.62498483e-01,  1.92015305e-03,  1.61087910e-01]])
        np.testing.assert_array_almost_equal(res, expected_res)

    def test_inner_product_2d(self):
        argvals = IrregularArgvals({
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
        })
        values = IrregularValues({
            0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
            1: np.array([[1, 2, 3], [1, 2, 3]]),
            2: np.array([[8, 9], [8, 9], [8, 9]])
        })
        data = IrregularFunctionalData(argvals, values)

        with self.assertRaises(NotImplementedError):
            data.inner_product()


class TestCenterDense(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5

        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=10)
        kl.add_noise_and_sparsify(0.01, 0.95)
        self.fdata_1d = kl.sparse_data

    def test_center_1d(self):
        fdata_center = self.fdata_1d.center()

        self.assertIsInstance(fdata_center, IrregularFunctionalData)

        expected_values = DenseValues([ 0.10438566,  0.0671942 , -0.20798275, -0.1555264 , -0.22701277, -0.29560242, -0.30637032, -0.1719037 , -0.44593386, -0.2962262 , -0.59100573, -0.48195829, -0.45409883, -0.42995399, -0.43209738, -0.43510004, -0.5572024 , -0.57326715, -0.4428713 , -0.54650734, -0.65085118, -0.62984952, -0.59917215, -0.44578824, -0.46723798, -0.39623002, -0.48976215, -0.41102478, -0.34235778, -0.41225955, -0.31054217, -0.39591921, -0.33825834, -0.36257115, -0.16340779, -0.22743484, -0.14699356, -0.06736396, -0.03743773,  0.01818149, -0.02431331, -0.02290194,  0.04537325, -0.08167476,  0.02116701,  0.2577794 ,  0.15839936,  0.43834243,  0.43744301,  0.29599486,  0.39298685,  0.34113607,  0.42319626,  0.5602522 ,  0.55207479,  0.54544164,  0.47288912,  0.39580635,  0.55311283,  0.49571809,  0.57182235,  0.54765269,  0.69948952,  0.50602086,  0.41497818,  0.51927536,  0.50393232,  0.59484562,  0.51567149,  0.43722516,  0.5133073 ,  0.20949221,  0.22164139,  0.14594531,  0.1482301 , -0.00633447,  0.13439169, -0.01649776, -0.21131538, -0.24068442, -0.1877294 , -0.22040845, -0.19488646, -0.19880851, -0.54979504, -0.79679154, -1.02329362, -0.90128145, -1.13312912, -1.22315553, -1.37876467, -1.4738207 , -1.50174103, -1.74760341, -1.94053565])
        np.testing.assert_array_almost_equal(fdata_center.values[0], expected_values)

        expected_values = DenseValues([0.03024885, 0.19574206, 0.08930958, 0.01581904, 0.14264388, 0.18532339, 0.08549022, 0.22746207, 0.20808386, 0.40202118, 0.33832925, 0.32993342, 0.36912185, 0.42180382, 0.47919198, 0.33388333, 0.34444214, 0.57227785, 0.43230687, 0.34215917, 0.53715998, 0.54673538, 0.35368249, 0.53487717, 0.58954092, 0.55641742, 0.58416391, 0.3815803 , 0.55528128, 0.49122701, 0.47173346, 0.66288612, 0.69100093, 0.63912071, 0.492262  , 0.53365075, 0.5698519 , 0.4998242 , 0.49524308, 0.55895514, 0.46497689, 0.39224272, 0.41814228, 0.51750946, 0.47783816, 0.5062069 , 0.39751741, 0.50696055, 0.46607244, 0.30217776, 0.56804715, 0.37245899, 0.31680688, 0.31921138, 0.42693744, 0.39501642, 0.389681  , 0.48653417, 0.4599204 , 0.46959446, 0.22016812, 0.39544265, 0.21456628, 0.43664859, 0.43272894, 0.53758619, 0.5513205 , 0.38049607, 0.47731492, 0.37373544, 0.42356584, 0.51890808, 0.49215161, 0.47393042, 0.49053699, 0.45528684, 0.38705165, 0.29114712, 0.27101146, 0.29169769, 0.22405083, 0.34891548, 0.16542701, 0.25250307, 0.26605896, 0.4806325 , 0.07814136, 0.3414786 , 0.18628597, 0.18797513, 0.13289027, 0.13841477])
        np.testing.assert_array_almost_equal(fdata_center.values[9], expected_values)

    def test_center_1d_with_given_mean(self):
        precomputed_mean = self.fdata_1d.mean()

        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=43
        )
        kl.new(n_obs=10)
        kl.add_noise_and_sparsify(0.01, 0.95)
        fdata_1d_new = kl.sparse_data

        fdata_center = fdata_1d_new.center(mean=precomputed_mean)
        self.assertIsInstance(fdata_center, IrregularFunctionalData)

        expected_values = DenseValues([-0.03830012,  0.11041963,  0.0133529 ,  0.13486167,  0.11601488,  0.02673493,  0.26161219,  0.05264892,  0.21947272,  0.21529531,  0.34548576,  0.11496152,  0.17869754,  0.22711824,  0.25744897,  0.32656903,  0.26941556,  0.23766012,  0.33322343,  0.43361173,  0.0805314 ,  0.20610473,  0.27180973,  0.19224299,  0.31367371,  0.12211837,  0.038296  ,  0.20980922, -0.17249251,  0.03825377,  0.30253514,  0.04777818,  0.05936786, -0.09381522, -0.02388574, -0.05931468,  0.11662645, -0.39495541, -0.10329284, -0.01430552, -0.26373148, -0.40295326, -0.35358216, -0.22864586, -0.35384616, -0.31886196, -0.58402877, -0.42517462, -0.31874351, -0.44804961, -0.54748107, -0.36278999, -0.39173361, -0.55905871, -0.63195494, -0.61432038, -0.43874177, -0.62088543, -0.70979504, -0.87869597, -0.6984956 , -0.73188039, -0.80702546, -0.67383268, -0.9176268 , -0.96924842, -0.91269204, -0.86267628, -1.03404308, -1.02619935, -1.01798303, -0.96478726, -1.05886662, -0.91808374, -1.26517565, -1.16220217, -1.08667737, -0.9666139 , -1.2094064 , -1.179404  , -1.26781805, -1.28970381, -1.42321509, -1.38381545, -1.60826748, -1.70678456, -1.52190155, -1.79211332, -1.72563241, -1.93438911, -1.91222869, -1.96850132])
        np.testing.assert_array_almost_equal(fdata_center.values[0], expected_values)

        expected_values = DenseValues([ 1.18903592,  1.30054853,  0.83387149,  0.91626648,  0.99512309,  0.72042696,  0.80899999,  0.78540731,  0.7086824 ,  0.68701057,  0.7369545 ,  0.7231971 ,  0.43901074,  0.41551093,  0.56201837,  0.37194754,  0.65485191,  0.42966761,  0.44122487,  0.3806152 ,  0.34342646,  0.2906656 ,  0.51810741,  0.37281023,  0.2514299 ,  0.3597168 ,  0.3415683 ,  0.41667987,  0.31618746,  0.26046766,  0.26611097,  0.33643347,  0.21864689,  0.25148888,  0.34318643,  0.30344429,  0.28205874,  0.38068737,  0.22745054,  0.36728697,  0.43476998,  0.31773075,  0.31571857,  0.22552548,  0.30119693,  0.27854707,  0.45155549,  0.33244839,  0.33683592,  0.48009732,  0.37095023,  0.44027704,  0.43571732,  0.3848733 ,  0.53611747,  0.4352297 ,  0.27569143,  0.37081442,  0.50412552,  0.45144068,  0.37449246,  0.30938863,  0.41584197,  0.34990566,  0.50450516,  0.3662828 ,  0.44356828,  0.34128634,  0.48874459,  0.26647934,  0.39917088,  0.49311288,  0.16517626,  0.16240897,  0.40505184,  0.307353  ,  0.26947685,  0.06383166,  0.16755697,  0.16137892,  0.09847752,  0.07263724,  0.08062353, -0.27724381, -0.27853154, -0.23697058, -0.15468214, -0.24974182, -0.39391868, -0.15069158, -0.38798605, -0.40460538, -0.39026313, -0.33243741])
        np.testing.assert_array_almost_equal(fdata_center.values[9], expected_values)


class TestNormIrregular(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=10)
        kl.sparsify(percentage=0.8, epsilon=0.05)
        self.data = kl.sparse_data

    def test_norm(self):
        res = self.data.norm()
        expected_res = np.array([0.53301318, 0.42249943, 0.67097958, 0.24237486, 0.27450066, 0.37765895, 0.65230671, 0.54117675, 0.28729627, 0.4933732 ])
        np.testing.assert_array_almost_equal(res, expected_res)

        res = self.data.norm(squared=True)
        expected_res = np.array([0.28410305, 0.17850577, 0.4502136 , 0.05874557, 0.07535061, 0.14262628, 0.42550404, 0.29287227, 0.08253915, 0.24341712])
        np.testing.assert_array_almost_equal(res, expected_res)

        res = self.data.norm(use_argvals_stand=True)
        expected_res = np.array([0.53301318, 0.42249943, 0.67097958, 0.24483558, 0.27450066, 0.37956153, 0.65559291, 0.54117675, 0.28729627, 0.4933732 ])
        np.testing.assert_array_almost_equal(res, expected_res)

    def test_norm_2d(self):
        argvals = IrregularArgvals({
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
        })
        values = IrregularValues({
            0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
            1: np.array([[1, 2, 3], [1, 2, 3]]),
            2: np.array([[8, 9], [8, 9], [8, 9]])
        })
        data = IrregularFunctionalData(argvals, values)

        res = data.norm()
        expected_res = np.array([ 6.78232998,  4.24264069, 12.04159458])
        np.testing.assert_array_almost_equal(res, expected_res)


class TestNormalizeIrregular(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=10)
        kl.sparsify(percentage=0.5, epsilon=0.05)
        self.data = kl.sparse_data

    def test_norm(self):
        res, weight = self.data.normalize()
        expected_weight = 0.16802008
        np.testing.assert_array_almost_equal(weight, expected_weight)
        self.assertIsInstance(res, IrregularFunctionalData)

        expected_values = np.array([  1.34925412,   0.91617373,   0.14071864,  -0.20316446,  -1.06848631,  -1.51430328,  -2.11853405,  -2.21411555,  -2.28901497,  -2.37978433,  -2.37967761,  -2.34632262,  -1.96359079,  -1.84913242,  -1.7237965 ,  -1.58833723,  -1.44350881,  -1.29006543,  -0.96035057,  -0.78558748,  -0.60522621,  -0.42002096,  -0.03809529,   0.35415597,   1.145501  ,   1.34036331,   2.42650329,   2.88252993,   3.14073215,   3.35538523,   3.63039808,   3.66269642,   3.67893493,   3.67837469,   3.66027676,   3.62390221,   3.49336752,   3.28085914,   3.14201749,   2.98046562,   2.79546459,   2.09237726,   0.78181069,  -0.0466806 ,  -1.52255352,  -2.67051369,  -4.65329994,  -5.38674336,  -6.15767567,  -8.70279527,  -9.63107248, -10.60053323, -11.61191648])
        np.testing.assert_array_almost_equal(res.values[0], expected_values)

        expected_values = np.array([1.30156224, 1.93967563, 2.30010235, 2.75042508, 2.99460299, 3.10095416, 3.42978895, 3.48964417, 3.62128624, 3.69834513, 3.70241234, 3.70111615, 3.69481434, 3.64945326, 3.54059975, 3.50713383, 3.43520351, 3.35899389, 3.28136737, 3.24291726, 3.16853235, 3.13331314, 3.06861021, 2.99126152, 2.95689447, 2.93689824, 2.92848729, 2.9383331 , 2.9441859 , 2.95079335, 2.95783697, 2.97195886, 2.98400379, 2.99142401, 2.99260368, 2.99167175, 2.98830976, 2.98219924, 2.94419176, 2.79615135, 2.64249761, 2.50774165, 2.34415719, 2.14919646, 1.79235162, 1.65495512, 1.18296239, 1.0046355 , 0.81527983, 0.40220828])
        np.testing.assert_array_almost_equal(res.values[9], expected_values)

        res, weight = self.data.normalize(use_argvals_stand=True)
        expected_weight = 0.16802008
        np.testing.assert_array_almost_equal(weight, expected_weight)
        self.assertIsInstance(res, IrregularFunctionalData)

    def test_norm_with_given_weights(self):
        res, weight = self.data.normalize(weights=1)
        expected_weight = 1
        np.testing.assert_array_almost_equal(weight, expected_weight)
        self.assertIsInstance(res, IrregularFunctionalData)

    def test_norm_2d(self):
        argvals = IrregularArgvals({
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
        })
        values = IrregularValues({
            0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
            1: np.array([[1, 2, 3], [1, 2, 3]]),
            2: np.array([[8, 9], [8, 9], [8, 9]])
        })
        data = IrregularFunctionalData(argvals, values)

        res, weight = data.normalize()
        expected_weight = 87.77777778
        np.testing.assert_array_almost_equal(weight, expected_weight)
        self.assertIsInstance(res, IrregularFunctionalData)

    def test_norm_2d_with_given_weights(self):
        argvals = IrregularArgvals({
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
        })
        values = IrregularValues({
            0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
            1: np.array([[1, 2, 3], [1, 2, 3]]),
            2: np.array([[8, 9], [8, 9], [8, 9]])
        })
        data = IrregularFunctionalData(argvals, values)

        res, weight = data.normalize(weights=1)
        expected_weight = 1
        np.testing.assert_array_almost_equal(weight, expected_weight)
        self.assertIsInstance(res, IrregularFunctionalData)


class TestCovariance(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=100)
        kl.sparsify(percentage=0.5, epsilon=0.05)
        self.data = kl.sparse_data

    def test_covariance_1d(self):
        self.data.covariance()

        expected_cov = np.array([ 0.46908558,  0.50183053,  0.5227941 ,  0.5317089 ,  0.53108778,  0.52308042,  0.51025201,  0.49428858,  0.47512707,  0.45338596,  0.42970782,  0.40409963,  0.37765384,  0.35024334,  0.32264314,  0.29495295,  0.26924584,  0.24584519,  0.22389991,  0.20381207,  0.18438351,  0.16545152,  0.14695299,  0.12976112,  0.11266059,  0.09546621,  0.07847628,  0.06203113,  0.04666676,  0.03296685,  0.01992275,  0.00812633, -0.00229425, -0.01128007, -0.01902733, -0.02595731, -0.03237438, -0.03827309, -0.04269715, -0.04620535, -0.04824466, -0.04981964, -0.05130755, -0.05341201, -0.05576454, -0.05852535, -0.06218736, -0.06689593, -0.07249118, -0.0780311 , -0.08371261, -0.09024318, -0.09768551, -0.10506797, -0.11219186, -0.11874241, -0.12425184, -0.12830129, -0.13093742, -0.13200077, -0.13131211, -0.12936798, -0.12622501, -0.1224811 , -0.11863617, -0.1144184 , -0.11010575, -0.10659236, -0.10365305, -0.10159059, -0.10027381, -0.09938413, -0.09966974, -0.10093416, -0.10343208, -0.10724144, -0.11178521, -0.11717679, -0.12263712, -0.12748813, -0.13053691, -0.13181844, -0.13283854, -0.13410203, -0.1346029 , -0.13512733, -0.13518738, -0.13504468, -0.13471035, -0.13345488, -0.13100382, -0.12697329, -0.1209482 , -0.11392364, -0.10663811, -0.09812225, -0.08841827, -0.07647037, -0.06213727, -0.04324978, -0.01771456])
        np.testing.assert_array_almost_equal(self.data._covariance.values[0, 1], expected_cov)

        expected_noise = 0.02312407511881056
        np.testing.assert_almost_equal(self.data._noise_variance_cov, expected_noise)

    def test_covariance_1d_points(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        self.data.covariance(points=points)

        expected_cov = np.array([ 0.39309119,  0.46122988,  0.35416482,  0.23828871, 0.15613467,  0.09909948,  0.02960858,  0.03170356, -0.00601762, -0.04725067, -0.00100574])
        np.testing.assert_array_almost_equal(self.data._covariance.values[0, 1], expected_cov)

        expected_noise = 0.018519423095566027
        np.testing.assert_almost_equal(self.data._noise_variance_cov, expected_noise)

    def test_covariance_2d(self):
        argvals = IrregularArgvals({
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
        })
        values = IrregularValues({
            0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
            1: np.array([[1, 2, 3], [1, 2, 3]]),
            2: np.array([[8, 9], [8, 9], [8, 9]])
        })
        data = IrregularFunctionalData(argvals, values)

        with self.assertRaises(NotImplementedError):
            data.covariance()