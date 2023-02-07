#!/usr/bin/python3
# -*-coding:utf8 -*

import numpy as np
import unittest

from FDApy.representation.functional_data import (
    FunctionalData,
    DenseFunctionalData,
    IrregularFunctionalData
)


class TestIrregularFunctionalData(unittest.TestCase):
    def setUp(self):
        self.argvals = {
            'input_dim_0': {
                0: np.array([0, 1, 2, 3, 4]),
                1: np.array([0, 2, 4]),
                2: np.array([2, 4]),
            }
        }
        self.values = {
            0: np.array([1, 2, 3, 4, 5]),
            1: np.array([2, 5, 6]),
            2: np.array([4, 7]),
        }
        self.fdata = IrregularFunctionalData(self.argvals, self.values)

        self.dense_argvals = {'input_dim_0': np.array([1, 2, 3, 4, 5])}
        self.dense_values = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        self.dense_data = DenseFunctionalData(
            self.dense_argvals, self.dense_values
        )

    def test_get_item_slice(self):
        fdata = self.fdata[1:3]
        self.assertIsInstance(fdata, IrregularFunctionalData)
        self.assertEqual(fdata.n_obs, 2)
        self.assertEqual(fdata.n_dim, 1)
        np.testing.assert_equal(
            fdata.argvals['input_dim_0'][1],
            self.argvals['input_dim_0'][1]
        )
        np.testing.assert_equal(
            fdata.argvals['input_dim_0'][2],
            self.argvals['input_dim_0'][2]
        )
        np.testing.assert_equal(
            fdata.values[1],
            self.values[1]
        )

    def test_get_item_index(self):
        fdata = self.fdata[1]
        self.assertIsInstance(fdata, IrregularFunctionalData)
        self.assertEqual(fdata.n_obs, 1)
        self.assertEqual(fdata.n_dim, 1)
        np.testing.assert_equal(
            fdata.argvals['input_dim_0'][1],
            self.argvals['input_dim_0'][1]
        )
        np.testing.assert_equal(
            fdata.values[1],
            self.values[1]
        )

    def test_argvals_getter(self):
        argvals = self.fdata.argvals
        self.assertEqual(argvals, self.argvals)

    def test_argvals_setter(self):
        new_argvals = {
            'input_dim_0': {
                0: np.array([5, 6, 7, 8, 9]),
                1: np.array([6, 8, 10]),
                2: np.array([6, 8]),
            }
        }
        self.fdata.argvals = new_argvals
        self.assertEqual(self.fdata._argvals, new_argvals)

        expected_argvals_stand = {
            'input_dim_0': {
                0: np.array([0, 0.2, 0.4, 0.6, 0.8]),
                1: np.array([0.2, 0.6, 1]),
                2: np.array([0.2, 0.6]),
            }
        }
        np.testing.assert_array_almost_equal(
            self.fdata._argvals_stand['input_dim_0'][0],
            expected_argvals_stand['input_dim_0'][0]
        )
        np.testing.assert_array_almost_equal(
            self.fdata._argvals_stand['input_dim_0'][1],
            expected_argvals_stand['input_dim_0'][1]
        )
        np.testing.assert_array_almost_equal(
            self.fdata._argvals_stand['input_dim_0'][1],
            expected_argvals_stand['input_dim_0'][1]
        )

    def test_values_property(self):
        values = self.fdata.values
        np.testing.assert_array_equal(values, self.values)

    def test_values_setter(self):
        new_values = {
            0: np.array([1, 4, 3, 4, 9]),
            1: np.array([1, 5, 3]),
            2: np.array([7, 7]),
        }
        self.fdata.values = new_values
        np.testing.assert_array_equal(self.fdata.values, new_values)

    def test_range_obs(self):
        expected_range = (1, 7)
        self.assertEqual(self.fdata.range_obs, expected_range)

    def test_n_points(self):
        expected_n_points = {'input_dim_0': 10 / 3}
        self.assertDictEqual(self.fdata.n_points, expected_n_points)

    def test_range_dim(self):
        expected_range_dim = {'input_dim_0': (0, 4)}
        self.assertDictEqual(expected_range_dim, self.fdata.range_dim)

    def test_shape(self):
        expected_shape = {'input_dim_0': 5}
        self.assertDictEqual(self.fdata.shape, expected_shape)

    def test_gather_points(self):
        expected_points = {'input_dim_0': np.arange(5)}
        np.testing.assert_array_equal(
            self.fdata.gather_points()['input_dim_0'],
            expected_points['input_dim_0']
        )

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
        self.assertTrue(self.fdata.is_compatible(self.fdata))

    def test_non_compatible_type(self):
        with self.assertRaises(TypeError):
            self.fdata.is_compatible(self.dense_data)

    def test_non_compatible_nobs(self):
        argvals = {
            'input_dim_0': {
                0: np.array([0, 1, 2, 3, 4]),
                1: np.array([0, 2, 4]),
            }
        }
        values = {
            0: np.array([1, 2, 3, 4, 5]),
            1: np.array([2, 5, 6]),
        }
        func_data = IrregularFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            self.fdata.is_compatible(func_data)

    def test_non_compatible_ndim(self):
        argvals = {
            'input_dim_0': {
                0: np.array([0, 1, 2, 3, 4]),
                1: np.array([0, 2, 4]),
                2: np.array([2, 4]),
            },
            'input_dim_1': {
                0: np.array([5, 6, 7]),
                1: np.array([1, 2, 3]),
                2: np.array([1, 2])
            }
        }
        values = {
            0: np.array([
                [1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4], [1, 2, 4]
            ]),
            1: np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            2: np.array([[1, 2], [3, 4]])
        }
        func_data = IrregularFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            self.fdata.is_compatible(func_data)

    def test_non_compatible_argvals_equality(self):
        argvals = {
            'input_dim_0': {
                0: np.array([0, 1, 2, 3, 5]),
                1: np.array([0, 2, 4]),
                2: np.array([2, 4]),
            }
        }
        values = {
            0: np.array([1, 2, 3, 4, 5]),
            1: np.array([2, 5, 6]),
            2: np.array([4, 7]),
        }
        func_data = IrregularFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            self.fdata.is_compatible(func_data)


class TestIrregularFunctionalData1D(unittest.TestCase):
    """Test class for the class IrregularFunctionalData in one dimension."""

    def setUp(self):
        argvals = {'input_dim_0': {0: np.array([1, 2, 3, 4]),
                                   1: np.array([2, 4]),
                                   2: np.array([0, 2, 3])}}
        values = {0: np.array([1, 2, 3, 4]),
                  1: np.array([5, 6]),
                  2: np.array([8, 9, 7],)}
        self.irregu_fd = IrregularFunctionalData(argvals, values)

    def test_argvals_stand(self):
        is_equal = [np.allclose(self.irregu_fd.argvals_stand['input_dim_0'][0],
                                np.array([0.25, 0.5, 0.75, 1.])),
                    np.allclose(self.irregu_fd.argvals_stand['input_dim_0'][1],
                                np.array([0.5, 1.])),
                    np.allclose(self.irregu_fd.argvals_stand['input_dim_0'][2],
                                np.array([0., 0.5, 0.75]))]
        self.assertTrue(np.all(is_equal))

    def test_n_obs(self):
        self.assertEqual(self.irregu_fd.n_obs, 3)

    def test_n_dim(self):
        self.assertEqual(self.irregu_fd.n_dim, 1)

    def test_range_obs(self):
        self.assertEqual(self.irregu_fd.range_obs, (1, 9))

    def test_range_dim(self):
        self.assertEqual(self.irregu_fd.range_dim, {'input_dim_0': (0, 4)})

    def test_shape(self):
        self.assertEqual(self.irregu_fd.shape, {'input_dim_0': 5})

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
        self.assertTrue(self.irregu_fd.is_compatible(self.irregu_fd))

    def test_mean(self):
        mean_fd = self.irregu_fd.mean()
        is_equal = np.allclose(mean_fd.values,
                               np.array([[8., 1., 5.33333333, 5., 5.]]))
        self.assertTrue(is_equal)


class TestIrregularFunctionalData2D(unittest.TestCase):
    """Test class for the class IrregularFunctionalData in two dimension."""

    def setUp(self):
        argvals = {'input_dim_0': {0: np.array([1, 2, 3, 4]),
                                   1: np.array([2, 4]),
                                   2: np.array([4, 5, 6])},
                   'input_dim_1': {0: np.array([5, 6, 7]),
                                   1: np.array([1, 2, 3]),
                                   2: np.array([8, 9])}}
        values = {0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
                  1: np.array([[1, 2, 3], [1, 2, 3]]),
                  2: np.array([[8, 9], [8, 9], [8, 9]])}
        self.irregu_fd = IrregularFunctionalData(argvals, values)

    def test_argvals_stand(self):
        is_equal = [np.allclose(self.irregu_fd.argvals_stand['input_dim_0'][0],
                                np.array([0., 0.2, 0.4, 0.6])),
                    np.allclose(self.irregu_fd.argvals_stand['input_dim_0'][1],
                                np.array([0.2, 0.6])),
                    np.allclose(self.irregu_fd.argvals_stand['input_dim_0'][2],
                                np.array([0.6, 0.8, 1.])),
                    np.allclose(self.irregu_fd.argvals_stand['input_dim_1'][0],
                                np.array([0.5, 0.625, 0.75])),
                    np.allclose(self.irregu_fd.argvals_stand['input_dim_1'][1],
                                np.array([0., 0.125, 0.25])),
                    np.allclose(self.irregu_fd.argvals_stand['input_dim_1'][2],
                                np.array([0.875, 1.]))]
        self.assertTrue(np.all(is_equal))

    def test_n_obs(self):
        self.assertEqual(self.irregu_fd.n_obs, 3)

    def test_n_dim(self):
        self.assertEqual(self.irregu_fd.n_dim, 2)

    def test_range_obs(self):
        self.assertEqual(self.irregu_fd.range_obs, (1, 9))

    def test_range_dim(self):
        self.assertEqual(self.irregu_fd.range_dim, {'input_dim_0': (1, 6),
                                                    'input_dim_1': (1, 9)})

    def test_shape(self):
        self.assertEqual(self.irregu_fd.shape, {'input_dim_0': 6,
                                                'input_dim_1': 8})

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
        self.assertTrue(self.irregu_fd.is_compatible(self.irregu_fd))

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
