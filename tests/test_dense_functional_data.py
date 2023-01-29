#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the utils.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import (
    FunctionalData,
    DenseFunctionalData,
    IrregularFunctionalData
)


class TestDenseFunctionalData1D(unittest.TestCase):
    """Test class for the class DenseFunctionalData in one dimension."""

    def setUp(self):
        # First FD
        argvals = {'input_dim_0': np.array([1, 2, 3, 4])}
        values = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 9],
                [3, 4, 5, 7],
                [3, 4, 6, 1],
                [3, 4, 7, 6]
            ]
        )
        self.dense_fd = DenseFunctionalData(argvals, values)

        # Second FD
        argvals = {'input_dim_0': np.array([1, 2, 3])}
        values = np.array(
            [
                [1, 2, 3],
                [5, 6, 7],
                [3, 4, 5]
            ]
        )
        self.dense_fd_2 = DenseFunctionalData(argvals, values)

        # Third FD
        argvals = {'input_dim_0': np.array([1, 2, 3, 4]),
                   'input_dim_1': np.array([5, 6, 7])}
        values = np.array(
            [
                [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]],
                [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]
            ]
        )
        self.dense_fd_3 = DenseFunctionalData(argvals, values)

    def test_check_nobs(self):
        with self.assertRaises(ValueError) as cm:
            FunctionalData._check_same_nobs(self.dense_fd, self.dense_fd_2)
        self.assertTrue('Elements' in str(cm.exception))

    def test_check_ndim(self):
        with self.assertRaises(ValueError) as cm:
            FunctionalData._check_same_ndim(self.dense_fd, self.dense_fd_3)
        self.assertTrue('dimensions' in str(cm.exception))

    def test_check_argvals_equality_dense(self):
        with self.assertRaises(ValueError) as cm:
            DenseFunctionalData._check_argvals_equality_dense(
                self.dense_fd.argvals,
                self.dense_fd_2.argvals
            )
        self.assertTrue('points' in str(cm.exception))

    def test_perform_computation(self):
        X = DenseFunctionalData._perform_computation(
            self.dense_fd,
            self.dense_fd,
            np.add
        )
        self.assertTrue(
            np.allclose(
                X.values,
                np.array(
                    [
                        [2, 4, 6, 8],
                        [10, 12, 14, 18],
                        [6, 8, 10, 14],
                        [6, 8, 12, 2],
                        [6, 8, 14, 12]
                    ]
                )
            )
        )

    def test_argvals(self):
        self.assertTrue(
            np.allclose(
                self.dense_fd.argvals['input_dim_0'],
                np.array([1, 2, 3, 4])
            )
        )

    def test_argvals_stand(self):
        self.assertTrue(
            np.allclose(
                self.dense_fd.argvals_stand['input_dim_0'],
                np.array([0., 0.33333333, 0.66666667, 1.])
            )
        )

    def test_n_obs(self):
        self.assertEqual(self.dense_fd.n_obs, 5)

    def test_n_dim(self):
        self.assertEqual(self.dense_fd.n_dim, 1)

    def test_range_obs(self):
        self.assertEqual(self.dense_fd.range_obs, (1, 9))

    def test_range_dim(self):
        self.assertEqual(self.dense_fd.range_dim, {'input_dim_0': (1, 4)})

    def test_shape(self):
        self.assertEqual(self.dense_fd.shape, {'input_dim_0': 4})

    def test_subset(self):
        new_dense_fd = self.dense_fd[2]
        self.assertIsInstance(new_dense_fd, DenseFunctionalData)
        self.assertEqual(new_dense_fd.n_obs, 1)
        new_dense_fd = self.dense_fd[1:4]
        self.assertIsInstance(new_dense_fd, DenseFunctionalData)
        self.assertEqual(new_dense_fd.n_obs, 3)

    def test_as_irregular(self):
        irregu_fd = self.dense_fd.as_irregular()
        self.assertIsInstance(irregu_fd, IrregularFunctionalData)
        self.assertEqual(irregu_fd.n_obs, 5)

    def test_is_compatible(self):
        self.assertTrue(self.dense_fd.is_compatible(self.dense_fd))

    def test_mean(self):
        mean_fd = self.dense_fd.mean()
        is_equal = np.allclose(mean_fd.values,
                               np.array([[3., 4., 5.6, 5.4]]))
        self.assertTrue(is_equal)


class TestDenseFunctionalData2D(unittest.TestCase):
    """Test class for the class DenseFunctionalData in two dimension."""

    def setUp(self):
        argvals = {'input_dim_0': np.array([1, 2, 3, 4]),
                   'input_dim_1': np.array([5, 6, 7])}

        values = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                           [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]],
                           [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]],
                           [[3, 4, 6], [3, 4, 5], [3, 4, 5], [3, 4, 5]],
                           [[3, 4, 7], [3, 4, 5], [3, 4, 5], [3, 4, 5]]])
        self.dense_fd = DenseFunctionalData(argvals, values)

    def test_argvals_stand(self):
        is_equal_dim0 = np.allclose(self.dense_fd.argvals_stand['input_dim_0'],
                                    np.array([0., 0.33333333, 0.66666667, 1.]))
        is_equal_dim1 = np.allclose(self.dense_fd.argvals_stand['input_dim_1'],
                                    np.array([0., 0.5, 1.]))
        self.assertTrue(is_equal_dim0 and is_equal_dim1)

    def test_n_obs(self):
        self.assertEqual(self.dense_fd.n_obs, 5)

    def test_n_dim(self):
        self.assertEqual(self.dense_fd.n_dim, 2)

    def test_range_obs(self):
        self.assertEqual(self.dense_fd.range_obs, (1, 7))

    def test_range_dim(self):
        self.assertEqual(self.dense_fd.range_dim, {'input_dim_0': (1, 4),
                                                   'input_dim_1': (5, 7)})

    def test_shape(self):
        self.assertEqual(self.dense_fd.shape, {'input_dim_0': 4,
                                               'input_dim_1': 3})

    def test_subset(self):
        new_dense_fd = self.dense_fd[2]
        self.assertIsInstance(new_dense_fd, DenseFunctionalData)
        self.assertEqual(new_dense_fd.n_obs, 1)
        new_dense_fd = self.dense_fd[1:4]
        self.assertIsInstance(new_dense_fd, DenseFunctionalData)
        self.assertEqual(new_dense_fd.n_obs, 3)

    def test_as_irregular(self):
        irregu_fd = self.dense_fd.as_irregular()
        self.assertIsInstance(irregu_fd, IrregularFunctionalData)
        self.assertEqual(irregu_fd.n_obs, 5)

    def test_is_compatible(self):
        self.assertTrue(self.dense_fd.is_compatible(self.dense_fd))

    def test_mean(self):
        mean_fd = self.dense_fd.mean()
        is_equal = np.allclose(mean_fd.values,
                               np.array([[[3., 4., 5.6],
                                          [3., 4., 5.],
                                          [3., 4., 5.],
                                          [3., 4., 5.]]]))
        self.assertTrue(is_equal)
