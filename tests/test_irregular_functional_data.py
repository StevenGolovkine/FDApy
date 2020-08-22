#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import unittest

from FDApy.representation.functional_data import (DenseFunctionalData,
                                                  IrregularFunctionalData)


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


if __name__ == '__main__':
    unittest.main()
