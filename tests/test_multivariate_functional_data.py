#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for MultivariateFunctionalData.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import (
    DenseFunctionalData,
    MultivariateFunctionalData
)


class MultivariateFunctionalDataTest(unittest.TestCase):
    def setUp(self):
        self.argvals = {'input_dim_0': np.array([1, 2, 3, 4, 5])}
        self.values = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])

        self.fdata1 = DenseFunctionalData(self.argvals, self.values)
        self.fdata2 = DenseFunctionalData(self.argvals, self.values)
        self.fdata3 = DenseFunctionalData(self.argvals, self.values)
        self.multivariate_data = MultivariateFunctionalData([self.fdata1, self.fdata2])

    def test_init(self):
        self.assertEqual(len(self.multivariate_data), 2)
        self.assertIsInstance(self.multivariate_data[0], DenseFunctionalData)
        self.assertIsInstance(self.multivariate_data[1], DenseFunctionalData)

    def test_repr(self):
        expected_repr = f"Multivariate functional data object with 2 functions of 3 observations."
        actual_repr = repr(self.multivariate_data)
        self.assertEqual(actual_repr, expected_repr)

    def test_n_obs(self):
        expected_n_obs = 3
        actual_n_obs = self.multivariate_data.n_obs
        self.assertEqual(actual_n_obs, expected_n_obs)

    def test_n_functional(self):
        expected_n_functional = 2
        actual_n_functional = self.multivariate_data.n_functional
        self.assertEqual(actual_n_functional, expected_n_functional)

    def test_n_dim(self):
        expected_n_dim = [1, 1]
        actual_n_dim = self.multivariate_data.n_dim
        self.assertEqual(actual_n_dim, expected_n_dim)

    def test_range_obs(self):
        expected_range_obs = [(1, 15), (1, 15)]
        actual_range_obs = self.multivariate_data.range_obs
        self.assertEqual(actual_range_obs, expected_range_obs)

    def test_n_points(self):
        expected_n_points = [{'input_dim_0': 5}, {'input_dim_0': 5}]
        actual_n_points = self.multivariate_data.n_points
        self.assertEqual(actual_n_points, expected_n_points)

    def test_range_points(self):
        expected_range_points = [{'input_dim_0': (1, 5)}, {'input_dim_0': (1, 5)}]
        actual_range_points = self.multivariate_data.range_points
        self.assertEqual(actual_range_points, expected_range_points)

    def test_shape(self):
        expected_shape = [{'input_dim_0': 5}, {'input_dim_0': 5}]
        actual_shape = self.multivariate_data.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_append(self):
        self.multivariate_data.append(self.fdata3)
        self.assertEqual(len(self.multivariate_data), 3)
        self.assertEqual(self.multivariate_data[2], self.fdata3)

        fdata = MultivariateFunctionalData([])
        fdata.append(self.fdata1)
        self.assertEqual(fdata[0], self.fdata1)

    def test_extend(self):
        self.multivariate_data.extend([self.fdata3])
        self.assertEqual(len(self.multivariate_data), 3)
        self.assertEqual(self.multivariate_data[2], self.fdata3)

    def test_insert(self):
        self.multivariate_data.insert(1, self.fdata3)
        self.assertEqual(len(self.multivariate_data), 3)
        self.assertEqual(self.multivariate_data[1], self.fdata3)

    def test_remove(self):
        with self.assertRaises(NotImplementedError):
            self.multivariate_data.remove(self.fdata1)

    def test_pop(self):
        popped_data = self.multivariate_data.pop(0)
        self.assertEqual(len(self.multivariate_data), 1)
        self.assertEqual(popped_data, self.fdata1)

    def test_clear(self):
        self.multivariate_data.clear()
        self.assertEqual(len(self.multivariate_data), 0)

    def test_reverse(self):
        self.multivariate_data.reverse()
        self.assertEqual(self.multivariate_data[0], self.fdata2)
        self.assertEqual(self.multivariate_data[1], self.fdata1)

    def test_copy(self):
        copied_data = self.multivariate_data.copy()
        self.assertEqual(len(copied_data), 2)
        self.assertIsNot(copied_data, self.multivariate_data)
        self.assertIs(copied_data[0], self.multivariate_data[0])
        self.assertIs(copied_data[1], self.multivariate_data[1])

    def test_items(self):
        items = list(self.multivariate_data.items())
        self.assertEqual(len(items), self.fdata1.n_obs)
        for item in items:
            self.assertIsInstance(item, MultivariateFunctionalData)
            self.assertIsInstance(item[0], DenseFunctionalData)
            self.assertIsInstance(item[1], DenseFunctionalData)

    def test_get_obs(self):
        obs = self.multivariate_data.get_obs(0)
        self.assertIsInstance(obs, MultivariateFunctionalData)
        self.assertEqual(len(obs), 2)
        np.testing.assert_array_equal(obs[0].values, np.array([[1, 2, 3, 4, 5]]))
        np.testing.assert_array_equal(obs[1].values, np.array([[1, 2, 3, 4, 5]]))
