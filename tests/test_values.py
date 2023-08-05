#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the _values.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
from FDApy.representation.values import DenseValues, IrregularValues


class TestDenseValues(unittest.TestCase):
    def test_n_obs(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        values = DenseValues(array)
        self.assertEqual(values.n_obs, 2)

    def test_n_points(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        values = DenseValues(array)
        self.assertEqual(values.n_points, (3,))

    def test_compatible_with(self):
        argvals1 = DenseArgvals()
        argvals1['key1'] = np.array([1, 2, 3])
        argvals1['key2'] = np.array([4, 5, 6])

        values = DenseValues(np.random.randn(10, 3, 3))
        values.compatible_with(argvals1)

        values = DenseValues(np.random.randn(10, 4, 3))
        with self.assertRaises(ValueError):
            values.compatible_with(argvals1)

    def test_concatenate(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        values = DenseValues(array)

        new_values = DenseValues.concatenate(values, values)
        expected_values = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]])
        np.testing.assert_equal(new_values, DenseValues(expected_values))


class TestIrregularValues(unittest.TestCase):
    def test_n_obs(self):
        values_dict = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}
        values = IrregularValues(values_dict)
        self.assertEqual(values.n_obs, 2)

    def test_n_points(self):
        values_dict = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}
        values = IrregularValues(values_dict)
        self.assertEqual(values.n_points, {0: (3,), 1: (3,)})

    def test_setitem(self):
        values = IrregularValues()
        values[0] = np.array([1, 2, 3])
        values[1] = np.array([4, 5, 6])
        self.assertEqual(len(values), 2)

    def test_setitem_invalid_key(self):
        values = IrregularValues()
        with self.assertRaises(TypeError):
            values['key'] = np.array([1, 2, 3])

    def test_setitem_invalid_value(self):
        values = IrregularValues()
        with self.assertRaises(TypeError):
            values[0] = 'value'

    def test_compatible_with(self):
        argvals_1 = DenseArgvals({
            'input_dim_0': np.random.randn(10),
            'input_dim_1': np.random.randn(11)
        })
        argvals_2 = DenseArgvals({
            'input_dim_0': np.random.randn(5),
            'input_dim_1': np.random.randn(7)
        })
        argvals_irr = IrregularArgvals({0: argvals_1, 1: argvals_2})

        values = IrregularValues({0: np.random.randn(10, 11), 1: np.random.randn(5, 7)})
        values.compatible_with(argvals_irr)

        values = IrregularValues({0: np.random.randn(10, 10), 1: np.random.randn(5, 7)})
        with self.assertRaises(ValueError):
            values.compatible_with(argvals_irr)

    def test_concatenate(self):
        values_dict = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}
        values = IrregularValues(values_dict)
        
        values_dict = {0: np.array([1, 2])}
        values_2 = IrregularValues(values_dict)

        new_values = IrregularValues.concatenate(values, values_2)
        expected_values = IrregularValues({
            0: np.array([1, 2, 3]),
            1: np.array([4, 5, 6]),
            2: np.array([1, 2])
        })
        np.testing.assert_allclose(new_values, expected_values)
