#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the _values.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation._values import DenseValues, IrregularValues


class TestDenseValues(unittest.TestCase):
    def test_n_obs(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        values = DenseValues(array)
        self.assertEqual(values.n_obs, 2)


class TestIrregularValues(unittest.TestCase):
    def test_n_obs(self):
        values_dict = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}
        values = IrregularValues(values_dict)
        self.assertEqual(values.n_obs, 2)

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
