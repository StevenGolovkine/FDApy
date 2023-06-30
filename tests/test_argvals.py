#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the _argvals.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation._argvals import DenseArgvals, IrregularArgvals


class TestDenseArgvals(unittest.TestCase):

    def test_setitem(self):
        argvals = DenseArgvals()
        argvals['key1'] = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(argvals['key1'], np.array([1, 2, 3])))

    def test_setitem_invalid_key_type(self):
        argvals = DenseArgvals()
        with self.assertRaises(TypeError):
            argvals[0] = np.array([1, 2, 3])

    def test_setitem_invalid_value_type(self):
        argvals = DenseArgvals()
        with self.assertRaises(TypeError):
            argvals['key1'] = 'value'

    def test_eq(self):
        argvals1 = DenseArgvals()
        argvals1['key1'] = np.array([1, 2, 3])
        argvals1['key2'] = np.array([4, 5, 6])

        argvals2 = DenseArgvals()
        argvals2['key1'] = np.array([1, 2, 3])
        argvals2['key2'] = np.array([4, 5, 6])

        self.assertTrue(argvals1 == argvals2)

        argvals3 = DenseArgvals()
        argvals3['key1'] = np.array([1, 2, 3])
        argvals3['key2'] = np.array([4, 5, 7])

        self.assertFalse(argvals1 == argvals3)

        argvals4 = DenseArgvals()
        argvals4['key1'] = np.array([1, 2, 3])

        self.assertFalse(argvals1 == argvals4)

        argvals5 = {}
        argvals5['key1'] = np.array([1, 2, 3])
        argvals5['key2'] = np.array([4, 5, 6])

        self.assertFalse(argvals1 == argvals5)
    
    def test_n_points(self):
        argvals1 = DenseArgvals()
        argvals1['key1'] = np.array([1, 2, 3])
        argvals1['key2'] = np.array([4, 5, 6])

        np.testing.assert_equal(argvals1.n_points, (3, 3))

    def test_compatible_with(self):
        argvals1 = DenseArgvals()
        argvals1['key1'] = np.array([1, 2, 3])
        argvals1['key2'] = np.array([4, 5, 6])

        values = np.random.randn(10, 3, 3)
        argvals1.compatible_with(values)

        values = np.random.randn(10, 4, 3)
        with self.assertRaises(ValueError):
            argvals1.compatible_with(values)


class TestIrregularArgvals(unittest.TestCase):

    def test_setitem_valid(self):
        argvals = IrregularArgvals()
        dense_argvals = DenseArgvals({
            'input_dim_0': np.random.randn(10),
            'input_dim_1': np.random.randn(11)
        })
        argvals[0] = dense_argvals
        self.assertEqual(len(argvals), 1)
        self.assertEqual(argvals[0], dense_argvals)

    def test_setitem_invalid_key_type(self):
        argvals = IrregularArgvals()
        dense_argvals = DenseArgvals({
            'input_dim_0': np.random.randn(10),
            'input_dim_1': np.random.randn(11)
        })
        with self.assertRaises(TypeError):
            argvals['key'] = dense_argvals

    def test_setitem_invalid_value_type(self):
        argvals = IrregularArgvals()
        with self.assertRaises(TypeError):
            argvals[0] = 'value'

    def test_eq_equal_objects(self):
        argvals_1 = DenseArgvals({
            'input_dim_0': np.random.randn(10),
            'input_dim_1': np.random.randn(11)
        })
        argvals_2 = DenseArgvals({
            'input_dim_0': np.random.randn(5),
            'input_dim_1': np.random.randn(7)
        })
        argvals_irr_1 = IrregularArgvals({0: argvals_1, 1: argvals_2})
        argvals_irr_2 = IrregularArgvals({0: argvals_1, 1: argvals_2})
        self.assertEqual(argvals_irr_1, argvals_irr_2)

    def test_eq_different_objects(self):
        argvals_1 = DenseArgvals({
            'input_dim_0': np.random.randn(10),
            'input_dim_1': np.random.randn(11)
        })
        argvals_2 = DenseArgvals({
            'input_dim_0': np.random.randn(5),
            'input_dim_1': np.random.randn(7)
        })
        argvals_irr_1 = IrregularArgvals({0: argvals_1, 1: argvals_2})
        argvals_irr_3 = IrregularArgvals({0: argvals_2, 1: argvals_1})
        self.assertNotEqual(argvals_irr_1, argvals_irr_3)

    def test_eq_different_type(self):
        argvals_1 = DenseArgvals({
            'input_dim_0': np.random.randn(10),
            'input_dim_1': np.random.randn(11)
        })
        argvals_2 = DenseArgvals({
            'input_dim_0': np.random.randn(5),
            'input_dim_1': np.random.randn(7)
        })
        argvals_irr_1 = IrregularArgvals({0: argvals_1, 1: argvals_2})
        argvals_irr_3 = {0: argvals_2, 1: argvals_1}
        self.assertNotEqual(argvals_irr_1, argvals_irr_3)
    
    def test_eq_different_length(self):
        argvals_1 = DenseArgvals({
            'input_dim_0': np.random.randn(10),
            'input_dim_1': np.random.randn(11)
        })
        argvals_2 = DenseArgvals({
            'input_dim_0': np.random.randn(5),
            'input_dim_1': np.random.randn(7)
        })
        argvals_irr_1 = IrregularArgvals({0: argvals_1, 1: argvals_2})
        argvals_irr_3 = IrregularArgvals({0: argvals_1, 1: argvals_2, 2: argvals_1})
        self.assertNotEqual(argvals_irr_1, argvals_irr_3)

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

        values = {0: np.random.randn(10, 11), 1: np.random.randn(5, 7)}
        argvals_irr.compatible_with(values)

        values = {0: np.random.randn(10, 10), 1: np.random.randn(5, 7)}
        with self.assertRaises(ValueError):
            argvals_irr.compatible_with(values)
