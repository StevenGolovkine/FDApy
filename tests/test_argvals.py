#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the _argvals.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation._argvals import DenseArgvals, IrregularArgvals
from FDApy.representation._values import DenseValues, IrregularValues

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

        values = DenseValues(np.random.randn(10, 3, 3))
        argvals1.compatible_with(values)

        values = DenseValues(np.random.randn(10, 4, 3))
        with self.assertRaises(ValueError):
            argvals1.compatible_with(values)

    def test_normalization(self):
        argvals = {'input_dim_0': np.arange(0, 51, 1), 'input_dim_1': np.arange(0, 21, 1)}
        argvals = DenseArgvals(argvals)

        expected_output = DenseArgvals({'input_dim_0': np.array([0.  , 0.02, 0.04, 0.06, 0.08, 0.1 , 0.12, 0.14, 0.16, 0.18, 0.2 , 0.22, 0.24, 0.26, 0.28, 0.3 , 0.32, 0.34, 0.36, 0.38, 0.4 , 0.42, 0.44, 0.46, 0.48, 0.5 , 0.52, 0.54, 0.56, 0.58, 0.6 , 0.62, 0.64, 0.66, 0.68, 0.7 , 0.72, 0.74, 0.76, 0.78, 0.8 , 0.82, 0.84, 0.86, 0.88, 0.9 , 0.92, 0.94, 0.96, 0.98, 1.  ]), 'input_dim_1': np.array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ])})
        np.testing.assert_equal(argvals.normalization(), expected_output)


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

    def test_n_points(self):
        argvals_1 = DenseArgvals({
            'input_dim_0': np.random.randn(10),
            'input_dim_1': np.random.randn(11)
        })
        argvals_2 = DenseArgvals({
            'input_dim_0': np.random.randn(5),
            'input_dim_1': np.random.randn(7)
        })
        argvals_irr = IrregularArgvals({0: argvals_1, 1: argvals_2})

        np.testing.assert_equal(argvals_irr.n_points, {0: (10, 11), 1: (5, 7)})

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
        argvals_irr.compatible_with(values)

        values = IrregularValues({0: np.random.randn(10, 10), 1: np.random.randn(5, 7)})
        with self.assertRaises(ValueError):
            argvals_irr.compatible_with(values)
