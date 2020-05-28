#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import unittest

from FDApy.irregular_functional import _check_argvals
from FDApy.irregular_functional import _check_values
from FDApy.irregular_functional import IrregularFunctionalData


class TestIrregularFunctionalData(unittest.TestCase):
    """Test class for the class IrregularFunctionalData.

    """

    # Tests _check_argvals function
    def test_check_argvals_type(self):
        argvals = (1, 2, 3)
        self.assertRaises(ValueError, _check_argvals, argvals)

    def test_check_argvals_type2(self):
        argvals = [[1, 2, 3]]
        self.assertRaises(ValueError, _check_argvals, argvals)

    def test_check_argvals_numeric(self):
        argvals = [(1, 2.5, 3), (None, 5, 3)]
        self.assertRaises(ValueError, _check_argvals, argvals)

    def test_check_argvals_work(self):
        argvals = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        test = _check_argvals(argvals)
        self.assertEquals(len(test), 2)

    def test_check_argvals_work2(self):
        argvals = np.array([1, 2, 3])
        test = _check_argvals(argvals)
        self.assertEquals(len(test), 1)

    # Tests _check_values function
    def test_check_values(self):
        values = [1, 2, 3]
        self.assertRaises(ValueError, _check_values, values)

    def test_check_values_work(self):
        values = [np.array([1, 2, 3]), np.array([1, 2])]
        values = _check_values(values)
        self.assertEquals(len(values), 2)

    def test_check_values_work2(self):
        values = np.array([1, 2, 3])
        values = _check_values(values)
        self.assertEquals(len(values), 1)

    # Tests __init__ function
    def test_init_dimensions(self):
        values = np.array([1, 2, 3])
        argvals = [(1, 2, 3), (4, 5, 6)]
        self.assertRaises(ValueError, IrregularFunctionalData, argvals, values)

    def test_init_sampling(self):
        values = np.array([1, 2, 3])
        argvals = (1, 3)
        self.assertRaises(ValueError, IrregularFunctionalData, argvals, values)

    # Tests nObs function
    def test_nObs(self):
        values = [np.array([1, 2, 3]), np.array([1, 2])]
        argvals = [np.array([1, 2, 3]), np.array([1, 2])]
        irr = IrregularFunctionalData(argvals, values)
        self.assertEqual(irr.nObs(), 2)

    # Tests rangeObs function
    def test_rangeObs(self):
        values = [np.array([1, 2, 3]), np.array([1, 2])]
        argvals = [np.array([1, 2, 3]), np.array([1, 2])]
        irr = IrregularFunctionalData(argvals, values)
        self.assertEqual(irr.rangeObs(), (1, 3))

    # Tests nObsPoint function
    def test_nObsPoint(self):
        values = [np.array([1, 2, 3]), np.array([1, 2])]
        argvals = [np.array([1, 2, 3]), np.array([1, 2])]
        irr = IrregularFunctionalData(argvals, values)
        self.assertEqual(irr.nObsPoint(), [3, 2])

    # Tests rangeObsPoint function
    def test_rangeObsPoint(self):
        values = [np.array([1, 2, 3]), np.array([1, 2])]
        argvals = [np.array([1, 2, 3]), np.array([1, 2])]
        irr = IrregularFunctionalData(argvals, values)
        self.assertEqual(irr.rangeObsPoint(), (1, 3))

    # Tests dimension function
    def test_dimension(self):
        values = [np.array([1, 2, 3]), np.array([1, 2])]
        argvals = [np.array([1, 2, 3]), np.array([1, 2])]
        irr = IrregularFunctionalData(argvals, values)
        self.assertEqual(irr.dimension(), 1)


if __name__ == '__main__':
    unittest.main()
