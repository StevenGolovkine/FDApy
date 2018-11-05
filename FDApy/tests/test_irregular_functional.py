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
        argvals = np.array([1, 2, 3])
        self.assertRaises(ValueError, _check_argvals, argvals)

    def test_check_argvals_type2(self):
        argvals = [[1, 2, 3]]
        self.assertRaises(ValueError, _check_argvals, argvals)

    def test_check_argvals_numeric(self):
        argvals = [(1, 2.5, 3), (None, 5, 3)]
        self.assertRaises(ValueError, _check_argvals, argvals)

    def test_check_argvals_work(self):
        argvals = [(1, 2, 3), (4, 5, 6)]
        test = _check_argvals(argvals)
        self.assertEquals(len(test), 2)

    def test_check_argvals_work2(self):
        argvals = (1, 2, 3)
        test = _check_argvals(argvals)
        self.assertEquals(len(test), 1)

    # Tests _check_values function
    def test_check_values(self):
        X = [1, 2, 3]
        self.assertRaises(ValueError, _check_values, X)

    def test_check_values_work(self):
        X = [np.array([1, 2, 3]), np.array([1, 2])]
        X = _check_values(X)
        self.assertEquals(len(X), 2)

    def test_check_values_work2(self):
        X = np.array([1, 2, 3])
        X = _check_values(X)
        self.assertEquals(len(X), 1)

    # Tests __init__ function
    def test_init_dimensions(self):
        X = np.array([1, 2, 3])
        argvals = [(1, 2, 3), (4, 5, 6)]
        self.assertRaises(ValueError, IrregularFunctionalData, argvals, X)

    def test_init_sampling(self):
        X = np.array([1, 2, 3])
        argvals = (1, 3)
        self.assertRaises(ValueError, IrregularFunctionalData, argvals, X)


if __name__ == '__main__':
    unittest.main()