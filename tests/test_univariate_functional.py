#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import unittest

from FDApy.univariate_functional import _check_argvals
from FDApy.univariate_functional import _check_values
from FDApy.univariate_functional import _check_argvals_values
from FDApy.univariate_functional import _check_argvals_equality
from FDApy.univariate_functional import UnivariateFunctionalData


class TestUnivariateFunctionalData(unittest.TestCase):
    """Test class for the class UnivariateFunctionalData.

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

    # Tests _check_argvals_values function
    def test_check_argvals_values_len(self):
        argvals = [(1, 2, 3), (1, 2)]
        values = np.array([[1, 2, 3], [1, 2, 3]])
        self.assertRaises(ValueError, _check_argvals_values, argvals, values)

    def test_check_argvals_values_dim(self):
        argvals = (1, 2, 3)
        values = np.array([[1, 2], [3, 4]])
        self.assertRaises(ValueError, _check_argvals_values, argvals, values)

    def test_check_argvals_values_work(self):
        argvals = [(1, 2, 3)]
        values = np.array([[1, 2, 3], [4, 5, 6]])
        res = _check_argvals_values(argvals, values)
        self.assertTrue(res)

    # Tests _check_argvals_equality function
    def test_check_argvals_equality(self):
        argvals1 = (1, 2, 3)
        argvals2 = (4, 5, 6)
        self.assertRaises(ValueError,
                          _check_argvals_equality, argvals1, argvals2)

    # Tests __init__ function
    def test_init_dimensions(self):
        values = np.array([[1, 2, 3], [4, 5, 6]])
        argvals = [(1, 2, 3), (4, 5, 6)]
        self.assertRaises(ValueError, UnivariateFunctionalData,
                          argvals, values)

    def test_init_sampling(self):
        values = np.array([[1, 2, 3], [4, 5, 6]])
        argvals = np.array([[1, 3]])
        self.assertRaises(ValueError, UnivariateFunctionalData,
                          argvals, values)

    # Tests nObs function
    def test_nObs(self):
        argvals = np.array([1, 2, 3])
        values = np.array([[1, 2, 3], [4, 5, 6]])
        uni = UnivariateFunctionalData(argvals, values)
        self.assertEquals(uni.nObs(), 2)

    # Tests rangeObs function
    def test_rangeObs(self):
        # Test in dimension 1
        argvals = np.array([1, 2, 3])
        values = np.array([[1, 2, 3],
                           [7, 8, 9]])
        uni = UnivariateFunctionalData(argvals, values)
        self.assertEquals(uni.rangeObs(), (1, 9))

    def test_rangeObs2(self):
        # Test in dimension 2
        argvals = [np.array([1, 2, 3]), np.array([1, 2])]
        values = np.array([[[1, 2], [3, 4], [5, 6]],
                           [[7, 8], [9, 10], [11, 12]]])
        uni = UnivariateFunctionalData(argvals, values)
        self.assertEquals(uni.rangeObs(), (1, 12))

    # Tests nObsPoint function
    def test_nObsPoint(self):
        argvals = [np.array([1, 2, 3]), np.array([1, 2])]
        values = np.array([[[1, 2], [3, 4], [5, 6]],
                           [[7, 8], [9, 10], [11, 12]]])
        uni = UnivariateFunctionalData(argvals, values)
        self.assertEquals(uni.nObsPoint(), [3, 2])

    # Tests rangeObsPoint function
    def test_rangeObsPoint(self):
        argvals = [np.array([1, 2, 3]), np.array([1, 2])]
        values = np.array([[[1, 2], [3, 4], [5, 6]],
                           [[7, 8], [9, 10], [11, 12]]])
        uni = UnivariateFunctionalData(argvals, values)
        self.assertEquals(uni.rangeObsPoint(), [(1, 3), (1, 2)])

    # Tests dimension function
    def test_dimension(self):
        argvals = np.array([1, 2, 3])
        values = np.array([[1, 2, 3], [4, 5, 6]])
        uni = UnivariateFunctionalData(argvals, values)
        self.assertEquals(uni.dimension(), 1)

    def test_dimension2(self):
        argvals = [np.array([1, 2, 3]), np.array([1, 2])]
        values = np.array([[[1, 2], [3, 4], [5, 6]],
                           [[7, 8], [9, 10], [11, 12]]])
        uni = UnivariateFunctionalData(argvals, values)
        self.assertEquals(uni.dimension(), 2)


if __name__ == '__main__':
    unittest.main()
