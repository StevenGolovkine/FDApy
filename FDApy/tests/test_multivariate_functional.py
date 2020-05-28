#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import unittest

from FDApy.multivariate_functional import _check_data
from FDApy.multivariate_functional import MultivariateFunctionalData
from FDApy.univariate_functional import UnivariateFunctionalData


class TestMultivariateFunctionalData(unittest.TestCase):
    """ Test class for the class MultivariateFunctionalData.

    """

    # Tests _check_data function
    def test_check_data_type(self):
        data = (1, 2, 3)
        self.assertRaises(ValueError, _check_data, data)

    def test_check_data_type2(self):
        data = [(1, 2, 3), (1, 2)]
        self.assertRaises(ValueError, _check_data, data)

    def test_check_data_uni(self):
        argvals = np.array([1, 2, 3])
        X = np.array([[1, 2, 3], [4, 5, 6]])
        uni = UnivariateFunctionalData(argvals, X)
        multi = MultivariateFunctionalData(uni)
        self.assertEquals(multi.nFunctions(), 1)

    def test_check_data_observations(self):
        argvals = np.array([1, 2, 3])
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        uni = UnivariateFunctionalData(argvals, X)
        uni1 = UnivariateFunctionalData(argvals, X1)
        self.assertRaises(ValueError, MultivariateFunctionalData, [uni, uni1])

    def test_check_data_work(self):
        argvals = np.array([1, 2, 3])
        X = np.array([[1, 2, 3], [4, 5, 6]])
        uni = UnivariateFunctionalData(argvals, X)
        multi = MultivariateFunctionalData([uni, uni])
        self.assertEquals(multi.nFunctions(), 2)

    # Tests nFunction function
    def test_nFunction(self):
        argvals = np.array([1, 2, 3])
        argvals1 = [np.array([1, 2, 3]), np.array([1, 2])]
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X1 = np.array([[[1, 2], [3, 4], [5, 6]],
                       [[7, 8], [9, 10], [11, 12]]])
        uni = UnivariateFunctionalData(argvals, X)
        uni1 = UnivariateFunctionalData(argvals1, X1)
        multi = MultivariateFunctionalData([uni, uni1])
        self.assertEquals(multi.nFunctions(), 2)

    # Tests nObs function
    def test_nObs(self):
        argvals = np.array([1, 2, 3])
        argvals1 = [np.array([1, 2, 3]), np.array([1, 2])]
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X1 = np.array([[[1, 2], [3, 4], [5, 6]],
                       [[7, 8], [9, 10], [11, 12]]])
        uni = UnivariateFunctionalData(argvals, X)
        uni1 = UnivariateFunctionalData(argvals1, X1)
        multi = MultivariateFunctionalData([uni, uni1])
        self.assertEquals(multi.nObs(), 2)

    # Tests rangeObs function
    def test_rangeObs(self):
        argvals = np.array([1, 2, 3])
        argvals1 = [np.array([1, 2, 3]), np.array([1, 2])]
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X1 = np.array([[[1, 2], [3, 4], [5, 6]],
                       [[7, 8], [9, 10], [11, 12]]])
        uni = UnivariateFunctionalData(argvals, X)
        uni1 = UnivariateFunctionalData(argvals1, X1)
        multi = MultivariateFunctionalData([uni, uni1])
        self.assertEquals(multi.rangeObs(), [(1, 6), (1, 12)])

    # Tests nObsPoint function
    def test_nObsPoint(self):
        argvals = np.array([1, 2, 3])
        argvals1 = [np.array([1, 2, 3]), np.array([1, 2])]
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X1 = np.array([[[1, 2], [3, 4], [5, 6]],
                       [[7, 8], [9, 10], [11, 12]]])
        uni = UnivariateFunctionalData(argvals, X)
        uni1 = UnivariateFunctionalData(argvals1, X1)
        multi = MultivariateFunctionalData([uni, uni1])
        self.assertEquals(multi.nObsPoint(), [[3], [3, 2]])

    # Tests rangeObsPoint function
    def test_rangeObsPoint(self):
        argvals = np.array([1, 2, 3])
        argvals1 = [np.array([1, 2, 3]), np.array([1, 2])]
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X1 = np.array([[[1, 2], [3, 4], [5, 6]],
                       [[7, 8], [9, 10], [11, 12]]])
        uni = UnivariateFunctionalData(argvals, X)
        uni1 = UnivariateFunctionalData(argvals1, X1)
        multi = MultivariateFunctionalData([uni, uni1])
        self.assertEquals(multi.rangeObsPoint(), [[(1, 3)], [(1, 3), (1, 2)]])

    # Tests dimension function
    def test_dimension(self):
        argvals = np.array([1, 2, 3])
        argvals1 = [np.array([1, 2, 3]), np.array([1, 2])]
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X1 = np.array([[[1, 2], [3, 4], [5, 6]],
                       [[7, 8], [9, 10], [11, 12]]])
        uni = UnivariateFunctionalData(argvals, X)
        uni1 = UnivariateFunctionalData(argvals1, X1)
        multi = MultivariateFunctionalData([uni, uni1])
        self.assertEquals(multi.dimension(), [1, 2])


if __name__ == '__main__':
    unittest.main()
