#!/usr/bin/python3.7
# -*-cooding:utf8 -*

import numpy as np
import unittest

from FDApy.representation.functional_data import (DenseFunctionalData,
                                                  IrregularFunctionalData)
from FDApy.representation.functional_data import (_check_dict_array,
                                                  _check_dict_dict,
                                                  _check_type,
                                                  _check_dict_type,
                                                  _check_dict_len,
                                                  _check_same_type,
                                                  _check_same_nobs,
                                                  _check_same_ndim)
from FDApy.representation.functional_data import (
                                            _check_argvals_equality_dense,
                                            _check_argvals_equality_irregular)


class TestCheckerFunctionalData(unittest.TestCase):
    """Test class for the checkers in the class FunctionalData."""

    def test_check_dict_array(self):
        argvals = {'input_dim_0': np.array([1, 2, 3, 4])}
        values = np.array([[1, 2, 3, 4, 5]])
        self.assertRaises(ValueError, _check_dict_array, argvals, values)

    def test_check_dict_dict(self):
        argvals = {'input_dim_0': {0: np.array([1, 2, 3, 4])}}
        values = {0: np.array([1, 2, 3, 4, 5])}
        self.assertRaises(ValueError, _check_dict_dict, argvals, values)

    def test_check_type(self):
        values = [1, 2, 3]
        self.assertRaises(TypeError, _check_type, values, np.ndarray)

    def test_check_dict_type(self):
        values = {0: np.array([1, 2, 3, 4]),
                  1: np.array([5, 6]),
                  2: [8, 9, 7]}
        self.assertRaises(TypeError, _check_dict_type, values, np.ndarray)

    def test_check_dict_len(self):
        argvals = {'input_dim_0': {0: np.array([1, 2, 3, 4]),
                                   1: np.array([2, 4]),
                                   2: np.array([4, 5, 6])},
                   'input_dim_1': {0: np.array([5, 6, 7]),
                                   1: np.array([1, 2, 3])}}
        self.assertRaises(ValueError, _check_dict_len, argvals)

    def test_check_same_type(self):
        argvals = {'input_dim_0': np.array([1, 2, 3, 4])}
        values = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 9],
                           [3, 4, 5, 7]])
        dense_fd = DenseFunctionalData(argvals, values)

        argvals = {'input_dim_0': {0: np.array([1, 2, 3, 4]),
                                   1: np.array([2, 4]),
                                   2: np.array([0, 2, 3])}}
        values = {0: np.array([1, 2, 3, 4]),
                  1: np.array([5, 6]),
                  2: np.array([8, 9, 7])}
        irregu_fd = IrregularFunctionalData(argvals, values)
        self.assertRaises(TypeError, _check_same_type, dense_fd, irregu_fd)

    def test_check_compatible_dense(self):
        argvals = {'input_dim_0': np.array([1, 2, 3, 4]),
                   'input_dim_1': np.array([5, 6, 7])}
        values = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                           [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]]])
        dense_fd = DenseFunctionalData(argvals, values)

        argvals = {'input_dim_0': np.array([1, 2, 3, 4])}
        values = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 9],
                           [3, 4, 5, 7]])
        dense_fd2 = DenseFunctionalData(argvals, values)
        self.assertRaises(ValueError, _check_same_nobs, dense_fd, dense_fd2)
        self.assertRaises(ValueError, _check_same_ndim, dense_fd, dense_fd2)

    def test_check_argvals_equality_dense(self):
        argvals = {'input_dim_0': np.array([1, 2, 3, 4])}
        argvals2 = {'input_dim_0': np.array([1, 2, 3, 4, 5])}
        self.assertRaises(ValueError, _check_argvals_equality_dense,
                          argvals, argvals2)

    def test_check_compatible_irregular(self):
        argvals = {'input_dim_0': {0: np.array([1, 2, 3, 4])},
                   'input_dim_1': {0: np.array([5, 6, 7])}}
        values = {0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]])}
        irregu_fd = IrregularFunctionalData(argvals, values)

        argvals = {'input_dim_0': {0: np.array([1, 2, 3, 4]),
                                   1: np.array([2, 4])}}
        values = {0: np.array([1, 2, 3, 4]),
                  1: np.array([5, 6])}
        irregu_fd2 = IrregularFunctionalData(argvals, values)
        self.assertRaises(ValueError, _check_same_nobs, irregu_fd, irregu_fd2)
        self.assertRaises(ValueError, _check_same_ndim, irregu_fd, irregu_fd2)

    def test_check_argvals_equality_irregular(self):
        argvals = {'input_dim_0': {0: np.array([1, 2, 3, 4])}}
        argvals2 = {'input_dim_0': {0: np.array([1, 2, 3, 4, 5])}}
        self.assertRaises(ValueError, _check_argvals_equality_irregular,
                          argvals, argvals2)


if __name__ == '__main__':
    unittest.main()
