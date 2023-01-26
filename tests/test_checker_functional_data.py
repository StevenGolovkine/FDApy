#!/usr/bin/python3
# -*-cooding:utf8 -*
"""Module that contains unit tests for the checkers of the FunctionalData
classe.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData
)
from FDApy.representation.functional_data import (
    _check_dict_array,
    _check_dict_dict,
    _check_dict_len,
    _check_same_type
)


class TestCheckDictArray(unittest.TestCase):
    def test_coherent_dimensions(self):
        argv_dict = {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}
        argv_array = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        _check_dict_array(argv_dict, argv_array)
        self.assertTrue(True)  # if no error is raised, test passed

    def test_incoherent_dimensions(self):
        argv_dict = {'a': np.array([1, 2]), 'b': np.array([4, 5])}
        argv_array = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        with self.assertRaises(ValueError):
            _check_dict_array(argv_dict, argv_array)

    def test_incoherent_dimensions_2(self):
        argv_dict = {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6, 7])}
        argv_array = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            _check_dict_array(argv_dict, argv_array)

    def test_incoherent_dimensions_3(self):
        argv_dict = {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}
        argv_array = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
        with self.assertRaises(ValueError):
            _check_dict_array(argv_dict, argv_array)


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

    def test_check_dict_len(self):
        argvals = {'input_dim_0': {0: np.array([1, 2, 3, 4]),
                                   1: np.array([2, 4]),
                                   2: np.array([4, 5, 6])},
                   'input_dim_1': {0: np.array([5, 6, 7]),
                                   1: np.array([1, 2, 3])}}
        self.assertRaises(ValueError, _check_dict_len, argvals)

    def test_check_same_type(self):
        argvals = {'input_dim_0': np.array([1, 2, 3, 4])}
        values = np.array([[1, 2, 3, 4]])
        dense_fd = DenseFunctionalData(argvals, values)

        argvals = {'input_dim_0': {0: np.array([1, 2, 3, 4])}}
        values = {0: np.array([1, 2, 3, 4])}
        irregu_fd = IrregularFunctionalData(argvals, values)
        self.assertRaises(TypeError, _check_same_type, dense_fd, irregu_fd)
