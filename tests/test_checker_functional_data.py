#!/usr/bin/python3
# -*-cooding:utf8 -*
"""Module that contains unit tests for the checkers of the FunctionalData
classe.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

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


class TestCheckDictDict(unittest.TestCase):
    def test_coherent_dimensions(self):
        argv1 = {'a': {1: np.ones((5, 2)), 2: np.ones((5, 2))}}
        argv2 = {1: np.ones((5, 2)), 2: np.ones((5, 2))}
        _check_dict_dict(argv1, argv2)
        self.assertTrue(True)  # if no error is raised, test passed

    def test_incoherent_dimensions(self):
        argv1 = {'a': {1: np.ones((5, 2)), 2: np.ones((5, 2))}}
        argv2 = {1: np.ones((5, 3)), 2: np.ones((5, 2))}
        with self.assertRaises(ValueError):
            _check_dict_dict(argv1, argv2)


class TestCheckDictLen(unittest.TestCase):
    def test_equal_length(self):
        argv = {'a': {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])},
                'b': {0: np.array([7, 8, 9]), 1: np.array([10, 11, 12])}}
        _check_dict_len(argv)
        self.assertTrue(True)  # if no error is raised, test passed

    def test_unequal_length(self):
        argv = {'a': {0: np.array([1, 2, 3]), 1: np.array([4, 5])},
                'b': {0: np.array([7, 8, 9])}}
        with self.assertRaises(ValueError):
            _check_dict_len(argv)


class TestCheckSameType(unittest.TestCase):
    def test_same_type(self):
        argv1 = [1, 2, 3]
        argv2 = [4, 5, 6]
        _check_same_type(argv1, argv2)
        self.assertTrue(True)  # if no error is raised, test passed

    def test_different_type(self):
        argv1 = [1, 2, 3]
        argv2 = "hello"
        self.assertRaises(TypeError, _check_same_type, argv1, argv2)
