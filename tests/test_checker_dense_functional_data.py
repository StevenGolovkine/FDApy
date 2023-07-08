#!/usr/bin/python3
# -*-cooding:utf8 -*
"""Module that contains unit tests for the checkers of the DenseFunctionalData
classe.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import DenseFunctionalData


class TestDenseCheckArgvalsValues(unittest.TestCase):
    def test_coherent_dimensions(self):
        argv_dict = {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}
        argv_array = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        DenseFunctionalData._check_argvals_values(argv_dict, argv_array)
        self.assertTrue(True)  # if no error is raised, test passed

    def test_incoherent_dimensions(self):
        argv_dict = {'a': np.array([1, 2]), 'b': np.array([4, 5])}
        argv_array = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        with self.assertRaises(ValueError):
            DenseFunctionalData._check_argvals_values(argv_dict, argv_array)

    def test_incoherent_dimensions_2(self):
        argv_dict = {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6, 7])}
        argv_array = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            DenseFunctionalData._check_argvals_values(argv_dict, argv_array)

    def test_incoherent_dimensions_3(self):
        argv_dict = {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}
        argv_array = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
        with self.assertRaises(ValueError):
            DenseFunctionalData._check_argvals_values(argv_dict, argv_array)
