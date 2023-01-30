#!/usr/bin/python3
# -*-cooding:utf8 -*
"""Module that contains unit tests for the checkers of the
IrregularFunctionalData classe.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import IrregularFunctionalData


class TestIrregularArgvalsLength(unittest.TestCase):
    def test_equal_length(self):
        argv = {
            'a': {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])},
            'b': {0: np.array([7, 8, 9]), 1: np.array([10, 11, 12])}
        }
        IrregularFunctionalData._check_argvals_length(argv)
        self.assertTrue(True)  # if no error is raised, test passed

    def test_unequal_length(self):
        argv = {
            'a': {0: np.array([1, 2, 3]), 1: np.array([4, 5])},
            'b': {0: np.array([7, 8, 9])}
        }
        with self.assertRaises(ValueError):
            IrregularFunctionalData._check_argvals_length(argv)


class TestIrregularCheckArgvalsEquality(unittest.TestCase):
    def test_equal_argvals(self):
        argv1 = {'x': {
            0: np.array([1, 2, 3]),
            1: np.array([4, 5])
        }}
        argv2 = {'x': {
            0: np.array([1, 2, 3]),
            1: np.array([4, 5])
        }}
        IrregularFunctionalData._check_argvals_equality(argv1, argv2)
        self.assertTrue(True)  # if no error is raised, test passed

    def test_unequal_argvals(self):
        argv1 = {'x': {
            0: np.array([1, 2, 3]),
            1: np.array([4, 6])
        }}
        argv2 = {'x': {
            0: np.array([1, 2, 3]),
            1: np.array([4, 7])
        }}
        with self.assertRaises(ValueError):
            IrregularFunctionalData._check_argvals_equality(argv1, argv2)


class TestIrregularArgvalsValues(unittest.TestCase):
    def test_coherent_dimensions(self):
        argv1 = {'a': {
            1: np.ones((5, 2)), 2: np.ones((5, 2))
        }}
        argv2 = {
            1: np.ones((5, 2)), 2: np.ones((5, 2))
        }
        IrregularFunctionalData._check_argvals_values(argv1, argv2)
        self.assertTrue(True)  # if no error is raised, test passed

    def test_incoherent_dimensions(self):
        argv1 = {'a': {
            1: np.ones((5, 2)), 2: np.ones((5, 2))
        }}
        argv2 = {
            1: np.ones((5, 3)), 2: np.ones((5, 2))
        }
        with self.assertRaises(ValueError):
            IrregularFunctionalData._check_argvals_values(argv1, argv2)
