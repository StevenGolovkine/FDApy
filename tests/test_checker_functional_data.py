#!/usr/bin/python3
# -*-cooding:utf8 -*
"""Module that contains unit tests for the checkers of the FunctionalData
classe.

Written with the help of ChatGPT.

"""
import unittest

from FDApy.representation.functional_data import (
    _check_same_type
)


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
