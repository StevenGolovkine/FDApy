#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for FunctionalData.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.argvals import DenseArgvals
from FDApy.representation.values import DenseValues
from FDApy.representation.functional_data import DenseFunctionalData


class TestConcatenate(unittest.TestCase):
    def setUp(self):
        self.argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 5])})
        self.values = DenseValues(np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ]))
        self.func_data = DenseFunctionalData(self.argvals, self.values)
