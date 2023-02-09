#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for FunctionalData.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData,
    _concatenate
)


class TestDenseFunctionalData(unittest.TestCase):
    def setUp(self):
        self.argvals = {'input_dim_0': np.array([1, 2, 3, 4, 5])}
        self.values = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        self.func_data = DenseFunctionalData(self.argvals, self.values)

        argvals = {'input_dim_0': {
            0: np.array([1, 2, 3, 4]),
            1: np.array([2, 4])
        }}
        values = {
            0: np.array([1, 6, 9, 4]),
            1: np.array([2, 3])
        }
        self.irreg_data = IrregularFunctionalData(argvals, values)