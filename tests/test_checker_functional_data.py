#!/usr/bin/python3
# -*-cooding:utf8 -*
"""Module that contains unit tests for the checkers of the FunctionalData
classe.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import (
    FunctionalData, DenseFunctionalData, IrregularFunctionalData
)
from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
from FDApy.representation.values import DenseValues, IrregularValues


class TestCheckSameType(unittest.TestCase):
    def setUp(self):
        # define DenseFunctionalData
        self.x = np.linspace(0, 1, num=10)
        self.y = np.random.randn(3, 10)
        self.argvals = {'input_dim_0': self.x}
        self.dense_fda = DenseFunctionalData(DenseArgvals(self.argvals), DenseValues(self.y))

        # define IrregularFunctionalData
        self.x = IrregularArgvals({
            0: DenseArgvals({'input_dim_0': np.array([1, 2, 3])}),
            1: DenseArgvals({'input_dim_0': np.array([1, 2])})
        })
        self.y = IrregularValues({0: np.array([4, 5, 6]), 1: np.array([2, 4])})
        self.irreg_data = IrregularFunctionalData(self.x, self.y)

    def test_same_type(self):
        FunctionalData._check_same_type(self.dense_fda, self.dense_fda)
        FunctionalData._check_same_type(self.irreg_data, self.irreg_data)
        self.assertTrue(True)  # if no error is raised, test passed

    def test_different_type(self):
        with self.assertRaises(TypeError):
            FunctionalData._check_same_type(self.dense_fda, self.irreg_data)
