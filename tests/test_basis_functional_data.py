#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the functional_data.py
file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from pathlib import Path

from FDApy.representation.argvals import DenseArgvals
from FDApy.representation.functional_data import BasisFunctionalData
from FDApy.representation.basis import Basis

THIS_DIR = Path(__file__)


class TestBasisFunctionalData(unittest.TestCase):
    def setUp(self):
        argvals = DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)})
        basis = Basis("fourier", n_functions=2, argvals=argvals)
        coefs = np.array([[1, 0.5], [0.5, 1]])
        self.func_data = BasisFunctionalData(basis, coefs)

    def test_n_obs(self):
        self.assertEqual(self.func_data.n_obs, 2)

    def test_n_dimension(self):
        self.assertEqual(self.func_data.n_dimension, 1)

    def test_n_points(self):
        expected_result = (11,)
        result = self.func_data.n_points
        np.testing.assert_equal(result, expected_result)
