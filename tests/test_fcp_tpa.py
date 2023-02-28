#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class FCPTPA in the fcp_tpa.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
import warnings

from FDApy.representation.functional_data import MultivariateFunctionalData
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fcp_tpa import (
    FCPTPA,
    _gcv
)


class TestGCV(unittest.TestCase):
    def test_gcv(self):
        alpha = 0.5
        dimension_length = 3
        vector = np.array([1, 2, 3])
        smoother = 0.9
        rayleigh = np.array([0.1, 0.2, 0.3])

        expected_output = 5.90094401041666
        output = _gcv(alpha, dimension_length, vector, smoother, rayleigh)
        np.testing.assert_almost_equal(output, expected_output)
