#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the karhunen.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.simulation.karhunen import (
    _eigenvalues_linear,
    _eigenvalues_exponential,
    _eigenvalues_wiener
)


class TestEigenvaluesLinear(unittest.TestCase):
    def test_eigenvalues_exponential_default(self):
        expected_output = np.array(
            [1.0, 0.6666666666666666, 0.3333333333333333]
        )
        result = _eigenvalues_linear()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_linear_n4(self):
        # Test that the function returns the correct values for n=4
        expected = np.array([1., 0.75, 0.5, 0.25])
        result = _eigenvalues_linear(n=4)
        np.testing.assert_allclose(result, expected)
    
    def test_eigenvalues_linear_n1(self):
        # Test that the function returns the correct values for n=1
        expected = np.array([1.0])
        result = _eigenvalues_linear(n=1)
        np.testing.assert_allclose(result, expected)


class TestEigenvaluesExponential(unittest.TestCase):
    def test_eigenvalues_exponential_default(self):
        expected_output = np.array(
            [0.36787944117144233, 0.22313016014842982, 0.1353352832366127]
        )
        result = _eigenvalues_exponential()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_exponential_n4(self):
        expected_output = np.array(
            [0.36787944, 0.22313016, 0.13533528, 0.082085]
        )
        result = _eigenvalues_exponential(n=4)
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_exponential_n1(self):
        expected_output = np.array([0.36787944117144233])
        result = _eigenvalues_exponential(n=1)
        np.testing.assert_allclose(result, expected_output)


class TestEigenvaluesWiener(unittest.TestCase):
    def test_eigenvalues_wiener_default(self):
        expected_output = np.array(
            [0.4052847345693511, 0.04503163717437235, 0.016211389382774045]
        )
        result = _eigenvalues_wiener()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_wiener_n4(self):
        expected_output = np.array(
            [
                0.4052847345693511, 0.04503163717437235,
                0.016211389382774045, 0.008271117032027573
            ]
        )
        result = _eigenvalues_wiener(n=4)
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_wiener_n1(self):
        expected_output = np.array([0.4052847345693511])
        result = _eigenvalues_wiener(n=1)
        np.testing.assert_allclose(result, expected_output)
