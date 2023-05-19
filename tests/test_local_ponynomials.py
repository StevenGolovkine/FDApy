#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class LocalPolynomial in the
local_polynomial.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.preprocessing.smoothing.local_polynomial import (
    _gaussian,
    _epanechnikov,
    _tri_cube,
    _bi_square
)


class TestGaussian(unittest.TestCase):
    def test_gaussian(self):
        x = np.array([0, 1, 2, 3])
        expected_result = np.array([0.39894228, 0.24197072, 0.05399097, 0.00443185])
        result = _gaussian(x)
        np.testing.assert_allclose(result, expected_result, rtol=10e-7)

    def test_gaussian_with_negative_values(self):
        x = np.array([-1, -2, -3])
        expected_result = np.array([0.24197072, 0.05399097, 0.00443185])
        result = _gaussian(x)
        np.testing.assert_allclose(result, expected_result, rtol=10e-7)


class TestEpanechnikov(unittest.TestCase):
    def test_epanechnikov(self):
        x = np.array([-1, -0.5, 0, 0.5, 1])
        expected_result = np.array([0.    , 0.5625, 0.75  , 0.5625, 0.    ])
        result = _epanechnikov(x)
        np.testing.assert_allclose(result, expected_result)

    def test_epanechnikov_with_negative_values(self):
        x = np.array([-3, -2, -1])
        expected_result = np.array([0, 0, 0])
        result = _epanechnikov(x)
        np.testing.assert_allclose(result, expected_result)

    def test_epanechnikov_with_large_values(self):
        x = np.array([-10, 10])
        expected_result = np.array([0, 0])
        result = _epanechnikov(x)
        np.testing.assert_allclose(result, expected_result)


class TestTriCube(unittest.TestCase):
    def test_tri_cube(self):
        x = np.array([-1, -0.5, 0, 0.5, 1])
        expected_result = np.array([0.        , 0.66992188, 1.        , 0.66992188, 0.        ])
        result = _tri_cube(x)
        np.testing.assert_allclose(result, expected_result)

    def test_tri_cube_with_negative_values(self):
        x = np.array([-3, -2, -1])
        expected_result = np.array([0, 0, 0])
        result = _tri_cube(x)
        np.testing.assert_allclose(result, expected_result)

    def test_tri_cube_with_large_values(self):
        x = np.array([-10, 10])
        expected_result = np.array([0, 0])
        result = _tri_cube(x)
        np.testing.assert_allclose(result, expected_result)


class TestBiSquare(unittest.TestCase):
    def test_bi_square(self):
        x = np.array([-1, -0.5, 0, 0.5, 1])
        expected_result = np.array([0.    , 0.5625, 1.    , 0.5625, 0.    ])
        result = _bi_square(x)
        np.testing.assert_allclose(result, expected_result)

    def test_bi_square_with_negative_values(self):
        x = np.array([-3, -2, -1])
        expected_result = np.array([0, 0, 0])
        result = _bi_square(x)
        np.testing.assert_allclose(result, expected_result)

    def test_bi_square_with_large_values(self):
        x = np.array([-10, 10])
        expected_result = np.array([0, 0])
        result = _bi_square(x)
        np.testing.assert_allclose(result, expected_result)
