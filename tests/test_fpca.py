#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.preprocessing.dim_reduction.fpca import (
    _compute_inner_product
)


class TestComputeInnerProduct(unittest.TestCase):
    def setUp(self):
        self.argvals = {'input_dim_0': np.array([1, 2, 3, 4])}
        self.values = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.func_data = DenseFunctionalData(self.argvals, self.values)

    def test_compute_inner_product(self):
        exp_eigenvalues = np.array([150.12070758,   0.87929242])
        exp_eigenvectors = np.array(
            [
                [ 0.37171245, -0.92834792],
                [ 0.92834792,  0.37171245]
            ]
        )

        eigenvalues, eigenvectors = _compute_inner_product(
            self.func_data
        )
        np.testing.assert_array_almost_equal(eigenvalues, exp_eigenvalues)
        np.testing.assert_array_almost_equal(eigenvectors, exp_eigenvectors)

    def test_compute_inner_product_n_components_int(self):
        exp_eigenvalues = np.array([150.12070758])
        exp_eigenvectors = np.array([[0.37171245], [0.92834792]])

        eigenvalues, eigenvectors = _compute_inner_product(
            self.func_data, 1
        )
        np.testing.assert_array_almost_equal(eigenvalues, exp_eigenvalues)
        np.testing.assert_array_almost_equal(eigenvectors, exp_eigenvectors)

    def test_compute_inner_product_n_components_float(self):
        exp_eigenvalues = np.array([150.12070758])
        exp_eigenvectors = np.array([[0.37171245], [0.92834792]])

        eigenvalues, eigenvectors = _compute_inner_product(
            self.func_data, 0.9
        )
        np.testing.assert_array_almost_equal(eigenvalues, exp_eigenvalues)
        np.testing.assert_array_almost_equal(eigenvectors, exp_eigenvectors)
