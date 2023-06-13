#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the datasets.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.simulation.datasets import (
    _zhang_chen,
    Datasets
)


class ZhangChangTestCase(unittest.TestCase):
    def test_zhang_chang_shape(self):
        n_obs = 100
        argvals = np.linspace(0, 1, 10)
        results = _zhang_chen(n_obs, argvals)
        expected_shape = (n_obs, len(argvals))
        self.assertEqual(results.shape, expected_shape)

    def test_zhang_chang_cos_sin(self):
        n_obs = 2
        argvals = np.linspace(0, 1, 10)
        rnorm = np.random.default_rng(42).normal

        results = _zhang_chen(n_obs, argvals, rnorm=rnorm)
        expected_results = np.array([[ 2.63139013,  5.02482289,  6.60973235,  5.8997611 ,  2.48634443, -1.16218953, -4.02114032, -3.39676508, -1.05722496,  2.36348728], [ 5.4050085 ,  6.01901495,  5.81901528,  3.17679982,  0.40184728, -1.65134235, -1.00260541,  0.10582   ,  2.66644022,  5.13092035]])
        np.testing.assert_array_almost_equal(results, expected_results, decimal=3)


class DatasetsTestCase(unittest.TestCase):
    def test_new_zhang_chen(self):
        basis_name = 'zhang_chen'
        random_state = 42
        dataset = Datasets(basis_name, random_state)

        n_obs = 100
        argvals = np.linspace(0, 1, 10)

        dataset.new(n_obs, argvals=argvals)

        self.assertIsInstance(dataset.data, DenseFunctionalData)
        np.testing.assert_almost_equal(dataset.data.argvals['input_dim_0'], argvals)
        np.testing.assert_equal(dataset.data.values.shape, (n_obs, len(argvals)))

    def test_new_not_implemented(self):
        basis_name = 'unknown_basis'
        random_state = 42
        dataset = Datasets(basis_name, random_state)

        n_obs = 100
        argvals = np.linspace(0, 1, 10)

        with self.assertRaises(NotImplementedError):
            dataset.new(n_obs, argvals=argvals)
