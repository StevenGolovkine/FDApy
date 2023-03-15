#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the bic.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
import warnings

from FDApy.clustering.criteria.bic import (
    _BICResult,
    _compute_bic
)


class TestBICResult(unittest.TestCase):
    def test_attributes(self):
        bic_result = _BICResult(n_cluster=3, value=10.0)
        self.assertEqual(bic_result.n_cluster, 3)
        self.assertEqual(bic_result.value, 10.0)

    def test_repr(self):
        bic_result = _BICResult(n_cluster=3, value=10.0)
        self.assertEqual(repr(bic_result), "Number of clusters: 3 - BIC: 10.0")


class TestComputeBIC(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.default_rng(42)

    def test_compute_bic(self):
        data = self.random_state.multivariate_normal(
            mean=np.array([0, 0]),
            cov=np.array([[1, 0], [0, 1]]),
            size=100
        )
        bic = _compute_bic(data, 5, random_state=42)
        self.assertIsInstance(bic, _BICResult)
        self.assertEqual(bic.n_cluster, 5)
        self.assertAlmostEqual(bic.value, 606.5293291803285)
