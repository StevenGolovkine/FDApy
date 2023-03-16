#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the bic.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import pandas as pd
import unittest
import warnings

from multiprocessing import cpu_count

from FDApy.clustering.criteria.bic import (
    _BICResult,
    _compute_bic,
    BIC
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


class TestBICInit(unittest.TestCase):
    def test_init_default(self):
        # Test initialization with default values
        bic = BIC()
        self.assertEqual(bic.n_jobs, cpu_count())
        self.assertEqual(bic.parallel_backend, 'multiprocessing')
        
    def test_init_non_default(self):
        # Test initialization with non-default values
        bic = BIC(n_jobs=2, parallel_backend='multiprocessing')
        self.assertEqual(bic.n_jobs, 2)
        self.assertEqual(bic.parallel_backend, 'multiprocessing')
        
    def test_init_error(self):
        # Test initialization with invalid parallel_backend value
        with self.assertRaises(ValueError):
            BIC(parallel_backend='invalid_backend')
    
    def test_init_n_jobs(self):
        # Test initialization with outside range n_jobs value
        bic = BIC(n_jobs=0)
        self.assertEqual(bic.n_jobs, 1)

        bic = BIC(n_jobs=cpu_count() + 5)
        self.assertEqual(bic.n_jobs, cpu_count())


class TestBICPrint(unittest.TestCase):
    def test_str(self):
        n_jobs = cpu_count()
        bic = BIC()
        self.assertEqual(str(bic), f'BIC(n_jobs={n_jobs}, parallel_backend=multiprocessing)')
        
        bic = BIC(n_jobs=2, parallel_backend=None)
        self.assertEqual(str(bic), 'BIC(n_jobs=1, parallel_backend=None)')
    
    def test_repr(self):
        n_jobs = cpu_count()
        bic = BIC()
        self.assertEqual(repr(bic), f'BIC(n_jobs={n_jobs}, parallel_backend=multiprocessing)')
        
        bic = BIC(n_jobs=2, parallel_backend=None)
        self.assertEqual(repr(bic), 'BIC(n_jobs=1, parallel_backend=None)')


class TestParallel(unittest.TestCase):
    def test_process_parallel(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        cluster_array = [1, 2, 3]
        bic = BIC(n_jobs=1, parallel_backend='multiprocessing')

        bic_results = list(bic._process_with_multiprocessing(data, cluster_array))
        self.assertEqual(len(bic_results), 3)
        self.assertIsInstance(bic_results[0], _BICResult)


class TestNonParallel(unittest.TestCase):
    def test_process_non_parallel(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        cluster_array = [1, 2, 3]
        bic = BIC(n_jobs=1, parallel_backend=None)

        bic_results = list(bic._process_non_parallel(data, cluster_array))
        self.assertEqual(len(bic_results), 3)
        self.assertIsInstance(bic_results[0], _BICResult)


class TestBIC(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(100, 5)
        self.n_clusters = np.arange(1, 6)
        self.bic = BIC(n_jobs=-1, parallel_backend='multiprocessing')

    def test_call_method(self) -> None:
        self.assertIsInstance(self.bic(self.data, self.n_clusters), np.int_)

    def test_bic_df_attr(self) -> None:
        self.bic(self.data, self.n_clusters)
        self.assertIsInstance(self.bic.bic, pd.DataFrame)
        self.assertEqual(self.bic.bic.shape[0], len(self.n_clusters))

    def test_n_clusters_attr(self) -> None:
        self.bic(self.data, self.n_clusters)
        self.assertIsInstance(self.bic.n_clusters, np.int_)
        self.assertGreater(self.bic.n_clusters, 0)
