#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the gap.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import pandas as pd
import unittest

from multiprocessing import cpu_count
from sklearn.cluster import KMeans

from FDApy.clustering.criteria.gap import (
    _GapResult,
    _compute_dispersion,
    _generate_uniform,
    _generate_pca,
    _clustering,
    _estimate_gap,
    Gap
)


class TestGapResults(unittest.TestCase):
    def test_attributes(self):
        gap_result = _GapResult(
            n_clusters=3,
            log_value=10.0, log_error=1.0,
            value=1.0, error=0.1
        )
        self.assertEqual(gap_result.n_clusters, 3)
        self.assertEqual(gap_result.log_value, 10.0)
        self.assertEqual(gap_result.log_error, 1.0)
        self.assertEqual(gap_result.value, 1.0)
        self.assertEqual(gap_result.error, 0.1)

    def test_repr(self):
        gap_result = _GapResult(
            n_clusters=3,
            log_value=10.0, log_error=1.0,
            value=1.0, error=0.1
        )
        self.assertEqual(repr(gap_result), "Number of clusters: 3 - Gap: 10.0")


class TestComputeDispersion(unittest.TestCase):
    def test_compute_dispersion(self):
        rnorm = np.random.default_rng(42).normal
        data = rnorm(0, 1, (10, 2))

        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(data)
        centroids = kmeans.cluster_centers_

        output = _compute_dispersion(data, labels, centroids)
        np.testing.assert_almost_equal(output, 5.982341862774169)


class TestGenerateUniform(unittest.TestCase):
    def test_generate_uniform(self):
        random = np.random.default_rng(42)
        data = random.normal(0, 1, (10, 2))
        
        output = _generate_uniform(data, random.uniform)
        expected_output = np.array([
            [ 0.19468149, -0.44088678],
            [ 0.79646069,  0.86758745],
            [ 0.25212728, -0.8293202 ],
            [-0.63001258, -1.19576173],
            [-1.51432909,  0.35723377],
            [ 0.15696432,  1.04830868],
            [-1.02880829, -0.40217702],
            [-0.62198885, -0.84187386],
            [-1.58330105, -0.14649211],
            [-1.30878344,  0.32508049]
        ])
        np.testing.assert_equal(output.shape, expected_output.shape)
        np.testing.assert_almost_equal(output, expected_output)


class TestGeneratePCA(unittest.TestCase):
    def test_generate_pca(self):
        random = np.random.default_rng(42)
        data = random.normal(0, 1, (10, 2))
        
        output = _generate_pca(data, random.uniform)
        expected_output = np.array([
            [-0.98638675, -1.01406288],
            [-0.78653804, -2.12402458],
            [-1.22769048, -0.90708657],
            [-0.73361217,  0.07637489],
            [ 0.73573877,  0.26570694],
            [-0.20195086, -1.59622967],
            [-0.02369329,  0.12308185],
            [-0.56029449, -0.0779803 ],
            [ 0.53339738,  0.53966164],
            [ 0.5609963 ,  0.08530684]
        ])
        np.testing.assert_equal(output.shape, expected_output.shape)
        np.testing.assert_almost_equal(output, expected_output)


class TestClustering(unittest.TestCase):
    def test_clustering(self):
        rnorm = np.random.default_rng(42).normal
        data = rnorm(0, 1, (10, 2))

        labels, centers = _clustering(data, 2, random_state=42)
        expected_labels = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        expected_centers = np.array([
            [ 0.64358254,  0.69891799],
            [-0.11650312, -0.88827087]
        ])
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_almost_equal(centers, expected_centers)


class TestEstimateGap(unittest.TestCase):
    def test_estimate_gap(self):
        dispersion = 10
        references = np.array([11, 11, 9])
        output = _estimate_gap(dispersion, references)
        np.testing.assert_almost_equal(output, (0.3333333333333339, 1.0886621079036347))


class TestGapInit(unittest.TestCase):     
    def test_init_non_default(self):
        # Test initialization with non-default values
        gap = Gap(
            n_jobs=1,
            parallel_backend=None,
            generating_process='uniform'
        )
        self.assertEqual(gap.n_jobs, 1)
        self.assertEqual(gap.parallel_backend, None)
        self.assertEqual(gap.generate_process, _generate_uniform)
        
    def test_init_error(self):
        # Test initialization with invalid parallel_backend value
        with self.assertRaises(ValueError):
            Gap(parallel_backend='invalid_backend')
        # Test initialization with invalid generate_process value
        with self.assertRaises(ValueError):
            Gap(generating_process='invalid_process')

    def test_init_n_jobs(self):
        # Test initialization with outside range n_jobs value
        gap = Gap(n_jobs=0)
        self.assertEqual(gap.n_jobs, 1)
        self.assertEqual(gap.parallel_backend, 'multiprocessing')
        self.assertEqual(gap.generate_process, _generate_pca)

        gap = Gap(n_jobs=2)
        self.assertEqual(gap.n_jobs, min(2, cpu_count()))
        self.assertEqual(gap.parallel_backend, 'multiprocessing')
        self.assertEqual(gap.generate_process, _generate_pca)
    
    def test_init_clusterer(self):
        # Default values
        gap = Gap()
        self.assertEqual(gap.clusterer, _clustering)
        self.assertEqual(gap.clusterer_kwargs, {'init': 'k-means++', 'n_init': 10})

        # Change clusterer
        def new_clusterer(data, n_clusters, **clusterer_kwargs):
            return np.array([0, 0]), np.array([0, 0])
        
        gap = Gap(clusterer=new_clusterer)
        self.assertEqual(gap.clusterer, new_clusterer)
        self.assertIsNone(gap.clusterer_kwargs)

        # Change clusterer_kwargs
        gap = Gap(clusterer_kwargs={'n_init': 15})
        self.assertEqual(gap.clusterer, _clustering)
        self.assertEqual(gap.clusterer_kwargs, {'n_init': 15})

        # Change both
        gap = Gap(clusterer=new_clusterer, clusterer_kwargs={'n_init': 15})
        self.assertEqual(gap.clusterer, new_clusterer)
        self.assertEqual(gap.clusterer_kwargs, {'n_init': 15})


class TestGapPrint(unittest.TestCase):
    def test_str(self):
        gap = Gap(n_jobs=2, parallel_backend=None)
        self.assertEqual(str(gap), 'Gap(n_jobs=1, parallel_backend=None)')
    
    def test_repr(self):
        gap = Gap(n_jobs=2, parallel_backend=None)
        self.assertEqual(repr(gap), 'Gap(n_jobs=1, parallel_backend=None)')


class TestComputeGap(unittest.TestCase):
    def test_compute_gap(self):
        random = np.random.default_rng(42)
        data = random.normal(0, 1, (10, 2))

        gap = Gap()
        gap_results = gap._compute_gap(data, 2, 3, runif=random.uniform)
        self.assertIsInstance(gap_results, _GapResult)
        np.testing.assert_equal(gap_results.n_clusters, 2)
        np.testing.assert_almost_equal(gap_results.log_value, -0.6982832252153512)
        np.testing.assert_almost_equal(gap_results.log_error, 0.25694825724342335)
        np.testing.assert_almost_equal(gap_results.value, -2.9352343877606577)
        np.testing.assert_almost_equal(gap_results.error, 0.7303479722776504)


class TestParallel(unittest.TestCase):
    def test_process_parallel(self):
        random = np.random.default_rng(42)
        data = random.normal(0, 1, (10, 2))
        cluster_array = [1, 2, 3]
        gap = Gap(n_jobs=2, parallel_backend='multiprocessing')

        gap_results = list(gap._process_with_multiprocessing(data, cluster_array))
        self.assertEqual(len(gap_results), 3)
        self.assertIsInstance(gap_results[0], _GapResult)


class TestNonParallel(unittest.TestCase):
    def test_process_non_parallel(self):
        random = np.random.default_rng(42)
        data = random.normal(0, 1, (10, 2))
        cluster_array = [1, 2, 3]
        gap = Gap(n_jobs=1, parallel_backend=None)

        gap_results = list(gap._process_non_parallel(data, cluster_array))
        self.assertEqual(len(gap_results), 3)
        self.assertIsInstance(gap_results[0], _GapResult)


class TestGap(unittest.TestCase):
    def setUp(self):
        self.random = np.random.default_rng(42)
        self.data = self.random.normal(0, 1, (10, 2))
        self.cluster_array = np.arange(1, 6)
        self.n_refs = 3
        self.gap = Gap(n_jobs=1, parallel_backend=None)

    def test_call_method(self) -> None:
        self.assertIsInstance(self.gap(self.data, self.cluster_array), np.int_)

    def test_call_multiprocessing(self):
        gap = Gap(n_jobs=2, parallel_backend='multiprocessing')
        gap.__call__(self.data, self.cluster_array)
        self.assertIsInstance(gap.n_clusters, np.int_)
        self.assertGreater(gap.n_clusters, 0)

    def test_gap_df_attr(self) -> None:
        self.gap(self.data, self.cluster_array)
        self.assertIsInstance(self.gap.gap, pd.DataFrame)
        self.assertEqual(self.gap.gap.shape[0], len(self.cluster_array))

    def test_n_clusters_attr(self) -> None:
        self.gap(self.data, self.cluster_array)
        self.assertIsInstance(self.gap.n_clusters, np.int_)
        self.assertGreater(self.gap.n_clusters, 0)
