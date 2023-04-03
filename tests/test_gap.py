#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the gap.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import pandas as pd
import unittest

from sklearn.cluster import KMeans

from FDApy.clustering.criteria.gap import (
    _GapResult,
    _compute_dispersion,
    _generate_uniform,
    _clustering,
    Gap
)


class TestGapResults(unittest.TestCase):
    def test_attributes(self):
        gap_result = _GapResult(n_clusters=3, value=10.0)
        self.assertEqual(gap_result.n_clusters, 3)
        self.assertEqual(gap_result.value, 10.0)

    def test_repr(self):
        gap_result = _GapResult(n_clusters=3, value=10.0)
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
        data = np.random.normal(0, 1, (10, 2))
        n_obs = 10
        low = 1.0
        high = 2.0
        runif = np.random.default_rng(42).uniform

        output = _generate_uniform(data, n_obs, low, high, runif)
        expected_output = np.array([[1.77395605, 1.43887844], [1.85859792, 1.69736803], [1.09417735, 1.97562235], [1.7611397 , 1.78606431], [1.12811363, 1.45038594], [1.37079802, 1.92676499], [1.64386512, 1.82276161], [1.4434142 , 1.22723872], [1.55458479, 1.06381726], [1.82763117, 1.6316644]])
        np.testing.assert_almost_equal(output, expected_output)


class TestClustering(unittest.TestCase):
    def test_clustering(self):
        rnorm = np.random.default_rng(42).normal
        data = rnorm(0, 1, (10, 2))

        labels, centers = _clustering(data, 2, random_state=42)
        expected_labels = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        np.testing.assert_array_equal(labels, expected_labels)


class TestGapPrint(unittest.TestCase):
    def test_str(self):
        gap = Gap(n_jobs=2, parallel_backend=None)
        self.assertEqual(str(gap), 'Gap(n_jobs=1, parallel_backend=None)')
    
    def test_repr(self):
        gap = Gap(n_jobs=2, parallel_backend=None)
        self.assertEqual(repr(gap), 'Gap(n_jobs=1, parallel_backend=None)')
