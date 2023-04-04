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
    _generate_pca,
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


class TestGapPrint(unittest.TestCase):
    def test_str(self):
        gap = Gap(n_jobs=2, parallel_backend=None)
        self.assertEqual(str(gap), 'Gap(n_jobs=1, parallel_backend=None)')
    
    def test_repr(self):
        gap = Gap(n_jobs=2, parallel_backend=None)
        self.assertEqual(repr(gap), 'Gap(n_jobs=1, parallel_backend=None)')
