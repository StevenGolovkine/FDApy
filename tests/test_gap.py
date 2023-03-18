#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the gap.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import pandas as pd
import unittest

from FDApy.clustering.criteria.gap import (
    _GapResult,
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


class TestGapPrint(unittest.TestCase):
    def test_str(self):
        gap = Gap(n_jobs=2, parallel_backend=None)
        self.assertEqual(str(gap), 'Gap(n_jobs=1, parallel_backend=None)')
    
    def test_repr(self):
        gap = Gap(n_jobs=2, parallel_backend=None)
        self.assertEqual(repr(gap), 'Gap(n_jobs=1, parallel_backend=None)')
