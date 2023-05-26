#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for SmoothingSplines.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.preprocessing.smoothing.smoothing_splines import (
    SmoothingSpline
)


class SmoothingSplineTest(unittest.TestCase):
    def test_init(self):
        # Test default initialization
        spline = SmoothingSpline()
        self.assertTrue(np.isnan(spline.smooth))

        # Test custom initialization
        spline = SmoothingSpline(smooth=0.5)
        self.assertEqual(spline.smooth, 0.5)

    def test_smooth(self):
        spline = SmoothingSpline()
        spline.smooth = 0.7
        self.assertEqual(spline.smooth, 0.7)
