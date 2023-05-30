#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for FLMM.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.regression.flmm import (
    FLMM
)


class FLMMTest(unittest.TestCase):
    def setUp(self):
        self.flmm = FLMM(n_components=3, smooth="method")

    def test_n_components(self):
        self.assertEqual(self.flmm.n_components, 3)

        new_n_components = 5
        self.flmm.n_components = new_n_components
        self.assertEqual(self.flmm.n_components, new_n_components)

    def test_smooth(self):
        self.assertEqual(self.flmm.smooth, "method")

        new_smooth = "new_method"
        self.flmm.smooth = new_smooth
        self.assertEqual(self.flmm.smooth, new_smooth)
