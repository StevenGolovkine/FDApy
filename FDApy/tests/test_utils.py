#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import unittest
from sklearn.preprocessing import StandardScaler

from FDApy.utils import *

class TestUtils(unittest.TestCase):
	""" Test class for the functions in utils.py
	"""

	# Test rowMean_ function
	def test_rowMean(self):
		X = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])
		mean_ = rowMean_(X)
		self.assertTrue(np.array_equal(mean_, np.array([1., 2., 3.])))

	# Test rowVar_ function
	def test_rowVar(self):
		X = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])
		var_ = rowVar_(X)
		self.assertTrue(np.array_equal(var_, np.array([0., 0., 0.])))

	# Test colMean_ function
	def test_colMean(self):
		X = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])
		mean_ = colMean_(X)
		self.assertTrue(np.array_equal(mean_, np.array([2., 2., 2., 2.])))

	# Test colVar_ function
	def test_colVar(self):
		X = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])
		var_ = colVar_(X)
		self.assertTrue(np.allclose(var_, np.array([2/3, 2/3, 2/3, 2/3])))

	# Test tensorProduct_ function
	def test_tensorProduct_(self):
		X = np.array([1, 2, 3])
		Y = np.array([-1, 2])
		tens_ = tensorProduct_(X, Y)
		self.assertTrue(
			np.array_equal(tens_, np.array([ [-1, 2], [-2, 4], [-3, 6]])))

	# Test integrate_ function
	def test_integrate_(self):
		X = np.array([1, 2, 4])
		Y = np.array([1, 4, 16])
		self.assertEqual(integrate_(X, Y), 21.0)