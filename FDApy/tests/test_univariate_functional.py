#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import unittest

from FDApy.univariate_functional import _check_argvals


class TestUnivariateFunctionalData(unittest.TestCase):
	"""Test class for the class UnivariateFunctionalData.

	"""
	def test_check_argvals_type(self):
		argvals = np.array([1, 2, 3])
		self.assertRaises(ValueError, _check_argvals, argvals)

	def test_check_argvals_type2(self):
		argvals = [[1, 2, 3]]
		self.assertRaises(ValueError, _check_argvals, argvals)

	def test_check_argvals_numeric(self):
		argvals = [(1, 2.5, 3), (None, 5, 3)]
		self.assertRaises(ValueError, _check_argvals, argvals)

	def test_check_argvals_work(self):
		argvals = [(1, 2, 3), (4, 5, 6)]
		test = _check_argvals(argvals)
		self.assertEquals(len(test), 2)

	def test_check_argvals_work2(self):
		argvals = (1, 2, 3)
		test = _check_argvals(argvals)
		self.assertEquals(len(test), 1)

	"""
	def test_init_instance(self):
		X = [ [1,2,3] ]
		argvals = [1,2,3]
		self.assertRaises(ValueError, UnivariateFunctionalData, argvals, X)

	def test_init_numeric(self):
		X = np.array([[1, 2, 3], [4, 5, 6]])
		argvals = np.array([[1, None, 3]])
		self.assertRaises(ValueError, UnivariateFunctionalData, argvals, X)

	def test_init_dimensions(self):
		X = np.array([[1, 2, 3], [4, 5, 6]])
		argvals = np.array([[1, 2, 3], [4, 5, 6]])
		self.assertRaises(ValueError, UnivariateFunctionalData, argvals, X)

	def test_init_sampling(self):
		X = np.array([[1, 2, 3], [4, 5, 6]])
		argvals = np.array([[1, 3]])
		self.assertRaises(ValueError, UnivariateFunctionalData, argvals, X)
	"""


if __name__ == '__main__':
	unittest.main()