#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np 


class UnivariateFunctionalData(object):
	"""An object for defining Univariate Functional Data.

	Parameters
	----------
	argvals : list
		An list of numeric vectors or a single numeric vector giving the sampling points in the domains.

	X : array-like
		An array, giving the observed values for N observations. Missing values should be included via `None` (or `np.nan`). The shape depends on `argvals`::

			(N, M) if `argvals` is a single numeric vector,
			(N, M_1, ..., M_d) if `argvals` is a list of numeric vectors.

	Attributes
	----------

	Notes
	-----

	References
	----------

	"""
	def __init__(self, argvals, X):

		if not all([all(i) for i in argvals]):
			raise ValueError('All argvals elements must be numeric!')
		if len(argvals) != len(X.shape[1:]):
			raise ValueError('argvals and X elements have different support dimensions!')
		if tuple(len(i) for i in argvals) == X.shape[1:]:
			raise ValueError('argvals and X have different number of sampling points!')
		self.argvals = argvals
		self.X = X
