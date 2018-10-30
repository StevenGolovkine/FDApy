#!/usr/bin/python3.7
# -*-coding:utf8 -*

import itertools
import numpy as np 


###############################################################################
# Checkers used by the UnivariateFunctionalData class.

def _check_argvals(argvals):
	"""Check the user provided `argvals`.
	
	Parameters
	---------
	argvals : list of tuples
		A list of numeric vectors (tuples) or a single numeric vector (tuple) giving the sampling points in the domainns. 

	Return
	------
	argvals : list of tuples
	"""
	if type(argvals) not in (tuple, list):
		raise ValueError('argvals has to be a list of tuples or a tuple.')

	if isinstance(argvals, list) and not isinstance(argvals[0], tuple):
		raise ValueError('argvals has to be a list of tuples or a tuple.')

	if isinstance(argvals, tuple):
		print('argvals is convert into one dimensional list.')
		argvals = [argvals]

	# Check if all entries of `argvals` are numeric. 
	argvals_ = list(itertools.chain.from_iterable(argvals))
	if not all([type(i) in (int, float) for i in argvals_]):
		raise ValueError('All argvals elements must be numeric!')

	return argvals


###############################################################################
# Class Univariate FunctionalData 
class UnivariateFunctionalData(object):
	"""An object for defining Univariate Functional Data.

	Parameters
	----------
	argvals : list of tuples
		A list of numeric vectors (tuples) or a single numeric vector (tuple) giving the sampling points in the domains.

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

		# TO DO: Modify the condition to deal with other types of data.
		if not isinstance(X, np.ndarray):
			raise ValueError('X have to be instance of numpy.ndarray!')

		if isinstance(argvals, tuple):
			argvals = [argvals]

		if not all([all(i) for i in argvals]):
			raise ValueError('All argvals elements must be numeric!')
		if len(argvals) != len(X.shape[1:]):
			raise ValueError('argvals and X elements have different support dimensions!')
		if tuple(len(i) for i in argvals) != X.shape[1:]:
			raise ValueError('argvals and X have different number of sampling points!')

		self.argvals = argvals
		self.X = X
