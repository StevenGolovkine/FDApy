#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import scipy

import FDApy

def basis_legendre(M=3, argvals=None):
	"""Define Legendre basis of function.
	
	Build a basis of `M` functions using Legendre polynomials on the interval `argvals`.

	Parameters
	----------
	M : int, default = 3
		Maximum degree of the Legendre polynomials. 
	argvals : tuple or numpy.ndarray, default = None
		The values on which evaluated the Legendre polynomials. If `None`, the polynomials are evaluated on the interval [-1, 1].

	Return
	------
	obj : FDApy.univariate_functional.UnivariateFunctionalData
		A UnivariateFunctionalData object containing the Legendre polynomial up to `M` functions evaluated on `argvals`.
	
	Reference
	---------
	* https://stackoverflow.com/questions/39537794/orthogonality-issue-in-scipys-legendre-polynomials

	"""

	if argvals is None:
		argvals = np.arange(-1, 1, 0.1)

	if isinstance(argvals, list):
		raise ValueError('argvals has to be a tuple or a numpy array!')

	if isinstance(argvals, tuple):
		argvals = np.array(argvals)

	values = np.empty((M, len(argvals)))

	for degree in range(M):
		legendre = scipy.special.eval_legendre(degree, argvals)
		values[degree, :] = legendre

	obj = FDApy.univariate_functional.UnivariateFunctionalData(
			tuple(argvals), values)
	return obj

def basis_wiener(M=3, argvals=None):
	"""Define Wiener basis of function.

	Build a basis of functions of the Wiener process.

	Parameters
	----------
	degree : int, default = 3
		Number of functions to compute.
	argvals : tuple or numpy.ndarray, default = None
		 The values on which evaluated the Wiener basis functions. If `None`, the functions are evaluated on the interval [0, 1].

	Return
	------
	obj : FDApy.univariate_functional.UnivariateFunctionalData
		A UnivariateFunctionalData object containing `M` Wiener basis functions evaluated on `argvals`.

	"""
	if argvals is None:
		argvals = np.arange(0, 1, 0.05)

	if isinstance(argvals, list):
		raise ValueError('argvals has to be a tuple or a numpy array!')

	if isinstance(argvals, tuple):
		argvals = np.array(argvals)

	values = np.empty((M, len(argvals)))

	for degree in range(M):
		wiener = np.sqrt(2) * np.sin( (degree - 0.5) * np.pi * argvals)
		values[degree, :] = wiener

	obj = FDApy.univariate_functional.UnivariateFunctionalData(
			tuple(argvals), values)
	return obj 

#############################################################################
# Class Simulation

class Simulation(object):
	"""An object to simulate functional data.

	The function are simulated using the Karhunen-Lo\`eve decomposition :
		X_i(t) = \mu(t) + \sum_{j = 1}^M c_{i,j}\phi_{i,j}(t), i = 1, ..., N

	Paramaters:
	-----------
	N : int
		Number of functions to simulate
	basis : str
		String which denotes the basis of functions to use.
	M : int
		Number of basis functions to use to simulate the data.
	noise : boolean, default = True
		Do we add noise to the data?

	Attributes
	----------

	Notes
	-----

	References
	---------

	"""
