#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import scipy

import FDApy


#############################################################################
# Definition of the basis (eigenfunctions)

def basis_legendre(M=3, argvals=None, norm=True):
	"""Define Legendre basis of function.
	
	Build a basis of `M` functions using Legendre polynomials on the interval `argvals`.

	Parameters
	----------
	M : int, default = 3
		Maximum degree of the Legendre polynomials. 
	argvals : tuple or numpy.ndarray, default = None
		The values on which evaluated the Legendre polynomials. If `None`, the polynomials are evaluated on the interval [-1, 1].
	norm : boolean, default = True
		Do we normalize the functions?

	Return
	------
	obj : FDApy.univariate_functional.UnivariateFunctionalData
		A UnivariateFunctionalData object containing the Legendre polynomial up to `M` functions evaluated on `argvals`.
	
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

		if norm:
			legendre = legendre / np.sqrt(scipy.integrate.simps(
						legendre * legendre, argvals))
		values[degree, :] = legendre

	obj = FDApy.univariate_functional.UnivariateFunctionalData(
			tuple(argvals), values)
	return obj

def basis_wiener(M=3, argvals=None, norm=True):
	"""Define Wiener basis of function.

	Build a basis of functions of the Wiener process.

	Parameters
	----------
	degree : int, default = 3
		Number of functions to compute.
	argvals : tuple or numpy.ndarray, default = None
		 The values on which evaluated the Wiener basis functions. If `None`, the functions are evaluated on the interval [0, 1].
	norm : boolean, default = True
		Do we normalize the functions?

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

	for degree in np.linspace(1, M, M):
		wiener = np.sqrt(2) * np.sin( (degree - 0.5) * np.pi * argvals)

		if norm:
			wiener = wiener / np.sqrt(scipy.integrate.simps(
						wiener * wiener, argvals))

		values[int(degree-1), :] = wiener

	obj = FDApy.univariate_functional.UnivariateFunctionalData(
			tuple(argvals), values)
	return obj 

#############################################################################
# Definition of the eigenvalues

def eigenvalues_linear(M=3):
	"""Function that generate linear decreasing eigenvalues.

	Parameters
	----------
	M : int, default = 3
		Number of eigenvalues to generates

	Return
	------
	val : list
		The generated eigenvalues 
	"""
	return [(M - m + 1) / M for m in range(M)]

def eigenvalues_exponential(M=3):
	"""Function that generate exponential decreasing eigenvalues.

	Parameters
	----------
	M : int, default = 3
		Number of eigenvalues to generates

	Return
	------
	val : list
		The generated eigenvalues 
	"""
	return [np.exp(-(m+1)/2) for m in range(M)]

def eigenvalues_wiener(M=3):
	"""Function that generate exponential decreasing eigenvalues.

	Parameters
	----------
	M : int, default = 3
		Number of eigenvalues to generates

	Return
	------
	val : list
		The generated eigenvalues 
	"""
	return [np.exp(-(m+1)/2) for m in np.linspace(1, M, M)]

#############################################################################
# Class Simulation

class Simulation(object):
	"""An object to simulate functional data.

	The function are simulated using the Karhunen-Loève decomposition :
		X_i(t) = \mu(t) + \sum_{j = 1}^M c_{i,j}\phi_{i,j}(t), i = 1, ..., N

	Parameters:
	-----------
	basis : str
		String which denotes the basis of functions to use.
	M : int
		Number of basis functions to use to simulate the data.
	eigenvalues : str
		Define the decreasing if the eigenvalues of the process.
	noise : boolean, default = True
		Do we add noise to the data?

	Attributes
	----------

	Notes
	-----

	References
	---------

	"""
	def __init__(self, basis, M, eigenvalues, noise=True):
		self.basis = basis
		self.M = M
		self.eigenvalues = eigenvalues
		self.noise = noise


	def new(self, argvals, N):
		"""Function that simulates `N` observations
		
		Parameters
		----------
		argvals : list of tuples
			A list of numeric vectors (tuples) or a single numeric vector (tuple) giving the sampling points in the domains.
		N : int
			Number of observations to generate.

		"""

		# Simulate the basis
		if self.basis == 'legendre':
			basis_ = basis_legendre(self.M, argvals, norm=True)
		elif self.basis == 'wiener':
			basis_ = basis_wiener(self.M, argvals, norm=True)
		else:
			raise ValueError('Basis not implemented!')

		# Define the decreasing of the eigenvalues
		if self.eigenvalues == 'linear':
			eigenvalues_ = eigenvalues_linear(self.M)
		elif self.eigenvalues == 'exponential':
			eigenvalues_ = eigenvalues_exponential(self.M)
		elif self.eigenvalues == 'wiener':
			eigenvalues_ = eigenvalues_wiener(self.M)
		else:
			raise ValueError('Eigenvalues not implemented!')

		# Simulate the coefficients
		coef_ = list(np.random.normal(0, eigenvalues_))

		prod_ = coef_ * basis_

		res = FDApy.univariate_functional.UnivariateFunctionalData(
			prod_.argvals, np.array(prod_.values.sum(axis=0), ndmin=2))
		return res