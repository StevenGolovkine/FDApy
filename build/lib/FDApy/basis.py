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
	M : int, default = 3
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

def simulate_basis_(basis_name, M, argvals, norm):
	"""Function that redirects to the right simulation basis function.

	Parameters
	----------
	basis_name : str
		Name of the basis to use.
	M : int
		Number of functions to compute.
	argvals : tuple or numpy.ndarray
		 The values on which evaluated the Wiener basis functions. If `None`, the functions are evaluated on the interval [0, 1].
	norm : boolean
		Do we normalize the functions?

	Return
	------
	basis_ : FDApy.univariate_functional.UnivariateFunctionalData
		A UnivariateFunctionalData object containing `M` basis functions evaluated on `argvals`.

	"""
	if basis_name == 'legendre':
		basis_ = basis_legendre(M, argvals, norm)
	elif basis_name == 'wiener':
		basis_ = basis_wiener(M, argvals, norm)
	else:
		raise ValueError('Basis not implemented!')
	return basis_


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

def simulate_eigenvalues_(eigenvalues_name, M):
	"""Function that redirects to the right simulation eigenvalues function.

	Parameters
	----------
	eigenvalues_name : str
		Name of the eigenvalues generation process to use.
	M : int
		Number of eigenvalues to generates

	"""
	if eigenvalues_name == 'linear':
		eigenvalues_ = eigenvalues_linear(M)
	elif eigenvalues_name == 'exponential':
		eigenvalues_ = eigenvalues_exponential(M)
	elif eigenvalues_name == 'wiener':
		eigenvalues_ = eigenvalues_wiener(M)
	else:
		raise ValueError('Eigenvalues not implemented!')
	return eigenvalues_

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

	Attributes
	----------

	Notes
	-----

	References
	---------

	"""
	def __init__(self, basis, M, eigenvalues):
		self.basis = basis
		self.M = M
		self.eigenvalues = eigenvalues


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
		basis_ = simulate_basis_(self.basis, self.M, argvals, norm=True)

		# Define the decreasing of the eigenvalues
		eigenvalues_ = simulate_eigenvalues_(self.eigenvalues, self.M)

		# Simulate the N observations
		obs = np.empty(shape=(N, len(argvals)))
		coef = np.empty(shape=(N, len(eigenvalues_)))
		for i in range(N):
			coef_ = list(np.random.normal(0, eigenvalues_))
			prod_ = coef_ * basis_
			
			obs[i, :] = prod_.values.sum(axis=0)
			coef[i, :] = coef_

		# Simulate K clusters into the data.
		#K = 8
		#if K % 2 == 1:
		#	mu = np.array([i/2 if i % 2 == 0 else -(i+1)/2 
		#		for i in np.arange(0, K, step=1)])
		#else:
		#	mu = np.array([i/2 if i % 2 == 0 else -(i+1)/2 
		#		for i in np.arange(1, K+1, step=1)])

		self.coef_ = coef
		self.obs = FDApy.univariate_functional.UnivariateFunctionalData(
			argvals, obs)

	def add_noise(self, noise_var):
		"""Add noise to the data.

		Parameters
		----------
		noise_var : float
			Variance of the noise to add.

		"""

		noisy_data = []
		for i in self.obs:
			noise = np.random.normal(0, np.sqrt(noise_var), 
				size=len(self.obs.argvals[0]))
			noise_func = FDApy.univariate_functional.UnivariateFunctionalData(self.obs.argvals, np.array(noise, ndmin=2))
			noisy_data.append(i + noise_func)

		data = FDApy.multivariate_functional.MultivariateFunctionalData(noisy_data)

		self.noisy_obs = data.asUnivariateFunctionalData()
