#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np

##############################################################################
# Inter functions for the LocalPolynomial class.

def _compute_kernel(x, x0, bandwith, kernel='gaussian'):
	"""Compute kernel at point (x - x0) / h.

	TODO: Add other kernels.
	
	Parameters
	----------
	kernel : string, default='gaussian'
		Kernel name used.
	x : array-like, shape = [n_samples]
		Training data, 1-D array.
	x0 : float
		Number around which compute the kernel.
	bandwith : float
		Bandwith to control the importance of points far from x0.
	
	Return
	------
	kernelMat : array-like , shape = [n_samples, n_samples]
	"""
	if kernel is 'gaussian':
		kernelMat = np.exp(-np.power(((x - x0) / bandwith),2) / 2) /\
					np.sqrt(2 * np.pi)
	else:
		raise ValueError(''.join[
			'The kernel `', kernel, '` is not implemented!'])

	return np.diag(kernelMat)

def _loc_poly(x, y, x0, kernel='gaussian', bandwith=0.05, degree=2):
	"""Local polynomial regression for one point.
	
	Let (x_1, Y_1), ...., (x_n, Y_n) be a random sample of bivariate data. Assume the following model: Y_i = f(x_i) + e_i. We would like to estimate the unknown regression function f(x) = E[Y | X = x]. We approximate f(x) using Taylor series.

	Parameters
	----------
	x : array-like, shape = [n_samples]
		1-D input array.
	y : array-like, shape = [n_samples]
		1-D input array such that y = f(x) + e.
	x0 : float
		1-D array on which estimate the function f(x). If None, the parameter x is used.
	kernel : string, default='gaussian'
		Kernel name used as weight.
	bandwith : float, default=0.05
		Bandwith for the kernel trick.
	degree : integer, default=2
		Degree of the local polynomial to fit.

	Return
	------
	y0_pred : float
		Prediction of y0, which is f(x0).

	References
	----------
	Zhang and Chen, Statistical Inferences for functional data, The Annals of Statistics, 1052-1079, No. 3, Vol. 35, 2007.

	"""
	# Compute kernel.
	kernelMat = _compute_kernel(x=x,
								x0=x0,
								bandwith=bandwith,
								kernel=kernel)
	# Compute the coefficients of the polynomials.
	X = np.vander(x=x-x0,
				  N=degree+1,
				  increasing=True)
	# Compute the estimation of f (and derivatives) at x0.
	beta = np.dot(
		np.linalg.inv(np.dot(np.dot(np.transpose(X), kernelMat), X)), 
		np.dot(np.dot(np.transpose(X), kernelMat), y)
	)

	return beta[0]

#############################################################################
# Class LocalPOlynomial


class LocalPolynomial():
	"""Local polynomial regression. 

	Let (x_1, Y_1), ...., (x_n, Y_n) be a random sample of bivariate data. Assume the following model: Y_i = f(x_i) + e_i. We would like to estimate the unknown regression function f(x) = E[Y | X = x]. We approximate f(x) using Taylor series.

	Parameters
	----------
	kernel : string, default="gaussian"
		Kernel name used as weight (default = 'gaussian').
	bandwith : float, default=0.05
		Strictly positive. Control the size of the associated neighborhood. 
	degree: integer, default=2
		Degree of the local polynomial to fit.

	Return
	------

	References
	----------
	Zhang and Chen, Statistical Inferences for functional data, The Annals of Statistics, 1052-1079, No. 3, Vol. 35, 2007.

	"""
	def __init__(self, kernel="gaussian", bandwith=0.05, degree=2):
		# TODO: Add test on parameters.
		self.kernel = kernel
		self.bandwith = bandwith
		self.degree = degree

	def fit(self, x, y):
		"""Fit local polynomial regression.
		
		Parameters:
		-----------
		x : array-like, shape = [n_samples]
			Training data, 1-D input array.
		y : array-like, shape = [n_samples]
			Target velues, 1-D input array

		Return
		------
		self : returns an instance of self. 		
		"""
		# TODO: Add tests on the parameters.
		self.X = x
		self.Y = y
		x0 = x
		self.X_fit_ = [_loc_poly(x, y, i, 
			self.kernel, self.bandwith, self.degree) for i in x0]

		return self

	def predict(self, X):
		""" Predict using local polynomial regression.

		Parameters
		----------
		X : array-like, shape = [n_samples]

		Return
		------
		y_pred : array-like, shape = [n_samples]
			Return predicted values.
		"""
		if type(X) in (int, float, np.int_, np.float_):
			X = [X]
		y_pred = [_loc_poly(self.X, self.Y, i, 
			self.kernel, self.bandwith, self.degree) for i in X]
		return y_pred

