#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np

from sklearn.preprocessing import PolynomialFeatures

##############################################################################
# Inter functions for the LocalPolynomial class.
def _gaussian(t):
	"""Compute the gaussian density with mean 0 and stadard deviation 1.

	Parameters
	----------
	t : array-like, shape = [n_samples]
		Array at which computes the gaussian density

	Return
	------
	K : array-like, shape = [n_samples]
	"""
	return np.exp(-np.power(t, 2) / 2) / np.sqrt(2 * np.pi)

def _epanechnikov(t):
	"""Compute the Epanechnikov kernel.

	Parameters
	----------
	t : array-like, shape = [n_samples]
		Array on which computes the Epanechnikov kernel

	Return
	------
	K : array-like, shape = [n_samples]

	References
	----------
	Hastie, Tibshirani and Friedman, Elements of Statistical Learning, 2009, 
	equation 6.4 
	"""
	K = np.zeros(t.shape)
	idx = np.where(t < 1)
	K[idx] = 0.75 * (1 - np.power(t[idx], 2))
	return K

def _tri_cube(t):
	"""Compute the tri-cube kernel.

	Parameters
	----------
	t : array-like, shape = [n_samples]
		Array on which computes the tri-cube kernel

	Return
	------
	K : array-like, shape = [n_samples]

	References
	----------
	Hastie, Tibshirani and Friedman, Elements of Statistical Learning, 2009, 
	equation 6.6 
	"""
	K = np.zeros(t.shape)
	idx = np.where(t < 1)
	K[idx] = np.power((1 - np.power(np.abs(t[idx]), 3)), 3)
	return K

def _bi_square(t):
	"""Compute the bi-square kernel.

	Parameters
	----------
	t : array-like, shape = [n_samples]
		Array on which computes the bi-square kernel

	Return
	------
	K : array-like, shape = [n_samples]

	References
	----------
	Cleveland, Robust Locally Weighted Regression and Smoothing Scatterplots,
	1979, p.831
	"""
	K = np.zeros(t.shape)
	idx = np.where(t < 1)
	K[idx] = np.power((1 - np.power(t[idx], 2)), 2)
	return K

def _compute_kernel(x, x0, h, kernel='gaussian'):
	"""Compute kernel at point (x - x0) / h.
	
	Parameters
	----------
	kernel : string, default='gaussian'
		Kernel name used.
	x : array-like, shape = [n_dim, n_samples]
		Training data.
	x0 : float-array, shape= [n_dim,]
		Number around which compute the kernel.
	h : float
		Bandwidth to control the importance of points far from x0.
	
	Return
	------
	K : array-like , shape = [n_samples, n_samples]

	References
	----------
	Hastie, Tibshirani and Friedman, Elements of Statistical Learning, 2009, 
	equation 6.13 
	"""

	if not np.iterable(x0):
		x0 = np.asarray([x0])

	t = np.sqrt(np.sum(np.power(x - x0.T, 2), axis = 0)) / h

	if kernel is 'gaussian':
		K = _gaussian(t)
	elif kernel is 'epanechnikov':
		K = _epanechnikov(t)
	elif kernel is 'tricube':
		K = _tri_cube(t)
	elif kernel is 'bisquare':
		K = _bi_square(t)
	else:
		raise ValueError(''.join[
			'The kernel `', kernel, '` is not implemented!'])

	return np.diag(K.flatten())

def _loc_poly(x, y, x0, B, 
				kernel='gaussian', h=0.05, degree=2):
	"""Local polynomial regression for one point.
	
	Let (x_1, Y_1), ...., (x_n, Y_n) be a random sample of bivariate data. Assume 
	the following model: Y_i = f(x_i) + e_i. We would like to estimate the unknown
	regression function f(x) = E[Y | X = x]. We approximate f(x) using Taylor series.

	Parameters
	----------
	x : array-like, shape = [n_samples]
		1-D input array.
	y : array-like, shape = [n_samples]
		1-D input array such that y = f(x) + e.
	x0 : float
		1-D array on which estimate the function f(x). 
	B : array-like, shape = [n_sample, degree+1]
		Design matrix.
	kernel : string, default='gaussian'
		Kernel name used as weight.
	h : float, default=0.05
		Bandwidth for the kernel trick.
	degree : integer, default=2
		Degree of the local polynomial to fit.

	Return
	------
	y0_pred : float
		Prediction of y0, which is f(x0).

	References
	----------
	Zhang and Chen, Statistical Inferences for functional data, The Annals of
	Statistics, 1052-1079, No. 3, Vol. 35, 2007.

	"""
	x0 = np.array([x0], ndmin=2)

	# Compute kernel.
	K = _compute_kernel(x=x, x0=x0, h=h, kernel=kernel)

	# Compute the estimation of f (and derivatives) at x0.
	BtW = np.dot(B.T, K)
	beta = np.dot(np.linalg.pinv(np.dot(BtW, B)), np.dot(BtW, y))
	
	poly_features = PolynomialFeatures(degree=degree)
	B0 = poly_features.fit_transform(x0)

	return np.dot(B0, beta)[0]

#############################################################################
# Class LocalPolynomial


class LocalPolynomial():
	"""Local polynomial regression. 

	Let (x_1, Y_1), ...., (x_n, Y_n) be a random sample of bivariate data. 
	For all i, x_i belongs to R^d and Y_i in R. Assume the following model: 
	Y_i = f(x_i) + e_i. We would like to estimate the unknown regression 
	function f(x) = E[Y | X = x]. We approximate f(x) using Taylor series.

	Parameters
	----------
	kernel : string, default="gaussian"
		Kernel name used as weight (default = 'gaussian').
	bandwidth : float, default=0.05
		Strictly positive. Control the size of the associated neighborhood. 
	degree: integer, default=2
		Degree of the local polynomial to fit.

	Return
	------

	References
	----------
	* Zhang and Chen, Statistical Inferences for functional data, The Annals of
	Statistics, 1052-1079, No. 3, Vol. 35, 2007.
	* https://github.com/arokem/lowess/blob/master/lowess/lowess.py

	"""
	def __init__(self, kernel="gaussian", bandwidth=0.05, degree=2):
		# TODO: Add test on parameters.
		self.kernel = kernel
		self.bandwidth = bandwidth
		self.degree = degree
		self.poly_features = PolynomialFeatures(degree=degree)

	def fit(self, x, y):
		"""Fit local polynomial regression.

		Parameters:
		-----------
		x : array-like, shape = [n_dim, n_samples]
			Training data, input array.
		y : array-like, shape = [n_samples, ]
			Target values, 1-D input array

		Return
		------
		self : returns an instance of self. 		
		"""
		# TODO: Add tests on the parameters.
		self.X = np.array(x, ndmin=2)
		self.Y = y

		x0 = np.unique(self.X, axis=1).squeeze()
		
		design_matrix = self.poly_features.fit_transform(self.X.T)

		self.X_fit_ = np.array([_loc_poly(x, y, i, design_matrix,
			self.kernel, self.bandwidth, self.degree) for i in x0.T])

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

		design_matrix = self.poly_features.fit_transform(self.X.T)

		y_pred = np.array([_loc_poly(self.X, self.Y, i, design_matrix,
			self.kernel, self.bandwidth, self.degree) for i in X])

		return y_pred

