#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import sklearn

import FDApy

#############################################################################
# Class UFPCA

class UFPCA():
	"""Univariate Functional Principal Components Analysis (UFPCA)

	Linear dimensionality reduction of univariate functional data using Singular Value Decomposition of the data to project it to a lower dimension space.

	It uses the PCA implementation of sklearn.

	Parameters
	----------
	n_components : int, float, None, default=None
		Number of components to keep.
		if n_components if None, all components are kept::

			n_components == min(n_samples, n_features)

		if n_components is int, n_components are kept.
		if 0 < n_components < 1, select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.

	whiten : bool, default=False
		When True (False by default) the `components_` vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.

	Attributes
	----------
	eigenfunctions : array, shape = (n_components, n_features)
		Principal axes in feature space, representing the directions of maximum variances in the data.
	eigenvalues : array, shape = (n_components, )
		The singular values corresponding to each of selected components.

	References
	----------
	"""
	def __init__(self, n_components=None, whiten=False):
		self.n_components = n_components
		self.whiten = whiten

	def fit(self, X):
		"""Fit the model with X.

		Parameters
		----------
		X : FDApy.univariate_fonctional.UnivariateFunctionalData
			Training data

		Return
		------
		self : object
			Returns the instance itself.
		"""
		self._fit(X)
		return self


	def _fit(self, X):
		"""Dispatch to the right submethod depending on the input."""
		if type(X) is FDApy.univariate_functional.UnivariateFunctionalData:
			self._fit_uni(X, self.n_components, self.whiten)
		else:
			raise TypeError('UFPCA only support FDApy.univariate_fonctional.UnivariateFunctionalData object!')

	def _fit_uni(self, X, n_components, whiten):
		"""Univariate Functional PCA."""
		pca = sklearn.decomposition.PCA(
				n_components=n_components, 
				whiten=whiten)
		pca.fit(X.values)

		self.eigenfunctions = pca.components_
		self.eigenvalues = pca.singular_values_

#############################################################################
# Class MFPCA

class MFPCA():
	"""Multivariate Functional Principal Components Analysis (MFPCA)

	Linear dimensionality reduction of multivariate functional data using Singular Value Decomposition of the data to project it to a lower dimension space.

	It uses the PCA implementation of sklearn.

	Parameters
	----------
	n_components : int, float, None, default=None
		Number of components to keep.
		if n_components if None, all components are kept::

			n_components == min(n_samples, n_features)

		if n_components is int, n_components are kept.
		if 0 < n_components < 1, select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.

	whiten : bool, default=False
		When True (False by default) the `components_` vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.

	Attributes
	----------
	eigenfunctions : array, shape = (n_components, n_features)
		Principal axes in feature space, representing the directions of maximum variances in the data.
	eigenvalues : array, shape = (n_components, )
		The singular values corresponding to each of selected components.

	References
	----------
	Happ and Greven, Multivariate Functional Principal Component Analysis for Data Observed on Different (Dimensional Domains), Journal of the American Statistical Association.

	"""
	def __init__(self, n_components=None, whiten=False):
		self.n_components = n_components
		self.whiten = whiten

	def fit(self, X):
		"""Fit the model with X.

		Parameters
		----------
		X : FDApy.multivariate_functional.MultivariateFunctionalData
			Training data

		Return
		------
		self : object
			Returns the instance itself.
		"""
		self._fit(X)
		return self


	def _fit(self, X):
		"""Dispatch to the right submethod depending on the input."""
		if type(X) is FDApy.multivariate_functional.MultivariateFunctionalData:
			self._fit_multi(X, self.n_components, self.whiten)
		else:
			raise TypeError('MFPCA only support FDApy.multivariate_functional.MultivariateFunctionalData object!')

	def _fit_multi(self, X, n_components, whiten):
		"""Multivariate Functional PCA."""

		# Step 1: Perform univariate fPCA on each functions.
		
		# Step 2: Estimation of the covariance of the scores.

		# Step 3: Eigenanalysis of the covariance of the scores.

		# Step 4: Estimation of the multivariate eigenfucntions and scores.
		raise NotImplementedError('Not implemented yet!')