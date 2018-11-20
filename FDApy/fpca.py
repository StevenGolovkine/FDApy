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

	def transform(self, X):
		"""Apply dimensionality reduction to X.
		
		Parameters
		----------
		X : FDApy.univariate_functional.UnivariateFunctionalData object
			Data

		Return
		------
		X_proj : array-like, shape = (n_samples, n_components)

		"""
		# TODO: Add checkers
		X_proj = np.dot(X.values, self.eigenfunctions.T)
		return X_proj


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
	ufpca_ : list of FDApy.UFPCA, shape = (X.nFunctions(),)
		List of FDApy.UFPCA where entry i is an object of the class FDApy.UFPCA which the univariate functional PCA of the function i of the multivariate functional data.
	uniScores_ : array-like, shape = (X.nObs(), X.nFunctions())
		List of array containing the projection of the data onto the univariate functional principal components.
	covariance_ : array_like, shape = (X.nFunctions(), X.nFunctions())
		Estimation of the covariance of the array uniScores_.
	eigenvaluesCovariance_ : array-like, shape = (X.nFunctions())
		Eigenvalues of the matrix covariance_.
	nbAxis_ : int
		Number of axis kept after the PCA of covariance_.
	eigenvectors_ : array-like, shape = (X.nFunctions(), nbAxis_)
		The nbAxis_ first eigenvectors of the matrix covariance_.
	basis_ : list, shape = (X.nFunctions())
		Multivariate basis of eigenfunctions.

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
		# TODO: Diffenrent possiblity for n_components
		if type(X) is FDApy.multivariate_functional.MultivariateFunctionalData:
			self._fit_multi(X, self.n_components, self.whiten)
		else:
			raise TypeError('MFPCA only support FDApy.multivariate_functional.MultivariateFunctionalData object!')

	def _fit_multi(self, X, n_components, whiten):
		"""Multivariate Functional PCA."""

		# Step 1: Perform univariate fPCA on each functions.
		ufpca = []
		scores = []		
		for function in X.data:
			uni = UFPCA(n_components, whiten)
			ufpca.append(uni.fit(function))
			scores.append(uni.transform(function))

		scores_ = np.concatenate(scores, axis=1)

		# Step 2: Estimation of the covariance of the scores.
		covariance = np.dot(scores_.T, scores_) / (len(scores_) - 1)

		# Step 3: Eigenanalysis of the covariance of the scores.
		eigenvalues, eigenvectors = np.linalg.eigh(covariance)
		eigenvalues = eigenvalues[::-1]
		eigenvectors = np.fliplr(eigenvectors)

		# Step 4: Estimation of the multivariate eigenfunctions.
		nb_axis = sum(eigenvalues.cumsum() / eigenvalues.sum() < n_components)
		eigenvectors = eigenvectors[:, :nb_axis]

		# Retrieve the number of eigenfunctions for each univariate funtion.
		nb_eigenfunction_uni = [0]
		for uni in ufpca:
			nb_eigenfunction_uni.append(len(uni.eigenvalues))
		nb_eigenfunction_uni_cum = np.cumsum(nb_eigenfunction_uni)

		# Compute the multivariate eigenbasis.
		basis_multi = []
		for idx, function in enumerate(ufpca):
			start = nb_eigenfunction_uni_cum[idx]
			end = nb_eigenfunction_uni_cum[idx+1]
			basis_multi.append(
				np.dot(function.eigenfunctions.T, eigenvectors[start:end, :]))

		self.ufpca_ = ufpca
		self.uniScores_ = scores_
		self.covariance_ = covariance
		self.eigenvaluesCovariance_ = eigenvalues
		self.nbAxis_ = nb_axis
		self.eigenvectors_ = eigenvectors
		self.basis_ = basis_multi

	def transform(self, X):
		"""Apply dimensionality reduction to X.
		
		Parameters
		----------
		X : FDApy.univariate_functional.Multivariate object
			Data

		Return
		------
		X_proj : array-like

		"""
		# TODO: Add checkers
		scores = []
		for idx, function in enumerate(X.data):
			scores.append(self.ufpca_[idx].transform(function))

		scores_ = np.concatenate(scores, axis=1)
		scores_multi = np.dot(scores_, self.eigenvectors_)

		return scores_multi
