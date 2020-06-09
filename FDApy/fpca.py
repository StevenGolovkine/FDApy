#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""Module for UFPCA and MFPCA classes.

This module is used to compute UFPCA and MFPCA eigencomponents on the provided
functional data. Univariate functional data and irregular functional data are
concerned with UFPCA, whereas multivariate functional data with MFPCA.
"""
import numpy as np

from .univariate_functional import UnivariateFunctionalData
from .multivariate_functional import MultivariateFunctionalData
from .utils import integrationWeights_


#############################################################################
# Class UFPCA

class UFPCA():
    """Univariate Functional Principal Components Analysis (UFPCA).

    Linear dimensionality reduction of univariate functional data using
    Singular Value Decomposition of the data to project it to a lower
    dimension space.

    It uses the PCA implementation of sklearn.

    Attributes
    ----------
    eigenfunctions : array, shape = (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variances in the data.
    eigenvalues : array, shape = (n_components, )
        The singular values corresponding to each of selected components.

    """

    def __init__(self, n_components=None):
        """Initaliaze UFPCA object.

        Parameters
        ----------
        n_components : int, float, None, default=None
            Number of components to keep.
            if n_components if None, all components are kept::
                n_components == min(n_samples, n_features)
            if n_components is int, n_components are kept.
            if 0 < n_components < 1, select the number of components such that
            the amount of variance that needs to be explained is greater than
            the percentage specified by n_components.

        """
        self.n_components = n_components

    def fit(self, X, **kwargs):
        """Fit the model with X.

        Parameters
        ----------
        X : UnivariateFunctionalData
            Training data

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.smoothing_parameters = {
            'method': kwargs.get('method', 'LocalLinear'),
            'kernel': kwargs.get('kernel', 'gaussian'),
            'bandwidth': kwargs.get('bandwidth', 1),
            'degree': kwargs.get('degree', 2),
            'n_basis': kwargs.get('n_basis', 10)
        }
        self._fit(X)
        return self

    def _fit(self, X):
        """Dispatch to the right submethod depending on the input."""
        if isinstance(X, UnivariateFunctionalData):
            self._fit_uni(X)
        else:
            raise TypeError(
                """UFPCA only support UnivariateFunctionalData object!""")

    def _fit_uni(self, X):
        """Univariate Functional PCA.

        Parameters
        ----------
        X: UnivariateFunctionalData
            Training data
        n_components : int, float, None, default=None
            Number of components to keep.
            if n_components if None, all components are kept::

            n_components == min(n_samples, n_features)

            if n_components is int, n_components are kept.
            if 0 < n_components < 1, select the number of components such that
            the amount of variance that needs to be explained is greater than
            the percentage specified by n_components.

        References
        ----------
        * Ramsey and Silverman, Functional Data Analysis, 2005, chapter 8
        * https://raw.githubusercontent.com/refunders/refund/master/R/fpca.sc.R

        Notes
        -----
        TODO : Add possibility to smooth the eigenfunctions

        """
        # Covariance estimation (also estimate the mean)
        if getattr(X, 'covariance_', None) is None:
            X.covariance(smooth=True, **self.smoothing_parameters)

        # Choose n, the wj's and the sj's (from Ramsey and Silverman, 2005)
        # N = X.nObsPoint()
        S = np.asarray(X.argvals).squeeze()
        W = integrationWeights_(S, method='trapz')

        # Compute the eigenvalues and eigenvectors of W^{1/2}VW^{1/2}
        Wsqrt = np.diag(np.sqrt(W))
        Winvsqrt = np.diag(1 / np.sqrt(W))
        V = np.dot(np.dot(Wsqrt, X.covariance_.values.squeeze()), Wsqrt)

        Evalues, Evectors = np.linalg.eigh(V)
        Evalues[Evalues < 0] = 0
        Evalues = Evalues[::-1]
        exp_variance = np.cumsum(Evalues) / np.sum(Evalues)
        npc = np.sum(exp_variance < self.n_components) + 1

        self.eigenvalues = Evalues[:npc]
        # Compute eigenfunction = W^{-1/2}U
        self.eigenfunctions = np.transpose(
            np.dot(Winvsqrt, np.fliplr(Evectors)[:, :npc]))

        # Estimation of the covariance
        self.covariance_hat = np.dot(
            np.dot(np.transpose(self.eigenfunctions),
                   np.diag(self.eigenvalues)),
            self.eigenfunctions)

        self.argvals = X.argvals
        self.mean = X.mean_

    def transform(self, X, method='PACE'):
        """Apply dimensionality reduction to X.

        The functional principal components scores are:
            c_ik = int (X_i(t) - mu(t))phi_k(t)dt

        Two methods are proposed to estimate these scores:
            * Numerical integration, works well in case of large density of
            the grid of measurements of each individuals.
            * PACE: Principal Components through Conditional Expectation,
            particularly suitable for sparse functional data.

        Parameters
        ----------
        X : UnivariateFunctionalData object
            Data
        method : 'PACE' or 'NumInt'
            Which method we should use for the estimation of the scores?

        Returns
        -------
        X_proj : array-like, shape = (n_samples, n_components)

        References
        ----------
        Yao, Müller and Wang (2005), Functional Data Analysis for Sparse
        Longitudinal Data, Journal of the American Statistical Association,
        Vol. 100, No. 470

        """
        # TODO: Add checkers
        X_unmean = X - self.mean
        if method == 'PACE':
            Sigma_inv = np.linalg.inv(
                self.covariance_hat + X.sigma2 * np.diagflat(
                    np.ones(shape=self.covariance_hat.shape[0]))
            )
            X_proj = self.eigenvalues * np.dot(
                np.dot(X_unmean.values, Sigma_inv), self.eigenfunctions.T)
        elif method == 'NumInt':
            prod = [traj * self.eigenfunctions for traj in X_unmean.values]
            # TODO: Modify to add other numrical integration methods
            X_proj = np.trapz(prod, X_unmean.argvals)
        else:
            raise ValueError('Method not implemented!')

        return X_proj

    def inverse_transform(self, X):
        """Transform the data back to its original space.

        Return a Univariate Functional data X_original whose transform would
        be X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_components)
            New data, where n_samples is the number of samples and n_components
            is the number of components.

        Returns
        -------
        X_original : UnivariateFunctionalData object

        """
        values = np.dot(X, self.eigenfunctions) + self.mean.values
        return UnivariateFunctionalData(self.argvals, values)


#############################################################################
# Class MFPCA

class MFPCA():
    """Multivariate Functional Principal Components Analysis (MFPCA).

    Linear dimensionality reduction of multivariate functional data using
    Singular Value Decomposition of the data to project it to a lower
    dimension space.

    It uses the PCA implementation of sklearn.

    Attributes
    ----------
    ufpca_ : list of UFPCA, shape = (X.nFunctions(),)
        List of UFPCA where entry i is an object of the class UFPCA which the
        univariate functional PCA of the function i of the multivariate
        functional data.
    uniScores_ : array-like, shape = (X.nObs(), X.nFunctions())
        List of array containing the projection of the data onto the univariate
        functional principal components.
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
    Happ and Greven, Multivariate Functional Principal Component Analysis for
    Data Observed on Different (Dimensional Domains), Journal of the American
    Statistical Association.

    """

    def __init__(self, n_components=None, method='PACE'):
        """Initialize MFPCA object.

        Parameters
        ----------
        n_components : list of integers of size X.nFunctions()
            int, float, None, default=None
            Number of components to keep.
            if n_components if None, all components are kept::
                n_components == min(n_samples, n_features)
            if n_components is int, n_components are kept.
            if 0 < n_components < 1, select the number of components such that
            the amount of variance that needs to be explained is greater than
            the percentage specified by n_components.
        method: str, default='PACE'
            Method for the estimation of the univariate scores.
            Should be 'PACE' or 'NumInt'.

        """
        self.n_components = n_components
        self.method = method

    def fit(self, X):
        """Fit the model with X.

        Parameters
        ----------
        X : MultivariateFunctionalData
            Training data

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def _fit(self, X):
        """Dispatch to the right submethod depending on the input."""
        # TODO: Different possiblity for n_components
        if isinstance(X, MultivariateFunctionalData):
            self._fit_multi(X, self.n_components, self.method)
        else:
            raise TypeError(
                'MFPCA only support MultivariateFunctionalData object!')

    def _fit_multi(self, X, n_components, method):
        """Multivariate Functional PCA.

        Notes
        -----
        TODO: Add a selection of the number of PC in FD. Maybe, consider the
        paper of Li, Wang and Carool (2013), Selecting the number of principal
        components in functional data.

        """
        # Step 1: Perform univariate fPCA on each functions.
        ufpca = []
        scores = []
        for function, n in zip(X.data, n_components):
            uni = UFPCA(n)
            ufpca.append(uni.fit(function))
            scores.append(uni.transform(function, method))

        scores_ = np.concatenate(scores, axis=1)

        # Step 2: Estimation of the covariance of the scores.
        covariance = np.dot(scores_.T, scores_) / (len(scores_) - 1)

        # Step 3: Eigenanalysis of the covariance of the scores.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = np.fliplr(eigenvectors)

        # Step 4: Estimation of the multivariate eigenfunctions.
        # nb_axis = sum(eigenvalues.cumsum() / eigenvalues.sum() < n_components
        # eigenvectors = eigenvectors[:, :nb_axis]

        # Retrieve the number of eigenfunctions for each univariate funtion.
        nb_eigenfunction_uni = [0]
        for uni in ufpca:
            nb_eigenfunction_uni.append(len(uni.eigenvalues))
        nb_eigenfunction_uni_cum = np.cumsum(nb_eigenfunction_uni)

        # Compute the multivariate eigenbasis.
        basis_multi = []
        for idx, function in enumerate(ufpca):
            start = nb_eigenfunction_uni_cum[idx]
            end = nb_eigenfunction_uni_cum[idx + 1]
            basis_multi.append(
                np.dot(function.eigenfunctions.T, eigenvectors[start:end, :]))

        self.ufpca_ = ufpca
        self.uniScores_ = scores_
        self.covariance_ = covariance
        self.eigenvaluesCovariance_ = eigenvalues
        # self.nbAxis_ = nb_axis
        self.eigenvectors_ = eigenvectors
        self.basis_ = basis_multi

    def transform(self, X):
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : FDApy.univariate_functional.Multivariate object
            Data

        Returns
        -------
        X_proj : array-like

        """
        # TODO: Add checkers
        scores_multi = np.dot(self.uniScores_, self.eigenvectors_)

        return scores_multi

    def inverse_transform(self, X):
        """Transform the data back to its original space.

        Return a Multivariate Functional data X_original whose transform would
        be X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_components)
            New data, where n_samples is the number of samples and n_components
            is the number of components.

        Returns
        -------
        X_original : UnivariateFunctionalData object

        Notes
        -----
        If whitening is enabled, inverse_tranform will compute the exact
        inverse operation, which includes reversing whitening.

        """
        res = []
        for idx, ufpca in enumerate(self.ufpca_):
            reconst = np.dot(X, self.basis_[idx].T) + ufpca.mean.values
            res.append(UnivariateFunctionalData(ufpca.argvals, reconst))

        return MultivariateFunctionalData(res)
