#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""Module for UFPCA and MFPCA classes.

This module is used to compute UFPCA and MFPCA eigencomponents on the provided
functional data. Univariate functional data and irregular functional data are
concerned with UFPCA, whereas multivariate functional data with MFPCA.
"""
import numpy as np

from ...representation.functional_data import (DenseFunctionalData,
                                               MultivariateFunctionalData)
from ...misc.utils import integration_weights_

from .fcp_tpa import FCPTPA


#############################################################################
# Class UFPCA

class UFPCA():
    """Univariate Functional Principal Components Analysis (UFPCA).

    Linear dimensionality reduction of univariate functional data using
    Singular Value Decomposition of the data to project it to a lower
    dimensional space.

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

    Attributes
    ----------
    eigenvalues: array, shape = (n_components, )
        The singular values corresponding to each of selected components.
    eigenfunctions: DenseFunctionalData
        Principal axes in feature space, representing the directions of
        maximum variances in the dataas a DenseFunctionalData.
    mean: DenseFunctionalData
        An estimation of the mean of the training data
    covariance: DenseFunctionalData
        An estimation of the covariance of the training data based on the
        results of the functional principal components analysis.

    """

    def __init__(self, n_components=None):
        """Initaliaze UFPCA object."""
        self.n_components = n_components

    def fit(self, data, mean=None, covariance=None, **kwargs):
        """Fit the model on data.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data
        mean: DenseFunctionalData, default=None
            An estimation of the mean of the training data.
        covariance: DenseFunctionalData, default=None
            An estimation of the covariance of the training data.

        Keyword Args
        ------------
        method: str, default='LocalLinear'
            Smoothing method.
        kernel: str, default='gaussian'
            Kernel used for the smoothing.
        bandwidth: float, default=1.0
            Bandwidth used for the smoothing.
        degree: int, default=2
            Degree for the smoothing (LocalLinear).
        n_basis: int, default=2
            Number of basis used for the smoothing (GAM).

        """
        self.smoothing_parameters = {
            'method': kwargs.get('method', None),
            'kernel': kwargs.get('kernel', 'gaussian'),
            'bandwidth': kwargs.get('bandwidth', 1.0),
            'degree': kwargs.get('degree', 2),
            'n_basis': kwargs.get('n_basis', 10)
        }
        self._fit(data)

    def _fit(self, data, mean=None, covariance=None):
        """Dispatch to the right submethod depending on the input."""
        if isinstance(data, DenseFunctionalData):
            self._fit_uni(data, mean, covariance)
        else:
            raise TypeError('UFPCA only support DenseFunctionalData'
                            ' object!')

    def _fit_uni(self, data, mean=None, covariance=None):
        """Univariate Functional PCA.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data
        mean: DenseFunctionalData, default=None
            An estimation of the mean of the training data.
        covariance: DenseFunctionalData, default=None
            An estimation of the covariance of the training data.

        References
        ----------
        * Ramsey and Silverman, Functional Data Analysis, 2005, chapter 8
        * https://raw.githubusercontent.com/refunders/refund/master/R/fpca.sc.R

        Notes
        -----
        TODO : Add possibility to smooth the eigenfunctions.

        """
        smoothing_method = self.smoothing_parameters['method']
        if mean is None:
            mean = data.mean(smooth=smoothing_method,
                             **self.smoothing_parameters)
        if covariance is None:
            covariance = data.covariance(mean=mean,
                                         smooth=smoothing_method,
                                         **self.smoothing_parameters)

        # Choose the W_j's and the S_j's (Ramsey and Silverman, 2005)
        argvals = data.argvals['input_dim_0']
        weight = integration_weights_(argvals, method='trapz')

        # Compute the eigenvalues and eigenvectors of W^{1/2}VW^{1/2}
        weight_sqrt = np.diag(np.sqrt(weight))
        weight_invsqrt = np.diag(1 / np.sqrt(weight))
        var = np.dot(np.dot(weight_sqrt, covariance.values[0]), weight_sqrt)

        eigenvalues, eigenvectors = np.linalg.eigh(var)
        eigenvalues[eigenvalues < 0] = 0
        eigenvalues = eigenvalues[::-1]
        if isinstance(self.n_components, int):
            npc = self.n_components
        elif isinstance(self.n_components, float) and (self.n_components < 1):
            exp_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            npc = np.sum(exp_variance < self.n_components) + 1
        elif self.n_components is None:
            npc = len(eigenvalues)
        else:
            raise ValueError('Wrong n_components')

        # Slice eigenvalues and compute eigenfunctions = W^{-1/2}U
        eigenvalues = eigenvalues[:npc]
        eigenfunctions = np.transpose(np.dot(weight_invsqrt,
                                             np.fliplr(eigenvectors)[:, :npc]))
        # Compute estimation of the covariance
        temp = np.dot(np.transpose(eigenfunctions), np.diag(eigenvalues))
        cov = np.dot(temp, eigenfunctions)

        # Save the results
        new_argvals = {'input_dim_0': argvals}
        new_argvals_2 = {'input_dim_0': argvals, 'input_dim_1': argvals}
        self.eigenvalues = eigenvalues
        self.eigenfunctions = DenseFunctionalData(new_argvals, eigenfunctions)
        self.mean = mean
        self.covariance = DenseFunctionalData(new_argvals_2, cov[np.newaxis])

    def transform(self, data, method='PACE'):
        """Apply dimensionality reduction to data.

        The functional principal components scores are given by:
            c_ik = int (X_i(t) - mu(t))phi_k(t)dt

        Two methods are proposed to estimate these scores:
            * Numerical integration, works well in case of large density of
            the grid of measurements of each individuals.
            * PACE: Principal Components through Conditional Expectation,
            particularly suitable for sparse functional data.

        Parameters
        ----------
        data: UnivariateFunctionalData object
            Data
        method: 'PACE' or 'NumInt'
            Which method we should use for the estimation of the scores?

        Returns
        -------
        scores: np.ndarrray, shape=(n_obs, n_components)
            An array representing the projection of the data onto the basis of
            functions defined by the eigenfunctions.

        References
        ----------
        Yao, Müller and Wang (2005), Functional Data Analysis for Sparse
        Longitudinal Data, Journal of the American Statistical Association,
        Vol. 100, No. 470

        """
        # TODO: Add checkers
        data_unmean = data.values - self.mean.values

        if method == 'PACE':
            sigma_inv = np.linalg.inv(
                self.covariance.values[0] + data.var_noise * np.diagflat(
                    np.ones(shape=self.covariance.values[0].shape[0]))
            )
            scores = self.eigenvalues * np.dot(
                np.dot(data_unmean, sigma_inv), self.eigenfunctions.values.T)
        elif method == 'NumInt':
            prod = [traj * self.eigenfunctions.values for traj in data_unmean]
            # TODO: Modify to add other numrical integration methods
            scores = np.trapz(prod, data.argvals['input_dim_0'])
        else:
            raise ValueError('Method not implemented!')

        return scores

    def inverse_transform(self, scores):
        """Transform the data back to its original space.

        Return a DenseFunctionalData data_original whose transform would
        be `scores`.

        Parameters
        ----------
        scores: np.ndarray, shape=(n_obs, n_components)
            New data, where n_obs is the number of observations and
            n_components is the number of components.

        Returns
        -------
        data_original: DenseFunctionalData object
            The transformation of the scores into the original space.

        """
        argvals = self.eigenfunctions.argvals
        values = np.dot(scores, self.eigenfunctions.values)
        return DenseFunctionalData(argvals, values + self.mean.values)


#############################################################################
# Class MFPCA

class MFPCA():
    """Multivariate Functional Principal Components Analysis (MFPCA).

    Linear dimensionality reduction of multivariate functional data using
    Singular Value Decomposition of the data to project it to a lower
    dimension space.

    Parameters
    ----------
    n_components : list of {int, float, None}, default=None
        Number of components to keep for each functions in data.
        if n_components if None, all components are kept::
            n_components == min(n_samples, n_features)
        if n_components is int, n_components are kept.
        if 0 < n_components < 1, select the number of components such that
        the amount of variance that needs to be explained is greater than
        the percentage specified by n_components.

    Attributes
    ----------
    ufpca: list of UFPCA, shape=(data.n_functional,)
        List of UFPCA where the :math:`i`th entry is an object of the class
        UFPCA which the univariate functional PCA of the :math:`i` th process
        of the multivariate functional data.
    scores_univariate: list of np.ndarray, shape=(data.n_functional,)
        List of array containing the projection of the data onto the univariate
        functional principal components. The :math:`i`th entry of the list
        have the following shape (data.n_obs, ufpca[i].n_components)
    covariance: np.ndarray, shape = (data.n_functional, data.n_functional)
        Estimation of the covariance of the array scores_univariate.
    covariance_eigenvalues: np.ndarray, shape=(data.n_functional)
        Eigenvalues of the matrix covariance.
    eigenvectors: np.ndarray, shape=(data.n_functional, n_axis)
        The n_axis first eigenvectors of the matrix covariance.
    basis: MultivariateFunctionalData
        Multivariate basis of eigenfunctions.

    References
    ----------
    Happ and Greven, Multivariate Functional Principal Component Analysis for
    Data Observed on Different (Dimensional Domains), Journal of the American
    Statistical Association.

    """

    def __init__(self, n_components=None):
        """Initialize MFPCA object."""
        self.n_components = n_components

    def fit(self, data, method='NumInt'):
        """Fit the model with X.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Training data
        method: str, {'PACE', 'NumInt'}, default='NumInt'
            Method for the estimation of the univariate scores.

        """
        self._fit(data, method)

    def _fit(self, data, method='NumInt'):
        """Dispatch to the right submethod depending on the input."""
        # TODO: Different possiblity for n_components
        if isinstance(data, MultivariateFunctionalData):
            self._fit_multi(data, method)
        else:
            raise TypeError('MFPCA only support MultivariateFunctionalData'
                            ' object!')

    def _fit_multi(self, data, method='NumInt'):
        """Multivariate Functional PCA.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Training data.
        method: str, {'PACE', 'NumInt'}, default='PACE'
            Method for the estimation of the univariate scores.

        Notes
        -----
        TODO: Add a selection of the number of PC in FD. Maybe, consider the
        paper of Li, Wang and Carool (2013), Selecting the number of principal
        components in functional data.

        """
        # Step 1: Perform univariate fPCA on each functions.
        ufpca_list, scores = [], []
        for function, n in zip(data, self.n_components):
            if function.n_dim == 1:
                ufpca = UFPCA(n_components=n)
                ufpca.fit(data=function, method='GAM')
                scores_uni = ufpca.transform(data=function, method='NumInt')
            elif function.n_dim == 2:
                n_points = function.n_points
                mat_v = np.diff(np.identity(n_points['input_dim_0']))
                mat_w = np.diff(np.identity(n_points['input_dim_1']))
                ufpca = FCPTPA(n_components=n)
                ufpca.fit(function, penal_mat={'v': np.dot(mat_v, mat_v.T),
                                               'w': np.dot(mat_w, mat_w.T)},
                          alpha_range={'v': np.array([1e-4, 1e4]),
                                       'w': np.array([1e-4, 1e4])},
                          tol=1e-4, max_iter=15,
                          adapt_tol=True)
                scores_uni = ufpca.transform(function)
            ufpca_list.append(ufpca)
            scores.append(scores_uni)

        scores_univariate = np.concatenate(scores, axis=1)

        # Step 2: Estimation of the covariance of the scores.
        temp = np.dot(scores_univariate.T, scores_univariate)
        covariance = temp / (len(scores_univariate) - 1)

        # Step 3: Eigenanalysis of the covariance of the scores.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = np.fliplr(eigenvectors)

        # Step 4: Estimation of the multivariate eigenfunctions.
        # nb_axis = sum(eigenvalues.cumsum() / eigenvalues.sum() < n_components
        # eigenvectors = eigenvectors[:, :nb_axis]

        # Retrieve the number of eigenfunctions for each univariate function.
        nb_eigenfunction_uni = [0]
        for ufpca in ufpca_list:
            nb_eigenfunction_uni.append(len(ufpca.eigenvalues))
        nb_eigenfunction_uni_cum = np.cumsum(nb_eigenfunction_uni)

        # Compute the multivariate eigenbasis.
        basis_multi = []
        for idx, function in enumerate(ufpca_list):
            start = nb_eigenfunction_uni_cum[idx]
            end = nb_eigenfunction_uni_cum[idx + 1]

            argvals = function.eigenfunctions.argvals
            values = np.dot(function.eigenfunctions.values.T,
                            eigenvectors[start:end, :]).T
            basis_multi.append(DenseFunctionalData(argvals, values))

        self.ufpca_list = ufpca_list
        self.scores_univariate = scores_univariate
        self.covariance = covariance
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.basis = MultivariateFunctionalData(basis_multi)

    def transform(self, data, method='NumInt'):
        """Apply dimensionality reduction to data.

        Parameters
        ----------
        data: MultivariateFunctionalData
            The data to be projected onto the eigenfunctions.
        method: str, default='NumInt'
            The method used to estimate the projection.

        Returns
        -------
        scores: array-like

        """
        # TODO: Add checkers
        scores_uni = list()
        for data_uni, ufpca in zip(data, self.ufpca_list):
            scores_uni.append(ufpca.transform(data_uni, method=method))
        scores_uni = np.concatenate(scores_uni, axis=1)
        return np.dot(scores_uni, self.eigenvectors)

    def inverse_transform(self, scores):
        """Transform the data back to its original space.

        Return a MultivariateFunctionalData data_original whose transform would
        be `scores`.

        Parameters
        ----------
        scores: np.ndarray, shape=(n_obs, n_components)
            New data, where n_obs is the number of observations and
            n_components is the number of components.

        Returns
        -------
        data_original: MultivariateFunctionalData object
            The transformation of the scores into the original space.

        """
        res = []
        for idx, ufpca in enumerate(self.ufpca_list):
            if isinstance(ufpca, UFPCA):
                mean = ufpca.mean
                reconst = np.dot(scores, self.basis[idx].values) + mean.values
                res.append(DenseFunctionalData(mean.argvals, reconst))
            elif isinstance(ufpca, FCPTPA):
                reconst = np.einsum('ij, jkl', scores, self.basis[idx].values)
                res.append(DenseFunctionalData(ufpca.eigenfunctions.argvals,
                                               reconst))
            else:
                raise TypeError("Something went wrong with univariate "
                                "decomposition.")
        return MultivariateFunctionalData(res)
