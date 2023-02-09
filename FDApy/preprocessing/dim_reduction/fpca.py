#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Functional Principal Components Analysis
----------------------------------------

This module is used to compute UFPCA and MFPCA eigencomponents on the provided
functional data. Univariate functional data and irregular functional data are
concerned with UFPCA, whereas multivariate functional data with MFPCA.
"""
import numpy as np
import numpy.typing as npt

from typing import Optional, List, Union

from ...representation.functional_data import (
    DenseFunctionalData, MultivariateFunctionalData
)
from ...misc.utils import (
    _compute_covariance, _integrate, _integration_weights,
    _select_number_eigencomponents
)

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
    n_components: int, float, None, default=None
        Number of components to keep. If `n_components` is `None`, all
        components are kept, ``n_components == min(n_samples, n_features)``.
        If `n_components` is an integer, `n_components` are kept. If
        `0 < n_components < 1`, select the number of components such that the
        amount of variance that needs to be explained is greater than the
        percentage specified by `n_components`.
    method: str, {'covariance', 'inner-product'}, default='covariance'
        Method used to estimate the eigencomponents. If
        ``method == 'covariance'``, the estimation is based on an
        eigendecomposition of the covariance operator. If
        ``method == 'inner-product'``, the estimation is based on an
        eigendecomposition of the inner-product matrix.
    normalize: bool, default=False
        Perform a normalization of the data.

    Attributes
    ----------
    eigenvalues: npt.NDArray[np.float64], shape=(n_components,)
        The singular values corresponding to each of selected components.
    eigenfunctions: DenseFunctionalData
        Principal axes in feature space, representing the directions of
        maximum variances in the data as a DenseFunctionalData.
    mean: DenseFunctionalData
        An estimation of the mean of the training data
    covariance: DenseFunctionalData
        An estimation of the covariance of the training data based on the
        results of the functional principal components analysis.

    """

    def __init__(
        self,
        n_components: Union[int, float, None] = None,
        method: str = 'covariance',
        normalize: bool = False
    ) -> None:
        """Initaliaze UFPCA object."""
        self.n_components = n_components
        self.method = method
        self.normalize = normalize
        self.weights = 1

    def fit(
        self,
        data: DenseFunctionalData,
        compute_covariance: bool = False,
        **kwargs
    ) -> None:
        """Estimate the eigencomponents of the data.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data used to estimate the eigencomponents.
        compute_covariance: bool, default=False
            Should we compute an estimate of the covariance of the data using
            Mercer's theorem and the estimated eigenfunctions.

        Keyword Args
        ------------
        method: str, default=None
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

        if not isinstance(data, DenseFunctionalData):
            raise TypeError('UFPCA only support DenseFunctionalData object!')
        if self.method == 'covariance':
            self._fit_covariance(
                data,
                compute_covariance=compute_covariance
            )
        elif self.method == 'inner-product':
            self._fit_inner_product(
                data,
                compute_covariance=compute_covariance
            )
        else:
            raise NotImplementedError(
                f"{self.method} method not implemented."
            )

    def _fit_covariance(
        self,
        data: DenseFunctionalData,
        mean: Optional[DenseFunctionalData] = None,
        covariance: Optional[DenseFunctionalData] = None,
        compute_covariance: bool = False
    ) -> None:
        """Univariate Functional PCA.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data
        mean: DenseFunctionalData, default=None
            An estimation of the mean of the training data.
        covariance: DenseFunctionalData, default=None
            An estimation of the covariance of the training data.
        compute_covariance: bool, default=False
            Should we compute an estimate of the covariance of the data using
            Mercer's theorem and the estimated eigenfunctions?

        References
        ----------
        * Ramsey and Silverman, Functional Data Analysis, 2005, chapter 8
        * https://raw.githubusercontent.com/refunders/refund/master/R/fpca.sc.R

        """
        if self.normalize:
            data, weights = data.normalize(use_argvals_stand=True)
            self.weights = weights

        smoothing_method = self.smoothing_parameters['method']
        if mean is None:
            mean = data.mean(
                smooth=smoothing_method,
                **self.smoothing_parameters
            )
        if covariance is None:
            covariance = data.covariance(
                mean=mean,
                smooth=smoothing_method,
                **self.smoothing_parameters
            )

        # Choose the W_j's and the S_j's (Ramsey and Silverman, 2005)
        argvals = data.argvals['input_dim_0']
        weight = _integration_weights(argvals, method='trapz')

        # Compute the eigenvalues and eigenvectors of W^{1/2}VW^{1/2}
        weight_sqrt = np.diag(np.sqrt(weight))
        weight_invsqrt = np.diag(1 / np.sqrt(weight))
        var = np.dot(np.dot(weight_sqrt, covariance.values[0]), weight_sqrt)

        eigenvalues, eigenvectors = np.linalg.eigh(var)
        eigenvalues[eigenvalues < 0] = 0
        eigenvalues = eigenvalues[::-1]

        npc = _select_number_eigencomponents(eigenvalues, self.n_components)

        # Slice eigenvalues and compute eigenfunctions = W^{-1/2}U
        eigenvalues = eigenvalues[:npc]
        eigenfunctions = np.transpose(
            np.dot(weight_invsqrt, np.fliplr(eigenvectors)[:, :npc])
        )

        # Save the results
        self.eigenvalues = eigenvalues
        self.eigenfunctions = DenseFunctionalData(data.argvals, eigenfunctions)

        # Compute estimation of the covariance
        self.mean = mean
        if compute_covariance:
            covariance = _compute_covariance(
                eigenvalues, eigenfunctions
            )
            self.covariance = DenseFunctionalData(
                {'input_dim_0': argvals, 'input_dim_1': argvals},
                covariance[np.newaxis]
            )

    def _fit_inner_product(
        self,
        data: DenseFunctionalData,
        compute_covariance: bool = False
    ) -> None:
        """Univariate Functional PCA using inner-product matrix decomposition.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data used to estimate the eigencomponents.
        compute_covariance: bool, default=False
            Should we compute an estimate of the covariance of the data using
            Mercer's theorem and the estimated eigenfunctions?

        """
        # Compute inner-product matrix
        inner_mat = data.inner_product()

        # Diagonalization of the inner-product matrix
        eigenvalues, eigenvectors = np.linalg.eigh(inner_mat)

        # Estimation of the number of components
        eigenvalues = np.real(eigenvalues[::-1])
        eigenvalues[eigenvalues < 0] = 0
        n_components = _select_number_eigencomponents(
           eigenvalues, self.n_components
        )

        # Estimation of the eigenvalues
        eigenvalues = eigenvalues[:n_components]

        # Estimation of the eigenfunctions
        eigenvectors = np.real(np.fliplr(eigenvectors)[:, :n_components])
        eigenfunctions = (
           np.matmul(data.values.T, eigenvectors) / np.sqrt(eigenvalues)
        )

        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues / data.n_obs
        self.eigenfunctions = DenseFunctionalData(
           data.argvals, eigenfunctions.T
        )

        if compute_covariance:
            argvals = data.argvals['input_dim_0']
            covariance = _compute_covariance(
                eigenvalues / data.n_obs, eigenfunctions.T
            )
            self.covariance = DenseFunctionalData(
                {'input_dim_0': argvals, 'input_dim_1': argvals},
                covariance[np.newaxis]
            )

    def transform(
        self,
        data: DenseFunctionalData,
        method: str = 'NumInt',
        **kwargs
    ) -> npt.NDArray[np.float64]:
        r"""Apply dimensionality reduction to data.

        The functional principal components scores are defined as the
        projection of the observation :math:`X_i` on the eigenfunction
        :math:`\phi_k`. These scores are given by:

        .. math::
            c_{ik} = \int_{\mathcal{T}} \{X_i(t) - \mu(t)\}\phi_k(t)dt.

        This integrale can be estimated using two ways. First, if data are
        sampled on a common fine grid, the estimation is done using
        numerical integration. Second, the PACE (Principal Components through
        Conditional Expectation) algorithm [1]_ is used for sparse functional
        data. If the eigenfunctions have been estimated using the inner-product
        matrix, the scores can also be estimated using the formula

        .. math::
            c_{ik} = \sqrt{l_k}v_{ik},

        where :math:`l_k` and :math:`v_{k}` are the eigenvalues and
        eigenvectors of the inner-product matrix.

        Parameters
        ----------
        data: DenseFunctionalData object
            Data
        method: str, {'PACE', 'NumInt', '}
            Which method we should use for the estimation of the scores?

        Keyword Args
        ------------
        tol: np.float64, default=1e-4
            Tolerance parameter to prevent overflow to inverse a matrix.
        int_method: str, {'trapz', 'simpson'}, default='trapz'
            Method used to perform numerical integration.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_components)
            An array representing the projection of the data onto the basis of
            functions defined by the eigenfunctions.

        References
        ----------
        .. [1] Yao, Müller and Wang (2005), Functional Data Analysis for Sparse Longitudinal Data, Journal of the American Statistical Association, Vol. 100, No. 470.

        """
        parameters = {
            'tol': kwargs.get('tol', 1e-4),
            'int_method': kwargs.get('int_method', 'trapz')
        }

        # TODO: Add checkers
        if self.normalize:
            values = data.values / self.weights
            data = DenseFunctionalData(data.argvals, values)

        #data_unmean = data.values - self.mean.values
        if method == 'PACE':
            return self._pace(data, parameters['tol'])
        elif method == 'NumInt':
            return self._numerical_integration(data, parameters['int_method'])
        elif method == 'InnPro':
            return np.sqrt(data.n_obs * self.eigenvalues) * self.eigenvectors
        else:
            raise ValueError('Method not implemented!')

    def _pace(
        self,
        data: DenseFunctionalData,
        tol: np.float64 = 1e-4
    ) -> npt.NDArray[np.float64]:
        """Estimate scores using PACE algorithm.

        Parameters
        ----------
        data: DenseFunctionalData
            Data
        tol: np.float64, default=1e-4
            Tolerance parameter to prevent overflow to inverse a matrix.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_components)
            An array representing the projection of the data onto the basis of
            functions defined by the eigenfunctions.

        References
        ----------
        [1] Yao, Müller and Wang (2005), Functional Data Analysis for Sparse Longitudinal Data, Journal of the American Statistical Association, Vol. 100, No. 470.

        """
        noise = max(tol, data.var_noise)
        noise_mat = noise * np.eye(self.covariance.values[0].shape[0])
        sigma_inv = np.linalg.pinv(self.covariance.values[0] + noise_mat)
        return self.eigenvalues * np.linalg.multi_dot(
            [data.values, sigma_inv, self.eigenfunctions.values.T]
        )

    def _numerical_integration(
        self,
        data: DenseFunctionalData,
        method: str = "trapz"
    ) -> npt.NDArray[np.float64]:
        """Estimate scores using numerical integration.

        Parameters
        ----------
        data: DenseFunctionalData
            Data
        int_method: str, {'trapz', 'simpson'}, default='trapz'
            Method used to perform numerical integration.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_components)
            An array representing the projection of the data onto the basis of
            functions defined by the eigenfunctions.

        """
        return _integrate(
            x=data.argvals['input_dim_0'],
            y=[traj * self.eigenfunctions.values for traj in data.values],
            method=method
        )

    def inverse_transform(
        self,
        scores: np.ndarray
    ) -> DenseFunctionalData:
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
    normalize: bool, default=False
        Perform a normalization of the data.

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

    def __init__(
        self,
        n_components: List[Union[int, float, None]] = None,
        method: str = 'covariance',
        normalize: bool = False
    ) -> None:
        """Initialize MFPCA object."""
        self.n_components = n_components
        self.method = method
        self.normalize = normalize

    def fit(
        self,
        data: MultivariateFunctionalData,
        method: str = 'NumInt'
    ) -> None:
        """Estimate the eigencomponents of the data.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Training data used to estimate the eigencomponents.
        method: str, {'PACE', 'NumInt'}, default='NumInt'
            Method for the estimation of the univariate scores.

        """
        if not isinstance(data, MultivariateFunctionalData):
            raise TypeError(
                'MFPCA only support MultivariateFunctionalData object!'
            )
        if self.method == 'covariance':
            self._fit_covariance(data, method)
        elif self.method == 'inner-product':
            self._fit_inner_product(data)
        else:
            raise NotImplementedError(
                "f{self.method} method not implemented."
            )

    def _fit_covariance(
        self,
        data: MultivariateFunctionalData,
        method: str = 'NumInt'
    ) -> None:
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
                ufpca = UFPCA(n_components=n, normalize=self.normalize)
                ufpca.fit(data=function)
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

    def _fit_inner_product(
        self,
        data: MultivariateFunctionalData,
        compute_covariance: bool = False
    ) -> None:
        """Multivariate FPCA using inner-product matrix decomposition.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Training data used to estimate the eigencomponents.
        compute_covariance: bool, default=False
            Should we compute an estimate of the covariance of the data using
            Mercer's theorem and the estimated eigenfunctions.

        """
        # Compute inner-product matrix
        inner_mat = data.inner_product()

        # Diagonalization of the inner-product matrix
        eigenvalues, eigenvectors = np.linalg.eigh(inner_mat)

        # Estimation of the number of components
        eigenvalues = np.real(eigenvalues[::-1])
        eigenvalues[eigenvalues < 0] = 0
        n_components = _select_number_eigencomponents(
           eigenvalues, self.n_components
        )

        # Estimation of the eigenvalues
        eigenvalues = eigenvalues[:n_components]

        # Estimation of the eigenfunctions
        eigenvectors = np.real(np.fliplr(eigenvectors)[:, :n_components])
        eigenfunctions = [
            DenseFunctionalData(
                data_uni.argvals,
                np.transpose(
                    np.matmul(
                        data_uni.values.T, eigenvectors
                    ) / np.sqrt(eigenvalues)
                )
            ) for data_uni in data
        ]

        self.eigenvalues = eigenvalues / data.n_obs
        self.eigenfunctions = MultivariateFunctionalData(eigenfunctions)

        # if compute_covariance:
        #     argvals = data.argvals['input_dim_0']
        #     covariance = _compute_covariance(eigenvalues, eigenfunctions)
        #     self.covariance = DenseFunctionalData(
        #         {'input_dim_0': argvals, 'input_dim_1': argvals},
        #         covariance[np.newaxis]
        #     )

    def transform(
        self,
        data: MultivariateFunctionalData,
        method: str = 'NumInt'
    ) -> np.ndarray:
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

    def inverse_transform(
        self,
        scores: np.ndarray
    ) -> MultivariateFunctionalData:
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
