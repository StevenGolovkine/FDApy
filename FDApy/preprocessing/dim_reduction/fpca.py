#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Functional Principal Components Analysis
----------------------------------------

This module is used to compute the eigencomponents of a functional dataset.
The class ``UFPCA`` concerns the decomposition of ``DenseFunctionalData``
object. The class ``MFPCA`` concerns the decomposition of
``MultivariateFunctionalData`` object. In both methods, multidimensional data
can be considered (such as curves, images).

"""
import numpy as np
import numpy.typing as npt
import warnings

from typing import Optional, List, Tuple, Union

from ...representation.functional_data import (
    FunctionalData, DenseFunctionalData, MultivariateFunctionalData
)
from ...misc.utils import (
    _compute_covariance, _integrate, _integration_weights,
    _select_number_eigencomponents
)

from .fcp_tpa import FCPTPA


#############################################################################
# Class UFPCA

class UFPCA():
    """UFPCA -- Univariate Functional Principal Components Analysis.

    Linear dimensionality reduction of a univariate functional dataset. The
    projection of the data in a lower dimensional space is performed using
    a diagomalization of the covariance operator or of the inner-product matrix
    of the data.

    Parameters
    ----------
    method: str, {'covariance', 'inner-product'}, default='covariance'
        Method used to estimate the eigencomponents. If
        ``method == 'covariance'``, the estimation is based on an
        eigendecomposition of the covariance operator. If
        ``method == 'inner-product'``, the estimation is based on an
        eigendecomposition of the inner-product matrix.
    n_components: Union[int, float, None], default=None
        Number of components to keep. If `n_components` is `None`, all
        components are kept, ``n_components == min(n_samples, n_features)``.
        If `n_components` is an integer, `n_components` are kept. If
        `0 < n_components < 1`, select the number of components such that the
        amount of variance that needs to be explained is greater than the
        percentage specified by `n_components`.
    normalize: bool, default=False
        Perform a normalization of the data.

    Attributes
    ----------
    mean: DenseFunctionalData
        An estimation of the mean of the training data.
    covariance: DenseFunctionalData
        An estimation of the covariance of the training data based on their
        eigendecomposition using the Mercer's theorem.
    eigenvalues: npt.NDArray[np.float64], shape=(n_components,)
        The singular values corresponding to each of selected components.
    eigenfunctions: DenseFunctionalData
        Principal axes in feature space, representing the directions of
        maximum variance in the data.

    """

    def __init__(
        self,
        method: str = 'covariance',
        n_components: Union[int, float, None] = None,
        normalize: bool = False
    ) -> None:
        """Initaliaze UFPCA object."""
        self.n_components = n_components
        self.method = method
        self.normalize = normalize
        self.weights = 1

    def fit(
        self,
        data: DenseFunctionalData
    ) -> None:
        """Estimate the eigencomponents of the data.

        Before estimating the eigencomponents, the data is centered. Using the
        covariance operator, the estimation is based on [RS]_.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data used to estimate the eigencomponents.

        References
        ----------
        .. [RS] Ramsey, J. O. and Silverman, B. W. (2005), Functional Data
            Analysis, Springer Science, Chapter 8.

        """
        # Checkers
        if not isinstance(data, DenseFunctionalData):
            raise TypeError('UFPCA only support DenseFunctionalData object!')

        # Center the data
        data_mean = data.mean()
        data_new = DenseFunctionalData(
            data.argvals, data.values - data_mean.values
        )

        # Estimate eigencomponents
        self.mean = data_mean
        if self.method == 'covariance':
            if data_new.n_dim == 1:
                self._fit_covariance(data_new)
            else:
                raise ValueError((
                    "Estimation of the eigencomponents is not implemented "
                    f"for {data_new.n_dim}-dimensional data."
                ))
        elif self.method == 'inner-product':
            self._fit_inner_product(data_new)
        else:
            raise NotImplementedError(
                f"{self.method} method not implemented."
            )

    def _fit_covariance(
        self,
        data: DenseFunctionalData,
        covariance: Optional[DenseFunctionalData] = None
    ) -> None:
        """Univariate Functional PCA.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data.
        covariance: DenseFunctionalData, default=None
            An estimation of the covariance of the training data.

        References
        ----------
        .. [RS] Ramsey, J. O. and Silverman, B. W. (2005), Functional Data
            Analysis, Springer Science, Chapter 8.

        """
        if self.normalize:
            data, weights = data.normalize(use_argvals_stand=True)
            self.weights = weights
        smoothing_method = None

        if covariance is None:
            covariance = data.covariance(
                mean=None,
                smooth=smoothing_method
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

        # Compute an estimation of the covariance
        covariance = _compute_covariance(eigenvalues, eigenfunctions)
        self.covariance = DenseFunctionalData(
            {'input_dim_0': argvals, 'input_dim_1': argvals},
            covariance[np.newaxis]
        )

    def _fit_inner_product(
        self,
        data: DenseFunctionalData
    ) -> None:
        """Univariate Functional PCA using inner-product matrix decomposition.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data used to estimate the eigencomponents.

        """
        # Compute inner product matrix and its eigendecomposition
        eigenvalues, eigenvectors = _compute_inner_product(
            data,
            self.n_components
        )
        eigenfunctions = (
            np.matmul(data.values.T, eigenvectors) / np.sqrt(eigenvalues)
        )

        self._eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues / data.n_obs
        self.eigenfunctions = DenseFunctionalData(
            data.argvals, eigenfunctions.T
        )

        # Compute an estimation of the covariance
        if data.n_dim == 1:
            argvals = data.argvals['input_dim_0']
            covariance = _compute_covariance(
                eigenvalues / data.n_obs, eigenfunctions.T
                )
            self.covariance = DenseFunctionalData(
                {'input_dim_0': argvals, 'input_dim_1': argvals},
                covariance[np.newaxis]
            )
        else:
            warnings.warn((
                "The estimation of the covariance is not performed for "
                f"{data.n_dim}-dimensional data."
            ), UserWarning)

    def transform(
        self,
        data: DenseFunctionalData,
        method: str = 'NumInt',
        **kwargs
    ) -> npt.NDArray[np.float64]:
        r"""Apply dimensionality reduction to the data.

        The functional principal components scores are defined as the
        projection of the observation :math:`X_i` on the eigenfunction
        :math:`\phi_k`. These scores are given by:

        .. math::
            c_{ik} = \int_{\mathcal{T}} \{X_i(t) - \mu(t)\}\phi_k(t)dt.

        This integrale can be estimated using two ways. First, if data are
        sampled on a common fine grid, the estimation is done using
        numerical integration. Second, the PACE (Principal Components through
        Conditional Expectation) algorithm [YMW]_ is used for sparse functional
        data. If the eigenfunctions have been estimated using the inner-product
        matrix, the scores can also be estimated using the formula

        .. math::
            c_{ik} = \sqrt{l_k}v_{ik},

        where :math:`l_k` and :math:`v_{k}` are the eigenvalues and
        eigenvectors of the inner-product matrix.

        Parameters
        ----------
        data: DenseFunctionalData
            Data
        method: str, {'NumInt', 'PACE', 'InnPro'}, default='NumInt'
            Method used to estimate the scores. If ``method == 'NumInt'``,
            numerical integration method is performed. If
            ``method == 'PACE'``, the PACE algorithm [YMW]_ is used. If
            ``method == 'InnPro'``, the estimation is performed using the
            inner product matrix of the data (can only be used if the
            eigencomponents have been estimated using the inner-product
            matrix.)

        Keyword Args
        ------------
        tol: np.float64, default=1e-4
            Tolerance parameter to prevent overflow to inverse a matrix, only
            used if ``method == 'PACE'``.
        integration_method: str, {'trapz', 'simpson'}, default='trapz'
            Method used to perform numerical integration, only used if
            ``method == 'NumInt'``.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_components)
            An array representing the projection of the data onto the basis of
            functions defined by the eigenfunctions.

        References
        ----------
        .. [YMW] Yao, Müller and Wang (2005), Functional Data Analysis for
            Sparse Longitudinal Data, Journal of the American Statistical
            Association, Vol. 100, No. 470.

        """
        # Get the keyword arguments
        parameters = {
            'tol': kwargs.get('tol', 1e-4),
            'integration_method': kwargs.get('integration_method', 'trapz')
        }

        # Checkers
        if method == 'InnPro' and not hasattr(self, '_eigenvectors'):
            raise ValueError((
                f"The method {method} can not be used as the eigencomponents "
                "have not been estimated using the inner-product matrix."
            ))

        # Center the data using the estimated mean in the fitting step.
        data_new = DenseFunctionalData(
            data.argvals, data.values - self.mean.values
        )

        # TODO: Add checkers
        if self.normalize:
            values = data.values / self.weights
            data = DenseFunctionalData(data.argvals, values)

        if method == 'PACE':
            return self._pace(data_new, parameters['tol'])
        elif method == 'NumInt':
            return self._numerical_integration(
                data_new, parameters['integration_method']
            )
        elif method == 'InnPro':
            temp = np.sqrt(data.n_obs * self.eigenvalues)
            return temp * self._eigenvectors
        else:
            raise ValueError(
                f"Method {method} not implemented."
            )

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
        .. [YWM] Yao, Müller and Wang (2005), Functional Data Analysis for
            Sparse Longitudinal Data, Journal of the American Statistical
            Association, Vol. 100, No. 470.

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
        method: str, {'trapz', 'simpson'}, default='trapz'
            Method used to perform numerical integration.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_components)
            An array representing the projection of the data onto the basis of
            functions defined by the eigenfunctions.

        """
        projection = _integrate(
            x=data.argvals['input_dim_0'],
            y=[traj * self.eigenfunctions.values for traj in data.values],
            method=method
        )
        if data.n_dim == 1:
            return projection
        elif data.n_dim == 2:
            return _integrate(
                x=data.argvals['input_dim_1'],
                y=projection,
                method=method
            )
        else:
            raise ValueError("The dimension of the data have to be 1 or 2.")

    def inverse_transform(
        self,
        scores: npt.NDArray[np.float64]
    ) -> DenseFunctionalData:
        r"""Transform the data back to its original space.

        Given a set of scores :math:`c_{ik}`, we reconstruct the observations
        using a truncation of the Karhunen-Loève expansion,

        .. math::
            X_{i}(t) = \mu(t) + \sum_{k = 1}^K c_{ik}\phi_k(t).

        Data can be multidimensional.

        Parameters
        ----------
        scores: npt.NDArray[np.float64], shape=(n_obs, n_components)
            New data, where `n_obs` is the number of observations and
            `n_components` is the number of components.

        Returns
        -------
        DenseFunctionalData
            A DenseFunctionalData object representing the transformation of the
            scores into the original curve space.

        """
        argvals = self.eigenfunctions.argvals
        if self.eigenfunctions.n_dim == 1:
            values = np.dot(scores, self.eigenfunctions.values)
        elif self.eigenfunctions.n_dim == 2:
            values = np.einsum(
                'ij,jkl->ikl',
                scores,
                self.eigenfunctions.values
            )
        else:
            raise ValueError("The dimension of the data have to be 1 or 2.")
        return DenseFunctionalData(argvals, values + self.mean.values)


#############################################################################
# Class MFPCA

class MFPCA():
    r"""MFPCA -- Multivariate Functional Principal Components Analysis.

    Linear dimensionality reduction of a multivariate functional dataset. The
    projection of the data in a lower dimensional space is performed using
    a diagomalization of the covariance operator of each univariate component
    or of the inner-product matrix of the data. It is assumed that the data
    have :math:`P` components.

    Parameters
    ----------
    method: str, {'covariance', 'inner-product'}, default='covariance'
        Method used to estimate the eigencomponents. If
        ``method == 'covariance'``, the estimation is based on an
        eigendecomposition of the covariance operator of each univariate
        components. If ``method == 'inner-product'``, the estimation is
        based on an eigendecomposition of the inner-product matrix.
    n_components: List[int, float, None], default=None
        Number of components to keep. If ``method=='covariance'``,
        `n_components` should be a list of length :math:`P`. Each entry
        represents the variance explained by each univariate component. Note
        that for 2-dimensional data, `n_components` has to be an integer, as we
        use the FCP-TPA algorithm. If ``method=='inner-product'``,
        `n_components` should not be a list and represents the variance
        explained by the multivariate components.
        If `n_components` is `None`, all components are kept,
        ``n_components == min(n_samples, n_features)``. If `n_components` is an
        integer, `n_components` are kept. If `0 < n_components < 1`, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by `n_components`.
    normalize: bool, default=False
        Perform a normalization of the data.

    Attributes
    ----------
    mean: MultivariateFunctionalData
        An estimation of the mean of the training data.
    covariance: MultivariateFunctionalData
        An estimation of the covariance of the training data based on their
        eigendecomposition using the Mercer's theorem.
    eigenvalues: npt.NDArray[np.float64], shape=(n_components,)
        The singular values corresponding to each of selected components.
    eigenfunctions: MultivariateFunctionalData
        Principal axes in feature space, representing the directions of
        maximum variances in the data as a MultivariateFunctionalData.

    """

    def __init__(
        self,
        method: str = 'covariance',
        n_components: List[Union[int, float, None]] = None,
        normalize: bool = False
    ) -> None:
        """Initialize MFPCA object."""
        self.n_components = n_components
        self.method = method
        self.normalize = normalize

    def fit(
        self,
        data: MultivariateFunctionalData,
        scores_method: str = 'NumInt'
    ) -> None:
        """Estimate the eigencomponents of the data.

        Before estimating the eigencomponents, the data is centered. Using the
        covariance operator, the estimation is based on [HG]_.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Training data used to estimate the eigencomponents.
        scores_method: str, {'NumInt', 'PACE'}, default='NumInt'
            Method for the estimation of the univariate scores for the
            diagonalization of the covariance operator.

        References
        ----------
        .. [HG] Happ C. & Greven S. (2018) Multivariate Functional Principal
            Component Analysis for Data Observed on Different (Dimensional)
            Domains, Journal of the American Statistical Association, 113:522,
            649-659, DOI: 10.1080/01621459.2016.1273115

        """
        # Checkers
        if not isinstance(data, MultivariateFunctionalData):
            raise TypeError(
                'MFPCA only support MultivariateFunctionalData object!'
            )

        # Center the data
        data_mean = data.mean()
        data_new = MultivariateFunctionalData([
            DenseFunctionalData(
                data_uni.argvals,
                data_uni.values - mean.values
            ) for data_uni, mean in zip(data, data_mean)
        ])

        # Estimate eigencomponents
        self.mean = data_mean
        if self.method == 'covariance':
            self._fit_covariance(data_new, scores_method)
        elif self.method == 'inner-product':
            self._fit_inner_product(data_new)
        else:
            raise NotImplementedError(
                f"{self.method} method not implemented."
            )

    def _fit_covariance(
        self,
        data: MultivariateFunctionalData,
        scores_method: str = 'NumInt'
    ) -> None:
        """Multivariate Functional PCA.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Training data.
        scores_method: str, {'NumInt', 'PACE'}, default='NumInt'
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
        eigenfunctions = []
        for idx, function in enumerate(ufpca_list):
            start = nb_eigenfunction_uni_cum[idx]
            end = nb_eigenfunction_uni_cum[idx + 1]

            argvals = function.eigenfunctions.argvals
            values = np.dot(function.eigenfunctions.values.T,
                            eigenvectors[start:end, :]).T
            eigenfunctions.append(DenseFunctionalData(argvals, values))

        self.ufpca_list = ufpca_list
        self.scores_univariate = scores_univariate
        self.covariance = covariance
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.eigenfunctions = MultivariateFunctionalData(eigenfunctions)

    def _fit_inner_product(
        self,
        data: MultivariateFunctionalData
    ) -> None:
        """Multivariate FPCA using inner-product matrix decomposition.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Training data used to estimate the eigencomponents.

        """
        # Compute inner product matrix and its eigendecomposition
        eigenvalues, eigenvectors = _compute_inner_product(
            data,
            self.n_components
        )
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

        self._eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues / data.n_obs
        self.eigenfunctions = MultivariateFunctionalData(eigenfunctions)

        # Compute an estimation of the covariance

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
        method: str = 'NumInt',
        **kwargs
    ) -> npt.NDArray[np.float64]:
        r"""Apply dimensionality reduction to the data.

        The functional principal components scores are defined as the
        projection of the observation :math:`X_i` on the eigenfunction
        :math:`\phi_k`. These scores are given by:

        .. math::
            c_{ik} = \sum_{p = 1}^P \int_{\mathcal{T}_p} \{X_i^{(p)}(t) -
            \mu^{(p)}(t)\}\phi_k^{(p)}(t)dt.

        This integrale can be estimated using numerical integration. If the
        eigenfunctions have been estimated using the inner-product matrix, the
        scores can also be estimated using the formula

        .. math::
            c_{ik} = \sqrt{l_k}v_{ik},

        where :math:`l_k` and :math:`v_{k}` are the eigenvalues and
        eigenvectors of the inner-product matrix.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Data
        method: str, {'NumInt', 'InnPro'}, default='NumInt'
            Method used to estimate the scores. If ``method == 'NumInt'``,
            numerical integration method is performed. If
            ``method == 'InnPro'``, the estimation is performed using the
            inner product matrix of the data (can only be used if the
            eigencomponents have been estimated using the inner-product
            matrix.)

        Keyword Args
        ------------
        integration_method: str, {'trapz', 'simpson'}, default='trapz'
            Method used to perform numerical integration, only used if
            ``method == 'NumInt'``.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_components)
            An array representing the projection of the data onto the basis of
            functions defined by the eigenfunctions.

        Notes
        -----
        Concerning the estimation of the scores using numerical integration, we
        directly estimate the scores using the projection of the data onto the
        multivariate eigenfunctions and not use the univariate components and
        the decomposition of the covariance of the univariate scores as Happ
        and Greven [HG]_ could do.

        References
        ----------
        .. [HG] Happ C. & Greven S. (2018) Multivariate Functional Principal
            Component Analysis for Data Observed on Different (Dimensional)
            Domains, Journal of the American Statistical Association, 113:522,
            649-659, DOI: 10.1080/01621459.2016.1273115

        """
        # Get the keyword arguments
        parameters = {
            'integration_method': kwargs.get('integration_method', 'trapz')
        }

        # Checkers
        if method == 'InnPro' and not hasattr(self, '_eigenvectors'):
            raise ValueError((
                f"The method {method} can not be used as the eigencomponents "
                "have not been estimated using the inner-product matrix."
            ))

        # Center the data using the estimated mean in the fitting step.
        data_new = MultivariateFunctionalData([
            DenseFunctionalData(
                data_uni.argvals,
                data_uni.values - mean.values
            ) for data_uni, mean in zip(data, self.mean)
        ])

        # TODO: Add checkers
        # if self.normalize:
        #     values = data.values / self.weights
        #     data = MultivariateFunctionalData(data.argvals, values)

        if method == 'PACE':
            raise ValueError("PACE method not implemented.")
        elif method == 'NumInt':
            return self._numerical_integration(
                data_new, parameters['integration_method']
            )
        elif method == 'InnPro':
            temp = np.sqrt(data.n_obs * self.eigenvalues)
            return temp * self._eigenvectors
        else:
            raise ValueError(
                f"Method {method} not implemented."
            )

    def _numerical_integration(
        self,
        data: MultivariateFunctionalData,
        method: str = "trapz"
    ) -> npt.NDArray[np.float64]:
        """Estimate scores using numerical integration.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Data
        method: str, {'trapz', 'simpson'}, default='trapz'
            Method used to perform numerical integration.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_components)
            An array representing the projection of the data onto the basis of
            functions defined by the eigenfunctions.

        """
        scores_uni = [None] * len(self.eigenfunctions)
        for idx, (dat_uni, eigen) in enumerate(zip(data, self.eigenfunctions)):
            projection = _integrate(
                x=dat_uni.argvals['input_dim_0'],
                y=[traj * eigen.values for traj in dat_uni.values],
                method=method
            )
            if eigen.n_dim == 1:
                scores_uni[idx] = projection
            elif eigen.n_dim == 2:
                scores_uni[idx] = _integrate(
                    x=dat_uni.argvals['input_dim_1'],
                    y=projection,
                    method=method
                )
            else:
                raise ValueError(
                    "The dimension of the data have to be 1 or 2."
                )

        return np.array(scores_uni).sum(axis=0)

    def inverse_transform(
        self,
        scores: npt.NDArray[np.float64]
    ) -> MultivariateFunctionalData:
        r"""Transform the data back to its original space.

        Given a set of scores :math:`c_{ik}`, we reconstruct the observations
        using a truncation of the Karhunen-Loève expansion,

        .. math::
            X_{i}(t) = \mu(t) + \sum_{k = 1}^K c_{ik}\phi_k(t).

        Data can be multidimensional. Recall that, here, :math:`X_{i}`,
        :math:`\mu` and :math:`\phi_k` are :math:`P`-dimensional functions.

        Parameters
        ----------
        scores: npt.NDArray[np.float64], shape=(n_obs, n_components)
            New data, where `n_obs` is the number of observations and
            `n_components` is the number of components.

        Returns
        -------
        MultivariateFunctionalData
            A ``MultivariateFunctionalData`` object representing the
            transformation of the scores into the original curve space.

        """
        res = [None] * self.eigenfunctions.n_functional
        for idx, (mean, eigenfunction) in enumerate(
            zip(self.mean, self.eigenfunctions)
        ):
            if eigenfunction.n_dim == 1:
                values = np.dot(scores, eigenfunction.values)
            elif eigenfunction.n_dim == 2:
                values = np.einsum('ij,jkl->ikl', scores, eigenfunction.values)
            else:
                raise ValueError(
                    "The dimension of the data have to be 1 or 2."
                )
            res[idx] = DenseFunctionalData(
                eigenfunction.argvals, values + mean.values
            )

        # res = []
        # for idx, ufpca in enumerate(self.ufpca_list):
        #     if isinstance(ufpca, UFPCA):
        #         mean = ufpca.mean
        #         reconst = np.dot(scores, self.basis[idx].values) + mean.values
        #         res.append(DenseFunctionalData(mean.argvals, reconst))
        #     elif isinstance(ufpca, FCPTPA):
        #         reconst = np.einsum('ij, jkl', scores, self.basis[idx].values)
        #         res.append(DenseFunctionalData(ufpca.eigenfunctions.argvals,
        #                                        reconst))
        #     else:
        #         raise TypeError("Something went wrong with univariate "
        #                         "decomposition.")
        return MultivariateFunctionalData(res)


def _compute_inner_product(
    data: FunctionalData,
    n_components: Optional[Union[np.float64, np.int64]] = None
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the inner-product matrix and its eigendecomposition.

    Parameters
    ----------
    data: FunctionalData
        Observed data, an instance of DenseFunctionalData or
        MultivariateFunctionalData.
    n_components: int, float, None, default=None
        Number of components to keep. If `n_components` is `None`, all
        components are kept, ``n_components == min(n_samples, n_features)``.
        If `n_components` is an integer, `n_components` are kept. If
        `0 < n_components < 1`, select the number of components such that the
        amount of variance that needs to be explained is greater than the
        percentage specified by `n_components`.

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        A tuple containing the eigenvalues and the eigenvectors of the inner
        product matrix.

    """
    # Compute inner-product matrix
    inner_mat = data.inner_product()

    # Diagonalization of the inner-product matrix
    eigenvalues, eigenvectors = np.linalg.eigh(inner_mat)

    # Estimation of the number of components
    eigenvalues = np.real(eigenvalues[::-1])
    eigenvalues[eigenvalues < 0] = 0
    npc = _select_number_eigencomponents(eigenvalues, n_components)

    # Estimation of the eigenvalues
    eigenvalues = eigenvalues[:npc]

    # Estimation of the eigenfunctions
    eigenvectors = np.real(np.fliplr(eigenvectors)[:, :npc])

    return eigenvalues, eigenvectors
