#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Multivariate Functional Principal Components Analysis
-----------------------------------------------------

"""
import numpy as np
import numpy.typing as npt

from typing import Dict, Optional, List, Union

from ...representation.argvals import DenseArgvals
from ...representation.values import DenseValues
from ...representation.functional_data import (
    DenseFunctionalData,
    MultivariateFunctionalData,
)
from .ufpca import (
    UFPCA,
    _transform_numerical_integration_dense,
    _transform_numerical_integration_irregular,
    _transform_innpro,
)
from .fcp_tpa import FCPTPA
from ...misc.utils import _compute_eigen


#############################################################################
# Utilities to fit


def _fit_covariance_multivariate(
    data: MultivariateFunctionalData,
    points: DenseArgvals,
    n_components: List[Union[int, float]],
    method_smoothing: str = "LP",
    scores_method: str = "NumInt",
    **kwargs,
) -> Dict[str, object]:
    """Multivariate Functional PCA using the covariance operator.

    Parameters
    ----------
    data: FunctionalData
        Training data.
    points: DenseArgvals
        The sampling points at which the covariance and the eigenfunctions
        will be estimated.
    n_components: List[Union[int, float]]
        Number of components to be estimated.
    method_smoothing: str, default='LP'
        Should the mean and covariance be smoothed?
    scores_method: str, {'NumInt', 'PACE'}, default='NumInt'
        Method for the estimation of the univariate scores.
    **kwargs:
        kernel_name: str, default='epanechnikov'
            Name of the kernel used for local polynomial smoothing.
        degree: int, default=1
            Degree used for local polynomial smoothing.
        bandwidth: float
            Bandwidth used for local polynomial smoothing. The default
            bandwitdth is set to be the number of sampling points to the
            power :math:`-1/5`.

    """
    # Step 1: Perform univariate fPCA on each functions.
    ufpca_list, scores = [], []
    for fdata_uni, n_comp in zip(data.data, n_components):
        if fdata_uni.n_dimension == 1:
            ufpca = UFPCA(n_components=n_comp, normalize=True)
            ufpca.fit(data=fdata_uni, points=None, method_smoothing=method_smoothing)
            scores_uni = ufpca.transform(
                method=scores_method, method_smoothing=method_smoothing
            )
        elif fdata_uni.n_dimension == 2:
            n_points = fdata_uni.n_points
            mat_v = np.diff(np.identity(n_points[0]))
            mat_w = np.diff(np.identity(n_points[1]))
            ufpca = FCPTPA(n_components=n_comp, normalize=True)
            ufpca.fit(
                fdata_uni,
                penalty_matrices={
                    "v": np.dot(mat_v, mat_v.T),
                    "w": np.dot(mat_w, mat_w.T),
                },
                alpha_range={"v": (1e-4, 1e4), "w": (1e-4, 1e4)},
                tolerance=1e-4,
                max_iteration=15,
                adapt_tolerance=True,
            )
            scores_uni = ufpca.transform(fdata_uni)
        ufpca_list.append(ufpca)
        scores.append(scores_uni)
    scores_univariate = np.concatenate(scores, axis=1)

    # Step 2: Estimation of the covariance of the scores.
    temp = np.dot(scores_univariate.T, scores_univariate)
    covariance = temp / (len(scores_univariate) - 1)

    # Step 3: Eigenanalysis of the covariance of the scores.
    # We choose to keep all the components here.
    eigenvalues, eigenvectors = _compute_eigen(covariance)

    # Step 4: Estimation of the multivariate eigenfunctions.
    # Retrieve the number of eigenfunctions for each univariate function.
    nb_eigenfunction_uni = [0]
    for ufpca in ufpca_list:
        nb_eigenfunction_uni.append(len(ufpca.eigenvalues))
    nb_eigenfunction_uni_cum = np.cumsum(nb_eigenfunction_uni)

    # Compute the multivariate eigenbasis.
    eigenfunctions = []
    for idx, ufpca in enumerate(ufpca_list):
        start = nb_eigenfunction_uni_cum[idx]
        end = nb_eigenfunction_uni_cum[idx + 1]
        values = np.dot(ufpca.eigenfunctions.values.T, eigenvectors[start:end, :])
        eigenfunctions.append(
            DenseFunctionalData(ufpca.eigenfunctions.argvals, values.T)
        )

    # Save the results
    results = dict()
    results["eigenvalues"] = eigenvalues
    results["eigenfunctions"] = MultivariateFunctionalData(eigenfunctions)
    results["_ufpca_list"] = ufpca_list
    results["_scores_univariate"] = scores_univariate
    results["_scores_eigenvectors"] = eigenvectors
    return results


def _fit_inner_product_multivariate(
    data: MultivariateFunctionalData,
    points: DenseArgvals,
    n_components: Union[int, float] = 1,
    method_smoothing: str = "LP",
    noise_variance: Optional[npt.NDArray[np.float64]] = None,
    **kwargs,
) -> Dict[str, object]:
    """Multivariate Functional PCA using inner-product matrix decomposition.

    Parameters
    ----------
    data: MultivariateFunctionalData
        Training data used to estimate the eigencomponents.
    points: Optional[List[DenseArgvals]]
        The sampling points at which the covariance and the eigenfunctions
        will be estimated.
    n_components: Union[int, float], default=1
        Number of components to be estimated.
    method_smoothing: str = 'LP',
            Should the mean and covariance be smoothed?
    noise_variance: Optional[npt.NDArray[np.float64]], default=None
            An estimation of the variance of the noise. If `None`, an
            estimation is computed using the methodology in [1]_.
    **kwargs:
        kernel_name: str, default='epanechnikov'
            Name of the kernel used for local polynomial smoothing.
        degree: int, default=1
            Degree used for local polynomial smoothing.
        bandwidth: float
            Bandwidth used for local polynomial smoothing. The default
            bandwitdth is set to be the number of sampling points to the
            power :math:`-1/5`.

    Returns
    -------
    Dict[str, object]
        A dictionary with entries:
            - "eigenvalues"
            - "eigenfunctions"
            - "eigenvectors"

    """
    # Compute inner product matrix and its eigendecomposition
    in_prod = data.inner_product(
        method_integration="trapz",
        method_smoothing=method_smoothing,
        noise_variance=noise_variance,
        **kwargs,
    )
    eigenvalues, eigenvectors = _compute_eigen(in_prod, n_components)

    # Compute the eigenfunctions
    temp = [
        np.matmul(data_uni._data_inpro.values.T, eigenvectors) / np.sqrt(eigenvalues)
        for data_uni in data.data
    ]
    eigenfunctions = [
        DenseFunctionalData(data_uni._data_inpro.argvals, eigenfunction.T)
        for data_uni, eigenfunction in zip(data.data, temp)
    ]
    eigenfunctions = MultivariateFunctionalData(eigenfunctions)

    # Save the results
    results = dict()
    results["eigenvectors"] = eigenvectors
    results["eigenvalues"] = eigenvalues / data.n_obs
    results["eigenfunctions"] = eigenfunctions
    #results["eigenfunctions"] = eigenfunctions.smooth(points=points, method="PS")
    return results


#############################################################################
# Utilities to transform


def _transform_numerical_integration_multivariate(
    data: MultivariateFunctionalData,
    eigenfunctions: MultivariateFunctionalData,
    method: str = "trapz",
) -> npt.NDArray[np.float64]:
    """Estimate scores using numerical integration.

    Parameters
    ----------
    data: DenseFunctionalData
        Data.
    eigenfunctions: DenseFunctionalData
        Estimate of the eigenfunctions.
    method: str, {'trapz', 'simpson'}, default='trapz'
        Method used to perform numerical integration.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs, n_components)
        An array representing the projection of the data onto the basis of
        functions defined by the eigenfunctions.

    """
    scores = [None] * eigenfunctions.n_functional
    for idx, (eigen, data) in enumerate(zip(eigenfunctions.data, data.data)):
        if isinstance(data, DenseFunctionalData):
            scores[idx] = _transform_numerical_integration_dense(
                data=data, eigenfunctions=eigen, method=method
            )
        else:
            scores[idx] = _transform_numerical_integration_irregular(
                data, eigen, method=method
            )
    return np.array(scores).sum(axis=0)


def _transform_pace_multivariate(
    eigenvectors: npt.NDArray[np.float64],
    scores_univariate: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Estimate scores using PACE.

    Parameters
    ----------
    eigenvectors: npt.NDArray[np.float64]
        Estimate of the eigenvectors of the scores.
    scores_univariate: npt.NDArray[np.float64]
        Estimate of the scores of the univariate components.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs, n_components)
        An array representing the projection of the data onto the basis of
        functions defined by the eigenfunctions.


    """
    return np.dot(scores_univariate, eigenvectors)


#############################################################################
# Class MFPCA


class MFPCA:
    r"""MFPCA -- Multivariate Functional Principal Components Analysis.

    Linear dimensionality reduction of a multivariate functional dataset. The
    projection of the data in a lower dimensional space is performed using
    a diagonalization of the covariance operator of each univariate component
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
    n_components: Optional[List[Union[int, float]]], default=None
        Number of components to keep. If ``method=='covariance'``,
        `n_components` should be a list of length :math:`P`. Each entry
        represents the variance explained by each univariate component. Note
        that for 2-dimensional data, `n_components` has to be an integer, as we
        use the FCP-TPA algorithm. If ``method=='inner-product'``,
        `n_components` should not be a list and represents the variance
        explained by the multivariate components. If `n_components` is `None`,
        all components are kept,
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

    References
    ----------
    .. [1] Happ and Greven (2018), Multivariate Functional Principal
        Component Analysis for Data Observed on Different (Dimensional)
        Domains. Journal of the American Statistical Association, 113,
        pp. 649--659.

    """

    def __init__(
        self,
        n_components: List[Union[int, float]],
        method: str = "covariance",
        normalize: bool = False,
    ) -> None:
        """Initialize MFPCA object."""
        self.n_components = n_components
        self.method = method
        self.normalize = normalize
        self.weights = None

    @property
    def method(self) -> str:
        """Getter for `method`."""
        return self._method

    @method.setter
    def method(self, new_method: str) -> None:
        self._method = new_method

    @property
    def n_components(self) -> List[Union[int, float]]:
        """Getter for `n_components`."""
        return self._n_components

    @n_components.setter
    def n_components(self, new_n_components: List[Union[int, float]]) -> None:
        self._n_components = new_n_components

    @property
    def normalize(self) -> bool:
        """Getter for `normalize`."""
        return self._normalize

    @normalize.setter
    def normalize(self, new_normalize: bool) -> None:
        self._normalize = new_normalize

    @property
    def mean(self) -> MultivariateFunctionalData:
        """Getter for `mean`."""
        return self._mean

    @property
    def covariance(self) -> MultivariateFunctionalData:
        """Getter for `covariance`."""
        return self._covariance

    @property
    def eigenvalues(self) -> npt.NDArray[np.float64]:
        """Getter for `eigenvalues`."""
        return self._eigenvalues

    @property
    def eigenfunctions(self) -> MultivariateFunctionalData:
        """Getter for `eigenfunctions`."""
        return self._eigenfunctions

    def fit(
        self,
        data: MultivariateFunctionalData,
        points: Optional[List[DenseArgvals]] = None,
        method_smoothing: str = "LP",
        scores_method: str = "NumInt",
        **kwargs,
    ) -> None:
        """Estimate the eigencomponents of the data.

        Before estimating the eigencomponents, the data is centered. Using the
        covariance operator, the estimation is based on [1]_.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Training data used to estimate the eigencomponents.
        points: Optional[List[DenseArgvals]]
            The sampling points at which the covariance and the eigenfunctions
            will be estimated.
        method_smoothing: str = 'LP',
            Should the mean and covariance be smoothed?
        scores_method: str, {'NumInt', 'PACE', 'InnPro'}, default='NumInt'
            Method for the estimation of the univariate scores for the
            diagonalization of the covariance operator.
        **kwargs:
            kernel_name: str, default='epanechnikov'
                Name of the kernel used for local polynomial smoothing.
            degree: int, default=1
                Degree used for local polynomial smoothing.
            bandwidth: float
                Bandwidth used for local polynomial smoothing. The default
                bandwitdth is set to be the number of sampling points to the
                power :math:`-1/5`.

        References
        ----------
        .. [1] Happ C. & Greven S. (2018), Multivariate Functional Principal
            Component Analysis for Data Observed on Different (Dimensional)
            Domains. Journal of the American Statistical Association, 113,
            pp. 649--659.

        """
        if points is None:
            points = [
                dd.argvals
                if isinstance(dd, DenseFunctionalData)
                else dd.argvals.to_dense()
                for dd in data.data
            ]
        if self.weights is None:
            self.weights = np.repeat(1, data.n_functional)

        # Compute the mean and center the data.
        self._mean = data.mean(
            points=points, method_smoothing=method_smoothing, **kwargs
        )
        data = data.center(mean=self._mean, method_smoothing=method_smoothing, **kwargs)

        # Normalize the data
        if self.normalize:
            data, self.weights = data.rescale(use_argvals_stand=True)

        # Estimate the variance of the noise
        self._noise_variance = data.noise_variance(order=2)

        # Estimate eigencomponents
        if self.method == "covariance":
            results = _fit_covariance_multivariate(
                data=data,
                points=points,
                n_components=self.n_components,
                method_smoothing=method_smoothing,
                scores_method=scores_method,
                **kwargs,
            )
        elif self.method == "inner-product":
            results = _fit_inner_product_multivariate(
                data=data,
                points=points,
                n_components=self.n_components,
                method_smoothing=method_smoothing,
                noise_variance=self._noise_variance,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"{self.method} method not implemented.")

        # Save the results
        self._eigenvalues = results.get("eigenvalues", None)
        self._eigenfunctions = results.get("eigenfunctions", None)
        self._eigenvectors = results.get("eigenvectors", None)
        self._scores_univariate = results.get("_scores_univariate", None)
        self._scores_eigenvectors = results.get("_scores_eigenvectors", None)
        self._training_data = data

        # TODO: Add covariance computation
        self._covariance = None

    def transform(
        self,
        data: Optional[MultivariateFunctionalData] = None,
        method: str = "NumInt",
        method_smoothing: str = "LP",
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        r"""Apply dimensionality reduction to the data.

        The functional principal components scores are defined as the
        projection of the observation :math:`X_i` on the eigenfunction
        :math:`\phi_k`. These scores are given by:

        .. math::
            c_{ik} = \sum_{p = 1}^P \int_{\mathcal{T}_p} \{X_i^{(p)}(t) -
            \mu^{(p)}(t)\}\phi_k^{(p)}(t)dt.

        This integral can be estimated using numerical integration. If the
        eigenfunctions have been estimated using the inner-product matrix, the
        scores can also be estimated using the formula

        .. math::
            c_{ik} = \sqrt{l_k}v_{ik},

        where :math:`l_k` and :math:`v_{k}` are the eigenvalues and
        eigenvectors of the inner-product matrix.

        TODO: Add PACE.
        TODO: Test for 2D functional data

        Parameters
        ----------
        data: Optional[MultivariateFunctionalData], default=None
            Data
        method: str, {'NumInt', 'PACE', 'InnPro'}, default='NumInt'
            Method used to estimate the scores. If ``method == 'NumInt'``,
            numerical integration method is performed. If
            ``method == 'InnPro'``, the estimation is performed using the
            inner product matrix of the data (can only be used if the
            eigencomponents have been estimated using the inner-product
            matrix.)
        method_smoothing: str = 'LP',
            Should the mean and covariance be smoothed?
        **kwargs:
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
        and Greven [1]_ could do.

        References
        ----------
        .. [1] Happ and Greven (2018), Multivariate Functional Principal
            Component Analysis for Data Observed on Different (Dimensional)
            Domains. Journal of the American Statistical Association, 113,
            pp. 649--659.

        """
        # Checkers
        if method == "InnPro" and data is not None:
            raise ValueError(
                f"The method {method} can not be used as the eigencomponents "
                "have not been estimated using the provided data."
            )
        if method == "InnPro" and self._eigenvectors is None:
            raise ValueError(
                f"The method {method} can not be used as the eigencomponents "
                "have not been estimated using the inner-product matrix."
            )
        if data is None:
            data_new = self._training_data
        else:
            # Center the data using the estimated mean in the fitting step.
            data_new = data.center(
                mean=self._mean, method_smoothing=method_smoothing, **kwargs
            )
            if self.normalize:
                data_new, _ = data.rescale(weights=self.weights)

        if method == "NumInt":
            return _transform_numerical_integration_multivariate(
                data_new,
                self._eigenfunctions,
                kwargs.get("integration_method", "trapz"),
            )
        elif method == "PACE":
            return _transform_pace_multivariate(
                self._scores_eigenvectors, self._scores_univariate
            )
        elif method == "InnPro":
            return _transform_innpro(data_new, self._eigenvectors, self.eigenvalues)
        else:
            raise ValueError(f"Method {method} not implemented.")

    def inverse_transform(
        self, scores: npt.NDArray[np.float64]
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
            A MultivariateFunctionalData object representing the transformation
            of the scores into the original curve space.

        """
        results = [None] * self.eigenfunctions.n_functional
        for idx, (mean, eigenfunction, weight) in enumerate(
            zip(self.mean.data, self.eigenfunctions.data, self.weights)
        ):
            values = np.einsum("ij,j... -> i...", scores, eigenfunction.values)
            results[idx] = DenseFunctionalData(
                DenseArgvals(eigenfunction.argvals),
                DenseValues(weight * values + mean.values),
            )
        return MultivariateFunctionalData(results)
