#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Functional Principal Components Analysis
----------------------------------------

"""
import numpy as np
import numpy.typing as npt
import warnings

from typing import Dict, Optional, List, Union

from ...representation.argvals import DenseArgvals
from ...representation.values import DenseValues
from ...representation.functional_data import (
    FunctionalData, DenseFunctionalData,
    IrregularFunctionalData, MultivariateFunctionalData
)
from ...misc.utils import (
    _compute_covariance,
    _integrate,
    _integration_weights,
    _compute_eigen
)
from .fcp_tpa import FCPTPA


#############################################################################
# Utilities to fit

def _fit_covariance(
    data: FunctionalData,
    points: DenseArgvals,
    n_components: Union[int, float] = 1,
    smooth: bool = True,
    **kwargs
) -> Dict[str, object]:
    """Univariate Functional PCA using the covariance operator.

    Parameters
    ----------
    data: FunctionalData
        Training data.
    points: DenseArgvals
        The sampling points at which the covariance and the eigenfunctions
        will be estimated.
    n_components: Union[int, float], default=1
        Number of components to be estimated.
    smooth: bool, default=True
        Should the mean and covariance be smoothed?
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
            - "noise_variance_cov"


    References
    ----------
    .. [1] Ramsey, J. O. and Silverman, B. W. (2005), Functional Data
        Analysis, Springer Science, Chapter 8.

    """
    # Compute the covariance
    covariance = data.covariance(points=points, smooth=smooth, **kwargs)

    # Choose the W_j's and the S_j's (Ramsey and Silverman, 2005)
    argvals = points['input_dim_0']
    weight = _integration_weights(argvals, method='trapz')

    # Compute the eigenvalues and eigenvectors of W^{1/2}VW^{1/2}
    weight_sqrt = np.diag(np.sqrt(weight))
    weight_invsqrt = np.diag(1 / np.sqrt(weight))
    var = np.linalg.multi_dot([weight_sqrt, covariance.values[0], weight_sqrt])
    eigenvalues, eigenvectors = _compute_eigen(var, n_components)

    # Compute eigenfunctions = W^{-1/2}U
    eigenfunctions = np.transpose(np.dot(weight_invsqrt, eigenvectors))

    # Save the results
    results = dict()
    results["noise_variance_cov"] = data._noise_variance_cov
    results["eigenvalues"] = eigenvalues
    results["eigenfunctions"] = DenseFunctionalData(
        points, DenseValues(eigenfunctions)
    )
    return results


def _fit_inner_product(
    data: FunctionalData,
    points: DenseArgvals,
    n_components: Union[int, float] = 1,
    smooth: bool = True,
    noise_variance: Optional[float] = None,
    **kwargs
) -> Dict[str, object]:
    """Univariate Functional PCA using inner-product matrix decomposition.

    Parameters
    ----------
    data: FunctionalData
        Training data used to estimate the eigencomponents.
    points: DenseArgvals
        The sampling points at which the covariance and the eigenfunctions
        will be estimated.
    n_components: Union[int, float], default=1
        Number of components to be estimated.
    smooth: bool, default=True
        Should the mean and covariance be smoothed?
    noise_variance: Optional[float], default=None
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
        method='trapz', smooth=smooth,
        noise_variance=noise_variance, **kwargs
    )
    eigenvalues, eigenvectors = _compute_eigen(in_prod, n_components)

    # Compute the eigenfunctions
    data_smooth = data.smooth(points)
    eigenfunctions = np.matmul(data_smooth.values.T, eigenvectors)
    eigenfunctions = eigenfunctions / np.sqrt(eigenvalues)

    # Save the results
    results = dict()
    results['eigenvectors'] = eigenvectors
    results['eigenvalues'] = eigenvalues / data_smooth.n_obs
    results['eigenfunctions'] = DenseFunctionalData(
        points, DenseValues(eigenfunctions.T)
    )
    return results


#############################################################################
# Utilities to transform

def _transform_numerical_integration_dense(
    data: DenseFunctionalData,
    eigenfunctions: DenseFunctionalData,
    method: str = "trapz"
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
    axis = [argvals for argvals in data.argvals.values()]
    temp = [obs * eigenfunctions.values for obs in data.values]

    scores = np.zeros((data.n_obs, eigenfunctions.n_obs))
    for idx, curves in enumerate(temp):
        for idx_eigen, curve in enumerate(curves):
            scores[idx, idx_eigen] = _integrate(curve, *axis, method=method)
    return scores


def _transform_numerical_integration_irregular(
    data: IrregularFunctionalData,
    eigenfunctions: DenseFunctionalData,
    method: str = "trapz"
) -> npt.NDArray[np.float64]:
    """Estimate scores using numerical integration.

    Parameters
    ----------
    data: IrregularFunctionalData
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
    scores = np.zeros((data.n_obs, eigenfunctions.n_obs))
    for idx, obs in enumerate(data):
        eigen_sampled = eigenfunctions.smooth(points=obs.argvals[idx])
        temp = eigen_sampled.values * obs.values[idx]
        for idx_eigen, curve in enumerate(temp):
            scores[idx, idx_eigen] = _integrate(
                curve, obs.argvals[idx]['input_dim_0'], method=method
            )
    return scores


def _transform_pace_dense(
    data: DenseFunctionalData,
    eigenfunctions: DenseFunctionalData,
    eigenvalues: npt.NDArray[np.float64],
    covariance: DenseFunctionalData,
    noise_variance: float
) -> npt.NDArray[np.float64]:
    """Estimate scores using PACE.

    Parameters
    ----------
    data: DenseFunctionalData
        Data.
    eigenfunctions: DenseFunctionalData
        Estimate of the eigenfunctions.
    eigenvalues: npt.NDArray[np.float64]
        Estimate of the eigenvalues
    covariance: DenseFunctionalData
        Estimate of the covariance
    noise_variance: float
        Estimate of the noise_variance

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs, n_components)
        An array representing the projection of the data onto the basis of
        functions defined by the eigenfunctions.

    """
    noise_mat = noise_variance * np.eye(covariance.values[0].shape[0])
    sigma_inv = np.linalg.pinv(covariance.values[0] + noise_mat)
    return eigenvalues * np.linalg.multi_dot([
        data.values, sigma_inv, eigenfunctions.values.T
    ])


def _transform_pace_irregular(
    data: DenseFunctionalData,
    eigenfunctions: DenseFunctionalData,
    eigenvalues: npt.NDArray[np.float64],
    covariance: DenseFunctionalData,
    noise_variance: float
) -> npt.NDArray[np.float64]:
    """Estimate scores using PACE.

    Parameters
    ----------
    data: DenseFunctionalData
        Data.
    eigenfunctions: DenseFunctionalData
        Estimate of the eigenfunctions.
    eigenvalues: npt.NDArray[np.float64]
        Estimate of the eigenvalues
    covariance: DenseFunctionalData
        Estimate of the covariance
    noise_variance: float
        Estimate of the noise_variance

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs, n_components)
        An array representing the projection of the data onto the basis of
        functions defined by the eigenfunctions.

    """
    points = data.argvals.to_dense()
    argvals_cov = DenseArgvals({
        'input_dim_0': data.argvals.to_dense()['input_dim_0'],
        'input_dim_1': data.argvals.to_dense()['input_dim_0'],
    })
    bandwidth = 1 / points.n_points[0]
    covariance_sampled = covariance.smooth(
        points=argvals_cov, bandwidth=bandwidth
    )
    eigenfunctions_sampled = eigenfunctions.smooth(
        points=points, bandwidth=bandwidth
    )

    scores = np.zeros((data.n_obs, eigenfunctions.n_obs))
    for idx, fdata in enumerate(data):
        obs_points = np.isin(
            data.argvals.to_dense()['input_dim_0'],
            fdata.argvals[idx]['input_dim_0']
        )

        mask = np.outer(obs_points, obs_points)
        cov_sampled = covariance_sampled.values[0, mask].\
            reshape(2 * fdata.n_points[idx])
        eigen_sampled = eigenfunctions_sampled.values[:, obs_points]

        noise_mat = noise_variance * np.eye(cov_sampled.shape[0])
        sigma_inv = np.linalg.pinv(cov_sampled + noise_mat)
        scores[idx, :] = eigenvalues * np.linalg.multi_dot([
            fdata.values[idx], sigma_inv, eigen_sampled.T
        ])
    return scores


def _transform_innpro(
    data: DenseFunctionalData,
    eigenvectors: npt.NDArray[np.float64],
    eigenvalues: npt.NDArray[np.float64]
):
    """Estimate scores using the eigenvectors of the inner-product matrix.

    Parameters
    ----------
    data: DenseFunctionalData
        Data.
    eigenvectors: npt.NDArray[np.float64]
        Estimate of the eigenvectors of the inner-product matrix.
    eigenvalues: npt.NDArray[np.float64]
        Estimate of the eigenvalues.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs, n_components)
        An array representing the projection of the data onto the basis of
        functions defined by the eigenfunctions.

    """
    return np.sqrt(data.n_obs * eigenvalues) * eigenvectors


#############################################################################
# Class UFPCA

class UFPCA():
    """UFPCA -- Univariate Functional Principal Components Analysis.

    Linear dimensionality reduction of a univariate functional dataset. The
    projection of the data in a lower dimensional space is performed using
    a diagonalization of the covariance operator or of the inner-product matrix
    of the data.

    Parameters
    ----------
    method: str, {'covariance', 'inner-product'}, default='covariance'
        Method used to estimate the eigencomponents. If
        ``method == 'covariance'``, the estimation is based on an
        eigendecomposition of the covariance operator. If
        ``method == 'inner-product'``, the estimation is based on an
        eigendecomposition of the inner-product matrix.
    n_components: Optional[Union[int, float]], default=None
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

    References
    ----------
    .. [1] Ramsey, J. O. and Silverman, B. W. (2005), Functional Data
        Analysis, Springer Science, Chapter 8.

    """

    def __init__(
        self,
        method: str = 'covariance',
        n_components: Optional[Union[int, float]] = None,
        normalize: bool = False
    ) -> None:
        """Initaliaze UFPCA object."""
        self.n_components = n_components
        self.method = method
        self.normalize = normalize
        self.weights = 1

    @property
    def method(self) -> str:
        """Getter for `method`."""
        return self._method

    @method.setter
    def method(self, new_method: str) -> None:
        self._method = new_method

    @property
    def n_components(self) -> int:
        """Getter for `n_components`."""
        return self._n_components

    @n_components.setter
    def n_components(self, new_n_components: int) -> None:
        self._n_components = new_n_components

    @property
    def normalize(self) -> bool:
        """Getter for `normalize`."""
        return self._normalize

    @normalize.setter
    def normalize(self, new_normalize: bool) -> None:
        self._normalize = new_normalize

    @property
    def mean(self) -> DenseFunctionalData:
        """Getter for `mean`."""
        return self._mean

    @property
    def covariance(self) -> DenseFunctionalData:
        """Getter for `covariance`."""
        return self._covariance

    @property
    def eigenvalues(self) -> npt.NDArray[np.float64]:
        """Getter for `eigenvalues`."""
        return self._eigenvalues

    @property
    def eigenfunctions(self) -> DenseFunctionalData:
        """Getter for `eigenfunctions`."""
        return self._eigenfunctions

    def fit(
        self,
        data: FunctionalData,
        points: Optional[DenseArgvals] = None,
        smooth: bool = True,
        **kwargs
    ) -> None:
        """Estimate the eigencomponents of the data.

        Before estimating the eigencomponents, the data is centered. Using the
        covariance operator, the estimation is based on [1]_.

        Parameters
        ----------
        data: FunctionalData
            Training data used to estimate the eigencomponents.
        points: DenseArgvals
            The sampling points at which the covariance and the eigenfunctions
            will be estimated.
        smooth: bool, default=True
            Should the mean and covariance be smoothed?
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
        .. [1] Ramsey, J. O. and Silverman, B. W. (2005), Functional Data
            Analysis, Springer Science, Chapter 8.

        """
        if self.method == 'covariance' and data.n_dimension > 1:
            raise ValueError((
                "Estimation of the eigencomponents using the covariance "
                f"operator is not implemented for {data.n_dimension}"
                "-dimensional data."
            ))

        if points is None:
            if isinstance(data, DenseFunctionalData):
                points = data.argvals
            else:
                points = data.argvals.to_dense()

        # Compute the mean and center the data.
        self._mean = data.mean(points=points, smooth=smooth, **kwargs)
        data = data.center(mean=self._mean, smooth=smooth, **kwargs)

        # Normalize the data
        if self.normalize:
            data, self.weights = data.normalize(use_argvals_stand=True)

        # Estimate the variance of the noise
        if data.n_dimension > 1:
            self._noise_variance = 0.0
        else:
            self._noise_variance = data.noise_variance(order=2)

        if self.method == 'covariance':
            results = _fit_covariance(
                data=data, points=points, n_components=self._n_components,
                smooth=smooth, **kwargs
            )
        elif self.method == 'inner-product':
            results = _fit_inner_product(
                data=data, points=points, n_components=self.n_components,
                smooth=smooth, noise_variance=self._noise_variance, **kwargs
            )
        else:
            raise NotImplementedError(
                f"The {self.method} method not implemented."
            )

        # Save the results
        self._eigenvalues = results.get("eigenvalues", None)
        self._eigenfunctions = results.get("eigenfunctions", None)
        self._eigenvectors = results.get("eigenvectors", None)
        self._training_data = data

        # Compute an estimation of the covariance
        if data.n_dimension == 1:
            covariance = _compute_covariance(
                self._eigenvalues, self._eigenfunctions.values
            )
            self._covariance = DenseFunctionalData(
                DenseArgvals({
                    'input_dim_0': points['input_dim_0'],
                    'input_dim_1': points['input_dim_0']
                }),
                DenseValues(covariance[np.newaxis])
            )
        else:
            warnings.warn((
                "The estimation of the covariance is not performed for "
                f"{data.n_dimension}-dimensional data."
            ), UserWarning)

    def transform(
        self,
        data: Optional[DenseFunctionalData] = None,
        method: str = 'NumInt',
        **kwargs
    ) -> npt.NDArray[np.float64]:
        r"""Apply dimensionality reduction to the data.

        The functional principal components scores are defined as the
        projection of the observation :math:`X_i` on the eigenfunction
        :math:`\phi_k`. These scores are given by:

        .. math::
            c_{ik} = \int_{\mathcal{T}} \{X_i(t) - \mu(t)\}\phi_k(t)dt.

        This integral can be estimated using two ways. First, if data are
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
        data: Optional[DenseFunctionalData], default=None
            The data to be transformed. If `None`, the data are the same than
            for the `fit` method.
        method: str, {'NumInt', 'PACE', 'InnPro'}, default='NumInt'
            Method used to estimate the scores. If ``method == 'NumInt'``,
            numerical integration method is performed. If
            ``method == 'PACE'``, the PACE algorithm [1]_ is used. If
            ``method == 'InnPro'``, the estimation is performed using the
            inner product matrix of the data (can only be used if the
            eigencomponents have been estimated using the inner-product
            matrix.)
        **kwargs:
            tol: float, default=1e-4
                Tolerance parameter to prevent overflow to inverse a matrix,
                only used if ``method == 'PACE'``.
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
        .. [1] Yao, Müller and Wang (2005), Functional Data Analysis for Sparse
            Longitudinal Data. Journal of the American Statistical Association,
            100, pp. 577--590.

        """
        # Checkers
        if method == 'InnPro' and data is not None:
            raise ValueError(
                f"The method {method} can not be used as the eigencomponents "
                "have not been estimated using the provided data."
            )
        if method == 'InnPro' and self._eigenvectors is None:
            raise ValueError(
                f"The method {method} can not be used as the eigencomponents "
                "have not been estimated using the inner-product matrix."
            )
        if data is None:
            data_new = self._training_data
        else:
            # Center the data using the estimated mean in the fitting step.
            data_new = data.center(mean=self._mean)
            if self.normalize:
                data_new, _ = data.normalize(weights=self.weights)

        if method == 'NumInt':
            if isinstance(data_new, DenseFunctionalData):
                return _transform_numerical_integration_dense(
                    data_new, self.eigenfunctions,
                    method=kwargs.get('integration_method', 'trapz')
                )
            else:
                return _transform_numerical_integration_irregular(
                    data_new, self._eigenfunctions,
                    method=kwargs.get('integration_method', 'trapz')
                )
        elif method == 'PACE':
            noise_variance = max(kwargs.get('tol', 1e-4), self._noise_variance)
            if isinstance(data_new, DenseFunctionalData):
                return _transform_pace_dense(
                    data_new, self.eigenfunctions, self.eigenvalues,
                    self.covariance, noise_variance
                )
            else:
                return _transform_pace_irregular(
                    data_new, self.eigenfunctions, self.eigenvalues,
                    self.covariance, noise_variance
                )
        elif method == 'InnPro':
            return _transform_innpro(
                data_new, self._eigenvectors, self.eigenvalues
            )
        else:
            raise ValueError(f"Method {method} not implemented.")

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
        if self.eigenfunctions.n_dimension == 1:
            values = np.dot(scores, self.eigenfunctions.values)
        elif self.eigenfunctions.n_dimension == 2:
            values = np.einsum(
                'ij,jkl->ikl',
                scores,
                self.eigenfunctions.values
            )
        else:
            raise ValueError("The dimension of the data have to be 1 or 2.")
        return DenseFunctionalData(
            DenseArgvals(argvals),
            DenseValues(self.weights * values + self.mean.values)
        )


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
    method: np.str_, {'covariance', 'inner-product'}, default='covariance'
        Method used to estimate the eigencomponents. If
        ``method == 'covariance'``, the estimation is based on an
        eigendecomposition of the covariance operator of each univariate
        components. If ``method == 'inner-product'``, the estimation is
        based on an eigendecomposition of the inner-product matrix.
    n_components: Optional[List[Union[np.int64, np.float64]]], default=None
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
    normalize: np.bool_, default=False
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
        method: np.str_ = 'covariance',
        n_components: Optional[List[Union[np.int64, np.float64]]] = None,
        normalize: np.bool_ = False
    ) -> None:
        """Initialize MFPCA object."""
        self.n_components = n_components
        self.method = method
        self.normalize = normalize
        self.weights = None

    @property
    def method(self) -> np.str_:
        """Getter for `method`."""
        return self._method

    @method.setter
    def method(self, new_method: np.str_) -> None:
        self._method = new_method

    @property
    def n_components(self) -> Optional[List[Union[np.int64, np.float64]]]:
        """Getter for `n_components`."""
        return self._n_components

    @n_components.setter
    def n_components(
        self,
        new_n_components: Optional[List[Union[np.int64, np.float64]]]
    ) -> None:
        self._n_components = new_n_components

    @property
    def normalize(self) -> np.bool_:
        """Getter for `normalize`."""
        return self._normalize

    @normalize.setter
    def normalize(self, new_normalize: np.bool_) -> None:
        self._normalize = new_normalize

    @property
    def mean(self) -> DenseFunctionalData:
        """Getter for `mean`."""
        return self._mean

    @property
    def covariance(self) -> DenseFunctionalData:
        """Getter for `covariance`."""
        return self._covariance

    @property
    def eigenvalues(self) -> npt.NDArray[np.float64]:
        """Getter for `eigenvalues`."""
        return self._eigenvalues

    @property
    def eigenfunctions(self) -> DenseFunctionalData:
        """Getter for `eigenfunctions`."""
        return self._eigenfunctions

    def fit(
        self,
        data: MultivariateFunctionalData,
        scores_method: np.str_ = 'NumInt'
    ) -> None:
        """Estimate the eigencomponents of the data.

        Before estimating the eigencomponents, the data is centered. Using the
        covariance operator, the estimation is based on [1]_.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Training data used to estimate the eigencomponents.
        scores_method: np.str_, {'NumInt', 'PACE'}, default='NumInt'
            Method for the estimation of the univariate scores for the
            diagonalization of the covariance operator.

        References
        ----------
        .. [1] Happ and Greven (2018), Multivariate Functional Principal
            Component Analysis for Data Observed on Different (Dimensional)
            Domains. Journal of the American Statistical Association, 113,
            pp. 649--659.

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
                DenseArgvals(data_uni.argvals),
                DenseValues(data_uni.values - mean.values)
            ) for data_uni, mean in zip(data.data, data_mean.data)
        ])

        # Normalize the data
        if self.normalize:
            data_new, self.weights = data_new.normalize(use_argvals_stand=True)

        # Estimate eigencomponents
        self._mean = data_mean
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
        scores_method: np.str_ = 'NumInt'
    ) -> None:
        """Multivariate Functional PCA.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Training data.
        scores_method: np.str_, {'NumInt', 'PACE'}, default='NumInt'
            Method for the estimation of the univariate scores.

        """
        # Step 1: Perform univariate fPCA on each functions.
        ufpca_list, scores = [], []
        for function, n in zip(data, self.n_components):
            if function.n_dimension == 1:
                ufpca = UFPCA(n_components=n, normalize=True)
                ufpca.fit(data=function)
                scores_uni = ufpca.transform(
                    data=function, method=scores_method
                )
            elif function.n_dimension == 2:
                n_points = function.n_points
                mat_v = np.diff(np.identity(n_points['input_dim_0']))
                mat_w = np.diff(np.identity(n_points['input_dim_1']))
                ufpca = FCPTPA(n_components=n, normalize=True)
                ufpca.fit(
                    function,
                    penalty_matrices={
                        'v': np.dot(mat_v, mat_v.T),
                        'w': np.dot(mat_w, mat_w.T)
                    },
                    alpha_range={
                        'v': (1e-4, 1e4),
                        'w': (1e-4, 1e4)
                    },
                    tolerance=1e-4,
                    max_iteration=15,
                    adapt_tolerance=True
                )
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
        npc = self.n_components[0]
        if isinstance(npc, float):
            nb_axis = sum(eigenvalues.cumsum() / eigenvalues.sum() < npc)
        elif isinstance(npc, int):
            # nb_axis = npc
            nb_axis = np.sum(self.n_components)
        else:
            nb_axis = eigenvectors.shape[1]
        eigenvalues = eigenvalues[:nb_axis]
        eigenvectors = eigenvectors[:, :nb_axis]

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
            eigenfunctions.append(
                DenseFunctionalData(DenseArgvals(argvals), DenseValues(values))
            )

        self.ufpca_list = ufpca_list
        self.scores_univariate = scores_univariate
        self._covariance = covariance
        self._eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self._eigenfunctions = MultivariateFunctionalData(eigenfunctions)

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
        inn_pro = data.inner_product()
        # Compute inner product matrix and its eigendecomposition
        eigenvalues, eigenvectors = _compute_eigen(
            inn_pro,
            self.n_components
        )
        eigenfunctions = [
            DenseFunctionalData(
                DenseArgvals(data_uni.argvals),
                DenseValues(np.transpose(
                    np.matmul(
                        data_uni.values.T, eigenvectors
                    ) / np.sqrt(eigenvalues)
                ))
            ) for data_uni in data.data
        ]

        self._eigenvectors = eigenvectors
        self._eigenvalues = eigenvalues / data.n_obs
        self._eigenfunctions = MultivariateFunctionalData(eigenfunctions)

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
        method: np.str_ = 'NumInt',
        **kwargs
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

        Parameters
        ----------
        data: MultivariateFunctionalData
            Data
        method: np.str_, {'NumInt', 'InnPro'}, default='NumInt'
            Method used to estimate the scores. If ``method == 'NumInt'``,
            numerical integration method is performed. If
            ``method == 'InnPro'``, the estimation is performed using the
            inner product matrix of the data (can only be used if the
            eigencomponents have been estimated using the inner-product
            matrix.)
        **kwargs:
            integration_method: np.str_, {'trapz', 'simpson'}, default='trapz'
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
                DenseArgvals(data_uni.argvals),
                DenseValues(data_uni.values - mean.values)
            ) for data_uni, mean in zip(data, self.mean)
        ])

        # TODO: Add checkers
        if self.normalize:
            data_new, self.weights = data_new.normalize(use_argvals_stand=True)
            # values = data.values / self.weights
            # data = MultivariateFunctionalData(data.argvals, values)

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
        method: np.str_ = "trapz"
    ) -> npt.NDArray[np.float64]:
        """Estimate scores using numerical integration.

        Parameters
        ----------
        data: MultivariateFunctionalData
            Data
        method: np.str_, {'trapz', 'simpson'}, default='trapz'
            Method used to perform numerical integration.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_components)
            An array representing the projection of the data onto the basis of
            functions defined by the eigenfunctions.

        TODO: Change _integrate

        """
        scores_uni = [None] * len(self.eigenfunctions)
        for idx, (dat_uni, eigen) in enumerate(zip(data, self.eigenfunctions)):
            projection = _integrate(
                [traj * eigen.values for traj in dat_uni.values],
                dat_uni.argvals['input_dim_0'],
                method=method
            )
            if eigen.n_dimension == 1:
                scores_uni[idx] = projection
            elif eigen.n_dimension == 2:
                scores_uni[idx] = _integrate(
                    projection,
                    dat_uni.argvals['input_dim_1'],
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
        if self.weights is None:
            self.weights = np.repeat(1, self.eigenfunctions.n_functional)
        res = [None] * self.eigenfunctions.n_functional
        for idx, (mean, eigenfunction, weight) in enumerate(
            zip(self.mean, self.eigenfunctions, self.weights)
        ):
            if eigenfunction.n_dimension == 1:
                values = np.dot(scores, eigenfunction.values)
            elif eigenfunction.n_dimension == 2:
                values = np.einsum('ij,jkl->ikl', scores, eigenfunction.values)
            else:
                raise ValueError(
                    "The dimension of the data have to be 1 or 2."
                )
            res[idx] = DenseFunctionalData(
                DenseArgvals(eigenfunction.argvals),
                DenseValues(weight * values + mean.values)
            )
        return MultivariateFunctionalData(res)
