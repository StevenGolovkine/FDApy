#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Univariate Functional Principal Components Analysis
---------------------------------------------------

"""
import numpy as np
import numpy.typing as npt
import warnings

from typing import Any, Dict

from ...representation.argvals import DenseArgvals
from ...representation.values import DenseValues
from ...representation.functional_data import (
    FunctionalData,
    DenseFunctionalData,
    IrregularFunctionalData,
)
from ...misc.utils import (
    _compute_covariance,
    _integrate,
    _integration_weights,
    _compute_eigen,
)


#############################################################################
# Utilities to fit


def _fit_covariance(
    data: FunctionalData,
    points: DenseArgvals,
    n_components: int | float = 1,
    method_smoothing: str = "LP",
    **kwargs: Any,
) -> Dict[str, object]:
    """Univariate Functional PCA using the covariance operator.

    This method estimate eigencomponents of a functional dataset using the
    diagonalization of the covariance operator.

    Parameters
    ----------
    data
        Training data.
    points
        The sampling points at which the covariance and the eigenfunctions
        will be estimated.
    n_components
        Number of components to be estimated.
    method_smoothing
        Method to smooth the covariance function.
    kwargs
        Other keyword arguments are passed to the function
        :meth:`FunctionalData.covariance`.

    Returns
    -------
    Dict[str, object]
        A dictionary with entries:

        - `'eigenvalues'`: the estimated eigenvalues;
        - `'eigenfunctions'`: the estimated eigenfunctions;
        - `'noise_variance_cov'`: the estimation of the noise.

    """
    # Compute the covariance
    covariance = data.covariance(
        points=points, method_smoothing=method_smoothing, center=False, **kwargs
    )

    # Choose the W_j's and the S_j's (Ramsay and Silverman, 2005)
    argvals = points["input_dim_0"]
    weight = _integration_weights(argvals, method="trapz")

    # Compute the eigenvalues and eigenvectors of W^{1/2}VW^{1/2}
    weight_sqrt = np.diag(np.sqrt(weight))
    weight_invsqrt = np.diag(1 / np.sqrt(weight))
    covariance_matrix = weight_sqrt @ covariance.values[0] @ weight_sqrt
    eigenvalues, eigenvectors = _compute_eigen(covariance_matrix, n_components)

    # Compute eigenfunctions = W^{-1/2}U
    eigenfunctions = (weight_invsqrt @ eigenvectors).T

    # Save the results
    results = dict()
    results["noise_variance_cov"] = data._noise_variance_cov
    results["eigenvalues"] = eigenvalues
    results["eigenfunctions"] = DenseFunctionalData(points, DenseValues(eigenfunctions))
    return results


def _fit_inner_product(
    data: FunctionalData,
    points: DenseArgvals,
    n_components: np.int64 | np.float64 = 1,
    method_smoothing: str = "LP",
    noise_variance: float | None = None,
    **kwargs: Any,
) -> Dict[str, object]:
    """Univariate Functional PCA using inner-product matrix decomposition.

    This method estimate the eigencomponents of a functional dataset using the
    diagonalization of the inner-product (or Gram) matrix.

    Parameters
    ----------
    data
        Training data used to estimate the eigencomponents.
    points
        The sampling points at which the covariance and the eigenfunctions
        will be estimated.
    n_components
        Number of components to be estimated.
    method_smoothing
        Method to smooth the covariance function.
    noise_variance
        An estimation of the variance of the noise. If `None`, an estimation is computed
        using the method :meth:`FunctionalData.noise_variance`.
    **kwargs
        Other keyword arguments are passed to the function
        :meth:`FunctionalData.inner_product`.

    Returns
    -------
    Dict[str, object]
        A dictionary with entries:

        - `'eigenvalues'`: the estimated eigenvalues;
        - `'eigenfunctions'`: the estimated eigenfunctions;
        - `'eigenvectors'`: the estimated eigenvectors of the Gram matrix.

    """
    # Compute inner product matrix and its eigendecomposition
    gram_matrix = data.inner_product(
        method_integration="trapz",
        method_smoothing=method_smoothing,
        noise_variance=noise_variance,
        **kwargs,
    )
    eigenvalues, eigenvectors = _compute_eigen(gram_matrix, n_components)

    # Compute the eigenfunctions
    eigenfunctions = data._data_inpro.values.T @ eigenvectors / np.sqrt(eigenvalues)
    eigenfunctions = DenseFunctionalData(data._data_inpro.argvals, eigenfunctions.T)

    # Save the results
    results = dict()
    results["eigenvectors"] = eigenvectors
    results["eigenvalues"] = eigenvalues / data._data_inpro.n_obs
    results["eigenfunctions"] = eigenfunctions
    return results


#############################################################################
# Utilities to transform


def _transform_numerical_integration_dense(
    data: DenseFunctionalData,
    eigenfunctions: DenseFunctionalData,
    method: str = "trapz",
) -> npt.NDArray[np.float64]:
    """Estimate scores using numerical integration.

    Parameters
    ----------
    data
        Data.
    eigenfunctions
        Estimate of the eigenfunctions.
    method
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
    method: str = "trapz",
) -> npt.NDArray[np.float64]:
    """Estimate scores using numerical integration.

    Parameters
    ----------
    data
        Data.
    eigenfunctions
        Estimate of the eigenfunctions.
    method
        Method used to perform numerical integration.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs, n_components)
        An array representing the projection of the data onto the basis of
        functions defined by the eigenfunctions.

    """
    data_s = data.smooth(method="interpolation")
    return _transform_numerical_integration_dense(data_s, eigenfunctions, method)


def _transform_pace_dense(
    data: DenseFunctionalData,
    eigenfunctions: DenseFunctionalData,
    eigenvalues: npt.NDArray[np.float64],
    covariance: DenseFunctionalData,
    noise_variance: float,
) -> npt.NDArray[np.float64]:
    """Estimate scores using PACE.

    Parameters
    ----------
    data
        Data.
    eigenfunctions
        Estimate of the eigenfunctions.
    eigenvalues
        Estimate of the eigenvalues
    covariance
        Estimate of the covariance
    noise_variance
        Estimate of the noise_variance

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs, n_components)
        An array representing the projection of the data onto the basis of
        functions defined by the eigenfunctions.

    """
    noise_mat = noise_variance * np.eye(covariance.values[0].shape[0])
    sigma_inv = np.linalg.pinv(covariance.values[0] + noise_mat)
    return eigenvalues * np.linalg.multi_dot(
        [data.values, sigma_inv, eigenfunctions.values.T]
    )


def _transform_pace_irregular(
    data: DenseFunctionalData,
    eigenfunctions: DenseFunctionalData,
    eigenvalues: npt.NDArray[np.float64],
    covariance: DenseFunctionalData,
    noise_variance: float,
) -> npt.NDArray[np.float64]:
    """Estimate scores using PACE.

    Parameters
    ----------
    data
        Data.
    eigenfunctions
        Estimate of the eigenfunctions.
    eigenvalues
        Estimate of the eigenvalues
    covariance
        Estimate of the covariance
    noise_variance
        Estimate of the noise_variance

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs, n_components)
        An array representing the projection of the data onto the basis of
        functions defined by the eigenfunctions.

    """
    data = data.smooth(method="interpolation")
    # points = data.argvals.to_dense()
    # argvals_cov = DenseArgvals(
    #     {
    #         "input_dim_0": data.argvals.to_dense()["input_dim_0"],
    #         "input_dim_1": data.argvals.to_dense()["input_dim_0"],
    #     }
    # )
    # covariance_sampled = covariance.smooth(points=argvals_cov, method="PS")
    # eigenfunctions_sampled = eigenfunctions.smooth(points=points, method="PS")

    # scores = np.zeros((data.n_obs, eigenfunctions.n_obs))
    # for idx, obs in enumerate(data):
    #     nan_mask = np.isnan(obs.values[idx])
    #     new_argvals = obs.argvals[idx]["input_dim_0"][~nan_mask]
    #     new_values = obs.values[idx][~nan_mask]

    #     obs_points = np.isin(points["input_dim_0"], new_argvals)

    #     mask = np.outer(obs_points, obs_points)
    #     cov_sampled = covariance_sampled.values[0, mask].\
    #       reshape(2 * (len(new_values),))
    #     eigen_sampled = eigenfunctions_sampled.values[:, obs_points]

    #     noise_mat = noise_variance * np.eye(cov_sampled.shape[0])
    #     sigma_inv = np.linalg.pinv(cov_sampled + noise_mat)
    #     scores[idx, :] = eigenvalues * np.linalg.multi_dot(
    #         [new_values, sigma_inv, eigen_sampled.T]
    #     )
    return _transform_pace_dense(
        data, eigenfunctions, eigenvalues, covariance, noise_variance
    )


def _transform_innpro(
    data: DenseFunctionalData,
    eigenvectors: npt.NDArray[np.float64],
    eigenvalues: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Estimate scores using the eigenvectors of the inner-product matrix.

    Parameters
    ----------
    data
        Data.
    eigenvectors
        Estimate of the eigenvectors of the inner-product matrix.
    eigenvalues
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


class UFPCA:
    """Univariate functional principal components analysis.

    Linear dimensionality reduction of a univariate functional dataset. The
    projection of the data in a lower dimensional space is performed using
    a diagonalization of the covariance operator or of the inner-product matrix
    of the data.

    Parameters
    ----------
    method
        Method used to estimate the eigencomponents. If
        ``method == 'covariance'``, the estimation is based on an
        eigendecomposition of the covariance operator. If
        ``method == 'inner-product'``, the estimation is based on an
        eigendecomposition of the inner-product matrix.
    n_components
        Number of components to keep. If `n_components` is `None`, all
        components are kept, ``n_components == min(n_samples, n_features)``.
        If `n_components` is an integer, `n_components` are kept. If
        `0 < n_components < 1`, select the number of components such that the
        amount of variance that needs to be explained is greater than the
        percentage specified by `n_components`.
    normalize
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
    .. [1] Ramsay, J. O. and Silverman, B. W. (2005), Functional Data
        Analysis, Springer Science, Chapter 8.
    .. [2] Yao, Müller and Wang (2005), Functional Data Analysis for Sparse
        Longitudinal Data. Journal of the American Statistical Association,
        100, pp. 577--590.


    """

    def __init__(
        self,
        method: str = "covariance",
        n_components: int | float | None = None,
        normalize: bool = False,
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
        points: DenseArgvals | None = None,
        method_smoothing: str = None,
        kwargs_mean: Dict[str, object] = {},
        kwargs_covariance: Dict[str, object] = {},
        kwargs_innpro: Dict[str, object] = {},
    ) -> None:
        """Estimate the eigencomponents of the data.

        Before estimating the eigencomponents, the data is centered. Using the
        covariance operator, the estimation is based on [1]_.

        Parameters
        ----------
        data
            Training data used to estimate the eigencomponents.
        points
            The sampling points at which the covariance and the eigenfunctions
            will be estimated.
        method_smoothing
            Should the mean and covariance be smoothed?
        kwargs_mean
            Keywords arguments to be passed to the function :meth:`FunctionalData.mean`.
        kwargs_covariance
            Keywords arguments to be passed to the function
            :meth:`preprocessing.ufpca._fit_covariance`.
        kwargs_innpro
            Keywords arguments to be passed to the function
            :meth:`preprocessing.ufpca._fit_inner_product`.

        """
        if self.method == "covariance" and data.n_dimension > 1:
            raise ValueError(
                (
                    "Estimation of the eigencomponents using the covariance "
                    f"operator is not implemented for {data.n_dimension}"
                    "-dimensional data."
                )
            )

        if points is None:
            if isinstance(data, DenseFunctionalData):
                points = data.argvals
            else:
                points = data.argvals.to_dense()

        # Compute the mean and center the data.
        self._mean = data.mean(
            points=points, method_smoothing=method_smoothing, **kwargs_mean
        )
        if isinstance(data, IrregularFunctionalData):
            penalty = data.n_dimension * (0,)
            data = data.center(mean=self._mean, method_smoothing="PS", penalty=penalty)
        else:
            data = data.center(mean=self._mean, method_smoothing=None)

        # Normalize the data
        if self.normalize:
            data, self.weights = data.rescale(
                method_smoothing=method_smoothing, use_argvals_stand=True
            )

        # Estimate the variance of the noise
        if data.n_dimension > 1:
            self._noise_variance = 0.0
        else:
            self._noise_variance = data.noise_variance(order=2)

        if self.method == "covariance":
            results = _fit_covariance(
                data=data,
                points=points,
                n_components=self._n_components,
                method_smoothing=method_smoothing,
                **kwargs_covariance,
            )
        elif self.method == "inner-product":
            results = _fit_inner_product(
                data=data,
                points=points,
                n_components=self.n_components,
                method_smoothing=method_smoothing,
                noise_variance=self._noise_variance,
                **kwargs_innpro,
            )
        else:
            raise NotImplementedError(f"The {self.method} method not implemented.")

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
                DenseArgvals(
                    {
                        "input_dim_0": points["input_dim_0"],
                        "input_dim_1": points["input_dim_0"],
                    }
                ),
                DenseValues(covariance[np.newaxis]),
            )
        else:
            warnings.warn(
                (
                    "The estimation of the covariance is not performed for "
                    f"{data.n_dimension}-dimensional data."
                ),
                UserWarning,
            )

    def transform(
        self,
        data: DenseFunctionalData | None = None,
        method: str = "NumInt",
        method_smoothing: str = "LP",
        **kwargs: Any,
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
        Conditional Expectation) algorithm [2]_ is used for sparse functional
        data. If the eigenfunctions have been estimated using the inner-product
        matrix, the scores can also be estimated using the formula

        .. math::
            c_{ik} = \sqrt{l_k}v_{ik},

        where :math:`l_k` and :math:`v_{k}` are the eigenvalues and
        eigenvectors of the inner-product matrix.

        Parameters
        ----------
        data
            The data to be transformed. If `None`, the data are the same than
            for the `fit` method.
        method
            Method used to estimate the scores. If ``method == 'NumInt'``,
            numerical integration method is performed. If
            ``method == 'PACE'``, the PACE algorithm [1]_ is used. If
            ``method == 'InnPro'``, the estimation is performed using the
            inner product matrix of the data (can only be used if the
            eigencomponents have been estimated using the inner-product
            matrix.)
        method_smoothing
            Should the mean and covariance be smoothed?
        kwargs
            See below

        Keyword Arguments
        -----------------
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
            if isinstance(data_new, DenseFunctionalData):
                return _transform_numerical_integration_dense(
                    data_new,
                    self.eigenfunctions,
                    method=kwargs.get("integration_method", "trapz"),
                )
            else:
                return _transform_numerical_integration_irregular(
                    data_new,
                    self._eigenfunctions,
                    method=kwargs.get("integration_method", "trapz"),
                )
        elif method == "PACE":
            noise_variance = max(kwargs.get("tol", 1e-4), self._noise_variance)
            if isinstance(data_new, DenseFunctionalData):
                return _transform_pace_dense(
                    data_new,
                    self.eigenfunctions,
                    self.eigenvalues,
                    self.covariance,
                    noise_variance,
                )
            else:
                return _transform_pace_irregular(
                    data_new,
                    self.eigenfunctions,
                    self.eigenvalues,
                    self.covariance,
                    noise_variance,
                )
        elif method == "InnPro":
            return _transform_innpro(data_new, self._eigenvectors, self.eigenvalues)
        else:
            raise ValueError(f"Method {method} not implemented.")

    def inverse_transform(self, scores: npt.NDArray[np.float64]) -> DenseFunctionalData:
        r"""Transform the data back to its original space.

        Given a set of scores :math:`c_{ik}`, we reconstruct the observations
        using a truncation of the Karhunen-Loève expansion,

        .. math::
            X_{i}(t) = \mu(t) + \sum_{k = 1}^K c_{ik}\phi_k(t).

        Data can be multidimensional.

        Parameters
        ----------
        scores
            New data, where `n_obs` is the number of observations and
            `n_components` is the number of components.

        Returns
        -------
        DenseFunctionalData
            A DenseFunctionalData object representing the transformation of the
            scores into the original curve space.

        """
        values = np.einsum("ij , j... -> i...", scores, self.eigenfunctions.values)
        return DenseFunctionalData(
            DenseArgvals(self.eigenfunctions.argvals),
            DenseValues(self.weights * values + self.mean.values),
        )
