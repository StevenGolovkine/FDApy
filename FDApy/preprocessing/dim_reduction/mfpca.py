#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Multivariate Functional Principal Components Analysis
-----------------------------------------------------

"""
import numpy as np
import numpy.typing as npt

from typing import Dict, List

from ...representation.argvals import DenseArgvals
from ...representation.values import DenseValues
from ...representation.functional_data import (
    FunctionalData,
    DenseFunctionalData,
    IrregularFunctionalData,
    BasisFunctionalData,
    MultivariateFunctionalData,
)
from ...representation.basis import Basis

from .ufpca import (
    UFPCA,
    _transform_numerical_integration_dense,
    _transform_numerical_integration_irregular,
    _transform_innpro,
)
from .fcp_tpa import FCPTPA
from ...misc.utils import _compute_eigen, _block_diag


#############################################################################
# Utilities to fit


def _univariate_decomposition(
    data: FunctionalData,
    method: str = "UFPCA",
    n_components: int | float = 2,
    **kwargs,
) -> BasisFunctionalData:
    """Univariate basis decomposition.

    This method estimates a univariate basis decomposition for a univariate functional
    dataset.

    Parameters
    ----------
    data
        Training data.
    method
        The method that specify the basis for which the decomposition is to be
        calculated. It should one of `{'UFPCA', 'PSplines', 'FCPTPA'}`.
    n_components
        Number of components to be estimated.
    kwargs
        Other keyword arguments are passed to the functions:
        :meth:`UFPCA.fit` and :meth:`UFPCA.transform` and
        :meth:`GridFunctionalData.to_basis`.

    Returns
    -------
    BasisFunctionalData
        The basis representation of the functional data.

    Raises
    ------
    ValueError
        If the provided method is not implemented.

    """
    if method == "UFPCA":
        ufpca = UFPCA(n_components=n_components, normalize=False)
        ufpca.fit(data=data, points=None, **kwargs)
        scores = ufpca.transform(data=None, method="PACE", **kwargs)
        basis = Basis(
            name="given",
            argvals=ufpca.eigenfunctions.argvals,
            values=ufpca.eigenfunctions.values,
        )
        data_basis = BasisFunctionalData(basis=basis, coefficients=scores)
    elif method == "PSplines":
        data_basis = data.to_basis(**kwargs)
    elif method == "FCPTPA":
        if isinstance(data, IrregularFunctionalData):
            data = data.smooth(method="interpolation")
        n_points = data.n_points
        mat_v = np.diff(np.identity(n_points[0]))
        mat_w = np.diff(np.identity(n_points[1]))
        ufpca = FCPTPA(n_components=n_components, normalize=False)
        ufpca.fit(
            data,
            penalty_matrices={"v": np.dot(mat_v, mat_v.T), "w": np.dot(mat_w, mat_w.T)},
            alpha_range={"v": (1e-5, 1e5), "w": (1e-5, 1e5)},
            tolerance=1e-4,
            max_iteration=30,
            adapt_tolerance=True,
        )
        scores = ufpca.transform(data, method="FCPTPA")
        basis = Basis(
            name="given",
            argvals=ufpca.eigenfunctions.argvals,
            values=ufpca.eigenfunctions.values,
        )
        data_basis = BasisFunctionalData(basis=basis, coefficients=scores)
    else:
        raise ValueError("Method not implemented.")
    return data_basis


def _fit_covariance_multivariate(
    data: MultivariateFunctionalData,
    n_components: int | float,
    univariate_expansions: List[Dict[str, object]],
    points: List[DenseArgvals] | None = None,
) -> Dict[str, object]:
    """Multivariate Functional PCA using the covariance operator.

    Parameters
    ----------
    data
        Training data.
    points
        The sampling points at which the covariance and the eigenfunctions
        will be estimated.
    n_components
        Number of components to keep.
    univariate_expansions
        List of dictionaries characterizing the univariate expansion computed for each
        component.
    points
        The sampling points at which the covariance and the eigenfunctions
        will be estimated.

    Returns
    -------
    Dict[str, object]
        A dictionary with entries:

        - `'eigenvalues'`: the estimated eigenvalues;
        - `'eigenfunctions'`: the estimated eigenfunctions;
        - `'_basis_univariate'`: the univariate basis decompositions;
        - `'_scores_univariate'`: the univariate scores;
        - `'_eigenvectors'`: the eigenvectors from the decomposition of the scores.

    """
    # Step 1: Perform univariate decomposition on each functions.
    basis_univariate = []
    for univariate_data, univariate_expansion in zip(data.data, univariate_expansions):
        method = univariate_expansion.pop("method", "PSplines")
        n_component = univariate_expansion.pop("n_components", 5)
        temp = _univariate_decomposition(
            data=univariate_data,
            method=method,
            n_components=n_component,
            **univariate_expansion,
        )
        basis_univariate.append(temp)
    scores_univariate = np.hstack([basis.coefficients for basis in basis_univariate])

    # Step 2: Estimation of the covariance of the scores.
    cholesky_matrices = [
        np.linalg.cholesky(basis.basis.inner_product()).T for basis in basis_univariate
    ]
    cholesky_matrix = _block_diag(*cholesky_matrices)

    scores_normed = scores_univariate / np.sqrt(len(scores_univariate) - 1)
    scores_cov = (cholesky_matrix.T @ cholesky_matrix) @ np.cov(scores_univariate.T)

    # Step 3: Eigenanalysis of the covariance of the scores.
    eigenvalues, eigenvectors = _compute_eigen(scores_cov, n_components=n_components)

    norm_factor = 1 / np.sqrt(
        np.diag((scores_normed @ eigenvectors).T @ (scores_normed @ eigenvectors))
    )
    scores = scores_normed @ eigenvectors * np.sqrt(len(scores_univariate) - 1)
    scores = scores @ np.diag(np.sqrt(eigenvalues) * norm_factor)

    # Step 4: Estimation of the multivariate eigenfunctions.
    # Retrieve the number of eigenfunctions for each univariate function.
    npc = [basis.coefficients.shape[1] for basis in basis_univariate]
    nb_eigenfunction_uni = [0]
    nb_eigenfunction_uni.extend(npc)
    nb_eigenfunction_uni_cum = np.cumsum(nb_eigenfunction_uni)

    # Compute the multivariate eigenbasis.
    weights = scores_normed.T @ scores_normed @ eigenvectors

    eigenfunctions = []
    for idx, basis in enumerate(basis_univariate):
        start = nb_eigenfunction_uni_cum[idx]
        end = nb_eigenfunction_uni_cum[idx + 1]
        values = 1 / np.sqrt(eigenvalues) * norm_factor * weights[start:end, :]
        univariate_eigen = BasisFunctionalData(coefficients=values.T, basis=basis.basis)
        eigenfunctions.append(univariate_eigen)

    # Save the results
    results = dict()
    results["eigenvalues"] = eigenvalues
    results["eigenfunctions"] = MultivariateFunctionalData(eigenfunctions)
    results["_basis_univariate"] = basis_univariate
    results["_scores_univariate"] = scores_univariate
    results["_eigenvectors"] = eigenvectors
    return results


def _fit_inner_product_multivariate(
    data: MultivariateFunctionalData,
    n_components: int | float = 1,
    points: List[DenseArgvals] | None = None,
    **kwargs,
) -> Dict[str, object]:
    """Multivariate Functional PCA using inner-product matrix decomposition.

    Parameters
    ----------
    data
        Training data used to estimate the eigencomponents.
    n_components
        Number of components to be estimated.
    points
        The sampling points at which the eigenfunctions will be estimated.
    kwargs:
        Other keyword arguments are passed to the function:
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
    in_prod = data.inner_product(
        method_integration="trapz", method_smoothing="PS", **kwargs
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
    results["_eigenvectors"] = eigenvectors
    results["eigenvalues"] = eigenvalues / data.n_obs
    results["eigenfunctions"] = eigenfunctions
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
    scores = [None] * eigenfunctions.n_functional
    for idx, (eigen, data) in enumerate(zip(eigenfunctions.data, data.data)):
        if data.n_dimension > 1:
            data = data.smooth(method="PS")
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
    eigenvectors
        Estimate of the eigenvectors of the scores.
    scores_univariate
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
    r"""Multivariate functional principal components analysis.

    Linear dimensionality reduction of a multivariate functional dataset. The
    projection of the data in a lower dimensional space is performed using
    a diagonalization of the covariance operator of each univariate component
    or of the inner-product matrix of the data. It is assumed that the data
    have :math:`P` components.

    Parameters
    ----------
    n_components
        Number of components to keep. If `n_components` is an integer, `n_components`
        are kept. If `0 < n_components < 1`, we select the number of components such
        that the amount of variance that needs to be explained is greater than the
        percentage specified by `n_components`.
    univariate_expansions
        List of dictionaries characterizing the univariate expansion computed for each
        component.
    method
        Method used to estimate the eigencomponents. If
        `method == 'covariance'`, the estimation is based on an
        eigendecomposition of the covariance operator of each univariate
        components. If `method == 'inner-product'`, the estimation is
        based on an eigendecomposition of the inner-product matrix.
    weights
        A vector of weights of length :math:`P`. If `None`, we set the weights to be
        equal to 1 for each component.
    normalize
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
    .. [1] Golovkine, S., Gunning, E., Simpkin, A.J., Bargary, N. (2023). On the use
        of the Gram matrix for multivariate functional principal components
        analysis.
    .. [2] Happ and Greven (2018), Multivariate Functional Principal
        Component Analysis for Data Observed on Different (Dimensional)
        Domains. Journal of the American Statistical Association, 113,
        pp. 649--659.

    """

    def __init__(
        self,
        n_components: int | float = 2,
        univariate_expansions: List[Dict[str, object]] | None = None,
        method: str = "covariance",
        weights: npt.NDArray[np.float64] | None = None,
        normalize: bool = False,
    ) -> None:
        """Initialize MFPCA object."""
        self.n_components = n_components
        self.univariate_expansions = univariate_expansions
        self.method = method
        self.weights = weights
        self.normalize = normalize

    @property
    def n_components(self) -> int | float:
        """Getter for `n_components`."""
        return self._n_components

    @n_components.setter
    def n_components(self, new_n_components: int | float) -> None:
        self._n_components = new_n_components

    @property
    def univariate_expansion(self) -> List[Dict[str, object]] | None:
        """Gettter for `univariate_expansion`."""
        return self._univariate_expansion

    @univariate_expansion.setter
    def univariate_expansion(
        self, new_univariate_expansion: List[Dict[str, object]] | None
    ) -> None:
        self._univariate_expansion = new_univariate_expansion

    @property
    def method(self) -> str:
        """Getter for `method`."""
        return self._method

    @method.setter
    def method(self, new_method: str) -> None:
        self._method = new_method

    @property
    def weights(self) -> npt.NDArray[np.float64] | None:
        """Getter for `weights`."""
        return self._weights

    @weights.setter
    def weights(self, new_weights: npt.NDArray[np.float64] | None) -> None:
        self._weights = new_weights

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
        points: List[DenseArgvals] | None = None,
        method_smoothing: str = None,
        **kwargs,
    ) -> None:
        """Estimate the eigencomponents of the data.

        Before estimating the eigencomponents, the data is centered. Using the
        covariance operator, the estimation is based on [2]_. Using the Gram matrix, the
        estimation is based on [1]_.

        Parameters
        ----------
        data
            Training data used to estimate the eigencomponents.
        points
            The sampling points at which the covariance and the eigenfunctions
            will be estimated.
        method_smoothing
            Should the mean and covariance be smoothed?
        kwargs
            Other keyword arguments are passed to the functions:
            :meth:`FunctionalData.mean`, :meth:`FunctionalData.center` and
            :meth:`preprocessing.dim_reduction.mfpca._fit_inner_product_multivariate`.

        """
        if points is None:
            points = [
                (
                    data_univariate.argvals
                    if isinstance(data_univariate, DenseFunctionalData)
                    else data_univariate.argvals.to_dense()
                )
                for data_univariate in data.data
            ]
        if self.weights is None:
            self.weights = np.repeat(1, data.n_functional)

        # Compute the mean and center the data.
        self._mean = data.mean(
            points=points, method_smoothing=method_smoothing, **kwargs
        )
        # pen = self._mean.n_dimension * (0,)
        data = data.center(mean=self._mean, method_smoothing="PS")

        # Normalize the data
        if self.normalize:
            data, self.weights = data.rescale(
                method_smoothing=method_smoothing, use_argvals_stand=True
            )

        # Estimate the variance of the noise
        self._noise_variance = data.noise_variance(order=2)

        # Estimate eigencomponents
        if self.method == "covariance":
            results = _fit_covariance_multivariate(
                data=data,
                n_components=self.n_components,
                univariate_expansions=self.univariate_expansions,
                points=points,
            )
        elif self.method == "inner-product":
            results = _fit_inner_product_multivariate(
                data=data,
                n_components=self.n_components,
                points=points,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"{self.method} method not implemented.")

        # Save the results
        self._eigenvalues = results.get("eigenvalues", None)
        self._eigenfunctions = results.get("eigenfunctions", None)
        self._eigenvectors = results.get("eigenvectors", None)
        self._scores_univariate = results.get("_scores_univariate", None)
        self._eigenvectors = results.get("_eigenvectors", None)
        self._basis_univariate = results.get("_basis_univariate", None)
        self._training_data = data

        self._covariance = None

    def transform(
        self,
        data: MultivariateFunctionalData | None = None,
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

        Parameters
        ----------
        data
            Data
        method
            Method used to estimate the scores. If ``method == 'NumInt'``,
            numerical integration method is performed. If
            ``method == 'InnPro'``, the estimation is performed using the
            inner product matrix of the data (can only be used if the
            eigencomponents have been estimated using the inner-product
            matrix.)
        method_smoothing
            Method to smooth the data before estimating the scores.
        kwargs
            Other keyword arguments are passed to the function:
            :meth:`FunctionalData.center`.

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
        and Greven [2]_ could do.

        """
        # Checkers
        if method == "InnPro" and data is not None:
            raise ValueError(
                f"The method {method} can not be used as the eigencomponents "
                "have not been estimated using the provided data."
            )
        if method == "InnPro" and self.method == "covariance":
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
                self._eigenfunctions.to_grid(),
                kwargs.get("integration_method", "trapz"),
            )
        elif method == "PACE":
            return _transform_pace_multivariate(
                self._eigenvectors, self._scores_univariate
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
        scores
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
            zip(self.mean.data, self.eigenfunctions.to_grid().data, self.weights)
        ):
            values = np.einsum("ij,j... -> i...", scores, eigenfunction.values)
            results[idx] = DenseFunctionalData(
                DenseArgvals(eigenfunction.argvals),
                DenseValues(weight * values + mean.values),
            )
        return MultivariateFunctionalData(results)
