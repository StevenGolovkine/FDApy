#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Functional Data
---------------

"""
from __future__ import annotations

import itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings

from functools import reduce

from abc import ABC, abstractmethod
from collections import UserList
from collections.abc import Iterator
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
    Type,
    TYPE_CHECKING,
)

from .argvals import Argvals, DenseArgvals, IrregularArgvals
from .values import Values, DenseValues, IrregularValues

from ..preprocessing.smoothing.local_polynomial import LocalPolynomial
from ..preprocessing.smoothing.psplines import PSplines, _format_data

from ..misc.utils import _cartesian_product
from ..misc.utils import _estimate_noise_variance
from ..misc.utils import _inner_product
from ..misc.utils import _integrate
from ..misc.utils import _outer

if TYPE_CHECKING:
    from .basis import Basis


###############################################################################
# Utilities function
def _tensor_product(
    data1: DenseFunctionalData, data2: DenseFunctionalData
) -> DenseFunctionalData:
    """Compute the tensor product between functional data.

    Compute the tensor product between all the observation of data1 with all
    the observation of data2.

    Parameters
    ----------
    data1
        First functional data.
    data2
        Second functional data.

    Returns
    -------
    DenseFunctionalData
        The tensor product between data1 and data2. It contains data1.n_obs *
        data2.n_obs observations.

    """
    arg = {
        "input_dim_0": data1.argvals["input_dim_0"],
        "input_dim_1": data2.argvals["input_dim_0"],
    }
    val = [_outer(i, j) for i in data1.values for j in data2.values]
    return DenseFunctionalData(DenseArgvals(arg), DenseValues(np.array(val)))


def _smooth_covariance(
    covariance_matrix: npt.NDArray[np.float64],
    argvals: DenseArgvals,
    points: DenseArgvals,
    method_smoothing: str = "LP",
    remove_diagonal: bool = True,
    weights: npt.NDArray[np.float64] | None = None,
    **kwargs,
):
    """Smooth the covariance.

    Parameters
    ----------
    covariance_matrix
        The samnpled covariance.
    argvals
        The sampling points at which the raw covariance is estimated.
    points
        The sampling points at which the smoothed covariance will be estimated.
    method_smoothing
        The method to used for the smoothing of the mean. If 'PS', the method is
        P-splines [1]_. If 'LP', the method is local polynomials [2]_.
    remove_diagonal
        Should the diagonal of the covariance be removed due to measurement errors.
    weights
        Matrix of weights.
    kwargs
        Other keyword arguments are passed to one of the following functions:
        :meth:`preprocessing.smoothing.PSplines` (``method='PS'``) and
        :meth:`preprocessing.smoothing.LocalPolynomial` (``method='LP'``).

    Returns
    -------
    npt.NDArray
        The smooth covariance.

    References
    ----------
    .. [1] Eilers, P. H. C., Marx, B. D. (2021). Practical Smoothing: The Joys
        of P-splines. Cambridge University Press, Cambridge.
    .. [2] Zhang, J.-T. and Chen J. (2007), Statistical Inferences for
        Functional Data, The Annals of Statistics, Vol. 35, No. 3.

    """
    if weights is None:
        weights = np.ones_like(covariance_matrix)
    if method_smoothing == "LP":
        # Remove covariance diagnonal because of measurements errors.
        kernel_name = kwargs.pop("kernel_name", "epanechnikov")
        bandwidth = kwargs.pop("bandwidth", np.prod(argvals.n_points) ** (-1 / 5))
        degree = kwargs.get("degree", 2)

        if remove_diagonal:
            np.fill_diagonal(covariance_matrix, np.nan)
        covariance_fd = DenseFunctionalData(
            argvals, DenseValues(covariance_matrix[np.newaxis])
        )
        fdata_long = covariance_fd.to_long()
        fdata_long = fdata_long.dropna()

        x = fdata_long.drop(["id", "values"], axis=1, inplace=False).values
        y = fdata_long["values"].values
        points_mat = _cartesian_product(*points.values())

        lp = LocalPolynomial(
            kernel_name=kernel_name,
            bandwidth=bandwidth,
            degree=degree,
        )
        covariance = lp.predict(y=y, x=x, x_new=points_mat)
        covariance = covariance.reshape(points.n_points)
    elif method_smoothing == "PS":
        # Remove covariance diagnonal because of measurements errors.
        n_segments = kwargs.pop("n_segments", 30)
        degree = kwargs.pop("degree", 3)
        order_penalty = kwargs.pop("order_penalty", 2)
        order_derivative = kwargs.pop("order_derivative", 0)
        penalty = kwargs.pop("penalty", (1, 1))

        if remove_diagonal:
            np.fill_diagonal(covariance_matrix, 0)
        ps = PSplines(
            n_segments=n_segments,
            degree=degree,
            order_penalty=order_penalty,
            order_derivative=order_derivative,
        )

        x = argvals["input_dim_0"]
        y = argvals["input_dim_1"]

        ps.fit(x=[x, y], y=covariance_matrix, penalty=penalty, sample_weights=weights)
        covariance = ps.predict([points["input_dim_0"], points["input_dim_1"]])
    else:
        raise ValueError("Method not implemented.")
    return covariance


def _estimate_noise_variance_with_covariance(
    raw_diagonal: npt.NDArray[np.float64],
    smooth_diagonal: npt.NDArray[np.float64],
    argvals: DenseArgvals,
    points: DenseArgvals,
):
    """Estimate the variance of the noise using the covariance diagonal.

    Parameters
    ----------
    raw_diagonal
        The raw covariance diagonal.
    smooth_diagonal
        The smooth covariance diagonal.
    argvals
        The sampling points at which the raw covariance is estimated.
    points
        The sampling points at which the smoothed covariance will be estimated.

    Returns
    -------
    float
        An estimation of the variance of the noise.

    References
    ----------
    .. [1] Yao, F., Müller, H.-G., Wang, J.-L. (2005). Functional Data
        Analysis for Sparse Longitudinal Data. Journal of the American
        Statistical Association 100, pp. 577--590.

    """
    lp = LocalPolynomial(
        kernel_name="epanechnikov", bandwidth=len(raw_diagonal) ** (-1 / 5), degree=1
    )
    var_hat = lp.predict(
        y=raw_diagonal,
        x=_cartesian_product(*argvals.values()),
        x_new=_cartesian_product(*points.values()),
    )
    lower = [int(np.round(0.25 * el)) for el in points.n_points]
    upper = [int(np.round(0.75 * el)) for el in points.n_points]
    bounds = slice(*tuple(lower + upper))
    temp = _integrate(
        (var_hat - smooth_diagonal)[bounds],
        points["input_dim_0"][bounds],
        method="trapz",
    )

    return np.maximum(2 * temp / points.range()["input_dim_0"], 0)


###############################################################################
# Class FunctionalData
class FunctionalData(ABC):
    """Define the structure of FunctionalData.

    Attributes
    ----------
    n_obs: int
        Number of observations of the functional data.
    n_dimension: int
        Number of input dimension of the functional data.
    n_points: Tuple[int, ...] | Dict[int, Tuple[int, ...]]
        Number of sampling points.

    """

    ###########################################################################
    # Checkers
    @staticmethod
    def _check_same_type(*fdata: FunctionalData) -> None:
        """Raise an error if elements in `fdata` have different type.

        Raises
        ------
        TypeError
            When all `fdata` do not have the same type.

        """
        types = set(type(obj) for obj in fdata)
        if len(types) > 1:
            raise TypeError("Elements do not have the same types.")

    @staticmethod
    def _check_same_nobs(*fdata: Type[FunctionalData]) -> None:
        """Raise an arror if elements in `fdata` have different number of obs.

        Raises
        ------
        ValueError
            When all `fdata` do not have the same number of observations.

        """
        n_obs = set(obj.n_obs for obj in fdata)
        if len(n_obs) > 1:
            raise ValueError("Elements do not have the same number of observations.")

    @staticmethod
    def _check_same_ndim(*fdata: Type[FunctionalData]) -> None:
        """Raise an error if elements in `fdata` have different dimension.

        Raises
        ------
        ValueError
            When all `fdata` do not have the same dimension.

        """
        n_dim = set(obj.n_dimension for obj in fdata)
        if len(n_dim) > 1:
            raise ValueError("Elements do not have the same dimensions.")

    @staticmethod
    @abstractmethod
    def _is_compatible(*fdata: Type[FunctionalData]) -> None:
        """Raise an error if elements in `fdata` are not compatible.

        Parameters
        ----------
        fdata
            Functional data to compare.

        Raises
        ------
        ValueError
            When all `fdata` do not have the same number of observations or
            when all `fdata` do not have the same dimension.
        TypeError
            When all `fdata` do not have the same type.

        """
        FunctionalData._check_same_type(*fdata)
        FunctionalData._check_same_nobs(*fdata)
        FunctionalData._check_same_ndim(*fdata)

    ###########################################################################

    ###########################################################################
    # Static methods
    @staticmethod
    @abstractmethod
    def _perform_computation(
        fdata1: Type[FunctionalData], fdata2: Type[FunctionalData], func: Callable
    ) -> Type[FunctionalData]:
        """Perform computation."""

    @staticmethod
    @abstractmethod
    def _perform_computation_number(
        fdata: Type[FunctionalData], number: float, func: Callable
    ) -> Type[FunctionalData]:
        """Perform computation with numbers."""

    @staticmethod
    @abstractmethod
    def concatenate(*fdata: Type[FunctionalData]) -> Type[FunctionalData]:
        """Concatenate FunctionalData objects.

        Parameters
        ----------
        fdata
            Functional data to concatenate.

        Raises
        ------
        ValueError
            When all `fdata` do not have the same dimension.

        TypeError
            When all `fdata` do not have the same type.

        """
        FunctionalData._check_same_type(*fdata)
        FunctionalData._check_same_ndim(*fdata)

    ###########################################################################

    ###########################################################################
    # Magic methods
    def __repr__(self) -> str:
        """Override print function."""
        return (
            f"Functional data object with {self.n_obs} observations on a "
            f"{self.n_dimension}-dimensional support."
        )

    @abstractmethod
    def __iter__(self):
        """Initialize the iterator."""

    @abstractmethod
    def __getitem__(self, index: int) -> Type[FunctionalData]:
        """Override getitem function, called when self[index]."""

    ###########################################################################

    ###########################################################################
    # Properties
    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Get the number of observations of the functional data."""

    @property
    @abstractmethod
    def n_dimension(self) -> int:
        """Get the number of input dimension of the functional data."""

    @property
    @abstractmethod
    def n_points(self) -> Tuple[int, ...] | Dict[int, Tuple[int, ...]]:
        """Get the number of sampling points."""

    ###########################################################################

    ###########################################################################
    # Abstract methods
    @abstractmethod
    def to_long(self, reindex: bool = False) -> pd.DataFrame:
        """Convert the data to long format."""

    @abstractmethod
    def noise_variance(self, order: int = 2) -> float:
        """Estimate the variance of the noise."""

    @abstractmethod
    def smooth(
        self,
        points: DenseArgvals | None = None,
        method: str = "PS",
        bandwidth: float | None = None,
        penalty: float | None = None,
        **kwargs,
    ) -> Type[FunctionalData]:
        """Smooth the data."""

    @abstractmethod
    def mean(
        self,
        points: DenseArgvals | None = None,
        method_smoothing: str = None,
        **kwargs,
    ) -> FunctionalData:
        """Compute an estimate of the mean."""

    @abstractmethod
    def center(
        self,
        mean: DenseFunctionalData | None = None,
        method_smoothing: str | None = None,
        **kwargs,
    ) -> FunctionalData:
        """Center the data."""

    @abstractmethod
    def norm(
        self,
        squared: bool = False,
        method_integration: str = "trapz",
        use_argvals_stand: bool = False,
    ) -> npt.NDArray[np.float64]:
        r"""Norm of each observation of the data.

        For each observation in the data, it computes its norm defined
        as

        .. math::
            \| X \| = \left\{\int_{\mathcal{T}} X(t)^2dt\right\}^{\frac12}.

        """

    @abstractmethod
    def normalize(self, **kwargs) -> FunctionalData:
        """Normalize the data."""

    @abstractmethod
    def standardize(self, center: bool = True, **kwargs) -> FunctionalData:
        """Standardize the data."""

    @abstractmethod
    def rescale(
        self,
        weights: float = 0.0,
        method_integration: str = "trapz",
        use_argvals_stand: bool = False,
        **kwargs,
    ) -> Tuple[FunctionalData, float]:
        """Rescale the data."""

    @abstractmethod
    def inner_product(
        self,
        method_integration: str = "trapz",
        method_smoothing: str | None = None,
        noise_variance: float | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """Compute an estimate of the inner product matrix."""

    @abstractmethod
    def covariance(
        self,
        points: DenseArgvals | None = None,
        method_smoothing: str | None = None,
        **kwargs,
    ) -> Type[FunctionalData]:
        """Compute an estimate of the covariance."""

    ###########################################################################


###############################################################################
# Class GridFunctionalData
class GridFunctionalData(FunctionalData):
    """Represent discretised functional data.

    Parameters
    ----------
    argvals: Type[Argvals]
        Sampling points of the functional data.
    values: Type[Values]
        Values of the functional data.

    Attributes
    ----------
    argvals_stand: Type[Argvals]
        Standardized sampling points of the functional data.
    n_obs: int
        Number of observations of the functional data.
    n_dimension: int
        Number of input dimension of the functional data.
    n_points: Tuple[int, ...] | Dict[int, Tuple[int, ...]]
        Number of sampling points.

    """

    ###########################################################################
    # Checkers
    @staticmethod
    def _is_compatible(*fdata: Type[FunctionalData]) -> None:
        """Raise an error if elements in `fdata` are not compatible.

        Parameters
        ----------
        fdata
            Functional data to compare.

        Raises
        ------
        ValueError
            When all `fdata` do not have the same argvals.

        """
        FunctionalData._is_compatible(*fdata)
        if not all(data.argvals == fdata[0].argvals for data in fdata):
            raise ValueError("Argvals are not equals.")

    ###########################################################################

    ###########################################################################
    # Magic methods
    def __init__(
        self,
        argvals: Type[Argvals],
        values: Type[Values],
    ) -> None:
        """Initialize GridFunctionalData object."""
        self.argvals = argvals
        self.values = values
        self._index = 0

    def __eq__(self, obj: GridFunctionalData) -> bool:
        """Override eq function."""
        if not isinstance(obj, GridFunctionalData):
            raise TypeError("Object does not have the right type.")
        return (self.argvals == obj.argvals) & np.allclose(self.values, obj.values)

    def __add__(self, obj: Type[FunctionalData] | float | int) -> Type[FunctionalData]:
        """Override add function."""
        if isinstance(obj, FunctionalData):
            return self._perform_computation(self, obj, np.add)
        elif isinstance(obj, (float, int)):
            return self._perform_computation_number(self, obj, np.add)
        else:
            raise TypeError("Operations not available for this type.")

    def __sub__(self, obj: Type[FunctionalData] | float | int) -> Type[FunctionalData]:
        """Override sub function."""
        if isinstance(obj, FunctionalData):
            return self._perform_computation(self, obj, np.subtract)
        elif isinstance(obj, (float, int)):
            return self._perform_computation_number(self, obj, np.subtract)
        else:
            raise TypeError("Operations not available for this type.")

    def __mul__(self, obj: Type[FunctionalData] | float | int) -> Type[FunctionalData]:
        """Override mul function."""
        if isinstance(obj, FunctionalData):
            return self._perform_computation(self, obj, np.multiply)
        elif isinstance(obj, (float, int)):
            return self._perform_computation_number(self, obj, np.multiply)
        else:
            raise TypeError("Operations not available for this type.")

    def __rmul__(self, obj: Type[FunctionalData] | float | int) -> Type[FunctionalData]:
        """Override rmul function."""
        return self * obj

    def __truediv__(
        self, obj: Type[FunctionalData] | float | int
    ) -> Type[FunctionalData]:
        """Override truediv function."""
        if isinstance(obj, FunctionalData):
            return self._perform_computation(self, obj, np.true_divide)
        elif isinstance(obj, (float, int)):
            return self._perform_computation_number(self, obj, np.true_divide)
        else:
            raise TypeError("Operations not available for this type.")

    def __floordiv__(
        self, obj: Type[FunctionalData] | float | int
    ) -> Type[FunctionalData]:
        """Override floordiv function."""
        if isinstance(obj, FunctionalData):
            return self._perform_computation(self, obj, np.floor_divide)
        elif isinstance(obj, (float, int)):
            return self._perform_computation_number(self, obj, np.floor_divide)
        else:
            raise TypeError("Operations not available for this type.")

    ###########################################################################

    ###########################################################################
    # Properties
    @property
    def argvals(self) -> Type[Argvals]:
        """Getter for argvals."""
        return self._argvals

    @argvals.setter
    @abstractmethod
    def argvals(self, new_argvals: Type[Argvals]) -> None:
        """Setter for argvals."""

    @property
    def argvals_stand(self) -> Type[Argvals]:
        """Getter for argvals_stand."""
        return self._argvals_stand

    @argvals_stand.setter
    def argvals_stand(self, new_argvals_stand: Type[Argvals]) -> None:
        """Setter for argvals_stand."""
        if not isinstance(new_argvals_stand, Argvals):
            raise TypeError("new_argvals_stand must be an Argvals object.")
        self._argvals_stand = new_argvals_stand

    @property
    def values(self) -> Type[Values]:
        """Getter for values."""
        return self._values

    @values.setter
    @abstractmethod
    def values(self, new_values: Type[Values]) -> None:
        """Setter for values."""

    @property
    def n_obs(self) -> int:
        """Get the number of observations of the functional data.

        Returns
        -------
        int
            Number of observations within the functional data.

        """
        return self.values.n_obs

    @property
    def n_dimension(self) -> int:
        """Get the number of input dimension of the functional data.

        Returns
        -------
        int
            Number of input dimension with the functional data.

        """
        return self.argvals.n_dimension

    @property
    def n_points(self) -> Tuple[int, ...] | Dict[int, Tuple[int, ...]]:
        """Get the number of sampling points.

        Returns
        -------
        Tuple[int, ...] | Dict[int, Tuple[int, ...]]
            Number of sampling points.

        """
        return self.argvals.n_points

    ###########################################################################

    ###########################################################################
    # Abstract methods
    @abstractmethod
    def to_basis(
        self, points: DenseArgvals | None = None, method: str = "PS", **kwargs
    ) -> BasisFunctionalData:
        """Convert the data to basis format."""

    @abstractmethod
    def to_long(self, reindex: bool = False) -> pd.DataFrame:
        """Convert the data to long format."""

    @abstractmethod
    def noise_variance(self, order: int = 2) -> float:
        """Estimate the variance of the noise."""

    @abstractmethod
    def smooth(
        self,
        points: DenseArgvals | None = None,
        method: str = "PS",
        bandwidth: float | None = None,
        penalty: float | None = None,
        **kwargs,
    ) -> Type[FunctionalData]:
        """Smooth the data."""

    @abstractmethod
    def mean(
        self,
        points: DenseArgvals | None = None,
        method_smoothing: str = None,
        **kwargs,
    ) -> FunctionalData:
        """Compute an estimate of the mean."""

    @abstractmethod
    def center(
        self,
        mean: DenseFunctionalData | None = None,
        method_smoothing: str | None = None,
        **kwargs,
    ) -> FunctionalData:
        """Center the data."""

    @abstractmethod
    def norm(
        self,
        squared: bool = False,
        method_integration: str = "trapz",
        use_argvals_stand: bool = False,
    ) -> npt.NDArray[np.float64]:
        r"""Norm of each observation of the data.

        For each observation in the data, it computes its norm defined
        as

        .. math::
            \| X \| = \left\{\int_{\mathcal{T}} X(t)^2dt\right\}^{\frac12}.

        """

    @abstractmethod
    def normalize(self, **kwargs) -> FunctionalData:
        """Normalize the data."""

    @abstractmethod
    def standardize(self, center: bool = True, **kwargs) -> FunctionalData:
        """Standardize the data."""

    @abstractmethod
    def rescale(
        self,
        weights: float = 0.0,
        method_integration: str = "trapz",
        use_argvals_stand: bool = False,
        **kwargs,
    ) -> Tuple[FunctionalData, float]:
        """Rescale the data."""

    @abstractmethod
    def inner_product(
        self,
        method_integration: str = "trapz",
        method_smoothing: str | None = None,
        noise_variance: float | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """Compute an estimate of the inner product matrix."""

    @abstractmethod
    def covariance(
        self,
        points: DenseArgvals | None = None,
        method_smoothing: str | None = None,
        **kwargs,
    ) -> Type[FunctionalData]:
        """Compute an estimate of the covariance."""

    ###########################################################################


###############################################################################
# Class DenseFunctionalDataIterator
class DenseFunctionalDataIterator(Iterator):
    """Iterator for dense functional data."""

    def __init__(self, fdata):
        """Initialize the Iterator object."""
        self._fdata = fdata
        self._index = 0

    def __next__(self):
        """Return the next item in the sequence."""
        if self._index < self._fdata.n_obs:
            item = self._fdata[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration


###############################################################################
# Class DenseFunctionalData
class DenseFunctionalData(GridFunctionalData):
    r"""Represent densely sampled functional data.

    A class used to define dense functional data. We denote by :math:`n`, the
    number of observations and by :math:`p`, the number of input dimensions.
    Here, we are in the case of univariate functional data, and so the output
    dimension will be :math:`\mathbb{R}`. We note by :math:`X` an observation,
    while we use :math:`X_1, \dots, X_n` if we refer to a particular set of
    observations. The observations are defined as:

    .. math::
        X(t): \mathcal{T} \longrightarrow \mathbb{R},

    where :math:`\mathcal{T} \subset \mathbb{R}^p`. We denote the mean function
    by

    .. math::
        \mu(t): \mathcal{T} \longrightarrow \mathbb{R},

    and the covariance function by:

    .. math::
        C(s, t): \mathcal{T} \times \mathcal{T} \longrightarrow \mathbb{R}.

    We also note :math:`\mathbf{M}` the Gram matrix of the set of observations.

    Parameters
    ----------
    argvals: DenseArgvals
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. The shape of the :math:`j` th
        dimension is :math:`(m_j,)` for :math:`0 \leq j \leq p`.
    values: DenseValues
        The values of the functional data. The shape of the array is
        :math:`(n, m_1, \dots, m_p)`.

    Attributes
    ----------
    argvals_stand: DenseArgvals
        Standardized sampling points of the functional data.
    n_obs: int
        Number of observations of the functional data.
    n_dimension: int
        Number of input dimension of the functional data.
    n_points: Tuple[int, ...]
        Number of sampling points.

    Examples
    --------
    For 1-dimensional dense data:

    >>> argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 5])})
    >>> values = DenseValues(np.array([
    ...     [1, 2, 3, 4, 5],
    ...     [6, 7, 8, 9, 10],
    ...     [11, 12, 13, 14, 15]
    ... ]))
    >>> DenseFunctionalData(argvals, values)

    For 2-dimensional dense data:

    >>> argvals = DenseArgvals({
    ...     'input_dim_0': np.array([1, 2, 3, 4]),
    ...     'input_dim_1': np.array([5, 6, 7])
    ... })
    >>> values = DenseValues(np.array([
    ...     [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
    ...     [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]],
    ...     [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]
    ... ]))
    >>> DenseFunctionalData(argvals, values)

    References
    ----------
    .. [1] Benko, M., Härdle, W. and Kneip, A. (2009). Common functional
        principal components. The Annals of Statistics 37, 1--34.
    .. [2] Cai, T.T., Yuan, M., (2011), Optimal estimation of the mean
        function based on discretely sampled functional data: Phase
        transition. The Annals of Statistics 39, 2330-2355.
    .. [3] Chiou, J.-M., Chen, Y.-T., Yang, Y.-F. (2014). Multivariate
        Functional Principal Component Analysis: A Normalization Approach.
        Statistica Sinica 24, 1571--1596.
    .. [4] Eilers, P. H. C., Marx, B. D. (2021). Practical Smoothing: The Joys
        of P-splines. Cambridge University Press, Cambridge.
    .. [5] Hall, P., Kay, J.W. and Titterington, D.M. (1990).
        Asymptotically Optimal Difference-Based Estimation of Variance in
        Nonparametric Regression. Biometrika 77, 521--528.
    .. [6] Happ, C., Greven, S. (2018). Multivariate Functional Principal
        Component Analysis for Data Observed on Different (Dimensional)
        Domains. Journal of the American Statistical Association 113, 649--659.
    .. [7] Ramsay, J. O. and Silverman, B. W. (2005), Functional Data
        Analysis, Springer Science, Chapter 8.
    .. [8] Tsybakov, A.B. (2008), Introduction to Nonparametric Estimation.
        Springer Series in Statistics.

    """

    ###########################################################################
    # Checkers

    ###########################################################################

    ###########################################################################
    # Static methods
    @staticmethod
    def _perform_computation(
        fdata1: DenseFunctionalData, fdata2: DenseFunctionalData, func: Callable
    ) -> DenseFunctionalData:
        """Perform computation defined by `func` if they are compatible.

        Parameters
        ----------
        fdata1
            First functional data to consider.
        fdata2
            Second functional data to consider.
        func
            The function to apply to combine `fdata1` and `fdata2`.

        Returns
        -------
        DenseFunctionalData
            The resulting functional data.

        """
        DenseFunctionalData._is_compatible(fdata1, fdata2)
        new_values = func(fdata1.values, fdata2.values)
        return DenseFunctionalData(fdata1.argvals, new_values)

    @staticmethod
    def _perform_computation_number(
        fdata: DenseFunctionalData, number: int | float, func: Callable
    ) -> DenseFunctionalData:
        """Perform computation with numbers.

        Parameters
        ----------
        fdata
            Functional data to consider.
        number
            number to consider.
        func
            The function to apply to combine `fdata` and `number`.

        Returns
        -------
        DenseFunctionalData
            The resulting functional data.

        """
        new_values = func(fdata.values, number)
        return DenseFunctionalData(fdata.argvals, DenseValues(new_values))

    @staticmethod
    def concatenate(*fdata: DenseFunctionalData) -> DenseFunctionalData:
        """Concatenate DenseFunctional objects.

        Parameters
        ----------
        fdata
            Functional data to concatenate.

        Returns
        -------
        DenseFunctionalData
            The concatenated object.

        """
        super(DenseFunctionalData, DenseFunctionalData).concatenate(*fdata)
        argvals = DenseArgvals.concatenate(*[el.argvals for el in fdata])
        values = DenseValues.concatenate(*[el.values for el in fdata])
        return DenseFunctionalData(argvals, values)

    ###########################################################################

    ###########################################################################
    # Magic methods
    def __init__(self, argvals: DenseArgvals, values: DenseValues) -> None:
        """Initialize UnivariateFunctionalData object."""
        super().__init__(argvals, values)

    def __iter__(self):
        """Initialize the iterator."""
        return DenseFunctionalDataIterator(self)

    def __getitem__(self, index: int) -> DenseFunctionalData:
        """Overrride getitem function, called when self[index].

        Parameters
        ----------
        index
            The observation(s) of the object to retrive.

        Returns
        -------
        DenseFunctionalData
            The selected observation(s) as DenseFunctionalData object.

        """
        argvals = self.argvals
        values = self.values[index]

        if len(argvals) == len(values.shape):
            values = values[np.newaxis]
        return DenseFunctionalData(argvals, values)

    ###########################################################################

    ###########################################################################
    # Properties
    @GridFunctionalData.argvals.setter
    def argvals(self, new_argvals: DenseArgvals) -> None:
        """Setter for argvals."""
        if not isinstance(new_argvals, DenseArgvals):
            raise TypeError("new_argvals must be a DenseArgvals object.")
        if hasattr(self, "values"):
            self._values.compatible_with(new_argvals)
        self._argvals = new_argvals
        self._argvals_stand = self._argvals.normalization()

    @GridFunctionalData.values.setter
    def values(self, new_values: DenseValues) -> None:
        """Setter for values."""
        if not isinstance(new_values, DenseValues):
            raise TypeError("new_values must be a DenseValues object.")
        if hasattr(self, "argvals"):
            self._argvals.compatible_with(new_values)
        self._values = new_values

    ###########################################################################

    ###########################################################################
    # Methods
    def to_basis(
        self,
        points: DenseArgvals | None = None,
        method: str = "PS",
        penalty: float | None = None,
        **kwargs,
    ) -> BasisFunctionalData:
        """Convert the data to basis format.

        This function transform a DenseFunctionalData object into a
        BasisFunctionalData object using `method`.

        Parameters
        ----------
        points
            The argvals of the basis.
        method
            The method to get the coefficients.
        penalty
            Strictly positive. Penalty used in the P-splined fitting of the
            data.
        kwargs
            Other keyword arguments are passed to the function:
            :meth:`preprocessing.smoothing.PSplines`

        Returns
        -------
        BasisFunctionalData
            The expanded data.

        """
        from .basis import Basis

        if method == "PS":
            if penalty is None:
                penalty = self.n_dimension * [1]
            ps = PSplines(**kwargs)
            n_functions = np.power(ps.n_segments + ps.degree, self.n_dimension)

            x = list(self.argvals.values())
            coefs = np.zeros((self.n_obs, n_functions))
            for idx, _ in enumerate(self):
                ps.fit(x=x, y=self.values[idx, :], penalty=penalty)
                coefs[idx, :] = ps.beta_hat.flatten()

            values = DenseValues(
                reduce(np.kron, ps.basis).reshape((n_functions, *self.argvals.n_points))
            )
            basis = Basis(name="given", argvals=self.argvals, values=values)
        else:
            raise ValueError("Method not implemented.")
        return BasisFunctionalData(basis=basis, coefficients=coefs)

    def to_long(self, reindex: bool = False) -> pd.DataFrame:
        """Convert the data to long format.

        This function transform a DenseFunctionalData object into pandas
        DataFrame. It uses the long format to represent the DenseFunctionalData
        object as a dataframe. This is a helper function as it might be easier
        for some computation, e.g., smoothing of the mean and covariance
        functions to have a long format.

        Parameters
        ----------
        reindex
            Not used here.

        Returns
        -------
        pd.DataFrame
            The data in a long format.

        Examples
        --------
        >>> argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 5])})
        >>> values = DenseValues(np.array([
        ...     [1, 2, 3, 4, 5],
        ...     [6, 7, 8, 9, 10],
        ...     [11, 12, 13, 14, 15]
        ... ]))
        >>> fdata = DenseFunctionalData(argvals, values)

        >>> fdata.to_long()
            input_dim_0  id  values
        0             1   0       1
        1             2   0       2
        2             3   0       3
        3             4   0       4
        4             5   0       5
        5             1   1       6
        6             2   1       7
        7             3   1       8
        8             4   1       9
        9             5   1      10
        10            1   2      11
        11            2   2      12
        12            3   2      13
        13            4   2      14
        14            5   2      15

        """
        sampling_points = list(itertools.product(*self.argvals.values()))
        temp = pd.DataFrame(self.n_obs * sampling_points)
        temp.columns = list(self.argvals.keys())
        temp["id"] = np.repeat(np.arange(self.n_obs), np.prod(self.n_points))
        temp["values"] = self.values.flatten()
        return temp

    def noise_variance(self, order: int = 2) -> float:
        """Estimate the variance of the noise.

        This function estimates the variance of the noise. The noise is
        estimated for each individual curve using the methodology in [5]_. As
        the curves are assumed to be generated by the same process, the
        estimation of the variance of the noise is the mean over the set of
        curves.

        Parameters
        ----------
        order
            Order of the difference sequence. The order has to be between
            1 and 10. See [5]_ for more information.

        Returns
        -------
        float
            The estimation of the variance of the noise.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=100)
        >>> kl.add_noise(0.05)
        >>> kl.noisy_data.noise_variance(order=2)
        0.051922438333740877

        """
        if self.n_dimension > 1:
            warnings.warn(
                (
                    "The estimation of the variance of the noise is not performed "
                    "for data with dimension larger than 1 and is set to 0."
                ),
                UserWarning,
            )
            return 0
        return np.nanmean(
            [_estimate_noise_variance(obs.values[0], order) for obs in self]
        )

    def smooth(
        self,
        points: DenseArgvals | None = None,
        method: str = "PS",
        bandwidth: float | None = None,
        penalty: float | None = None,
        **kwargs,
    ) -> DenseFunctionalData:
        """Smooth the data.

        This function smooths each curves individually. Based on [2]_, it fits
        a local polynomial smoother to the data. Based on [4]_, it fits
        P-splines to the data.

        Parameters
        ----------
        points
            Points at which the curves are estimated. The default is None,
            meaning we use the argvals as estimation points.
        method
            The method to used for the smoothing. If 'PS', the method is
            P-splines [4]_. If 'LP', the method is local polynomials [2]_.
            Otherwise, it raises an error.
        bandwidth
            Strictly positive. Control the size of the associated neighborhood.
            If ``bandwidth=None``, it is assumed that the curves are twice
            differentiable and the bandwidth is set to :math:`n^{-1/5}` [8]_
            where :math:`n` is the number of sampling points per curve. Be
            careful with the results if the curves are not sampled on
            :math:`[0, 1]`.
        penalty
            Strictly positive. Penalty used in the P-splined fitting of the
            data.
        kwargs
            Other keyword arguments are passed to one of the following
            functions :meth:`preprocessing.smoothing.PSplines` (``method='PS'``) and
            :meth:`preprocessing.smoothing.LocalPolynomial` (``method='LP'``).

        Returns
        -------
        DenseFunctionalData
            Smoothed data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=1)
        >>> kl.add_noise(0.05)
        >>> kl.noisy_data.smooth()
        Functional data object with 1 observations on a 1-dimensional support.

        """
        if points is None:
            points = self.argvals

        if method == "LP":
            if bandwidth is None:
                bandwidth = np.prod(self.n_points) ** (-1 / 5)

            argvals_mat = _cartesian_product(*self.argvals.values())
            points_mat = _cartesian_product(*points.values())

            lp = LocalPolynomial(bandwidth=bandwidth, **kwargs)

            smooth = np.zeros((self.n_obs, *points.n_points))
            for idx, obs in enumerate(self):
                smooth[idx, :] = lp.predict(
                    y=obs.values.flatten(), x=argvals_mat, x_new=points_mat
                ).reshape(smooth.shape[1:])
        elif method == "PS":
            if penalty is None:
                penalty = self.n_dimension * [1]

            ps = PSplines(**kwargs)

            x = list(self.argvals.values())
            new_x = list(points.values())

            smooth = np.zeros((self.n_obs, *points.n_points))
            for idx, _ in enumerate(self):
                ps.fit(x=x, y=self.values[idx, :], penalty=penalty)
                smooth[idx, :] = ps.predict(x=new_x)
        else:
            raise NotImplementedError("Method not implemented.")
        return DenseFunctionalData(points, DenseValues(smooth))

    def mean(
        self,
        points: DenseArgvals | None = None,
        method_smoothing: str | None = None,
        **kwargs,
    ) -> DenseFunctionalData:
        """Compute an estimate of the mean.

        This function computes an estimate of the mean curve of a
        DenseFunctionalData object. As the curves are sampled on a common grid,
        we consider the sample mean, as defined in [7]_. The sampled mean is
        rate optimal [2]_. We included some smoothing using Local Polynonial
        Estimators [8]_ or P-Splines [4]_.

        Parameters
        ----------
        points
            The sampling points at which the mean is estimated. If `None`, the
            DenseArgvals of the DenseFunctionalData is used.
        method_smoothing
            The method to used for the smoothing. If 'None', no smoothing is
            performed. If 'PS', the method is P-splines [4]_. If 'LP', the
            method is local polynomials [8]_.
        kwargs
            Other keyword arguments are passed to the following function
            :meth:`DenseFunctionalData.smooth`.

        Returns
        -------
        DenseFunctionalData
            An estimate of the mean as a DenseFunctionalData object.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=100)
        >>> kl.add_noise(0.01)
        >>> kl.noisy_data.mean(smooth=True)
        Functional data object with 1 observations on a 1-dimensional support.

        """
        # Set parameters
        if points is None:
            points = self.argvals

        mean_estim = self.values.mean(axis=0)
        self._mean = DenseFunctionalData(self.argvals, mean_estim[np.newaxis])

        if method_smoothing:
            self._mean = self._mean.smooth(
                points=points, method=method_smoothing, **kwargs
            )
        return self._mean

    def center(
        self,
        mean: DenseFunctionalData | None = None,
        method_smoothing: str | None = None,
        **kwargs,
    ) -> DenseFunctionalData:
        r"""Center the data.

        The centering is done by estimating the mean from the data and then
        substracting it to the data. It results in

        .. math::
            \widetilde{X}(t) = X(t) - \mu(t).

        Parameters
        ----------
        mean
            A precomputed mean as a DenseFunctionalData object.
        method_smoothing
            The method to used for the smoothing of the mean. If 'None', no
            smoothing is performed. If 'PS', the method is P-splines [4]_. If
            'LP', the method is local polynomials [2]_.
        kwargs
            Other keyword arguments are passed to one of the following
            functions: :meth:`DenseFunctionalData.mean` (``mean=None``) and
            :meth:`DenseFunctionalData.smooth`.

        Returns
        -------
        DenseFunctionalData
            The centered version of the data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.data.center(smooth=True)
        Functional data object with 10 observations on a 1-dimensional support.

        """
        if mean is None:
            data_mean = self.mean(method_smoothing=method_smoothing, **kwargs)
        elif (mean is not None) and (method_smoothing is not None):
            data_mean = mean.smooth(self.argvals, method=method_smoothing, **kwargs)
        else:
            data_mean = mean
        return DenseFunctionalData(
            DenseArgvals(self.argvals), DenseValues(self.values - data_mean.values)
        )

    def norm(
        self,
        squared: bool = False,
        method_integration: str = "trapz",
        use_argvals_stand: bool = False,
    ) -> npt.NDArray[np.float64]:
        r"""Norm of each observation of the data.

        For each observation in the data, it computes its norm defined in [6]_
        as

        .. math::
            \| X \| = \left\{\int_{\mathcal{T}} X(t)^2dt\right\}^{\frac12}.

        Parameters
        ----------
        squared
            If ``True``, the function calculates the squared norm, otherwise it
            returns the norm.
        method_integration
            The method used to estimate the integral.
        use_argvals_stand
            Use standardized argvals to compute the normalization of the data.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs,)
            The norm of each observations.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.data.norm()
        array([
            0.53253351, 0.42212112, 0.6709846 , 0.26672898, 0.27440755,
            0.37906252, 0.65277413, 0.53998411, 0.2872874 , 0.4934973
        ])

        """
        # Get parameters
        n_obs = self.n_obs
        if use_argvals_stand:
            axis = [argvals for argvals in self.argvals_stand.values()]
        else:
            axis = [argvals for argvals in self.argvals.values()]

        sq_values = np.power(self.values, 2)

        norm_fd = np.zeros(n_obs)
        for idx in np.arange(n_obs):
            norm_fd[idx] = _integrate(sq_values[idx], *axis, method=method_integration)

        if squared:
            return np.array(norm_fd)
        else:
            return np.power(norm_fd, 0.5)

    def normalize(
        self,
        **kwargs,
    ) -> DenseFunctionalData:
        r"""Normalize the data.

        The normalization is performed by divising each functional datum
        :math:`X` by its norm :math:`\| X \|`. It results in

        .. math::
            \widetilde{X} = \frac{X}{\| X \|}.

        Parameters
        ----------
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`DenseFunctionalData.norm`.

        Returns
        -------
        DenseFunctionalData
            The normalized data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.data.normalize()
        Functional data object with 10 observations on a 1-dimensional support.

        """
        norm = np.moveaxis(self.values, 0, -1) / self.norm(**kwargs)
        fdata_new = DenseFunctionalData(self.argvals, np.moveaxis(norm, -1, 0))
        return fdata_new

    def standardize(self, center: bool = True, **kwargs) -> DenseFunctionalData:
        r"""Standardize the data.

        The standardization is performed by first centering the data and then
        dividing by the standard deviation curve [3]_. It results in

        .. math::
            \widetilde{X}(t) = C(t, t)^{-\frac12}\{X(t) - \mu(t)\}, \quad
            t \in \mathcal{T}.

        Parameters
        ----------
        center
            Should the data be centered?
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`DenseFunctionalData.center`.

        Returns
        -------
        DenseFunctionalData
            The standardized data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.data.standardize()
        Functional data object with 10 observations on a 1-dimensional support.

        """
        if center:
            fdata = self.center(**kwargs)
        else:
            fdata = self
        std = np.std(self.values, axis=0)
        new_values = np.divide(fdata.values, std, where=(std != 0))
        return DenseFunctionalData(self.argvals, new_values)

    def rescale(
        self,
        weights: float = 0.0,
        method_integration: str = "trapz",
        use_argvals_stand: bool = False,
        **kwargs,
    ) -> Tuple[DenseFunctionalData, float]:
        r"""Rescale the data.

        The rescaling is performed by first centering the data and then
        multiplying with a common weight:

        .. math::
            \widetilde{X}(t) = w\{X(t) - \mu(t)\}.

        The weights are defined in [6]_.

        Parameters
        ----------
        weights
            The weights used to normalize the data. If `weights = 0.0`, the
            weights are estimated by integrating the variance function [3]_.
        method_integration
            The method used to estimate the integral.
        use_argvals_stand
            Use standardized argvals to compute the normalization of the data.

        Returns
        -------
        Tuple[DenseFunctionalData, float]
            The rescaled data and the weight.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.data.rescale()
        Functional data object with 10 observations on a 1-dimensional support.

        """
        if weights == 0.0:
            if use_argvals_stand:
                axis = [argvals for argvals in self.argvals_stand.values()]
            else:
                axis = [argvals for argvals in self.argvals.values()]
            variance = np.var(self.values, axis=0)
            weights = _integrate(variance, *axis, method=method_integration)
        new_data = self / np.sqrt(float(weights))
        return new_data, weights

    def inner_product(
        self,
        method_integration: str = "trapz",
        method_smoothing: str | None = None,
        noise_variance: float | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        r"""Compute the inner product matrix of the data.

        The inner product matrix is a ``n_obs`` by ``n_obs`` matrix where each
        entry is defined as

        .. math::
            \langle x, y \rangle = \int_{\mathcal{T}} x(t)y(t)dt,
            t \in \mathcal{T},

        where :math:`\mathcal{T}` is a one- or multi-dimensional domain [1]_.

        Parameters
        ----------
        method_integration
            The method used to integrated.
        method_smoothing
            The method to used for the smoothing of the mean. If 'None', no
            smoothing is performed. If 'PS', the method is P-splines [4]_. If
            'LP', the method is local polynomials [2]_.
        noise_variance
            An estimation of the variance of the noise. If `None`, an
            estimation is computed using the methodology in [5]_.
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`DenseFunctionalData.center`.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_obs)
            Inner product matrix of the data.

        Examples
        --------
        For one-dimensional functional data:

        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines', n_functions=5, random_state=42
        ... )
        >>> kl.new(n_obs=3)
        >>> kl.data.inner_product(noise_variance=0)
        array([
            [ 0.16288536,  0.01958865, -0.10017322],
            [ 0.01958865,  0.17701988, -0.2459348 ],
            [-0.10017322, -0.2459348 ,  0.42008035]
        ])

        For two-dimensional functional data:

        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines', dimension='2D', n_functions=5,
        ...     random_state=42, argvals=np.linspace(0, 1, 11)
        ... )
        >>> kl.new(n_obs=3)
        >>> kl.data.inner_product(noise_variance=0)
        array([
            [ 0.01669878,  0.00349892, -0.00817676],
            [ 0.00349892,  0.03208174, -0.03777796],
            [-0.00817676, -0.03777796,  0.05083159]
        ])

        """
        # Center the data
        data = self.center(method_smoothing=method_smoothing, **kwargs)

        # Estimate the noise of the variance
        if noise_variance is None:
            self._noise_variance = self.noise_variance(order=2)
        else:
            self._noise_variance = noise_variance

        # Get parameters
        n_obs = data.n_obs
        axis = [argvals for argvals in data.argvals.values()]

        inner_mat = np.zeros((n_obs, n_obs))
        for i, j in itertools.product(np.arange(n_obs), repeat=2):
            if i <= j:
                inner_mat[i, j] = _inner_product(
                    data.values[i], data.values[j], *axis, method=method_integration
                )
        inner_mat = inner_mat - np.diag(np.repeat(self._noise_variance, n_obs))

        # Estimate the diagonal of the inner-product matrix
        inner_mat = inner_mat + inner_mat.T
        np.fill_diagonal(inner_mat, np.diag(inner_mat) / 2)

        self._data_inpro = data
        self._inner_product_matrix = inner_mat
        return self._inner_product_matrix

    def covariance(
        self,
        points: DenseArgvals | None = None,
        method_smoothing: str | None = None,
        center: bool = True,
        kwargs_center: Dict[str, object] = {},
        **kwargs,
    ) -> DenseFunctionalData:
        r"""Compute an estimate of the covariance function.

        This function computes an estimate of the covariance surface of a
        DenseFunctionalData object. As the curves are sampled on a common grid,
        we consider the sample covariance [7]_.

        Parameters
        ----------
        points
            The sampling points at which the covariance is estimated. If
            `None`, the DenseArgvals of the DenseFunctionalData is used. If
            `smooth` is False, the DenseArgvals of the DenseFunctionalData is
            used.
        method_smoothing
            The method to used for the smoothing of the mean. If 'None', no
            smoothing is performed. If 'PS', the method is P-splines [4]_. If
            'LP', the method is local polynomials [2]_.
        center
            Should the data be centered before computing the covariance.
        kwargs_center
            Keyword arguments to be passed to the function
            :meth:`FunctionalData.center`.
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`functional_data._smooth_covariance`.

        Returns
        -------
        DenseFunctionalData
            An estimate of the covariance as a two-dimensional
            DenseFunctionalData object.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=100)
        >>> kl.add_noise(0.01)
        >>> kl.noisy_data.covariance(smooth=True)
        Functional data object with 1 observations on a 2-dimensional support.

        """
        if self.n_dimension > 1:
            raise ValueError("Only one dimensional functional data are supported.")

        if points is None:
            points = self.argvals
        argvals_cov = DenseArgvals(
            {
                "input_dim_0": self.argvals["input_dim_0"],
                "input_dim_1": self.argvals["input_dim_0"],
            }
        )
        points_cov = DenseArgvals(
            {
                "input_dim_0": points["input_dim_0"],
                "input_dim_1": points["input_dim_0"],
            }
        )

        # Center the data
        data = self
        if center:
            data = data.center(method_smoothing=method_smoothing, **kwargs_center)

        # Estimate the covariance
        cov = np.dot(data.values.T, data.values) / (self.n_obs - 1)
        raw_diag_cov = np.diag(cov).copy()
        if method_smoothing:
            weights = np.ones_like(cov)
            weights[cov == 0] = 0

            cov = _smooth_covariance(
                cov,
                argvals_cov,
                points_cov,
                method_smoothing=method_smoothing,
                weights=weights,
                **kwargs,
            )

        # Ensure the covariance is symmetric.
        cov = (cov + cov.T) / 2

        # Estimate noise variance ([2], [3])
        self._noise_variance_cov = _estimate_noise_variance_with_covariance(
            raw_diag_cov, np.diag(cov), self.argvals, points
        )

        self._covariance = DenseFunctionalData(points_cov, DenseValues(cov[np.newaxis]))
        return self._covariance

    ###########################################################################


###############################################################################
# Class IrregularFunctionalDataIterator
class IrregularFunctionalDataIterator(Iterator):
    """Iterator for irregular functional data."""

    def __init__(self, fdata):
        """Initialize the Iterator object."""
        self._fdata = fdata
        self._index = list(fdata.argvals)

    def __next__(self):
        """Return the next item in the sequence."""
        if len(self._index) > 0:
            idx = self._index.pop(0)
            item = self._fdata[idx]
            return item
        else:
            raise StopIteration


###############################################################################
# Class IrregularFunctionalData
class IrregularFunctionalData(GridFunctionalData):
    r"""Represent irregularly sampled functional data.

    Parameters
    ----------
    argvals: IrregularArgvals
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. Then, each dimension is a
        dictionary where entries are the different observations. So, the
        observation :math:`i` for the dimension :math:`j` is a `np.ndarray`
        with shape :math:`(m^i_j,)` for :math:`0 \leq i \leq n` and
        :math:`0 \leq j \leq p`.
    values: IrregularValues
        The values of the functional data. Each entry of the dictionary is an
        observation of the process. And, an observation is represented by a
        `np.ndarray` of shape :math:`(n, m_1, \dots, m_p)`. It should not
        contain any missing values.

    Attributes
    ----------
    argvals_stand: IrregularArgvals
        Standardized sampling points of the functional data.
    n_obs: int
        Number of observations of the functional data.
    n_dimension: int
        Number of input dimension of the functional data.
    n_points: Dict[int, Tuple[int, ...]]
        Number of sampling points.

    Examples
    --------
    For 1-dimensional irregular data:

    >>> argvals = IrregularArgvals({
    ...     0: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4])}),
    ...     1: DenseArgvals({'input_dim_0': np.array([0, 2, 4])}),
    ...     2: DenseArgvals({'input_dim_0': np.array([2, 4])})
    ... })
    >>> values = IrregularValues({
    ...     0: np.array([1, 2, 3, 4, 5]),
    ...     1: np.array([2, 5, 6]),
    ...     2: np.array([4, 7])
    ... })
    >>> IrregularFunctionalData(argvals, values)

    For 2-dimensional irregular data:

    >>> argvals = IrregularArgvals({
    ...     0: DenseArgvals({
    ...         'input_dim_0': np.array([1, 2, 3, 4]),
    ...         'input_dim_1': np.array([5, 6, 7])
    ...     }),
    ...     1: DenseArgvals({
    ...         'input_dim_0': np.array([2, 4]),
    ...         'input_dim_1': np.array([1, 2, 3])
    ...     }),
    ...     2: DenseArgvals({
    ...         'input_dim_0': np.array([4, 5, 6]),
    ...         'input_dim_1': np.array([8, 9])
    ...     })
    ... })
    >>> values = IrregularValues({
    ...     0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
    ...     1: np.array([[1, 2, 3], [1, 2, 3]]),
    ...     2: np.array([[8, 9], [8, 9], [8, 9]])
    ... })
    >>> IrregularFunctionalData(argvals, values)

    References
    ----------
    .. [1] Benko, M., Härdle, W., Kneip, A., (2009), Common functional
        principal components. The Annals of Statistics 37, 1-34.
    .. [2] Cai, T.T., Yuan, M., (2011), Optimal estimation of the mean
        function based on discretely sampled functional data: Phase
        transition. The Annals of Statistics 39, 2330-2355.
    .. [3] Chiou, J.-M., Chen, Y.-T., Yang, Y.-F. (2014). Multivariate
        Functional Principal Component Analysis: A Normalization Approach.
        Statistica Sinica 24, 1571--1596.
    .. [4] Eilers, P. H. C., Marx, B. D. (2021). Practical Smoothing: The Joys
        of P-splines. Cambridge University Press, Cambridge.
    .. [5] Hall, P., Kay, J.W. and Titterington, D.M. (1990).
        Asymptotically Optimal Difference-Based Estimation of Variance in
        Nonparametric Regression. Biometrika 77, 521--528.
    .. [6] Happ and Greven (2018), Multivariate Functional Principal
        Component Analysis for Data Observed on Different (Dimensional)
        Domains. Journal of the American Statistical Association, 113,
        pp. 649--659.
    .. [7] Tsybakov, A.B. (2008), Introduction to Nonparametric Estimation.
        Springer Series in Statistics.
    .. [8] Yao, F., Müller, H.-G., Wang, J.-L. (2005). Functional Data
        Analysis for Sparse Longitudinal Data. Journal of the American
        Statistical Association 100, pp. 577--590.

    """

    ###########################################################################
    # Checkers

    ###########################################################################

    ###########################################################################
    # Static methods
    @staticmethod
    def _perform_computation(
        fdata1: IrregularFunctionalData, fdata2: IrregularFunctionalData, func: Callable
    ) -> IrregularFunctionalData:
        """Perform computation defined by `func` if they are compatible.

        Parameters
        ----------
        fdata1
            First functional data to consider.
        fdata2
            Second functional data to consider.
        func
            The function to apply to combine `fdata1` and `fdata2`.

        Returns
        -------
        IrregularFunctionalData
            The resulting functional data.

        """
        IrregularFunctionalData._is_compatible(fdata1, fdata2)

        new_values = {
            idx: func(obs1, obs2)
            for (idx, obs1), (_, obs2) in zip(
                fdata1.values.items(), fdata2.values.items()
            )
        }
        return IrregularFunctionalData(fdata1.argvals, IrregularValues(new_values))

    @staticmethod
    def _perform_computation_number(
        fdata: Type[FunctionalData], number: int | float, func: Callable
    ) -> Type[FunctionalData]:
        """Perform computation with numbers.

        Parameters
        ----------
        fdata
            Functional data to consider.
        number
            number to consider.
        func
            The function to apply to combine `fdata` and `number`.

        Returns
        -------
        IrregularFunctionalData
            The resulting functional data.

        """
        new_values = {idx: func(obs, number) for (idx, obs) in fdata.values.items()}
        return IrregularFunctionalData(fdata.argvals, IrregularValues(new_values))

    @staticmethod
    def concatenate(*fdata: IrregularFunctionalData) -> IrregularFunctionalData:
        """Concatenate IrregularFunctionalData objects.

        Parameters
        ----------
        fdata
            Functional data to concatenate.

        Returns
        -------
        IrregularFunctionalData
            The concatenated objects.

        """
        super(IrregularFunctionalData, IrregularFunctionalData).concatenate(*fdata)
        argvals = IrregularArgvals.concatenate(*[el.argvals for el in fdata])
        values = IrregularValues.concatenate(*[el.values for el in fdata])
        return IrregularFunctionalData(argvals, values)

    ###########################################################################

    ###########################################################################
    # Magic methods
    def __init__(self, argvals: IrregularArgvals, values: IrregularValues) -> None:
        """Initialize IrregularFunctionalData object."""
        super().__init__(argvals, values)

    def __iter__(self):
        """Initialize the iterator."""
        return IrregularFunctionalDataIterator(self)

    def __getitem__(self, index: int) -> IrregularFunctionalData:
        """Overrride getitem function, called when self[index].

        Parameters
        ----------
        index
            The observation(s) of the object to retrive.

        Returns
        -------
        IrregularFunctionalData
            The selected observation(s) as IrregularFunctionalData object.

        """
        if isinstance(index, slice):
            indices = index.indices(self.n_obs)
            argvals = {obs: self.argvals.get(obs) for obs in range(*indices)}
            values = {obs: self.values.get(obs) for obs in range(*indices)}
        elif isinstance(index, np.ndarray):
            argvals = {int(obs): self.argvals.get(obs) for obs in index}
            values = {int(obs): self.values.get(obs) for obs in index}
        else:
            argvals = {index: self.argvals[index]}
            values = {index: self.values[index]}
        return IrregularFunctionalData(
            IrregularArgvals(argvals), IrregularValues(values)
        )

    ###########################################################################

    ###########################################################################
    # Properties
    @GridFunctionalData.argvals.setter
    def argvals(self, new_argvals: IrregularArgvals) -> None:
        """Setter for argvals."""
        if not isinstance(new_argvals, IrregularArgvals):
            raise TypeError("new_argvals must be a IrregularArgvals object.")
        if hasattr(self, "values"):
            self._values.compatible_with(new_argvals)
        self._argvals = new_argvals
        self._argvals_stand = self._argvals.normalization()

    @GridFunctionalData.values.setter
    def values(self, new_values: IrregularValues) -> None:
        """Setter for values."""
        if not isinstance(new_values, IrregularValues):
            raise TypeError("new_values must be a IrregularValues object.")
        if hasattr(self, "argvals"):
            self._argvals.compatible_with(new_values)
        self._values = new_values

    ###########################################################################

    ###########################################################################
    # Methods
    def to_basis(
        self,
        points: DenseArgvals | None = None,
        method: str = "PS",
        penalty: float | None = None,
        **kwargs,
    ) -> BasisFunctionalData:
        """Convert the data to basis format.

        This function transforms a IrregularFunctionalData object into a
        BasisFunctionalData object using `method`.

        Parameters
        ----------
        points
            The argvals of the basis.
        method
            The method to get the coefficients.
        penalty
            Strictly positive. Penalty used in the P-splined fitting of the
            data.
        kwargs
            Other keyword arguments are passed to the function:
            :meth:`preprocessing.smoothing.PSplines`

        Returns
        -------
        BasisFunctionalData
            The expanded data.

        """
        from .basis import Basis

        argvals = self.argvals.to_dense()
        domain_min = tuple(val[0] for val in argvals.min_max.values())
        domain_max = tuple(val[1] for val in argvals.min_max.values())

        if method == "PS":
            if penalty is None:
                penalty = self.n_dimension * [1]
            ps = PSplines(**kwargs)
            basis = Basis(
                name=self.n_dimension * ("bsplines",),
                n_functions=self.n_dimension * (int(ps.n_segments + ps.degree),),
                degree=int(ps.degree),
                argvals=argvals,
            )

            n_functions = np.power(ps.n_segments + ps.degree, self.n_dimension)
            coefs = np.zeros((self.n_obs, n_functions))
            for idx, obs in enumerate(self):
                x = list(obs.argvals[idx].values())
                y = np.array(obs.values[idx], copy=True)

                weights = np.ones_like(y)
                weights[np.isnan(y)] = 0
                y[np.isnan(y)] = 0

                ps.fit(
                    x=x,
                    y=y,
                    sample_weights=weights,
                    penalty=penalty,
                    domain_min=domain_min,
                    domain_max=domain_max,
                )
                coefs[idx, :] = ps.beta_hat.flatten()
        else:
            raise ValueError("Method not implemented.")

        return BasisFunctionalData(basis=basis, coefficients=coefs)

    def to_long(self, reindex: bool = False) -> pd.DataFrame:
        """Convert the data to long format.

        This function transform a IrregularFunctionalData object into pandas
        DataFrame. It uses the long format to represent the
        IrregularFunctionalData object as a dataframe. This is a helper
        function as it might be easier for some computation, e.g., smoothing of
        the mean and covariance functions to have a long format.

        Parameters
        ----------
        reindex
            Should the observations be reindexed?

        Returns
        -------
        pd.DataFrame
            The data in a long format.

        Examples
        --------
        For one-dimensional functional data:

        >>> argvals = IrregularArgvals({
        ...     0: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4])}),
        ...     1: DenseArgvals({'input_dim_0': np.array([0, 2, 4])}),
        ...     2: DenseArgvals({'input_dim_0': np.array([2, 4])})
        ... })
        >>> values = IrregularValues({
        ...     0: np.array([1, 2, 3, 4, 5]),
        ...     1: np.array([2, 5, 6]),
        ...     2: np.array([4, 7])
        ... })
        >>> fdata = IrregularFunctionalData(argvals, values)
        >>> fdata.to_long()
           input_dim_0  id  values
        0            0   0       1
        1            1   0       2
        2            2   0       3
        3            3   0       4
        4            4   0       5
        5            0   1       2
        6            2   1       5
        7            4   1       6
        8            2   2       4
        9            4   2       7

        """
        temp_list = []
        for i, obs in enumerate(self):
            idx, cur_argvals = obs.argvals.popitem()
            cur_values = obs.values[idx]
            sampling_points = list(itertools.product(*cur_argvals.values()))

            temp = pd.DataFrame(sampling_points)
            temp.columns = list(cur_argvals.keys())
            temp["id"] = i if reindex else idx
            temp["values"] = cur_values.flatten()
            temp_list.append(temp)
            # temp_list.append(temp.dropna())
        return pd.concat(temp_list, ignore_index=True).dropna()

    def noise_variance(self, order: int = 2) -> float:
        """Estimate the variance of the noise.

        This function estimates the variance of the noise. The noise is
        estimated for each individual curve using the methodology in [3]_. As
        the curves are assumed to be generated by the same process, the
        estimation of the variance of the noise is the mean over the set of
        curves.

        Parameters
        ----------
        order
            Order of the difference sequence. The order has to be between
            1 and 10. See [3]_ for more information.

        Returns
        -------
        float
            The estimation of the variance of the noise.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=100)
        >>> kl.sparsify(0.5)
        >>> kl.sparse_data.noise_variance(order=2)
        0.006671248206782777

        """
        if self.n_dimension > 1:
            warnings.warn(
                (
                    "The estimation of the variance of the noise is not "
                    "performed for data with dimension larger than 1 and is "
                    "set to 0."
                ),
                UserWarning,
            )
            return 0
        variances = [
            _estimate_noise_variance(obs.values[idx][~np.isnan(obs.values[idx])], order)
            for idx, obs in enumerate(self)
        ]
        return np.nanmean(variances)

    def smooth(
        self,
        points: DenseArgvals | None = None,
        method: str = "PS",
        bandwidth: float | None = None,
        penalty: float | None = None,
        **kwargs,
    ) -> DenseFunctionalData:
        """Smooth the data.

        This function smooths each curves individually. Based on [2]_, it fits
        a local polynomial smoother to the data. Based on [4]_, it fits
        P-splines to the data.

        Parameters
        ----------
        points
            Points at which the curves are estimated. The default is None,
            meaning we use the argvals as estimation points.
        method
            The method to used for the smoothing. If 'PS', the method is
            P-splines [4]_. If 'LP', the method is local polynomials [2]_.
            Otherwise, it raises an error.
        bandwidth
            Strictly positive. Control the size of the associated neighborhood.
            If ``bandwidth=None``, it is assumed that the curves are twice
            differentiable and the bandwidth is set to :math:`n^{-1/5}` [7]_
            where :math:`n` is the number of sampling points per curve. Be
            careful with the results if the curves are not sampled on
            :math:`[0, 1]`.
        penalty
            Strictly positive. Penalty used in the P-splined fitting of the
            data.
        kwargs
            Other keyword arguments are passed to one of the following
            functions: :meth:`preprocessing.smoothing.PSplines` (``method='PS'``) and
            :meth:`preprocessing.smoothing.LocalPolynomial` (``method='LP'``).

        Returns
        -------
        DenseFunctionalData
            Smoothed data.

        Examples
        --------
        For one-dimensional functional data:

        >>> argvals = IrregularArgvals({
        ...     0: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4])}),
        ...     1: DenseArgvals({'input_dim_0': np.array([0, 2, 4])}),
        ...     2: DenseArgvals({'input_dim_0': np.array([2, 4])})
        ... })
        >>> values = IrregularValues({
        ...     0: np.array([1, 2, 3, 4, 5]),
        ...     1: np.array([2, 5, 6]),
        ...     2: np.array([4, 7])
        ... })
        >>> fdata = IrregularFunctionalData(argvals, values)
        >>> fdata.smooth()
        Functional data object with 3 observations on a 1-dimensional support.

        """
        if points is None:
            points = self.argvals.to_dense()
        domain_min = tuple(val[0] for val in points.min_max.values())
        domain_max = tuple(val[1] for val in points.min_max.values())

        if method == "LP":
            if bandwidth is None:
                n_points = np.mean([obs for obs in self.n_points.values()])
                bandwidth = n_points ** (-1 / 5)

            points_mat = _cartesian_product(*points.values())

            lp = LocalPolynomial(bandwidth=bandwidth, **kwargs)

            smooth = np.zeros((self.n_obs, *points.n_points))
            for idx, obs in enumerate(self):
                argvals_mat = _cartesian_product(*obs.argvals[idx].values())
                smooth[idx, :] = lp.predict(
                    y=obs.values[idx].flatten(), x=argvals_mat, x_new=points_mat
                ).reshape(smooth.shape[1:])
        elif method == "PS":
            if penalty is None:
                penalty = self.n_dimension * [1]

            ps = PSplines(**kwargs)
            smooth = np.zeros((self.n_obs, *points.n_points))
            for idx, obs in enumerate(self):
                x = list(obs.argvals[idx].values())
                y = np.array(obs.values[idx], copy=True)

                weights = np.ones_like(y)
                weights[np.isnan(y)] = 0
                y[np.isnan(y)] = 0

                ps.fit(
                    x=x,
                    y=y,
                    sample_weights=weights,
                    penalty=penalty,
                    domain_min=domain_min,
                    domain_max=domain_max,
                )
                smooth[idx, :] = ps.predict(x=list(points.values()))
        elif method == "interpolation":
            from scipy.interpolate import NearestNDInterpolator

            smooth = np.zeros((self.n_obs, *points.n_points))
            for idx, obs in enumerate(self):
                fdata_long = obs.to_long()
                x = fdata_long.drop(["id", "values"], axis=1, inplace=False).values
                y = fdata_long["values"].values

                if self.n_dimension == 1:
                    smooth[idx, :] = np.interp(points["input_dim_0"], x.flatten(), y)
                else:
                    new_X = [pp for pp in points.values()]
                    X_matrices = np.meshgrid(*new_X, indexing="ij")

                    interp = NearestNDInterpolator(x, y)
                    smooth[idx, :] = interp(*X_matrices)
        else:
            raise NotImplementedError("Method not implemented.")
        return DenseFunctionalData(points, DenseValues(smooth))

    def mean(
        self,
        points: DenseArgvals | None = None,
        method_smoothing: str = "LP",
        approx: bool = True,
        **kwargs,
    ) -> DenseFunctionalData:
        """Compute an estimate of the mean.

        This function computes an estimate of the mean curve of a
        IrregularFunctionalData object. The curves are not sampled on a common
        grid. We implement the methodology from [2]_.

        Parameters
        ----------
        points
            The sampling points at which the mean is estimated. If `None`, the
            concatenation of the argvals of the IrregularFunctionalData is used.
        method_smoothing
            The method to used for the smoothing. If 'PS', the method is
            P-splines [4]_. If 'LP', the method is local polynomials [2]_.
        approx
            Approximation of the estimation.
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`IrregularFunctionalData.smooth`.

        Returns
        -------
        DenseFunctionalData
            An estimate of the mean as a DenseFunctionalData object.

        Examples
        --------
        For one-dimensional functional data:

        >>> argvals = IrregularArgvals({
        ...     0: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4])}),
        ...     1: DenseArgvals({'input_dim_0': np.array([0, 2, 4])}),
        ...     2: DenseArgvals({'input_dim_0': np.array([2, 4])})
        ... })
        >>> values = IrregularValues({
        ...     0: np.array([1, 2, 3, 4, 5]),
        ...     1: np.array([2, 5, 6]),
        ...     2: np.array([4, 7])
        ... })
        >>> fdata = IrregularFunctionalData(argvals, values)
        >>> fdata.mean()
        Functional data object with 1 observations on a 1-dimensional support.

        """
        if points is None:
            points = self.argvals.to_dense()

        fdata_long = self.to_long()
        if approx and len(fdata_long) > 2000:
            str_sub = [f"input_dim_{idx}" for idx in np.arange(self.n_dimension)]
            temp = fdata_long.groupby(str_sub).mean().reset_index()
            x = temp.drop(["id", "values"], axis=1, inplace=False).values
            y = temp["values"].values
        else:
            x = fdata_long.drop(["id", "values"], axis=1, inplace=False).values
            y = fdata_long["values"].values

        if method_smoothing == "LP":
            bandwidth = kwargs.pop("bandwidth", None)
            if bandwidth is None:
                n_points = np.mean([obs for obs in self.n_points.values()])
                bandwidth = n_points ** (-1 / 5)
            points_mat = _cartesian_product(*points.values())

            lp = LocalPolynomial(bandwidth=bandwidth, **kwargs)
            pred = lp.predict(y=y, x=x, x_new=points_mat).reshape(points.n_points)
        elif method_smoothing == "PS":
            penalty = kwargs.pop("penalty", self.n_dimension * (1,))

            x, y, weights = _format_data(x, y)
            ps = PSplines(**kwargs)

            ps.fit(x=x, y=y, sample_weights=weights, penalty=penalty)
            pred = ps.predict([pp for pp in points.values()])
        elif method_smoothing == "interpolation":
            from scipy.interpolate import NearestNDInterpolator

            if self.n_dimension == 1:
                pred = np.interp(points["input_dim_0"], x.flatten(), y)
            else:
                new_X = [pp for pp in points.values()]
                X_matrices = np.meshgrid(*new_X, indexing="ij")

                interp = NearestNDInterpolator(x, y)
                pred = interp(*X_matrices)
        else:
            raise ValueError("Method not implemented.")

        self._mean = DenseFunctionalData(points, DenseValues(pred[np.newaxis]))
        return self._mean

    def center(
        self,
        mean: DenseFunctionalData | None = None,
        method_smoothing: str = "LP",
        **kwargs,
    ) -> IrregularFunctionalData:
        """Center the data.

        Parameters
        ----------
        mean
            A precomputed mean as a DenseFunctionalData object.
        method_smoothing
            The method to used for the smoothing of the mean. If 'PS', the
            method is P-splines [4]_. If 'LP', the method is local polynomials
            [2]_.
        kwargs
            Other keyword arguments are passed to one of the following
            functions: :meth:`IrregularFunctionalData.mean` (``mean=None``) and
            :meth:`IrregularFunctionalData.smooth`.

        Returns
        -------
        IrregularFunctionalData
            The centered version of the data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.add_noise_and_sparsify(0.01, 0.95)
        >>> kl.sparse_data.center(smooth=True)
        Functional data object with 10 observations on a 1-dimensional support.

        """
        new_argvals = self.argvals.to_dense()
        if mean is None:
            data_mean = self.mean(
                points=new_argvals, method_smoothing=method_smoothing, **kwargs
            )
        else:
            data_mean = mean.smooth(new_argvals, method=method_smoothing, **kwargs)

        obs_centered = {}
        for idx, obs in enumerate(self):
            obs_points = np.isin(
                new_argvals["input_dim_0"], obs.argvals[idx]["input_dim_0"]
            )
            mean_obs = data_mean.values[0][obs_points]
            obs_centered[idx] = obs.values[idx] - mean_obs
        return IrregularFunctionalData(self.argvals, IrregularValues(obs_centered))

    def norm(
        self,
        squared: bool = False,
        method_integration: str = "trapz",
        use_argvals_stand: bool = False,
    ) -> npt.NDArray[np.float64]:
        r"""Norm of each observation of the data.

        For each observation in the data, it computes its norm defined in [6]_
        as

        .. math::
            \| X \| = \left\{\int_{\mathcal{T}} X(t)^2dt\right\}^{\frac12}.

        Parameters
        ----------
        squared
            If `True`, the function calculates the squared norm, otherwise the
            result is not squared.
        method_integration
            The method used to integrated.
        use_argvals_stand
            Use standardized argvals to compute the normalization of the data.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs,)
            The norm of each observations.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.sparsify(percentage=0.5, epsilon=0.05)
        >>> kl.sparse_data.norm()
        array([
            0.53419879, 0.40750272, 0.67092435, 0.26762124, 0.27425138,
            0.37419987, 0.65775515, 0.54579643, 0.25830787, 0.49324345
        ])

        """
        data_interp = self.smooth(method="interpolation")
        return data_interp.norm(
            squared=squared,
            method_integration=method_integration,
            use_argvals_stand=use_argvals_stand,
        )
        # norm_fd = np.zeros(data_interp.n_obs)
        # for idx, obs in enumerate(data_interp):
        #    if use_argvals_stand:
        #        axis = [argvals for argvals in obs.argvals_stand[idx].values()]
        #    else:
        #        axis = [argvals for argvals in obs.argvals[idx].values()]
        #    values = obs.values[idx][~np.isnan(obs.values[idx])]
        #    sq_values = np.power(values, 2)
        #    norm_fd[idx] = _integrate
        # (sq_values, *axis, method=method_integration)

        # if squared:
        #    return np.array(norm_fd)
        # else:
        #    return np.power(norm_fd, 0.5)

    def normalize(self, **kwargs) -> IrregularFunctionalData:
        r"""Normalize the data.

        The normalization is performed by divising each functional datum
        :math:`X` by its norm :math:`\| X \|`. It results in

        .. math::
            \widetilde{X} = \frac{X}{\| X \|}.

        Parameters
        ----------
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`IrregularFunctionalData.norm`.

        Returns
        -------
        IrregularFunctionalData
            The normalized data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.sparsify(percentage=0.5, epsilon=0.05)
        >>> kl.sparse_data.normalize()
        Functional data object with 10 observations on a 1-dimensional support.

        """
        norm_val = self.norm(**kwargs)

        new_values = IrregularValues()
        for idx, (obs, norm) in enumerate(zip(self, norm_val)):
            new_values[idx] = obs.values[idx] / norm
        return IrregularFunctionalData(self.argvals, new_values)

    def standardize(self, center: bool = True, **kwargs) -> IrregularFunctionalData:
        r"""Standardize the data.

        The standardization is performed by first centering the data and then
        dividing by the standard deviation curve [3]_. It results in

        .. math::
            \widetilde{X}(t) = C(t, t)^{-\frac12}\{X(t) - \mu(t)\}, \quad
            t \in \mathcal{T}.

        Parameters
        ----------
        center: bool, default=True
            Should the data be centered?
        **kwargs
            Other keyword arguments are passed to the following functions:

            - :meth:`IrregularFunctionalData.center`,
            - :meth:`IrregularFunctionalData.covariance`.

        Returns
        -------
        IrregularFunctionalData
            The standardized data.


        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.sparsify(percentage=0.5, epsilon=0.05)
        >>> kl.sparse_data.standardize()
        Functional data object with 10 observations on a 1-dimensional support.

        """
        if center:
            fdata = self.center(**kwargs)
        else:
            fdata = self
        covariance = fdata.covariance(**kwargs)
        variance = np.diag(covariance.values.squeeze())

        obs_standardized = {}
        for idx, obs in enumerate(fdata):
            obs_points = np.isin(
                covariance.argvals["input_dim_0"], obs.argvals[idx]["input_dim_0"]
            )
            std_obs = np.sqrt(variance[obs_points])
            obs_standardized[idx] = np.divide(
                obs.values[idx], std_obs, where=(std_obs > 1e-12)
            )
        return IrregularFunctionalData(self.argvals, IrregularValues(obs_standardized))

    def rescale(
        self,
        weights: float = 0.0,
        method_integration: str = "trapz",
        method_smoothing: str = "LP",
        use_argvals_stand: bool = False,
        **kwargs,
    ) -> Tuple[IrregularFunctionalData, float]:
        r"""Rescale the data.

        The rescaling is performed by first centering the data and then
        multiplying with a common weight:

        .. math::
            \widetilde{X}(t) = w\{X(t) - \mu(t)\}.

        The weights are defined in [6]_.

        Parameters
        ----------
        weights
            The weights used to normalize the data. If `weights = 0.0`, the
            weights are estimated by integrating the variance function [3]_.
        method_integration
            The method used to integrated.
        use_argvals_stand
            Use standardized argvals to compute the normalization of the data.
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`IrregularFunctionalData.smooth`.

        Returns
        -------
        Tuple[IrregularFunctionalData, float]
            The rescaled data and the weight.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.sparsify(percentage=0.5, epsilon=0.05)
        >>> kl.sparse_data.normalize()
        (Functional data object with 10 observations on a 1-dimensional
        support., DenseValues(0.16802008))

        """
        if weights == 0.0:
            data_smooth = self.smooth(method=method_smoothing, **kwargs)
            if use_argvals_stand:
                argvals_stand = data_smooth.argvals_stand.values()
                axis = [argvals for argvals in argvals_stand]
            else:
                axis = [argvals for argvals in data_smooth.argvals.values()]
            variance = np.var(data_smooth.values, axis=0)
            weights = _integrate(variance, *axis, method=method_integration)
        return self / np.sqrt(float(weights)), weights

    def inner_product(
        self,
        method_integration: str = "trapz",
        method_smoothing: str = "LP",
        noise_variance: float | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        r"""Compute the inner product matrix of the data.

        The inner product matrix is a ``n_obs`` by ``n_obs`` matrix where each
        entry is defined as

        .. math::
            \langle x, y \rangle = \int_{\mathcal{T}} x(t)y(t)dt,
            t \in \mathcal{T},

        where :math:`\mathcal{T}` is a one- or multi-dimensional domain [1]_.

        Parameters
        ----------
        method_integration
            The method used to integrated.
        method_smoothing
            Should the mean be smoothed?
        noise_variance
            An estimation of the variance of the noise. If `None`, an
            estimation is computed using the methodology in [5]_.
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`IrregularFunctionalData.center`.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_obs)
            Inner product matrix of the data.

        Raises
        ------
        NotImplementedError
            Not implement for higher-dimensional data.

        Examples
        --------
        For one-dimensional functional data:

        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines', n_functions=5, random_state=5
        ... )
        >>> kl.new(n_obs=3)
        >>> kl.sparsify(percentage=0.8, epsilon=0.05)
        >>> kl.sparse_data.inner_product(noise_variance=0)
        array([
            [ 0.15749721,  0.01983093, -0.09607059],
            [ 0.01983093,  0.17937531, -0.24773228],
            [-0.09607059, -0.24773228,  0.41648575]
        ])

        """
        # Center the data
        data = self.center(method_smoothing=method_smoothing, **kwargs)

        # Estimate the noise of the variance
        if noise_variance is None:
            self._noise_variance = self.noise_variance(order=2)
        else:
            self._noise_variance = noise_variance

        self._data_inpro = data.smooth(method="interpolation")
        return self._data_inpro.inner_product(
            method_integration=method_integration,
            method_smoothing=None,
            noise_variance=self._noise_variance,
        )

    def covariance(
        self,
        points: DenseArgvals | None = None,
        method_smoothing: str = "LP",
        center: bool = True,
        smooth: bool = True,
        kwargs_center: Dict[str, object] = {},
        **kwargs,
    ) -> DenseFunctionalData:
        """Compute an estimate of the covariance function.

        This function computes an estimate of the covariance surface of a
        IrregularFunctionalData object. As the curves are not sampled on a
        common grid, we consider the method in [8]_.

        Parameters
        ----------
        points
            The sampling points at which the covariance is estimated. If
            `None`, the concatenation of the IrregularArgvals of the
            IrregularFunctionalData is used.
        method_smoothing
            The method to used for the smoothing of the mean. If 'PS', the
            method is P-splines [4]_. If 'LP', the method is local polynomials
            [2]_.
        center
            Should the data be centered before computing the covariance.
        smooth
            Should the covariance be smoothed.
        kwargs_center
            Keyword arguments to be passed to the function
            :meth:`FunctionalData.center`.
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`FunctionalData._smooth_covariance`.

        Returns
        -------
        DenseFunctionalData
            An estimate of the covariance as a two-dimensional
            DenseFunctionalData object.

        Raises
        ------
        NotImplementedError
            Not implement for higher-dimensional data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=100)
        >>> kl.sparsify(percentage=0.5, epsilon=0.05)
        >>> kl.sparse_data.covariance()
        Functional data object with 1 observations on a 2-dimensional support.

        """
        if self.n_dimension > 1:
            raise NotImplementedError(
                "Only implemented for one-dimensional irregular ", "functional data."
            )
        if points is None:
            points = self.argvals.to_dense()
        argvals_cov = DenseArgvals(
            {
                "input_dim_0": self.argvals.to_dense()["input_dim_0"],
                "input_dim_1": self.argvals.to_dense()["input_dim_0"],
            }
        )
        points_cov = DenseArgvals(
            {
                "input_dim_0": points["input_dim_0"],
                "input_dim_1": points["input_dim_0"],
            }
        )
        n_points = self.argvals.to_dense().n_points

        # Center the data
        data = self
        if center:
            data = data.center(method_smoothing=method_smoothing, **kwargs_center)

        # Compute the covariance
        cov = np.zeros(np.power(n_points, 2))
        cov_sum = np.zeros(np.power(n_points, 2))
        cov_count = np.zeros(np.power(n_points, 2))
        for idx, obs in enumerate(data):
            nan_mask = np.isnan(obs.values[idx])
            new_argvals = obs.argvals[idx]["input_dim_0"][~nan_mask]
            new_values = obs.values[idx][~nan_mask]

            obs_points = np.isin(self.argvals.to_dense()["input_dim_0"], new_argvals)
            mask = np.outer(obs_points, obs_points).flatten()
            cov_inner = np.outer(new_values, new_values).flatten()

            cov_count[mask] += 1
            cov_sum[mask] += cov_inner
        _ = np.divide(cov_sum, cov_count, where=(cov_count != 0), out=cov)
        cov = cov.reshape(2 * n_points)
        # cov[cov < 1e-12] = 0
        raw_diag_cov = np.diag(cov).copy()

        # Smooth the covariance
        if smooth:
            weights = np.ones_like(cov)
            weights[cov == 0] = 0

            cov = _smooth_covariance(
                cov,
                argvals_cov,
                points_cov,
                method_smoothing=method_smoothing,
                weights=weights,
                **kwargs,
            )

        # Ensure the covariance is symmetric.
        cov = (cov + cov.T) / 2

        # Estimate noise variance ([2], [3])
        self._noise_variance_cov = _estimate_noise_variance_with_covariance(
            raw_diag_cov, np.diag(cov), self.argvals.to_dense(), points
        )

        self._covariance = DenseFunctionalData(points_cov, DenseValues(cov[np.newaxis]))
        return self._covariance

    ###########################################################################


###############################################################################
# Class BasisFunctionalDataIterator
class BasisFunctionalDataIterator(Iterator):
    """Iterator for functional data represented as a basis."""

    def __init__(self, fdata):
        """Initialize the Iterator object."""
        self._fdata = fdata
        self._index = 0

    def __next__(self):
        """Return the next item in the sequence."""
        if self._index < self._fdata.n_obs:
            item = self._fdata[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration


###############################################################################
# Class BasisFunctionalData
class BasisFunctionalData(FunctionalData):
    r"""Represent functional data with a basis.

    A class used to defined functional data with a basis expansion. We denote by
    :math:`n`, the number of observations and by :math:`p`, the number of input
    dimensions. Here, we are in the case of univariate functional data, and so the
    output dimension will be :math:`\mathbb{R}`. We note by :math:`X` an observation,
    while we use :math:`X_1, \dots, X_n` if we refer to a particular set of
    observations. The observations are defined as:

    .. math::
        X(t) = \sum_{k = 1}^K c_k \phi_k(t), \quad t \in \mathcal{T},

    where :math:`\mathcal{T} \subset \mathbb{R}^p` and the :math:`\phi_k(t)` is a set of
    functions.

    Parameters
    ----------
    basis: Basis
        The basis of the functional data.
    coefficients: npt.NDArray[np.float64]
        The set of coefficients.

    Attributes
    ----------
    n_obs: int
        Number of observations of the functional data.
    n_dimension: int
        Number of input dimension of the functional data.
    n_points: Tuple[int, ...]
        Number of sampling points.

    """

    ###########################################################################
    # Checkers
    @staticmethod
    def _is_compatible(*fdata: Type[FunctionalData]) -> None:
        """Raise an error if elements in `fdata` are not compatible.

        Parameters
        ----------
        fdata
            Functional data to compare.

        """
        FunctionalData._is_compatible(*fdata)

    ###########################################################################

    ###########################################################################
    # Static methods
    @staticmethod
    def _perform_computation(
        fdata1: Type[FunctionalData], fdata2: Type[FunctionalData], func: Callable
    ) -> Type[FunctionalData]:
        """Perform computation."""

    @staticmethod
    def _perform_computation_number(
        fdata: Type[FunctionalData], number: float, func: Callable
    ) -> Type[FunctionalData]:
        """Perform computation with numbers."""

    @staticmethod
    def concatenate(*fdata: BasisFunctionalData) -> BasisFunctionalData:
        """Concatenate FunctionalData objects.

        Parameters
        ----------
        fdata
            Functional data to concatenate.

        Returns
        -------
        BasisFunctionalData
            Concatenated data.

        Raises
        ------
        NotImplementedError
            Not implemented for BasisFunctionalData.

        """
        raise NotImplementedError()

    ###########################################################################

    ###########################################################################
    # Magic methods
    def __init__(
        self,
        basis: Type[Basis],
        coefficients: npt.NDArray[np.float64],
    ) -> None:
        """Initialize GridFunctionalData object."""
        self.basis = basis
        self.coefficients = coefficients
        self._index = 0

    def __iter__(self):
        """Initialize the iterator."""
        return BasisFunctionalDataIterator(self)

    def __getitem__(self, index: int) -> Type[FunctionalData]:
        """Override getitem function, called when self[index]."""
        new_coefs = self.coefficients[index]
        if len(new_coefs.shape) == 1:
            new_coefs = new_coefs[np.newaxis]
        return BasisFunctionalData(coefficients=new_coefs, basis=self.basis)

    ###########################################################################

    ###########################################################################
    # Properties
    @property
    def n_obs(self) -> int:
        """Get the number of observations of the functional data."""
        return self.coefficients.shape[0]

    @property
    def n_dimension(self) -> int:
        """Get the number of input dimension of the functional data."""
        return self.basis.n_dimension

    @property
    def n_points(self) -> Tuple[int, ...]:
        """Get the number of sampling points."""
        return self.basis.n_points

    ###########################################################################

    ###########################################################################
    # Methods
    def to_grid(self) -> DenseFunctionalData:
        """Convert the data to grid format.

        Returns
        -------
        DenseFunctionalData
            The data in grid format.

        """
        new_argvals = self.basis.argvals
        new_values = np.einsum("ij,j... -> i...", self.coefficients, self.basis.values)
        return DenseFunctionalData(new_argvals, DenseValues(new_values))

    def to_long(self, reindex: bool = False) -> pd.DataFrame:
        """Convert the data to long format.

        Parameters
        ----------
        reindex
            Not used here.

        Returns
        -------
        pd.DataFrame
            The data in a long format.

        Raises
        ------
        NotImplementedError
            Not implemented for BasisFunctionalData.

        """
        raise NotImplementedError()

    def noise_variance(self, order: int = 2) -> float:
        """Estimate the variance of the noise.

        Parameters
        ----------
        order
            Order of the difference sequence. The order has to be between
            1 and 10.

        Returns
        -------
        float
            The estimation of the variance of the noise.

        Raises
        ------
        NotImplementedError
            Not implemented for BasisFunctionalData.

        """
        raise NotImplementedError()

    def smooth(
        self,
        points: DenseArgvals | None = None,
        method: str = "PS",
        bandwidth: float | None = None,
        penalty: float | None = None,
        **kwargs,
    ) -> BasisFunctionalData:
        """Smooth the data.

        Parameters
        ----------
        points
            Points at which the curves are estimated. The default is None,
            meaning we use the argvals as estimation points.
        method
            The method to used for the smoothing. If 'PS', the method is
            P-splines. If 'LP', the method is local polynomials.
            Otherwise, it raises an error.
        bandwidth
            Strictly positive. Control the size of the associated neighborhood.
            If ``bandwidth=None``, it is assumed that the curves are twice
            differentiable and the bandwidth is set to :math:`n^{-1/5}`
            where :math:`n` is the number of sampling points per curve. Be
            careful with the results if the curves are not sampled on
            :math:`[0, 1]`.
        penalty
            Strictly positive. Penalty used in the P-splined fitting of the
            data.
        kwargs
            Other keyword arguments are passed to one of the following
            functions :meth:`preprocessing.smoothing.PSplines` (``method='PS'``) and
            :meth:`preprocessing.smoothing.LocalPolynomial` (``method='LP'``).

        Returns
        -------
        BasisFunctionalData
            Smoothed data.

        Raises
        ------
        NotImplementedError
            Not implemented for BasisFunctionalData.

        """
        raise NotImplementedError()

    def mean(
        self,
        points: DenseArgvals | None = None,
        method_smoothing: str = None,
        **kwargs,
    ) -> FunctionalData:
        """Compute an estimate of the mean.

        Parameters
        ----------
        points
            The sampling points at which the mean is estimated. If `None`, the
            DenseArgvals of the DenseFunctionalData is used.
        method_smoothing
            The method to used for the smoothing. If 'None', no smoothing is
            performed. If 'PS', the method is P-splines. If 'LP', the
            method is local polynomials.
        kwargs
            Other keyword arguments are passed to the following function
            :meth:`FunctionalData.smooth`.

        Returns
        -------
        DenseFunctionalData
            An estimate of the mean as a DenseFunctionalData object.

        """
        mean = np.mean(self.coefficients, axis=0)[np.newaxis]
        return BasisFunctionalData(self.basis, mean)

    def center(
        self,
        mean: None = None,
        method_smoothing: None = None,
        **kwargs,
    ) -> BasisFunctionalData:
        """Center the data.

        Parameters
        ----------
        mean
            Not used here.
        method_smoothing
            Not used here.
        kwargs
            Not used here.

        Returns
        -------
        BasisFunctionalData
            The centered version of the data.

        """
        new_coefs = self.coefficients - np.mean(self.coefficients, axis=0)
        return BasisFunctionalData(self.basis, new_coefs)

    def norm(
        self,
        squared: bool = False,
        method_integration: str = "trapz",
        use_argvals_stand: bool = False,
    ) -> npt.NDArray[np.float64]:
        r"""Norm of each observation of the data.

        For each observation in the data, it computes its norm defined
        as

        .. math::
            \| X \| = \left\{\int_{\mathcal{T}} X(t)^2dt\right\}^{\frac12}.

        Parameters
        ----------
        squared
            If ``True``, the function calculates the squared norm, otherwise it
            returns the norm.
        method_integration
            The method used to estimate the integral.
        use_argvals_stand
            Not used here.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs,)
            The norm of each observations.

        """
        inner_product = self.inner_product(method_integration=method_integration)
        norm_obs = np.diag(inner_product)
        if squared:
            return np.array(norm_obs)
        else:
            return np.power(norm_obs, 0.5)

    def normalize(self, **kwargs) -> BasisFunctionalData:
        """Normalize the data.

        Parameters
        ----------
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`BasisFunctionalData.norm`.

        Returns
        -------
        BasisFunctionalData
            The normalized data.

        """
        norm = np.moveaxis(self.coefficients, 0, -1) / self.norm(**kwargs)
        return BasisFunctionalData(self.basis, np.moveaxis(norm, -1, 0))

    def standardize(self, center: bool = True, **kwargs) -> BasisFunctionalData:
        """Standardize the data.

        Parameters
        ----------
        center
            Should the data be centered?
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`BasisFunctionalData.center`.

        Returns
        -------
        DenseFunctionalData
            The standardized data.

        """
        if center:
            fdata = self.center(**kwargs)
        else:
            fdata = self
        std = np.sqrt(np.diag(fdata.covariance().to_grid().values.squeeze()))
        basis_values = np.divide(fdata.basis.values, std, where=(std != 0))
        fdata.basis.values = basis_values
        return BasisFunctionalData(fdata.basis, fdata.coefficients)

    def rescale(
        self,
        weights: float = 0.0,
        method_integration: str = "trapz",
        use_argvals_stand: bool = False,
        **kwargs,
    ) -> Tuple[BasisFunctionalData, float]:
        """Rescale the data.

        Parameters
        ----------
        weights
            The weights used to normalize the data. If `weights = 0.0`, the
            weights are estimated by integrating the variance function.
        method_integration
            The method used to estimate the integral.
        use_argvals_stand
            Use standardized argvals to compute the normalization of the data.

        Returns
        -------
        Tuple[BasisFunctionalData, float]
            The rescaled data and the weight.

        """
        if weights == 0.0:
            if use_argvals_stand:
                axis = [argvals for argvals in self.basis.argvals_stand.values()]
            else:
                axis = [argvals for argvals in self.basis.argvals.values()]
            variance = np.diag(self.covariance().to_grid().values.squeeze())
            weights = _integrate(variance, *axis, method=method_integration)
        new_coefs = self.coefficients / np.sqrt(float(weights))
        return BasisFunctionalData(self.basis, new_coefs), weights

    def inner_product(
        self,
        method_integration: str = "trapz",
        method_smoothing: str | None = None,
        noise_variance: float | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """Compute an estimate of the inner product matrix.

        Parameters
        ----------
        method_integration
            The method used to integrated.
        method_smoothing
            The method to used for the smoothing of the mean. If 'None', no
            smoothing is performed. If 'PS', the method is P-splines. If
            'LP', the method is local polynomials.
        noise_variance
            An estimation of the variance of the noise. If `None`, an
            estimation is computed using the methodology.
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`BasisFunctionalData.center`.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_obs)
            Inner product matrix of the data.

        """
        inner_product = self.basis.inner_product(method_integration=method_integration)
        return self.coefficients @ inner_product @ self.coefficients.T

    def covariance(
        self,
        points: None = None,
        method_smoothing: None = None,
        **kwargs,
    ) -> Type[FunctionalData]:
        """Compute an estimate of the covariance.

        Parameters
        ----------
        points
            Not used here.
        method_smoothing
            Not used here.
        kwargs
            Not used here.

        Returns
        -------
        BasisFunctionalData
            An estimate of the covariance as a two-dimensional
            BasisFunctionalData object.

        """
        # Center the data
        data = self.center()

        # Estimate the covariance
        cov = (data.coefficients.T @ data.coefficients) / data.n_obs
        cov = cov.flatten()[np.newaxis]

        # Build the represetation
        new_argvals = DenseArgvals()
        for idx, values in enumerate(self.basis.argvals.values()):
            new_argvals[f"input_dim_{2 * idx}"] = values
            new_argvals[f"input_dim_{2 * idx + 1}"] = values

        new_dim = (self.basis.n_obs**2, *(2 * self.n_points))
        new_values = np.kron(self.basis.values, self.basis.values).reshape(new_dim)

        new_basis = DenseFunctionalData(new_argvals, new_values)
        return BasisFunctionalData(new_basis, cov)

    ###########################################################################


###############################################################################
# Class MultivariateFunctionalData
class MultivariateFunctionalData(UserList[Type[FunctionalData]]):
    r"""Represent multivariate functional data.

    An instance of MultivariateFunctionalData is a list containing objects of
    the class DenseFunctionalData or IrregularFunctionalData.

    Parameters
    ----------
    initlist: List[Type[FunctionalData]]
        The list containing the elements of the MultivariateFunctionalData.

    Attributes
    ----------
    n_obs: int
        Number of observations of the functional data.
    n_functional: int
        Number of components of the multivariate functional data.
    n_dimension: List[int]
        Number of input dimension of the functional data.
    n_points: List[Dict[str, int]]
        Number of sampling points.

    Examples
    --------
    >>> argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 5])})
    >>> values = DenseValues(np.array([
    ...     [1, 2, 3, 4, 5],
    ...     [6, 7, 8, 9, 10],
    ...     [11, 12, 13, 14, 15]
    ... ]))
    >>> fdata_dense = DenseFunctionalData(argvals, values)

    >>> argvals = IrregularArgvals({
    ...     0: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4])}),
    ...     1: DenseArgvals({'input_dim_0': np.array([0, 2, 4])}),
    ...     2: DenseArgvals({'input_dim_0': np.array([2, 4])})
    ... })
    >>> values = IrregularValues({
    ...     0: np.array([1, 2, 3, 4, 5]),
    ...     1: np.array([2, 5, 6]),
    ...     2: np.array([4, 7])
    ... })
    >>> fdata_irregular = IrregularFunctionalData(argvals, values)

    >>> MultivariateFunctionalData([fdata_dense, fdata_irregular])

    Notes
    -----
    Be careful that we will not check if all the elements have the same type.
    It is possible to create MultivariateFunctionalData containing both
    Dense, Iregular and Basis functional data. The number of observations has to be the
    same for each element of the list.

    References
    ----------
    .. [1] Benko, M., Härdle, W. and Kneip, A. (2009). Common functional
        principal components. The Annals of Statistics 37, 1--34.
    .. [2] Chiou, J.-M., Chen, Y.-T., Yang, Y.-F. (2014). Multivariate
        Functional Principal Component Analysis: A Normalization Approach.
        Statistica Sinica 24, 1571--1596.
    .. [3] Eilers, P. H. C., Marx, B. D. (2021). Practical Smoothing: The Joys
        of P-splines. Cambridge University Press, Cambridge.
    .. [4] Hall, P., Kay, J.W. and Titterington, D.M. (1990).
        Asymptotically Optimal Difference-Based Estimation of Variance in
        Nonparametric Regression. Biometrika 77, 521--528.
    .. [5] Happ and Greven (2018), Multivariate Functional Principal
        Component Analysis for Data Observed on Different (Dimensional)
        Domains. Journal of the American Statistical Association, 113,
        pp. 649--659.
    .. [6] Tsybakov, A.B. (2008), Introduction to Nonparametric Estimation.
        Springer Series in Statistics.
    .. [7] Zhang, J.-T. and Chen J. (2007), Statistical Inferences for
        Functional Data, The Annals of Statistics, Vol. 35, No. 3.

    """

    ###########################################################################
    # Static methods
    @staticmethod
    def concatenate(*fdata: MultivariateFunctionalData) -> MultivariateFunctionalData:
        """Concatenate MultivariateFunctionalData objects.

        Parameters
        ----------
        data
            The data to concatenate with self.

        Returns
        -------
        MultivariateFunctionalData
            The concatenation of self and data.

        Raises
        ------
        ValueError
            When all `fdata` do not have the same number of elements.

        """
        if len(set(data.n_functional for data in fdata)) > 1:
            raise ValueError(
                "The MultivariateFunctionalData must have the same number "
                "of elements."
            )

        n_functional = fdata[0].n_functional

        new = n_functional * [None]
        for idx in np.arange(n_functional):
            data_uni = [el.data[idx] for el in fdata]
            if isinstance(data_uni[0], DenseFunctionalData):
                new[idx] = DenseFunctionalData.concatenate(*data_uni)
            else:
                new[idx] = IrregularFunctionalData.concatenate(*data_uni)
        return MultivariateFunctionalData(new)

    ###########################################################################

    ###########################################################################
    # Magic methods
    def __init__(self, initlist: List[Type[FunctionalData]]) -> None:
        """Initialize MultivariateFunctionalData object."""
        FunctionalData._check_same_nobs(*initlist)
        self.data = initlist

    def __repr__(self) -> str:
        """Override print function."""
        return (
            f"Multivariate functional data object with {self.n_functional}"
            f" functions of {self.n_obs} observations."
        )

    def __getitem__(self, index: int) -> MultivariateFunctionalData:
        """Overrride getitem function, called when self[index].

        Parameters
        ----------
        index
            The observation(s) of the object to retrive.

        Returns
        -------
        MultivariateFunctionalData
            The selected observation(s) as MultivariateFunctionalData object.

        """
        return MultivariateFunctionalData([obs[index] for obs in self.data])

    ###########################################################################

    ###########################################################################
    # Properties
    @property
    def n_obs(self) -> int:
        """Get the number of observations of the functional data.

        Returns
        -------
        int
            Number of observations within the functional data.

        """
        return self.data[0].n_obs if len(self.data) > 0 else 0

    @property
    def n_functional(self) -> int:
        """Get the number of functional data with `self`.

        Returns
        -------
        int
            Number of functions in the list.

        """
        return len(self.data)

    @property
    def n_dimension(self) -> List[int]:
        """Get the number of input dimension of the functional data.

        Returns
        -------
        List[int]
            List containing the dimension of each component in the functional
            data.

        """
        return [fdata.n_dimension for fdata in self.data]

    @property
    def n_points(self) -> List[Dict[str, int]]:
        """Get the mean number of sampling points.

        Returns
        -------
        List[Union[Tuple[int, ...], Dict[int, Tuple[int, ...]]]]
            A list containing the number of sampling points along each axis
            for each function.

        """
        return [fdata.n_points for fdata in self.data]

    ###########################################################################

    ###########################################################################
    # List related functions
    def append(self, item: Type[FunctionalData]) -> None:
        """Add an item to `self`.

        Parameters
        ----------
        item: Type[FunctionalData]
            Item to add.

        """
        if len(self.data) == 0:
            self.data = [item]
        else:
            FunctionalData._check_same_nobs(*self.data, item)
            self.data.append(item)

    def extend(self, other: Iterable[Type[FunctionalData]]) -> None:
        """Extend the list of FunctionalData by appending from iterable."""
        super().extend(other)

    def insert(self, i: int, item: Type[FunctionalData]) -> None:
        """Insert an item `item` at a given position `i`."""
        super().insert(i, item)

    def remove(self, item: Type[FunctionalData]) -> None:
        """Remove the first item from `self` where value is `item`."""
        super().remove(item)

    def pop(self, i: int = -1) -> Type[FunctionalData]:
        """Remove the item at the given position in the list, and return it."""
        return super().pop(i)

    def clear(self) -> None:
        """Remove all items from the list."""
        super().clear()

    def reverse(self) -> None:
        """Reserve the elements of the list in place."""
        super().reverse()

    ###########################################################################

    ###########################################################################
    # Methods
    def to_basis(self, **kwargs) -> MultivariateFunctionalData:
        """Convert the data to basis format.

        This function transforms a MultivariateFunctionalData object into a
        MultivariateFunctionalData that contains BasisFunctionalData.

        Parameters
        ----------
        kwargs
            Other keyword arguments are passed to the functions
            :meth:`representation.functional_data.DenseFunctionalData` and
            :meth:`representation.functional_data.IrregularFunctionalData`.

        Returns
        -------
        MultivariateFunctionalData
            The expanded data.

        """
        data_list = []
        for data in self.data:
            if isinstance(data, BasisFunctionalData):
                data_list.append(data)
            else:
                data_basis = data.to_basis(**kwargs)
                data_list.append(data_basis)
        return MultivariateFunctionalData(data_list)

    def to_grid(self) -> MultivariateFunctionalData:
        """Convert the data to grid.

        Returns
        -------
        MultivariateFunctionalData
            The data in grid format.

        """
        data_list = []
        for data in self.data:
            if isinstance(data, GridFunctionalData):
                data_list.append(data)
            else:
                data_grid = data.to_grid()
                data_list.append(data_grid)
        return MultivariateFunctionalData(data_list)

    def to_long(self, reindex: bool = True) -> List[pd.DataFrame]:
        """Convert the data to long format.

        This function transform a MultivariateFunctionalData object into a list
        of pandas DataFrame. It uses the long format to represent each element
        of the MultivariateFunctionalData object as a dataframe. This is a
        helper function as it might be easier for some computation.

        Parameters
        ----------
        reindex
            Should the observations be reindexed.

        Returns
        -------
        List[pd.DataFrame]
            The data in a long format.

        Examples
        --------
        >>> argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 5])})
        >>> values = DenseValues(np.array([
        ...     [1, 2, 3, 4, 5],
        ...     [6, 7, 8, 9, 10],
        ...     [11, 12, 13, 14, 15]
        ... ]))
        >>> fdata_dense = DenseFunctionalData(argvals, values)

        >>> argvals = IrregularArgvals({
        ...     0: DenseArgvals({'input_dim_0': np.array([0, 1, 2, 3, 4])}),
        ...     1: DenseArgvals({'input_dim_0': np.array([0, 2, 4])}),
        ...     2: DenseArgvals({'input_dim_0': np.array([2, 4])})
        ... })
        >>> values = IrregularValues({
        ...     0: np.array([1, 2, 3, 4, 5]),
        ...     1: np.array([2, 5, 6]),
        ...     2: np.array([4, 7])
        ... })
        >>> fdata_irregular = IrregularFunctionalData(argvals, values)
        >>> fdata = MultivariateFunctionalData([fdata_dense, fdata_irregular])

        >>> fdata.to_long()
        [    input_dim_0  id  values
        0             1   0       1
        1             2   0       2
        2             3   0       3
        3             4   0       4
        4             5   0       5
        5             1   1       6
        6             2   1       7
        7             3   1       8
        8             4   1       9
        9             5   1      10
        10            1   2      11
        11            2   2      12
        12            3   2      13
        13            4   2      14
        14            5   2      15,
           input_dim_0  id  values
        0            0   0       5
        1            1   0       4
        2            2   0       3
        3            3   0       2
        4            4   0       1
        5            0   1       5
        6            2   1       3
        7            4   1       1
        8            2   2       5
        9            4   2       3]

        """
        return [fdata.to_long(reindex) for fdata in self.data]

    def noise_variance(self, order: int = 2) -> float:
        """Estimate the variance of the noise.

        This function estimates the variance of the noise. The noise is
        estimated for each individual curve using the methodology in [4]_. As
        the curves are assumed to be generated by the same process, the
        estimation of the variance of the noise is the mean over the set of
        curves.

        Parameters
        ----------
        order
            Order of the difference sequence. The order has to be between
            1 and 10. See [4]_ for more information.

        Returns
        -------
        float
            The estimation of the variance of the noise.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=100)
        >>> kl.add_noise(0.05)
        >>> kl.sparsify(0.5)
        >>> fdata = MultivariateFunctionalData([kl.noisy_data, kl.sparse_data])
        >>> fdata.noise_variance
        [0.051922438333740877, 0.006671248206782777]

        """
        return [fdata.noise_variance(order=order) for fdata in self.data]

    def smooth(
        self,
        points: DenseArgvals | None = None,
        method: str = "PS",
        bandwidth: float | None = None,
        penalty: float | None = None,
        **kwargs,
    ):
        """Smooth the data.

        This function smooths each curves individually. It fits a local
        smoother to the data (the argument ``degree`` controls the degree of
        the local fits). All the paraneters have to be passed as a list of the
        same length of the MultivariateFunctionalData.

        Parameters
        ----------
        points
            Points at which the curves are estimated. The default is None,
            meaning we use the argvals as estimation points.
        method
            The method to used for the smoothing. If 'PS', the method is
            P-splines [3]_. If 'LP', the method is local polynomials [7]_.
            Otherwise, it raises an error.
        bandwidth
            Strictly positive. Control the size of the associated neighborhood.
            If ``bandwidth == None``, it is assumed that the curves are twice
            differentiable and the bandwidth is set to :math:`n^{-1/5}` [6]_
            where :math:`n` is the number of sampling points per curve. Be
            careful with the results if the curves are not sampled on
            :math:`[0, 1]`.
        penalty
            Strictly positive. Penalty used in the P-splined fitting of the
            data.
        kwargs
            Other keyword arguments are passed to one of the following
            functions :meth:`DenseFunctionalData.smooth` an
            :meth:`IrregularFunctionalData.smooth`.


        Returns
        -------
        MultivariateFunctionalData
            Smoothed data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines', n_functions=5, random_state=42
        ... )
        >>> kl.new(n_obs=50)
        >>> kl.add_noise_and_sparsify(0.05, 0.5)

        >>> fdata_1 = kl.data
        >>> fdata_2 = kl.noisy_data
        >>> fdata = MultivariateFunctionalData([fdata_1, fdata_2])

        >>> points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        >>> fdata_smooth = fdata.smooth(
        ...     points=[points, points],
        ...     kernel_name=['epanechnikov', 'epanechnikov'],
        ...     bandwidth=[0.05, 0.1],
        ...     degree=[1, 2]
        ... )
        Multivariate functional data object with 2 functions of 50 observations

        """
        if points is None:
            points = self.n_functional * [None]

        bandwidth = kwargs.get("bandwidth", None)
        kernel_name = kwargs.get("kernel", None)
        degree = kwargs.get("degree", None)
        if kernel_name is None:
            kernel_name = self.n_functional * ["epanechnikov"]
        if bandwidth is None:
            bandwidth = self.n_functional * [None]
        if degree is None:
            degree = self.n_functional * [1]
        if (
            not isinstance(points, list)
            or not isinstance(kernel_name, list)
            or not isinstance(bandwidth, list)
            or not isinstance(degree, list)
        ):
            raise TypeError("Each parameter has to be a list.")
        if (
            len(points) != self.n_functional
            or len(kernel_name) != self.n_functional
            or len(bandwidth) != self.n_functional
            or len(degree) != self.n_functional
        ):
            raise ValueError(
                "Each parameter has to be a list of length " f"{self.n_functional}."
            )
        return MultivariateFunctionalData(
            [
                fdata.smooth(pp, method=method, **kwargs)
                for (fdata, pp) in zip(self.data, points)
            ]
        )

    def mean(
        self,
        points: List[DenseArgvals] | None = None,
        method_smoothing: str | None = None,
        **kwargs,
    ) -> MultivariateFunctionalData:
        """Compute an estimate of the mean.

        This function computes an estimate of the mean curve of a
        MultivariateFunctionalData object.

        Parameters
        ----------
        points
            Points at which the mean is estimated. The default is None,
            meaning we use the argvals as estimation points.
        method_smoothing
            The method to used for the smoothing. If 'None', no smoothing is
            performed. If 'PS', the method is P-splines [3]_. If 'LP', the
            method is local polynomials [7]_.
        kwargs
            Other keyword arguments are passed to the following function:
            :meth:`MultivariateFunctionalData.smooth`.

        Returns
        -------
        MultivariateFunctionalData
            An estimate of the mean as a MultivariateFunctionalData object.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines', n_functions=5, random_state=42
        ... )
        >>> kl.new(n_obs=50)
        >>> kl.add_noise_and_sparsify(0.05, 0.5)

        >>> fdata_1 = kl.data
        >>> fdata_2 = kl.noisy_data
        >>> fdata = MultivariateFunctionalData([fdata_1, fdata_2])

        >>> points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        >>> fdata.mean(points=points)
        Multivariate functional data object with 2 functions of 1 observations.

        """
        if points is None:
            points = self.n_functional * [None]
        if not isinstance(points, list):
            raise TypeError("`points` has to be a list.")
        if len(points) != self.n_functional:
            raise ValueError(
                f"`points` has to be a list of length {self.n_functional}."
            )
        self._mean = MultivariateFunctionalData(
            [
                fdata.mean(points=pp, method_smoothing=method_smoothing, **kwargs)
                for (fdata, pp) in zip(self.data, points)
            ]
        )
        return self._mean

    def center(
        self,
        mean: MultivariateFunctionalData | None = None,
        method_smoothing: str | None = None,
        **kwargs,
    ) -> MultivariateFunctionalData:
        """Center the data.

        Parameters
        ----------
        mean
            A precomputed mean as a MultivariateFunctionalData object.
        method_smoothing
            The method to used for the smoothing of the mean. If 'None', no
            smoothing is performed. If 'PS', the method is P-splines [3]_. If
            'LP', the method is local polynomials [7]_.
        kwargs
            Other keyword arguments are passed to one of the following
            functions :meth:`DenseFunctionalData.mean` (``mean=None``) and
            :meth:`DenseFunctionalData.smooth`.

        Returns
        -------
        MultivariateFunctionalData
            The centered version of the data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name=name, n_functions=n_functions, random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.add_noise_and_sparsify(0.05, 0.5)

        >>> fdata_1 = kl.data
        >>> fdata_2 = kl.sparse_data
        >>> fdata = MultivariateFunctionalData([fdata_1, fdata_2])
        >>> fdata.center(smooth=True)
        Functional data object with 10 observations on a 1-dimensional support.

        """
        if mean is None:
            return MultivariateFunctionalData(
                [
                    fdata.center(method_smoothing=method_smoothing, **kwargs)
                    for fdata in self.data
                ]
            )
        else:
            return MultivariateFunctionalData(
                [
                    fdata.center(mean=mean, method_smoothing=method_smoothing, **kwargs)
                    for (fdata, mean) in zip(self.data, mean.data)
                ]
            )

    def norm(
        self,
        squared: bool = False,
        method_integration: str = "trapz",
        use_argvals_stand: bool = False,
    ) -> npt.NDArray[np.float64]:
        r"""Norm of each observation of the data.

        For each observation in the data, it computes its norm defined in [2]_
        as

        .. math::
            \| X \| = \left\{\int_{\mathcal{T}} X(t)^2dt\right\}^{\frac12}.

        Parameters
        ----------
        squared
            If `True`, the function calculates the squared norm, otherwise it
            returns the norm.
        method_integration
            The method used to integrated.
        use_argvals_stand
            Use standardized argvals to compute the normalization of the data.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs,)
            The norm of each observations.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name=name, n_functions=n_functions, random_state=42
        ... )
        >>> kl.new(n_obs=4)
        >>> kl.add_noise_and_sparsify(0.05, 0.5)

        >>> fdata_1 = kl.data
        >>> fdata_2 = kl.sparse_data
        >>> fdata = MultivariateFunctionalData([fdata_1, fdata_2])
        >>> fdata.norm()
        array([1.05384959, 0.84700578, 1.37439764, 0.59235447])

        """
        norm_univariate = np.array(
            [
                fdata.norm(squared, method_integration, use_argvals_stand)
                for fdata in self.data
            ]
        )
        return np.sum(norm_univariate, axis=0)

    def normalize(self, **kwargs) -> MultivariateFunctionalData:
        r"""Normalize the data.

        The normalization is performed by divising each functional datum
        :math:`X` by its norm :math:`\| X \|`. It results in

        .. math::
            \widetilde{X} = \frac{X}{\| X \|}.

        Parameters
        ----------
        kwargs
            Other keyword arguments are passed to the following function
            :meth:`MultivariateFunctionalData.norm`.

        Returns
        -------
        MultivariateFunctionalData
            The normalized data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name=name, n_functions=n_functions, random_state=42
        ... )
        >>> kl.new(n_obs=4)
        >>> kl.add_noise_and_sparsify(0.05, 0.5)

        >>> fdata_1 = kl.data
        >>> fdata_2 = kl.sparse_data
        >>> fdata = MultivariateFunctionalData([fdata_1, fdata_2])
        >>> fdata.normalize()
        Functional data object with 10 observations on a 1-dimensional support.

        """
        norm_vector = self.norm(**kwargs)

        list_multivariate = []
        for obs, norm in zip(self, norm_vector):
            list_univariate = []
            for component in obs.data:
                list_univariate.append(component / norm)
            list_multivariate.append(MultivariateFunctionalData(list_univariate))
        return MultivariateFunctionalData.concatenate(*list_multivariate)

    def standardize(self, center: bool = True, **kwargs) -> MultivariateFunctionalData:
        r"""Standardize the data.

        The standardization is performed by first centering the data and then
        dividing by the standard deviation curve [2]_. It results in

        .. math::
            \widetilde{X}(t) = C(t, t)^{-\frac12}\{X(t) - \mu(t)\}, \quad
            t \in \mathcal{T}.

        Parameters
        ----------
        center
            Should the data be centered?
        kwargs
            Other keyword arguments are passed to the following function
            :meth:`MultivariateFunctionalData.center`,
            :meth:`DenseFunctionalData.standardize` and
            :meth:`IrregularFunctionalData.stansardize`.

        Returns
        -------
        MultivariateFunctionalData
            The standardized data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name=name, n_functions=n_functions, random_state=42
        ... )
        >>> kl.new(n_obs=4)
        >>> kl.add_noise_and_sparsify(0.05, 0.5)

        >>> fdata_1 = kl.data
        >>> fdata_2 = kl.sparse_data
        >>> fdata = MultivariateFunctionalData([fdata_1, fdata_2])
        >>> fdata.standardize()
        Functional data object with 10 observations on a 1-dimensional support.

        """
        if center:
            fdata = self.center(**kwargs)
        else:
            fdata = self

        list_multivariate = []
        for components in fdata.data:
            list_multivariate.append(components.standardize(center=False, **kwargs))
        return MultivariateFunctionalData(list_multivariate)

    def rescale(
        self,
        weights: npt.NDArray[np.float64] | None = None,
        method_integration: str = "trapz",
        method_smoothing: str = "LP",
        use_argvals_stand: bool = False,
        **kwargs,
    ) -> Tuple[MultivariateFunctionalData, npt.NDArray[np.float64]]:
        r"""Rescale the data.

        The normalization is performed by divising each functional datum by
        :math:`w_j = \int_{T} Var(X(t))dt`.

        Parameters
        ----------
        weights
            The weights used to normalize the data. If `weights = None`, the
            weights are estimated by integrating the variance function [5]_.
        method_integration
            The method used to integrated.
        use_argvals_stand
            Use standardized argvals to compute the normalization of the data.
        kwargs
            Keyword parameters for the smoothing of the observations.

        Returns
        -------
        Tuple[MultivariateFunctionalData, npt.NDArray[np.float64]]
            The normalized data.


        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name=name, n_functions=n_functions, random_state=42
        ... )
        >>> kl.new(n_obs=4)
        >>> kl.add_noise_and_sparsify(0.05, 0.5)

        >>> fdata_1 = kl.data
        >>> fdata_2 = kl.sparse_data
        >>> fdata = MultivariateFunctionalData([fdata_1, fdata_2])
        >>> fdata.normalize()
        (Multivariate functional data object with 2 functions of 4
        observations., array([0.20365764, 0.19388443]))

        """
        if weights is None:
            weights = np.zeros(self.n_functional)
        normalization = [
            fdata.rescale(
                weights=weight,
                method_integration=method_integration,
                method_smoothing=method_smoothing,
                use_argvals_stand=use_argvals_stand,
                **kwargs,
            )
            for (fdata, weight) in zip(self.data, weights)
        ]
        data_norm = [data for data, _ in normalization]
        weights = np.array([weight for _, weight in normalization])
        return MultivariateFunctionalData(data_norm), weights

    def inner_product(
        self,
        method_integration: str = "trapz",
        method_smoothing: str | None = None,
        noise_variance: npt.NDArray[np.float64] | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        r"""Compute the inner product matrix of the data.

        The inner product matrix is a ``n_obs`` by ``n_obs`` matrix where each
        entry is defined as

        .. math::
            \langle\langle x, y \rangle\rangle =
            \sum_{p = 1}^P \int_{\mathcal{T}_k} x^{(p)}(t)y^{(p)}(t)dt,
            t \in \mathcal{T},

        where :math:`\mathcal{T}` is a one- or multi-dimensional domain [1]_.

        Parameters
        ----------
        method_integration
            The method used to integrated.
        method_smoothing
            Should the mean be smoothed?
        noise_variance
            An estimation of the variance of the noise. If `None`, an
            estimation is computed using the methodology in [4]_.
        kwargs
            Other keyword arguments are passed to the following function
            :meth:`DenseFunctionalData.inner_product()` and
            :meth:`IrregularFunctionalData.inner_product()`.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_obs)
            Inner product matrix of the data.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name=name, n_functions=n_functions, random_state=42
        ... )
        >>> kl.new(n_obs=4)
        >>> kl.add_noise_and_sparsify(0.05, 0.5)

        >>> fdata_1 = kl.data
        >>> fdata_2 = kl.sparse_data
        >>> fdata = MultivariateFunctionalData([fdata_1, fdata_2])
        >>> fdata.inner_product(noise_variance=0)
        array([
            [ 0.39261306,  0.06899153, -0.14614219, -0.0836462 ],
            [ 0.06899153,  0.32580074, -0.4890299 ,  0.07577286],
            [-0.14614219, -0.4890299 ,  0.94953678, -0.09322892],
            [-0.0836462 ,  0.07577286, -0.09322892,  0.17157688]
        ])

        """
        if noise_variance is None:
            self._noise_variance = self.noise_variance(order=2)
        else:
            self._noise_variance = noise_variance
        return np.sum(
            [
                data.inner_product(
                    method_integration, method_smoothing, noise_variance, **kwargs
                )
                for (data, noise_variance) in zip(self.data, self._noise_variance)
            ],
            axis=0,
        )

    def covariance(
        self,
        points: List[DenseArgvals] | None = None,
        method_smoothing: str | None = None,
        **kwargs,
    ) -> MultivariateFunctionalData:
        """Compute an estimate of the covariance.

        This function computes an estimate of the covariance surface of a
        MultivariateFunctionalData object.

        Parameters
        ----------
        points
            Points at which the mean is estimated. The default is None,
            meaning we use the argvals as estimation points.
        method_smoothing
            Should the mean be smoothed?
        kwargs
            Other keyword arguments are passed to the following function
            :meth:`DenseFunctionalData.covariance()` and
            :meth:`IrregularFunctionalData.covariance()`.

        Returns
        -------
        MultivariateFunctionalData
            An estimate of the covariance as a two-dimensional
            MultivariateFunctionalData object with same argvals as `self`.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines', n_functions=5, random_state=42
        ... )
        >>> kl.new(n_obs=50)
        >>> kl.add_noise_and_sparsify(0.05, 0.5)

        >>> fdata_1 = kl.data
        >>> fdata_2 = kl.noisy_data
        >>> fdata = MultivariateFunctionalData([fdata_1, fdata_2])

        >>> points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        >>> fdata.covariance(points=[points, points])
        Multivariate functional data object with 2 functions of 1 observations.

        """
        if points is None:
            points = self.n_functional * [None]
        if not isinstance(points, list):
            raise TypeError("`points` has to be a list.")
        if len(points) != self.n_functional:
            raise ValueError(
                f"`points` has to be a list of length {self.n_functional}."
            )
        self._covariance = MultivariateFunctionalData(
            [
                fdata.covariance(pp, method_smoothing=method_smoothing, **kwargs)
                for fdata, pp in zip(self.data, points)
            ]
        )
        self._noise_variance_cov = [fdata._noise_variance_cov for fdata in self.data]
        return self._covariance

    ###########################################################################
