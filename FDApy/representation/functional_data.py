#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Module for the definition of FunctionalData types.

This modules is used to defined different types of functional data. The
different types are: Univariate Functional Data, Irregular Functional Data and
Multivariate Functional Data.
"""
from __future__ import annotations

import itertools
import numpy as np
import numpy.typing as npt
import pygam
import warnings

from abc import ABC, abstractmethod
from collections import UserList
from typing import (
    cast, Any, Dict, Iterable, Iterator, Optional, List,
    Tuple, TYPE_CHECKING, Union
)

from sklearn.metrics import pairwise_distances

from ..preprocessing.smoothing.local_polynomial import LocalPolynomial
from ..preprocessing.smoothing.smoothing_splines import SmoothingSpline
from ..misc.utils import _get_dict_dimension, _get_obs_shape
from ..misc.utils import _inner_product, _inner_product_2d
from ..misc.utils import _integrate, _integration_weights
from ..misc.utils import _normalization, _outer

if TYPE_CHECKING:
    from ..representation.basis import Basis

DenseArgvals = Dict[str, npt.NDArray[np.float64]]
DenseValues = npt.NDArray[np.float64]
IrregArgvals = Dict[str, Dict[int, npt.NDArray[np.float64]]]
IrregValues = Dict[int, npt.NDArray[np.float64]]


###############################################################################
# Class FunctionalData
class FunctionalData(ABC):
    """Metaclass for the definition of diverse functional data objects.

    Parameters
    ----------
    argvals: list
    values: list
    category: str, {'univariate', 'irregular', 'multivariate'}

    """

    @staticmethod
    def _check_same_type(
        argv1: FunctionalData,
        argv2: FunctionalData
    ) -> None:
        """Raise an error if `argv1` and `argv2` have different type.

        Parameters
        ----------
        argv1: FunctionalData
            An object.
        argv2: FunctionalData
            An object.

        Raises
        ------
        TypeError
            When `argv1` and `argv2` do not have the same type.

        """
        if not isinstance(argv2, type(argv1)):
            raise TypeError(f"{argv1} and {argv2} do not have the same type.")

    @staticmethod
    def _check_same_nobs(
        *argv: FunctionalData
    ) -> None:
        """Raise an arror if elements in argv have different number of obs."""
        n_obs = set(obj.n_obs for obj in argv)
        if len(n_obs) > 1:
            raise ValueError(
                "Elements do not have the same number of observations."
            )

    @staticmethod
    def _check_same_ndim(
        argv1: FunctionalData,
        argv2: FunctionalData
    ) -> None:
        """Raise an error if `argv1` and `argv2` have different dim."""
        if argv1.n_dim != argv2.n_dim:
            raise ValueError(
                f"{argv1} and {argv2} do not have the same number"
                " of dimensions."
            )

    @staticmethod
    @abstractmethod
    def _check_argvals_values(
        argvals: Any,
        values: Any
    ) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _perform_computation(
        fdata1: Any,
        fdata2: Any,
        func: np.ufunc
    ) -> FunctionalData:
        pass

    def __init__(
        self,
        argvals: Union[DenseArgvals, IrregArgvals],
        values: Union[DenseValues, IrregValues],
        category: str
    ) -> None:
        """Initialize FunctionalData object."""
        super().__init__()
        self.argvals = argvals
        self.values = values
        self.category = category

    def __repr__(self) -> str:
        """Override print function."""
        return (
            f"{self.category.capitalize()} functional data object with"
            f" {self.n_obs} observations on a {self.n_dim}-dimensional"
            " support."
        )

    @abstractmethod
    def __getitem__(
        self,
        index: int
    ) -> FunctionalData:
        """Override getitem function, called when self[index]."""
        pass

    def __add__(
        self,
        obj: FunctionalData
    ) -> FunctionalData:
        """Override add function."""
        return self._perform_computation(self, obj, np.add)

    def __sub__(
        self,
        obj: FunctionalData
    ) -> FunctionalData:
        """Override sub function."""
        return self._perform_computation(self, obj, np.subtract)

    def __mul__(
        self,
        obj: FunctionalData
    ) -> FunctionalData:
        """Overrude mul function."""
        return self._perform_computation(self, obj, np.multiply)

    def __rmul__(
        self,
        obj: FunctionalData
    ) -> FunctionalData:
        """Override rmul function."""
        return self * obj

    def __truediv__(
        self,
        obj: FunctionalData
    ) -> FunctionalData:
        """Override truediv function."""
        return self._perform_computation(self, obj, np.divide)

    def __floordiv__(
        self,
        obj: FunctionalData
    ) -> FunctionalData:
        """Override floordiv function."""
        return self / obj

    @property
    def argvals(
        self
    ) -> Any:
        """Getter for argvals."""
        return self._argvals  # type: ignore

    @argvals.setter
    def argvals(
        self,
        new_argvals: Union[DenseArgvals, IrregArgvals]
    ) -> None:
        if hasattr(self, 'values'):
            self._check_argvals_values(new_argvals, self.values)
        self._argvals = new_argvals

    @property
    def argvals_stand(
        self
    ) -> Union[DenseArgvals, IrregArgvals]:
        """Getter for argvals_stand."""
        return self._argvals_stand

    @argvals_stand.setter
    def argvals_stand(
        self,
        new_argvals_stand: Union[DenseArgvals, IrregArgvals]
    ) -> None:
        self._argvals_stand = new_argvals_stand

    @property
    def values(
        self
    ) -> Any:
        """Getter for values."""
        return self._values

    @values.setter
    def values(
        self,
        new_values: Union[DenseValues, IrregValues]
    ) -> None:
        if hasattr(self, 'argvals'):
            self._check_argvals_values(self.argvals, new_values)
        self._values = new_values

    @property
    def category(self) -> str:
        """Getter for category."""
        return self._category

    @category.setter
    def category(self, new_category: str) -> None:
        self._category = new_category

    @property
    def n_obs(self) -> int:
        """Get the number of observations of the functional data.

        Returns
        -------
        n_obs: int
            Number of observations within the functional data.

        """
        return len(self.values)

    @property
    @abstractmethod
    def range_obs(self) -> Tuple[float, float]:
        """Get the range of the observations of the object."""
        pass

    @property
    def n_dim(self) -> int:
        """Get the number of input dimension of the functional data.

        Returns
        -------
        n_dim: int
            Number of input dimension with the functional data.

        """
        return len(self.argvals)

    @property
    @abstractmethod
    def n_points(self) -> Dict[str, int]:
        """Get the mean number of sampling points."""
        pass

    @property
    @abstractmethod
    def range_dim(self) -> Dict[str, Tuple[int, int]]:
        """Range of the `argvals` for each of the dimension."""
        pass

    @property
    @abstractmethod
    def shape(self) -> Dict[str, int]:
        """Shape of the data for each dimension."""
        pass

    @abstractmethod
    def is_compatible(
        self,
        fdata: FunctionalData
    ) -> bool:
        """Check if `fdata` is compatible with `self`."""
        FunctionalData._check_same_type(self, fdata)
        FunctionalData._check_same_nobs(self, fdata)
        FunctionalData._check_same_ndim(self, fdata)
        return True

    @abstractmethod
    def to_basis(
        self,
        basis: 'Basis'
    ) -> None:
        """Expand the FunctionalData into a basis."""
        pass

    @abstractmethod
    def mean(
        self,
        smooth: Optional[str] = None,
        **kwargs: Any
    ) -> FunctionalData:
        """Compute an estimate of the mean."""
        pass

    @abstractmethod
    def covariance(
        self,
        mean: Optional[FunctionalData] = None,
        smooth: Optional[str] = None,
        **kwargs: Any
    ) -> FunctionalData:
        """Compute an estimate of the covariance."""
        pass

    @abstractmethod
    def inner_product(
        self
    ) -> npt.NDArray:
        """Compute an estimate of the inner product matrix."""
        pass

    @abstractmethod
    def smooth(
        self,
        points: npt.NDArray[np.float64],
        neighborhood: npt.NDArray[np.float64],
        points_estim: Optional[npt.NDArray[np.float64]] = None,
        degree: int = 0,
        kernel: str = "epanechnikov",
        bandwidth: Optional[List[float]] = None
    ) -> FunctionalData:
        """Smooth the data."""
        pass


###############################################################################
# Class DenseFunctionalData
class DenseFunctionalData(FunctionalData):
    r"""A class for defining Dense Functional Data.

    A class used to define dense functional data. We denote by :math:`n`, the
    number of observations and by :math:`p`, the number of input dimensions.
    Here, we are in the case of univariate functional data, and so the output
    dimension will be :math:`\mathbb{R}`.

    Parameters
    ----------
    argvals: dict
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. The shape of the :math:`j`th
        dimension is :math:`(m_j,)` for :math:`0 \leq j \leq p`.
    values: np.ndarray
        The values of the functional data. The shape of the array is
        :math:`(n, m_1, \dots, m_p)`.

    Examples
    --------
    argvals = {
        'input_dim_0': np.array([1, 2, 3, 4]),
        'input_dim_1': np.array([5, 6, 7])
    }
    values = np.array(
        [
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
            [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]],
            [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]],
            [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]],
            [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]
        ]
    )
    DenseFunctionalData(argvals, values)

    """

    @staticmethod
    def _check_argvals_equality(
        argv1: DenseArgvals,
        argv2: DenseArgvals
    ) -> None:
        """Check if `argv1` and `argv2` are equal.

        Parameters
        ----------
        argv1 : DenseArgvals
            The first set of argument values.
        argv2 : DenseArgvals
            The second set of argument values.

        Raises
        ------
        ValueError
            If `argv1` and `argv2` do not have the same sampling points.

        """
        argvs_equal = all(
            np.array_equal(argv1[key], argv2[key]) for key in argv1
        )
        if not argvs_equal:
            raise ValueError(
                f"{argv1} and {argv2} do not have the same sampling points."
            )

    @staticmethod
    def _check_argvals_values(
        argvals: DenseArgvals,
        values: DenseValues
    ) -> None:
        """Raise an error in case of dimension conflicts between the arguments.

        An error is raised when `argvals` (a dictionary) and `values`
        (a np.ndarray) do not have coherent common dimensions. The first
        dimension of `values` is assumed to represented the number of
        observation.

        Parameters
        ----------
        argvals: DenseArgvals
            A dictionary with key as string and value as numpy array.
        values: DenseValues
            A numpy array

        Raises
        ------
        ValueError
            When `argvals` and `values` do not have coherent common
            dimensions. The first dimension of `argvals` is assumed to
            represented the number of observations.

        """
        dim_dict = _get_dict_dimension(argvals)
        dim_array = values.shape[1:]
        if dim_dict != dim_array:
            raise ValueError(
                f"{argvals} and {values} do not have coherent dimension."
            )

    @staticmethod
    def _perform_computation(
        fdata1: DenseFunctionalData,
        fdata2: DenseFunctionalData,
        func: np.ufunc
    ) -> DenseFunctionalData:
        """Perform computation defined by `func`."""
        if fdata1.is_compatible(fdata2):
            new_values = func(fdata1.values, fdata2.values)
        return DenseFunctionalData(fdata1.argvals, new_values)

    def __init__(
        self,
        argvals: DenseArgvals,
        values: DenseValues
    ) -> None:
        """Initialize UnivariateFunctionalData object."""
        super().__init__(argvals, values, 'univariate')

    def __getitem__(
        self,
        index: int
    ) -> DenseFunctionalData:
        """Overrride getitem function, called when self[index].

        Parameters
        ----------
        index: int
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

    @property
    def argvals(self) -> DenseArgvals:
        """Getter for argvals."""
        return cast(DenseArgvals, super().argvals)

    @argvals.setter
    def argvals(
        self,
        new_argvals: DenseArgvals
    ) -> None:
        """Setter for argvals."""
        self._argvals = new_argvals
        argvals_stand = {}
        for dim, points in new_argvals.items():
            argvals_stand[dim] = _normalization(points)
        self._argvals_stand = argvals_stand

    @property
    def values(
        self
    ) -> DenseValues:
        """Getter for values."""
        return cast(DenseValues, self._values)

    @values.setter
    def values(
        self,
        new_values: DenseValues
    ) -> None:
        """Setter for values."""
        if hasattr(self, 'argvals'):
            self._check_argvals_values(self.argvals, new_values)
        self._values = new_values

    @property
    def range_obs(self) -> Tuple[float, float]:
        """Get the range of the observations of the object.

        Returns
        -------
        tuple
            Tuple containing the mimimum and maximum values taken by all the
            observations for the object.

        """
        return np.min(self.values), np.max(self.values)

    @property
    def n_points(self) -> Dict[str, int]:
        """Get the mean number of sampling points.

        Returns
        -------
        dict
            A dictionary with the same shape than argvals with the number of
            sampling points along each axis.

        Notes
        -----
        For DenseFunctionalData, this function is equivalent to shape().

        """
        return {idx: len(points) for idx, points in self.argvals.items()}

    @property
    def range_dim(self) -> Dict[str, Tuple[int, int]]:
        """Get the range of the `argvals` for each of the dimension.

        Returns
        -------
        dict
            Dictionary containing the range of the argvals for each of the
            input dimension.

        """
        return {
            idx: (min(argval), max(argval))
            for idx, argval in self.argvals.items()
        }

    @property
    def shape(self) -> Dict[str, int]:
        r"""Get the shape of the data for each dimension.

        Returns
        -------
        dict
            Dictionary containing the number of points for each of the
            dimension. It corresponds to :math:`m_j` for
            :math:`0 \leq j \leq p`.

        Notes
        -----
        For DenseFunctionalData, this function is equivalent to n_points().

        """
        return {idx: len(points) for idx, points in self.argvals.items()}

    def as_irregular(self) -> IrregularFunctionalData:
        """Convert `self` from Dense to Irregular functional data.

        Coerce a DenseFunctionalData object into an IrregularFunctionalData
        object.

        Returns
        -------
        obj: IrregularFunctionalData
            An object of the class IrregularFunctionalData

        """
        new_argvals: IrregArgvals = dict.fromkeys(self.argvals.keys(), {})
        for dim in new_argvals.keys():
            temp = {}
            for idx in range(self.n_obs):
                temp[idx] = self.argvals[dim]
            new_argvals[dim] = temp

        new_values = {}
        for idx in range(self.n_obs):
            new_values[idx] = self.values[idx]

        return IrregularFunctionalData(new_argvals, new_values)

    def is_compatible(
        self,
        fdata: FunctionalData
    ) -> bool:
        """Check if `fdata` is compatible with `self`.

        Two DenseFunctionalData object are said to be compatible if they
        have the same number of observations and dimensions. Moreover, they
        must have (strictly) the same sampling points.

        Parameters
        ----------
        fdata: DenseFunctionalData object
            The object to compare with `self`.

        Returns
        -------
        True
            If the objects are compatible, otherwise an error is raised before
            the return statement.

        """
        super().is_compatible(fdata)
        DenseFunctionalData._check_argvals_equality(
            self.argvals, fdata.argvals
        )
        return True

    def to_basis(
        self,
        basis: Basis
    ) -> None:
        """Convert to basis

        Parameters
        ----------
        basis: Basis

        """
        xtx = np.linalg.inv(np.matmul(basis.values, basis.values.T))
        xty = np.matmul(basis.values, self.values.T)
        self.basis = basis
        self.coefs = np.matmul(xtx, xty).T

    def mean(
        self,
        smooth: Optional[str] = None,
        **kwargs: Any
    ) -> DenseFunctionalData:
        """Compute an estimate of the mean.

        Parameters
        ----------
        smooth: str, default=None
            Name of the smoothing method to use. Currently, not implemented.

        Keyword Args
        ------------
        kernel_name: str, default='epanechnikov'
            Name of the kernel used for local polynomial smoothing.
        degree: int, default=1
            Degree used for local polynomial smoothing.
        bandwidth: float, default=1
            Bandwidth used for local polynomial smoothing.
        n_basis: int, default=10
            Number of splines basis used for GAM smoothing.

        Returns
        -------
        obj: DenseFunctionalData object
            An estimate of the mean as a DenseFunctionalData object with the
            same argvals as `self` and one observation.

        """
        mean_estim = self.values.mean(axis=0)

        if smooth is not None:
            argvals = self.argvals['input_dim_0']
            if self.n_dim > 1:
                raise ValueError('Only one dimensional data can be smoothed.')
            if smooth == 'LocalLinear':
                p = self.n_points['input_dim_0']
                points = kwargs.get('points', 0.5)
                neigh = kwargs.get(
                    'neighborhood',
                    np.int32(p * np.exp(-(np.log(np.log(p)))**2))
                )
                data_smooth = self.smooth(points=points,
                                          neighborhood=neigh)
                mean_estim = data_smooth.values.mean(axis=0)
            elif smooth == 'GAM':
                n_basis = kwargs.get('n_basis', 10)
                argvals = self.argvals['input_dim_0']
                mean_estim = pygam.LinearGAM(pygam.s(0, n_splines=n_basis)).\
                    fit(argvals, mean_estim).\
                    predict(argvals)
            elif smooth == 'SmoothingSpline':
                ss = SmoothingSpline()
                mean_estim = ss.fit_predict(argvals, mean_estim)
            else:
                raise NotImplementedError('Smoothing method not implemented.')
        return DenseFunctionalData(self.argvals, mean_estim[np.newaxis])

    def covariance(
        self,
        mean: Optional[FunctionalData] = None,
        smooth: Optional[str] = None,
        **kwargs: Any
    ) -> DenseFunctionalData:
        """Compute an estimate of the covariance.

        Parameters
        ----------
        smooth: str, default=None
            Name of the smoothing method to use. Currently, not implemented.
        mean: DenseFunctionalData, default=None
            An estimate of the mean of self. If None, an estimate is computed.

        Returns
        -------
        obj: DenseFunctionalData object
            An estimate of the covariance as a two-dimensional
            DenseFunctionalData object with same argvals as `self`.

        Keyword Args
        ------------
        kernel_name: str, default='epanechnikov'
            Name of the kernel used for local polynomial smoothing.
        degree: int, default=1
            Degree used for local polynomial smoothing.
        bandwidth: float, default=1
            Bandwidth used for local polynomial smoothing.
        n_basis: int, default=10
            Number of splines basis used for GAM smoothing.

        References
        ----------
        * Yao, Müller and Wang (2005), Functional Data Analysis for Sparse
        Longitudinal Data,
        Journal of the American Statistical Association, Vol. 100, No. 470
        * Staniswalis, J. G., and Lee, J. J. (1998), “Nonparametric Regression
        Analysis of Longitudinal Data,” Journal of the American Statistical
        Association, 93, 1403-1418.

        """
        if self.n_dim > 1:
            raise ValueError('Only one dimensional functional data are'
                             ' supported')

        p = self.n_points['input_dim_0']
        argvals = self.argvals['input_dim_0']
        if mean is None:
            mean = self.mean(smooth)
        data = self.values - mean.values
        cov = np.dot(data.T, data) / (self.n_obs - 1)
        cov_diag = np.copy(np.diag(cov))

        if smooth is not None:
            # Remove covariance diagonale because of measurement errors.
            np.fill_diagonal(cov, np.nan)
            cov = cov[~np.isnan(cov)]

            # Define train vector
            train_ = np.vstack((
                np.repeat(argvals, repeats=len(argvals)),
                np.tile(argvals, reps=len(argvals)))
            )

            train = train_[:, train_[0, :] != train_[1, :]]

            if smooth == 'LocalLinear':
                points = kwargs.get('points', 0.5)
                neigh = kwargs.get(
                    'neighborhood',
                    np.int32(p * np.exp(-(np.log(np.log(p)))**2))
                )
                data_smooth = self.smooth(points=points,
                                          neighborhood=neigh)
                data = data_smooth.values - mean.values
                cov = np.dot(data.T, data) / (self.n_obs - 1)
            elif smooth == 'GAM':
                n_basis = kwargs.get('n_basis', 10)

                cov = pygam.LinearGAM(pygam.te(0, 1, n_splines=n_basis)).\
                    fit(np.transpose(train), cov).\
                    predict(np.transpose(train_)).\
                    reshape((len(argvals), len(argvals)))
            else:
                raise NotImplementedError('Smoothing method not implemented.')

        # Ensure the covariance is symmetric.
        cov = (cov + cov.T) / 2

        # Smoothing the diagonal of the covariance (Yao, Müller and Wang, 2005)
        lp = LocalPolynomial(kernel_name=kwargs.get('kernel_name', 'gaussian'),
                             bandwidth=kwargs.get('bandwidth', 1),
                             degree=kwargs.get('degree', 1))
        var_hat = lp.fit_predict(argvals, cov_diag, argvals)
        # Estimate noise variance (Staniswalis and Lee, 1998)
        ll = argvals[len(argvals) - 1] - argvals[0]
        lower = np.sum(~(argvals >= (argvals[0] + 0.25 * ll)))
        upper = np.sum((argvals <= (argvals[len(argvals) - 1] - 0.25 * ll)))
        weights = _integration_weights(argvals[lower:upper], method='trapz')
        nume = np.dot(weights, (var_hat - cov_diag)[lower:upper])
        self.var_noise = np.maximum(nume / argvals[upper] - argvals[lower], 0)

        new_argvals = {'input_dim_0': argvals, 'input_dim_1': argvals}
        return DenseFunctionalData(new_argvals, cov[np.newaxis])

    def inner_product(
        self
    ) -> npt.NDArray:
        r"""Compute the inner product matrix of the data.

        The inner product matrix is a ``n_obs`` by ``n_obs`` matrix where each
        entry is defined as

        .. math::
            \langle x, y \rangle = \int_{\mathcal{T}} x(t)y(t)dt,
            t \in \mathcal{T},

        where :math:`\mathcal{T}` is a one- or multi-dimensional domain.

        Returns
        -------
        np.array, shape=(n_obs, n_obs)
            Inner product matrix of the data.

        Examples
        --------
        # For one-dimensional functional data
        argvals = {'input_dim_0': np.array([0., 0.25, 0.5 , 0.75])}
        values = np.array(
            [
                [ 2.48466259, -3.38397716, -1.2367073 , -1.85052901],
                [ 1.44853118,  0.67716255,  1.79711043,  4.76950236],
                [-5.13173463,  0.35830122,  0.56648942, -0.20965252]
            ]
        )
        data = DenseFunctionalData(argvals, values)
        data.inner_product()
        > array(
        >     [
        >         [ 4.44493731, -1.78187445, -2.02359881],
        >         [-1.78187445,  4.02783817, -0.73900893],
        >         [-2.02359881, -0.73900893,  3.40965432]
        >     ]
        > )

        # For two-dimensional functional data
        argvals = {
            'input_dim_0': np.array([0.  , 0.25, 0.5 , 0.75]),
            'input_dim_1': np.array([0.  , 0.25, 0.5 , 0.75])
        }
        values = np.array(
            [
                [
                    [  6.30864764, -18.37912204,   6.15515232,  29.8027036 ],
                    [ -6.076622  , -15.48586803, -11.39997792,   8.40599319],
                    [-20.4094798 ,  -1.3872093 ,  -0.59922597,  -6.42013363],
                    [  5.78626375,  -1.83874696,  -0.87225549,   2.75000303]
                ],
                [
                    [ -4.83576968,  18.85512513, -18.73086523,  15.1511348 ],
                    [-24.41254888,  12.37333951,  28.85176939,  16.41806885],
                    [-10.02681278,  14.76500118,   1.83114017,  -2.78985647],
                    [  4.29268032,   8.1781319 ,  30.10132687,  -0.72828334]
                ],
                [
                    [ -5.85921132,   1.85573561,  -5.11291405, -12.89441767],
                    [ -4.79384081,  -0.93863074,  18.81909033,   4.55041973],
                    [-13.27810529,  28.08961819, -13.79482673,  35.25677906],
                    [  9.10058173, -16.43979436, -11.88561292,  -5.86481318]
                ]
            ]
        )
        data = DenseFunctionalData(argvals, values)
        data.inner_product()
        > array(
        >     [
        >         [ 67.93133466, -26.76503879, -17.70996479],
        >         [-26.76503879, 162.59040715,  51.40230074],
        >         [-17.70996479,  51.40230074, 147.86839738]
        >     ]
        > )
        """
        # Get parameters
        n_obs = self.n_obs
        if self.n_dim == 1:
            inner_func = _inner_product
            axis = self.argvals['input_dim_0']
            params = {'axis': axis}
        elif self.n_dim == 2:
            inner_func = _inner_product_2d
            primary_axis = self.argvals['input_dim_0']
            secondary_axis = self.argvals['input_dim_1']
            params = {
                'primary_axis': primary_axis,
                'secondary_axis': secondary_axis
            }
        else:
            raise ValueError(
                'The data dimension is not correct.'
            )

        inner_mat = np.zeros((n_obs, n_obs))
        for (i, j) in itertools.product(np.arange(n_obs), repeat=2):
            if i <= j:
                inner_mat[i, j] = inner_func(
                    self.values[i],
                    self.values[j],
                    **params
                )
        inner_mat = inner_mat + inner_mat.T
        np.fill_diagonal(inner_mat, np.diag(inner_mat) / 2)
        return inner_mat

    def smooth(
        self,
        points: npt.NDArray[np.float64],
        neighborhood: npt.NDArray[np.float64],
        points_estim: Optional[npt.NDArray[np.float64]] = None,
        degree: int = 0,
        kernel: str = "epanechnikov",
        bandwidth: Optional[List[float]] = None
    ) -> DenseFunctionalData:
        """Smooth the data.

        Parameters
        ----------
        points: np.array
            Points at which the Bandwidth is estimated.
        neighborhood: np.array
            Neighborhood considered for each each points. Should have the same
            shape than points.
        points_estim: np.array, default=None
            Points at which the curves are estimated. The default is None,
            meaning we use the argvals as estimation points.
        degree: int, default=2
            Degree for the local polynomial smoothing.
        kernel: str, default='epanechnikov'
            The name of the kernel to use.
        bandwidth: Bandwidth, default=None
            An instance of Bandwidth for the smoothing.

        Returns
        -------
        obj: DenseFunctionalData
            A smoothed version of the data.

        Notes
        -----
        Only, one dimensional IrregularFunctionalData can be smoothed.

        """
        if self.n_dim != 1:
            raise NotImplementedError(
                'Only one dimensional data can be smoothed.'
            )

        data = self.as_irregular()
        data_smooth = data.smooth(points, neighborhood,
                                  points_estim=points_estim,
                                  degree=degree,
                                  kernel=kernel,
                                  bandwidth=bandwidth)
        return data_smooth.as_dense()

    def pairwise_distance(
        self,
        metric: str = 'euclidean'
    ) -> npt.NDArray[np.float64]:
        """Compute the pairwise distance between the data.

        Parameters
        ----------
        metric: str, default='euclidean'
            The metric to use when calculating distance between instances in a
            functional data object.

        Returns
        -------
        D: np.ndarray, shape=(n_obs, n_obs)
            A distance matrix D such that D_{i, j} is the distance between the
            ith and jth observations of the functional data object,

        """
        if self.n_dim > 1:
            raise NotImplementedError('The distance computation is not'
                                      ' implemented for data with dimension'
                                      ' greater than 1.')
        return cast(
            npt.NDArray[np.float64],
            pairwise_distances(self.values, metric=metric)
        )

    def concatenate(
        self,
        data: DenseFunctionalData
    ) -> DenseFunctionalData:
        """Concatenate two DenseFunctionalData.

        Parameters
        ----------
        data: DenseFunctionalData
            The data to concatenate with self.

        Returns
        -------
        res: DenseFunctionalData
            The concatenation of self and data.

        """
        return cast(DenseFunctionalData, _concatenate([self, data]))

    def normalize(
        self,
        use_argvals_stand: bool = False
    ) -> Tuple[DenseFunctionalData, float]:
        r"""Normalize the data.

        The normalization is performed by divising each functional datum by
        :math:`w_j = \int_{T} Var(X(t))dt`.

        Parameters
        ----------
        use_argvals_stand: bool, default=False
            Use standardized argvals to compute the normalization of the data.

        Returns
        -------
        res: DenseFunctionalData
            The normalized data.

        Todo
        ----
        - Add other normalization schames
        - Add the possibility to normalize multidemsional data

        References
        ----------
        Happ and Greven, Multivariate Functional Principal Component Analysis
        for Data Observed on Different (Dimensional Domains), Journal of the
        American Statistical Association.
        """
        if self.n_dim > 1:
            raise ValueError(
                "Normalization can only be performed on one dimensional data"
            )

        if use_argvals_stand:
            argvals = self.argvals_stand['input_dim_0']
        else:
            argvals = self.argvals['input_dim_0']
            weights = _integrate(argvals, np.var(self.values, axis=0))
        new_values = self.values / weights
        return DenseFunctionalData(self.argvals, new_values), weights


###############################################################################
# Class IrregularFunctionalData

class IrregularFunctionalData(FunctionalData):
    r"""A class for defining Irregular Functional Data.

    Parameters
    ----------
    argvals: dict
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. Then, each dimension is a
        dictionary where entries are the different observations. So, the
        observation :math:`i` for the dimension :math:`j` is a `np.ndarray`
        with shape :math:`(m^i_j,)` for :math:`0 \leq i \leq n` and
        :math:`0 \leq j \leq p`.
    values: dict
        The values of the functional data. Each entry of the dictionary is an
        observation of the process. And, an observation is represented by a
        `np.ndarray` of shape :math:`(n, m_1, \dots, m_p)`. It should not
        contain any missing values.

    Examples
    --------
    argvals = {
        'input_dim_0': {
            0: np.array([1, 2, 3, 4]),
            1: np.array([2, 4])
        },
        'input_dim_1': {
            0: np.array([5, 6, 7]),
            1: np.array([1, 2, 3])
        }
    }
    values = {
        0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
        1: np.array([[1, 2, 3], [1, 2, 3]])
    }
    IrregularFunctionalData(argvals, values)

    """

    @staticmethod
    def _check_argvals_length(
        argv: IrregArgvals
    ) -> None:
        """Raise an error if all elements of `argv` do not have equal length.

        Parameters
        ----------
        argv: IrregArgvals
            A nested dictionary with key as string and value as dictionary with
            key as integer and value as numpy array.

        Raises
        ------
        ValueError
            When the number of observations is different across the dimensions.

        """
        lengths = [len(obj) for obj in argv.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                "The number of observations is different across the dimensions"
            )

    @staticmethod
    def _check_argvals_equality(
        argv1: IrregArgvals,
        argv2: IrregArgvals
    ) -> None:
        """Check if `argv1` and `argv2` are equal.

        Parameters
        ----------
        argv1 : IrregArgvals
            The first set of argument values.
        argv2 : IrregArgvals
            The second set of argument values.

        Raises
        ------
        ValueError
            If `argv1` and `argv2` do not have the same sampling points.

        """
        temp = all([
            all(
                np.array_equal(points1[key], points2[key])
                for key in points1
            ) for points1, points2 in zip(argv1.values(), argv2.values())
        ])
        if not temp:
            raise ValueError(
                f"{argv1} and {argv2} do not have the same sampling points."
            )

    @staticmethod
    def _check_argvals_values(
        argvals: IrregArgvals,
        values: IrregValues
    ) -> None:
        """Raise an error in case of dimension conflicts between the arguments.

        Parameters
        ----------
        argvals: IrregArgvals
            A nested dictionary with key as string and value as dictionary with
            key as integer and value as numpy array.
        values: IrregValues
            A dictionary with key as integer and value as numpy array.

        Raises
        ------
        ValueError
            When `argvals` and `values` do not have coherent common dimensions.

        """
        has_obs_shape = [
            obs.shape == _get_obs_shape(argvals, idx)
            for idx, obs in values.items()
        ]
        if not np.all(has_obs_shape):
            raise ValueError(
                f"{argvals} and {values} do not have coherent dimension."
            )

    @staticmethod
    def _perform_computation(
        fdata1: IrregularFunctionalData,
        fdata2: IrregularFunctionalData,
        func: np.ufunc
    ) -> IrregularFunctionalData:
        """Perform computation defined by `func`."""
        if fdata1.is_compatible(fdata2):
            new_values = {}
            for (idx, obs1), (_, obs2) in zip(fdata1.values.items(),
                                              fdata2.values.items()):
                new_values[idx] = func(obs1, obs2)
        return IrregularFunctionalData(fdata1.argvals, new_values)

    def __init__(
        self,
        argvals: IrregArgvals,
        values: IrregValues
    ) -> None:
        """Initialize IrregularFunctionalData object."""
        super().__init__(argvals, values, 'irregular')

    def __getitem__(
        self,
        index: int
    ) -> IrregularFunctionalData:
        """Overrride getitem function, called when self[index].

        Parameters
        ----------
        index: int
            The observation(s) of the object to retrive.

        Returns
        -------
        IrregularFunctionalData
            The selected observation(s) as IrregularFunctionalData object.

        """
        argvals: IrregArgvals = {}
        if isinstance(index, slice):
            indices = index.indices(self.n_obs)
            for idx, dim in self.argvals.items():
                argvals[idx] = {i: dim.get(i) for i in range(*indices)}
            values = {i: self.values.get(i) for i in range(*indices)}
        else:
            argvals = {
                idx: {
                    index: cast(npt.NDArray[np.float64], points.get(index))
                } for idx, points in self.argvals.items()
            }
            values = {index: self.values.get(index)}
        return IrregularFunctionalData(argvals, values)

    @property
    def argvals(self) -> IrregArgvals:
        """Getter for argvals."""
        return cast(IrregArgvals, super().argvals)

    @argvals.setter
    def argvals(
        self,
        new_argvals: IrregArgvals
    ) -> None:
        IrregularFunctionalData._check_argvals_length(new_argvals)
        self._argvals = new_argvals
        points = self.gather_points()
        
        argvals_stand: IrregArgvals = {}
        for dim, obss in new_argvals.items():
            max_x, min_x = np.max(points[dim]), np.min(points[dim])

            argvals_stand[dim] = {}
            for obs, point in obss.items():
                argvals_stand[dim][obs] = _normalization(
                    point, max_x, min_x
                )
        self.argvals_stand = argvals_stand

    @property
    def range_obs(self) -> Tuple[float, float]:
        """Get the range of the observations of the object.

        Returns
        -------
        min, max: tuple
            Tuple containing the mimimum and maximum values taken by all the
            observations for the object.

        """
        ranges = [(np.min(obs), np.max(obs)) for obs in self.values.values()]
        return min(min(ranges)), max(max(ranges))

    @property
    def n_points(self) -> Dict[str, int]:
        """Get the mean number of sampling points.

        Returns
        -------
        n_points: dict
            A dictionary with the same shape than argvals with the number of
            sampling points along each axis.

        """
        n_points = {}
        for i, points in self.argvals.items():
            n_points[i] = np.mean([len(p) for p in points.values()])
        return n_points

    @property
    def range_dim(self) -> Dict[str, Tuple[int, int]]:
        """Get the range of the `argvals` for each of the dimension.

        Returns
        -------
        ranges: dict
            Dictionary containing the range of the argvals for each of the
            input dimension.

        """
        ranges = {idx: list(argval.values())
                  for idx, argval in self.argvals.items()}
        return {idx: (
            cast(int, min(map(min, dim))),
            cast(int, max(map(max, dim)))
        ) for idx, dim in ranges.items()}

    @property
    def shape(self) -> Dict[str, int]:
        r"""Get the shape of the data for each dimension.

        Returns
        -------
        shape: dict
            Dictionary containing the number of points for each of the
            dimension. It corresponds to :math:`m_j` for
            :math:`0 \leq j \leq p`.

        """
        return {idx: len(dim) for idx, dim in self.gather_points().items()}

    def gather_points(self) -> DenseArgvals:
        """Gather all the `argvals` for each of the dimensions separetely.

        Returns
        -------
        argvals: dict
            Dictionary containing all the unique observations points for each
            of the input dimension.

        """
        return {idx: np.unique(np.hstack(list(dim.values())))
                for idx, dim in self.argvals.items()}

    def as_dense(self) -> DenseFunctionalData:
        """Convert `self` from Irregular to Dense functional data.

        Coerce an IrregularFunctionalData object into a DenseFunctionalData
        object.

        Note
        ----
        We coerce an IrregularFunctionalData object into a DenseFunctionalData
        object by gathering all the sampling points from the different
        dimension into one, and set the value to `np.nan` for the not observed
        points.

        Returns
        -------
        obj: DenseFunctionalData
            An object of the class DenseFunctionalData

        """
        new_argvals = self.gather_points()
        new_values = np.full((self.n_obs,) + tuple(self.shape.values()),
                             np.nan)

        # Create the index definition domain for each of the observation
        index_obs = {}
        for obs in self.values.keys():
            index_obs_dim = []
            for dim in new_argvals.keys():
                _, idx, _ = np.intersect1d(new_argvals[dim],
                                           self.argvals[dim][obs],
                                           return_indices=True)
                index_obs_dim.append(idx)
            index_obs[obs] = index_obs_dim

        # Create mask arrays
        mask_obs = {obs: np.full(tuple(self.shape.values()), False)
                    for obs in self.values.keys()}
        for obs in self.values.keys():
            mask_obs[obs][tuple(np.meshgrid(*index_obs[obs]))] = True

        # Assign values
        for obs in self.values.keys():
            new_values[obs][mask_obs[obs]] = self.values[obs].flatten()

        return DenseFunctionalData(new_argvals, new_values)

    def is_compatible(self, fdata: FunctionalData) -> bool:
        """Check if `fdata` is compatible with `self`.

        Two IrregularFunctionalData object are said to be compatible if they
        have the same number of observations and dimensions. Moreover, they
        must have (strictly) the same sampling points.

        Parameters
        ----------
        fdata : IrregularFunctionalData object
            The object to compare with `self`.

        Returns
        -------
        True
            If the objects are compatible.

        """
        super().is_compatible(fdata)
        IrregularFunctionalData._check_argvals_equality(
            self.argvals, fdata.argvals)
        return True

    def to_basis(
        self,
        basis: Basis
    ) -> None:
        """Convert to basis"""
        raise NotImplementedError()

    def mean(
        self,
        smooth: Optional[str] = None,
        **kwargs: Any
    ) -> DenseFunctionalData:
        """Compute an estimate of the mean.

        Parameters
        ----------
        smooth: str, default=None
            Name of the smoothing method. Currently, not implemented.

        Returns
        -------
        obj: DenseFunctionalData object
            An estimate of the mean as a DenseFunctionalData object with a
            concatenation of the self.argvals as argvals and one observation.

        """
        dense_self = self.as_dense()

        # Catch this warning as 2D data might have empty slice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_estim = np.nanmean(dense_self.values, axis=0, keepdims=True)
        return DenseFunctionalData(dense_self.argvals, mean_estim)

    def covariance(
        self,
        mean: Optional[FunctionalData] = None,
        smooth: Optional[str] = None,
        **kwargs: Any
    ) -> FunctionalData:
        """Compute an estimate of the covariance."""
        pass

    def inner_product(self) -> npt.NDArray:
        return super().inner_product()

    def smooth(
        self,
        points: npt.NDArray[np.float64],
        neighborhood: npt.NDArray[np.float64],
        points_estim: Optional[npt.NDArray[np.float64]] = None,
        degree: int = 0,
        kernel: str = "epanechnikov",
        bandwidth: Optional[List[float]] = None
    ) -> IrregularFunctionalData:
        """Smooth the data.

        Notes
        -----
        Only, one dimensional IrregularFunctionalData can be smoothed.

        Parameters
        ----------
        points: np.array
            Points at which the Bandwidth is estimated.
        neighborhood: np.array
            Neighborhood considered for each each points. Should have the same
            shape than points.
        points_estim: np.array, default=None
            Points at which the curves are estimated. The default is None,
            meaning we use the argvals as estimation points.
        degree: int, default=0
            Degree for the local polynomial smoothing.
        kernel: str, default='epanechnikov'
            The name of the kernel to use.
        bandwidth: Bandwidth, default=None
            An instance of Bandwidth for the smoothing.

        Returns
        -------
        obj: IrregularFunctionalData
            A smoothed version of the data.

        """
        if self.n_dim > 1:
            raise NotImplementedError('Currently, only one dimensional data'
                                      ' can be smoothed.')

        # TODO: Provide an estimator for the bandwidth
        bandwidth = 0.5

        argvals = self.argvals['input_dim_0'].values()
        values = self.values.values()
        smooth_argvals, smooth_values = {}, {}
        for i, (arg, val) in enumerate(zip(argvals, values)):
            if points_estim is None:
                points_estim = arg

            lp = LocalPolynomial(kernel_name=kernel,
                                 bandwidth=bandwidth,
                                 degree=degree)
            pred = lp.fit_predict(arg, val, points_estim)
            smooth_argvals[i] = points_estim
            smooth_values[i] = pred
        return IrregularFunctionalData(
            {'input_dim_0': smooth_argvals}, smooth_values
        )


###############################################################################
# Class MultivariateFunctionalData

class MultivariateFunctionalData(UserList[FunctionalData]):
    r"""A class for defining Multivariate Functional Data.

    An instance of MultivariateFunctionalData is a list containing objects of
    the class DenseFunctionalData or IrregularFunctionalData.

    Notes
    -----
    Be careful that we will not check if all the elements have the same type.
    It is possible to create MultivariateFunctionalData containing both
    Dense and Iregular functional data. However, only this two types are
    allowed to be in the list.

    Parameters
    ----------
    data: list
        The list containing the elements of the MultivariateFunctionalData.

    """

    def __init__(
        self,
        initlist: List[FunctionalData]
    ) -> None:
        """Initialize MultivariateFunctionalData object."""
        self.data = initlist

    def __repr__(self) -> str:
        """Override print function."""
        return (f"Multivariate functional data object with {self.n_functional}"
                f" functions of {self.n_obs} observations.")

    @property
    def n_obs(self) -> int:
        """Get the number of observations of the functional data.

        Returns
        -------
        n_obs: int
            Number of observations within the functional data.

        """
        return self.data[0].n_obs if len(self) > 0 else 0

    @property
    def n_functional(self) -> int:
        """Get the number of functional data with `self`.

        Returns
        -------
        n_functional: int
            Number of functions in the list.

        """
        return len(self)

    @property
    def n_dim(self) -> List[int]:
        """Get the dimension of the functional data.

        Returns
        -------
        dim: list
            List containing the dimension of each component in the functional
            data.

        """
        return [i.n_dim for i in self]

    @property
    def range_obs(self) -> List[Tuple[float, float]]:
        """Get the range of the observations of the object.

        Returns
        -------
        (min, max): list of tuples
            List of tuples containing the mimimum and maximum values taken by
            all the observations for the object for each function.

        """
        return [i.range_obs for i in self]

    @property
    def n_points(self) -> List[Dict[str, int]]:
        """Get the mean number of sampling points.

        Returns
        -------
        n_points: list of dict
            A list of dictionary with the same shape than argvals with the
            number of sampling points along each axis for each function.

        """
        return [i.n_points for i in self]

    @property
    def range_points(self) -> List[Dict[str, Tuple[int, int]]]:
        """Get the range of the `argvals` for each of the dimension.

        Returns
        -------
        ranges: list of dict
            List of dictionary containing the range of the argvals for each of
            the input dimension for each function.

        """
        return [i.range_dim for i in self]

    @property
    def shape(self) -> List[Dict[str, int]]:
        r"""Get the shape of the data for each dimension.

        Returns
        -------
        shape: list of dict
            List of dictionary containing the number of points for each of the
            dimension for each function. It corresponds to :math:`m_j` for
            :math:`0 \leq j \leq p`.

        """
        return [i.shape for i in self]

    def append(self, item: FunctionalData) -> None:
        """Add an item to `self`.

        Parameters
        ----------
        item: DenseFunctionalData or IrregularFunctionalData
            Item to add.

        """
        if len(self.data) == 0:
            self.data = [item]
        else:
            FunctionalData._check_same_nobs(*self, item)
            self.data.append(item)

    def extend(self, other: Iterable[FunctionalData]) -> None:
        """Extend the list of FunctionalData by appending from iterable."""
        super().extend(other)

    def insert(self, i: int, item: FunctionalData) -> None:
        """Insert an item `item` at a given position `i`."""
        FunctionalData._check_same_nobs(*self, item)
        self.data.insert(i, item)

    def remove(self, item: FunctionalData) -> None:
        """Remove the first item from `self` where value is `item`."""
        raise NotImplementedError

    def pop(self, i: int = -1) -> FunctionalData:
        """Remove the item at the given position in the list, and return it."""
        return super().pop(i)

    def clear(self) -> None:
        """Remove all items from the list."""
        super().clear()

    def sort(self, *args: Any, **kwargs: Any) -> None:
        """Sort the items of the list in place."""
        raise NotImplementedError

    def reverse(self) -> None:
        """Reserve the elements of the list in place."""
        super().reverse()

    def copy(self) -> MultivariateFunctionalData:
        """Return a shallow copy of the list."""
        return super().copy()

    def get_obs(self) -> Iterator[MultivariateFunctionalData]:
        """Return a generator over the observation."""
        for idx in range(self.n_obs):
            yield MultivariateFunctionalData([obs[idx] for obs in self])

    def mean(
        self,
        smooth: Optional[str] = None,
        **kwargs: Any
    ) -> MultivariateFunctionalData:
        """Compute an estimate of the mean.

        Parameters
        ----------
        smooth: str, default=None
            Name of the smoothing method. Currently, not implemented.

        Returns
        -------
        obj: MultivariateFunctionalData object
            An estimate of the mean as a MultivariateFunctionalData object
            with a concatenation of the self.argvals as argvals and one
            observation.

        """
        return MultivariateFunctionalData(
            [i.mean(smooth, **kwargs) for i in self]
        )

    def covariance(
        self,
        mean: Optional[MultivariateFunctionalData] = None,
        smooth: Optional[str] = None,
        **kwargs: Any
    ) -> MultivariateFunctionalData:
        """Compute an estimate of the covariance.

        Parameters
        ----------
        smooth: str, default=None
            Name of the smoothing method to use. Currently, not implemented.
        mean: MultivariateFunctionalData, default=None
            An estimate of the mean of self. If None, an estimate is computed.

        Returns
        -------
        obj: MultivariateFunctionalData object
            An estimate of the covariance as a two-dimensional
            MultivariateFunctionalData object with same argvals as `self`.

        """
        if mean is not None:
            return MultivariateFunctionalData(
                [i.covariance(m, smooth, **kwargs) for i, m in zip(self, mean)]
            )
        else:
            return MultivariateFunctionalData(
                [i.covariance(None, smooth, **kwargs) for i in self]
            )

    def inner_product(self) -> npt.NDArray:
        r"""Compute the inner product matrix of the data.

        The inner product matrix is a ``n_obs`` by ``n_obs`` matrix where each
        entry is defined as

        .. math::
            \langle\langle x, y \rangle\rangle =
            \sum_{p = 1}^P \int_{\mathcal{T}_k} x^{(p)}(t)y^{(p)}(t)dt,
            t \in \mathcal{T}.

        Returns
        -------
        np.array, shape=(n_obs, n_obs)
            Inner product matrix of the data.

        Examples
        --------
        argvals = {'input_dim_0': np.array([0., 0.25, 0.5 , 0.75])}
        values = np.array(
            [
                [ 2.48466259, -3.38397716, -1.2367073 , -1.85052901],
                [ 1.44853118,  0.67716255,  1.79711043,  4.76950236],
                [-5.13173463,  0.35830122,  0.56648942, -0.20965252]
            ]
        )
        data_1D = DenseFunctionalData(argvals, values)

        argvals = {
            'input_dim_0': np.array([0.  , 0.25, 0.5 , 0.75]),
            'input_dim_1': np.array([0.  , 0.25, 0.5 , 0.75])
        }
        values = np.array(
            [
                [
                    [  6.30864764, -18.37912204,   6.15515232,  29.8027036 ],
                    [ -6.076622  , -15.48586803, -11.39997792,   8.40599319],
                    [-20.4094798 ,  -1.3872093 ,  -0.59922597,  -6.42013363],
                    [  5.78626375,  -1.83874696,  -0.87225549,   2.75000303]
                ],
                [
                    [ -4.83576968,  18.85512513, -18.73086523,  15.1511348 ],
                    [-24.41254888,  12.37333951,  28.85176939,  16.41806885],
                    [-10.02681278,  14.76500118,   1.83114017,  -2.78985647],
                    [  4.29268032,   8.1781319 ,  30.10132687,  -0.72828334]
                ],
                [
                    [ -5.85921132,   1.85573561,  -5.11291405, -12.89441767],
                    [ -4.79384081,  -0.93863074,  18.81909033,   4.55041973],
                    [-13.27810529,  28.08961819, -13.79482673,  35.25677906],
                    [  9.10058173, -16.43979436, -11.88561292,  -5.86481318]
                ]
            ]
        )
        data_2D = DenseFunctionalData(argvals, values)
        data = MultivariateFunctionalData([data_1D, data_2D])
        data.inner_product()
        > array(
        >     [
        >         [ 72.37627198, -28.54691325, -19.7335636 ],
        >         [-28.54691325, 166.61824532,  50.66329182],
        >         [-19.7335636 ,  50.66329182, 151.2780517 ]
        >     ]
        > )
        """
        if not all([isinstance(data, DenseFunctionalData) for data in self]):
            raise TypeError(
                "All the univariate data must be DenseFunctionalData"
            )
        return np.sum([data.inner_product() for data in self], axis=0)

    def concatenate(
        self,
        data: MultivariateFunctionalData
    ) -> MultivariateFunctionalData:
        """Concatenate two MultivariateFunctionalData.

        Parameters
        ----------
        data: MultivariateFunctionalData
            The data to concatenate with self.

        Returns
        -------
        res: MultivariateFunctionalData
            The concatenation of self and data.

        """
        new = [_concatenate([d1, d2]) for d1, d2 in zip(self, data)]
        return MultivariateFunctionalData(new)


##############################################################################
# Functional data manipulation

def _concatenate(
    data: List[FunctionalData]
) -> FunctionalData:
    """Concatenate functional data.

    Compute multiple DenseFunctionalData into one. It works with higher
    dimension for the input data.

    Parameters
    ----------
    data: DenseFunctionalData
        DenseFunctionalData to concatenate.

    Returns
    -------
    data: DenseFunctionalData
        The concatenation of the input data.

    Notes
    -----
    TODO :
    * Add tests, in particular check that the data are compatible.

    """
    new_argvals = data[0].argvals
    new_values = np.vstack([d.values for d in data])
    return DenseFunctionalData(new_argvals, new_values)


def _tensor_product(
    data1: DenseFunctionalData,
    data2: DenseFunctionalData
) -> DenseFunctionalData:
    """Compute the tensor product between functional data.

    Compute the tensor product between all the observation of data1 with all
    the observation of data2.

    Parameters
    ----------
    data1: DenseFunctionalData
        First functional data.
    data2: DenseFunctionalData
        Second functional data.

    Returns
    -------
    data: DenseFunctionalData
        The tensor product between data1 and data2. It contains data1.n_obs *
        data2.n_obs observations.

    Notes
    -----
    TODO:
    * Add tests.

    """
    arg = {'input_dim_0': data1.argvals['input_dim_0'],
           'input_dim_1': data2.argvals['input_dim_0']}
    val = [_outer(i, j) for i in data1.values for j in data2.values]
    return DenseFunctionalData(arg, np.array(val))
