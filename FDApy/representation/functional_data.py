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

from abc import ABC, abstractmethod
from collections import UserList
from collections.abc import Iterator
from typing import (
    Callable, Dict, Iterable, Optional, List,
    Tuple, Type, Union
)

from .argvals import Argvals, DenseArgvals, IrregularArgvals
from .values import Values, DenseValues, IrregularValues

from ..preprocessing.smoothing.local_polynomial import LocalPolynomial
from ..misc.utils import _cartesian_product
from ..misc.utils import _inner_product
from ..misc.utils import _integrate
from ..misc.utils import _outer
from ..misc.utils import _shift


###############################################################################
# Class FunctionalData
class FunctionalData(ABC):
    """Metaclass for the definition of diverse functional data objects.

    Parameters
    ----------
    argvals: Type[Argvals]
        Sampling points of the functional data.
    values: Type[Values]
        Values of the functional data.

    """

    ###########################################################################
    # Checkers
    @staticmethod
    def _check_same_type(
        *fdata: Type[FunctionalData]
    ) -> None:
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
    def _check_same_nobs(
        *fdata: Type[FunctionalData]
    ) -> None:
        """Raise an arror if elements in `fdata` have different number of obs.

        Raises
        ------
        ValueError
            When all `fdata` do not have the same number of observations.

        """
        n_obs = set(obj.n_obs for obj in fdata)
        if len(n_obs) > 1:
            raise ValueError(
                "Elements do not have the same number of observations."
            )

    @staticmethod
    def _check_same_ndim(
        *fdata: Type[FunctionalData]
    ) -> None:
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
    def _is_compatible(
        *fdata: Type[FunctionalData]
    ) -> None:
        """Raise an error if elements in `fdata` are not compatible.

        Parameters
        ----------
        *fdata: FunctionalData
            Functional data to compare.

        Raises
        ------
        ValueError
            When all `fdata` do not have the same number of observations or
            when all `fdata` do not have the same dimension.
            When all `fdata` do not have the same argvals.
        TypeError
            When all `fdata` do not have the same type.

        """
        FunctionalData._check_same_type(*fdata)
        FunctionalData._check_same_nobs(*fdata)
        FunctionalData._check_same_ndim(*fdata)
        if not all(data.argvals == fdata[0].argvals for data in fdata):
            raise ValueError("Argvals are not equals.")

    ###########################################################################

    ###########################################################################
    # Static methods
    @staticmethod
    @abstractmethod
    def _perform_computation(
        fdata1: Type[FunctionalData],
        fdata2: Type[FunctionalData],
        func: Callable
    ) -> Type[FunctionalData]:
        """Perform computation."""

    @staticmethod
    @abstractmethod
    def concatenate(
        *fdata: Type[FunctionalData]
    ) -> Type[FunctionalData]:
        """Concatenate FunctionalData objects.

        Parameters
        ----------
        *fdata: FunctionalData
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
    # Magic methods
    def __init__(
        self,
        argvals: Type[Argvals],
        values: Type[Values],
    ) -> None:
        """Initialize FunctionalData object."""
        super().__init__()
        self.argvals = argvals
        self.values = values
        self._index = 0

    def __repr__(self) -> str:
        """Override print function."""
        return (
            f"Functional data object with {self.n_obs} observations on a "
            f"{self.n_dimension}-dimensional support."
        )

    def __iter__(self):
        """Initialize the iterator."""
        return FunctionalDataIterator(self)

    @abstractmethod
    def __getitem__(
        self,
        index: int
    ) -> Type[FunctionalData]:
        """Override getitem function, called when self[index]."""

    def __add__(
        self,
        obj: Type[FunctionalData]
    ) -> Type[FunctionalData]:
        """Override add function."""
        return self._perform_computation(self, obj, np.add)

    def __sub__(
        self,
        obj: Type[FunctionalData]
    ) -> Type[FunctionalData]:
        """Override sub function."""
        return self._perform_computation(self, obj, np.subtract)

    def __mul__(
        self,
        obj: Type[FunctionalData]
    ) -> Type[FunctionalData]:
        """Override mul function."""
        return self._perform_computation(self, obj, np.multiply)

    def __rmul__(
        self,
        obj: Type[FunctionalData]
    ) -> Type[FunctionalData]:
        """Override rmul function."""
        return self * obj

    def __truediv__(
        self,
        obj: Type[FunctionalData]
    ) -> Type[FunctionalData]:
        """Override truediv function."""
        return self._perform_computation(self, obj, np.true_divide)

    def __floordiv__(
        self,
        obj: Type[FunctionalData]
    ) -> Type[FunctionalData]:
        """Override floordiv function."""
        return self._perform_computation(self, obj, np.floor_divide)

    ###########################################################################

    ###########################################################################
    # Properties
    @property
    def argvals(
        self
    ) -> Type[Argvals]:
        """Getter for argvals."""
        return self._argvals

    @argvals.setter
    @abstractmethod
    def argvals(
        self,
        new_argvals: Type[Argvals]
    ) -> None:
        """Setter for argvals."""

    @property
    def argvals_stand(
        self
    ) -> Type[Argvals]:
        """Getter for argvals_stand."""
        return self._argvals_stand

    @argvals_stand.setter
    def argvals_stand(
        self,
        new_argvals_stand: Type[Argvals]
    ) -> None:
        """Setter for argvals_stand."""
        if not isinstance(new_argvals_stand, Argvals):
            raise TypeError('new_argvals_stand must be an Argvals object.')
        self._argvals_stand = new_argvals_stand

    @property
    def values(
        self
    ) -> Type[Values]:
        """Getter for values."""
        return self._values

    @values.setter
    @abstractmethod
    def values(
        self,
        new_values: Type[Values]
    ) -> None:
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
    def n_points(self) -> Union[Tuple[int, ...], Dict[int, Tuple[int, ...]]]:
        """Get the number of sampling points.

        Returns
        -------
        Union[Tuple[int, ...], Dict[int, Tuple[int, ...]]]
            Number of sampling points.

        """
        return self.argvals.n_points

    ###########################################################################

    ###########################################################################
    # Abstract methods
    @abstractmethod
    def to_long(self) -> pd.DataFrame:
        """Convert the data to long format."""

    @abstractmethod
    def smooth(
        self,
        points: Optional[DenseArgvals] = None,
        kernel_name: Optional[str] = "epanechnikov",
        bandwidth: Optional[float] = None,
        degree: Optional[int] = 1
    ) -> Type[FunctionalData]:
        """Smooth the data."""

    @abstractmethod
    def mean(
        self,
        points: Optional[DenseArgvals] = None,
        smooth: bool = True,
        **kwargs
    ) -> DenseFunctionalData:
        """Compute an estimate of the mean."""

    @abstractmethod
    def inner_product(
        self,
        method: str = 'trapz'
    ) -> npt.NDArray[np.float64]:
        """Compute an estimate of the inner product matrix."""

    @abstractmethod
    def norm(
        self,
        squared: bool = False,
        method: str = 'trapz',
        use_argvals_stand: bool = False
    ) -> npt.NDArray[np.float64]:
        """Norm of each observation of the data.

        TODO: Incorporate different type of norms. Especially, the one from the
        paper.

        """

    @abstractmethod
    def normalize(
        self,
        use_argvals_stand: bool = False
    ) -> Tuple[FunctionalData, float]:
        """Normalize the data."""

    @abstractmethod
    def covariance(
        self,
        mean: Optional[Type[FunctionalData]] = None,
        smooth: Optional[str] = None,
        **kwargs
    ) -> Type[FunctionalData]:
        """Compute an estimate of the covariance."""

    ###########################################################################


###############################################################################
# Class FunctionalDataIterator
class FunctionalDataIterator(Iterator):
    """Iterator for FunctionalData object."""

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
class DenseFunctionalData(FunctionalData):
    r"""Class for defining Dense Functional Data.

    A class used to define dense functional data. We denote by :math:`n`, the
    number of observations and by :math:`p`, the number of input dimensions.
    Here, we are in the case of univariate functional data, and so the output
    dimension will be :math:`\mathbb{R}`.

    Parameters
    ----------
    argvals: DenseArgvals
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. The shape of the :math:`j` th
        dimension is :math:`(m_j,)` for :math:`0 \leq j \leq p`.
    values: DenseValues
        The values of the functional data. The shape of the array is
        :math:`(n, m_1, \dots, m_p)`.

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

    """

    ###########################################################################
    # Checkers

    ###########################################################################

    ###########################################################################
    # Static methods
    @staticmethod
    def _perform_computation(
        fdata1: DenseFunctionalData,
        fdata2: DenseFunctionalData,
        func: Callable
    ) -> DenseFunctionalData:
        """Perform computation defined by `func` if they are compatible.

        Parameters
        ----------
        fdata1: DenseFunctionalData
            First functional data to consider.
        fdata2: DenseFunctionalData
            Second functional data to consider.
        func: Callable
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
    def concatenate(
        *fdata: DenseFunctionalData
    ) -> DenseFunctionalData:
        """Concatenate DenseFunctional objects.

        Returns
        -------
        DenseFunctionalData
            The concatenated object.

        """
        super(
            DenseFunctionalData, DenseFunctionalData
        ).concatenate(*fdata)
        argvals = DenseArgvals.concatenate(*[el.argvals for el in fdata])
        values = DenseValues.concatenate(*[el.values for el in fdata])
        return DenseFunctionalData(argvals, values)

    ###########################################################################

    ###########################################################################
    # Magic methods
    def __init__(
        self,
        argvals: DenseArgvals,
        values: DenseValues
    ) -> None:
        """Initialize UnivariateFunctionalData object."""
        super().__init__(argvals, values)

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

    ###########################################################################
    # Properties
    @FunctionalData.argvals.setter
    def argvals(
        self,
        new_argvals: DenseArgvals
    ) -> None:
        """Setter for argvals."""
        if not isinstance(new_argvals, DenseArgvals):
            raise TypeError('new_argvals must be a DenseArgvals object.')
        if hasattr(self, 'values'):
            self._values.compatible_with(new_argvals)
        self._argvals = new_argvals
        self._argvals_stand = self._argvals.normalization()

    @FunctionalData.values.setter
    def values(
        self,
        new_values: DenseValues
    ) -> None:
        """Setter for values."""
        if not isinstance(new_values, DenseValues):
            raise TypeError('new_values must be a DenseValues object.')
        if hasattr(self, 'argvals'):
            self._argvals.compatible_with(new_values)
        self._values = new_values

    ###########################################################################

    ###########################################################################
    # Methods
    def to_long(self) -> pd.DataFrame:
        """Convert the data to long format.

        This function transform a DenseFunctionalData object into pandas
        DataFrame. It uses the long format to represent the DenseFunctionalData
        object as a dataframe. This is a helper function as it might be easier
        for some computation, e.g., smoothing of the mean and covariance
        functions to have a long format.

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
        temp['id'] = np.repeat(np.arange(self.n_obs), np.prod(self.n_points))
        temp['values'] = self.values.flatten()
        return temp

    def smooth(
        self,
        points: Optional[DenseArgvals] = None,
        kernel_name: str = "epanechnikov",
        bandwidth: Optional[float] = None,
        degree: int = 1
    ) -> DenseFunctionalData:
        """Smooth the data.

        This function smooths each curves individually. Based on [1]_, it fits
        a local smoother to the data (the argument ``degree`` controls the
        degree of the local fits).

        Parameters
        ----------
        points: Optional[DenseArgvals], default=None
            Points at which the curves are estimated. The default is None,
            meaning we use the argvals as estimation points.
        kernel_name: str, default="epanechnikov"
            Kernel name used as weight (`gaussian`, `epanechnikov`, `tricube`,
            `bisquare`).
        bandwidth: float, default=None
            Strictly positive. Control the size of the associated neighborhood.
            If ``bandwidth == None``, it is assumed that the curves are twice
            differentiable and the bandwidth is set to :math:`n^{-1/5}` [2]_
            where :math:`n` is the number of sampling points per curve. Be
            careful that it will not work if the curves are not sampled on
            :math:`[0, 1]`.
        degree: int, default=1
            Degree of the local polynomial to fit. If ``degree=0``, we fit
            the local constant estimator (equivalent to the Nadaraya-Watson
            estimator). If ``degree=1``, we fit the local linear estimator.
            If ``degree=2``, we fit the local quadratic estimator.

        Returns
        -------
        DenseFunctionalData
            Smoothed data.

        References
        ----------
        .. [1] Zhang, J.-T. and Chen J. (2007), Statistical Inferences for
            Functional Data, The Annals of Statistics, Vol. 35, No. 3.
        .. [2] Tsybakov, A.B. (2008), Introduction to Nonparametric Estimation.
            Springer Series in Statistics.

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
        if bandwidth is None:
            bandwidth = np.product(self.n_points)**(-1 / 5)

        argvals_mat = _cartesian_product(*self.argvals.values())
        points_mat = _cartesian_product(*points.values())

        lp = LocalPolynomial(
            kernel_name=kernel_name, bandwidth=bandwidth, degree=degree
        )

        smooth = np.zeros((self.n_obs, *points.n_points))
        for idx, obs in enumerate(self):
            smooth[idx, :] = lp.predict(
                y=obs.values.flatten(),
                x=argvals_mat,
                x_new=points_mat
            ).reshape(smooth.shape[1:])
        return DenseFunctionalData(points, DenseValues(smooth))

    def mean(
        self,
        points: Optional[DenseArgvals] = None,
        smooth: bool = True,
        **kwargs
    ) -> DenseFunctionalData:
        """Compute an estimate of the mean.

        This function computes an estimate of the mean curve of a
        DenseFunctionalData object. As the curves are sampled on a common grid,
        we consider the sample mean, as defined in [1]_. The sampled mean is
        rate optimal [2]_. We included some smoothing using Local Polynonial
        Estimators.

        Parameters
        ----------
        points: Optional[DenseArgvals], default=None
            The sampling points at which the mean is estimated. If `None`, the
            DenseArgvals of the DenseFunctionalData is used. If `smooth` is
            False, the DenseArgvals of the DenseFunctionalData is used.
        smooth: bool, default=True
            Should the mean be smoothed?
        **kwargs:
            kernel_name: str, default='epanechnikov'
                Name of the kernel used for local polynomial smoothing.
            degree: int, default=1
                Degree used for local polynomial smoothing.
            bandwidth: float
                Bandwidth used for local polynomial smoothing. The default
                bandwitdth is set to be the number of sampling points to the
                power :math:`-1/5` [3]_.

        Returns
        -------
        DenseFunctionalData
            An estimate of the mean as a DenseFunctionalData object.

        References
        ----------
        .. [1] Ramsey, J. O. and Silverman, B. W. (2005), Functional Data
            Analysis, Springer Science, Chapter 8.
        .. [2] Cai, T.T., Yuan, M., (2011), Optimal estimation of the mean
            function based on discretely sampled functional data: Phase
            transition. The Annals of Statistics 39, 2330-2355.
        .. [3] Tsybakov, A.B. (2008), Introduction to Nonparametric Estimation.
            Springer Series in Statistics.

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

        if smooth:
            self._mean = self._mean.smooth(
                points=points,
                kernel_name=kwargs.get('kernel_name', 'epanechnikov'),
                bandwidth=kwargs.get(
                    'bandwidth',
                    np.product(self.n_points)**(-1 / 5)
                ),
                degree=kwargs.get('degree', 1)
            )
        return self._mean

    def inner_product(
        self,
        method: str = 'trapz'
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
        method: str, {'simpson', 'trapz'}, default = 'trapz'
            The method used to integrated.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_obs)
            Inner product matrix of the data.

        References
        ----------
        .. [1] Ramsey, J. O. and Silverman, B. W. (2005), Functional Data
            Analysis, Springer Science, Chapter 2.

        Examples
        --------
        For one-dimensional functional data:

        >>> argvals = {'input_dim_0': np.array([0., 0.25, 0.5 , 0.75])}
        >>> values = np.array(
        ...     [
        ...         [ 2.48466259, -3.38397716, -1.2367073 , -1.85052901],
        ...         [ 1.44853118,  0.67716255,  1.79711043,  4.76950236],
        ...         [-5.13173463,  0.35830122,  0.56648942, -0.20965252]
        ...     ]
        ... )
        >>> data = DenseFunctionalData(argvals, values)
        >>> data.inner_product()
        array(
            [
                [ 4.44493731, -1.78187445, -2.02359881],
                [-1.78187445,  4.02783817, -0.73900893],
                [-2.02359881, -0.73900893,  3.40965432]
            ]
        )

        For two-dimensional functional data:

        >>> argvals = {
        ...     'input_dim_0': np.array([0.  , 0.25, 0.5 , 0.75]),
        ...     'input_dim_1': np.array([0.  , 0.25, 0.5 , 0.75])
        ... }
        >>> values = np.array([
        ...     [
        ...         [  6.30864764, -18.37912204,   6.15515232,  29.8027036 ],
        ...         [ -6.076622  , -15.48586803, -11.39997792,   8.40599319],
        ...         [-20.4094798 ,  -1.3872093 ,  -0.59922597,  -6.42013363],
        ...         [  5.78626375,  -1.83874696,  -0.87225549,   2.75000303]
        ...     ],
        ...     [
        ...         [ -4.83576968,  18.85512513, -18.73086523,  15.1511348 ],
        ...         [-24.41254888,  12.37333951,  28.85176939,  16.41806885],
        ...         [-10.02681278,  14.76500118,   1.83114017,  -2.78985647],
        ...         [  4.29268032,   8.1781319 ,  30.10132687,  -0.72828334]
        ...     ],
        ...     [
        ...         [ -5.85921132,   1.85573561,  -5.11291405, -12.89441767],
        ...         [ -4.79384081,  -0.93863074,  18.81909033,   4.55041973],
        ...         [-13.27810529,  28.08961819, -13.79482673,  35.25677906],
        ...         [  9.10058173, -16.43979436, -11.88561292,  -5.86481318]
        ...     ]
        ... ])
        >>> data = DenseFunctionalData(argvals, values)
        >>> data.inner_product()
        array(
            [
                [ 67.93133466, -26.76503879, -17.70996479],
                [-26.76503879, 162.59040715,  51.40230074],
                [-17.70996479,  51.40230074, 147.86839738]
            ]
        )

        """
        # Get parameters
        n_obs = self.n_obs
        axis = [argvals for argvals in self.argvals.values()]

        inner_mat = np.zeros((n_obs, n_obs))
        for (i, j) in itertools.product(np.arange(n_obs), repeat=2):
            if i <= j:
                inner_mat[i, j] = _inner_product(
                    self.values[i],
                    self.values[j],
                    *axis,
                    method=method
                )
        inner_mat = inner_mat + inner_mat.T
        np.fill_diagonal(inner_mat, np.diag(inner_mat) / 2)
        return inner_mat

    def norm(
        self,
        squared: bool = False,
        method: str = 'trapz',
        use_argvals_stand: bool = False
    ) -> npt.NDArray[np.float64]:
        r"""Norm of each observation of the data.

        For each observation in the data, it computes its norm [1]_ defined as

        .. math::
            \lvert\lvert f \rvert\rvert = \left(\int_{\mathcal{T}}
            f(t)^2dt\right)^{1\2}, t \in \mathcal{T},

        Parameters
        ----------
        squared: bool, default=False
            If `True`, the function calculates the squared norm, otherwise it
            returns the norm.
        method: str, {'simpson', 'trapz'}, default = 'trapz'
            The method used to integrated.
        use_argvals_stand: bool, default=False
            Use standardized argvals to compute the normalization of the data.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs,)
            The norm of each observations.

        References
        ----------
        .. [1] Ramsey, J. O. and Silverman, B. W. (2005), Functional Data
            Analysis, Springer Science, Chapter 2.

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
            norm_fd[idx] = _integrate(
                sq_values[idx], *axis, method=method
            )

        if squared:
            return np.array(norm_fd)
        else:
            return np.power(norm_fd, 0.5)

    def normalize(
        self,
        method: str = 'trapz',
        use_argvals_stand: bool = False
    ) -> Tuple[DenseFunctionalData, float]:
        r"""Normalize the data.

        The normalization is performed by divising each functional datum by
        :math:`w_j = \int_{T} Var(X(t))dt`.

        Parameters
        ----------
        method: str, {'simpson', 'trapz'}, default = 'trapz'
            The method used to integrated.
        use_argvals_stand: bool, default=False
            Use standardized argvals to compute the normalization of the data.

        Returns
        -------
        Tuple[DenseFunctionalData, float]
            The normalized data and its weight.

        References
        ----------
        .. [1] Happ and Greven (2018), Multivariate Functional Principal
            Component Analysis for Data Observed on Different (Dimensional)
            Domains. Journal of the American Statistical Association, 113,
            pp. 649--659.

        Examples
        --------
        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=10)
        >>> kl.data.normalize()
        (Functional data object with 10 observations on a 1-dimensional
        support., DenseValues(0.21227413))

        """
        if use_argvals_stand:
            axis = [argvals for argvals in self.argvals_stand.values()]
        else:
            axis = [argvals for argvals in self.argvals.values()]
        variance = np.var(self.values, axis=0)
        weights = _integrate(variance, *axis, method=method)
        new_values = self.values / weights
        return DenseFunctionalData(self.argvals, new_values), weights

    def covariance(
        self,
        points: Optional[DenseArgvals] = None,
        mean: Optional[DenseFunctionalData] = None,
        smooth: bool = True,
        **kwargs
    ) -> DenseFunctionalData:
        r"""Compute an estimate of the covariance function.

        This function computes an estimate of the covariance surface of a
        DenseFunctionalData object. As the curves are sampled on a common grid,
        we consider the sample covariance [1]_.

        Parameters
        ----------
        points: Optional[DenseArgvals], default=None
            The sampling points at which the covariance is estimated. If
            `None`, the DenseArgvals of the DenseFunctionalData is used. If
            `smooth` is False, the DenseArgvals of the DenseFunctionalData is
            used.
        mean: Optional[DenseFunctionalData], default=None
            An estimate of the mean of self. If None, an estimate is computed.
        smooth: bool, default=True
            Should the mean be smoothed?
        **kwargs:
            kernel_name: str, default='epanechnikov'
                Name of the kernel used for local polynomial smoothing.
            degree: int, default=1
                Degree used for local polynomial smoothing.
            bandwidth: float
                Bandwidth used for local polynomial smoothing. The default
                bandwitdth is set to be the number of sampling points to the
                power :math:`-1/5` [3]_.

        Returns
        -------
        DenseFunctionalData
            An estimate of the covariance as a two-dimensional
            DenseFunctionalData object.

        References
        ----------
        .. [1] Ramsey, J. O. and Silverman, B. W. (2005), Functional Data
            Analysis, Springer Science, Chapter 8.
        .. [2] Yao, F., Müller, H.-G., Wang, J.-L. (2005). Functional Data
            Analysis for Sparse Longitudinal Data. Journal of the American
            Statistical Association 100, pp. 577--590.
        .. [3] Staniswalis and Lee (1998), Nonparametric Regression Analysis of
            Longitudinal Data, Journal of the American Statistical Association,
            93, pp. 1403--1418.
        .. [4] Tsybakov, A.B. (2008), Introduction to Nonparametric Estimation.
            Springer Series in Statistics.

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
            raise ValueError(
                'Only one dimensional functional data are supported.'
            )

        if points is None:
            points = self.argvals
        argvals_cov = DenseArgvals({
            'input_dim_0': self.argvals['input_dim_0'],
            'input_dim_1': self.argvals['input_dim_0'],
        })
        points_cov = DenseArgvals({
            'input_dim_0': points['input_dim_0'],
            'input_dim_1': points['input_dim_0'],
        })
        if mean is None:
            mean = self.mean(smooth=smooth, **kwargs)

        data = self.values - mean.values
        cov = np.dot(data.T, data) / (self.n_obs - 1)
        raw_diag_cov = np.diag(cov).copy()

        if smooth:
            # Remove covariance diagonal because of measurement errors.
            np.fill_diagonal(cov, np.nan)

            cov_temp = DenseFunctionalData(argvals_cov, cov[np.newaxis])
            fdata_long = cov_temp.to_long()
            fdata_long = fdata_long.dropna()

            x = fdata_long.drop(['id', 'values'], axis=1, inplace=False).values
            y = fdata_long['values'].values
            points_mat = _cartesian_product(*points_cov.values())

            lp = LocalPolynomial(
                kernel_name=kwargs.get('kernel_name', 'epanechnikov'),
                bandwidth=kwargs.get(
                    'bandwidth',
                    np.product(cov_temp.n_points)**(-1 / 5)
                ),
                degree=kwargs.get('degree', 2)
            )
            cov = lp.predict(y=y, x=x, x_new=points_mat)
            cov = cov.reshape(points_cov.n_points)

        # Ensure the covariance is symmetric.
        cov = (cov + cov.T) / 2

        # Estimate noise variance ([2], [3])
        lp = LocalPolynomial(
            kernel_name='epanechnikov',
            bandwidth=len(raw_diag_cov)**(- 1 / 5),
            degree=1
        )
        var_hat = lp.predict(
            y=raw_diag_cov,
            x=_cartesian_product(*self.argvals.values()),
            x_new=_cartesian_product(*points.values())
        )
        lower = [int(np.round(0.25 * el)) for el in points.n_points]
        upper = [int(np.round(0.75 * el)) for el in points.n_points]
        bounds = slice(*tuple(lower + upper))
        temp = _integrate(
            (var_hat - np.diag(cov))[bounds],
            points['input_dim_0'][bounds],
            method='trapz'
        )

        self._noise_variance = np.maximum(
            2 * temp / points.range()['input_dim_0'], 0
        )
        self._covariance = DenseFunctionalData(
            points_cov, DenseValues(cov[np.newaxis])
        )
        return self._covariance
    ###########################################################################


###############################################################################
# Class IrregularFunctionalData
class IrregularFunctionalData(FunctionalData):
    r"""A class for defining Irregular Functional Data.

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

    """

    ###########################################################################
    # Checkers

    ###########################################################################

    ###########################################################################
    # Static methods
    @staticmethod
    def _perform_computation(
        fdata1: IrregularFunctionalData,
        fdata2: IrregularFunctionalData,
        func: Callable
    ) -> IrregularFunctionalData:
        """Perform computation defined by `func` if they are compatible.

        Parameters
        ----------
        fdata1: IrregularFunctionalData
            First functional data to consider.
        fdata2: IrregularFunctionalData
            Second functional data to consider.
        func: Callable
            The function to apply to combine `fdata1` and `fdata2`.

        Returns
        -------
        IrregularFunctionalData
            The resulting functional data.

        """
        IrregularFunctionalData._is_compatible(fdata1, fdata2)

        new_values = {
            idx: func(obs1, obs2)
            for (idx, obs1), (_, obs2)
            in zip(fdata1.values.items(), fdata2.values.items())
        }
        return IrregularFunctionalData(
            fdata1.argvals, IrregularValues(new_values)
        )

    @staticmethod
    def concatenate(
        *fdata: IrregularFunctionalData
    ) -> IrregularFunctionalData:
        """Concatenate IrregularFunctionalData objects.

        Returns
        -------
        IrregularFunctionalData
            The concatenated objects.

        """
        super(
            IrregularFunctionalData, IrregularFunctionalData
        ).concatenate(*fdata)
        argvals = IrregularArgvals.concatenate(*[el.argvals for el in fdata])
        values = IrregularValues.concatenate(*[el.values for el in fdata])
        return IrregularFunctionalData(argvals, values)

    ###########################################################################

    ###########################################################################
    # Magic methods
    def __init__(
        self,
        argvals: IrregularArgvals,
        values: IrregularValues
    ) -> None:
        """Initialize IrregularFunctionalData object."""
        super().__init__(argvals, values)

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
        if isinstance(index, slice):
            indices = index.indices(self.n_obs)
            argvals = {obs: self.argvals.get(obs) for obs in range(*indices)}
            values = {obs: self.values.get(obs) for obs in range(*indices)}
        else:
            argvals = {index: self.argvals[index]}
            values = {index: self.values[index]}
        return IrregularFunctionalData(
            IrregularArgvals(argvals),
            IrregularValues(values)
        )

    ###########################################################################

    ###########################################################################
    # Properties
    @FunctionalData.argvals.setter
    def argvals(
        self,
        new_argvals: IrregularArgvals
    ) -> None:
        """Setter for argvals."""
        if not isinstance(new_argvals, IrregularArgvals):
            raise TypeError('new_argvals must be a IrregularArgvals object.')
        if hasattr(self, 'values'):
            self._values.compatible_with(new_argvals)
        self._argvals = new_argvals
        self._argvals_stand = self._argvals.normalization()

    @FunctionalData.values.setter
    def values(
        self,
        new_values: IrregularValues
    ) -> None:
        """Setter for values."""
        if not isinstance(new_values, IrregularValues):
            raise TypeError('new_values must be a IrregularValues object.')
        if hasattr(self, 'argvals'):
            self._argvals.compatible_with(new_values)
        self._values = new_values

    ###########################################################################

    ###########################################################################
    # Methods
    def to_long(self) -> pd.DataFrame:
        """Convert the data to long format.

        This function transform a IrregularFunctionalData object into pandas
        DataFrame. It uses the long format to represent the
        IrregularFunctionalData object as a dataframe. This is a helper
        function as it might be easier for some computation, e.g., smoothing of
        the mean and covariance functions to have a long format.

        Returns
        -------
        pd.DataFrame
            The data in a long format.

        Examples
        --------
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
        for idx, obs in enumerate(self):
            cur_argvals = obs.argvals[idx]
            cur_values = obs.values[idx]
            sampling_points = list(itertools.product(*cur_argvals.values()))

            temp = pd.DataFrame(sampling_points)
            temp.columns = list(cur_argvals.keys())
            temp['id'] = idx
            temp['values'] = cur_values.flatten()
            temp_list.append(temp)
        return pd.concat(temp_list, ignore_index=True)

    def smooth(
        self,
        points: Optional[DenseArgvals] = None,
        kernel_name: str = "epanechnikov",
        bandwidth: Optional[float] = None,
        degree: int = 1
    ) -> DenseFunctionalData:
        """Smooth the data.

        This function smooths each curves individually. Based on [1]_, it fits
        a local smoother to the data (the argument ``degree`` controls the
        degree of the local fits).

        Parameters
        ----------
        points: Optional[DenseArgvals], default=None
            Points at which the curves are estimated. The default is None,
            meaning we use the argvals as estimation points.
        kernel_name: str, default="epanechnikov"
            Kernel name used as weight (`gaussian`, `epanechnikov`, `tricube`,
            `bisquare`).
        bandwidth: float, default=None
            Strictly positive. Control the size of the associated neighborhood.
            If ``bandwidth == None``, it is assumed that the curves are twice
            differentiable and the bandwidth is set to :math:`n^{-1/5}` where
            :math:`n` is the number of sampling points per curve [2]_. Be
            careful that it will not work if the curves are not sampled on
            :math:`[0, 1]`.
        degree: int, default=1
            Degree of the local polynomial to fit. If ``degree = 0``, we fit
            the local constant estimator (equivalent to the Nadaraya-Watson
            estimator). If ``degree = 1``, we fit the local linear estimator.
            If ``degree = 2``, we fit the local quadratic estimator.

        Returns
        -------
        DenseFunctionalData
            A smoothed version of the data.

        References
        ----------
        .. [1] Zhang, J.-T. and Chen J. (2007), Statistical Inferences for
            Functional Data, The Annals of Statistics, Vol. 35, No. 3.
        .. [2] Tsybakov, A.B. (2008), Introduction to Nonparametric Estimation.
            Springer Series in Statistics.

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
        if bandwidth is None:
            n_points = np.mean([obs for obs in self.n_points.values()])
            bandwidth = n_points**(-1 / 5)

        points_mat = _cartesian_product(*points.values())

        lp = LocalPolynomial(
            kernel_name=kernel_name, bandwidth=bandwidth, degree=degree
        )

        smooth = np.zeros((self.n_obs, *points.n_points))
        for idx, obs in enumerate(self):
            argvals_mat = _cartesian_product(*obs.argvals[idx].values())
            smooth[idx, :] = lp.predict(
                y=obs.values[idx],
                x=argvals_mat,
                x_new=points_mat
            ).reshape(smooth.shape[1:])
        return DenseFunctionalData(points, DenseValues(smooth))

    def mean(
        self,
        points: Optional[DenseArgvals] = None,
        smooth: bool = True,
        **kwargs
    ) -> DenseFunctionalData:
        """Compute an estimate of the mean.

        This function computes an estimate of the mean curve of a
        IrregularFunctionalData object. The curves are not sampled on a common
        grid. We implement the methodology from [1]_.

        Parameters
        ----------
        points: Optional[DenseArgvals], default=None
            The sampling points at which the mean is estimated. If `None`, the
            DenseArgvals of the DenseFunctionalData is used. If `smooth` is
            False, the DenseArgvals of the DenseFunctionalData is used.
        smooth: bool, default=True
            Not used in this context. The mean curve is always smoothed for
            IrregularFunctionalData.
        **kwargs:
            kernel_name: str, default='epanechnikov'
                Name of the kernel used for local polynomial smoothing.
            degree: int, default=1
                Degree used for local polynomial smoothing.
            bandwidth: float
                Bandwidth used for local polynomial smoothing. The default
                bandwitdth is set to be the number of sampling points to the
                power :math:`-1/5` [2]_.

        Returns
        -------
        DenseFunctionalData
            An estimate of the mean as a DenseFunctionalData object.

        References
        ----------
        .. [1] Cai, T.T., Yuan, M., (2011), Optimal estimation of the mean
            function based on discretely sampled functional data: Phase
            transition. The Annals of Statistics 39, 2330-2355.
        .. [2] Tsybakov, A.B. (2008), Introduction to Nonparametric Estimation.
            Springer Series in Statistics.

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
        bandwidth = kwargs.get('bandwidth', None)
        if bandwidth is None:
            n_points = np.mean([obs for obs in self.n_points.values()])
            bandwidth = n_points**(-1 / 5)

        fdata_long = self.to_long()
        x = fdata_long.drop(['id', 'values'], axis=1, inplace=False).values
        y = fdata_long['values'].values
        points_mat = _cartesian_product(*points.values())

        lp = LocalPolynomial(
            kernel_name=kwargs.get('kernel_name', 'epanechnikov'),
            bandwidth=bandwidth,
            degree=kwargs.get('degree', 1)
        )
        pred = lp.predict(y=y, x=x, x_new=points_mat).reshape(points.n_points)

        self._mean = DenseFunctionalData(points, DenseValues(pred[np.newaxis]))
        return self._mean

    def inner_product(
        self,
        method: str = 'trapz'
    ) -> npt.NDArray[np.float64]:
        r"""Compute the inner product matrix of the data.

        The inner product matrix is a ``n_obs`` by ``n_obs`` matrix where each
        entry is defined as

        .. math::
            \langle x, y \rangle = \int_{\mathcal{T}} x(t)y(t)dt,
            t \in \mathcal{T},

        where :math:`\mathcal{T}` is a one- or multi-dimensional domain.

        Parameters
        ----------
        method: str, {'simpson', 'trapz'}, default = 'trapz'
            The method used to integrated.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_obs)
            Inner product matrix of the data.

        Raises
        ------
        NotImplementedError
            Not implement for higher-dimensional data.

        References
        ----------
        .. [1] Benko, M., Härdle, W., Kneip, A., (2009), Common functional
            principal components. The Annals of Statistics 37, 1-34.

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
        >>> fdata.inner_product()
        array(
            [
                [42. , 55. , 55.5],
                [55. , 74. , 73. ],
                [55.5, 73. , 80.5]
            ]
        )

        """
        if self.n_dimension > 1:
            raise NotImplementedError(
                "Only implemented for one-dimensional irregular ",
                "functional data."
            )

        dense_argvals = self.argvals.to_dense()['input_dim_0']
        new_values = np.zeros((self.n_obs, len(dense_argvals)))
        for idx, obs in enumerate(self):
            tt = obs.argvals[idx]['input_dim_0']
            intervals = (tt + _shift(tt, -1)) / 2
            indices = np.searchsorted(intervals, dense_argvals)
            new_values[idx, :] = obs.values[idx][indices]

        data_smooth = DenseFunctionalData(
            DenseArgvals({'input_dim_0': dense_argvals}),
            DenseValues(new_values)
        )
        return data_smooth.inner_product(method=method)

    def norm(
        self,
        squared: bool = False,
        method: str = 'trapz',
        use_argvals_stand: bool = False
    ) -> npt.NDArray[np.float64]:
        r"""Norm of each observation of the data.

        For each observation in the data, it computes its norm [1]_ defined as

        .. math::
            \lvert\lvert f \rvert\rvert = \left(\int_{\mathcal{T}}
            f(t)^2dt\right)^{1\2}, t \in \mathcal{T},

        Parameters
        ----------
        squared: bool, default=False
            If `True`, the function calculates the squared norm, otherwise the
            result is not squared.
        method: str, {'simpson', 'trapz'}, default = 'trapz'
            The method used to integrated.
        use_argvals_stand: bool, default=False
            Use standardized argvals to compute the normalization of the data.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs,)
            The norm of each observations.

        References
        ----------
        .. [1] Ramsey, J. O. and Silverman, B. W. (2005), Functional Data
            Analysis, Springer Science, Chapter 2.

        Raises
        ------
        NotImplementedError
            Currently not implemented.

        """
        raise NotImplementedError()

    def normalize(
        self,
        use_argvals_stand: bool = False
    ) -> Tuple[FunctionalData, float]:
        r"""Normalize the data.

        The normalization is performed by divising each functional datum by
        :math:`w_j = \int_{T} Var(X(t))dt`.

        Parameters
        ----------
        use_argvals_stand: bool, default=False
            Use standardized argvals to compute the normalization of the data.

        Returns
        -------
        Tuple[DenseFunctionalData, float]
            The normalized data.

        TODO: Add other normalization schames and Add the possibility to
        normalize multidemsional data

        Raises
        ------
        NotImplementedError
            Currently not implemented.

        """
        raise NotImplementedError()

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
        DenseFunctionalData
            An object of the class DenseFunctionalData

        TODO: Consider removing this

        """
        new_argvals = self.argvals.to_dense()
        new_values = np.full(
            (self.n_obs,) + new_argvals.n_points, np.nan
        )

        # Create the index definition domain for each of the observation
        index_obs = {}
        for obs in self.values.keys():
            index_obs_dim = []
            for dim in new_argvals.keys():
                _, idx, _ = np.intersect1d(
                    new_argvals[dim],
                    self.argvals[obs][dim],
                    return_indices=True
                )
                index_obs_dim.append(idx)
            index_obs[obs] = index_obs_dim

        # Create mask arrays
        mask_obs = {
            obs: np.full(new_argvals.n_points, False)
            for obs in self.values.keys()
        }
        for obs in self.values.keys():
            mask_obs[obs][tuple(np.meshgrid(*index_obs[obs]))] = True

        # Assign values
        for obs in self.values.keys():
            new_values[obs][mask_obs[obs]] = self.values[obs].flatten()

        return DenseFunctionalData(
            DenseArgvals(new_argvals), DenseValues(new_values)
        )

    def covariance(
        self,
        mean: Optional[IrregularFunctionalData] = None,
        smooth: Optional[str] = None,
        **kwargs
    ) -> IrregularFunctionalData:
        """Compute an estimate of the covariance.

        Raises
        ------
        NotImplementedError
            Currently not implemented.

        TODO: Implement this function.

        """
        raise NotImplementedError()

    ###########################################################################


###############################################################################
# Class MultivariateFunctionalData
class MultivariateFunctionalData(UserList[Type[FunctionalData]]):
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
    data: List[Type[FunctionalData]]
        The list containing the elements of the MultivariateFunctionalData.

    TODO: Loop through MultivariateFunctionalData and through components.

    """

    ###########################################################################
    # Magic methods
    def __init__(
        self,
        initlist: List[Type[FunctionalData]]
    ) -> None:
        """Initialize MultivariateFunctionalData object."""
        self.data = initlist

    def __repr__(self) -> str:
        """Override print function."""
        return (
            f"Multivariate functional data object with {self.n_functional}"
            f" functions of {self.n_obs} observations."
        )

    def __getitem__(
        self,
        index: int
    ) -> MultivariateFunctionalData:
        """Overrride getitem function, called when self[index].

        Parameters
        ----------
        index: int
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
        """Get the dimension of the functional data.

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
        List[Dict[str, int]]
            A list of dictionary with the same shape than argvals with the
            number of sampling points along each axis for each function.

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
            FunctionalData._check_same_nobs(*self, item)
            self.data.append(item)

    def extend(self, other: Iterable[Type[FunctionalData]]) -> None:
        """Extend the list of FunctionalData by appending from iterable."""
        super().extend(other)

    def insert(self, i: int, item: Type[FunctionalData]) -> None:
        """Insert an item `item` at a given position `i`."""
        FunctionalData._check_same_nobs(*self, item)
        self.data.insert(i, item)

    def remove(self, item: Type[FunctionalData]) -> None:
        """Remove the first item from `self` where value is `item`."""
        raise NotImplementedError

    def pop(self, i: int = -1) -> Type[FunctionalData]:
        """Remove the item at the given position in the list, and return it."""
        return super().pop(i)

    def clear(self) -> None:
        """Remove all items from the list."""
        super().clear()

    def reverse(self) -> None:
        """Reserve the elements of the list in place."""
        super().reverse()

    def copy(self) -> MultivariateFunctionalData:
        """Return a shallow copy of the list."""
        return super().copy()

    ###########################################################################
    # Methods
    def mean(
        self,
        smooth: Optional[str] = None,
        **kwargs
    ) -> MultivariateFunctionalData:
        """Compute an estimate of the mean.

        Parameters
        ----------
        smooth: Optional[str], default=None
            Name of the smoothing method. Currently, not implemented.

        Returns
        -------
        MultivariateFunctionalData
            An estimate of the mean as a MultivariateFunctionalData object
            with a concatenation of the self.argvals as argvals and one
            observation.

        """
        return MultivariateFunctionalData(
            [fdata.mean(smooth, **kwargs) for fdata in self.data]
        )

    def covariance(
        self,
        mean: Optional[MultivariateFunctionalData] = None,
        smooth: Optional[str] = None,
        **kwargs
    ) -> MultivariateFunctionalData:
        """Compute an estimate of the covariance.

        Parameters
        ----------
        mean: Optional[MultivariateFunctionalData], default=None
            An estimate of the mean of self. If None, an estimate is computed.
        smooth: Optional[str], default=None
            Name of the smoothing method to use. Currently, not implemented.

        Returns
        -------
        MultivariateFunctionalData
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

    def inner_product(self) -> npt.NDArray[np.float64]:
        r"""Compute the inner product matrix of the data.

        The inner product matrix is a ``n_obs`` by ``n_obs`` matrix where each
        entry is defined as

        .. math::
            \langle\langle x, y \rangle\rangle =
            \sum_{p = 1}^P \int_{\mathcal{T}_k} x^{(p)}(t)y^{(p)}(t)dt,
            t \in \mathcal{T}.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_obs)
            Inner product matrix of the data.

        Examples
        --------
        >>> argvals = {'input_dim_0': np.array([0., 0.25, 0.5 , 0.75])}
        >>> values = np.array(
        ...     [
        ...         [ 2.48466259, -3.38397716, -1.2367073 , -1.85052901],
        ...         [ 1.44853118,  0.67716255,  1.79711043,  4.76950236],
        ...         [-5.13173463,  0.35830122,  0.56648942, -0.20965252]
        ...     ]
        ... )
        >>> data_1D = DenseFunctionalData(argvals, values)

        >>> argvals = {
        ...     'input_dim_0': np.array([0.  , 0.25, 0.5 , 0.75]),
        ...     'input_dim_1': np.array([0.  , 0.25, 0.5 , 0.75])
        ... }
        >>> values = np.array(
        ...     [
        ...         [
        ...             [6.30864764, -18.37912204, 6.15515232, 29.8027036],
        ...             [-6.076622, -15.48586803, -11.39997792, 8.40599319],
        ...             [-20.4094798, -1.3872093, -0.59922597, -6.42013363],
        ...             [5.78626375, -1.83874696, -0.87225549, 2.75000303]
        ...         ],
        ...         [
        ...             [-4.83576968, 18.85512513, -18.73086523, 15.1511348],
        ...             [-24.41254888, 12.37333951, 28.85176939, 16.41806885],
        ...             [-10.02681278, 14.76500118, 1.83114017, -2.78985647],
        ...             [4.29268032, 8.1781319, 30.10132687, -0.72828334]
        ...         ],
        ...         [
        ...             [-5.85921132, 1.85573561, -5.11291405, -12.89441767],
        ...             [-4.79384081, -0.93863074, 18.81909033, 4.55041973],
        ...             [-13.27810529, 28.08961819, -13.79482673, 35.25677906],
        ...             [9.10058173, -16.43979436, -11.88561292, -5.86481318]
        ...         ]
        ...     ]
        ... )
        >>> data_2D = DenseFunctionalData(argvals, values)
        >>> data = MultivariateFunctionalData([data_1D, data_2D])
        >>> data.inner_product()
        array(
            [
                [ 72.37627198, -28.54691325, -19.7335636 ],
                [-28.54691325, 166.61824532,  50.66329182],
                [-19.7335636 ,  50.66329182, 151.2780517 ]
            ]
        )

        """
        if not all(
            [isinstance(data, DenseFunctionalData) for data in self.data]
        ):
            raise TypeError(
                "All the univariate data must be DenseFunctionalData"
            )
        return np.sum([data.inner_product() for data in self.data], axis=0)

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
        MultivariateFunctionalData
            The concatenation of self and data.

        """
        new = [
            FunctionalData.concatenate([d1, d2]) for d1, d2 in zip(self, data)
        ]
        return MultivariateFunctionalData(new)

    def normalize(
        self,
        use_argvals_stand: bool = False
    ) -> Tuple[MultivariateFunctionalData, npt.NDArray[np.float64]]:
        r"""Normalize the data.

        The normalization is performed by divising each functional datum by
        :math:`w_j = \int_{T} Var(X(t))dt`.

        Parameters
        ----------
        use_argvals_stand: bool, default=False
            Use standardized argvals to compute the normalization of the data.

        Returns
        -------
        Tuple[MultivariateFunctionalData, npt.NDArray[np.float64]]
            The normalized data.

        References
        ----------
        Happ and Greven, Multivariate Functional Principal Component Analysis
        for Data Observed on Different (Dimensional Domains), Journal of the
        American Statistical Association.

        """
        normalization = [
            data_uni.normalize(use_argvals_stand=use_argvals_stand)
            for data_uni in self
        ]
        data_norm = [data for data, _ in normalization]
        weights = np.array([weight for _, weight in normalization])
        return MultivariateFunctionalData(data_norm), weights

    ###########################################################################


##############################################################################
# Functional data manipulation
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
    DenseFunctionalData
        The tensor product between data1 and data2. It contains data1.n_obs *
        data2.n_obs observations.

    Notes
    -----
    TODO:
    * Add tests.

    """
    arg = {
        'input_dim_0': data1.argvals['input_dim_0'],
        'input_dim_1': data2.argvals['input_dim_0']
    }
    val = [_outer(i, j) for i in data1.values for j in data2.values]
    return DenseFunctionalData(DenseArgvals(arg), DenseValues(np.array(val)))
