#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Module for the definition of FunctionalData types.

This modules is used to defined different types of functional data. The
different types are: Univariate Functional Data, Irregular Functional data and
Multivariate Functional Data.
"""
import numpy as np
import pygam

from abc import ABC, abstractmethod
from collections import UserList

from sklearn.metrics import pairwise_distances

from ..preprocessing.smoothing.bandwidth import Bandwidth
from ..preprocessing.smoothing.local_polynomial import LocalPolynomial
from ..preprocessing.smoothing.smoothing_splines import SmoothingSpline
from ..misc.utils import get_dict_dimension_, get_obs_shape_
from ..misc.utils import integration_weights_, outer_
from ..misc.utils import range_standardization_


###############################################################################
# Checkers for parameters

def _check_dict_array(argv_dict, argv_array):
    """Raise an error in case of dimension conflicts between the arguments.

    An error is raised when `argv_dict` (a dictionary) and `argv_array`
    (a np.ndarray) do not have coherent common dimensions. The first dimension
    of `arg_array` is assumed to represented the number of observation.
    """
    dim_dict = get_dict_dimension_(argv_dict)
    dim_array = argv_array.shape[1:]
    if dim_dict != dim_array:
        raise ValueError(f"{argv_dict} and {argv_array} do not have coherent"
                         " dimension.")


def _check_dict_dict(argv1, argv2):
    """Raise an error in case of dimension conflicts between the arguments.

    An error is raised when `argv1` (a nested dictonary) and `argv2` (a
    dictionary) do not have coherent common dimensions.
    """
    has_obs_shape = [obs.shape == get_obs_shape_(argv1, idx)
                     for idx, obs in argv2.items()]
    if not np.all(has_obs_shape):
        raise ValueError(f"{argv1} and {argv2} do not"
                         " have coherent dimension.")


def _check_type(argv, category):
    """Raise an error if `argv` is not of type category."""
    if not isinstance(argv, category):
        raise TypeError(f"Argument must be FunctionalData, not"
                        f" {type(argv).__name__}")


def _check_dict_type(argv, category):
    """Raise an error if all elements of `argv` are not of type `category`."""
    is_cat = [isinstance(obj, category) for obj in argv.values()]
    if not np.all(is_cat):
        raise TypeError(f"Argument values must be {category.__name__}")


def _check_dict_len(argv):
    """Raise an error if all elements of `argv` do not have equal length."""
    lengths = [len(obj) for obj in argv.values()]
    if len(set(lengths)) > 1:
        raise ValueError("The number of observations is different across the"
                         " dimensions.""")


def _check_same_type(argv1, argv2):
    """Raise an error if `argv1` and `argv2` have different type."""
    if not isinstance(argv2, type(argv1)):
        raise TypeError(f"{argv1} and {argv2} do not have the same type.")


def _check_same_nobs(*argv):
    """Raise an arror if elements in argv have different number of obs."""
    n_obs = set(obj.n_obs for obj in argv)
    if len(n_obs) > 1:
        raise ValueError("Elements do not have the same number"
                         " of observations.")


def _check_same_ndim(argv1, argv2):
    """Raise an error if `argv1` and `argv2` have different number of dim."""
    if argv1.n_dim != argv2.n_dim:
        raise ValueError(f"{argv1} and {argv2} do not have the same number"
                         " of dimensions.")


def _check_argvals_equality_dense(argv1, argv2):
    """Raise an error if `argv1` and `argv2` are not equal."""
    argvs_equal = all(np.array_equal(argv1[key], argv2[key]) for key in argv1)
    if not argvs_equal:
        raise ValueError(f"{argv1} and {argv2} do not have the same sampling"
                         " points.")


def _check_argvals_equality_irregular(argv1, argv2):
    """Raise an error if `argv1` and `argv2` are not equal."""
    temp = []
    for points1, points2 in zip(argv1.values(), argv2.values()):
        temp.append(all(np.array_equal(points1[key], points2[key])
                        for key in points1))

    argvs_equal = all(temp)
    if not argvs_equal:
        raise ValueError(f"{argv1} and {argv2} do not have the same sampling"
                         " points.")


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
    @abstractmethod
    def _check_argvals(argvals):
        _check_type(argvals, dict)

    @staticmethod
    @abstractmethod
    def _check_values(values):
        pass

    @staticmethod
    @abstractmethod
    def _check_argvals_values(argvals, values):
        pass

    @staticmethod
    @abstractmethod
    def _perform_computation(fdata1, fdata2, func):
        pass

    def __init__(self, argvals, values, category):
        """Initialize FunctionalData object."""
        super().__init__()
        self.argvals = argvals
        self.values = values
        self.category = category

    def __repr__(self):
        """Override print function."""
        return (f"{self.category.capitalize()} functional data object with"
                f" {self.n_obs} observations on a {self.n_dim}-dimensional"
                " support.")

    @abstractmethod
    def __getitem__(self, index):
        """Override getitem function, called when self[index]."""
        pass

    def __add__(self, obj):
        """Override add function."""
        return self._perform_computation(self, obj, np.add)

    def __sub__(self, obj):
        """Override sub function."""
        return self._perform_computation(self, obj, np.subtract)

    def __mul__(self, obj):
        """Overrude mul function."""
        return self._perform_computation(self, obj, np.multiply)

    def __rmul__(self, obj):
        """Override rmul function."""
        return self * obj

    def __truediv__(self, obj):
        """Override truediv function."""
        return self._perform_computation(self, obj, np.divide)

    def __floordiv__(self, obj):
        """Override floordiv function."""
        return self / obj

    @property
    def argvals(self):
        """Getter for argvals."""
        return self._argvals

    @argvals.setter
    def argvals(self, new_argvals):
        self._check_argvals(new_argvals)
        if hasattr(self, 'values'):
            self._check_argvals_values(new_argvals, self.values)
        self._argvals = new_argvals

    @property
    def argvals_stand(self):
        """Getter for argvals_stand."""
        return self._argvals_stand

    @argvals_stand.setter
    def argvals_stand(self, new_argvals_stand):
        self._argvals_stand = new_argvals_stand

    @property
    def values(self):
        """Getter for values."""
        return self._values

    @values.setter
    def values(self, new_values):
        self._check_values(new_values)
        if hasattr(self, 'argvals'):
            self._check_argvals_values(self.argvals, new_values)
        self._values = new_values

    @property
    def category(self):
        """Getter for category."""
        return self._category

    @category.setter
    def category(self, new_category):
        self._category = new_category

    @property
    def n_obs(self):
        """Get the number of observations of the functional data.

        Returns
        -------
        n_obs: int
            Number of observations within the functional data.

        """
        return len(self.values)

    @property
    @abstractmethod
    def range_obs(self):
        """Get the range of the observations of the object."""
        pass

    @property
    def n_dim(self):
        """Get the number of input dimension of the functional data.

        Returns
        -------
        n_dim: int
            Number of input dimension with the functional data.

        """
        return len(self.argvals)

    @property
    @abstractmethod
    def n_points(self):
        """Get the mean number of sampling points."""
        pass

    @property
    @abstractmethod
    def range_dim(self):
        """Range of the `argvals` for each of the dimension."""
        pass

    @property
    @abstractmethod
    def shape(self):
        """Shape of the data for each dimension."""
        pass

    @abstractmethod
    def is_compatible(self, fdata):
        """Check if `fdata` is compatible with `self`."""
        _check_same_type(self, fdata)
        _check_same_nobs(self, fdata)
        _check_same_ndim(self, fdata)

    @abstractmethod
    def mean(self, smooth=None, **kwargs):
        """Compute an estimate of the mean."""
        pass

    @abstractmethod
    def covariance(self, mean=None, smooth=None, **kwargs):
        """Compute an estimate of the covariance."""
        pass

    @abstractmethod
    def smooth(self, points, neighborhood, points_estim=None, degree=0,
               kernel="epanechnikov", bandwidth=None):
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
    >>> argvals = {'input_dim_0': np.array([1, 2, 3, 4]),
                   'input_dim_1': np.array([5, 6, 7])}

    >>> values = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                   [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]],
                   [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]],
                   [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]],
                   [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]])

    >>> DenseFunctionalData(argvals, values)

    """

    @staticmethod
    def _check_argvals(argvals):
        """Check the user provided `argvals`."""
        FunctionalData._check_argvals(argvals)
        _check_dict_type(argvals, np.ndarray)

    @staticmethod
    def _check_values(values):
        """Check the use provided `values`."""
        _check_type(values, np.ndarray)

    @staticmethod
    def _check_argvals_values(argvals, values):
        """Check the compatibility of argvals and values."""
        _check_dict_array(argvals, values)

    @staticmethod
    def _perform_computation(fdata1, fdata2, func):
        """Perform computation defined by `func`."""
        if fdata1.is_compatible(fdata2):
            new_values = func(fdata1.values, fdata2.values)
        return DenseFunctionalData(fdata1.argvals, new_values)

    def __init__(self, argvals, values):
        """Initialize UnivariateFunctionalData object."""
        super().__init__(argvals, values, 'univariate')

    def __getitem__(self, index):
        """Overrride getitem function, called when self[index].

        Parameters
        ----------
        index: int
            The observation(s) of the object to retrive.

        Returns
        -------
        data: DenseFunctionalData object
            The selected observation(s) as DenseFunctionalData object.

        """
        argvals = self.argvals
        values = self.values[index]

        if len(argvals) == len(values.shape):
            values = values[np.newaxis]
        return DenseFunctionalData(argvals, values)

    @property
    def argvals(self):
        """Getter for argvals."""
        return super().argvals

    @argvals.setter
    def argvals(self, new_argvals):
        super(DenseFunctionalData, self.__class__).\
            argvals.fset(self, new_argvals)

        argvals_stand = {}
        for dim, points in new_argvals.items():
            argvals_stand[dim] = range_standardization_(points)
        self.argvals_stand = argvals_stand

    @property
    def range_obs(self):
        """Get the range of the observations of the object.

        Returns
        -------
        min, max: tuple
            Tuple containing the mimimum and maximum values taken by all the
            observations for the object.

        """
        return np.min(self.values), np.max(self.values)

    @property
    def n_points(self):
        """Get the mean number of sampling points.

        Returns
        -------
        n_points: dict
            A dictionary with the same shape than argavls with the number of
            sampling points along each axis.

        """
        return {i: len(points) for i, points in self.argvals.items()}

    @property
    def range_dim(self):
        """Get the range of the `argvals` for each of the dimension.

        Returns
        -------
        ranges: dict
            Dictionary containing the range of the argvals for each of the
            input dimension.

        """
        return {idx: (min(argval), max(argval))
                for idx, argval in self.argvals.items()}

    @property
    def shape(self):
        r"""Get the shape of the data for each dimension.

        Returns
        -------
        shape: dict
            Dictionary containing the number of points for each of the
            dimension. It corresponds to :math:`m_j` for
            :math:`0 \leq j \leq p`.

        """
        return {idx: len(dim) for idx, dim in self.argvals.items()}

    def as_irregular(self):
        """Convert `self` from Dense to Irregular functional data.

        Coerce a DenseFunctionalData object into an IrregularFunctionalData
        object.

        Returns
        -------
        obj: IrregularFunctionalData
            An object of the class IrregularFunctionalData

        """
        new_argvals = dict.fromkeys(self.argvals.keys(), {})
        for dim in new_argvals.keys():
            temp = {}
            for idx in range(self.n_obs):
                temp[idx] = self.argvals[dim]
            new_argvals[dim] = temp

        new_values = {}
        for idx in range(self.n_obs):
            new_values[idx] = self.values[idx]

        return IrregularFunctionalData(new_argvals, new_values)

    def is_compatible(self, fdata):
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
            If the objects are compatible.

        """
        super().is_compatible(fdata)
        _check_argvals_equality_dense(self.argvals, fdata.argvals)
        return True

    def mean(self, smooth=None, **kwargs):
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
                neigh = kwargs.get('neighborhood',
                                   np.int(p * np.exp(-(np.log(np.log(p)))**2)))
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

    def covariance(self, mean=None, smooth=None, **kwargs):
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
        Association, 93, 1403–1418.

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
            np.fill_diagonal(cov, None)
            cov = cov[~np.isnan(cov)]

            # Define train vector
            train_ = np.vstack((
                np.repeat(argvals, repeats=len(argvals)),
                np.tile(argvals, reps=len(argvals)))
            )

            train = train_[:, train_[0, :] != train_[1, :]]

            if smooth == 'LocalLinear':
                points = kwargs.get('points', 0.5)
                neigh = kwargs.get('neighborhood',
                                   np.int(p * np.exp(-(np.log(np.log(p)))**2)))
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
        weights = integration_weights_(argvals[lower:upper], method='trapz')
        nume = np.dot(weights, (var_hat - cov_diag)[lower:upper])
        self.var_noise = np.maximum(nume / argvals[upper] - argvals[lower], 0)

        new_argvals = {'input_dim_0': argvals, 'input_dim_1': argvals}
        return DenseFunctionalData(new_argvals, cov[np.newaxis])

    def smooth(self, points, neighborhood, points_estim=None, degree=0,
               kernel="epanechnikov", bandwidth=None):
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
        degree: int, default=2
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
        if self.n_dim != 1:
            raise NotImplementedError('Only one dimensional data can be'
                                      ' smoothed.')

        data = self.as_irregular()
        data_smooth = data.smooth(points, neighborhood,
                                  points_estim=points_estim,
                                  degree=degree,
                                  kernel=kernel,
                                  bandwidth=bandwidth)
        return data_smooth.as_dense()

    def pairwise_distance(self, metric='euclidean'):
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
        return pairwise_distances(self.values, metric=metric)

    def concatenate(self, data):
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
        return concatenate_(self, data)


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
    >>> argvals = {'input_dim_0': {
                        0: np.array([1, 2, 3, 4]),
                        1: np.array([2, 4])},
                   'input_dim_1': {
                        0: np.array([5, 6, 7]),
                        1: np.array([1, 2, 3])}
                  }

    >>> values = {0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]]),
                  1: np.array([[1, 2, 3], [1, 2, 3]])}

    >>> IrregularFunctionalData(argvals, values)

    """

    @staticmethod
    def _check_argvals(argvals):
        """Check the user provided `argvals`."""
        FunctionalData._check_argvals(argvals)
        for obj in argvals.values():
            _check_type(obj, dict)
            _check_dict_type(obj, np.ndarray)
        _check_dict_len(argvals)

    @staticmethod
    def _check_values(values):
        """Check the user provided `values`."""
        _check_type(values, dict)
        for obj in values.values():
            _check_type(obj, np.ndarray)

    @staticmethod
    def _check_argvals_values(argvals, values):
        """Check the compatibility of argvals and values."""
        _check_dict_dict(argvals, values)

    @staticmethod
    def _perform_computation(fdata1, fdata2, func):
        """Perform computation defined by `func`."""
        if fdata1.is_compatible(fdata2):
            new_values = {}
            for (idx, obs1), (_, obs2) in zip(fdata1.values.items(),
                                              fdata2.values.items()):
                new_values[idx] = func(obs1, obs2)
        return IrregularFunctionalData(fdata1.argvals, new_values)

    def __init__(self, argvals, values):
        """Initialize IrregularFunctionalData object."""
        super().__init__(argvals, values, 'irregular')

    def __getitem__(self, index):
        """Overrride getitem function, called when self[index].

        Parameters
        ----------
        index: int
            The observation(s) of the object to retrive.

        Returns
        -------
        data: IrregularFunctionalData object
            The selected observation(s) as IrregularFunctionalData object.

        """
        if isinstance(index, slice):
            indices = index.indices(self.n_obs)

            argvals = {}
            for idx, dim in self.argvals.items():
                argvals[idx] = {i: dim.get(i) for i in range(*indices)}
            values = {i: self.values.get(i) for i in range(*indices)}
        else:
            argvals = {idx: {index: points.get(index)}
                       for idx, points in self.argvals.items()}
            values = {index: self.values.get(index)}
        return IrregularFunctionalData(argvals, values)

    @property
    def argvals(self):
        """Getter for argvals."""
        return super().argvals

    @argvals.setter
    def argvals(self, new_argvals):
        super(IrregularFunctionalData, self.__class__).\
            argvals.fset(self, new_argvals)

        points = self.gather_points()
        argvals_stand = {}
        for dim, obss in new_argvals.items():
            max_x, min_x = np.max(points[dim]), np.min(points[dim])

            argvals_stand[dim] = {}
            for obs, point in obss.items():
                argvals_stand[dim][obs] = range_standardization_(point,
                                                                 max_x, min_x)
        self.argvals_stand = argvals_stand

    @property
    def range_obs(self):
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
    def n_points(self):
        """Get the mean number of sampling points.

        Returns
        -------
        n_points: dict
            A dictionary with the same shape than argavls with the number of
            sampling points along each axis.

        """
        n_points = {}
        for i, points in self.argvals.items():
            n_points[i] = np.mean([len(p) for p in points.values()])
        return n_points

    @property
    def range_dim(self):
        """Get the range of the `argvals` for each of the dimension.

        Returns
        -------
        ranges: dict
            Dictionary containing the range of the argvals for each of the
            input dimension.

        """
        ranges = {idx: list(argval.values())
                  for idx, argval in self.argvals.items()}
        return {idx: (min(map(min, dim)), max(map(max, dim)))
                for idx, dim in ranges.items()}

    @property
    def shape(self):
        r"""Get the shape of the data for each dimension.

        Returns
        -------
        shape: dict
            Dictionary containing the number of points for each of the
            dimension. It corresponds to :math:`m_j` for
            :math:`0 \leq j \leq p`.

        """
        return {idx: len(dim) for idx, dim in self.gather_points().items()}

    def gather_points(self):
        """Gather all the `argvals` for each of the dimensions separetely.

        Returns
        -------
        argvals: dict
            Dictionary containing all the unique observations points for each
            of the input dimension.

        """
        return {idx: np.unique(np.hstack(list(dim.values())))
                for idx, dim in self.argvals.items()}

    def as_dense(self):
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

    def is_compatible(self, fdata):
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
        _check_argvals_equality_irregular(self.argvals, fdata.argvals)
        return True

    def mean(self, smooth=None, **kwargs):
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
        mean_estim = np.nanmean(dense_self.values, axis=0, keepdims=True)
        return DenseFunctionalData(dense_self.argvals, mean_estim)

    def covariance(self, mean=None, smooth=None, **kwargs):
        """Compute an estimate of the covariance."""
        pass

    def smooth(self, points, neighborhood, points_estim=None, degree=0,
               kernel="epanechnikov", bandwidth=None):
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

        if bandwidth is None:
            band_obj = Bandwidth(points=points, neighborhood=neighborhood)
            bandwidth = band_obj(self).bandwidths

        argvals = self.argvals['input_dim_0'].values()
        values = self.values.values()
        smooth_argvals, smooth_values = {}, {}
        for i, (arg, val, b) in enumerate(zip(argvals, values, bandwidth)):
            if points_estim is None:
                points_estim = arg

            lp = LocalPolynomial(kernel_name=kernel,
                                 bandwidth=b,
                                 degree=degree)
            pred = lp.fit_predict(arg, val, points_estim)
            smooth_argvals[i] = points_estim
            smooth_values[i] = pred
        return IrregularFunctionalData({'input_dim_0': smooth_argvals},
                                       smooth_values)


###############################################################################
# Class MultivariateFunctionalData

class MultivariateFunctionalData(UserList):
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

    @staticmethod
    def _check_data(new_data):
        """Check the user provided `data`."""
        for obj in new_data:
            _check_type(obj, (DenseFunctionalData, IrregularFunctionalData))
        _check_same_nobs(*new_data)

    def __init__(self, initlist=None):
        """Initialize MultivariateFunctionalData object."""
        self.data = initlist

    def __repr__(self):
        """Override print function."""
        return (f"Multivariate functional data object with {self.n_functional}"
                f" functions of {self.n_obs} observations.")

    @property
    def data(self):
        """Getter for data."""
        return self._data

    @data.setter
    def data(self, new_data):
        if new_data is not None:
            self._check_data(new_data)
            self._data = new_data
        else:
            self._data = []

    @property
    def n_obs(self):
        """Get the number of observations of the functional data.

        Returns
        -------
        n_obs: int
            Number of observations within the functional data.

        """
        return self.data[0].n_obs if len(self) > 0 else 0

    @property
    def n_functional(self):
        """Get the number of functional data with `self`.

        Returns
        -------
        n_functional: int
            Number of functions in the list.

        """
        return len(self)

    @property
    def n_dim(self):
        """Get the dimension of the functional data.

        Returns
        -------
        dim: list
            List containing the dimension of each component in the functional
            data.

        """
        return [i.n_dim for i in self]

    @property
    def range_obs(self):
        """Get the range of the observations of the object.

        Returns
        -------
        (min, max): list of tuples
            List of tuples containing the mimimum and maximum values taken by
            all the observations for the object for each function.

        """
        return [i.range_obs for i in self]

    @property
    def n_points(self):
        """Get the mean number of sampling points.

        Returns
        -------
        n_points: list of dict
            A list of dictionary with the same shape than argvals with the
            number of sampling points along each axis for each function.

        """
        return [i.n_points for i in self]

    @property
    def range_points(self):
        """Get the range of the `argvals` for each of the dimension.

        Returns
        -------
        ranges: list of dict
            List of dictionary containing the range of the argvals for each of
            the input dimension for each function.

        """
        return [i.range_dim for i in self]

    @property
    def shape(self):
        r"""Get the shape of the data for each dimension.

        Returns
        -------
        shape: list of dict
            List of dictionary containing the number of points for each of the
            dimension for each function. It corresponds to :math:`m_j` for
            :math:`0 \leq j \leq p`.

        """
        return [i.shape for i in self]

    def append(self, item):
        """Add an item to `self`.

        Parameters
        ----------
        item: DenseFunctionalData or IrregularFunctionalData
            Item to add.

        """
        if len(self.data) == 0:
            self.data = [item]
        else:
            _check_same_nobs(self, item)
            self.data.append(item)

    def extend(self, other):
        """Extend the list of FunctionalData by appending from iterable."""
        super().extend(other)

    def insert(self, i, item):
        """Insert an item `item` at a given position `i`."""
        _check_same_nobs(self, item)
        self.data.insert(i, item)

    def remove(self, item):
        """Remove the first item from `self` where value is `item`."""
        raise NotImplementedError

    def pop(self, i=-1):
        """Remove the item at the given position in the list, and return it."""
        return super().pop(i)

    def clear(self):
        """Remove all items from the list."""
        super().clear()

    def index(self, item, *args):
        """Return first item of the list equald to x."""
        raise NotImplementedError

    def count(self, item):
        """Return the number of times `item` appears in the list."""
        raise NotImplementedError

    def sort(self, *args, **kwds):
        """Sort the items of the list in place."""
        raise NotImplementedError

    def reverse(self):
        """Reserve the elements of the list in place."""
        super().reverse()

    def copy(self):
        """Return a shallow copy of the list."""
        return super().copy()

    def get_obs(self):
        """Return a generator over the observation."""
        for idx in range(self.n_obs):
            yield MultivariateFunctionalData([obs[idx] for obs in self])

    def mean(self, smooth=None, **kwargs):
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
        return MultivariateFunctionalData([i.mean(smooth, **kwargs)
                                           for i in self])

    def covariance(self, mean=None, smooth=None, **kwargs):
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
                [i.covariance(m, smooth, **kwargs)
                    for i, m in zip(self, mean)])
        else:
            return MultivariateFunctionalData(
                [i.covariance(None, smooth, **kwargs) for i in self])

    def concatenate(self, data):
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
        new = [data1.concatenate(data2) for data1, data2 in zip(self, data)]
        return MultivariateFunctionalData(new)


##############################################################################
# Functional data manipulation

def concatenate_(*data):
    """Concatenate functional data.

    Compute multiple DenseFunctionalData into one. It works with higher
    dimension for the input data.

    Parameters
    ----------
    *data: DenseFunctionalData
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


def tensor_product_(data1, data2):
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
    val = [outer_(i, j) for i in data1.values for j in data2.values]
    return DenseFunctionalData(arg, np.array(val))
