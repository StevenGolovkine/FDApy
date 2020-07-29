#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""
Module for the definition of FunctionalData types.

This modules is used to defined different types of functional data. The
different types are: Univariate Functional Data, Irregular Functional data and
Multivariate Functional Data.
"""
import numpy as np

from abc import ABC, abstractmethod

from ..misc.utils import get_axis_dimension, get_dict_dimension, get_obs_shape


###############################################################################
# Checkers for parameters
def _check_len(argv1, argv2):
    """Raise an arror if `argv1` and `argv2` do not have the same length."""
    if len(argv1) != len(argv2):
        raise ValueError(f"{type(argv1).__name__} and {type(argv2).__name__}"
                         " must have the same length.")
    return None


def _check_dict_dimension(argv1, argv2, axis=0):
    """Raise an error in case of dimension conflicts along the `axis`.

    An error is raised when elements of `argv1` and `argv2`, which are assumed
    to be dictionary, do not have the length along the `axis`. Elements are
    assumed to be numpy.ndarray.
    """
    has_len = [i.shape[0] == get_axis_dimension(j, axis)
               for i, j in zip(argv1.values(), argv2.values())]
    if not np.all(has_len):
        raise ValueError(f"Dimension are not the same.")
    return None


def _check_dict_array(argv_dict, argv_array):
    """Raise an error in case of dimension conflicts between the arguments.

    An error is raised when `argv_dict` (a dictionary) and `argv_array`
    (a np.ndarray) do not have coherent common dimensions. The first dimension
    of `arg_array` is assumed to represented the number of observation."""
    dim_dict = get_dict_dimension(argv_dict)
    dim_array = argv_array.shape[1:]
    if dim_dict != dim_array:
        raise ValueError(f"{argv_dict.__name__} and {argv_array.__name__}"
                         " do not have coherent dimension.")


def _check_dict_dict(argv1, argv2):
    """Raise an error in case of dimension conflicts between the arguments.

    An error is raised when `argv1` (a nested dictonary) and `argv2` (a
    dictionary) do not have coherent common dimensions."""
    has_obs_shape = [obs.shape == get_obs_shape(argv1, idx)
                     for idx, obs in argv2.items()]
    if not np.all(has_obs_shape):
        raise ValueError(f"{argv1.__name} and {argv2.__name__} do not"
                         " have coherent dimension.")


def _check_type(argv, category):
    """Raise an error if `argv` is not of type category."""
    if not isinstance(argv, category):
        raise ValueError(f"Argument must be {category.__name__}, not"
                         f" {type(argv).__name__}")
    return None


def _check_dict_type(argv, category):
    """Raise an error if all elements of `argv` are not of type `category`."""
    is_cat = [isinstance(obj, category) for obj in argv.values()]
    if not np.all(is_cat):
        raise ValueError(f"Argument values must be {category.__name__}")
    return None


def _check_dict_len(argv):
    """Raise an error if all elements of `argv` do not have equal length."""
    lengths = [len(obj) for obj in argv.values()]
    if len(set(lengths)) > 1:
        raise ValueError("The number of observations is different across the"
                         " dimensions.""")
    return None

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
        """Function call when self[index]."""
        pass

    def __add__(self, obj):
        """Override add function."""
        pass

    def __sub__(self, obj):
        """Override sub function."""
        pass

    def __mul__(self, obj):
        """Overrude mul function."""
        pass

    def __rmul__(self, obj):
        """Override rmul function."""
        return self * obj

    @property
    def argvals(self):
        """Getter for argvals."""
        return self._argvals

    @argvals.setter
    @abstractmethod
    def argvals(self, new_argvals):
        pass

    @property
    def argvals_stand(self):
        """Getter for argvals_stand."""
        return self._argvals_stand

    @property
    def values(self):
        """Getter for values."""
        return self._values

    @values.setter
    @abstractmethod
    def values(self, new_values):
        pass

    @property
    def category(self):
        """Getter for category."""
        return self._category

    @category.setter
    def category(self, new_category):
        self._category = new_category

    @property
    def n_obs(self):
        """Number of observations of the functional data.

        Returns
        -------
        n_obs: int
            Number of observations within the functional data.

        """
        return len(self.values)

    @property
    @abstractmethod
    def range_obs(self):
        pass

    @property
    def n_dim(self):
        """Number of input dimension of the functional data.

        Returns
        -------
        n_dim: int
            Number of input dimension with the functional data.
        """
        return len(self.argvals)

    @property
    @abstractmethod
    def range_dim(self):
        pass

    @property
    @abstractmethod
    def shape(self):
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
        :math:`(n, m_1, \dots, m_p)`. It should not contain any missing values.

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

    def __init__(self, argvals, values):
        """Initialize UnivariateFunctionalData object."""
        super().__init__(argvals, values, 'univariate')

    def __getitem__(self, index):
        """Function call when self[index].

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
        self._check_argvals(new_argvals)
        if hasattr(self, 'values'):
            self._check_argvals_values(new_argvals, self.values)
        self._argvals = new_argvals

    @property
    def values(self):
        """Getter for values."""
        return super().values

    @values.setter
    def values(self, new_values):
        self._check_values(new_values)
        if hasattr(self, 'argvals'):
            self._check_argvals_values(self.argvals, new_values)
        self._values = new_values

    @property
    def range_obs(self):
        """Range of the observations of the object.

        Returns
        -------
        min, max: tuple
            Tuple containing the mimimum and maximum values taken by all the
            observations for the object.
        """
        return np.min(self.values), np.max(self.values)

    @property
    def range_dim(self):
        """Range of the `argvals` for each of the dimension.

        Returns
        -------
        ranges: dict
            Dictionary containing the range of the argvals for each of the
            input dimension."""
        return {idx: (min(argval), max(argval))
                for idx, argval in self.argvals.items()}

    @property
    def shape(self):
        r"""Shape of the data for each dimension.

        Returns
        -------
        shape: dict
            Dictionary containing the number of points for each of the
            dimension. It corresponds to :math:`m_j` for
            :math:`0 \leq j \leq p`.
        """
        return {idx: len(dim) for idx, dim in self.argvals.items()}


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

    Examples:
    ---------
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

    def __init__(self, argvals, values):
        """Initialize IrregularFunctionalData object."""
        super().__init__(argvals, values, 'irregular')

    def __getitem__(self, index):
        """Function call when self[index].

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
        self._check_argvals(new_argvals)
        if hasattr(self, 'values'):
            self._check_argvals_values(new_argvals, self.values)
        self._argvals = new_argvals

    @property
    def values(self):
        """Getter for values."""
        return super().values

    @values.setter
    def values(self, new_values):
        self._check_values(new_values)
        if hasattr(self, 'argvals'):
            self._check_argvals_values(self.argvals, new_values)
        self._values = new_values

    @property
    def range_obs(self):
        """Range of the observations of the object.

        Returns
        -------
        min, max: tuple
            Tuple containing the mimimum and maximum values taken by all the
            observations for the object.
        """
        ranges = [(np.min(obs), np.max(obs)) for obs in self.values.values()]
        return min(min(ranges)), max(max(ranges))

    @property
    def range_dim(self):
        """Range of the `argvals` for each of the dimension.

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
        r"""Shape of the data for each dimension.

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
