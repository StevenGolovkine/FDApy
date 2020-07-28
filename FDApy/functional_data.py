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

from .utils import get_axis_dimension


###############################################################################
# Checkers for parameters
def _check_len(argv1, argv2):
    """Raise an arror if `argv1` and `argv2` do not have the same length."""
    if len(argv1) != len(argv2):
        raise ValueError(f"""{type(argv1).__name__} and {type(argv2).__name__}
                         must have the same length.""")
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
        raise ValueError(f"""Dimension are not the same.""")
    return None


def _check_type(argv, category):
    """Raise an error if `argv` is not of type category."""
    if not isinstance(argv, category):
        raise ValueError(f"""Argument must be {category.__name__}, not
                         {type(argv).__name__}""")
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
        raise ValueError("""The number of observations is different across the
                         dimensions.""")
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
    def _check_argv(argv):
        _check_type(argv, dict)

    @staticmethod
    def _check_values(values):
        _check_dict_len(values)

    @staticmethod
    @abstractmethod
    def _check_argvals_values(argvals, values):
        _check_len(argvals, values)

    def __init__(self, argvals, values, category):
        """Initialize FunctionalData object."""
        super().__init__()
        self.argvals = argvals
        self.values = values
        self.category = category

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
    def n_dim(self):
        """Number of dimensions of the functional data.

        Returns
        -------
        n_dim: int
            Number of dimension within the functional data.

        """
        return len(self.values)


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
                           [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]])

    >>> DenseFunctionalData(argvals, values)
    """

    @staticmethod
    def _check_argv(argv):
        """Check the user provided `argv`."""
        FunctionalData._check_argv(argv)
        _check_dict_type(argv, np.ndarray)

    @staticmethod
    def _check_argvals_values(argvals, values):
        """Check the compatibility of argvals and values."""
        FunctionalData._check_argvals_values(argvals, values)
        _check_dict_dimension(argvals, values, axis=1)

    def __init__(self, argvals, values):
        """Initialize UnivariateFunctionalData object."""
        super().__init__(argvals, values, 'univariate')

    @property
    def argvals(self):
        """Getter for argvals."""
        return super().argvals

    @argvals.setter
    def argvals(self, new_argvals):
        self._check_argv(new_argvals)
        if hasattr(self, 'values'):
            self._check_argvals_values(new_argvals, self.values)
        self._argvals = new_argvals

    @property
    def values(self):
        """Getter for values."""
        return super().values

    @values.setter
    def values(self, new_values):
        self._check_argv(new_values)
        FunctionalData._check_values(new_values)
        if hasattr(self, 'argvals'):
            self._check_argvals_values(self.argvals, new_values)
        self._values = new_values


###############################################################################
# Class IrregularFunctionalData

class IrregularFunctionalData(FunctionalData):
    """A class for defining Irregular Functional Data.

    Parameters
    ----------
    """

    @staticmethod
    def _check_argv(argv):
        """Check the user provided `argv`."""
        FunctionalData._check_argv(argv)
        for obj in argv.values():
            _check_type(obj, dict)
            _check_dict_type(obj, np.ndarray)

    @staticmethod
    def _check_argvals_values(argvals, values):
        """Check the compatibility of argvals and values."""
        FunctionalData._check_argvals_values(argvals, values)
        for points, obj in zip(argvals.values(), values.values()):
            _check_dict_dimension(points, obj, axis=0)

    def __init__(self, argvals, values):
        """Initialize IrregularFunctionalData object."""
        super().__init__(argvals, values, 'irregular')

    @property
    def argvals(self):
        """Getter for argvals."""
        return super().argvals

    @argvals.setter
    def argvals(self, new_argvals):
        self._check_argv(new_argvals)
        if hasattr(self, 'values'):
            self._check_argvals_values(new_argvals, self.values)
        self._argvals = new_argvals

    @property
    def values(self):
        """Getter for values."""
        return super().values

    @values.setter
    def values(self, new_values):
        self._check_argv(new_values)
        FunctionalData._check_values(new_values)
        if hasattr(self, 'argvals'):
            self._check_argvals_values(self.argvals, new_values)
        self._values = new_values
