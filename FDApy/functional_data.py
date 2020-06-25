#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""
Module for the definition of FunctionalData types.

This modules is used to defined different types of functional data. The
different types are: Univariate Functional Data, Irregular Functional data and
Multivariate Functional Data.
"""
import itertools
import numpy as np

from abc import ABC, abstractmethod


###############################################################################
# Checkers for parameters
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
        return self._category

    @category.setter
    def category(self, new_category):
        self._category = new_category

    @property
    def n_obs(self):
        """Number of observations within the functional data.

        Returns
        -------
        n_obs: int
            Number of observations within the functional data.

        """
        return len(self.values)


###############################################################################
# Class UnivariateFunctionalData

class UnivariateFunctionalData(FunctionalData):
    """A class for defining Univariate Functional Data.

    Parameters
    ----------
    """

    @staticmethod
    def _check_argv(argv):
        """Check the user provided `argv`."""
        FunctionalData._check_argv(argv)
        _check_dict_type(argv, np.ndarray)
        return argv

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
        self._argvals = new_argvals

    @property
    def values(self):
        """Getter for values."""
        return super().values

    @values.setter
    def values(self, new_values):
        self._check_argv(new_values)
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
        return argv

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
        self._argvals = new_argvals

    @property
    def values(self):
        """Getter for values."""
        return super().values

    @values.setter
    def values(self, new_values):
        self._check_argv(new_values)
        self._values = new_values
