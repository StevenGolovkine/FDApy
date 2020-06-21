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

    def __init__(self, argvals, values, category):
        """Initialize FunctionalData object."""
        super().__init__()
        self.argvals = argvals
        self.values = values
        self.category = category

    def __repr__(self):
        """Override print function."""
        pass

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
        return self._argvals

    @argvals.setter
    def argvals(self, new_argvals):
        self._argvals = new_argvals

    @property
    def argvals_stand(self):
        return self._argvals_stand

    @argvals_stand.setter
    def argvals_stand(self, new_argvals_stand):
        self._argvals_stand = new_argvals_stand

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        self._values = new_values

    @property
    def n_obs(self):
        """Number of observations within the functional data.

        Returns
        -------
        n_obs: int
            Number of observations within the functional data.

        """
        return len(self.values)

    @abstractmethod
    @property
    def n_points(self):
        """Number of sampling points within the functional data."""
        return [len(i) for i in self.argvals]


###############################################################################
# Class UnivariateFunctionalData

class UnivariateFunctionalData(FunctionalData):
    """A class for defining Univariate Functional Data.

    Parameters
    ----------
    """

    def __init__(self, argvals, values):
        """Initialize UnivariateFunctionalData object."""
        super().__init__(argvals, values, 'univariate')


###############################################################################
# Class IrregularFunctionalData

class IrregularFunctionalData(FunctionalData):
    """A class for defining Irregular Functional Data.

    Parameters
    ----------
    """

    def __init__(self, argvals, values):
        """Initialize IrregularFunctionalData object."""
        self().__init__(argvals, values, 'irregular')
