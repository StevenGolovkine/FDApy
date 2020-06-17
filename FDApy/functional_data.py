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

from abs import ABC, abstractmethods

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
    type: str, {'univariate', 'irregular', 'multivariate'}
    """

    def __init__(self, argvals, values, type):
        """Initialize FunctionalData object."""
        pass

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
        pass

    @property
    def argvals_stand(self):
        return self._argvals_stand

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        pass
