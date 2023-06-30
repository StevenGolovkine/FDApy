#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Values
------

"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from collections import UserDict
from typing import Any, Type


###############################################################################
# Class Values
class Values(ABC):
    """Metaclass for the definition of values of functional data."""

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Return the number of observations."""


###############################################################################
# Class DenseValues
class DenseValues(Values, np.ndarray):
    """Class representing an array of values for DenseFunctionalData.

    This class extends the `Values` class to represent values for
    DenseFunctionalData. It provides additional functionality for working with
    argument values in scientific computing.

    """

    def __new__(cls, input_array: npt.NDArray[np.float64]) -> None:
        """Create a new instance of DenseValues.

        This method is responsible for creating and initializing the object.
        It converts the input array into an ndarray object and creates a new
        instance of DenseValues by calling `view(cls)`.

        Parameters
        ----------
        input_array: npt.NDArray[np.float64]
            The input array to be converted into a DenseValues object.

        Returns
        -------
        DenseValues
            A new instance of DenseValues initialized with the input array.

        """
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def n_obs(self) -> int:
        """Return the number of observations.

        Returns
        -------
        int
            The number of observations is the first dimension of the array.

        Examples
        --------
        >>> array = np.array([[1, 2, 3], [4, 5, 6]])
        >>> values = DenseValues(array)
        >>> values.n_obs
        2

        """
        return self.shape[0]


###############################################################################
# Class IrregularValues
class IrregularValues(Values, UserDict):
    """Class representing a dictionary of values for IrregularFunctionalData.

    This class extends the `Values` class to represent values for
    IrregularFunctionalData. It provides additional functionality for working
    with argument values in scientific computing.

    """

    def __setitem__(self, key: int, value: npt.NDArray[np.float64]) -> None:
        """Set the value for a given key.

        This method sets the value for the specified key in the
        `IrregularValues` object. If the key already exists, the value is
        updated with the new value. If the key does not exist, a new key-value
        pair is added.

        Parameters
        ----------
        key: int
            The key to set or update.
        value: npt.NDArray[np.float64]
            The value to associate with the key.

        Raises
        ------
        TypeError
            Raise a type error if the key is not a int or if the value is
            not a np.ndarray.

        Returns
        -------
        None

        """
        if not isinstance(key, int):
            raise TypeError("Key must be an int")
        if not isinstance(value, np.ndarray):
            raise TypeError("Value must be an np.ndarray")
        super().__setitem__(key, value)

    @property
    def n_obs(self) -> int:
        """Return the number of observations.

        Returns
        -------
        int
            The number of observations is the length of the dictionary.

        Examples
        --------
        >>> values_dict = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}
        >>> values = IrregularValues(values_dict)
        >>> values.n_obs
        2

        """
        return len(self)
