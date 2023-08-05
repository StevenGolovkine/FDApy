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
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .argvals import Argvals


###############################################################################
# Class Values
class Values(ABC):
    """Metaclass for the definition of values of functional data."""

    @staticmethod
    @abstractmethod
    def concatenate(*values) -> Type[Values]:
        """Concatenate Values objects."""

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Return the number of observations."""

    @property
    @abstractmethod
    def n_points(self):
        """Return the number of sampling points for each dimension."""

    def compatible_with(self, argvals: Type[Argvals]) -> None:
        """Raise an error if Values is not compatible with Argvals.

        Parameters
        ----------
        argvals: Type[Argvals]
            A Argvals object.

        Raises
        ------
        ValueError
            When `self` and `argvals` do not have coherent common
            sampling points.

        """
        if self.n_points != argvals.n_points:
            raise ValueError(
                "The Values and the Argvals do not have coherent number"
                " of sampling points."
            )


###############################################################################
# Class DenseValues
class DenseValues(Values, np.ndarray):
    """Class representing an array of values for DenseFunctionalData.

    This class extends the `Values` class to represent values for
    DenseFunctionalData. It provides additional functionality for working with
    argument values in scientific computing.

    """

    @staticmethod
    def concatenate(*values) -> DenseValues:
        """Concatenate DenseValues objects.

        Parameters
        ----------
        *values:
            The DenseValues objects to concatenate.

        Returns
        -------
        DenseValues
            The concatenated DenseValues.

        """
        return DenseValues(np.vstack([el for el in values]))

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

    @property
    def n_points(self):
        """Return the number of sampling points for each dimension.

        The number of sampling points is the dimensions of the array,
        except the first.

        """
        return self.shape[1:]


###############################################################################
# Class IrregularValues
class IrregularValues(Values, UserDict):
    """Class representing a dictionary of values for IrregularFunctionalData.

    This class extends the `Values` class to represent values for
    IrregularFunctionalData. It provides additional functionality for working
    with argument values in scientific computing.

    """

    @staticmethod
    def concatenate(*values) -> IrregularValues:
        """Concatenate IrregularValues objects.

        Parameters
        ----------
        *values:
            The IrregularValues objects to concatenate.

        Returns
        -------
        IrregularValues
            The concatenated IrregularValues.

        """
        new_values = {}
        for el in values:
            temp = len(new_values)
            for key, values in el.items():
                new_values[temp + key] = values
        return IrregularValues(new_values)

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

    @property
    def n_points(self):
        """Return the number of sampling points for each dimension.

        The number of sampling points is the dimension of the array for each
        entry of the dictionary.

        """
        return {obs: pts.shape for obs, pts in self.items()}
