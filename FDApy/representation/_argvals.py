#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Argvals
-------

"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from abc import abstractmethod
from collections import UserDict
from typing import Any, Type


###############################################################################
# Class Argvals
class Argvals(UserDict):
    """Metaclass for the definition of argvals of functional data."""

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> None:
        """Set the value for a given key.

        This method sets the value for the specified key in the `Argvals`
        object. If the key already exists, the value is updated with the new
        value. If the key does not exist, a new key-value pair is added.

        Parameters
        ----------
        key: Any
            The key to set or update.
        value: Any
            The value to associate with the key.

        Returns
        -------
        None

        """
        super().__setitem__(key, value)

    @abstractmethod
    def __eq__(self, other: Type[Argvals]) -> bool:
        """Check if two Argvals are equals.

        This method if two Argvals objects have the same type and if their
        length are equals.

        Parameters
        ----------
        other: Type[Argvals]
            The object to compare with the current Argvals object.

        Returns
        -------
        bool
            False if the objects have different types or different lengths,
            True otherwise.

        """
        if not isinstance(self, type(other)):
            return False
        if len(self) != len(other):
            return False
        return True


###############################################################################
# Class DenseArgvals
class DenseArgvals(Argvals):
    """Class representing a dictionary of argvals for DenseFunctionalData.

    This class extends the `Argvals` class to represent a dictionary where the
    keys are strings and the values are np.ndarray. It provides additional
    functionality for working with argument values in scientific computing.

    """

    def __setitem__(self, key: str, value: npt.NDArray[np.float64]) -> None:
        """Set the value for a given key.

        This method sets the value for the specified key in the `DenseArgvals`
        object. If the key already exists, the value is updated with the new
        value. If the key does not exist, a new key-value pair is added.

        Parameters
        ----------
        key: str
            The key to set or update.
        value: npt.NDArray[np.float64]
            The value to associate with the key.

        Raises
        ------
        TypeError
            Raise a type error if the key is not a string or if the value is
            not a np.ndarray.

        Returns
        -------
        None

        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        if not isinstance(value, np.ndarray):
            raise TypeError("Value must be an np.ndarray")
        super().__setitem__(key, value)

    def __eq__(self, other: DenseArgvals) -> bool:
        """Check if two DenseArgvals objects are equal.

        This method compares the DenseArgvals object with another object to
        check if they are equal. Two DenseArgvals objects are considered equal
        if they have the same keys and the corresponding values (np.ndarray)
        are equal element-wise.

        Parameters
        ----------
        other: DenseArgvals
            The object to compare with the current DenseArgvals object.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.

        Examples
        --------
        >>> argvals1 = DenseArgvals({
        ...     'input_dim_0': np.array([1, 2, 3]),
        ...     'input_dim_0': np.array([4, 5, 6])
        ... })
        >>> argvals2 = DenseArgvals({
        ...     'input_dim_0': np.array([1, 2, 3]),
        ...     'input_dim_1': np.array([4, 5, 6])
        ... })
        >>> argvals1 == argvals2
        True

        >>> argvals3 = DenseArgvals({
        ...     'input_dim_0': np.array([1, 2, 3]),
        ...     'input_dim_1': np.array([4, 5, 7])
        ... })
        >>> argvals1 == argvals3
        False

        """
        if not super(DenseArgvals, self).__eq__(other):
            return False
        for key, value in self.items():
            if key not in other or not np.array_equal(value, other[key]):
                return False
        return True


###############################################################################
# Class IrregularArgvals
class IrregularArgvals(Argvals):
    """Class representing a dictionary of argvals for IrregularFunctionalData.

    This class extends the `Argvals` class to represent a dictionary where the
    keys are strings and the values are DenseArgvals. It provides additional
    functionality for working with argument values in scientific computing.

    """

    def __setitem__(self, key, value):
        """Set the value for a given key.

        This method sets the value for the specified key in the
        `IrregularArgvals` object. If the key already exists, the value is
        updated with the new value. If the key does not exist, a new key-value
        pair is added.

        Parameters
        ----------
        key: int
            The key to set or update.
        value: DenseArgvals
            The value to associate with the key.

        Raises
        ------
        TypeError
            Raise a type error if the key is not a string or if the value is
            not a DenseArgvals object.

        Returns
        -------
        None

        """
        if not isinstance(key, int):
            raise TypeError("Key must be an integer")
        if not isinstance(value, DenseArgvals):
            raise TypeError("Value must be a DenseArgvals")
        super().__setitem__(key, value)

    def __eq__(self, other: IrregularArgvals) -> bool:
        """Check for equality between two IrregularArgvals objects.

        Parameters
        ----------
        other: IrregularArgvals
            The object to compare with the current IrregularArgvals object.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.

        Examples
        --------
        >>> argvals_1 = DenseArgvals({
        ...     'input_dim_0': np.random.randn(10),
        ...     'input_dim_1': np.random.randn(11)
        ... })
        >>> argvals_2 = DenseArgvals({
        ...     'input_dim_0': np.random.randn(5),
        ...     'input_dim_1': np.random.randn(7)
        ... })
        >>> argvals_2 = DenseArgvals(argvals)

        >>> argvals_irr = IrregularArgvals({0: argvals_1, 1: argvals_2})
        >>> argvals_irr_2 = IrregularArgvals({0: argvals_1, 1: argvals_2})
        >>> argvals_irr_3 = IrregularArgvals({0: argvals_2, 1: argvals_1})

        >>> argvals_irr == argvals_irr_2
        True

        >>> argvals_irr == argvals_irr_3
        False

        """
        if not super(IrregularArgvals, self).__eq__(other):
            return False
        for key, value in self.items():
            if key not in other or value != other[key]:
                return False
        return True
