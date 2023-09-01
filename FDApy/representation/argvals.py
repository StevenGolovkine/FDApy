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
from typing import Any, Dict, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .values import Values


###############################################################################
# Class Argvals
class Argvals(UserDict):
    """Metaclass for the definition of argvals of functional data."""

    @staticmethod
    @abstractmethod
    def concatenate(*argvals) -> Type[Argvals]:
        """Concatenate Argvals objects."""

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

    @property
    @abstractmethod
    def n_points(self):
        """Get the number of sampling points of each dimension."""

    @property
    @abstractmethod
    def n_dimension(self):
        """Get the number of dimension of the data."""

    @property
    @abstractmethod
    def min_max(self):
        """Get the minimum and maximum sampling points for each dimension."""

    @abstractmethod
    def normalization(self):
        """Normalize the Argvals."""

    def compatible_with(self, values: Type[Values]) -> None:
        """Raise an error if Argvals is not compatible with Values.

        Parameters
        ----------
        values: Type[Values]
            A Values object.

        Raises
        ------
        ValueError
            When `self` and `values` do not have coherent common
            sampling points. The first dimension of `values` is assumed to
            represented the number of observations.

        """
        if self.n_points != values.n_points:
            raise ValueError(
                "The Argvals and the Values do not have coherent number"
                " of sampling points."
            )


###############################################################################
# Class DenseArgvals
class DenseArgvals(Argvals):
    """Class representing a dictionary of argvals for DenseFunctionalData.

    This class extends the `Argvals` class to represent a dictionary where the
    keys are strings and the values are np.ndarray. It provides additional
    functionality for working with argument values in scientific computing.

    """

    @staticmethod
    def concatenate(*argvals) -> DenseArgvals:
        """Concatenate DenseArgvals objects.

        It does not make sense to concatenate DenseArgvals. This function
        checks that all the DenseArgvals objects pass as arguments are the same
        and return the first one. It raises an error if one is different.

        Parameters
        ----------
        *argvals:
            The DenseArgvals objects to concatenate.

        Returns
        -------
        DenseArgvals
            The first elements of the input list.

        Raises
        ------
        ValueError
            When all `argvals` are not equal.

        """
        if not all(el == argvals[0] for el in argvals):
            raise ValueError("Argvals are not equals.")
        return argvals[0]

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

    @property
    def n_points(self) -> Tuple[int, ...]:
        """Get the number of sampling points of each dimension."""
        return tuple(dim.shape[0] for dim in self.values())

    @property
    def n_dimension(self) -> int:
        """Get the number of dimension of the data."""
        return len(self)

    @property
    def min_max(self) -> Dict(SyntaxWarning, Tuple[float, float]):
        """Get the minimum and maximum sampling points for each dimension."""
        return {
            idx: (min(argval), max(argval)) for idx, argval in self.items()
        }

    def range(
        self,
        percentage: float = 1.0
    ) -> Dict[str, float]:
        """Get the range of sampling points for each dimension.

        Parameters
        ----------
        percentage: float, default=1.0
            Specify a percentage of the range to retrieve.

        Returns
        -------
        Dict[str, float]
            A percentage of the range of the sampling points in each dimension.

        """
        return {
            idx: percentage * (max - min)
            for idx, (min, max) in self.min_max.items()
        }

    def normalization(self) -> DenseArgvals:
        r"""Normalize the DenseArgvals.

        This function normalizes the Argvals by applying the following
        transformation to each dimension of the Argvals:

        ..math:: X_{norm} = \frac{X - \min{X}}{\max{X} - \min{X}}.

        Returns
        -------
        DenseArgvals
            Normalized argvals.

        """
        return DenseArgvals({
            dim: (points - min(points)) / (max(points) - min(points))
            for dim, points in self.items()
        })


###############################################################################
# Class IrregularArgvals
class IrregularArgvals(Argvals):
    """Class representing a dictionary of argvals for IrregularFunctionalData.

    This class extends the `Argvals` class to represent a dictionary where the
    keys are strings and the values are DenseArgvals. It provides additional
    functionality for working with argument values in scientific computing.

    """

    @staticmethod
    def concatenate(*argvals) -> IrregularArgvals:
        """Concatenate IrregularArgvals objects.

        It does not make sense to concatenate IrregularArgvals. This function
        checks that all the IrregularArgvals objects pass as arguments are the
        same and return the first one. It raises an error if one is different.

        Parameters
        ----------
        *argvals:
            The IrregularArgvals objects to concatenate.

        Returns
        -------
        IrregularArgvals
            The concatenated IrregularArgvals.

        """
        new_argvals = {}
        for el in argvals:
            temp = len(new_argvals)
            for key, values in el.items():
                new_argvals[temp + key] = values
        return IrregularArgvals(new_argvals)

    def __setitem__(self, key: int, value: DenseArgvals) -> None:
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

    @property
    def n_points(self) -> Dict[int, Tuple[int, ...]]:
        """Get the number of sampling points of each dimension."""
        return {obs: argvals.n_points for obs, argvals in self.items()}

    @property
    def n_dimension(self) -> int:
        """Get the number of dimension of the data."""
        return len(next(iter(self.values())))

    @property
    def min_max(self) -> Dict(int, Tuple[float, float]):
        """Get the minimum and maximum sampling points for each dimension."""
        return self.to_dense().min_max

    def normalization(self) -> IrregularArgvals:
        r"""Normalize the IrregularArgvals.

        This function normalizes the Argvals by applying the following
        transformation to each observation:

        ..math:: X_{norm} = \frac{X - \min{X}}{\max{X} - \min{X}}.

        Returns
        -------
        IrregularArgvals
            Normalized argvals.

        """
        min_max = self.min_max

        stand_dict = {}
        for out_key, inner_dict in self.items():
            for in_key, value in inner_dict.items():
                min_x, max_x = min_max[in_key]
                if out_key not in stand_dict:
                    stand_dict[out_key] = DenseArgvals({})
                stand_dict[out_key][in_key] = (value - min_x) / (max_x - min_x)
        return IrregularArgvals(stand_dict)

    def switch(self) -> Dict[str, Dict[int, npt.NDArray[np.float64]]]:
        """Switch the dictionary.

        This function switches nested dictionaries. It convert an
        IrregularArgvals object (with signature
        `dict[int, dict[str, npt.NDArray[np.float64]]]`) into a dictionary with
        signature `dict[str, dict[int, npt.NDArray[np.float64]]]`.

        Returns
        -------
        dict[str, dict[int, npt.NDArray[np.float64]]]
            The switched dictionnary.

        Examples
        --------
        >>> argvals_1 = DenseArgvals(
        ...     {
        ...         'input_dim_0': np.array([1, 2, 3]),
        ...         'input_dim_1': np.array([4, 5, 6])
        ...     }
        ... )
        >>> argvals_2 = DenseArgvals(
        ...     {
        ...         'input_dim_0': np.array([2, 4, 6]),
        ...         'input_dim_1': np.array([1, 3, 5])
        ...     }
        ... )
        >>> argvals = IrregularArgvals({0: argvals_1, 1: argvals_2})

        >>> argvals.switch()
        {
            'input_dim_0': {0: array([1, 2, 3]), 1: array([2, 4, 6])},
            'input_dim_1': {0: array([4, 5, 6]), 1: array([1, 3, 5])}
        }

        """
        switched_dict = {}
        for outer_key, inner_dict in self.items():
            for inner_key, value in inner_dict.items():
                if inner_key not in switched_dict:
                    switched_dict[inner_key] = {}
                switched_dict[inner_key][outer_key] = value
        return switched_dict

    def to_dense(self) -> DenseArgvals:
        """Concatenate IrregularArgvals to DenseArgvals.

        This function concatenates the sampling points of IrregularArgvals into
        a DenseArgvals object. The duplicated sampling points are dropped.

        Returns
        -------
        DenseArgvals
            The concatenation of the IrregularArgvals.

        Examples
        --------
        >>> argvals_1 = DenseArgvals(
        ...     {
        ...         'input_dim_0': np.array([1, 2, 3]),
        ...         'input_dim_1': np.array([4, 5, 6])
        ...     }
        ... )
        >>> argvals_2 = DenseArgvals(
        ...     {
        ...         'input_dim_0': np.array([2, 4, 6]),
        ...         'input_dim_1': np.array([1, 3, 5])
        ...     }
        ... )
        >>> argvals = IrregularArgvals({0: argvals_1, 1: argvals_2})

        >>> argvals.to_dense()
        {
            'input_dim_0': array([1, 2, 3, 4, 6]),
            'input_dim_1': array([1, 3, 4, 5, 6])
        }

        """
        dense_dict = {}
        for idx, dim in self.switch().items():
            dense_dict[idx] = np.unique(np.concatenate(list(dim.values())))
        return DenseArgvals(dense_dict)
