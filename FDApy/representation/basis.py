#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Basis
-----

"""
from __future__ import annotations

import itertools
import numpy as np
import numpy.typing as npt

from functools import reduce
from scipy.integrate import simpson

from typing import Optional, List, Tuple, Union

from .functional_data import DenseFunctionalData, MultivariateFunctionalData

from .argvals import DenseArgvals
from .values import DenseValues

from ..misc.basis import _basis_wiener, _basis_legendre, _basis_fourier, _basis_bsplines
from ..misc.utils import _inner_product


#######################################################################
# Definition of the basis (eigenfunctions)
def _simulate_basis(
    name: str,
    argvals: npt.NDArray[np.float64],
    n_functions: int = 5,
    is_normalized: bool = False,
    add_intercept: bool = True,
    **kwargs,
) -> npt.NDArray[np.float64]:
    """Redirect to the right simulation basis function.

    Parameters
    ----------
    name: str, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Name of the basis to use.
    argvals: npt.NDArray[np.float64]
        The values on which the basis functions are evaluated.
    n_functions: int, default=5
        Number of functions to compute.
    is_normalized: bool
        Should we normalize the functions?
    add_intercept: bool, default=True
        Should the constant functions be into the basis?
    kwargs
        Other keyword arguments are passed to the function:

        - :meth:`misc.basis._basis_bsplines`.

    Returns
    -------
    values: npt.NDArray[np.float64], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions.

    Example
    -------
    >>> _simulate_basis(
    ...     'legendre',
    ...     n_functions=3,
    ...     argvals=np.arange(-1, 1, 0.1),
    ...     is_normalized=True
    ... )

    """
    if not add_intercept:
        n_functions = n_functions + 1

    if name == "legendre":
        values = _basis_legendre(argvals, n_functions)
    elif name == "wiener":
        values = _basis_wiener(argvals, n_functions)
    elif name == "fourier":
        values = _basis_fourier(argvals, n_functions)
    elif name == "bsplines":
        values = _basis_bsplines(
            argvals,
            n_functions,
            degree=kwargs.get("degree", 3),
            domain_min=kwargs.get("domain_min", np.min(argvals)),
            domain_max=kwargs.get("domain_max", np.max(argvals)),
        )
    else:
        raise NotImplementedError(f"Basis {name!r} not implemented!")

    if is_normalized:
        norm2 = np.sqrt(simpson(values * values, argvals))
        values = np.divide(values, norm2[:, np.newaxis])

    if add_intercept:
        return values
    else:
        return values[1:]


###############################################################################
# Class Basis
class Basis(DenseFunctionalData):
    r"""Define univariate orthonormal basis.

    Parameters
    ----------
    name: Union[Tuple[str], str], {'given', 'legendre', 'wiener', 'fourier', 'bsplines'}
        Denotes the basis of functions to use. The default is `bsplines`. If
        `name=given`, it uses a user defined basis (defined with the `argvals` and
        `values` parameters). For higher dimensional data, `name` is a tuple for the
        marginal basis.
    n_functions: Union[Tuple[int], int], default=5
        Number of functions in the basis.
    argvals: Optional[DenseArgvals]
        The sampling points of the functional data.
    values: Optional[DenseValues]
        The values of the functional data. Only used if `name='given'`.
    is_normalized: bool, default=False
        Should we normalize the basis function?
    add_intercept: bool, default=True
        Should the constant functions be into the basis?
    kwargs
        Other keyword arguments are passed to the function:

        - :meth:`representation.basis._simulate_basis`.

    """

    ###########################################################################
    # Magic methods
    def __init__(
        self,
        name: Union[Tuple[str], str] = "bsplines",
        n_functions: Union[Tuple[int], int] = 5,
        argvals: Optional[DenseArgvals] = None,
        values: Optional[DenseValues] = None,
        is_normalized: bool = False,
        add_intercept: bool = True,
        **kwargs,
    ) -> None:
        """Initialize Basis object."""
        if name == "given":
            n_functions = values.n_obs
        self.name = name
        self.n_functions = n_functions
        self.is_normalized = is_normalized
        self.add_intercept = add_intercept

        if argvals is None:
            argvals = DenseArgvals(
                {
                    f"input_dim_{idx}": np.arange(0, 1.1, 0.1)
                    for idx in np.arange(len(self.n_functions))
                }
            )

        if name != "given":
            values_list = []
            for name, n_function, argval in zip(
                self.name, self.n_functions, argvals.values()
            ):
                temp = _simulate_basis(
                    name, argval, n_function, is_normalized, add_intercept, **kwargs
                )
                values_list.append(temp)

            values = DenseValues(
                reduce(np.kron, values_list).reshape(
                    (np.prod(self.n_functions), *argvals.n_points)
                )
            )
        super().__init__(argvals, values)

    ###########################################################################

    ###########################################################################
    # Properties
    @property
    def name(self) -> Tuple[str]:
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, new_name: Union[Tuple[str], str]) -> None:
        if isinstance(new_name, str):
            new_name = (new_name,)
        self._name = new_name

    @property
    def n_functions(self) -> Tuple[int]:
        """Getter for n_functions."""
        return self._n_functions

    @n_functions.setter
    def n_functions(self, new_n_functions: Union[Tuple[int], int]) -> None:
        if isinstance(new_n_functions, int):
            new_n_functions = (new_n_functions,)
        self._n_functions = new_n_functions

    @property
    def is_normalized(self) -> bool:
        """Getter for is_normalized."""
        return self._is_normalized

    @is_normalized.setter
    def is_normalized(self, new_is_normalized: bool) -> None:
        self._is_normalized = new_is_normalized

    @property
    def add_intercept(self) -> bool:
        """Getter for add_intercept."""
        return self._add_intercept

    @add_intercept.setter
    def add_intercept(self, new_add_intercept: bool) -> None:
        self._add_intercept = new_add_intercept

    ###########################################################################
    # Methods
    def inner_product(
        self,
        method_integration: str = "trapz",
        method_smoothing: Optional[str] = None,
        noise_variance: Optional[float] = None,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """Compute the inner product matrix of the basis."""
        # Get parameters
        n_obs = self.n_obs
        axis = [argvals for argvals in self.argvals.values()]

        inner_mat = np.zeros((n_obs, n_obs))
        for i, j in itertools.product(np.arange(n_obs), repeat=2):
            if i <= j:
                inner_mat[i, j] = _inner_product(
                    self.values[i], self.values[j], *axis, method=method_integration
                )

        # Estimate the diagonal of the inner-product matrix
        inner_mat[np.abs(inner_mat) < 1e-12] = 0
        inner_mat = inner_mat + inner_mat.T
        np.fill_diagonal(inner_mat, np.diag(inner_mat) / 2)
        
        try:
            np.linalg.cholesky(inner_mat)
        except np.linalg.LinAlgError as err:
            from statsmodels.stats.correlation_tools import cov_nearest
            print(f"The inner product has been made positive definite ({err}).")
            inner_mat = cov_nearest(inner_mat)

        self._inner_product_matrix = inner_mat
        return self._inner_product_matrix

    ###########################################################################


###############################################################################
# Class MultivariateBasis
class MultivariateBasis(MultivariateFunctionalData):
    r"""Define multivariate orthonormal basis.

    Parameters
    ----------
    name: List[Union[Tuple[str], str]]
        Name of the basis to use. One of
        `{'legendre', 'wiener', 'fourier', 'bsplines'}`.
    n_functions: List[Union[Tuple[int], int]]
        Number of functions in the basis.
    argvals: Optional[List[DenseArgvals]]
        The sampling points of the functional data.
    values: Optional[List[DenseValues]]
        The values of the functional data. Only used if `name='given'`.
    is_normalized: bool, default=False
        Should we normalize the basis function?
    kwargs
        Other keywords arguments are passed to the function:

        - :meth:`representation.basis.Basis`.

    """

    ###########################################################################
    # Magic methods
    def __init__(
        self,
        name: List[Union[Tuple[str], str]] = ["fourier", "legendre"],
        n_functions: List[Union[Tuple[int], int]] = [5, 5],
        argvals: Optional[List[DenseArgvals]] = None,
        values: Optional[List[DenseValues]] = None,
        is_normalized: bool = False,
        add_intercept: bool = True,
        **kwargs,
    ) -> None:
        """Initialize Basis object."""
        self.name = name
        self.n_functions = n_functions
        self.is_normalized = is_normalized
        self.add_intercept = add_intercept

        if argvals is None:
            argvals = [
                DenseArgvals(
                    {
                        f"input_dim_{idx}": np.arange(0, 1.1, 0.1)
                        for idx in np.arange(len(component))
                    }
                )
                for component in self.n_functions
            ]

        if name == "given":
            basis_list = [
                Basis(name="given", argvals=arg, values=val)
                for arg, val in zip(argvals, values)
            ]
        else:
            basis_list = [
                Basis(
                    name=name,
                    n_functions=n_function,
                    argvals=arg,
                    is_normalized=is_normalized,
                    add_intercept=add_intercept,
                    **kwargs,
                )
                for n_function, name, arg in zip(self.n_functions, self.name, argvals)
            ]
        super().__init__(basis_list)

    ###########################################################################

    ###########################################################################
    # Properties
    @property
    def name(self) -> Union[str, List[str]]:
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, new_name: Union[str, List[str]]) -> None:
        self._name = new_name

    @property
    def n_functions(self) -> List[Union[Tuple[int]]]:
        """Getter for n_functions."""
        return self._n_functions

    @n_functions.setter
    def n_functions(self, new_n_functions: List[Union[Tuple[int], int]]) -> None:
        temp = []
        for n_function in new_n_functions:
            if isinstance(n_function, int):
                temp.append((n_function,))
            else:
                temp.append(n_function)
        self._n_functions = temp

    @property
    def is_normalized(self) -> bool:
        """Getter for is_normalized."""
        return self._is_normalized

    @is_normalized.setter
    def is_normalized(self, new_is_normalized: bool) -> None:
        self._is_normalized = new_is_normalized

    @property
    def add_intercept(self) -> bool:
        """Getter for add_intercept."""
        return self._add_intercept

    @add_intercept.setter
    def add_intercept(self, new_add_intercept: bool) -> None:
        self._add_intercept = new_add_intercept

    ###########################################################################
