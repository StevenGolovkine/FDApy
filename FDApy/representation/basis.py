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

from typing import List, Tuple

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
    name
        Name of the basis to use.
    argvals
        The values on which the basis functions are evaluated.
    n_functions
        Number of functions to compute.
    is_normalized
        Should we normalize the functions?
    add_intercept
        Should the constant functions be into the basis?
    kwargs
        Other keyword arguments are passed to the function
        :meth:`misc.basis._basis_bsplines`.

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
        norm2 = np.sqrt(simpson(values * values, x=argvals))
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
    name
        Denotes the basis of functions to use. The default is `bsplines`. If
        `name=given`, it uses a user defined basis (defined with the `argvals` and
        `values` parameters). For higher dimensional data, `name` is a tuple for the
        marginal basis.
    n_functions
        Number of functions in the basis.
    argvals
        The sampling points of the functional data.
    values
        The values of the functional data. Only used if `name='given'`.
    is_normalized
        Should we normalize the basis function?
    add_intercept
        Should the constant functions be into the basis?
    kwargs
        Other keyword arguments are passed to the function
        :meth:`representation.basis._simulate_basis`.

    Attributes
    ----------
    argvals_stand: DenseArgvals
        Standardized sampling points of the functional data.
    n_obs: int
        Number of observations of the functional data.
    n_dimension: int
        Number of input dimension of the functional data.
    n_points: Tuple[int, ...]
        Number of sampling points.

    References
    ----------
    .. [1] Benko, M., Härdle, W. and Kneip, A. (2009). Common functional
        principal components. The Annals of Statistics 37, 1--34.
    .. [2] Cai, T.T., Yuan, M., (2011), Optimal estimation of the mean
        function based on discretely sampled functional data: Phase
        transition. The Annals of Statistics 39, 2330-2355.
    .. [3] Chiou, J.-M., Chen, Y.-T., Yang, Y.-F. (2014). Multivariate
        Functional Principal Component Analysis: A Normalization Approach.
        Statistica Sinica 24, 1571--1596.
    .. [4] Eilers, P. H. C., Marx, B. D. (2021). Practical Smoothing: The Joys
        of P-splines. Cambridge University Press, Cambridge.
    .. [5] Hall, P., Kay, J.W. and Titterington, D.M. (1990).
        Asymptotically Optimal Difference-Based Estimation of Variance in
        Nonparametric Regression. Biometrika 77, 521--528.
    .. [6] Happ, C., Greven, S. (2018). Multivariate Functional Principal
        Component Analysis for Data Observed on Different (Dimensional)
        Domains. Journal of the American Statistical Association 113, 649--659.
    .. [7] Ramsay, J. O. and Silverman, B. W. (2005), Functional Data
        Analysis, Springer Science, Chapter 8.
    .. [8] Tsybakov, A.B. (2008), Introduction to Nonparametric Estimation.
        Springer Series in Statistics.

    """

    ###########################################################################
    # Magic methods
    def __init__(
        self,
        name: Tuple[str] | str = "bsplines",
        n_functions: Tuple[int] | int = 5,
        argvals: DenseArgvals | None = None,
        values: DenseValues | None = None,
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
    def name(self, new_name: Tuple[str] | str) -> None:
        if isinstance(new_name, str):
            new_name = (new_name,)
        self._name = new_name

    @property
    def n_functions(self) -> Tuple[int]:
        """Getter for n_functions."""
        return self._n_functions

    @n_functions.setter
    def n_functions(self, new_n_functions: Tuple[int] | int) -> None:
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
        method_smoothing: str | None = None,
        noise_variance: float | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
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
        except np.linalg.LinAlgError as error:
            from statsmodels.stats.correlation_tools import cov_nearest

            print(f"The inner product has been made positive definite ({error}).")
            inner_mat = cov_nearest(inner_mat)
            # inner_mat = 1e-10 * np.eye(inner_mat.shape[0])

        self._inner_product_matrix = inner_mat
        return self._inner_product_matrix

    ###########################################################################


###############################################################################
# Class MultivariateBasis
class MultivariateBasis(MultivariateFunctionalData):
    r"""Define multivariate orthonormal basis.

    Parameters
    ----------
    name
        Name of the basis to use. One of
        `{'legendre', 'wiener', 'fourier', 'bsplines'}`.
    n_functions
        Number of functions in the basis.
    argvals
        The sampling points of the functional data.
    values
        The values of the functional data. Only used if `name='given'`.
    is_normalized
        Should we normalize the basis function?
    kwargs
        Other keywords arguments are passed to the function
        :meth:`representation.basis.Basis`.

    Attributes
    ----------
    n_obs: int
        Number of observations of the functional data.
    n_functional: int
        Number of components of the multivariate functional data.
    n_dimension: List[int]
        Number of input dimension of the functional data.
    n_points: List[Dict[str, int]]
        Number of sampling points.

    References
    ----------
    .. [1] Benko, M., Härdle, W. and Kneip, A. (2009). Common functional
        principal components. The Annals of Statistics 37, 1--34.
    .. [2] Chiou, J.-M., Chen, Y.-T., Yang, Y.-F. (2014). Multivariate
        Functional Principal Component Analysis: A Normalization Approach.
        Statistica Sinica 24, 1571--1596.
    .. [3] Eilers, P. H. C., Marx, B. D. (2021). Practical Smoothing: The Joys
        of P-splines. Cambridge University Press, Cambridge.
    .. [4] Hall, P., Kay, J.W. and Titterington, D.M. (1990).
        Asymptotically Optimal Difference-Based Estimation of Variance in
        Nonparametric Regression. Biometrika 77, 521--528.
    .. [5] Happ and Greven (2018), Multivariate Functional Principal
        Component Analysis for Data Observed on Different (Dimensional)
        Domains. Journal of the American Statistical Association, 113,
        pp. 649--659.
    .. [6] Tsybakov, A.B. (2008), Introduction to Nonparametric Estimation.
        Springer Series in Statistics.
    .. [7] Zhang, J.-T. and Chen J. (2007), Statistical Inferences for
        Functional Data, The Annals of Statistics, Vol. 35, No. 3.

    """

    ###########################################################################
    # Magic methods
    def __init__(
        self,
        name: List[Tuple[str] | str] = ["fourier", "legendre"],
        n_functions: List[Tuple[int] | int] = [5, 5],
        argvals: List[DenseArgvals] | None = None,
        values: List[DenseValues] | None = None,
        is_normalized: bool = False,
        add_intercept: bool = True,
        **kwargs,
    ) -> None:
        """Initialize MultivariateBasis object."""
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
    def name(self) -> str | List[str]:
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, new_name: str | List[str]) -> None:
        self._name = new_name

    @property
    def n_functions(self) -> List[Tuple[int]]:
        """Getter for n_functions."""
        return self._n_functions

    @n_functions.setter
    def n_functions(self, new_n_functions: List[Tuple[int] | int]) -> None:
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
