#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Basis
-----

"""
import numpy as np
import numpy.typing as npt
import scipy

from typing import Dict, Optional, List

from .functional_data import DenseFunctionalData, MultivariateFunctionalData
from .functional_data import _tensor_product


#######################################################################
# Definition of the basis (eigenfunctions)

def _basis_legendre(
    argvals: npt.NDArray[np.float64],
    n_functions: np.int64 = 3,
) -> npt.NDArray[np.float64]:
    r"""Define Legendre basis of function.

    Build a basis of :math:`K` functions using Legendre polynomials on the
    interval defined by ``argvals``.

    Parameters
    ----------
    argvals: npt.NDArray[np.float64]
        The values on which evaluated the Legendre polynomials.
    n_functions: np.int64, default=3
        Maximum degree of the Legendre polynomials.

    Returns
    -------
    values: npt.NDArray[np.float64], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions of
        Legendre basis.

    Notes
    -----
    The Legendre basis is defined by induction as:

    .. math::
        (n + 1)P_{n + 1}(t) = (2n + 1)tP_n(t) - nP_{n - 1}(t), \quad\text{for}
        \quad n \geq 1,

    with :math:`P_0(t) = 1` and :math:`P_1(t) = t`.

    Examples
    --------
    >>> _basis_legendre(argvals=np.arange(-1, 1, 0.1), n_functions=3)

    """
    values = np.empty((n_functions, len(argvals)))
    for degree in np.arange(0, n_functions):
        legendre = scipy.special.eval_legendre(degree, argvals)
        values[degree, :] = legendre
    return values


def _basis_wiener(
    argvals: npt.NDArray[np.float64],
    n_functions: np.int64 = 3
) -> npt.NDArray[np.float64]:
    r"""Define Wiener basis of function.

    Build a basis of :math:`K` functions using the eigenfunctions of a Wiener
    process on the interval defined by ``argvals``.

    Parameters
    ----------
    argvals: npt.NDArray[np.float64]
        The values on which the eigenfunctions of a Wiener process are
        evaluated.
    n_functions: np.int64, default=3
        Number of functions to consider.

    Returns
    -------
    values: npt.NDArray[np.float64], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions of
        Wiener basis.

    Notes
    -----
    The Wiener basis is defined as the eigenfunctions of the Brownian motion:

    .. math::
        \phi_k(t) = \sqrt{2}\sin\left(\left(k - \frac{1}{2}\right)\pi t\right),
        \quad 1 \leq k \leq K

    Example
    -------
    >>> _basis_wiener(argvals=np.arange(0, 1, 0.05), n_functions=3)

    """
    values = np.empty((n_functions, len(argvals)))
    for degree in np.arange(1, n_functions + 1):
        wiener = np.sqrt(2) * np.sin((degree - 0.5) * np.pi * argvals)
        values[(degree - 1), :] = wiener
    return values


def _basis_fourier(
    argvals: npt.NDArray[np.float64],
    n_functions: np.int64 = 3,
    period: np.float64 = 2 * np.pi,
) -> npt.NDArray[np.float64]:
    r"""Define Fourier basis of function.

    Build a basis of :math:`K` functions using Fourier series on the
    interval defined by ``argvals``.

    Parameters
    ----------
    argvals: npt.NDArray[np.float64]
        The values on which evaluated the Fourier series.
    n_functions: np.int64, default=3
        Number of considered Fourier series. Should be odd.
    period: np.float64, default=2 * np.pi
        The period of the circular functions.

    Returns
    -------
    values: npt.NDArray[np.float64], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions of
        Wiener basis.

    Notes
    -----
    The Fourier basis is defined as:

    .. math::
        \Phi(t) = \left(1, \sin(\omega t), \cos(\omega t), \dots \right)

    where :math:`\omega` is the period.

    Examples
    --------
    >>> _basis_fourier(argvals=np.arange(0, 2*np.pi, 0.1), n_functions=3)

    """
    n_functions = n_functions + 1 if n_functions % 2 == 0 else n_functions
    values = np.ones((n_functions, len(argvals)))
    for k in np.arange(1, (n_functions + 1) // 2):
        sin = np.sin(2 * np.pi * k * argvals / period)
        cos = np.cos(2 * np.pi * k * argvals / period)
        values[(2 * k - 1), :] = sin
        values[(2 * k), :] = cos
    return values[:n_functions, :]


def _basis_bsplines(
    argvals: npt.NDArray[np.float64],
    n_functions: np.int64 = 5,
    degree: np.int64 = 3,
) -> npt.NDArray[np.float64]:
    """Define B-splines basis of function.

    Build a basis of :math:`n_functions` functions using B-splines basis on the
    interval defined by ``argvals``. We assume that the knots are not given,
    and we compute them from the quantiles of the ``argvals``.

    Parameters
    ----------
    argvals: npt.NDArray[np.float64]
        The values on which evaluated the B-splines.
    n_functions: np.int64, default=5
        Number of considered B-splines.
    degree: np.int64, default=3
        Degree of the B-splines. The default gives cubic splines.

    Returns
    -------
    values: npt.NDArray[np.float64], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions of
        Wiener basis.

    Examples
    --------
    >>> _basis_bsplines(argvals=np.arange(0, 1, 0.01), n_functions=5)

    """
    n_inner_knots = n_functions - degree - 1
    if n_inner_knots < 0:
        raise ValueError(
            f"n_functions={n_functions} is too small for degree={degree}; "
            f"must be >= {degree + 1}."
        )

    inner_knots = np.linspace(0, 1, n_inner_knots + 2)
    inner_knots = np.quantile(argvals, inner_knots)
    knots = np.pad(inner_knots, (degree, degree), 'edge')
    coefs = np.eye(n_functions)
    basis = scipy.interpolate.splev(argvals, (knots, coefs, degree))
    return np.vstack(basis)


def _basis_natural_cubic_splines(
    argvals: npt.NDArray[np.float64],
    n_functions: np.int64 = 5,
    degree: np.int64 = 3
) -> npt.NDArray[np.float64]:
    """Define natural cubic splines basis of functions."""


def _basis_cyclic_cubic_splines(
    argvals: npt.NDArray[np.float64],
    n_functions: np.int64 = 5,
    degree: np.int64 = 3
) -> npt.NDArray[np.float64]:
    """Define cyclic cubic splines basis of functions."""


def _simulate_basis(
    name: np.str_,
    argvals: npt.NDArray[np.float64],
    n_functions: np.int64 = 5,
    norm: np.bool_ = False,
    **kwargs
) -> npt.NDArray[np.float64]:
    """Redirect to the right simulation basis function.

    Parameters
    ----------
    name: np.str_, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Name of the basis to use.
    argvals: npt.NDArray[np.float64]
        The values on which the basis functions are evaluated.
    n_functions: np.int64, default=5
        Number of functions to compute.
    norm: np.bool_
        Should we normalize the functions?

    Keyword Args
    ------------
    period: np.float64, default = 2 * np.pi
        The period of the circular functions for the Fourier basis.
    degree: np.int64, default = 3
        Degree of the B-splines. The default gives cubic splines.

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
    ...     norm=True
    ... )

    """
    if name == 'legendre':
        values = _basis_legendre(argvals, n_functions)
    elif name == 'wiener':
        values = _basis_wiener(argvals, n_functions)
    elif name == 'fourier':
        values = _basis_fourier(
            argvals, n_functions, kwargs.get('period', 2 * np.pi)
        )
    elif name == 'bsplines':
        values = _basis_bsplines(argvals, n_functions, kwargs.get('degree', 3))
    else:
        raise NotImplementedError(f'Basis {name!r} not implemented!')

    if norm:
        norm2 = np.sqrt(scipy.integrate.simpson(values * values, argvals))
        values = np.divide(values, norm2[:, np.newaxis])
    return values


def _simulate_basis_multivariate_weighted(
    basis_name: List[np.str_],
    argvals: List[npt.NDArray[np.float64]],
    n_functions: np.int64 = 5,
    norm: np.bool_ = False,
    **kwargs
):
    """Simulate function for multivariate functional data.

    Using weighted

    Parameters
    ----------
    basis_name: List[np.str_]
        Name of the basis to used.
    argvals: List[npt.NDArray[np.float64]]
        The values on which the basis functions are evaluated.
    n_functions: np.int64
        Number of basis functions to used.
    norm: np.bool_
        Should we normalize the functions?

    Keyword Args
    ------------
    period: np.float64, default = 2 * np.pi
        The period of the circular functions for the Fourier basis.
    degree: np.int64, default = 3
        Degree of the B-splines. The default gives cubic splines.

    Returns
    -------
    List[npt.NDArray[np.float64]], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions.

    """
    # Define weights
    alpha = np.random.uniform(low=0.2, high=0.8, size=len(basis_name))
    weights = np.sqrt(alpha / np.sum(alpha))

    return [
        weight * _simulate_basis(name, argval, n_functions, norm, **kwargs)
        for name, argval, weight in zip(basis_name, argvals, weights)
    ]


def _simulate_basis_multivariate_split(
    basis_name: List[np.str_],
    argvals: List[npt.NDArray[np.float64]],
    n_functions: np.int64 = 5,
    norm: np.bool_ = False,
    **kwargs
):
    """Simulate function for multivariate functional data.

    Using split.

    Parameters
    ----------
    basis_name: List[np.str_]
        Name of the basis to used.
    argvals: List[npt.NDArray[np.float64]]
        The values on which the basis functions are evaluated.
    n_functions: np.int64
        Number of basis functions to used.
    norm: np.bool_
        Should we normalize the functions?

    Keyword Args
    ------------
    period: np.float64, default = 2 * np.pi
        The period of the circular functions for the Fourier basis.
    degree: np.int64, default = 3
        Degree of the B-splines. The default gives cubic splines.

    Returns
    -------
    List[npt.NDArray[np.float64]], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions.

    """
    x = len(argvals) * [None]
    split_vals = np.repeat(0, len(argvals) + 1)

    x[0] = argvals[0]
    split_vals[1] = len(x[0])
    for idx in np.arange(1, len(argvals), 1):
        x[idx] = argvals[idx] - np.min(argvals[idx]) + np.max(x[idx - 1])
        split_vals[idx + 1] = split_vals[idx] + len(x[idx])

    x_concat = np.concatenate(x)
    values = _simulate_basis(basis_name, x_concat, n_functions, norm, **kwargs)
    flips = np.random.choice((-1, 1), size=len(argvals))
    return [
        flips[idx] * values[:, split_vals[idx]:split_vals[idx + 1]]
        for idx in range(len(argvals))
    ]


###############################################################################
# Class Basis

class Basis(DenseFunctionalData):
    r"""Functional data object representing an orthogonal basis of functions.

    Parameters
    ----------
    name: np.str_, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Denotes the basis of functions to use.
    n_functions: np.int64
        Number of functions in the basis.
    dimension: np.str_, {'1D', '2D'}, default='1D'
        Dimension of the basis to simulate. If '2D', the basis is simulated as
        the tensor product of the one dimensional basis of functions by itself.
        The number of functions in the 2D basis will be :math:`n_function^2`.
    argvals: Optional[Dict[np.str_, npt.NDArray[np.float64]]]
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. The shape of the :math:`j` th
        dimension is :math:`(m_j,)` for :math:`0 \leq j \leq p`.
    norm: np.bool_, default=False
        Should we normalize the basis function?

    Keyword Args
    ------------
    period: np.float64, default=2 * np.pi
        The period of the circular functions for the Fourier basis.
    degree: np.int64, default=3
        Degree of the B-splines. The default gives cubic splines.

    """

    def __init__(
        self,
        name: np.str_,
        n_functions: np.int64 = 5,
        dimension: np.str_ = '1D',
        argvals: Optional[Dict[np.str_, npt.NDArray[np.float64]]] = None,
        norm: np.bool_ = False,
        **kwargs
    ) -> None:
        """Initialize Basis object."""
        self.name = name
        self.norm = norm
        self.dimension = dimension

        if argvals is None:
            argvals = {'input_dim_0': np.arange(0, 1.01, 0.01)}

        if len(argvals) > 1:
            raise NotImplementedError(
                'Only one dimensional basis are implemented.'
            )

        values = _simulate_basis(
            name, argvals['input_dim_0'], n_functions, norm, **kwargs
        )

        if dimension == '1D':
            super().__init__(argvals, values)
        elif dimension == '2D':
            basis1d = DenseFunctionalData(argvals, values)
            basis2d = _tensor_product(basis1d, basis1d)
            super().__init__(basis2d.argvals, basis2d.values)
        else:
            raise ValueError(f"{dimension} is not a valid dimension!")

    @property
    def name(self) -> np.str_:
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, new_name: np.str_) -> None:
        if not isinstance(new_name, str):
            raise TypeError(f'{new_name!r} has to be `str`.')
        self._name = new_name

    @property
    def norm(self) -> np.bool_:
        """Getter for norm."""
        return self._norm

    @norm.setter
    def norm(self, new_norm: np.bool_) -> None:
        self._norm = new_norm

    @property
    def dimension(self) -> np.str_:
        """Getter for dimension."""
        return self._dimension

    @dimension.setter
    def dimension(self, new_dimension: np.str_) -> None:
        self._dimension = new_dimension


###############################################################################
# Class MultivariateBasis
class MultivariateBasis(MultivariateFunctionalData):
    r"""Functional data object representing an orthogonal basis of functions.

    Parameters
    ----------
    n_components: np.int64
        Number of components in the data multivariate basis.
    name: np.str_, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Denotes the basis of functions to use.
    n_functions: np.int64
        Number of functions in the basis.
    dimension: np.str_, {'1D', '2D'}, default='1D'
        Dimension of the basis to simulate. If '2D', the basis is simulated as
        the tensor product of the one dimensional basis of functions by itself.
        The number of functions in the 2D basis will be :math:`n_function^2`.
    argvals: Optional[Dict[np.str_, npt.NDArray[np.float64]]]
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. The shape of the :math:`j` th
        dimension is :math:`(m_j,)` for :math:`0 \leq j \leq p`.
    norm: np.bool_, default=False
        Should we normalize the basis function?

    Keyword Args
    ------------
    period: np.float64, default=2 * np.pi
        The period of the circular functions for the Fourier basis.
    degree: np.int64, default=3
        Degree of the B-splines. The default gives cubic splines.

    """

    def __init__(
        self,
        n_components: np.int64,
        name: np.str_,
        n_functions: np.int64 = 5,
        dimension: np.str_ = '1D',
        argvals: Optional[Dict[np.str_, npt.NDArray[np.float64]]] = None,
        norm: np.bool_ = False,
        **kwargs
    ) -> None:
        """Initialize Basis object."""
        self.name = name
        self.norm = norm
        self.dimension = dimension

    @property
    def name(self) -> np.str_:
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, new_name: np.str_) -> None:
        if not isinstance(new_name, str):
            raise TypeError(f'{new_name!r} has to be `str`.')
        self._name = new_name

    @property
    def norm(self) -> np.bool_:
        """Getter for norm."""
        return self._norm

    @norm.setter
    def norm(self, new_norm: np.bool_) -> None:
        self._norm = new_norm

    @property
    def dimension(self) -> np.str_:
        """Getter for dimension."""
        return self._dimension

    @dimension.setter
    def dimension(self, new_dimension: np.str_) -> None:
        self._dimension = new_dimension
