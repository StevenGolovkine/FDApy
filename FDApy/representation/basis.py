#!/usr/bin/env python
# -*-coding:utf8 -*

"""Basis functions.

This module is used to define a Basis class and diverse classes derived from
it. These are used to define basis of functions as DenseFunctionalData object.
"""
import numpy as np
import numpy.typing as npt
import scipy

from patsy import bs
from typing import Any, Dict, Optional

from .functional_data import DenseFunctionalData
from .functional_data import _tensor_product


#######################################################################
# Definition of the basis (eigenfunctions)

def basis_legendre(
    argvals: npt.NDArray[np.float64],
    n_functions: int = 3,
) -> npt.NDArray[np.float64]:
    r"""Define Legendre basis of function.

    Build a basis of :math:`K` functions using Legendre polynomials on the
    interval defined by ``argvals``.

    Parameters
    ----------
    argvals: numpy.ndarray
        The values on which evaluated the Legendre polynomials.
    n_functions: int, default=3
        Maximum degree of the Legendre polynomials.

    Returns
    -------
    values: np.ndarray, shape=(n_functions, len(argvals))
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
    >>> basis_legendre(argvals=np.arange(-1, 1, 0.1), n_functions=3)

    """
    values = np.empty((n_functions, len(argvals)))
    for degree in np.arange(0, n_functions):
        legendre = scipy.special.eval_legendre(degree, argvals)
        values[degree, :] = legendre
    return values


def basis_wiener(
    argvals: npt.NDArray[np.float64],
    n_functions: int = 3
) -> npt.NDArray[np.float64]:
    r"""Define Wiener basis of function.

    Build a basis of :math:`K` functions using the eigenfunctions of a Wiener
    process on the interval defined by ``argvals``.

    Parameters
    ----------
    argvals: numpy.ndarray
        The values on which the eigenfunctions of a Wiener process are
        evaluated.
    n_functions: int, default=3
        Number of functions to consider.

    Returns
    -------
    values: np.ndarray, shape=(n_functions, len(argvals))
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
    >>> basis_wiener(argvals=np.arange(0, 1, 0.05), n_functions=3)

    """
    values = np.empty((n_functions, len(argvals)))
    for degree in np.arange(1, n_functions + 1):
        wiener = np.sqrt(2) * np.sin((degree - 0.5) * np.pi * argvals)
        values[(degree - 1), :] = wiener
    return values


def basis_fourier(
    argvals: npt.NDArray[np.float64],
    n_functions: int = 3,
    period: float = 2 * np.pi,
) -> npt.NDArray[np.float64]:
    r"""Define Fourier basis of function.

    Build a basis of :math:`K` functions using Fourier series on the
    interval defined by ``argvals``.

    Parameters
    ----------
    argvals: numpy.ndarray
        The values on which evaluated the Fourier series.
    n_functions: int, default=3
        Number of considered Fourier series. Should be odd.
    period: float, default=2*numpy.pi
        The period of the circular functions.

    Returns
    -------
    values: np.ndarray, shape=(n_functions, len(argvals))
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
    >>> basis_fourier(argvals=np.arange(0, 2*np.pi, 0.1), n_functions=3)

    """
    n_functions = n_functions + 1 if n_functions % 2 == 0 else n_functions
    values = np.ones((n_functions, len(argvals)))
    for k in np.arange(1, (n_functions + 1) // 2):
        sin = np.sin(2 * np.pi * k * argvals / period)
        cos = np.cos(2 * np.pi * k * argvals / period)
        values[(2 * k - 1), :] = sin
        values[(2 * k), :] = cos
    return values[:n_functions, :]


def basis_bsplines(
    argvals: npt.NDArray[np.float64],
    n_functions: int = 5,
    degree: int = 3,
    knots: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray[np.float64]:
    """Define B-splines basis of function.

    Build a basis of :math:`n_functions` functions using B-splines basis on the
    interval defined by ``argvals``.

    Parameters
    ----------
    argvals: numpy.ndarray
        The values on which evaluated the B-splines.
    n_functions: int, default=5
        Number of considered B-splines.
    degree: int, default=3
        Degree of the B-splines. The default gives cubic splines.
    knots: numpy.ndarray, (n_knots,)
        Specify the break points defining the B-splines. If ``knots``
        are provided, the provided value of ``n_functions`` is ignored. And the
        number of basis functions is ``n_knots + degree - 1``.

    Returns
    -------
    values: np.ndarray, shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions of
        Wiener basis.

    Examples
    --------
    >>> basis_bsplines(argvals=np.arange(0, 1, 0.01), n_functions=5)

    """
    if knots is not None:
        n_knots = len(knots)
        n_functions = n_knots + degree - 1
    else:
        n_knots = n_functions - degree + 1
        knots = np.linspace(argvals[0], argvals[-1], n_knots)

    values = bs(
        argvals, df=n_functions, knots=knots[1:-1],
        degree=degree, include_intercept=True
    )
    return values.T  # type: ignore


def simulate_basis(
    name: str,
    argvals: npt.NDArray[np.float64],
    n_functions: int = 3,
    norm: bool = False,
    **kwargs: Any
) -> npt.NDArray[np.float64]:
    """Redirect to the right simulation basis function.

    Parameters
    ----------
    name: str, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Name of the basis to use.
    argvals: numpy.ndarray
        The values on which the basis functions are evaluated.
    n_functions: int, default=3
        Number of functions to compute.
    norm: boolean
        Should we normalize the functions?

    Keyword Args
    ------------
    period: float, default = 2*numpy.pi
        The period of the circular functions for the Fourier basis.
    degree: int, default = 3
        Degree of the B-splines. The default gives cubic splines.
    knots: numpy.ndarray, (n_knots,)
        Specify the break points defining the B-splines.

    Returns
    -------
    values: np.ndarray, shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions of
        Wiener basis.

    Example
    -------
    >>> simulate_basis('legendre', n_functions=3,
    >>>                argvals=np.arange(-1, 1, 0.1), norm=True)

    """
    if name == 'legendre':
        values = basis_legendre(argvals, n_functions)
    elif name == 'wiener':
        values = basis_wiener(argvals, n_functions)
    elif name == 'fourier':
        values = basis_fourier(
            argvals, n_functions, kwargs.get('period', 2 * np.pi)
        )
    elif name == 'bsplines':
        values = basis_bsplines(
            argvals, n_functions,
            kwargs.get('degree', 3), kwargs.get('knots', None)
        )
    else:
        raise NotImplementedError(f'Basis {name!r} not implemented!')

    if norm:
        norm2 = np.sqrt(scipy.integrate.simpson(values * values, argvals))
        values = np.divide(values, norm2[:, np.newaxis])
    return values


###############################################################################
# Class Basis

class Basis(
    DenseFunctionalData
):
    r"""A functional data object representing an orthogonal basis of functions.

    Parameters
    ----------
    name: str, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Denotes the basis of functions to use.
    n_functions: int
        Number of functions in the basis.
    dimension: str, ('1D', '2D'), default='1D'
        Dimension of the basis to simulate. If '2D', the basis is simulated as
        the tensor product of the one dimensional basis of functions by itself.
        The number of functions in the 2D basis will be :math:`n_function^2`.
    argvals: dict
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. The shape of the :math:`j`th
        dimension is :math:`(m_j,)` for :math:`0 \leq j \leq p`.
    norm: bool, default=False
        Should we normalize the basis function?

    Keyword Args
    ------------
    period: float, default = 2*numpy.pi
        The period of the circular functions for the Fourier basis.
    degree: int, default = 3
        Degree of the B-splines. The default gives cubic splines.
    knots: numpy.ndarray, (n_knots,)
        Specify the break points defining the B-splines.

    """

    def __init__(
        self,
        name: str,
        n_functions: int,
        dimension: str = '1D',
        argvals: Optional[Dict[str, npt.NDArray[np.float64]]] = None,
        norm: bool = False,
        **kwargs: Any
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

        values = simulate_basis(
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
    def name(self) -> str:
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        if not isinstance(new_name, str):
            raise TypeError(f'{new_name!r} has to be `str`.')
        self._name = new_name

    @property
    def norm(self) -> bool:
        """Getter for norm."""
        return self._norm

    @norm.setter
    def norm(self, new_norm: bool) -> None:
        self._norm = new_norm

    @property
    def dimension(self) -> str:
        """Getter for dimension."""
        return self._dimension

    @dimension.setter
    def dimension(self, new_dimension: str) -> None:
        self._dimension = new_dimension
