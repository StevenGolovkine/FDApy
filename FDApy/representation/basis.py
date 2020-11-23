#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""Basis functions.

This module is used to define a Basis class and diverse classes derived from
it. These are used to define basis of functions as DenseFunctionalData object.
"""
import numpy as np
import scipy

from patsy import bs

from .functional_data import DenseFunctionalData
from .functional_data import tensor_product_


#######################################################################
# Definition of the basis (eigenfunctions)

def basis_legendre(n_functions=3, argvals=None, norm=False):
    r"""Define Legendre basis of function.

    Build a basis of :math:`K` functions using Legendre polynomials on the
    interval defined by ``argvals``.

    Parameters
    ----------
    n_functions: int, default=3
        Maximum degree of the Legendre polynomials.
    argvals: numpy.ndarray, default=None
        The values on which evaluated the Legendre polynomials. If ``None``,
        the polynomials are evaluated on the interval :math:`[-1, 1]`.
    norm: boolean, default=True
        Should we normalize the functions?

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
    >>> basis_legendre(n_functions=3, argvals=np.arange(-1, 1, 0.1), norm=True)

    """
    if argvals is None:
        argvals = np.arange(-1, 1, 0.1)

    if isinstance(argvals, list):
        raise ValueError('argvals has to be a numpy array!')

    values = np.empty((n_functions, len(argvals)))

    for degree in np.arange(0, n_functions):
        legendre = scipy.special.eval_legendre(degree, argvals)

        if norm:
            norm2 = np.sqrt(scipy.integrate.simps(
                legendre * legendre, argvals))
            legendre = legendre / norm2
        values[degree, :] = legendre
    return values


def basis_wiener(n_functions=3, argvals=None, norm=False):
    r"""Define Wiener basis of function.

    Build a basis of :math:`K` functions using the eigenfunctions of a Wiener
    process on the interval defined by ``argvals``.

    Parameters
    ----------
    n_functions: int, default=3
        Number of functions to consider.
    argvals: numpy.ndarray, default=None
        The values on which the eigenfunctions of a Wiener process are
        evaluated. If ``None``, the functions are evaluated on the interval
        :math:`[0, 1]`.
    norm: boolean, default=True
        Should we normalize the functions?

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
    >>> basis_wiener(n_functions=3, argvals=np.arange(0, 1, 0.05), norm=True)

    """
    if argvals is None:
        argvals = np.arange(0, 1, 0.05)

    if isinstance(argvals, list):
        raise ValueError('argvals has to be a numpy array!')

    values = np.empty((n_functions, len(argvals)))

    for degree in np.arange(1, n_functions + 1):
        wiener = np.sqrt(2) * np.sin((degree - 0.5) * np.pi * argvals)

        if norm:
            wiener = wiener / np.sqrt(scipy.integrate.simps(
                wiener * wiener, argvals))
        values[(degree - 1), :] = wiener
    return values


def basis_fourier(n_functions=3, argvals=None, period=2 * np.pi, norm=True):
    r"""Define Fourier basis of function.

    Build a basis of :math:`K` functions using Fourier series on the
    interval defined by ``argvals``.

    Parameters
    ----------
    n_functions: int, default=3
        Number of considered Fourier series. Should be odd.
    argvals: numpy.ndarray, default = None
        The values on which evaluated the Fourier series. If ``None``,
        the polynomials are evaluated on the interval :math:`[0, period]`.
    period: float, default=2*numpy.pi
        The period of the circular functions.
    norm: boolean, default=True
        Should we normalize the functions?

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
    >>> basis_fourier(n_functions=3, argvals=np.arange(0, 2*np.pi, 0.1))

    """
    n_functions = n_functions + 1 if n_functions % 2 == 0 else n_functions
    if argvals is None:
        argvals = np.arange(0, period, 0.1)
    if isinstance(argvals, list):
        raise ValueError('argvals has to be a numpy array!')

    values = np.empty((n_functions, len(argvals)))
    values[0, :] = 1
    for k in np.arange(1, (n_functions + 1) // 2):
        sin = np.sin(2 * np.pi * k * argvals / period)
        cos = np.cos(2 * np.pi * k * argvals / period)

        if norm:
            sin_norm2 = np.sqrt(scipy.integrate.simps(
                sin * sin, argvals))
            cos_norm2 = np.sqrt(scipy.integrate.simps(
                cos * cos, argvals))
            sin = sin / sin_norm2
            cos = cos / cos_norm2
        values[(2 * k - 1), :] = sin
        values[(2 * k), :] = cos
    return values[:n_functions, :]


def basis_bsplines(n_functions=5, argvals=None, degree=3, knots=None,
                   norm=False):
    """Define B-splines basis of function.

    Build a basis of :math:`K` functions using B-splines basis on the
    interval defined by ``argvals``.

    Parameters
    ----------
    n_functions: int, default=5
        Number of considered B-splines.
    argvals: numpy.ndarray, default = None
        The values on which evaluated the B-splines. If ``None``,
        the polynomials are evaluated on the interval :math:`[0, 1]`.
    degree: int, default=3
        Degree of the B-splines. The default gives cubic splines.
    knots: numpy.ndarray, (n_knots,)
        Specify the break points defining the B-splines. If ``knots``
        are provided, the provided value of ``K`` is ignored. And the
        number of basis functions is ``n_knots + degree - 1``.
    norm: boolean, default=True
        Should we normalize the functions?

    Returns
    -------
    values: np.ndarray, shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions of
        Wiener basis.

    Examples
    --------
    >>> basis_bsplines(n_functions=5, argvals=np.arange(0, 1, 0.01))

    """
    if argvals is None:
        argvals = np.arange(0, 1, 0.01)
    if isinstance(argvals, list):
        raise ValueError('argvals has to be a numpy array!')

    if knots is not None:
        n_knots = len(knots)
        n_functions = n_knots + degree - 1
    else:
        n_knots = n_functions - degree + 1
        knots = np.linspace(argvals[0], argvals[-1], n_knots)

    values = bs(argvals, df=n_functions, knots=knots[1:-1], degree=degree,
                include_intercept=True)
    if norm:
        norm2 = np.sqrt(scipy.integrate.simps(values * values, argvals,
                                              axis=0))
        values = values / norm2
    return values.T


def simulate_basis(name, n_functions=3, argvals=None, norm=False, **kwargs):
    """Redirect to the right simulation basis function.

    Parameters
    ----------
    name: str, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Name of the basis to use.
    n_functions: int, default=3
        Number of functions to compute.
    argvals: numpy.ndarray, default=None
        The values on which the basis functions are evaluated. If ``None``,
        the functions are evaluated on the diverse interval depending on the
        basis.
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
        values = basis_legendre(n_functions, argvals, norm)
    elif name == 'wiener':
        values = basis_wiener(n_functions, argvals, norm)
    elif name == 'fourier':
        values = basis_fourier(n_functions, argvals,
                               kwargs.get('period', 2 * np.pi), norm)
    elif name == 'bsplines':
        values = basis_bsplines(n_functions, argvals,
                                kwargs.get('degree', 3),
                                kwargs.get('knots', None), norm)
    else:
        raise NotImplementedError(f'Basis {name!r} not implemented!')
    return values


###############################################################################
# Class Basis

class Basis(DenseFunctionalData):
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

    def __init__(self, name, n_functions, dimension='1D', argvals=None,
                 norm=False, **kwargs):
        """Initialize Basis object."""
        self.name = name
        self.norm = norm
        self.dimension = dimension

        if argvals is None:
            argvals = {'input_dim_0': np.arange(0, 1, 0.01)}

        super()._check_argvals(argvals)
        if len(argvals) > 1:
            raise NotImplementedError('Only one dimensional basis are'
                                      ' implemented.')

        values = simulate_basis(name, n_functions, argvals['input_dim_0'],
                                norm, **kwargs)

        if dimension == '1D':
            super().__init__(argvals, values)
        elif dimension == '2D':
            basis1d = DenseFunctionalData(argvals, values)
            basis2d = tensor_product_(basis1d, basis1d)
            super().__init__(basis2d.argvals, basis2d.values)
        else:
            raise ValueError(f"{dimension} is not a valid dimension!")

    @property
    def name(self):
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise TypeError(f'{new_name!r} has to be `str`.')
        self._name = new_name

    @property
    def norm(self):
        """Getter for norm."""
        return self._norm

    @norm.setter
    def norm(self, new_norm):
        self._norm = new_norm

    @property
    def dimension(self):
        """Getter for dimension."""
        return self._dimension

    @dimension.setter
    def dimension(self, new_dimension):
        self._dimension = new_dimension
