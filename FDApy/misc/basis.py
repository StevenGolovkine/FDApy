#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Basis
-----

"""
import numpy as np
import numpy.typing as npt

from scipy.special import gamma, eval_legendre


def _basis_legendre(
    argvals: npt.NDArray[np.float64],
    n_functions: int = 3,
) -> npt.NDArray[np.float64]:
    r"""Define Legendre basis of function.

    Build a basis of :math:`K` functions using Legendre polynomials on the
    interval defined by ``argvals``.

    Parameters
    ----------
    argvals
        The values on which evaluated the Legendre polynomials.
    n_functions
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
        legendre = eval_legendre(degree, argvals)
        values[degree, :] = legendre
    return values


def _basis_wiener(
    argvals: npt.NDArray[np.float64], n_functions: int = 3
) -> npt.NDArray[np.float64]:
    r"""Define Wiener basis of function.

    Build a basis of :math:`K` functions using the eigenfunctions of a Wiener
    process on the interval defined by ``argvals``.

    Parameters
    ----------
    argvals
        The values on which the eigenfunctions of a Wiener process are
        evaluated.
    n_functions
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
    argvals: npt.NDArray[np.float64], n_functions: int = 3
) -> npt.NDArray[np.float64]:
    r"""Define Fourier basis of function.

    Build a basis of :math:`K` functions using Fourier series on the
    interval defined by ``argvals``.

    Parameters
    ----------
    argvals
        The values on which evaluated the Fourier series.
    n_functions
        Number of considered Fourier series. Should be odd.

    Returns
    -------
    values: npt.NDArray[np.float64], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions of Wiener
        basis.

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
    values = np.ones((n_functions, len(argvals))) / np.sqrt(np.ptp(argvals))

    norm = np.sqrt(2 / np.ptp(argvals))
    xx = (2 * np.pi * (argvals - np.min(argvals)) / np.ptp(argvals)) - np.pi
    for k in np.arange(1, n_functions):
        # We consider k + 1 because of Python indexation
        if k % 2:  # k + 1 even
            values[k, :] = norm * np.cos(((k + 1) // 2) * xx)
        else:  # k + 1 odd
            values[k, :] = norm * np.sin(((k + 1) // 2) * xx)
    return values


def _basis_bsplines(
    argvals: npt.NDArray[np.float64],
    n_functions: int = 10,
    degree: int = 3,
    domain_min: float | None = None,
    domain_max: float | None = None,
) -> npt.NDArray[np.float64]:
    """Define B-splines basis of function.

    Build a basis of :math:`n_functions` functions using B-splines basis on the
    interval defined by ``argvals``. We assume that the knots are regularly spaced. The
    number of knots is equal to ``n_functions - degree``.

    Parameters
    ----------
    argvals
        The values on which evaluated the B-splines.
    n_functions
        Number of considered B-splines.
    degree
        Degree of the B-splines. The default gives cubic splines.
    domain_min
        Minimum number for the argvals.
    domain_max
        Maximum number for hte argvals.

    Returns
    -------
    values: npt.NDArray[np.float64], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions of a B-splines
        basis.

    Notes
    -----
    This function is adapted from the `bbase` function in the R package `JOPS` _[2]. It
    computes a proper B-splines basis function (see _[1], Section 8.1).

    Examples
    --------
    >>> _basis_bsplines(argvals=np.arange(0, 1, 0.01), n_functions=10)

    References
    ----------
    .. [1] Eilers, P., Marx, B.D., (2021) Practical Smoothing: The Joys of
        P-splines. Cambridge University Press, Cambridge.
    .. [2] Eilers, P., Marx, B., Li, B., Gampe, J., Rodriguez-Alvarez, M.X., (2023)
        JOPS: Practical Smoothing with P-Splines.

    """

    def _tpower(x, knots, p):
        res = np.zeros((len(x), len(knots)))
        for idx, knot in enumerate(knots):
            res[:, idx] = np.power(x - knot, p) * (x >= knot)
        return res

    if domain_min is None:
        domain_min = min(argvals)
    if domain_max is None:
        domain_max = max(argvals)

    # Compute the B-splines
    n_segments = n_functions - degree
    dx = (domain_max - domain_min) / n_segments
    knots = np.linspace(
        domain_min - degree * dx,
        domain_max + degree * dx,
        num=int(n_segments + 2 * degree) + 1,
        endpoint=True,
    )
    p_mat = _tpower(argvals, knots, degree)
    d_mat = np.diff(np.eye(p_mat.shape[1]), n=degree + 1, axis=0) / (
        gamma(degree + 1) * np.power(dx, degree)
    )
    basis_mat = np.power(-1, degree + 1) * p_mat @ d_mat.T

    # Make B-splines exactly zero beyond their end knots
    sk = knots[np.arange(basis_mat.shape[1]) + degree + 1]
    mask = np.zeros((len(argvals), len(sk)))
    for idx, val in enumerate(argvals):
        mask[idx, :] = val < sk
    return (basis_mat * mask).T
