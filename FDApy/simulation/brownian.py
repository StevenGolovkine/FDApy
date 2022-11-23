#!/usr/bin/env python
# -*-coding:utf8 -*

"""Brownian motions.

"""
import numpy as np

from typing import Optional, Tuple

from ..representation.functional_data import DenseFunctionalData
from .simulation import Simulation


#############################################################################
# Definition of the different Browian motion

def init_brownian(
    argvals: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """Initialize Brownian motion.

    Initialize the different parameters used in the simulation of the
    different type of Brownian motion.

    Parameters
    ----------
    argvals: numpy.ndarray, default=None
        The values on which the Brownian motion are evaluated. If ``None``,
        the functions are evaluated on the interval :math:`[0, 1]`.

    Returns
    -------
    delta, argvals: (float, numpy.ndarray)
        A tuple containing the step size, ``delta``, and the ``argvals``.

    """
    if argvals is None:
        argvals = np.arange(0, 1, 0.05)

    delta = (np.max(argvals) - np.min(argvals)) / np.size(argvals)
    return delta, argvals


def standard_brownian(
    argvals: Optional[np.ndarray] = None,
    x0: float = 0.0
) -> np.ndarray:
    """Generate standard Brownian motion.

    Parameters
    ----------
    argvals: numpy.ndarray, default=None, shape=(n,)
        The values on which the Brownian motion is evaluated. If ``None``,
        the functions are evaluated on the interval :math:`[0, 1]`.
    x0: float, default=0.0
        Start of the Brownian motion.

    Returns
    -------
    values: np.ndarray, shape=(n,)
        An array representing a standard brownian motion with the same shape
        than argvals.

    References
    ----------
    - https://github.com/cran/somebm/blob/master/R/bm.R

    Example
    -------
    >>> standard_brownian(argvals=np.arange(0, 1, 0.01), x0=0.0)

    """
    delta, argvals = init_brownian(argvals)

    values = np.zeros(np.size(argvals))
    values[0] = x0
    for idx in np.arange(1, np.size(argvals)):
        values[idx] = values[idx - 1] + np.sqrt(delta) * np.random.normal()
    return values


def geometric_brownian(
    argvals: Optional[np.ndarray] = None,
    x0: float = 1.0,
    mu: float = 0.0,
    sigma: float = 1.0
) -> np.ndarray:
    """Generate geometric Brownian motion.

    Parameters
    ----------
    argvals: numpy.ndarray, default=None, shape=(n,)
        The values on which the geometric brownian motion is evaluated. If
        ``None``, the Brownian is evaluated on the interval :math:`[0, 1]`.
    x0: float, default = 1.0
        Start of the Brownian motion. Careful, ``x0`` should be stricly
        greater than 0.
    mu: float, default = 0
        The interest rate
    sigma: float, default = 1
        The diffusion coefficient

    Returns
    -------
    values: np.ndarray, shape=(n,)
        An array representing a geometric brownian motion with the same shape
        than argvals.

    References
    ----------
    - https://github.com/cran/somebm/blob/master/R/bm.R

    Example
    -------
    >>> geometric_brownian(argvals=np.arange(0, 1, 0.01), x0=1.0)

    """
    if not x0 > 0:
        raise ValueError('x0 must be stricly greater than 0.')

    delta, argvals = init_brownian(argvals)
    const = mu - sigma**2 / 2
    values = np.random.normal(0, np.sqrt(delta), size=len(argvals))
    in_exp = const * delta + sigma * values
    return x0 * np.cumprod(np.exp(in_exp))


def fractional_brownian(
    argvals: Optional[np.ndarray] = None,
    hurst: float = 0.5
) -> np.ndarray:
    """Generate fractional Brownian motion.

    Parameters
    ----------
    argvals: numpy.ndarray, default=None, shape=(n,)
        The values on which the fractional Brownian motion is evaluated. If
        ``None``, the Brownian is evaluated on the interval :math:`[0, 1]`.
    hurst: float, default=0.5
        Hurst parameter

    Returns
    -------
    values: np.ndarray, shape=(n,)
        An array representing a standard brownian motion with the same shape
        than argvals.

    References
    ----------
    - https://github.com/cran/somebm/blob/master/R/bm.R

    Example
    -------
    >>> fractional_brownian(argvals=np.arange(0, 1, 0.01), hurst=0.7)

    """
    def p(idx, hurst):
        return np.power(idx, 2 * hurst)

    _, argvals = init_brownian(argvals)
    n = np.size(argvals)

    vec = np.ones(n + 1)
    for idx in np.arange(1, n + 1):
        temp = (p(idx + 1, hurst) - 2 * p(idx, hurst) + p(idx - 1, hurst))
        vec[idx] = 0.5 * temp
    inv_vec = vec[::-1]
    vec = np.append(vec, inv_vec[1:len(inv_vec) - 1])
    lamb = np.real(np.fft.fft(vec) / (2 * n))

    rng = (np.random.normal(size=2 * n) + np.random.normal(size=2 * n) * 1j)
    values = np.fft.fft(np.sqrt(lamb) * rng)
    return np.power(n, -hurst) * np.cumsum(np.real(values[1:(n + 1)]))


def simulate_brownian(
    name: str,
    argvals: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """Redirect to the right brownian motion function.

    Parameters
    ----------
    name: str, {'standard', 'geometric', 'fractional'}
        Name of the Brownian motion to simulate.
    argvals: numpy.ndarray, shape=(n,)
        The sampling points on which the Brownian motion is evaluated. If
        ``None``, the Brownian is evaluated on the interval :math:`[0, 1]`.

    Keyword Args
    ------------
    x0: float, default=0.0 or 1.0
        Start of the Brownian motion. Should be strictly positive if
        ``brownian_type=='geometric'``.
    mu: float, default=0
        The interest rate
    sigma: float, default=1
        The diffusion coefficient
    hurst: float, default=0.5
        Hurst parameter

    Returns
    -------
    values: np.ndarray, shape=(n,)
        An array representing a standard brownian motion with the same shape
        than argvals.

    Example
    -------
    >>> simulate_brownian(brownian_type='standard',
    >>>                   argvals=np.arange(0, 1, 0.05))

    """
    if name == 'standard':
        return standard_brownian(argvals, x0=kwargs.get('x0', 0.0))
    elif name == 'geometric':
        return geometric_brownian(argvals,
                                  x0=kwargs.get('x0', 1.0),
                                  mu=kwargs.get('mu', 0.0),
                                  sigma=kwargs.get('sigma', 1.0))
    elif name == 'fractional':
        return fractional_brownian(argvals, hurst=kwargs.get('hurst', 0.5))
    else:
        raise NotImplementedError('Brownian type not implemented!')


#############################################################################
# Definition of the Brownian class

class Brownian(Simulation):
    """Class for the simulation of diverse Brownian motions.

    Parameters
    ----------
    name: str, {'standard', 'geometric', 'fractional'}
        Type of Brownian motion to simulate.

    Arguments
    ---------
    data: DenseFunctionalData
        An object that represents the simulated data.
    noisy_data: DenseFunctionalData
        An object that represents a noisy version of the simulated data.
    sparse_data: IrregularFunctionalData
        An object that represents a sparse version of the simulated data.

    """

    def __init__(
        self,
        name: str,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Brownian object."""
        super().__init__(name, random_state)

    def new(
        self,
        n_obs: int,
        argvals: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """Simulate ``n_obs`` realizations of a Brownian on ``argvals``.

        Parameters
        ----------
        n_obs: int
            Number of observations to simulate.
        argvals: numpy.ndarray, shape=(n,), default=None
            The sampling points on which the Brownian motion is evaluated. If
            ``None``, the Brownian is evaluated on the interval :math:`[0, 1]`.

        Keyword Args
        ------------
        x0: float, default=0.0 or 1.0
            Start of the Brownian motion. Should be strictly positive if
            ``brownian_type=='geometric'``.
        mu: float, default=0
            The interest rate
        sigma: float, default=1
            The diffusion coefficient
        hurst: float, default=0.5
            Hurst parameter.

        """
        values = np.zeros(shape=(n_obs, len(argvals)))
        for idx in range(n_obs):
            values[idx, :] = simulate_brownian(self.name, argvals, **kwargs)
        self.data = DenseFunctionalData({'input_dim_0': argvals}, values)
