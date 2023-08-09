#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Brownian motions
----------------

"""
import numpy as np
import numpy.typing as npt

from typing import Callable, Optional, Tuple

from ..representation.argvals import DenseArgvals
from ..representation.values import DenseValues
from ..representation.functional_data import DenseFunctionalData
from .simulation import Simulation


#############################################################################
# Definition of the different Browian motion

def _init_brownian(
    argvals: npt.NDArray[np.float64]
) -> Tuple[float, npt.NDArray[np.float64]]:
    """Initialize Brownian motions.

    Initialize different parameters used for the simulation of different
    types of Brownian motions.

    Parameters
    ----------
    argvals: npt.NDArray[np.float64], shape=(n,)
        Values at which Brownian motions are evaluated.

    Returns
    -------
    Tuple[float, npt.NDArray[np.float64]]
        A tuple containing the step size, ``delta``, and the sampling points
        ``argvals``.

    """
    delta = (np.max(argvals) - np.min(argvals)) / np.size(argvals)
    return delta, argvals


def _standard_brownian(
    argvals: npt.NDArray[np.float64],
    init_point: float = 0.0,
    rnorm: Callable = np.random.normal
) -> npt.NDArray[np.float64]:
    """Generate standard Brownian motion.

    Parameters
    ----------
    argvals: npt.NDArray[np.float64], shape=(n,)
        Values at which Brownian motions are evaluated.
    init_point: float, default=0.0
        Start value of the Brownian motion.
    rnorm: Callable, default=np.random.normal
        Function to use for normal random variables generation (used for
        reproducibility purpose).

    Returns
    -------
    npt.NDArray[np.float64], shape=(n,)
        Array representing a standard Brownian motion with the same shape
        than argvals.

    Notes
    -----
    The sampling points have to be regularly spaced. Otherwise, the covariance
    of the generated data will not be the good one.

    References
    ----------
    .. [1] https://github.com/cran/somebm/blob/master/R/bm.R

    Example
    -------
    >>> _standard_brownian(argvals=np.arange(0, 1, 0.01), init_point=0.0)

    """
    delta, argvals = _init_brownian(argvals)

    values = np.zeros(np.size(argvals))
    values[0] = init_point
    for idx in np.arange(1, np.size(argvals)):
        values[idx] = values[idx - 1] + np.sqrt(delta) * rnorm()
    return values


def _geometric_brownian(
    argvals: npt.NDArray[np.float64],
    init_point: float = 1.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    rnorm: Callable = np.random.normal
) -> npt.NDArray[np.float64]:
    """Generate geometric Brownian motion.

    Parameters
    ----------
    argvals: npt.NDArray[np.float64], shape=(n,)
        Values at which Brownian motions are evaluated.
    init_point: float, default=1.0
        Start value of the Brownian motion. For geometric Brownian motion,
        ``init_point`` should be stricly positive.
    mu: float, default=0
        Interest rate (or percentage drift).
    sigma: float, default=1
        Diffusion coefficient (or percentage volatility).
    rnorm: Callable, default=np.random.normal
        Function to use for normal random variables generation (used for
        reproducibility purpose).

    Returns
    -------
    npt.NDArray[np.float64], shape=(n,)
        Array representing a geometric Brownian motion with the same shape
        than argvals.

    Notes
    -----
    The sampling points have to be regularly spaced. Otherwise, the covariance
    of the generated data will not be the good one.

    References
    ----------
    .. [1] https://github.com/cran/somebm/blob/master/R/bm.R

    Example
    -------
    >>> _geometric_brownian(argvals=np.arange(0, 1, 0.01), init_point=1.0)

    """
    if not init_point > 0:
        raise ValueError(
            'The parameter `init_point` must be stricly positive.'
        )

    delta, argvals = _init_brownian(argvals)
    const = mu - sigma**2 / 2
    values = rnorm(0, np.sqrt(delta), size=len(argvals))
    in_exp = const * delta + sigma * values
    return init_point * np.cumprod(np.exp(in_exp))


def _fractional_brownian(
    argvals: npt.NDArray[np.float64],
    hurst: float = 0.5,
    rnorm: Callable = np.random.normal
) -> npt.NDArray[np.float64]:
    """Generate fractional Brownian motion.

    Parameters
    ----------
    argvals: npt.NDArray[np.float64], shape=(n,)
        Values at which Brownian motions are evaluated.
    hurst: float, default=0.5
        Hurst parameter. If ``hurst = 0.5``. the fractional Brownian motion is
        equivalent to the standard Brownian motion.
    rnorm: Callable, default=np.random.normal
        Function to use for normal random variables generation (used for
        reproducibility purpose).

    Returns
    -------
    npt.NDArray[np.float64], shape=(n,)
        Array representing a geometric Brownian motion with the same shape
        than argvals.

    Notes
    -----
    The sampling points have to be regularly spaced. Otherwise, the covariance
    of the generated data will not be the good one.

    References
    ----------
    .. [1] https://github.com/cran/somebm/blob/master/R/bm.R

    Example
    -------
    >>> _fractional_brownian(argvals=np.arange(0, 1, 0.01), hurst=0.7)

    """
    def p(idx, hurst):
        return np.power(idx, 2 * hurst)

    if hurst <= 0:
        raise ValueError('The Hurst parameter has to be strictly positive.')

    _, argvals = _init_brownian(argvals)
    n = np.size(argvals)

    vec = np.ones(n + 1)
    for idx in np.arange(1, n + 1):
        temp = p(idx + 1, hurst) - 2 * p(idx, hurst) + p(idx - 1, hurst)
        vec[idx] = 0.5 * temp
    inv_vec = vec[::-1]
    vec = np.append(vec, inv_vec[1:len(inv_vec) - 1])
    lamb = np.real(np.fft.fft(vec) / (2 * n))

    rng = rnorm(size=2 * n) + rnorm(size=2 * n) * 1j
    values = np.fft.fft(np.sqrt(lamb) * rng)
    return np.power(n, -hurst) * np.cumsum(np.real(values[1:(n + 1)]))


def _simulate_brownian(
    name: str,
    argvals: npt.NDArray[np.float64],
    rnorm: Callable = np.random.normal,
    **kwargs
) -> npt.NDArray[np.float64]:
    """Simulate Brownian motion.

    Parameters
    ----------
    name: str, {'standard', 'geometric', 'fractional'}
        Name of the Brownian motion type to simulate.
    argvals: npt.NDArray[np.float64], shape=(n,)
        Values at which Brownian motions are evaluated.
    rnorm: Callable, default=np.random.normal
        Function to use for normal random variables generation (used for
        reproducibility purpose).
    **kwargs:
        init_point: float, default=0.0 or 1.0
            Start value of the Brownian motion. For geometric Brownian motion,
            ``init_point`` should be stricly positive.
        mu: float, default=0
            Interest rate (or percentage drift).
        sigma: float, default=1
            Diffusion coefficient (or percentage volatility).
        hurst: float, default=0.5
            Hurst parameter. If ``hurst = 0.5``. the fractional Brownian motion
            is equivalent to the standard Brownian motion.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n,)
        An array representing a standard brownian motion with the same shape
        than argvals.

    Example
    -------
    >>> simulate_brownian(brownian_type='standard')

    """
    if name == 'standard':
        return _standard_brownian(
            argvals,
            init_point=kwargs.get('init_point', 0.0),
            rnorm=rnorm
        )
    elif name == 'geometric':
        return _geometric_brownian(
            argvals,
            init_point=kwargs.get('init_point', 1.0),
            mu=kwargs.get('mu', 0.0),
            sigma=kwargs.get('sigma', 1.0),
            rnorm=rnorm
        )
    elif name == 'fractional':
        return _fractional_brownian(
            argvals,
            hurst=kwargs.get('hurst', 0.5),
            rnorm=rnorm
        )
    else:
        raise NotImplementedError('Brownian type not implemented!')


#############################################################################
# Definition of the Brownian class

class Brownian(Simulation):
    """Class that defines Brownian motions simulation.

    Parameters
    ----------
    name: str, {'standard', 'geometric', 'fractional'}
        Name of the Brownian motion type to simulate.
    random_state: int, default=None
        A seed to initialize the random number generator.

    Attributes
    ----------
    data: DenseFunctionalData
        An object that represents the simulated data.
    noisy_data: DenseFunctionalData
        An object that represents a noisy version of the simulated data.
    sparse_data: IrregularFunctionalData
        An object that represents a sparse version of the simulated data.

    Notes
    -----
    The sampling points have to be regularly spaced. Otherwise, the covariance
    of the generated data will not be the good one.

    The implementation is adapted from [1]_.

    References
    ----------
    .. [1] https://github.com/cran/somebm/blob/master/R/bm.R

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
        n_clusters: int = 1,
        argvals: Optional[npt.NDArray[np.float64]] = None,
        **kwargs
    ) -> None:
        """Simulate realizations of a Brownian motion.

        This function generates ``n_obs`` realizations of a Brownian motion
        on a common grid ``argvals``.

        Parameters
        ----------
        n_obs: int
            Number of observations to simulate.
        n_clusters: int
            Not used.
        argvals: Optional[npt.NDArray[np.float64]], shape=(n,)
            Values at which Brownian motions are evaluated. If ``None``, the
            functions are evaluated on the interval :math:`[0, 1]` with
            :math:`21` regularly spaced sampled points.
        **kwargs:
            init_point: float
                Start value of the Brownian motion. For geometric Brownian
                motion, ``init_point`` should be stricly positive. Default
                value is 0 for standard Brownian motion and 1 for geometric
                Brownian motion.
            mu: float, default=0
                Interest rate (or percentage drift).
            sigma: float, default=1
                Diffusion coefficient (or percentage volatility).
            hurst: float, default=0.5
                Hurst parameter. If ``hurst = 0.5``. the fractional Brownian
                motion is equivalent to the standard Brownian motion.

        """
        if self.random_state is None:
            rnorm = np.random.normal
        else:
            rnorm = self.random_state.normal

        if argvals is None:
            argvals = np.arange(0, 1.05, 0.05)

        values = np.zeros(shape=(n_obs, len(argvals)))
        for idx in range(n_obs):
            values[idx, :] = _simulate_brownian(
                name=self.basis_name,
                argvals=argvals,
                rnorm=rnorm,
                **kwargs
            )
        self.data = DenseFunctionalData(
            DenseArgvals({'input_dim_0': argvals}),
            DenseValues(values)
        )
