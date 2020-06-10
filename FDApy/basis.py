#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""Simulation functions

This module is used to define an abstract Simulation class and two classes
derived from it, the Basis class and the Brownian class. Thus, we may simulate
different data from a linear combination of basis functions or multiple
realizations of diverse Brownian motion.
"""
import numpy as np
import scipy

from .univariate_functional import UnivariateFunctionalData
from .multivariate_functional import MultivariateFunctionalData


#######################################################################
# Definition of the basis (eigenfunctions)

def basis_legendre(K=3, argvals=None, norm=True):
    """Define Legendre basis of function.

    Build a basis of :math:`K` functions using Legendre polynomials on the
    interval defined by ``argvals``.

    Parameters
    ----------
    K : int, default = 3
        Maximum degree of the Legendre polynomials.
    argvals : numpy.ndarray, default = None
        The values on which evaluated the Legendre polynomials. If ``None``,
        the polynomials are evaluated on the interval :math:`[-1, 1]`.
    norm : boolean, default = True
        Should we normalize the functions?

    Returns
    -------
    obj : UnivariateFunctionalData
        A UnivariateFunctionalData object containing the Legendre polynomial
        up to :math:`K` functions evaluated on ``argvals``.

    Examples
    --------
    >>> basis_legendre(K=3, argvals=np.arange(-1, 1, 0.1), norm=True)

    """
    if argvals is None:
        argvals = np.arange(-1, 1, 0.1)

    if isinstance(argvals, list):
        raise ValueError('argvals has to be a tuple or a numpy array!')

    values = np.empty((K, len(argvals)))

    for degree in np.arange(0, K):
        legendre = scipy.special.eval_legendre(degree, argvals)

        if norm:
            norm2 = np.sqrt(scipy.integrate.simps(
                legendre * legendre, argvals))
            legendre = legendre / norm2
        values[degree, :] = legendre

    obj = UnivariateFunctionalData(argvals, values)
    return obj


def basis_wiener(K=3, argvals=None, norm=True):
    """Define Wiener basis of function.

    Build a basis of :math:`K` functions using the eigenfunctions of a Wiener
    process on the interval defined by ``argvals``.

    Parameters
    ----------
    K : int, default = 3
        Number of functions to consider.
    argvals : numpy.ndarray, default = None
         The values on which the eigenfunctions of a Wiener process are
         evaluated. If ``None``, the functions are evaluated on the interval
         :math:`[0, 1]`.
    norm : boolean, default = True
        Should we normalize the functions?

    Returns
    -------
    obj : UnivariateFunctionalData
        A UnivariateFunctionalData object containing the Wiener process
        eigenvalues up to :math:`K` functions evaluated on ``argvals``.

    Example
    -------
    >>> basis_wiener(K=3, argvals=np.arange(0, 1, 0.05), norm=True)

    """
    if argvals is None:
        argvals = np.arange(0, 1, 0.05)

    if isinstance(argvals, list):
        raise ValueError('argvals has to be a numpy array!')

    values = np.empty((K, len(argvals)))

    for degree in np.arange(1, K + 1):
        wiener = np.sqrt(2) * np.sin((degree - 0.5) * np.pi * argvals)

        if norm:
            wiener = wiener / np.sqrt(scipy.integrate.simps(
                wiener * wiener, argvals))
        values[(degree - 1), :] = wiener

    obj = UnivariateFunctionalData(argvals, values)
    return obj


def simulate_basis(basis_name, K=3, argvals=None, norm=False):
    """Redirects to the right simulation basis function.

    Parameters
    ----------
    basis_name : str, {'legendre', 'wiener'}
        Name of the basis to use.
    K : int, default = 3
        Number of functions to compute.
    argvals : numpy.ndarray, default = None
        The values on which the basis functions are evaluated. If ``None``,
        the functions are evaluated on the interval :math:`[0, 1]`.
    norm : boolean
        Should we normalize the functions?

    Returns
    -------
    basis : UnivariateFunctionalData
        A UnivariateFunctionalData object containing :math:`K` basis functions
        evaluated on ``argvals``.

    Example
    -------
    >>> simulate_basis_('legendre', M=3,
    >>>                 argvals=np.arange(-1, 1, 0.1), norm=True)

    """
    if basis_name == 'legendre':
        basis = basis_legendre(K, argvals, norm)
    elif basis_name == 'wiener':
        basis = basis_wiener(K, argvals, norm)
    else:
        raise ValueError('Basis not implemented!')
    return basis


#############################################################################
# Definition of the different Browian motion

def init_brownian(argvals=None):
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
    delta, argvals : (float, numpy.ndarray)
        A tuple containing the step size, ``delta``, and the ``argvals``.

    """
    if argvals is None:
        argvals = np.arange(0, 1, 0.05)

    delta = (np.max(argvals) - np.min(argvals)) / np.size(argvals)
    return delta, argvals


def standard_brownian(argvals=None, x0=0.0):
    """Generate standard Brownian motion.

    Parameters
    ----------
    argvals: numpy.ndarray, default=None
        The values on which the Brownian motion is evaluated. If ``None``,
        the functions are evaluated on the interval :math:`[0, 1]`.
    x0: float, default=0.0
        Start of the Brownian motion.

    Returns
    -------
    obj : UnivariateFunctionalData
        A univariate functional data object containing one Brownian motion.

    References
    ----------
    - https://github.com/cran/somebm/blob/master/R/bm.R

    Example
    -------
    >>> standard_brownian(argvals=np.arange(0, 1, 0.01), x0=0.0)

    """
    delta, argvals = init_brownian(argvals)

    W = np.zeros(np.size(argvals))
    W[0] = x0
    for idx in range(1, np.size(argvals)):
        W[idx] = W[idx - 1] + np.sqrt(delta) * np.random.normal()

    obj = UnivariateFunctionalData(
        argvals=argvals, values=W[np.newaxis])
    return obj


def geometric_brownian(argvals=None, x0=1.0, mu=0.0, sigma=1.0):
    """Generate geometric Brownian motion.

    Parameters
    ----------
    argvals : numpy.ndarray, default=None
        The values on which the geometric brownian motion is evaluated. If
        ``None``, the Brownian is evaluated on the interval :math:`[0, 1]`.
    x0 : float, default = 1.0
        Start of the Brownian motion. Careful, ``x0`` should be stricly
        greater than 0.
    mu : float, default = 0
        The interest rate
    sigma : float, default = 1
        The diffusion coefficient

    Returns
    -------
    obj : UnivariateFunctionalData
        A univariate functional data object containing one geometric Brownian
        motion.

    References
    ----------
    - https://github.com/cran/somebm/blob/master/R/bm.R

    Example
    -------
    >>> geometric_brownian(argvals=np.arange(0, 1, 0.01), x0=1.0)

    """
    delta, argvals = init_brownian(argvals)

    W = np.zeros(np.size(argvals))
    for idx in range(1, np.size(argvals)):
        W[idx] = W[idx - 1] + np.sqrt(delta) * np.random.normal()

    in_exp = (mu - np.power(sigma, 2) / 2) * (argvals - argvals[0]) + sigma * W
    S = x0 * np.exp(in_exp)

    obj = UnivariateFunctionalData(
        argvals=argvals, values=S[np.newaxis])
    return obj


def fractional_brownian(argvals=None, H=0.5):
    """Generate fractional Brownian motion.

    Parameters
    ----------
    argvals: numpy.ndarray, default=None
        The values on which the fractional Brownian motion is evaluated. If
        ``None``, the Brownian is evaluated on the interval :math:`[0, 1]`.
    H: double, default = 0.5
        Hurst parameter

    Returns
    -------
    obj : UnivariateFunctionalData
        A univariate functional data object containing one fractional Brownian
        motion.

    References
    ----------
    - https://github.com/cran/somebm/blob/master/R/bm.R

    Example
    -------
    >>> fractional_brownian(argvals=np.arange(0, 1, 0.01), H=0.7)

    """
    def p(idx, H):
        return np.power(idx, 2 * H)

    if argvals is None:
        argvals = np.arange(0, 1, 0.05)

    M = np.size(argvals)
    R = np.zeros(M + 1)
    R[0] = 1
    for idx in range(1, M + 1):
        R[idx] = 0.5 * (p(idx + 1, H) - 2 * p(idx, H) + p(idx - 1, H))
    invR = R[::-1]
    R = np.append(R, invR[1:len(invR) - 1])
    lamb = np.real(np.fft.fft(R) / (2 * M))

    rng = (np.random.normal(size=2 * M) + np.random.normal(size=2 * M) * 1j)
    W = np.fft.fft(np.sqrt(lamb) * rng)
    W = np.power(M, -H) * np.cumsum(np.real(W[1:(M + 1)]))

    obj = UnivariateFunctionalData(
        argvals=argvals, values=W[np.newaxis])
    return obj


def simulate_brownian(brownian_type, argvals=None, norm=False, **kwargs):
    """Redirects to the right brownian motion function.

    Parameters
    ----------
    brownian_type : str
        Name of the Brownian motion to simulate.
    argvals : numpy.ndarray
        The sampling points on which the Brownian motion is evaluated. If
        ``None``, the Brownian is evaluated on the interval :math:`[0, 1]`.
    norm : boolean
        Should we normalize the simulation?

    Keyword Args
    ------------
    x0 : float, default = 0.0 or 1.0
        Start of the Brownian motion. Should be strictly positive if
        ``brownian_type == 'geometric'``.
    mu : float, default = 0
        The interest rate
    sigma : float, default = 1
        The diffusion coefficient
    H: double, default = 0.5
        Hurst parameter

    Returns
    -------
    simu : UnivariateFunctionalData
        A UnivariateFunctionalData object containing the simulated brownian
        motion evaluated on ``argvals``.

    Example
    -------
    >>> simulate_brownian_(brownian_type='standard',
    >>>                    argvals=np.arange(0, 1, 0.05),
    >>>                    norm=False)

    """
    if brownian_type == 'standard':
        simu = standard_brownian(argvals, x0=kwargs.get('x0', 0.0))
    elif brownian_type == 'geometric':
        simu = geometric_brownian(argvals,
                                  x0=kwargs.get('x0', 1.0),
                                  mu=kwargs.get('mu', 0.0),
                                  sigma=kwargs.get('sigma', 1.0))
    elif brownian_type == 'fractional':
        simu = fractional_brownian(argvals, H=kwargs.get('H', 0.5))
    else:
        raise ValueError('Brownian type not implemented!')
    return simu


#############################################################################
# Definition of the eigenvalues

def eigenvalues_linear(M=3):
    """Generate linear decreasing eigenvalues.

    Parameters
    ----------
    M : int, default = 3
        Number of eigenvalues to generates

    Returns
    -------
    val : numpy.ndarray
        The generated eigenvalues

    Example
    -------
    >>> eigenvalues_linear(M=3)
    array([1.0, 0.6666666666666666, 0.3333333333333333])

    """
    return np.array([(M - m + 1) / M for m in np.linspace(1, M, M)])


def eigenvalues_exponential(M=3):
    """Generate exponential decreasing eigenvalues.

    Parameters
    ----------
    M : int, default = 3
        Number of eigenvalues to generates

    Returns
    -------
    val : numpy.ndarray
        The generated eigenvalues

    Example
    -------
    >>> eigenvalues_exponential(M=3)
    array([0.36787944117144233, 0.22313016014842982, 0.1353352832366127])

    """
    return [np.exp(-(m + 1) / 2) for m in np.linspace(1, M, M)]


def eigenvalues_wiener(M=3):
    """Generate eigenvalues from a Wiener process.

    Parameters
    ----------
    M : int, default = 3
        Number of eigenvalues to generates

    Returns
    -------
    val : numpy.ndarray
        The generated eigenvalues

    Example
    -------
    >>> eigenvalues_wiener(M=3)
    array([0.4052847345693511, 0.04503163717437235, 0.016211389382774045])

    """
    return np.array([np.power((np.pi / 2) * (2 * m - 1), -2)
                    for m in np.linspace(1, M, M)])


def simulate_eigenvalues_(eigenvalues_name, M=3):
    """Redirects to the right simulation eigenvalues function.

    Parameters
    ----------
    eigenvalues_name : str
        Name of the eigenvalues generation process to use.
    M : int, default = 3
        Number of eigenvalues to generates.

    Returns
    -------
    eigenvalues: numpy.ndarray
        The generated eigenvalues

    Example
    -------
    >>> simulate_eigenvalues('linear', M=3)
    array([1.0, 0.6666666666666666, 0.3333333333333333])

    """
    if eigenvalues_name == 'linear':
        eigenvalues = eigenvalues_linear(M)
    elif eigenvalues_name == 'exponential':
        eigenvalues = eigenvalues_exponential(M)
    elif eigenvalues_name == 'wiener':
        eigenvalues = eigenvalues_wiener(M)
    else:
        raise ValueError('Eigenvalues not implemented!')
    return eigenvalues


#############################################################################
# Class Simulation


class Simulation(object):
    """An object to simulate functional data."""

    def __init__(self, N, M, G=1):
        """Initialize Simulation object.

        Parameters
        ----------
        N: int
            Number of curves to simulate.
        M: int or numpy.ndarray
            Sampling points.
            If M is int, we use np.linspace(0, 1, M) as sampling points.
            Otherwise, we use the provided numpy.ndarray.
        G: int
            Number of clusters to simulate.

        """
        self.N_ = N
        if isinstance(M, int):
            M = np.linspace(0, 1, M)
        self.M_ = M
        self.G_ = G

    def new(self, **kwargs):
        """Function to simulate observations.

        TODO: To redefine.

        """
        pass

    def add_noise(self, noise_var=1, sd_function=None):
        r"""Add noise to the data.

        Model:
        .. math:: Z(t) = f(t) + \sigma(f(t))\epsilon

        If sd_function is None, sigma(f(t)) = 1 and epsilon ~ N(0, noise_var)
        Else, we consider heteroscedastic noise with:
            - sigma(f(t)) = sd_function(self.obs.values)
            - epsilon ~ N(0,1)

        Parameters
        ----------
        noise_var : float
            Variance of the noise to add.
        sd_function : callable
            Standard deviation function for heteroscedatic noise.

        """
        noisy_data = []
        for i in self.obs_:
            if sd_function is None:
                noise = np.random.normal(0, np.sqrt(noise_var),
                                         size=len(self.M_))
            else:
                noise = sd_function(i.values) *\
                    np.random.normal(0, 1, size=len(self.obs_.argvals[0]))
            noise_func = UnivariateFunctionalData(
                self.obs_.argvals, np.array(noise, ndmin=2))
            noisy_data.append(i + noise_func)

        data = MultivariateFunctionalData(noisy_data)
        self.noisy_obs_ = data.asUnivariateFunctionalData()


class Basis(Simulation):
    r"""A functional data object representing an orthogonal basis of functions.

    The function are simulated using the Karhunen-Loève decomposition :
    .. math::
        X_i(t) = \mu(t) + \sum_{j = 1}^M c_{i,j}\phi_{i,j}(t), i = 1, ..., N

    Parameters:
    -----------
    basis: str or numpy.ndarray
        If basis is str, denotes the basis of functions to use.
        If basis is numpy.ndarray, provides the basis to use.
    K: int
        Number of basis functions to use to simulate the data.
    eigenvalues: str or numpy.ndarray
        Define the decreasing of the eigenvalues of the process.
        If `eigenvalues` is str, we define the eigenvalues as using the
        corresponding function. Otherwise, we keep it like that.
    norm: bool
        Should we normalize the basis function?

    Attributes
    ----------
    coef_: numpy.ndarray
        Array of coefficients c_{i,j}
    obs: FDApy.univariate_functional.UnivariateFunctionalData
        Simulation of univariate functional data

    """

    def __init__(self, N, M, basis, K, eigenvalues, norm):
        """Initialize Basis object."""
        Simulation.__init__(self, N, M)
        self.K_ = K
        self.norm_ = norm

        # Define the basis
        if isinstance(basis, str):
            self.basis_name_ = basis
            self.basis_ = simulate_basis_(self.basis_name_, self.K_,
                                          self.M_, self.norm_).values
        elif isinstance(basis, np.ndarray):
            self.basis_name_ = 'user_provided'
            self.basis_ = basis
        else:
            raise ValueError('Error with the basis.')

        # Define the decreasing of the eigenvalues
        if isinstance(eigenvalues, str):
            eigenvalues = simulate_eigenvalues_(eigenvalues, self.K_)
        self.eigenvalues_ = eigenvalues

    def new(self, **kwargs):
        """Function that simulates :math::`N` observations."""
        # Simulate the N observations
        obs = np.empty(shape=(self.N_, len(self.M_)))
        coef = np.empty(shape=(self.N_, len(self.eigenvalues_)))
        for i in range(self.N_):
            coef_ = np.random.normal(0, np.sqrt(self.eigenvalues_))
            prod_ = np.matmul(coef_[np.newaxis], self.basis_)

            obs[i, :] = prod_
            coef[i, :] = coef_

        self.coef_ = coef
        self.obs_ = UnivariateFunctionalData(self.M_, obs)


class Brownian(Simulation):
    """A functional data object representing a brownian motion.

    Parameters
    ----------
    N: int, default=100
        Number of curves to simulate.
    brownian_type: str, default='regular'
        Type of brownian motion to simulate.
        One of 'regular', 'geometric' or 'fractional'.

    """

    def __init__(self, N, M, brownian_type='standard'):
        """Initialize Brownian object."""
        Simulation.__init__(self, N, M)
        self.brownian_type_ = brownian_type

    def new(self, **kwargs):
        """Function that simulates `N` observations."""
        param_dict = {k: kwargs.pop(k) for k in dict(kwargs)}

        # Simulate the N observations
        obs = []
        for _ in range(self.N_):
            obs.append(simulate_brownian_(self.brownian_type_,
                                          self.M_, **param_dict))

        data = MultivariateFunctionalData(obs)

        self.obs_ = data.asUnivariateFunctionalData()
