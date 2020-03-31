#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import scipy

from .univariate_functional import UnivariateFunctionalData
from .multivariate_functional import MultivariateFunctionalData


#######################################################################
# Definition of the basis (eigenfunctions)

def basis_legendre(K=3, argvals=None, norm=True):
    """Define Legendre basis of function.

    Build a basis of `K` functions using Legendre polynomials on the interval
    `argvals`.

    Parameters
    ----------
    K : int, default = 3
        Maximum degree of the Legendre polynomials.
    argvals : numpy.ndarray, default = None
        The values on which evaluated the Legendre polynomials. If `None`, the
        polynomials are evaluated on the interval [-1, 1].
    norm : boolean, default = True
        Do we normalize the functions?

    Return
    ------
    obj : UnivariateFunctionalData
        A UnivariateFunctionalData object containing the Legendre
        polynomial up to `K` functions evaluated on `argvals`.
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

    Build a basis of functions of the Wiener process.

    Parameters
    ----------
    K : int, default = 3
        Number of functions to compute.
    argvals : numpy.ndarray, default = None
         The values on which evaluated the Wiener basis functions.
         If `None`, the functions are evaluated on the interval [0, 1].
    norm : boolean, default = True
        Do we normalize the functions?

    Return
    ------
    obj : UnivariateFunctionalData
        A UnivariateFunctionalData object containing `K` Wiener basis functions
        evaluated on `argvals`.

    Example
    -------
    >>>basis_wiener(K=3, argvals=np.arange(0, 1, 0.05), norm=True)
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


def simulate_basis_(basis_name, K, argvals, norm):
    """Function that redirects to the right simulation basis function.

    Parameters
    ----------
    basis_name : str
        Name of the basis to use.
    K : int
        Number of functions to compute.
    argvals : tuple or numpy.ndarray
        The values on which evaluated the Wiener basis functions. If `None`,
        the functions are evaluated on the interval [0, 1].
    norm : boolean
        Do we normalize the functions?

    Return
    ------
    basis_ : UnivariateFunctionalData
        A UnivariateFunctionalData object containing `M` basis functions
        evaluated on `argvals`.

    Example
    -------
    >>>simulate_basis_('legendre', M=3,
        argvals=np.arange(-1, 1, 0.1), norm=True)
    """
    if basis_name == 'legendre':
        basis_ = basis_legendre(K, argvals, norm)
    elif basis_name == 'wiener':
        basis_ = basis_wiener(K, argvals, norm)
    else:
        raise ValueError('Basis not implemented!')
    return basis_


#############################################################################
# Definition of the different Browian motion

def standard_brownian_(argvals=None, x0=0.0):
    """Function that generate standard brownian motions.

    Generate one dimensional standard brownian motion.

    Parameters
    ----------
    argvals: tuple or numpy.ndarray, default=None
        The values on which evaluated the brownian motion. If `None`,
        the functions are evaluated on the interval [0, 1].
    x0: double, default=0.0
        Start of the brownian motion.

    Return
    ------
    A univariate functional data object.

    References
    ----------
    - https://github.com/cran/somebm/blob/master/R/bm.R
    """

    if argvals is None:
        argvals = np.arange(0, 1, 0.05)

    t0 = np.min(argvals)
    t1 = np.max(argvals)
    M = np.size(argvals)

    # For one brownian motion
    delta = (t1 - t0) / M
    W = np.zeros(M)
    W[0] = x0

    for idx in range(1, M):
        W[idx] = W[idx - 1] + np.sqrt(delta) * np.random.normal()

    obj = UnivariateFunctionalData(
        argvals=argvals, values=W[np.newaxis])
    return obj


def geometric_brownian_(argvals=None, x0=1.0, mu=0, sigma=1):
    """Function that generate geometric brownian motions.

    Generate one dimensional geometric brownian motion.

    Parameters
    ----------
    argvals: tuple or numpy.ndarray, default=None
        The values on which evaluated the geometric brownian motion. If `None`,
        the brownian is evaluated on the interval [0, 1].
    x0: double, default=1.0
        Start of the brownian motion. Careful, should be stricly greater than 0
    mu: double, default=0
        The interest rate
    sigma: double, default=1
        The diffusion coefficient

    Return
    ------
    A univariate functional data object.

    References
    ----------
    - https://github.com/cran/somebm/blob/master/R/bm.R
    """

    if argvals is None:
        argvals = np.arange(0, 1, 0.05)

    t0 = np.min(argvals)
    t1 = np.max(argvals)
    M = np.size(argvals)

    # For one geometric brownian motion.
    delta = (t1 - t0) / M
    W = np.zeros(M)
    W[0] = 0

    for idx in range(1, M):
        W[idx] = W[idx - 1] + np.sqrt(delta) * np.random.normal()

    S = x0 * np.exp((mu - np.power(sigma, 2) / 2) * (argvals - t0) + sigma * W)

    obj = UnivariateFunctionalData(
        argvals=argvals, values=S[np.newaxis])
    return obj


def fractional_brownian_(argvals=None, hurst=0.5):
    """Function that generate fractional brownian moitions.

    Generate one dimension fractional brownian motion with a given Hurst
    parameter.

    Parameters
    ----------
    argvals: tuple or numpy.ndarray, default=None
        The values on which evaluated the fractional brownian motion.
        If `None`, the brownian is evaluated on the interval [0, 1].
    hurst: double, default=0.5
        Hurst parameter

    Return
    ------
    A univariate functional data object.

    References
    ----------
    - https://github.com/cran/somebm/blob/master/R/bm.R
    """

    if argvals is None:
        argvals = np.arange(0, 1, 0.05)

    M = np.size(argvals)
    R = np.zeros(M + 1)
    R[0] = 1
    for idx in range(1, M + 1):
        R[idx] = 0.5 * (np.power(idx + 1, 2 * hurst) - 2 *
                        np.power(idx, 2 * hurst) +
                        np.power(idx - 1, 2 * hurst))

    invR = R[::-1]
    R = np.append(R, invR[1:len(invR) - 1])
    lamb = np.real(np.fft.fft(R) / (2 * M))

    W = np.fft.fft(np.sqrt(lamb) * (np.random.normal(size=2 * M) +
                                    np.random.normal(size=2 * M) * (0 + 1j)))
    W = np.power(M, -hurst) * np.cumsum(np.real(W[1:(M + 1)]))

    obj = UnivariateFunctionalData(
        argvals=argvals, values=W[np.newaxis])
    return obj


def simulate_brownian_(brownian_type, argvals=None, norm=False, **kwargs):
    """Fonction that redirects to the right brownian motion function.

    Parameters
    ----------
    brownian_type: str
        Name of the brownian motion to simulate.
    argvals: tuple or numpy.ndarray
        The sampling points for the brownian motion.
    norm: boolean
        Do we normalize the simulation?

    Return
    ------
    simu_: FDApy.univariate_functional.UnivariateFunctionalData
        A UnivariateFunctionalData object containing the simulated brownian
        motion evaluated on `argvals`.

    Example
    -------
    >>>simulate_brownian_(brownian_type='standard',
        argvals=np.arange(0, 1, 0.05), norm=False)
    """
    if brownian_type == 'standard':
        simu_ = standard_brownian_(argvals, x0=kwargs['x0'])
    elif brownian_type == 'geometric':
        simu_ = geometric_brownian_(argvals,
                                    x0=kwargs['x0'],
                                    mu=kwargs['mu'],
                                    sigma=kwargs['sigma'])
    elif brownian_type == 'fractional':
        simu_ = fractional_brownian_(argvals, hurst=kwargs['hurst'])
    else:
        raise ValueError('Brownian type not implemented!')
    return simu_


#############################################################################
# Definition of the eigenvalues

def eigenvalues_linear(M=3):
    """Function that generate linear decreasing eigenvalues.

    Parameters
    ----------
    M : int, default = 3
        Number of eigenvalues to generates

    Return
    ------
    val : list
        The generated eigenvalues

    Example
    -------
    >>>eigenvalues_linear(M=3)
    [1.0, 0.6666666666666666, 0.3333333333333333]
    """
    return [(M - m + 1) / M for m in np.linspace(1, M, M)]


def eigenvalues_exponential(M=3):
    """Function that generate exponential decreasing eigenvalues.

    Parameters
    ----------
    M : int, default = 3
        Number of eigenvalues to generates

    Return
    ------
    val : list
        The generated eigenvalues

    Example
    -------
    >>>eigenvalues_exponential(M=3)
    [0.36787944117144233, 0.22313016014842982, 0.1353352832366127]
    """
    return [np.exp(-(m + 1) / 2) for m in np.linspace(1, M, M)]


def eigenvalues_wiener(M=3):
    """Function that generate eigenvalues from a Wiener process.

    Parameters
    ----------
    M : int, default = 3
        Number of eigenvalues to generates

    Return
    ------
    val : list
        The generated eigenvalues
    """
    return [np.power((np.pi / 2) * (2 * m - 1), -2)
            for m in np.linspace(1, M, M)]


def simulate_eigenvalues_(eigenvalues_name, M):
    """Function that redirects to the right simulation eigenvalues function.

    Parameters
    ----------
    eigenvalues_name : str
        Name of the eigenvalues generation process to use.
    M : int
        Number of eigenvalues to generates

    Return
    ------
    eigenvalues_: list
        The generated eigenvalues

    Example
    -------
    >>>simulate_eigenvalues_('linear', M=3)
    [1.0, 0.6666666666666666, 0.3333333333333333]
    """
    if eigenvalues_name == 'linear':
        eigenvalues_ = eigenvalues_linear(M)
    elif eigenvalues_name == 'exponential':
        eigenvalues_ = eigenvalues_exponential(M)
    elif eigenvalues_name == 'wiener':
        eigenvalues_ = eigenvalues_wiener(M)
    else:
        raise ValueError('Eigenvalues not implemented!')
    return eigenvalues_


#############################################################################
# Class Simulation


class Simulation(object):
    """An object to simulate functional data.

    Parameters
    ----------
    N: int
        Number of curves to simulate.
    M: int or numpy.ndarray
        Sampling points.
        If M is int, we use np.linspace(0, 1, M) as sampling points.
        Otherwise, we use the numpy.ndarray.
    """
    def __init__(self, N, M):
        self.N_ = N
        if isinstance(M, int):
            M = np.linspace(0, 1, M)
        self.M_ = M

    def new():
        """Function to simulate observations.
        To redefine.
        """
        pass

    def add_noise(self, noise_var=1, sd_function=None):
        """Add noise to the data.

        Model: Z(t) = f(t) + sigma(f(t))epsilon

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
    """A functional data object representing an orthogonal (or orthonormal)
    basis of functions.

    The function are simulated using the Karhunen-Loève decomposition :
        X_i(t) = mu(t) + sum_{j = 1}^M c_{i,j}phi_{i,j}(t), i = 1, ..., N

    Parameters:
    -----------
    basis_name: str
        String which denotes the basis of functions to use.
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

    Notes
    -----

    References
    ---------

    """
    def __init__(self, N, M, basis_name, K, eigenvalues, norm):
        Simulation.__init__(self, N, M)
        self.basis_name_ = basis_name
        self.K_ = K
        self.norm_ = norm

        # Define the basis
        self.basis_ = simulate_basis_(self.basis_name_,
                                      self.K_, self.M_, self.norm_)

        # Define the decreasing of the eigenvalues
        if isinstance(eigenvalues, str):
            eigenvalues = simulate_eigenvalues_(eigenvalues, self.K_)
        self.eigenvalues_ = eigenvalues

    def new(self):
        """Function that simulates `N` observations.

        Parameters
        ----------

        """

        # Simulate the N observations
        obs = np.empty(shape=(self.N_, len(self.M_)))
        coef = np.empty(shape=(self.N_, len(self.eigenvalues_)))
        for i in range(self.N_):
            coef_ = list(np.random.normal(0, self.eigenvalues_))
            prod_ = coef_ * self.basis_

            obs[i, :] = prod_.values.sum(axis=0)
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
        Simulation.__init__(self, N, M)
        self.brownian_type_ = brownian_type

    def new(self, **kwargs):
        """Function that simulates `N` observations.

        Parameters
        ----------

        """
        param_dict = {k: kwargs.pop(k) for k in dict(kwargs)}

        # Simulate the N observations
        obs = []
        for i in range(self.N_):
            obs.append(simulate_brownian_(self.brownian_type_,
                                          self.M_, **param_dict))

        data = MultivariateFunctionalData(obs)

        self.obs_ = data.asUnivariateFunctionalData()
