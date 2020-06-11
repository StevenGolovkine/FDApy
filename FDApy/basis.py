#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""Simulation functions.

This module is used to define an abstract Simulation class and two classes
derived from it, the Basis class and the Brownian class. Thus, we may simulate
different data from a linear combination of basis functions or multiple
realizations of diverse Brownian motion.
"""
import numpy as np
import scipy

from abc import ABC, abstractmethod
from sklearn.datasets import make_blobs

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
    K: int, default = 3
        Maximum degree of the Legendre polynomials.
    argvals: numpy.ndarray, default = None
        The values on which evaluated the Legendre polynomials. If ``None``,
        the polynomials are evaluated on the interval :math:`[-1, 1]`.
    norm: boolean, default = True
        Should we normalize the functions?

    Returns
    -------
    obj: UnivariateFunctionalData
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
    K: int, default = 3
        Number of functions to consider.
    argvals: numpy.ndarray, default = None
         The values on which the eigenfunctions of a Wiener process are
         evaluated. If ``None``, the functions are evaluated on the interval
         :math:`[0, 1]`.
    norm: boolean, default = True
        Should we normalize the functions?

    Returns
    -------
    obj: UnivariateFunctionalData
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
    """Redirect to the right simulation basis function.

    Parameters
    ----------
    basis_name: str, {'legendre', 'wiener'}
        Name of the basis to use.
    K: int, default = 3
        Number of functions to compute.
    argvals: numpy.ndarray, default = None
        The values on which the basis functions are evaluated. If ``None``,
        the functions are evaluated on the interval :math:`[0, 1]`.
    norm: boolean
        Should we normalize the functions?

    Returns
    -------
    basis: UnivariateFunctionalData
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
    delta, argvals: (float, numpy.ndarray)
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
    obj: UnivariateFunctionalData
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
    argvals: numpy.ndarray, default=None
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
    obj: UnivariateFunctionalData
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
    obj: UnivariateFunctionalData
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
    """Redirect to the right brownian motion function.

    Parameters
    ----------
    brownian_type: str, {'standard', 'geometric', 'fractional'}
        Name of the Brownian motion to simulate.
    argvals: numpy.ndarray
        The sampling points on which the Brownian motion is evaluated. If
        ``None``, the Brownian is evaluated on the interval :math:`[0, 1]`.
    norm: boolean
        Should we normalize the simulation?

    Keyword Args
    ------------
    x0: float, default = 0.0 or 1.0
        Start of the Brownian motion. Should be strictly positive if
        ``brownian_type == 'geometric'``.
    mu: float, default = 0
        The interest rate
    sigma: float, default = 1
        The diffusion coefficient
    H: double, default = 0.5
        Hurst parameter

    Returns
    -------
    simu: UnivariateFunctionalData
        A UnivariateFunctionalData object containing the simulated brownian
        motion evaluated on ``argvals``.

    Example
    -------
    >>> simulate_brownian(brownian_type='standard',
    >>>                   argvals=np.arange(0, 1, 0.05),
    >>>                   norm=False)

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
    M: int, default = 3
        Number of eigenvalues to generates

    Returns
    -------
    val: numpy.ndarray
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
    M: int, default = 3
        Number of eigenvalues to generates

    Returns
    -------
    val: numpy.ndarray
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
    M: int, default = 3
        Number of eigenvalues to generates

    Returns
    -------
    val: numpy.ndarray
        The generated eigenvalues

    Example
    -------
    >>> eigenvalues_wiener(M=3)
    array([0.4052847345693511, 0.04503163717437235, 0.016211389382774045])

    """
    return np.array([np.power((np.pi / 2) * (2 * m - 1), -2)
                    for m in np.linspace(1, M, M)])


def simulate_eigenvalues(eigenvalues_name, M=3):
    """Redirect to the right simulation eigenvalues function.

    Parameters
    ----------
    eigenvalues_name: str
        Name of the eigenvalues generation process to use.
    M: int, default = 3
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


class Simulation(ABC):
    """An abstract class for the simulation of functional data.

    Parameters
    ----------
    N: int
        Number of curves to simulate.
    M: int or numpy.ndarray
        Sampling points. If ``M`` is an integer, we use
        ``np.linspace(0, 1, M)`` as sampling points. Otherwise, we use the
        provided numpy.ndarray.

    Attributes
    ----------
    basis_name: str
        Name of the basis used.

    """

    def __init__(self, N, M):
        """Initialize Simulation object."""
        if isinstance(M, int):
            M = np.linspace(0, 1, M)

        super().__init__()
        self.basis_name = None
        self.N = N
        self.M = M

    @abstractmethod
    def new(self, **kwargs):
        """Function to simulate observations."""
        pass

    def make_coef(self, n_features, n_clusters, centers, cluster_std):
        """Simulate coefficient for the Karhunen-Loève decomposition.

        Parameters
        ----------
        n_features: int
            Number of features to simulate.
        n_clusters: int
            Number of clusters to simulate.
        centers: numpy.ndarray, (n_features, n_clusters)
            The centers of the clusters to generate. The ``n_features``
            correspond to the number of functions within the basis.
        cluster_std: np.ndarray, (n_features, n_clusters)
            The standard deviation of the clusters to generate. The
            ``n_features`` correspond to the number of functions within the
            basis.

        """
        coef = np.zeros((self.N, n_features))
        for idx in np.arange(n_features):
            X, y = make_blobs(n_samples=self.N, n_features=1,
                              centers=centers[idx, :].reshape(-1, 1),
                              cluster_std=cluster_std[idx, :],
                              shuffle=False)
            coef[:, idx] = X.squeeze()
        labels = y
        return coef, labels

    def add_noise(self, noise_var=1, sd_function=None):
        r"""Add noise to the data.

        Parameters
        ----------
        noise_var : float
            Variance of the noise to add.
        sd_function : callable
            Standard deviation function for heteroscedatic noise.

        Notes
        -----

        Model:

        .. math::
            Z(t) = f(t) + \sigma(f(t))\epsilon

        If ``sd_function is None``, :math:`\sigma(f(t)) = 1` and
        :math:`\epsilon \sim \mathcal{N}(0, \sigma^2)`.
        Else, we consider heteroscedastic noise with:
            - :math:`\sigma(f(t)) =` sd_function(self.obs.values)
            - :math:`\epsilon \sim \mathcal{N}(0,1)`.

        """
        noisy_data = []
        for i in self.data:
            if sd_function is None:
                noise = np.random.normal(0, np.sqrt(noise_var),
                                         size=len(self.M))
            else:
                noise = sd_function(i.values) *\
                    np.random.normal(0, 1, size=len(self.data.argvals[0]))
            noise_func = UnivariateFunctionalData(
                self.data.argvals, np.array(noise, ndmin=2))
            noisy_data.append(i + noise_func)

        data = MultivariateFunctionalData(noisy_data)
        self.noisy_obs = data.asUnivariateFunctionalData()


class Basis(Simulation):
    r"""A functional data object representing an orthogonal basis of functions.

    Parameters
    ----------
    basis: str, {'legendre', 'wiener'}
        Denotes the basis of functions to use.
    n_features: int, default = 1
        Number of basis functions to use to simulate the data.
    n_clusters: int, default = 1
        Number of clusters to simulate.
    centers: numpy.ndarray, (n_features, n_clusters)
        The centers of the clusters to generate.
    cluster_std: str, np.ndarray, (n_features, n_clusters)
        The standard deviation of the clusters to generate.
    norm: bool, default=False
        Should we normalize the basis function?

    Attributes
    ----------
    data: UnivariateFunctionalData or MultivariateFunctionalData
        The simulated data :math:`X_i(t)`.
    labels: numpy.array, (N, )
        True class labels for each data.
    coef: numpy.ndarray, (N, n_features)
        The simulated coefficient :math:`c_{i,j}`.

    Notes
    -----

    The function are simulated using the Karhunen-Loève decomposition:

    .. math::
        X_i(t) = \mu(t) + \sum_{j = 1}^K c_{i,j}\phi_{j}(t), i = 1, ..., N


    """

    def __init__(self, N, M, basis, n_features=1, n_clusters=1, centers=0,
                 cluster_std=1, norm=False):
        """Initialize Basis object."""
        super().__init__(N, M)
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.norm = norm
        self.centers = centers

        # Define the basis
        self.basis_name = basis
        self.basis = simulate_basis(self.basis_name,
                                    self.n_features,
                                    self.M,
                                    self.norm)

        # Define the decreasing of the eigenvalues
        if isinstance(cluster_std, str):
            eigenvalues = simulate_eigenvalues(cluster_std,
                                               self.n_features)
            eigenvalues = np.repeat(eigenvalues, self.n_clusters)
            self.cluster_std = eigenvalues.reshape((self.n_features,
                                                    self.n_clusters))
        else:
            self.cluster_std = cluster_std

    def new(self, **kwargs):
        """Function that simulates :math:`N` observations.

        Returns
        -------
        coef: numpy.ndarray, (N, n_features)
            Array of simulated coefficients :math:`c_{i,j}`.
        data: UnivariateFunctionalData
            The simulated data :math:`X_i(t)`.

        """
        coef, y = self.make_coef(self.n_features,
                                 self.n_clusters,
                                 self.centers,
                                 self.cluster_std)
        obs = np.matmul(coef, self.basis.values)
        self.data = UnivariateFunctionalData(self.M, obs)
        self.labels = y
        self.coef = coef


class BasisFPCA(Simulation):
    r"""Class for the simulation of data using a FPCA basis.

    Parameters
    ----------
    basis: FPCA or MFPCA object
        Results of a functional principal component analysis or a multivariate
        functional data analysis.
    n_clusters: int, default = 1
        Number of clusters to simulate.
    centers: numpy.ndarray, (n_features, n_clusters)
        The centers of the clusters to generate. The ``n_features`` correspond
        to the number of functions within the FPCA or MFPCA basis.
    cluster_std: np.ndarray, (n_features, n_clusters)
        The standard deviation of the clusters to generate. The ``n_features``
        correspond to the number of functions within the FPCA or MFPCA basis.

    Attributes
    ----------
    n_features: int
        Number of functions within the FPCA or MFPCA basis.
    data: UnivariateFunctionalData or MultivariateFunctionalData
        The simulated data :math:`X_i(t)`.
    labels: numpy.array, (N, )
        True class labels for each data.
    coef: numpy.ndarray, (N, n_features)
        The simulated coefficient :math:`c_{i,j}`.

    Notes
    -----
    The function are simulated using the Karhunen-Loève decomposition:

    .. math::
        X_i(t) = \mu(t) + \sum_{j = 1}^M c_{i, j}\phi_{i, j}(t), i = , \dots, N

    The number of sampling points :math:`M` is not used for the simulation of
    data using FPCA or MFPCA. The simulated curves will have the same length
    than the eigenfunctions.

    """

    def __init__(self, N, M, basis, n_clusters=1, centers=0, cluster_std=1):
        """Initialize BasisFPCA object."""
        super().__init__(N, M)
        self.basis = basis
        self.n_clusters = n_clusters
        self.n_features = len(basis.eigenvalues)
        self.centers = centers
        self.cluster_std = cluster_std

    def new(self, **kwargs):
        """Function that simulates :math:`N` observations."""
        coef, y = self.make_coef(self.n_features,
                                 self.n_clusters,
                                 self.centers,
                                 self.cluster_std)

        self.data = self.basis.inverse_transform(coef)
        self.labels = y
        self.coef = coef


class Brownian(Simulation):
    """A functional data object representing a Brownian motion.

    Parameters
    ----------
    N: int
        Number of curves to simulate.
    M: int or numpy.ndarray
        Sampling points. If ``M`` is an integer, we use
        ``np.linspace(0, 1, M)`` as sampling points. Otherwise, we use the
        provided numpy.ndarray.
    brownian_type: str, {'standard', 'geometric', 'fractional'}
        Type of brownian motion to simulate.
    n_clusters: int, default = 1
        Number of clusters to simulate. Not used in this context.

    """

    def __init__(self, N, M, brownian_type='standard', n_clusters=1):
        """Initialize Brownian object."""
        super().__init__(N, M, n_clusters)
        self.basis_name = brownian_type

    def new(self, **kwargs):
        """Function that simulates `N` observations.

        Keyword Args
        ------------
        x0: float, default = 0.0 or 1.0
            Start of the Brownian motion. Should be strictly positive if
            ``brownian_type == 'geometric'``.
        mu: float, default = 0
            The interest rate
        sigma: float, default = 1
            The diffusion coefficient
        H: double, default = 0.5
            Hurst parameter

        Returns
        -------
        data: UnivariateFunctionalData
            The simulated functions
        """
        param_dict = {k: kwargs.pop(k) for k in dict(kwargs)}

        # Simulate the N observations
        obs = []
        for _ in range(self.N):
            obs.append(simulate_brownian(self.basis_name,
                                         self.M, **param_dict))

        data = MultivariateFunctionalData(obs)
        self.data = data.asUnivariateFunctionalData()
