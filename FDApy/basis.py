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
from patsy import bs
from sklearn.datasets import make_blobs

from .fpca import MFPCA, UFPCA
from .univariate_functional import UnivariateFunctionalData
from .multivariate_functional import MultivariateFunctionalData


#######################################################################
# Definition of the basis (eigenfunctions)

def basis_legendre(K=3, argvals=None, norm=True):
    r"""Define Legendre basis of function.

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

    Notes
    -----

    The Legendre basis is defined by induction as:

    .. math::
        (n + 1)P_{n + 1}(t) = (2n + 1)tP_n(t) - nP_{n - 1}(t), \quad\text{for}
        \quad n \geq 1,

    with :math:`P_0(t) = 1` and :math:`P_1(t) = t`.

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
    r"""Define Wiener basis of function.

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
        eigenfunctions up to :math:`K` functions evaluated on ``argvals``.

    Notes
    -----

    The Wiener basis is defined as the eigenfunctions of the Brownian motion:

    .. math::
        \phi_k(t) = \sqrt{2}\sin\left(\left(k - \frac{1}{2}\right)\pi t\right),
        \quad 1 \leq k \leq K


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


def basis_fourier(K=3, argvals=None, period=2 * np.pi, norm=True):
    r"""Define Fourier basis of function.

    Build a basis of :math:`K` functions using Fourier series on the
    interval defined by ``argvals``.

    Parameters
    ----------
    K: int, default = 3
        Number of considered Fourier series. Should be odd.
    argvals: numpy.ndarray, default = None
        The values on which evaluated the Fourier series. If ``None``,
        the polynomials are evaluated on the interval :math:`[0, period]`.
    period: float, default = 2*numpy.pi
        The period of the circular functions.
    norm: boolean, default = True
        Should we normalize the functions?

    Returns
    -------
    obj: UnivariateFunctionalData
        A UnivariateFunctionalData object containing the Fourier series
        up to :math:`K` functions evaluated on ``argvals``.

    Notes
    -----

    The Fourier basis is defined as:

    .. math::
        \Phi(t) = \left(1, \sin(\omega t), \cos(\omega t), \dots \right)

    where :math:`\omega` is the period.

    Examples
    --------
    >>> basis_fourier(K=3, argvals=np.arange(0, 2*np.pi, 0.1), norm=True)

    """
    K_new = K + 1 if K % 2 == 0 else K
    if argvals is None:
        argvals = np.arange(0, period, 0.1)
    if isinstance(argvals, list):
        raise ValueError('argvals has to be a tuple or a numpy array!')

    values = np.empty((K_new, len(argvals)))
    values[0, :] = 1
    for k in np.arange(1, (K_new + 1) // 2):
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

    obj = UnivariateFunctionalData(argvals, values[:K, :])
    return obj


def basis_bsplines(K=5, argvals=None, degree=3, knots=None, norm=False):
    """Define B-splines basis of function.

    Build a basis of :math:`K` functions using B-splines basis on the
    interval defined by ``argvals``.

    Parameters
    ----------
    K: int, default = 5
        Number of considered B-splines.
    argvals: numpy.ndarray, default = None
        The values on which evaluated the B-splines. If ``None``,
        the polynomials are evaluated on the interval :math:`[0, 1]`.
    degree: int, default = 3
        Degree of the B-splines. The default gives cubic splines.
    knots: numpy.ndarray, (n_knots,)
        Specify the break points defining the B-splines. If ``knots``
        are provided, the provided value of ``K`` is ignored. And the
        number of basis functions is ``n_knots + degree - 1``.
    norm: boolean, default = True
        Should we normalize the functions?

    Returns
    -------
    obj: UnivariateFunctionalData
        A UnivariateFunctionalData object containing the ``K`` B-splines
        functions evaluated on ``argvals``.

    Examples
    --------
    >>> basis_bsplines(K=5, argvals=np.arange(0, 1, 0.01), norm=False)

    """
    if argvals is None:
        argvals = np.arange(0, 1, 0.01)
    if isinstance(argvals, list):
        raise ValueError('argvals has to be a tuple or a numpy array!')

    if knots is not None:
        n_knots = len(knots)
        K = n_knots + degree - 1
    else:
        n_knots = K - degree + 1
        knots = np.linspace(argvals[0], argvals[-1], n_knots)

    values = bs(argvals, df=K, knots=knots[1:-1], degree=degree,
                include_intercept=True)
    if norm:
        norm2 = np.sqrt(scipy.integrate.simps(values * values, argvals,
                                              axis=0))
        values = values / norm2

    obj = UnivariateFunctionalData(argvals, values.T)
    return obj


def simulate_basis(basis_name, K=3, argvals=None, norm=False, **kwargs):
    """Redirect to the right simulation basis function.

    Parameters
    ----------
    basis_name: str, {'legendre', 'wiener', 'fourier'}
        Name of the basis to use.
    K: int, default = 3
        Number of functions to compute.
    argvals: numpy.ndarray, default = None
        The values on which the basis functions are evaluated. If ``None``,
        the functions are evaluated on the interval :math:`[0, 1]`.
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
    basis: UnivariateFunctionalData
        A UnivariateFunctionalData object containing :math:`K` basis functions
        evaluated on ``argvals``.

    Example
    -------
    >>> simulate_basis('legendre', M=3,
    >>>                argvals=np.arange(-1, 1, 0.1), norm=True)

    """
    if basis_name == 'legendre':
        basis = basis_legendre(K, argvals, norm)
    elif basis_name == 'wiener':
        basis = basis_wiener(K, argvals, norm)
    elif basis_name == 'fourier':
        basis = basis_fourier(K, argvals,
                              kwargs.get('period', 2 * np.pi), norm)
    elif basis_name == 'bsplines':
        basis = basis_bsplines(K, argvals,
                               kwargs.get('degree', 3),
                               kwargs.get('knots', None), norm)
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
    for idx in np.arange(1, np.size(argvals)):
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
    for idx in np.arange(1, np.size(argvals)):
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
    for idx in np.arange(1, M + 1):
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
# Definition of clusters
def make_coef(N, n_features, centers, cluster_std):
    """Simulate coefficient for the Karhunen-Loève decomposition.

    Parameters
    ----------
    N: int
        Number of wanted observations.
    n_features: int
        Number of features to simulate.
    centers: numpy.ndarray, (n_features, n_clusters)
        The centers of the clusters to generate. The ``n_features``
        correspond to the number of functions within the basis.
    cluster_std: np.ndarray, (n_features, n_clusters)
        The standard deviation of the clusters to generate. The
        ``n_features`` correspond to the number of functions within the
        basis.

    Returns
    -------
    coef: numpy.ndarray, (N, n_features)
        Array of generated coefficients
    labels: numpy.ndarray, (N, )
        The integer labels for cluster membership of each observations.

    Notes
    -----
    The function :func:`sklearn.datasets.make_blobs` does not allow different
    standard deviations for the different features. It only permits to change
    the standard deviations between clusters. To bypass that, we loop through
    the ``n_features``.

    """
    coef = np.zeros((N, n_features))
    for idx in np.arange(n_features):
        X, y = make_blobs(n_samples=N, n_features=1,
                          centers=centers[idx, :].reshape(-1, 1),
                          cluster_std=cluster_std[idx, :],
                          shuffle=False)
        coef[:, idx] = X.squeeze()
    labels = y
    return coef, labels


def initialize_centers(n_features, n_clusters, centers=None):
    """Initialize the centers of the clusters.

    Parameters
    ----------
    n_features: int
        Number of features to simulate.
    n_clusters: int
        Number of clusters to simulate.
    centers: numpy.ndarray, (n_features, n_clusters), default = None
        The centers of each cluster per feature.
    """
    return np.zeros((n_features, n_clusters)) if centers is None else centers


def initialize_cluster_std(n_features, n_clusters, cluster_std=None):
    """Initialize the standard deviation of the clusters."""
    if isinstance(cluster_std, str):
        eigenvalues = simulate_eigenvalues(cluster_std, n_features)
        eigenvalues = np.repeat(eigenvalues, n_clusters)
        return eigenvalues.reshape((n_features, n_clusters))
    elif cluster_std is None:
        return np.ones((n_features, n_clusters))
    else:
        return cluster_std


#############################################################################
# Metaclass Simulation
class Simulation(ABC):
    """Metaclass for the simulation of functional data.

    Parameters
    ----------
    N: int
        Number of curves to simulate.
    name: str, default = None
        Name of the simulation
    n_features: int, default = 1
        Number of features to simulate.
    n_clusters: int, default = 1
        Number of clusters to simulate.
    centers: numpy.ndarray, (n_features, n_clusters)
        The centers of the clusters to generate. The ``n_features``
        correspond to the number of functions within the basis.
    cluster_std: np.ndarray, (n_features, n_clusters)
        The standard deviation of the clusters to generate. The
        ``n_features`` correspond to the number of functions within the
        basis.

    Arguments
    ---------
    coef: numpy.ndarray, (N, n_features)
        Array of generated coefficients
    labels: numpy.ndarray, (N, )
        The integer labels for cluster membership of each observations.

    """

    def __init__(self, N, name=None, n_features=1, n_clusters=1,
                 centers=None, cluster_std=None):
        """Initialize Simulation object."""
        super().__init__()
        self.N = N
        self.name = name
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.centers = initialize_centers(n_features, n_clusters, centers)
        self.cluster_std = initialize_cluster_std(n_features, n_clusters,
                                                  cluster_std)

    @abstractmethod
    def new(self, **kwargs):
        """Function to simulate observations."""
        self.coef, self.labels = make_coef(self.N, self.n_features,
                                           self.centers, self.cluster_std)


#############################################################################
# Class SimulationUni


class SimulationUni(Simulation):
    """An abstract class for the simulation of univariate functional data.

    Parameters
    ----------
    M: int or numpy.ndarray
        Sampling points. If ``M`` is an integer, we use
        ``np.linspace(0, 1, M)`` as sampling points. Otherwise, we use the
        provided numpy.ndarray.

    """

    def __init__(self, N, M, name=None, n_features=1, n_clusters=1,
                 centers=None, cluster_std=None):
        """Initialize Simulation object."""
        if isinstance(M, int):
            M = np.linspace(0, 1, M)

        super().__init__(N, name, n_features, n_clusters, centers, cluster_std)
        self.M = M

    @abstractmethod
    def new(self, **kwargs):
        """Function to simulate observations."""
        super().new()


class Basis(SimulationUni):
    r"""A functional data object representing an orthogonal basis of functions.

    Parameters
    ----------
    name: str, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Denotes the basis of functions to use.
    norm: bool, default=False
        Should we normalize the basis function?

    Attributes
    ----------
    basis: UnivariateFunctionalData
        The basis of functions to use.

    Keyword Args
    ------------
    period: float, default = 2*numpy.pi
        The period of the circular functions for the Fourier basis.
    degree: int, default = 3
        Degree of the B-splines. The default gives cubic splines.
    knots: numpy.ndarray, (n_knots,)
        Specify the break points defining the B-splines.

    Notes
    -----

    The function are simulated using the Karhunen-Loève decomposition:

    .. math::
        X_i(t) = \mu(t) + \sum_{j = 1}^K c_{i,j}\phi_{j}(t), i = 1, ..., N


    """

    def __init__(self, N, M, name, n_features=1, n_clusters=1, centers=None,
                 cluster_std=None, norm=False, **kwargs):
        """Initialize Basis object."""
        if not isinstance(name, str):
            raise TypeError(f'{name:r} has to be `str`.')

        super().__init__(N, M, name, n_features, n_clusters,
                         centers, cluster_std)
        self.basis = simulate_basis(name, self.n_features, self.M,
                                    norm, **kwargs)
        self.norm = norm

    def new(self, **kwargs):
        """Function that simulates :math:`N` observations."""
        super().new()
        obs = np.matmul(self.coef, self.basis.values)
        return UnivariateFunctionalData(self.M, obs)


class BasisUFPCA(SimulationUni):
    r"""Class for the simulation of data using a UFPCA basis.

    Parameters
    ----------
    basis: UFPCA object
        Results of a univariate functional principal component analysis.

    Attributes
    ----------
    n_features: int
        Number of functions within the FPCA basis.
    data: UnivariateFunctionalData
        The simulated data :math:`X_i(t)`.

    Notes
    -----
    The function are simulated using the Karhunen-Loève decomposition:

    .. math::
        X_i(t) = \mu(t) + \sum_{j = 1}^M c_{i, j}\phi_{i, j}(t), i = , \dots, N

    """

    def __init__(self, N, basis, n_clusters=1, centers=None, cluster_std=None):
        """Initialize BasisFPCA object."""
        if isinstance(basis, UFPCA):
            n_features = len(basis.eigenvalues)
        else:
            raise TypeError('Wrong basis type!')

        super().__init__(N, None, 'ufpca', n_features, n_clusters,
                         centers, cluster_std)
        self.basis = basis

    def new(self, **kwargs):
        """Function that simulates :math:`N` observations."""
        super().new()
        return self.basis.inverse_transform(self.coef)


class Brownian(SimulationUni):
    """A functional data object representing a Brownian motion.

    Parameters
    ----------
    brownian_type: str, {'standard', 'geometric', 'fractional'}
        Type of brownian motion to simulate.

    """

    def __init__(self, N, M, brownian_type='standard'):
        """Initialize Brownian object."""
        super().__init__(N, M, brownian_type, n_features=1, n_clusters=1,
                         centers=None, cluster_std=None)

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
            The generated observations

        """
        param_dict = {k: kwargs.pop(k) for k in dict(kwargs)}

        # Simulate the N observations
        obs = []
        for _ in np.arange(self.N):
            obs.append(simulate_brownian(self.name, self.M, **param_dict))

        data = MultivariateFunctionalData(obs)
        return data.asUnivariateFunctionalData()

#############################################################################
# Class SimulationMulti


class SimulationMulti(Simulation):
    """An abstract class for the simulation of multivariate functional data.

    Parameters
    ----------
    M: list of int or list of numpy.ndarray
        Sampling points. If ``M`` is an integer, we use
        ``np.linspace(0, 1, M)`` as sampling points. Otherwise, we use the
        provided numpy.ndarray.

    """

    def __init__(self, N, M, name=None, n_features=1, n_clusters=1,
                 centers=None, cluster_std=None):
        """Initialize SimulationMulti object."""
        if not isinstance(M, list):
            raise TypeError('M have to be a list!')

        super().__init__(N, name, n_features, n_clusters, centers, cluster_std)
        self.M = []
        for m in M:
            if isinstance(m, int):
                self.M.append(np.linspace(0, 1, m))
            elif isinstance(m, np.ndarray):
                self.M.append(m)
            else:
                raise TypeError(f"""An element of M have a wrong type!\
                                Have to be int or numpy.ndarray and not\
                                {type(m)}.""")

    @abstractmethod
    def new(self, **kwargs):
        """Function to simulate observations."""
        super().new()


class BasisMulti(SimulationMulti):
    r"""A multivariate functional data object representing a basis.

    Parameters
    ----------
    name: list of str, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Denotes the basis of functions to use.
    norm: bool, default=False
        Should we normalize the basis function?

    Attributes
    ----------
    basis: MultivariateFunctionalData
        A MultivariateFunctionalData object that contains a list of basis of
        functions to use.
    data: MultivariateFunctionalData
        The simulated data :math:`X_i(t)`.

    Keyword Args
    ------------
    period: float, default = 2*numpy.pi
        The period of the circular functions for the Fourier basis.
    degree: int, default = 3
        Degree of the B-splines. The default gives cubic splines.
    knots: numpy.ndarray, (n_knots,)
        Specify the break points defining the B-splines.

    Notes
    -----

    The function are simulated using the Karhunen-Loève decomposition:

    .. math::
        X_i(t) = \mu(t) + \sum_{j = 1}^K c_{i,j}\phi_{j}(t), i = 1, ..., N


    """

    def __init__(self, N, M, name=None, n_features=1, n_clusters=1,
                 centers=None, cluster_std=None, norm=False, **kwargs):
        """Initialize BasisMulti object."""
        super().__init__(N, M, name, n_features, n_clusters,
                         centers, cluster_std)

        # Define the basis
        basis = [simulate_basis(n, self.n_features, m, norm, **kwargs)
                 for n, m in zip(self.name, self.M)]
        self.basis = MultivariateFunctionalData(basis)

    def new(self, **kwargs):
        """Function that simulates :math:`N` observations."""
        super().new()
        obs_uni = []
        for basis in self.basis:
            obs = np.matmul(self.coef, basis.values)
            obs_uni.append(UnivariateFunctionalData(basis.argvals, obs))
        return MultivariateFunctionalData(obs_uni)


class BasisMFPCA(SimulationMulti):
    r"""Class for the simulation of data using a MFPCA basis.

    Parameters
    ----------
    basis: MFPCA object
        Results of a multivariate functional principal component analysis.

    Attributes
    ----------
    n_features: int
        Number of functions within the MFPCA basis.
    data: MultivariateFunctionalData
        The simulated data :math:`X_i(t)`.

    Notes
    -----
    The function are simulated using the Karhunen-Loève decomposition:

    .. math::
        X_i(t) = \mu(t) + \sum_{j = 1}^M c_{i, j}\phi_{i, j}(t), i = , \dots, N

    The number of sampling points :math:`M` is not used for the simulation of
    data using FPCA. The simulated curves will have the same length
    than the eigenfunctions.

    """

    def __init__(self, N, basis, n_clusters=1, centers=None, cluster_std=None):
        """Initialize BasisMFPCA object."""
        if isinstance(basis, MFPCA):
            n_features = len(basis.eigenvaluesCovariance_)
        else:
            raise TypeError('Wrong basis type!')

        # TODO: Modify the class MFPCA to have argvals.
        M = [ufpca.argvals[0] for ufpca in basis.ufpca_]

        super().__init__(N, M, 'mfpca', n_features, n_clusters,
                         centers, cluster_std)
        self.basis = basis

    def new(self, **kwargs):
        """Function that simulates :math:`N` observations."""
        super().new()
        return self.basis.inverse_transform(self.coef)
