#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""Simulation functions.

This module is used to define an abstract Simulation class. We may simulate
different data from a linear combination of basis functions or multiple
realizations of diverse Brownian motion.
"""
import numpy as np

from abc import ABC, abstractmethod


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


def geometric_brownian(argvals=None, x0=1.0, mu=0.0, sigma=1.0):
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


def fractional_brownian(argvals=None, hurst=0.5):
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


def simulate_brownian(name, argvals=None, **kwargs):
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
    hust: float, default=0.5
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
# Definition of the decreasing of the eigenvalues

def eigenvalues_linear(n=3):
    """Generate linear decreasing eigenvalues.

    Parameters
    ----------
    n: int, default=3
        Number of eigenvalues to generates.

    Returns
    -------
    values: numpy.ndarray, shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> eigenvalues_linear(n=3)
    array([1.0, 0.6666666666666666, 0.3333333333333333])

    """
    return np.array([(n - m + 1) / n for m in np.linspace(1, n, n)])


def eigenvalues_exponential(n=3):
    """Generate exponential decreasing eigenvalues.

    Parameters
    ----------
    n: int, default=3
        Number of eigenvalues to generates.

    Returns
    -------
    values: numpy.ndarray, shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> eigenvalues_exponential(n=3)
    array([0.36787944117144233, 0.22313016014842982, 0.1353352832366127])

    """
    return [np.exp(-(m + 1) / 2) for m in np.linspace(1, n, n)]


def eigenvalues_wiener(n=3):
    """Generate eigenvalues from a Wiener process.

    Parameters
    ----------
    n: int, default=3
        Number of eigenvalues to generates.

    Returns
    -------
    values: numpy.ndarray, shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> eigenvalues_wiener(n=3)
    array([0.4052847345693511, 0.04503163717437235, 0.016211389382774045])

    """
    return np.array([np.power((np.pi / 2) * (2 * m - 1), -2)
                     for m in np.linspace(1, n, n)])


def simulate_eigenvalues(name, n=3):
    """Redirect to the right simulation eigenvalues function.

    Parameters
    ----------
    eigenvalues_name: str, {'linear', 'exponential', 'wiener'}
        Name of the eigenvalues generation process to use.
    n: int, default=3
        Number of eigenvalues to generates.

    Returns
    -------
    eigenvalues: numpy.ndarray, shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> simulate_eigenvalues('linear', n=3)
    array([1.0, 0.6666666666666666, 0.3333333333333333])

    """
    if name == 'linear':
        return eigenvalues_linear(n)
    elif name == 'exponential':
        return eigenvalues_exponential(n)
    elif name == 'wiener':
        return eigenvalues_wiener(n)
    else:
        raise NotImplementedError('Eigenvalues not implemented!')


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
