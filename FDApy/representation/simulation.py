#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""Simulation functions.

This module is used to define an abstract Simulation class. We may simulate
different data from a linear combination of basis functions or multiple
realizations of diverse Brownian motion.
"""
import inspect

import numpy as np

from abc import ABC, abstractmethod

from sklearn.datasets import make_blobs

from .functional_data import DenseFunctionalData, IrregularFunctionalData
from .basis import Basis


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
    name: str, {'linear', 'exponential', 'wiener'}
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
def make_coef(n_obs, n_features, centers, cluster_std):
    """Simulate a set of coefficients for the Karhunen-Loève decomposition.

    Parameters
    ----------
    n_obs: int
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
    coef: numpy.ndarray, (n_obs, n_features)
        Array of generated coefficients.
    labels: numpy.ndarray, (n_obs,)
        The integer labels for cluster membership of each observations.

    Notes
    -----
    The function :func:`sklearn.datasets.make_blobs` does not allow different
    standard deviations for the different features. It only permits to change
    the standard deviations between clusters. To bypass that, we loop through
    the ``n_features``.

    Examples
    --------
    >>> centers = np.array([[1, 2, 3], [0, 4, 6]])
    >>> cluster_std = cluster_std = np.array([[0.5, 0.25, 1], [1, 0.1, 0.5]])
    >>> make_coef(100, 2, centers, cluster_std)

    """
    coef = np.zeros((n_obs, n_features))
    for idx in np.arange(n_features):
        x, labels = make_blobs(n_samples=n_obs, n_features=1,
                               centers=centers[idx, :].reshape(-1, 1),
                               cluster_std=cluster_std[idx, :],
                               shuffle=False)
        coef[:, idx] = x.squeeze()
    return coef, labels


def initialize_centers(n_features, n_clusters, centers=None):
    """Initialize the centers of the clusters.

    Parameters
    ----------
    n_features: int
        Number of features to simulate.
    n_clusters: int
        Number of clusters to simulate.
    centers: numpy.ndarray, shape=(n_features, n_clusters), default=None
        The centers of each cluster per feature.

    Returns
    -------
    centers: np.ndarray, shape=(n_features, n_clusters)
        An array with good shape for the initialization of the centers of the
        cluster.

    """
    return np.zeros((n_features, n_clusters)) if centers is None else centers


def initialize_cluster_std(n_features, n_clusters, cluster_std=None):
    """Initialize the standard deviation of the clusters.

    Parameters
    ----------
    n_features: int
        Number of features to simulate.
    n_clusters: int
        Number of clusters to simulate.
    cluster_std: str or np.ndarray or None
        The standard deviation of each cluster per feature.

    Returns
    -------
    cluster_std: np.ndarray, shape=(n_features, n_clusters)
        An array with good shape for the initialization of the standard
        deviation of the cluster.

    """
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
    name: str
        Name of the simulation

    Arguments
    ---------
    data: DenseFunctionalData
        An object that represents the simulated data.
    noisy_data: DenseFunctionalData
        An object that represents a noisy version of the simulated data.
    sparse_data: IrregularFunctionalData
        An object that represents a sparse version of the simulated data.

    """

    def _check_data(self):
        """Check if self has the attribut data."""
        if not hasattr(self, 'data'):
            raise ValueError('No data have been found in the simulation.'
                             ' Please run new() before add_noise().')

    def __init__(self, name):
        """Initialize Simulation object."""
        super().__init__()
        self.name = name

    @property
    def name(self):
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @abstractmethod
    def new(self, n_obs, argvals=None, **kwargs):
        """Simulate a new set of data."""
        pass

    def add_noise(self, var_noise=1):
        r"""Add noise to the data.

        Parameters
        ----------
        var_noise: float or Callable, default=1
            Variance of the noise to add. May be a callable for heteroscedastic
            noise.

        Notes
        -----
        Model used to generate the data:

        .. math::
            Z(t) = f(t) + \sigma(f(t))\epsilon

        """
        self._check_data()

        shape_simu = self.data.n_obs, *tuple(self.data.n_points.values())
        noisy_data = np.random.normal(0, 1, shape_simu)

        if inspect.isfunction(var_noise):
            var_noise = var_noise(self.data.values)

        std_noise = np.sqrt(var_noise)
        noisy_data = self.data.values + np.multiply(std_noise, noisy_data)
        self.noisy_data = DenseFunctionalData(self.data.argvals, noisy_data)

    def sparsify(self, percentage=0.9, epsilon=0.05):
        """Sparsify the simulated data.

        Parameters
        ----------
        percentage: float, default = 0.9
            Percentage of data to keep.
        epsilon: float, default = 0.05
            Uncertainty on the percentage to keep.

        """
        if self.data.n_dim > 1:
            raise ValueError("The sparsification is not implemented for data"
                             "with dimension larger than 1.")
        self._check_data()

        argvals = {}
        values = {}
        for idx, obs in enumerate(self.data):
            s = obs.values.size
            p = np.random.uniform(max(0, percentage - epsilon),
                                  min(1, percentage + epsilon))
            indices = np.sort(np.random.choice(np.arange(0, s),
                                               size=int(p * s),
                                               replace=False))
            argvals[idx] = obs.argvals['input_dim_0'][indices]
            values[idx] = obs.values[0][indices]
        self.sparse_data = IrregularFunctionalData({'input_dim_0': argvals},
                                                   values)


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

    def __init__(self, name):
        """Initialize Brownian object."""
        super().__init__(name)

    def new(self, n_obs, argvals=None, **kwargs):
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


class KarhunenLoeve(Simulation):
    r"""Class for the simulation of data using a basis of function.

    Parameters
    ----------
    name: str, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Type of basis to use.
    basis: DenseFunctionalData or None
        A basis of functions as a DenseFunctionalData object. Used to have a
        user-defined basis of function.
    n_functions: int, default=5
        Number of functions to use to generate the basis.
    dimension: str, ('1D', '2D'), default='1D'
        Dimension of the basis to generate.

    Arguments
    ---------
    data: DenseFunctionalData
        An object that represents the simulated data.
    noisy_data: DenseFunctionalData
        An object that represents a noisy version of the simulated data.
    sparse_data: IrregularFunctionalData
        An object that represents a sparse version of the simulated data.
    labels: np.ndarray, shape=(n_obs,)
        Data labels

    Notes
    -----
    The function are simulated using the Karhunen-Loève decomposition:

    .. math::
        X_i(t) = \mu(t) + \sum_{j = 1}^M c_{i, j}\phi_{i, j}(t),
        i = 1, \dots, N

    """

    def __init__(self, name, basis=None, n_functions=5, dimension='1D',
                 **kwargs_basis):
        """Initialize Basis object."""
        if (name is not None) and (basis is not None):
            raise ValueError('Name or basis have to be None. Do not know'
                             ' which basis to use.')
        if not isinstance(basis, DenseFunctionalData) and (basis is not None):
            raise ValueError('Basis have to be an instance of'
                             ' DenseFunctionalData')
        if (name is None) and isinstance(basis, DenseFunctionalData):
            name = 'user-defined'
        if isinstance(name, str) and (basis is None):
            basis = Basis(name, n_functions, dimension, **kwargs_basis)

        super().__init__(name)
        self.basis = basis

    def new(self, n_obs, argvals=None, **kwargs):
        """Simulate ``n_obs`` realizations from a basis of function.

        Parameters
        ----------
        n_obs: int
            Number of observations to simulate.
        argvals: None
            Not used in this context. We will use the `argvals` from the Basis
            object as `argvals` of the simulation.

        Keyword Args
        ------------
        n_clusters: int, default=1
            Number of clusters to generate
        centers: numpy.ndarray, shape=(n_features, n_clusters)
            The centers of the clusters to generate. The ``n_features``
            correspond to the number of functions within the basis.
        cluster_std: np.ndarray, shape=(n_features, n_clusters)
            The standard deviation of the clusters to generate. The
            ``n_features`` correspond to the number of functions within the
            basis.

        """
        n_features = self.basis.n_obs
        n_clusters = kwargs.get('n_clusters', 1)
        centers = initialize_centers(n_features, n_clusters,
                                     kwargs.get('centers', None))
        cluster_std = initialize_cluster_std(n_features, n_clusters,
                                             kwargs.get('cluster_std', None))
        coef, labels = make_coef(n_obs, n_features, centers, cluster_std)

        if self.basis.dimension == '1D':
            values = np.matmul(coef, self.basis.values)
        elif self.basis.dimension == '2D':
            values = np.tensordot(coef, self.basis.values, axes=1)
        else:
            raise ValueError("Something went wrong with the basis dimension.")

        self.labels = labels
        self.data = DenseFunctionalData(self.basis.argvals, values)
