#!/usr/bin/env python
# -*-coding:utf8 -*

"""Simulation functions.

This module is used to define an abstract Simulation class. We may simulate
different data from a linear combination of basis functions or multiple
realizations of diverse Brownian motion.
"""
import numpy as np

from typing import Optional, Union

from sklearn.datasets import make_blobs

from ..representation.functional_data import DenseFunctionalData
from ..representation.basis import Basis
from .simulation import Simulation


#############################################################################
# Definition of the decreasing of the eigenvalues

def eigenvalues_linear(
    n: int = 3
) -> np.ndarray:
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


def eigenvalues_exponential(
    n: int = 3
) -> np.ndarray:
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


def eigenvalues_wiener(
    n: int = 3
) -> np.ndarray:
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


def simulate_eigenvalues(
    name: str,
    n: int = 3
) -> np.ndarray:
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
def make_coef(
    n_obs: int,
    n_features: int,
    centers: np.ndarray,
    cluster_std: np.ndarray
) -> np.ndarray:
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


def initialize_centers(
    n_features: int,
    n_clusters: int,
    centers: Optional[np.ndarray] = None
) -> np.ndarray:
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


def initialize_cluster_std(
    n_features: int,
    n_clusters: int,
    cluster_std: Union[str, np.ndarray, None] = None
) -> np.ndarray:
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
# Definition of the KarhunenLoeve

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

    def __init__(
        self,
        name: str,
        basis: Optional[DenseFunctionalData] = None,
        n_functions: int = 5,
        dimension: str = '1D',
        **kwargs_basis
    ) -> None:
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

    def new(
        self,
        n_obs: int,
        argvals: Optional[np.ndarray] = None,
        **kwargs
    ):
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
