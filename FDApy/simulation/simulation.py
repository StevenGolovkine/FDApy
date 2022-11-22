#!/usr/bin/env python
# -*-coding:utf8 -*

"""Simulation functions.

This module is used to define an abstract Simulation class. We may simulate
different data from a linear combination of basis functions or multiple
realizations of diverse Brownian motion.
"""
import inspect

import numpy as np

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from sklearn.datasets import make_blobs

from ..representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData
)


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

    def _check_data(self) -> None:
        """Check if self has the attribut data."""
        if not hasattr(self, 'data'):
            raise ValueError('No data have been found in the simulation.'
                             ' Please run new() before add_noise().')

    def __init__(self, name: str) -> None:
        """Initialize Simulation object."""
        super().__init__()
        self.name = name

    @property
    def name(self) -> str:
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        self._name = new_name

    @abstractmethod
    def new(
        self,
        n_obs: int,
        argvals: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """Simulate a new set of data."""
        pass

    def add_noise(
        self,
        var_noise: Union[float, Callable[[np.ndarray], np.ndarray]] = 1.0
    ) -> None:
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

    def sparsify(
        self,
        percentage: float = 0.9,
        epsilon: float = 0.05
    ) -> None:
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
