#!/usr/bin/env python
# -*-coding:utf8 -*

"""Karhunen-Loève decomposition
-------------------------------

"""
import numpy as np
import numpy.typing as npt

from collections import namedtuple
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from ..representation.functional_data import (
    DenseFunctionalData, MultivariateFunctionalData
)
from ..representation.basis import Basis
from .simulation import Simulation

#############################################################################
# Class Data
Data = namedtuple('Data', ['labels', 'eigenvalues', 'data'])


#############################################################################
# Definition of the decreasing of the eigenvalues

def _eigenvalues_linear(
    n: int = 3
) -> npt.NDArray:
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
    >>> _eigenvalues_linear(n=3)
    array([1.0, 0.6666666666666666, 0.3333333333333333])

    """
    return np.array([(n - m + 1) / n for m in np.arange(1, n + 1)])


def _eigenvalues_exponential(
    n: int = 3
) -> npt.NDArray:
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
    >>> _eigenvalues_exponential(n=3)
    array([0.36787944117144233, 0.22313016014842982, 0.1353352832366127])

    """
    return np.array([np.exp(-(m + 1) / 2) for m in np.arange(1, n + 1)])


def _eigenvalues_wiener(
    n: int = 3
) -> npt.NDArray:
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
    >>> _eigenvalues_wiener(n=3)
    array([0.4052847345693511, 0.04503163717437235, 0.016211389382774045])

    """
    return np.array(
        [np.power((np.pi / 2) * (2 * m - 1), -2) for m in np.arange(1, n + 1)]
    )


def _simulate_eigenvalues(
    name: str,
    n: int = 3
) -> npt.NDArray:
    """Generate eigenvalues.

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
    >>> _simulate_eigenvalues('linear', n=3)
    array([1.0, 0.6666666666666666, 0.3333333333333333])

    """
    if name == 'linear':
        return _eigenvalues_linear(n)
    elif name == 'exponential':
        return _eigenvalues_exponential(n)
    elif name == 'wiener':
        return _eigenvalues_wiener(n)
    else:
        raise NotImplementedError(
            'Eigenvalues generation method is not implemented!'
        )


#############################################################################
# Definition of clusters
def _make_coef(
    n_obs: int,
    n_features: int,
    centers: npt.NDArray,
    cluster_std: npt.NDArray,
    rnorm: Callable = np.random.multivariate_normal
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Simulate a set of coefficients for the Karhunen-Loève decomposition.

    Parameters
    ----------
    n_obs: int
        Number of observations to simulate.
    n_features: int
        Number of features to simulate.
    centers: numpy.ndarray, shape=(n_features, n_clusters)
        The centers of the clusters to generate. The ``n_features`` parameter
        corresponds to the number of functions within the basis.
    cluster_std: numpy.ndarray, shape=(n_features, n_clusters)
        The standard deviation of the clusters to generate. The
        ``n_features`` parameter corresponds to the number of functions within
        the basis.

    Returns
    -------
    coef: numpy.ndarray, shape=(n_obs, n_features)
        Array of generated coefficients.
    labels: numpy.ndarray, shape=(n_obs,)
        The integer labels for cluster membership of each observations.

    Notes
    -----
    This function is inspired by :func:`sklearn.datasets.make_blobs`, which
    has been reimplement to allow different standard deviations for the
    different features and between clusters.

    As this function is used for the simulation of functional data using the
    Karhunen-Loève decomposition, we did not include correlation between the
    generated coefficients.

    Examples
    --------
    >>> centers = np.array([[1, 2, 3], [0, 4, 6]])
    >>> cluster_std = np.array([[0.5, 0.25, 1], [1, 0.1, 0.5]])
    >>> _make_coef(100, 2, centers, cluster_std)

    """
    n_centers = centers.shape[1]

    n_obs_per_center = [int(n_obs // n_centers)] * n_centers
    for idx in range(n_obs % n_centers):
        n_obs_per_center[idx] += 1
    cum_sum_n_obs = np.cumsum(n_obs_per_center)

    coefs = np.empty(shape=(n_obs, n_features), dtype=np.float64)
    labels = np.empty(shape=(n_obs,), dtype=int)
    for idx, n_obs in enumerate(n_obs_per_center):
        start_idx = cum_sum_n_obs[idx - 1] if idx > 0 else 0
        end_idx = cum_sum_n_obs[idx]

        coefs[start_idx:end_idx, :] = rnorm(
            mean=centers[:, idx],
            cov=np.diag(cluster_std[:, idx]),
            size=n_obs
        )
        labels[start_idx:end_idx] = idx
    return coefs, labels


def _initialize_centers(
    n_features: int,
    n_clusters: int,
    centers: Optional[npt.NDArray] = None
) -> npt.NDArray:
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
    centers: numpy.ndarray, shape=(n_features, n_clusters)
        An array with good shape for the initialization of the centers of the
        cluster.

    """
    return np.zeros((n_features, n_clusters)) if centers is None else centers


def _initialize_cluster_std(
    n_features: int,
    n_clusters: int,
    cluster_std: Union[str, npt.NDArray, None] = None
) -> npt.NDArray:
    """Initialize the standard deviation of the clusters.

    Parameters
    ----------
    n_features: int
        Number of features to simulate.
    n_clusters: int
        Number of clusters to simulate.
    cluster_std: str or np.ndarray or None
        The standard deviation of each cluster per feature. If the parameter
        is given as a string, it has to be one of {`linear`, `exponential`,
        `wiener`}. If `None`, the standard deviation of each cluster per
        feature is set to :math:`1`.

    Returns
    -------
    cluster_std: numpy.ndarray, shape=(n_features, n_clusters)
        An array with good shape for the initialization of the standard
        deviation of the cluster.

    """
    if isinstance(cluster_std, str):
        eigenvalues = _simulate_eigenvalues(cluster_std, n_features)
        eigenvalues = np.repeat(eigenvalues, n_clusters)
        return eigenvalues.reshape((n_features, n_clusters))
    elif cluster_std is None:
        return np.ones((n_features, n_clusters))
    else:
        return cluster_std


#############################################################################
# Generation of univariate functional data
def _generate_univariate_data(
    basis: DenseFunctionalData,
    n_obs: int,
    n_clusters: int = 1,
    centers: Optional[npt.NDArray] = None,
    cluster_std: Union[str, npt.NDArray, None] = None,
    rnorm: Callable = np.random.multivariate_normal,
) -> Data:
    r"""Generate univariate functional data.

    This function can be used to simulate univariate functional data
    :math:`X_1, \dots, X_N` based on a truncated Karhunen-Loève decomposition:

    .. math::
        X_i(t) = \sum_{K = 1}^K c_{i, k}\phi_{k}(t), i = 1, \dots, N,

    on one- or higher-dimensional domains. The eigenfunctions
    :math:`\phi_{k}(t)` could be generated using different basis functions or
    be user-defined. The scores :math:`c_{i, k}` are simulated independently
    from a normal distribution with zero mean and decreasing variance. For
    higher-dimensional domains, the eigenfunctions are constructed as tensors
    of marginal orthonormal function systems.

    Parameters
    ----------
    basis: DenseFunctionalData
        Basis of functions to use for the generation of the data.
    n_obs: int
        Number of observations to simulate.
    n_clusters: int, default=1
        Number of clusters to generate.
    centers: numpy.ndarray, shape=(n_features, n_clusters)
        The centers of the clusters to generate. The ``n_features``
        correspond to the number of functions within the basis.
    cluster_std: numpy.ndarray, shape=(n_features, n_clusters)
        The standard deviation of the clusters to generate. The
        ``n_features`` correspond to the number of functions within the
        basis.
    rnorm: Callable, default=np.random.multivariate_normal
        Random data generator.

    Returns
    -------
    simu: Data
        An element of the class Data with the labels, eigenvalues and simulated
        data.

    """
    # Initialize parameters
    n_features = basis.n_obs

    centers = _initialize_centers(n_features, n_clusters, centers)
    cluster_std = _initialize_cluster_std(n_features, n_clusters, cluster_std)

    # Generate data
    coef, labels = _make_coef(
        n_obs, n_features, centers, cluster_std, rnorm
    )

    if basis.dimension == '1D':
        values = np.matmul(coef, basis.values)
    elif basis.dimension == '2D':
        values = np.tensordot(coef, basis.values, axes=1)
    else:
        raise ValueError("Something went wrong with the basis dimension.")
    return Data(
        labels=labels,
        eigenvalues=cluster_std[:, 0],
        data=DenseFunctionalData(basis.argvals, values)
    )


#############################################################################
# Definition of the KarhunenLoeve simulation

class KarhunenLoeve(Simulation):
    r"""Class that defines simulation based on Karhunen-Loève decomposition.

    This class is used to simulate functional data
    :math:`X_1, \dots, X_N` based on a truncated Karhunen-Loève decomposition:

    .. math::
        X_i(t) = \sum_{K = 1}^K c_{i, k}\phi_{k}(t), i = 1, \dots, N,

    on one- or higher-dimensional domains. The eigenfunctions
    :math:`\phi_{k}(t)` could be generated using different basis functions or
    be user-defined. The scores :math:`c_{i, k}` are simulated independently
    from a normal distribution with zero mean and decreasing variance. For
    higher-dimensional domains, the eigenfunctions are constructed as tensors
    of marginal orthonormal function systems.

    Parameters
    ----------
    name: str or list of str, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Type of basis to use.
    n_functions: int or list of int, default=5
        Number of functions to use to generate the basis.
    dimension: str or list of str, {'1D', '2D'}, default='1D'
        Dimension of the basis to generate.
    argvals: dict
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. The shape of the :math:`j`th
        dimension is :math:`(m_j,)` for :math:`0 \leq j \leq p`.
    basis: DenseFunctionalData, default=None
        Basis of functions as a DenseFunctionalData object. Used to have a
        user-defined basis of function.
    random_state: int, default=None
        A seed to initialize the random number generator.

    Attributes
    ----------
    data: DenseFunctionalData or MultivariateFunctionalData
        An object that represents the simulated data.
    noisy_data: DenseFunctionalData or MultivariateFunctionalData
        An object that represents a noisy version of the simulated data.
    sparse_data: IrregularFunctionalData or MultivariateFunctionalData
        An object that represents a sparse version of the simulated data.
    labels: numpy.ndarray, shape=(n_obs,)
        The integer labels for cluster membership of each sample.
    basis: DenseFunctionalData of list of DenseFunctionalData
        The eigenfunctions used to simulate the data.
    eigenvalues: numpy.ndarray, shape=(n_functions,)
        The eigenvalues used to simulate the data.

    """

    def __init__(
        self,
        basis_name: Union[str, Sequence[str]],
        n_functions: Union[int, Sequence[int]] = 5,
        dimension: Union[str, Sequence[str]] = '1D',
        argvals: Optional[Dict[str, npt.NDArray]] = None,
        basis: Optional[
            Union[DenseFunctionalData, Sequence[DenseFunctionalData]]
        ] = None,
        random_state: Optional[int] = None,
        **kwargs_basis: Any
    ) -> None:
        """Initialize KarhunenLoeve object."""
        if (basis_name is not None) and (basis is not None):
            raise ValueError(
                'Name or basis have to be None. Do not know'
                ' which basis to use.'
            )
        if (
            not isinstance(basis, (DenseFunctionalData, list)) and
            (basis is not None)
        ):
            raise ValueError(
                'Basis have to be an instance of DenseFunctionalData or a list'
                ' of DenseFunctionalData'
            )
        if (basis_name is None) and isinstance(basis, DenseFunctionalData):
            basis_name = ['user-defined']
            basis = [basis]
        if (basis_name is None) and isinstance(basis, list):
            basis_name = len(basis) * ['user_defined']

        if isinstance(basis_name, str):
            basis_name = [basis_name]
            n_functions = [n_functions]
            dimension = [dimension]
        if isinstance(basis_name, list) and isinstance(n_functions, int):
            n_functions = len(basis_name) * [n_functions]
        if isinstance(basis_name, list) and isinstance(dimension, str):
            dimension = len(basis_name) * [dimension]

        if basis is None:
            basis = [
                Basis(
                    name=name,
                    n_functions=n_func,
                    dimension=dim,
                    argvals=argvals,
                    **kwargs_basis
                ) for name, n_func, dim in zip(
                    basis_name, n_functions, dimension
                )
            ]

        super().__init__(basis_name, random_state)
        self.basis = basis

    def new(
        self,
        n_obs: int,
        n_clusters: int = 1,
        argvals: Optional[Dict[str, npt.NDArray]] = None,
        **kwargs
    ):
        """Simulate realizations from Karhunen-Loève decomposition.

        This function generates ``n_obs`` realizations of a Gaussian process
        using the Karhunen-Loève decomposition on a common grid ``argvals``.

        Parameters
        ----------
        n_obs: int
            Number of observations to simulate.
        n_clusters: int, default=1
            Number of clusters to generate.
        argvals: None
            Not used in this context. We will use the ``argvals`` from the
            :mod:`Basis` object as ``argvals`` of the simulation.

        Keyword Args
        ------------
        centers: numpy.ndarray, shape=(n_features, n_clusters)
            The centers of the clusters to generate. The ``n_features``
            correspond to the number of functions within the basis.
        cluster_std: numpy.ndarray, shape=(n_features, n_clusters)
            The standard deviation of the clusters to generate. The
            ``n_features`` correspond to the number of functions within the
            basis.

        """
        if self.random_state is None:
            rnorm = np.random.multivariate_normal
        else:
            rnorm = self.random_state.multivariate_normal

        # Get parameters
        centers = kwargs.get('centers', len(self.basis) * [None])
        clusters_std = kwargs.get('cluster_std', len(self.basis) * [None])

        if isinstance(centers, np.ndarray) and len(self.basis) == 1:
            centers = [centers]
        if isinstance(clusters_std, str):
            clusters_std = len(self.basis) * [clusters_std]

        # Generate data
        simus_univariate = [
            _generate_univariate_data(
                basis=basis,
                n_obs=n_obs,
                n_clusters=n_clusters,
                rnorm=rnorm,
                centers=center,
                cluster_std=cluster_std
            ) for basis, center, cluster_std in zip(
                self.basis, centers, clusters_std
            )
        ]

        data_univariate = [simu.data for simu in simus_univariate]
        if len(data_univariate) > 1:
            self.data = MultivariateFunctionalData(data_univariate)
        else:
            self.data = data_univariate[0]
        self.labels = simus_univariate[0].labels
        self.eigenvalues = [simu.eigenvalues for simu in simus_univariate]
