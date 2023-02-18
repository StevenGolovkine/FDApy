#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Karhunen-Loève decomposition
----------------------------

This module defines simulation based on the Karhunen-Loève decomposition.

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
    npt.NDArray, shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> _eigenvalues_linear(n=3)
    array([1.0, 0.6666666666666666, 0.3333333333333333])

    """
    return (n - np.arange(1, n + 1) + 1) / n


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
    npt.NDArray, shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> _eigenvalues_exponential(n=3)
    array([0.36787944117144233, 0.22313016014842982, 0.1353352832366127])

    """
    return np.exp(-(np.arange(1, n + 1) + 1) / 2)


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
    npt.NDArray, shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> _eigenvalues_wiener(n=3)
    array([0.4052847345693511, 0.04503163717437235, 0.016211389382774045])

    """
    return np.power((np.pi / 2) * (2 * np.arange(1, n + 1) - 1), -2)


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
        Number of eigenvalues to generates. Should be strictly positive.

    Returns
    -------
    npt.NDArray, shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> _simulate_eigenvalues('linear', n=3)
    array([1.0, 0.6666666666666666, 0.3333333333333333])

    """
    if n < 1:
        raise ValueError(f'Parameter has to be strictly positive (now {n})')
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
    centers: npt.NDArray, shape=(n_features, n_clusters)
        The centers of the clusters to generate. The ``n_features`` parameter
        corresponds to the number of functions within the basis.
    cluster_std: npt.NDArray, shape=(n_features, n_clusters)
        The standard deviation of the clusters to generate. The
        ``n_features`` parameter corresponds to the number of functions within
        the basis.
    rnorm: Callable, default=np.random.multivariate_normal
        Method used to generate Gaussian coefficients.

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray]
        Returns a tuple containing the coefficients, as `npt.NDArray` of shape
        `(n_obs, n_features)`, and the labels, as `npt.NDArray` of shape
        `(n_obs,)`. The labels refer to the cluster membership of each
        observation.

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

    n_obs_per_center = np.ones(n_centers, dtype=int) * (n_obs // n_centers)
    n_obs_per_center[:n_obs % n_centers] += 1
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
    npt.NDArray, shape=(n_features, n_clusters)
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
    npt.NDArray, shape=(n_features, n_clusters)
        An array with good shape for the initialization of the standard
        deviation of the cluster.

    """
    if isinstance(cluster_std, str):
        eigenvalues = _simulate_eigenvalues(cluster_std, n_features)
        return np.full((n_clusters, n_features), eigenvalues).T
    elif cluster_std is None:
        return np.ones((n_features, n_clusters))
    else:
        return cluster_std


#############################################################################
# Generation of univariate functional data
def _compute_data(
    basis: Basis,
    coefficients: npt.NDArray
) -> DenseFunctionalData:
    r"""Compute functional data.

    This function can be used to compute functional data
    :math:`X_1, \dots, X_N` based on a truncated Karhunen-Loève decomposition:

    .. math::
        X_i(t) = \sum_{K = 1}^K c_{i, k}\phi_{k}(t), i = 1, \dots, N,

    on one- or higher-dimensional domains. For
    higher-dimensional domains, the eigenfunctions are constructed as tensors
    of marginal orthonormal function systems.

    Parameters
    ----------
    basis: Basis
        Basis of functions to use for the generation of the data.
    coefficients: npt.NDArray, shape=(n_obs, n_features)
        A set of coefficients of shape the number of observations times the
        number of elements in the basis.

    Returns
    -------
    DenseFunctionalData
        Generated data as a DenseFunctionalData object.

    """
    
    if basis.dimension == '1D':
        values = np.matmul(coefficients, basis.values)
    elif basis.dimension == '2D':
        values = np.tensordot(coefficients, basis.values, axes=1)
    else:
        raise ValueError(
            f"The basis dimension {basis.dimension} has to be 1D or 2D"
        )
    return DenseFunctionalData(basis.argvals, values)


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

    Notes
    -----
    In the case of multivariate functional data, :math:`X_i` and
    :math:`\phi_{k}` are vectors and according to the multivariate
    Karhunen-Loève theorem (see, e.g, [HG]_), the coefficients do not depend
    on the component :math:`p`.
    
    If the basis is user-defined, the object has to be an element of the class
    Basis and not just DenseFunctionalData or MultivariateFunctionalData.

    Parameters
    ----------
    name: Sequence[str], {'legendre', 'wiener', 'fourier', 'bsplines'}
        Name of the basis to use. For multivariate functional data, this is a
        list of `str` of length `n_features`.
    n_functions: Sequence[int], default=5
        Number of functions to use to generate the basis. For multivariate
        functional data, this is a list of `int` of length `n_features`.
    dimension: Sequence[str], {'1D', '2D'}, default='1D'
        Dimension of the basis to generate. For multivariate functional data,
        this is a list of `str` of length `n_features`.
    argvals: Dict[str, npt.NDArray]
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. The shape of the :math:`j`th
        dimension is :math:`(m_j,)` for :math:`0 \leq j \leq p`.
    basis: Union[Basis, Sequence[Basis]], default=None
        Basis of functions as a Basis object. Used to have a user-defined basis
        of function.
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

    References
    ----------
    .. [HG] Happ C. & Greven S. (2018) Multivariate Functional Principal
        Component Analysis for Data Observed on Different (Dimensional)
        Domains, Journal of the American Statistical Association, 113:522,
        649-659, DOI: 10.1080/01621459.2016.1273115

    """

    @staticmethod
    def _check_basis_none(
        basis_name: Union[str, Sequence[str]],
        basis: Optional[Basis]
    ) -> None:
        """Check if `basis_name` of `basis` is None.
        
        Parameters
        ----------
        basis_name: Union[str, Sequence[str]]
            A str or a sequence of str indicating the name or names of the basis.
        basis: Basis
            A Basis instance.

        Raises
        ------
        ValueError
            If both `basis_name` and `basis` are not None.

        """
        if (basis_name is not None) and (basis is not None):
            raise ValueError(
                'One of the arguments `basis_name` or `basis` have to be None.'
                ' Do not know which basis to use.'
            )

    @staticmethod
    def _check_basis_type(
        basis: Optional[Basis]
    ) -> None:
        """Check if `basis` has the right type.

        Parameters
        ----------
        basis: Optional[Basis]
            A Basis instance or None.

        Raises
        ------
        ValueError
            If the basis argument is not None and is not an instance of the
            Basis class.

        """
        if not isinstance(basis, (Basis, list)) and (basis is not None):
            raise ValueError(
                'The basis argument has to be an instance of Basis.'
            )

    @staticmethod
    def _format_basis_name_none(
        basis: Union[Basis, Sequence[Basis]]
    ) -> Tuple[Sequence[str], Sequence[Basis]]:
        """Format `basis_name` and `basis` if `basis_name==None`.
        
        Parameters
        ----------
        basis: Union[Basis, Sequence[Basis]]
            Basis of functions as a Basis object.

        Returns
        -------
        Tuple[Sequence[str], Sequence[Basis]]
            Tuple containing the basis names and the basis objects as a list.

        """
        if isinstance(basis, Basis):
            basis = [basis]
        basis_name = len(basis) * ['user-defined']
        return basis_name, basis

    @staticmethod
    def _format_basis_name_not_none(
        basis_name: Union[str, Sequence[str]],
        n_functions: Union[int, Sequence[int]],
        dimension: Union[str, Sequence[str]]
    ) -> Tuple[Sequence[str], Sequence[int], Sequence[str]]:
        """"Format different arguments if `basis_name != None`.

        name: Sequence[str]
            Name of the basis to use.
        n_functions: Sequence[int]
            Number of functions to use to generate the basis.
        dimension: Sequence[str]
            Dimension of the basis to generate.

        Returns
        -------
        Tuple[Sequence[str], Sequence[int], Sequence[str]]
            Tuple containing the basis names, the number of functions ans the
            dimensions as list.

        """
        if isinstance(basis_name, str):
            basis_name = [basis_name]
        if isinstance(n_functions, int):
            n_functions = len(basis_name) * [n_functions]
        if isinstance(dimension, str):
            dimension = len(basis_name) * [dimension]
        return basis_name, n_functions, dimension

    def __init__(
        self,
        basis_name: Union[str, Sequence[str]],
        n_functions: Union[int, Sequence[int]] = 5,
        dimension: Union[str, Sequence[str]] = '1D',
        argvals: Optional[Dict[str, npt.NDArray]] = None,
        basis: Optional[Union[Basis, Sequence[Basis]]] = None,
        random_state: Optional[int] = None,
        **kwargs_basis: Any
    ) -> None:
        """Initialize KarhunenLoeve object."""
        
        # Checkers
        KarhunenLoeve._check_basis_none(basis_name, basis)
        KarhunenLoeve._check_basis_type(basis)
        
        if basis_name is None:
            basis_name, basis = KarhunenLoeve._format_basis_name_none(basis)
        else:
            arguments = KarhunenLoeve._format_basis_name_not_none(
                basis_name, n_functions, dimension
            )
            basis_name, n_functions, dimension = arguments

        # Create the Basis list using the basis_name list.
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

        # Initialize parameters
        n_features = self.basis.n_obs

        centers = _initialize_centers(n_features, n_clusters, centers)
        cluster_std = _initialize_cluster_std(n_features, n_clusters, cluster_std)

        # Generate data
        coef, labels = _make_coef(
            n_obs, n_features, centers, cluster_std, rnorm
        )

        # Generate data
        simus_univariate = [
            _compute_data(
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
