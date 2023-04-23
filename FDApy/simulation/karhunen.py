#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Karhunen-Loève decomposition
----------------------------

"""
import numpy as np
import numpy.typing as npt

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from ..representation.functional_data import (
    DenseFunctionalData, MultivariateFunctionalData
)
from ..representation.basis import Basis, MultivariateBasis
from .simulation import Simulation


#############################################################################
# Definition of the decreasing of the eigenvalues

def _eigenvalues_linear(
    n: np.int64 = 3
) -> npt.NDArray[np.float64]:
    """Generate linear decreasing eigenvalues.

    Parameters
    ----------
    n: np.int64, default=3
        Number of eigenvalues to generates.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> _eigenvalues_linear(n=3)
    array([1.0, 0.6666666666666666, 0.3333333333333333])

    """
    return (n - np.arange(1, n + 1) + 1) / n


def _eigenvalues_exponential(
    n: np.int64 = 3
) -> npt.NDArray[np.float64]:
    """Generate exponential decreasing eigenvalues.

    Parameters
    ----------
    n: np.int64, default=3
        Number of eigenvalues to generates.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> _eigenvalues_exponential(n=3)
    array([1., 0.60653066, 0.36787944])

    """
    return np.exp(-(np.arange(0, n)) / 2)


def _eigenvalues_quadratic(
    n: np.int64 = 3
) -> npt.NDArray[np.float64]:
    """Generate quadratic decreasing eigenvalues.

    Parameters
    ----------
    n: np.int64, default=3
        Number of eigenvalues to generates.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> _eigenvalues_quadratic(n=3)
    array([1., 0.25, 0.11111111])

    """
    return np.power(np.arange(1., n + 1), -2)


def _eigenvalues_inverse(
    n: np.int64 = 3
) -> npt.NDArray[np.float64]:
    """Generate inverse decreasing eigenvalues.

    Parameters
    ----------
    n: np.int64, default=3
        Number of eigenvalues to generates.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> _eigenvalues_inverse(n=3)
    array([1., 0.5, 0.33333333])

    """
    return np.power(np.arange(1., n + 1), -1)


def _eigenvalues_sqrt(
    n: np.int64 = 3
) -> npt.NDArray[np.float64]:
    """Generate square-root decreasing eigenvalues.

    Parameters
    ----------
    n: np.int64, default=3
        Number of eigenvalues to generates.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> _eigenvalues_sqrt(n=3)
    array([1., 0.70710678, 0.57735027, 0.5])

    """
    return np.power(np.arange(1., n + 1), -1 / 2)


def _eigenvalues_wiener(
    n: np.int64 = 3
) -> npt.NDArray[np.float64]:
    """Generate eigenvalues from a Wiener process.

    Parameters
    ----------
    n: np.int64, default=3
        Number of eigenvalues to generates.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n,)
        The generated eigenvalues.

    Example
    -------
    >>> _eigenvalues_wiener(n=3)
    array([0.4052847345693511, 0.04503163717437235, 0.016211389382774045])

    """
    return np.power((np.pi / 2) * (2 * np.arange(1, n + 1) - 1), -2)


def _simulate_eigenvalues(
    name: np.str_,
    n: np.int64 = 3
) -> npt.NDArray[np.float64]:
    """Generate eigenvalues.

    Parameters
    ----------
    name: np.str_,
        Name of the eigenvalues generation process to use. One of
        `{'linear', 'exponential', 'wiener', 'quadratic', 'inverse', 'sqrt'}`.
    n: np.int64, default=3
        Number of eigenvalues to generates. Should be strictly positive.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n,)
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
    elif name == 'quadratic':
        return _eigenvalues_quadratic(n)
    elif name == 'inverse':
        return _eigenvalues_inverse(n)
    elif name == 'sqrt':
        return _eigenvalues_sqrt(n)
    elif name == 'wiener':
        return _eigenvalues_wiener(n)
    else:
        raise NotImplementedError(
            'Eigenvalues generation method is not implemented!'
        )


#############################################################################
# Definition of clusters
def _make_coef(
    n_obs: np.int64,
    n_features: np.int64,
    centers: npt.NDArray[np.float64],
    clusters_std: npt.NDArray[np.float64],
    rnorm: Callable = np.random.multivariate_normal
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate a set of coefficients for the Karhunen-Loève decomposition.

    Parameters
    ----------
    n_obs: np.int64
        Number of observations to simulate.
    n_features: np.int64
        Number of features to simulate.
    centers: npt.NDArray[np.float64], shape=(n_features, n_clusters)
        The centers of the clusters to generate. The ``n_features`` parameter
        corresponds to the number of functions within the basis.
    clusters_std: npt.NDArray[np.float64], shape=(n_features, n_clusters)
        The standard deviation of the clusters to generate. The
        ``n_features`` parameter corresponds to the number of functions within
        the basis.
    rnorm: Callable, default=np.random.multivariate_normal
        Method used to generate Gaussian coefficients.

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
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
            cov=np.diag(clusters_std[:, idx]),
            size=n_obs
        )
        labels[start_idx:end_idx] = idx
    return coefs, labels


def _initialize_centers(
    n_features: np.int64,
    n_clusters: np.int64,
    centers: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray[np.float64]:
    """Initialize the centers of the clusters.

    Parameters
    ----------
    n_features: np.int64
        Number of features to simulate.
    n_clusters: np.int64
        Number of clusters to simulate.
    centers: Optional[npt.NDArray[np.float64]], shape=(n_features, n_clusters)
        The centers of each cluster per feature.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_features, n_clusters)
        An array with good shape for the initialization of the centers of the
        cluster.

    """
    return np.zeros((n_features, n_clusters)) if centers is None else centers


def _initialize_clusters_std(
    n_features: np.int64,
    n_clusters: np.int64,
    clusters_std: Optional[Union[np.str_, npt.NDArray[np.float64]]] = None
) -> npt.NDArray[np.float64]:
    """Initialize the standard deviation of the clusters.

    Parameters
    ----------
    n_features: np.int64
        Number of features to simulate.
    n_clusters: np.int64
        Number of clusters to simulate.
    clusters_std: Optional[Union[np.str_, npt.NDArray[np.float64]]]
        The standard deviation of each cluster per feature. If the parameter
        is given as a string, it has to be one of {`linear`, `exponential`,
        `wiener`}. If `None`, the standard deviation of each cluster per
        feature is set to :math:`1`.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_features, n_clusters)
        An array with good shape for the initialization of the standard
        deviation of the cluster.

    """
    if isinstance(clusters_std, str):
        eigenvalues = _simulate_eigenvalues(clusters_std, n_features)
        return np.full((n_clusters, n_features), eigenvalues).T
    elif clusters_std is None:
        return np.ones((n_features, n_clusters))
    else:
        return clusters_std


#############################################################################
# Generation of univariate functional data
def _compute_data(
    basis: DenseFunctionalData,
    coefficients: npt.NDArray[np.float64]
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
    basis: DenseFunctionalData
        Basis of functions to use for the generation of the data.
    coefficients: npt.NDArray[np.float64], shape=(n_obs, n_features)
        Set of coefficients of shape the number of observations times the
        number of elements in the basis.

    Returns
    -------
    DenseFunctionalData
        Generated data as a DenseFunctionalData object.

    """
    if basis.n_dim == 1:
        values = np.matmul(coefficients, basis.values)
    elif basis.n_dim == 2:
        values = np.tensordot(coefficients, basis.values, axes=1)
    else:
        raise ValueError(
            f"The basis dimension {basis.n_dim} has to be 1D or 2D."
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
    name: Sequence[np.str_], {'legendre', 'wiener', 'fourier', 'bsplines'}
        Name of the basis to use. For multivariate functional data, this is a
        list of `str` of length `n_features`.
    n_functions: Sequence[np.int64], default=5
        Number of functions to use to generate the basis. For multivariate
        functional data, this is a list of `int` of length `n_features`.
    dimension: Sequence[np.str_], {'1D', '2D'}, default='1D'
        Dimension of the basis to generate. For multivariate functional data,
        this is a list of `str` of length `n_features`.
    argvals: Dict[np.str_, npt.NDArray[np.float64]]
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. The shape of the :math:`j`th
        dimension is :math:`(m_j,)` for :math:`0 \leq j \leq p`.
    basis: Optional[Union[Basis, MultivariateBasis]], default=None
        Basis of functions as a Basis object. Used to have a user-defined basis
        of function.
    random_state: np.int64, default=None
        A seed to initialize the random number generator.

    Attributes
    ----------
    data: Union[DenseFunctionalData, MultivariateFunctionalData]
        An object that represents the simulated data.
    noisy_data: Union[DenseFunctionalData, MultivariateFunctionalData]
        An object that represents a noisy version of the simulated data.
    sparse_data: Union[IrregularFunctionalData, MultivariateFunctionalData]
        An object that represents a sparse version of the simulated data.
    labels: npt.NDArray[np.float64], shape=(n_obs,)
        The integer labels for cluster membership of each sample.
    basis: Union[Basis, MultivariateBasis]
        The eigenfunctions used to simulate the data.
    eigenvalues: npt.NDArray[np.float64], shape=(n_functions,)
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
        basis_name: Optional[Union[np.str_, Sequence[np.str_]]],
        basis: Optional[Union[Basis, MultivariateBasis]]
    ) -> None:
        """Check if `basis_name` of `basis` is `None`.

        Parameters
        ----------
        basis_name: Optional[Union[np.str_, Sequence[np.str_]]]
            A str or a sequence of str indicating the name or names of the
            basis.
        basis: Optional[Union[Basis, MultivariateBasis]]
            A Basis instance.

        Raises
        ------
        ValueError
            If both `basis_name` and `basis` are not None.

        """
        if (
            ((basis_name is not None) and (basis is not None)) or
            ((basis_name is None) and (basis is None))
        ):
            raise ValueError(
                'Only one of the arguments `basis_name` or `basis` have to be '
                'provided. Do not know which basis to use.'
            )

    @staticmethod
    def _check_basis_type(
        basis: Optional[Union[Basis, MultivariateBasis]]
    ) -> None:
        """Check if `basis` has the right type.

        Parameters
        ----------
        basis: Optional[Basis, MultivariateBasis]
            A Basis instance or None.

        Raises
        ------
        ValueError
            If the basis argument is not None and is not an instance of the
            Basis class.

        """
        if (
            (not isinstance(basis, (Basis, MultivariateBasis))) and
            (basis is not None)
        ):
            raise ValueError(
                'The basis argument has to be an instance of Basis.'
            )

    @staticmethod
    def _format_basis_name_none(
        basis: Union[Basis, MultivariateBasis]
    ) -> Tuple[Sequence[np.str_], Sequence[Basis]]:
        """Format ``basis_name`` and ``basis`` if ``basis_name==None``.

        Parameters
        ----------
        basis: Union[Basis, MultivariateBasis]
            Basis of functions as a Basis object.

        Returns
        -------
        Tuple[Sequence[np.str_], Sequence[Basis]]
            Tuple containing the basis names and the basis objects as a list.

        """
        #TODO: incorporate MultivariateBasis
        #I think it should return a MultivariateBasis in every case.
        if isinstance(basis, Basis):
            basis = [basis]
        basis_name = len(basis) * ['user-defined']
        return basis_name, basis

    @staticmethod
    def _format_basis_name_not_none(
        basis_name: Union[np.str_, Sequence[np.str_]],
        dimension: Union[np.str_, Sequence[np.str_]]
    ) -> Tuple[Sequence[np.str_], Sequence[np.str_]]:
        """Format different arguments if ``basis_name != None``.

        Parameters
        ----------
        basis_name: Union[np.str_, Sequence[np.str_]]
            Name of the basis to use.
        dimension: Union[np.str_, Sequence[np.str_]
            Dimension of the basis to generate.

        Returns
        -------
        Tuple[Sequence[np.str_], Sequence[np.str_]]
            Tuple containing the basis names, the number of functions ans the
            dimensions as list.

        """
        if isinstance(basis_name, str):
            basis_name = [basis_name]
        if isinstance(dimension, str):
            dimension = len(basis_name) * [dimension]
        return basis_name, dimension

    @staticmethod
    def _create_basis(
        basis_name: Sequence[np.str_],
        dimension: Sequence[np.str_],
        n_functions: np.int64,
        argvals: Optional[npt.NDArray[np.float64]] = None,
        **kwargs_basis
    ) -> MultivariateBasis:
        """Create a list of Basis given some parameters.

        Parameters
        ----------
        basis_name: Sequence[np.str_]
            A sequence of basis names.
        dimension: Sequence[np.str_]
            A sequence of basis dimensions.
        n_functions: np.int64
            The number of functions to generate for each basis object.
        argvals: Optional[npt.NDArray[np.float64]]
            The argument values used to generate the basis functions.
        **kwargs_basis
            Additional keyword arguments used to generate the Basis objects.

        Returns
        -------
        MultivariateBasis
            MultivariateBasis objects generated for each name-dimension pair.

        """
        return MultivariateBasis(
            simulation_type='weighted',
            n_components=len(basis_name),
            name=basis_name,
            n_functions=n_functions,
            dimension=dimension,
            argvals=argvals,
            rchoice=None,
            **kwargs_basis
        )

    def __init__(
        self,
        basis_name: Union[np.str_, Sequence[np.str_]],
        n_functions: np.int_ = 5,
        dimension: Union[np.str_, Sequence[np.str_]] = '1D',
        argvals: Optional[npt.NDArray[np.float64]] = None,
        basis: Optional[Union[Basis, Sequence[Basis]]] = None,
        random_state: Optional[np.int64] = None,
        **kwargs_basis: Any
    ) -> None:
        """Initialize KarhunenLoeve object."""
        # Checkers
        KarhunenLoeve._check_basis_none(basis_name, basis)
        KarhunenLoeve._check_basis_type(basis)

        if basis_name is None:
            basis_name, basis = KarhunenLoeve._format_basis_name_none(basis)
        else:
            basis_name, dimension = KarhunenLoeve._format_basis_name_not_none(
                basis_name, dimension
            )

        # Create the Basis list using the basis_name list.
        if basis is None:
            if isinstance(argvals, np.ndarray):
                argvals = [argvals]
            basis = KarhunenLoeve._create_basis(
                basis_name, dimension, n_functions, argvals, **kwargs_basis
            )
        super().__init__(basis_name, random_state)
        self.basis = basis

    def new(
        self,
        n_obs: np.int64,
        n_clusters: np.int64 = 1,
        argvals: Optional[npt.NDArray[np.float64]] = None,
        **kwargs
    ) -> None:
        """Simulate realizations from Karhunen-Loève decomposition.

        This function generates ``n_obs`` realizations of a Gaussian process
        using the Karhunen-Loève decomposition on a common grid ``argvals``.

        Parameters
        ----------
        n_obs: np.int64
            Number of observations to simulate.
        n_clusters: np.int64, default=1
            Number of clusters to generate.
        argvals: None
            Not used in this context. We will use the ``argvals`` from the
            :mod:`Basis` object as ``argvals`` of the simulation. Here to be
            compliant with the class :mod:`Simulation`.

        Keyword Args
        ------------
        centers: npt.NDArray[np.float64], shape=(n_features, n_clusters)
            The centers of the clusters to generate. The ``n_features``
            correspond to the number of functions within the basis.
        cluster_std: npt.NDArray[np.float64], shape=(n_features, n_clusters)
            The standard deviation of the clusters to generate. The
            ``n_features`` correspond to the number of functions within the
            basis.

        """
        if self.random_state is None:
            rnorm = np.random.multivariate_normal
        else:
            rnorm = self.random_state.multivariate_normal

        # Get parameters
        centers = kwargs.get('centers', None)
        clusters_std = kwargs.get('clusters_std', None)

        # Initialize parameters
        n_features = self.basis[0].n_obs
        centers = _initialize_centers(n_features, n_clusters, centers)
        clusters_std = _initialize_clusters_std(
            n_features, n_clusters, clusters_std
        )

        # Generate coefficients
        coef, labels = _make_coef(
            n_obs, n_features, centers, clusters_std, rnorm
        )
        simus_univariate = [
            _compute_data(basis=basis, coefficients=coef)
            for basis in self.basis
        ]

        if len(simus_univariate) > 1:
            self.data = MultivariateFunctionalData(simus_univariate)
        else:
            self.data = simus_univariate[0]
        self.labels = labels
        self.eigenvalues = clusters_std[:, 0]
