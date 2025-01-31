#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Simulation class
----------------

"""
import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import Callable

from ..representation.argvals import DenseArgvals, IrregularArgvals
from ..representation.values import DenseValues, IrregularValues
from ..representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData,
    BasisFunctionalData,
    MultivariateFunctionalData,
)


#############################################################################
# Noise for univariate functional data
def _add_noise_univariate_data(
    data: DenseFunctionalData,
    noise_variance: float = 1.0,
    rnorm: Callable = np.random.normal,
) -> DenseFunctionalData:
    r"""Add noise to univariate functional data.

    This function generates an artificial noisy version of a functional
    data object of class :mod:`DenseFunctionalData` by adding realizations
    of Gaussian random variables :math:`\epsilon \sim \mathcal{N}(0, \sigma^2)`
    to the observations. The variance :math:`\sigma^2` can be supplied by the
    user. The generated data are given by

    .. math::
        Y(t) = X(t) + \epsilon.

    Parameters
    ----------
    data
        Functional data to add the noise.
    noise_variance
        The variance :math:`\sigma^2` of the Gaussian noise that is added
        to the data.
    rnorm
        Random data generator.

    Returns
    -------
    DenseFunctionalData
        Noisy version of the functional data.

    """
    # Get parameter of the data
    shape_simu = data.n_obs, *data.n_points

    noisy_data = rnorm(0, 1, shape_simu)
    std_noise = np.sqrt(noise_variance)
    noisy_data = data.values + np.multiply(std_noise, noisy_data)
    return DenseFunctionalData(DenseArgvals(data.argvals), DenseValues(noisy_data))


#############################################################################
# Sparsify univariate functional data
def _sparsify_univariate_data(
    data: DenseFunctionalData,
    percentage: float = 0.9,
    epsilon: float = 0.05,
    runif: Callable = np.random.uniform,
    rchoice: Callable = np.random.choice,
) -> IrregularFunctionalData:
    r"""Sparsify univariate functional data.

    This function generates an artificially sparsified version of a
    functional data object of class :mod:`DenseFunctionalData`. The
    percentage (and the uncertainty around it) of the number of observation
    points retained can be supplied by the user. Let :math:`p` be the
    defined percentage and :math:`\epsilon` be the uncertainty value. The
    retained number of observations will be different for each curve and be
    between :math:`p - \epsilon` and :math:`p + \epsilon`.

    Parameters
    ----------
    data
        Functional data to sparsify.
    percentage
        The percentage of observations to be retained.
    epsilon
        The uncertainty around the percentage of observations to be
        retained.
    runif
        Random data generator.
    rchoice
        Random data generator.

    Returns
    -------
    IrregularFunctionalData
        Sparse version of the functional data.

    """
    # Get parameters of the data
    n_obs, n_points = data.n_obs, np.prod(data.n_points)

    perc = runif(max(0, percentage - epsilon), min(1, percentage + epsilon), n_obs)

    argvals, values = {}, {}
    for idx, (obs, perc_obs) in enumerate(zip(data, perc)):
        mask = rchoice([False, True], size=n_points, p=(1 - perc_obs, perc_obs))
        if np.sum(mask) < 2:
            mask[rchoice(np.arange(n_points), size=2)] = True
        val = obs[0].values.flatten()
        val[~mask] = np.nan
        argvals[idx] = data.argvals
        values[idx] = val.reshape(data.n_points)
    return IrregularFunctionalData(IrregularArgvals(argvals), IrregularValues(values))


#############################################################################
# Class Simulation
class Simulation(ABC):
    """Define the structure of Simulation.

    Parameters
    ----------
    basis_name
        Name of the simulation
    random_state
        A seed to initialize the random number generator.

    Attributes
    ----------
    data: DenseFunctionalData | MultivariateFunctionalData
        An object that represents the simulated data.
    noisy_data: DenseFunctionalData | MultivariateFunctionalData
        An object that represents a noisy version of the simulated data.
    sparse_data: IrregularFunctionalData | MultivariateFunctionalData
        An object that represents a sparse version of the simulated data.

    """

    ###########################################################################
    # Checkers
    def _check_data(self) -> None:
        """Check if self has the attribut data."""
        if self.data is None:
            raise ValueError(
                "No data have been found in the simulation."
                " Please run new() before add_noise() or sparsify()."
            )

    def _check_dimension(self) -> None:
        """Check if self.data has the right dimension."""
        if (
            (isinstance(self.data, DenseFunctionalData) and self.data.n_dimension > 1)
            or (
                isinstance(self.data, BasisFunctionalData) and self.data.n_dimension > 1
            )
            or (
                isinstance(self.data, MultivariateFunctionalData)
                and all(n_dim > 1 for n_dim in self.data.n_dimension)
            )
        ):
            raise ValueError(
                "The sparsification is not implemented for data"
                " with dimension larger than 1."
            )

    ###########################################################################

    ###########################################################################
    # Magic methods
    def __init__(self, basis_name: str, random_state: int | None = None) -> None:
        """Initialize Simulation object."""
        super().__init__()
        self.data = None
        self.basis_name = basis_name

        if random_state is not None:
            self.random_state = np.random.default_rng(random_state)
        else:
            self.random_state = None

    ###########################################################################

    ###########################################################################
    # Properties
    @property
    def basis_name(self) -> str:
        """Getter for basis_name."""
        return self._basis_name

    @basis_name.setter
    def basis_name(self, new_basis_name: str) -> None:
        self._basis_name = new_basis_name

    ###########################################################################

    ###########################################################################
    # Abstract methods
    @abstractmethod
    def new(
        self,
        n_obs: int,
        n_clusters: int = 1,
        argvals: npt.NDArray[np.float64] | None = None,
        **kwargs,
    ) -> None:
        """Simulate a new set of curves."""

    ###########################################################################

    ###########################################################################
    # Methods
    def add_noise(self, noise_variance: float = 1.0) -> None:
        r"""Add noise to functional data objects.

        This function generates an artificial noisy version of a functional
        data object of class :mod:`DenseFunctionalData` by adding realizations
        of Gaussian random variables
        :math:`\epsilon \sim \mathcal{N}(0, \sigma^2)` to the observations. The
        variance :math:`\sigma^2` can be supplied by the user. The generated
        data are given by

        .. math::
            Y(t) = X(t) + \epsilon.

        Parameters
        ----------
        noise_variance
            The variance :math:`\sigma^2` of the Gaussian noise that is added
            to the data.

        Returns
        -------
        None
            Create the class attribute `noisy_data`.

        """
        self._check_data()

        # Define function for reproducibility
        if self.random_state is None:
            rnorm = np.random.normal
        else:
            rnorm = self.random_state.normal

        if isinstance(self.data, DenseFunctionalData):
            self.noisy_data = _add_noise_univariate_data(
                self.data, noise_variance, rnorm
            )
        else:
            self.noisy_data = MultivariateFunctionalData(
                [
                    _add_noise_univariate_data(data, noise_variance, rnorm)
                    for data in self.data.data
                ]
            )

    def sparsify(self, percentage: float = 0.9, epsilon: float = 0.05) -> None:
        r"""Generate a sparse version of functional data objects.

        This function generates an artificially sparsified version of a
        functional data object of class :mod:`DenseFunctionalData`. The
        percentage (and the uncertainty around it) of the number of observation
        points retained can be supplied by the user. Let :math:`p` be the
        defined percentage and :math:`\epsilon` be the uncertainty value. The
        retained number of observations will be different for each curve and be
        between :math:`p - \epsilon` and :math:`p + \epsilon`.

        Parameters
        ----------
        percentage
            The percentage of observations to be retained.
        epsilon
            The uncertainty around the percentage of observations to be
            retained.

        Returns
        -------
        None
            Create the class attribute `sparse_data`.

        """
        self._check_data()
        self._check_dimension()

        # Define functions for reproducibility
        if self.random_state is None:
            runif = np.random.uniform
            rchoice = np.random.choice
        else:
            runif = self.random_state.uniform
            rchoice = self.random_state.choice

        if isinstance(self.data, DenseFunctionalData):
            self.sparse_data = _sparsify_univariate_data(
                self.data, percentage, epsilon, runif, rchoice
            )
        else:
            self.sparse_data = MultivariateFunctionalData(
                [
                    _sparsify_univariate_data(data, percentage, epsilon, runif, rchoice)
                    for data in self.data.data
                ]
            )

    def add_noise_and_sparsify(
        self,
        noise_variance: float = 1.0,
        percentage: float = 0.9,
        epsilon: float = 0.05,
    ) -> None:
        r"""Generate a noisy and sparse version of functional data objects.

        This function generates an artificially noisy and sparse version of a
        functional datasets. From a functional dataset, it first generates the
        noisy version and then the sparse version based on the noisy one.

        Parameters
        ----------
        noise_variance
            The variance :math:`\sigma^2` of the Gaussian noise that is added
            to the data.
        percentage
            The percentage of observations to be retained.
        epsilon
            The uncertainty around the percentage of observations to be
            retained.

        Returns
        -------
        None
            Create the class attributes `noisy_data` and `sparse_data`.

        """
        self.add_noise(noise_variance=noise_variance)

        tmp = self.data
        self.data = self.noisy_data
        self.sparsify(percentage=percentage, epsilon=epsilon)
        self.data = tmp

    ###########################################################################
