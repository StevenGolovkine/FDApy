#!/usr/bin/env python
# -*-coding:utf8 -*

"""Simulation class
-------------------

"""
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional

from ..representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData
)


#############################################################################
# Class Simulation
class Simulation(ABC):
    """Class that defines functional data simulation.

    Parameters
    ----------
    name: str
        Name of the simulation
    random_state: int, default=None
        A seed to initialize the random number generator.

    Attributes
    ----------
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
            raise ValueError(
                'No data have been found in the simulation.'
                ' Please run new() before add_noise() or sparsify().'
            )

    def __init__(
        self,
        name: str,
        random_state: Optional[int] = None
    ) -> None:
        """Initialize Simulation object."""
        super().__init__()
        self.name = name

        if random_state is not None:
            self.random_state = np.random.default_rng(random_state)
        else:
            self.random_state = None

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
        """Simulate a new set of curves."""
        pass

    def add_noise(
        self,
        noise_variance: float = 1.0
    ) -> None:
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
        noise_variance: float, default=1.0
            The variance :math:`\sigma^2` of the Gaussian noise that is added
            to the data.

        """
        self._check_data()

        # Get parameter of the data
        shape_simu = self.data.n_obs, *tuple(self.data.n_points.values())

        # Define function for reproducibility
        if self.random_state is None:
            rnorm = np.random.normal
        else:
            rnorm = self.random_state.normal

        noisy_data = rnorm(0, 1, shape_simu)
        std_noise = np.sqrt(noise_variance)
        noisy_data = self.data.values + np.multiply(std_noise, noisy_data)
        self.noisy_data = DenseFunctionalData(self.data.argvals, noisy_data)

    def sparsify(
        self,
        percentage: float = 0.9,
        epsilon: float = 0.05
    ) -> None:
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
        percentage: float, default=0.9
            The percentage of observations to be retained.
        epsilon: float, default=0.05
            The uncertainty around the percentage of observations to be
            retained.

        """
        if self.data.n_dim > 1:
            raise ValueError(
                'The sparsification is not implemented for data'
                ' with dimension larger than 1.'
            )
        self._check_data()

        # Get parameters of the data
        n_obs = self.data.n_obs
        points = np.arange(0, n_obs)

        # Define functions for reproducibility
        if self.random_state is None:
            runif = np.random.uniform
            rchoice = np.random.choice
        else:
            runif = self.random_state.uniform
            rchoice = self.random_state.choice

        perc = np.around(100 * runif(
            max(0, percentage - epsilon),
            min(1, percentage + epsilon),
            n_obs
        )).astype(int)

        argvals, values = {}, {}
        for idx, (obs, n_pts) in enumerate(zip(self.data, perc.astype(int))):
            indices = np.sort(rchoice(points, size=n_pts, replace=False))
            argvals[idx] = obs.argvals['input_dim_0'][indices]
            values[idx] = obs.values[0][indices]

        self.sparse_data = IrregularFunctionalData(
            {'input_dim_0': argvals}, values
        )
