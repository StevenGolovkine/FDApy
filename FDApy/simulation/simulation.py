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
# Class Simulation
class Simulation(ABC):
    """Class that defines functional data simulation.

    Parameters
    ----------
    name: str
        Name of the simulation

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
        noise_variance: Union[float, Callable[[np.ndarray], np.ndarray]] = 1.0
    ) -> None:
        r"""Add noise to functional data objects.

        This function generates an artificial noisy version of a functional
        data object of class :mod:`DenseFunctionalData` by adding realizations
        of Gaussian random variables
        :math:`\epsilon \sim \mathcal{N}(0, \sigma^2)` to the observations. The
        variance :math:`\sigma^2` can be supplied by the user. Heteroscedastic
        noise is considered if a function is given as parameter. The generated data are given by
        
        .. math::
            Y(t) = X(t) + \epsilon.

        For heteroscedastic noise, the parameter :mod:`noise_variance` should
        accept two parameters and we will consider

        .. math::
            \epsilon \sim \mathcal{N}(0, \sigma^2(X(t), t)).

        Parameters
        ----------
        noise_variance: float or callable, default=1
            The variance :math:`\sigma^2` of the Gaussian noise that is added to
            the data.

        Notes
        -----
        TODO: Add checkers for the :mod:`noise_variance` parameter.

        """
        self._check_data()

        shape_simu = self.data.n_obs, *tuple(self.data.n_points.values())
        noisy_data = np.random.normal(0, 1, shape_simu)

        if inspect.isfunction(noise_variance):
            if len(inspect.signature(noise_variance)) == 2:
                noise_variance = noise_variance(
                    self.data.values, self.data.argvals
                )
            else:
                raise AttributeError(
                    'If the parameter `noise_variance` is supplied as a'
                    ' function, it should accept two parameters.'
                )

        std_noise = np.sqrt(noise_variance)
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
