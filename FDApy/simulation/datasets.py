#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Datasets
--------

"""
import numpy as np
import numpy.typing as npt

from typing import Callable

from ..representation.argvals import DenseArgvals
from ..representation.values import DenseValues
from ..representation.functional_data import DenseFunctionalData
from .simulation import Simulation


#############################################################################
# Definition of the simulation settings


def _zhang_chen(
    n_obs: int, argvals: npt.NDArray[np.float64], rnorm: Callable = np.random.normal
) -> npt.NDArray[np.float64]:
    """Define a simulation from Zhang and Chen (2007).

    This function reproduces simulation in [1]_.

    References
    ----------
    .. [1] Zhang, J.-T. and Chen J. (2007), Statistical Inferences for
        Functional Data, The Annals of Statistics, Vol. 35, No. 3.

    """
    cos = np.cos(2 * np.pi * argvals)
    sin = np.sin(2 * np.pi * argvals)

    mu = 1.2 + 2.3 * cos + 4.2 * sin

    results = np.zeros((n_obs, len(argvals)))
    for idx in np.arange(n_obs):
        coefs = rnorm(0, (1, np.sqrt(2), np.sqrt(3)))
        vi = coefs[0] + coefs[1] * cos + coefs[2] * sin
        eps = rnorm(0, np.sqrt(0.1 * (1 + argvals)))
        results[idx, :] = mu + vi + eps
    return results


#############################################################################
# Definition of the Datasets simulation


class Datasets(Simulation):
    r"""Simulate published paper datasets.

    Parameters
    ----------
    basis_name: str
        Name of the datasets to simulate.

    Attributes
    ----------
    data
        An object that represents the simulated data.
    noisy_data
        An object that represents a noisy version of the simulated data.
    sparse_data
        An object that represents a sparse version of the simulated data.

    """

    def __init__(self, basis_name: str, random_state: int | None = None) -> None:
        """Initialize Datasets object."""
        super().__init__(basis_name, random_state)

    def new(
        self,
        n_obs: int,
        n_clusters: int = 1,
        argvals: npt.NDArray[np.float64] | None = None,
        **kwargs,
    ) -> None:
        """Simulate realizations of the Datasets.

        This function generates ``n_obs`` realizations of the Datasets object.

        Parameters
        ----------
        n_obs
            Number of observations to simulate.
        n_clusters
            Not used in this context.
        argvals
            Not used in this context. We will use the ``argvals`` from the
            :mod:`Basis` object as ``argvals`` of the simulation. Here to be
            compliant with the class :mod:`Simulation`.

        Returns
        -------
        None
            Create the class attributes `data`.

        """
        if self.basis_name == "zhang_chen":
            self.data = DenseFunctionalData(
                argvals=DenseArgvals({"input_dim_0": argvals}),
                values=DenseValues(_zhang_chen(n_obs=n_obs, argvals=argvals)),
            )
        else:
            raise NotImplementedError
