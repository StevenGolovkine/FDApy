#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Datasets
--------

"""
from typing import Optional
import numpy as np
import numpy.typing as npt

from ..representation.functional_data import (
    DenseFunctionalData, MultivariateFunctionalData
)
from .simulation import Simulation


#############################################################################
# Definition of the Datasets simulation

class Datasets(Simulation):
    r"""Class that defines simulation based on published papers.

    Parameters
    ----------
    basis_name: np.str_
        Name of the datasets to simulate.

    """

    def __init__(
        self,
        basis_name: np.str_,
        random_state: Optional[np.int64] = None
    ) -> None:
        """Initialize Datasets object."""
        super().__init__(basis_name, random_state)

    def new(
        self,
        n_obs: np.int64,
        n_clusters: np.int64 = 1,
        argvals: Optional[npt.NDArray[np.float64]] = None,
        **kwargs
    ) -> None:
        """Simulate realizations of the Datasets.

        This function generates ``n_obs`` realizations of the Datasets object.

        """
        return super().new(n_obs, n_clusters, argvals, **kwargs)
