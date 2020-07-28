#!/usr/bin/env python
# -*-coding:utf8 -*

"""Module for the representation of the Gap statistic.

This module is used to represent the Gap statistic as an object.
"""
from typing import Callable, NamedTuple


##############################################################################
# Class GapResult

class GapResult(NamedTuple):
    """An object containing the Gap statistic for given dataset and k."""
    value: float
    k: int
    sd_k: float

    def __repr__(self) -> str:
        return (f"For {self.k} clusters considered, the Gap statistic is"
                f" {self.value} ({self.sd_k})")


##############################################################################
# Class Gap

class Gap():
    """A module for the computation of the Gap statistic.

    This module is used to compute the Gap statistic for a given dataset and
    a given number of cluster k.

    Attributes
    ----------
    result: GapResult

    """

    def __init__(
        self,
        clusterer: Callable = None,
        clusterer_kwargs: dict = None,
    ) -> None:
        """Initialize Gap object.

        Parameters
        ----------
        clusterer : Callable, default=None
            An user-provided function for the clustering of the dataset.
        clusterer : dict, default=None
            The parameters to be used by the clustering function.

        """
        pass
