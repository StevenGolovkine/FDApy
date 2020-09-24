#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""
Module for the definition of k-means algorithms.

This module is used to defined the k-means algorithm for both univariate and
multivariate functional data.

References
----------
- Amanda Hernando Bernabé - https://github.com/GAA-UAM/scikit-fda/blob/develop/
skfda/ml/clustering/kmeans.py

"""
from abc import ABC, abstractmethod


###############################################################################
# Class BaseKMeans

class BaseKMeans(ABC):
    """Metaclass for the definition of k-means algorithms.

    Parameters
    ----------
    n_clusters: int
        Number of clusters into which the data are grouped.
    init: FunctionalData, default=None
        Contains the initial centers of the different clusters. If None, the
        centroïds are randomly initialized.

    """

    def __init__(self, n_clusters):
        pass
