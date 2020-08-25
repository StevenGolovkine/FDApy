#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""
Module for the definition of the functional CUBT.

This module is used to defined the algorithm functional CUBT. It is used to
cluster functional data using binary trees.
"""
import numpy as np

from ..representation.functional_data import DenseFunctionalData
from ..preprocessing.dim_reduction.fpca import UFPCA
from .optimalK.gap import Gap

from sklearn.mixture import GaussianMixture


###############################################################################
# Class Node

class Node():
    """A class defining a node of the tree.

    A class used to define a node in a tree. A node is represented as a
    FunctionalData object linked that can be splitted into two groups: left
    and right.

    Parameters
    ----------
    data: FunctionalData
        The data as FunctionalData object.

    Attributes
    ----------
    id: int
        An unique identifier of the node.
    left: Node
        Left child of the node.
    right: Node
        Right child of the node.
    is_root: boolean
        Is the node a root node?
    is_leaf: boolean
        Is the node a leaf node?

    """

    @staticmethod
    def _check_data(new_data):
        """Check the user provided `data`."""
        if not isinstance(new_data, DenseFunctionalData):
            raise TypeError("Provided data do not have the right type.")

    def __init__(self, data):
        """Initialiaze Node object."""
        self.data = data
        self.left = None
        self.right = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._check_data(new_data)
        self._data = new_data

    def split(self):
        """Split the data into two groups."""
        ufpca = UFPCA(n_components=1)
        ufpca.fit(data=self.data)
        scores = ufpca.transform(data=self.data)

        gap_stat = Gap(generating_process='uniform', metric='euclidean')
        best_k = gap_stat(scores, np.arange(1, 5), n_refs=3)
        if best_k > 1:
            gm = GaussianMixture(n_components=2)
            prediction = gm.fit_predict(scores)
            self.left = Node(self.data[prediction == 0])
            self.right = Node(self.data[prediction == 1])


###############################################################################
# Class fCUBT

class fCUBT():
    """A class defining a functional CUBT."""

    def __init__(self):
        """Initialize fCIBT object."""
        pass

    def grow(self):
        """Grow a complete tree."""
        pass

    def prune(self):
        """Prune the tree."""
        pass

    def join(self):
        """Join elements of the tree."""
        pass
