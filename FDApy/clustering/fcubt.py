#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""
Module for the definition of the functional CUBT.

This module is used to defined the algorithm functional CUBT. It is used to
cluster functional data using binary trees.
"""
import matplotlib.pyplot as plt
import numpy as np

from ..representation.functional_data import DenseFunctionalData
from ..preprocessing.dim_reduction.fpca import UFPCA
from .optimalK.gap import Gap
from .optimalK.bic import BIC

from matplotlib import colors as mcolors
from sklearn.mixture import GaussianMixture

COLORS = [v for v in mcolors.BASE_COLORS.values()]


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
    is_root: boolean
        Is the node a root node?
    is_leaf: boolean
        Is the node a leaf node?

    Attributes
    ----------
    identifier: int
        An unique identifier of the node. If the Node is a root node, the id
        will be 1. The identifier of the left will be 2 * id and the identifier
        of the right will be 2 * id + 1.
    labels: np.array, shape (n_samples,)
        Component labels.
    left: Node
        Left child of the node.
    right: Node
        Right child of the node.

    """

    @staticmethod
    def _check_data(new_data):
        """Check the user provided `data`."""
        if not isinstance(new_data, DenseFunctionalData):
            raise TypeError("Provided data do not have the right type.")

    def __init__(self, data, identifier=0, is_root=False, is_leaf=False):
        """Initialiaze Node object."""
        self.identifier = 1 if is_root else identifier
        self.data = data
        self.labels = np.repeat(0, data.n_obs)
        self.left = None
        self.right = None
        self.is_root = is_root
        self.is_leaf = is_leaf

    def __str__(self):
        """Override __str__ function."""
        return (f'Node(id={self.identifier}, is_root={self.is_root}'
                f', is_leaf={self.is_leaf})')

    def __repr__(self):
        """Override __repr__ function."""
        return self.__str__()

    @property
    def data(self):
        """Getter for data."""
        return self._data

    @data.setter
    def data(self, new_data):
        self._check_data(new_data)
        self._data = new_data

    @property
    def identifier(self):
        """Getter for identifier."""
        return self._identifier

    @identifier.setter
    def identifier(self, new_identifier):
        self._identifier = new_identifier

    @property
    def labels(self):
        """Getter for labels."""
        return self._labels

    @labels.setter
    def labels(self, new_labels):
        self._labels = new_labels

    @property
    def left(self):
        """Getter for left."""
        return self._left

    @left.setter
    def left(self, new_left):
        self._left = new_left

    @property
    def right(self):
        """Getter for right."""
        return self._right

    @right.setter
    def right(self, new_right):
        self._right = new_right

    @property
    def is_root(self):
        """Getter for is_root."""
        return self._is_root

    @is_root.setter
    def is_root(self, new_is_root):
        self._is_root = new_is_root

    @property
    def is_leaf(self):
        """Getter for is_left."""
        return self._is_leaf

    @is_leaf.setter
    def is_leaf(self, new_is_leaf):
        self._is_leaf = new_is_leaf

    def split(self, splitting_criteria='bic', n_components=1):
        """Split the data into two groups.

        Parameters
        ----------
        splitting_criteria: str, {'gap', 'bic'}, default='bic'
            The splitting criteria used to decide if a split is done or not.
        n_components : int, float, None, default=None
            Number of components to keep.
            if n_components is int, n_components are kept.
            if 0 < n_components < 1, select the number of components such that
            the amount of variance that needs to be explained is greater than
            the percentage specified by n_components.

        """
        ufpca = UFPCA(n_components=n_components)
        ufpca.fit(data=self.data, method='GAM')
        scores = ufpca.transform(data=self.data, method='NumInt')

        if splitting_criteria == 'bic':
            bic_stat = BIC(parallel_backend='multiprocessing')
            best_k = bic_stat(scores, np.arange(1, 5))
        elif splitting_criteria == 'gam':
            gap_stat = Gap(generating_process='uniform', metric='euclidean')
            best_k = gap_stat(scores, np.arange(1, 5), n_refs=3)
        else:
            raise NotImplementedError('Splitting criteria not implemented.')

        if best_k > 1:
            gm = GaussianMixture(n_components=2)
            prediction = gm.fit_predict(scores)
            self.labels = prediction
            self.left = Node(self.data[prediction == 0],
                             2 * self.identifier)
            self.right = Node(self.data[prediction == 1],
                              2 * self.identifier + 1)
        else:
            self.is_leaf = True

    def plot(self, axes=None, **plt_kwargs):
        """Plot of a Node object.

        Parameters
        ----------
        axes: matplotlib.axes
            Axes object onto which the objects are plotted.
        **plt_kwargs:
            Keywords plotting arguments.

        Returns
        -------
        axes: matplotlib.axes
            Axes object containing the graphs.

        """
        if axes is None:
            axes = plt.gca()

        for o, i in zip(self.data.values, self.labels):
            axes.plot(self.data.argvals['input_dim_0'], o,
                      c=COLORS[i], **plt_kwargs)
        return axes


###############################################################################
# Class fCUBT

class fCUBT():
    """A class defining a functional CUBT.

    Parameters
    ----------
    root_node: Node, default=Node
        The root node of the tree.

    Attributes
    ----------
    tree: list of Node
        A tree represented as a list of Node.
    n_nodes: int
        Number of nodes in the tree.
    labels: np.array, shape (n_samples,)
        Component labels.
    n_nodes: int
        Number of nodes in the tree.
    n_leaf: int
        Number of leaves in the tree.
    height: int
        Height of the tree.

    """

    def __init__(self, root_node=None):
        """Initialize fCUBT object."""
        self.root_node = root_node
        self.tree = [root_node]

    @property
    def root_node(self):
        """Getter for root_node."""
        return self._root_node

    @root_node.setter
    def root_node(self, new_root_node):
        self._root_node = new_root_node

    @property
    def tree(self):
        """Getter for tree."""
        return self._tree

    @tree.setter
    def tree(self, new_tree):
        self._tree = new_tree

    @property
    def labels(self):
        """Getter for labels."""
        return self._labels

    @labels.setter
    def labels(self, new_labels):
        self._labels = new_labels

    @property
    def n_nodes(self):
        """Get the number of nodes in the tree."""
        return len(self.tree)

    @property
    def n_leaf(self):
        """Get the number of leaves in the tree."""
        return len([True for node in self.tree if node.is_leaf])

    @property
    def height(self):
        """Get the height of the tree."""
        return int(np.ceil(np.log2(self.n_nodes + 1)))

    def grow(self):
        """Grow a complete tree."""
        self.tree = self._recursive_clustering(self.tree)

    def prune(self):
        """Prune the tree."""
        pass

    def join(self):
        """Join elements of the tree."""
        pass

    def _recursive_clustering(self, list_nodes):
        """Perform the binary clustering recursively."""
        tree = []
        for node in list_nodes:
            if node is not None:
                tree.append(node)
                node.split(splitting_criteria='bic', n_components=0.95)
                tree.extend(self._recursive_clustering([node.left,
                                                        node.right]))
        return tree
