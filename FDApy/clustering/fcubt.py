#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""
Module for the definition of the functional CUBT.

This module is used to defined the algorithm functional CUBT. It is used to
cluster functional data using binary trees.
"""
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..representation.functional_data import (DenseFunctionalData,
                                              MultivariateFunctionalData)
from ..preprocessing.dim_reduction.fpca import UFPCA, MFPCA
from ..preprocessing.dim_reduction.fcp_tpa import FCPTPA
from .optimalK.gap import Gap
from .optimalK.bic import BIC

from sklearn.mixture import GaussianMixture

COLORS = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']


###############################################################################
# Utility functions

def joining_step(list_nodes, siblings, n_components=0.95, max_group=5):
    """Perform a joining step.

    Parameters
    ----------
    list_nodes: list of Nodes
        List of nodes to consider for the joining.
    siblings: set of tuples
        Set of tuples where each tuple contains two siblings nodes.
    n_components: int or float, default=0.95
        Number of components to keep for the aggregation of the nodes.
    max_group: int, default=5
        Number of models to try to split the data.

    Returns
    -------
    nodes: list of Nodes
        The resulting list of nodes after the joining.

    """
    nodes_combinations = set(itertools.combinations(list_nodes, 2))
    edges = nodes_combinations - siblings

    graph = nx.Graph()
    graph.add_nodes_from(list_nodes)
    graph.add_edges_from(edges)

    edges_to_remove = []
    for node1, node2 in graph.edges:
        new_data = node1.data.concatenate(node2.data)

        if isinstance(new_data, DenseFunctionalData):
            if new_data.n_dim == 1:
                ufpca = UFPCA(n_components=n_components)
                ufpca.fit(data=new_data, method='GAM')
                scores = ufpca.transform(data=new_data, method='NumInt')
            elif new_data.n_dim == 2:
                n_points = new_data.n_points
                Pv = np.diff(np.identity(n_points['input_dim_0']))
                Pw = np.diff(np.identity(n_points['input_dim_1']))
                fcptpa = FCPTPA(n_components=n_components)
                fcptpa.fit(new_data, penal_mat={'v': np.dot(Pv, Pv.T),
                                                'w': np.dot(Pw, Pw.T)},
                           alpha_range={'v': np.array([1e-4, 1e4]),
                                        'w': np.array([1e-4, 1e4])},
                           tol=1e-4, max_iter=15,
                           adapt_tol=True)
                scores = fcptpa.transform(new_data)
            else:
                raise ValueError("The dimension of the input data should "
                                 "be 1 or 2.")
        elif isinstance(new_data, MultivariateFunctionalData):
            mfpca = MFPCA(n_components=n_components)
            mfpca.fit(data=new_data, method='NumInt')
            scores = mfpca.transform(new_data)
        else:
            raise TypeError("Not the right data type!")

        max_group = min(max_group, new_data.n_obs)
        bic_stat = BIC(parallel_backend=None)
        best_k = bic_stat(scores, np.arange(1, max_group))
        if best_k > 1:
            edges_to_remove.append((node1, node2))
        else:
            graph[node1][node2]['bic'] = bic_stat.bic_df['bic_value'].min()

    graph.remove_edges_from(edges_to_remove)

    if graph.number_of_edges() != 0:
        bic_dict = nx.get_edge_attributes(graph, 'bic')
        nodes_to_concat = min(bic_dict, key=bic_dict.get)
        nodes_concat = nodes_to_concat[0].unite(nodes_to_concat[1])

        graph.add_node(nodes_concat)
        graph.remove_node(nodes_to_concat[0])
        graph.remove_node(nodes_to_concat[1])

    return list(graph.nodes)


def format_label(list_nodes):
    """Format the labels.

    Parameters
    ----------
    list_nodes: list of Nodes
        A list of nodes representing a clustering of the observations.
        Typically, it should the leaves of the grown tree or the result of the
        joining step.

    Returns
    -------
    labels: np.ndarray
        The labels ordered using the index observation within the nodes.

    """
    mapping, labels, order = {}, [], []
    for idx, node in enumerate(list_nodes):
        mapping[node] = idx
        labels = np.hstack([labels, np.repeat(idx, len(node.idx_obs))])
        order = np.hstack([order, node.idx_obs])
    return mapping, labels[np.argsort(order)].astype(int)


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
    identifier: tuple of int
        An unique identifier of the node. The format is (depth, position). If
        the Node is a root node, the id will be (0, 0). Then, for a node with
        identifier (d, j), the identifier of the left child will be
        (d + 1, 2 * j) and the identifier of the right node will be
        (d + 1, 2 * j + 1).
    idx_obs: np.array, shape=(n_samples,), default=None
        Array to remember the observation in the node. If None, it will be
        initialized as np.arange(data.n_obs).
    is_root: boolean
        Is the node a root node?
    is_leaf: boolean
        Is the node a leaf node?

    Attributes
    ----------
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
        if not isinstance(new_data, (DenseFunctionalData,
                                     MultivariateFunctionalData)):
            raise TypeError("Provided data do not have the right type.")

    def __init__(self, data, identifier=(0, 0), idx_obs=None,
                 is_root=False, is_leaf=False):
        """Initialiaze Node object."""
        self.identifier = (0, 0) if is_root else identifier
        self.data = data
        self.idx_obs = np.arange(data.n_obs) if idx_obs is None else idx_obs
        self.labels = np.repeat(0, data.n_obs)
        self.left = None
        self.right = None
        self.is_root = is_root
        self.is_leaf = is_leaf

    def __str__(self):
        """Override __str__ function."""
        return (f"Node(id={self.identifier}, is_root={self.is_root}"
                f", is_leaf={self.is_leaf})")

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
    def labels_grow(self):
        """Getter for labels_grow."""
        return self._labels_grow

    @labels_grow.setter
    def labels_grow(self, new_labels):
        self._labels_grow = new_labels

    @property
    def labels_join(self):
        """Getter for labels_join."""
        return self._labels_join

    @labels_join.setter
    def labels_join(self, new_labels):
        self._labels_join = new_labels

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

    def split(self, splitting_criteria='bic', n_components=1, min_size=10,
              max_group=5):
        """Split a node into two groups.

        Parameters
        ----------
        splitting_criteria: str, {'gap', 'bic'}, default='bic'
            The splitting criteria used to decide if a split is done or not.
        n_components: int, float, None, default=None
            Number of components to keep.
            if n_components is int, n_components are kept.
            if 0 < n_components < 1, select the number of components such that
            the amount of variance that needs to be explained is greater than
            the percentage specified by n_components.
        min_size: int, default=10
            Minimum number of observation within the node in order to try to
            be split.
        max_group: int, default=5
            Number of models to try to split the data.

        """
        if self.data.n_obs > min_size:
            if isinstance(self.data, DenseFunctionalData):
                if self.data.n_dim == 1:
                    ufpca = UFPCA(n_components=n_components)
                    ufpca.fit(data=self.data, method='GAM')
                    scores = ufpca.transform(data=self.data, method='NumInt')
                    self.fpca = ufpca
                elif self.data.n_dim == 2:
                    n_points = self.data.n_points
                    mat_v = np.diff(np.identity(n_points['input_dim_0']))
                    mat_w = np.diff(np.identity(n_points['input_dim_1']))
                    fcptpa = FCPTPA(n_components=n_components)
                    fcptpa.fit(self.data,
                               penal_mat={'v': np.dot(mat_v, mat_v.T),
                                          'w': np.dot(mat_w, mat_w.T)},
                               alpha_range={'v': np.array([1e-4, 1e4]),
                                            'w': np.array([1e-4, 1e4])},
                               tol=1e-4, max_iter=15,
                               adapt_tol=True)
                    scores = fcptpa.transform(self.data)
                    self.fpca = fcptpa
                else:
                    raise ValueError("The dimension of the input data should "
                                     "be 1 or 2.")
            elif isinstance(self.data, MultivariateFunctionalData):
                mfpca = MFPCA(n_components=n_components)
                mfpca.fit(data=self.data, method='NumInt')
                scores = mfpca.transform(self.data, method='NumInt')
                self.fpca = mfpca
            else:
                raise TypeError("Not the right data type!")

            max_group = min(max_group, self.data.n_obs)
            if splitting_criteria == 'bic':
                bic_stat = BIC(parallel_backend=None)
                best_k = bic_stat(scores, np.arange(1, max_group))
            elif splitting_criteria == 'gap':
                gap_stat = Gap(generating_process='uniform',
                               metric='euclidean',
                               parallel_backend=None)
                best_k = gap_stat(scores, np.arange(1, max_group), n_refs=3)
            else:
                raise NotImplementedError('Not implemented.')

            if best_k > 1:
                gm = GaussianMixture(n_components=2)
                prediction = gm.fit_predict(scores)

                if isinstance(self.data, DenseFunctionalData):
                    left_data = self.data[prediction == 0]
                    right_data = self.data[prediction == 1]
                elif isinstance(self.data, MultivariateFunctionalData):
                    left_data = MultivariateFunctionalData(
                        [obj[prediction == 0] for obj in self.data])
                    right_data = MultivariateFunctionalData(
                        [obj[prediction == 1] for obj in self.data])
                self.gaussian_model = gm
                self.labels = prediction
                self.left = Node(left_data,
                                 identifier=(self.identifier[0] + 1,
                                             2 * self.identifier[1]),
                                 idx_obs=self.idx_obs[prediction == 0])
                self.right = Node(right_data,
                                  identifier=(self.identifier[0] + 1,
                                              2 * self.identifier[1] + 1),
                                  idx_obs=self.idx_obs[prediction == 1])
            else:
                self.is_leaf = True
        else:
            self.is_leaf = True

    def unite(self, node):
        """Unite two nodes into one.

        Parameters
        ----------
        node: Node
            The node to unite with self.

        Returns
        -------
        res: Node
            The unification of self and node.

        """
        data = self.data.concatenate(node.data)

        if isinstance(self.identifier, tuple):
            if isinstance(node.identifier, tuple):
                new_id = [self.identifier, node.identifier]
            elif isinstance(node.identifier, list):
                new_id = [self.identifier, *node.identifier]
            else:
                raise TypeError("Wrong type for node.identifier.")
        elif isinstance(self.identifier, list):
            if isinstance(node.identifier, tuple):
                new_id = [*self.identifier, node.identifier]
            elif isinstance(node.identifier, list):
                new_id = [*self.identifier, *node.identifier]
            else:
                raise TypeError("Wrong type for node.identifier.")
        else:
            raise TypeError("Wrong type for self.identifier.")
        return Node(data,
                    identifier=new_id,
                    idx_obs=np.hstack([self.idx_obs, node.idx_obs]),
                    is_root=(self.is_root & node.is_root),
                    is_leaf=(self.is_leaf & node.is_leaf))

    def isin(self, node):
        """Test whether self is include in node."""
        return self.identifier in node.identifier

    def predict(self, new_obs):
        """Predict the label for a new observation."""
        score = self.fpca.transform(new_obs, method='NumInt')
        pred = self.gaussian_model.predict(score)
        if pred == 0:
            return self.left
        elif pred == 1:
            return self.right
        else:
            raise ValueError(f"Error in the prediction for {self}.")

    def predict_proba(self, new_obs):
        """Predict the probability for a new observation."""
        score = self.fpca.transform(new_obs, method='NumInt')
        proba = self.gaussian_model.predict_proba(score)
        return proba

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
        if self.data.n_dim > 1:
            raise ValueError("Prediction is not available for data with "
                             "dimension strictly greater than 1.")
        if axes is None:
            axes = plt.gca()

        for o, i in zip(self.data.values, self.labels):
            axes.plot(self.data.argvals['input_dim_0'], o,
                      c=COLORS[i], **plt_kwargs)
        axes.set_title(f'{self.identifier}')
        return axes


###############################################################################
# Class fCUBT

class FCUBT():
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
    mapping: dict
        A mapping between leaf nodes and cluster labels.
    labels: np.array, shape (n_samples,)
        Component labels after the tree has been grown.
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
    def n_nodes(self):
        """Get the number of nodes in the tree."""
        return len(self.tree)

    @property
    def n_leaf(self):
        """Get the number of leaves in the tree."""
        return len([True for node in self.tree if node.is_leaf])

    @property
    def height(self):
        """Get the height of the tree.

        The height of the tree is defined starting at 1. So, a tree with only
        a root node will have height 1.

        Returns
        -------
        height: int
            The height of the tree.

        """
        return self.tree[-1].identifier[0] + 1

    def grow(self, n_components=0.95, min_size=10, max_group=5):
        """Grow a complete tree."""
        tree = self._recursive_clustering(self.tree, n_components=n_components,
                                          min_size=min_size,
                                          max_group=max_group)
        self.tree = sorted(tree, key=lambda node: node.identifier)
        self.mapping_grow, self.labels_grow = format_label(self.get_leaves())

    def join(self, n_components=0.95, max_group=5):
        """Join elements of the tree."""
        leaves = self.get_leaves()
        siblings = self.get_siblings()
        final_cluster = self._recursive_joining(leaves, siblings,
                                                n_components, max_group)
        self.mapping_join, self.labels_join = format_label(final_cluster)

    def predict(self, new_data, step="join"):
        """Predict labels for a set of new observation."""
        if isinstance(new_data, DenseFunctionalData):
            return np.array([self._predict(obs, step) for obs in new_data])
        elif isinstance(new_data, MultivariateFunctionalData):
            return np.array([self._predict(obs, step)
                             for obs in new_data.get_obs()])
        else:
            raise TypeError("Wrong data type.")

    def predict_proba(self, new_data, step="join"):
        """Predict the probability for new obs to be in each classes."""
        if isinstance(new_data, DenseFunctionalData):
            return [self._predict_proba(obs, step) for obs in new_data]
        elif isinstance(new_data, MultivariateFunctionalData):
            return np.array([self._predict_proba(obs, step)
                             for obs in new_data.get_obs()])
        else:
            raise TypeError("Wrong data type.")

    def get_node(self, idx):
        """Get a particular node in the tree.

        Parameters
        ----------
        idx: tuple of int
            The identifier of a node.

        Returns
        -------
        node: Node
            The node which identifier `idx`.

        """
        for node in self.tree:
            if node.identifier == idx:
                return node

    def get_parent(self, node):
        """Get the parent of the node.

        Parameters
        ----------
        node: Node
            The considered node.

        Returns
        -------
        res: Node
            The parent

        """
        depth_index = node.identifier[0] - 1
        node_index = int(node.identifier[1] / 2)
        return self.get_node((depth_index, node_index))

    def get_leaves(self):
        """Get the leaves of the tree.

        Returns
        -------
        res: list of Node
            A list with only the leaf Node.

        """
        return [node for node in self.tree if node.is_leaf]

    def get_siblings(self):
        """Get the siblings in the tree.

        A siblings couple is defined as a pair of nodes that are leaf node and
        share the same parent node.

        Returns
        -------
        list_siblings: list of tuple
            A list of tuple where each tuple represent a siblings couple.

        """
        return set([(self.get_node(node.identifier),
                     self.get_node((node.identifier[0],
                                    node.identifier[1] + 1)))
                    for node in self.tree
                    if node.is_leaf and node.identifier[1] % 2 == 0])

    def plot(self, fig=None, **plt_kwargs):
        """Plot the tree.

        Parameters
        ----------
        fig: matplotlib.figure.Figure
            A matplotlib Figure object.
        **plt_kwargs:
            Keywords plotting arguments

        """
        if fig is None:
            fig = plt.figure(constrained_layout=True, **plt_kwargs)
        gs = fig.add_gridspec(self.height, 2**self.height)

        row_idx = 0
        col_idx = 2**self.height // 2 - 1
        for node in self.tree:
            if node.identifier[0] > row_idx:
                row_idx += 1
                if node.identifier[1] == 0:
                    col_idx = 2**(fcubt.height - row_idx) // 2 - 1
                else:
                    col_idx = 2**(node.identifier[1] + 1) + 1
            if not row_idx == (self.height - 1):
                ax = fig.add_subplot(gs[row_idx, col_idx:(col_idx + 2)])
                col_idx += 2**(self.height - row_idx)
            else:
                col_idx = 2 * node.identifier[1]
                ax = fig.add_subplot(gs[row_idx, col_idx:(col_idx + 2)])
            node.plot(axes=ax)

    def _recursive_clustering(self, list_nodes, n_components=0.95,
                              min_size=10, max_group=5):
        """Perform the binary clustering recursively."""
        tree = []
        for node in list_nodes:
            if node is not None:
                tree.append(node)
                node.split(splitting_criteria='bic', n_components=n_components,
                           min_size=min_size, max_group=max_group)
                tree.extend(self._recursive_clustering(
                    [node.left, node.right],
                    n_components=n_components,
                    min_size=min_size,
                    max_group=max_group))
        return tree

    def _recursive_joining(self, list_nodes, siblings, n_components=0.95,
                           max_group=5):
        """Perform the joining recursively.

        Parameters
        ----------
        list_nodes: list of Nodes
            List of nodes to consider for the joining.
        siblings: set of tuples
            Set of tuples where each tuple contains two siblings nodes.
        n_components: int or float, default=0.95
            Number of components to keep for the aggregation of the nodes.
        max_group: int, default=5
            Number of models to try to split the data.

        Returns
        -------
        nodes: list of Nodes
            The resulting list of nodes after the joining.

        """
        new_list_nodes = joining_step(list_nodes, siblings, n_components,
                                      max_group)
        if len(new_list_nodes) == len(list_nodes):
            return new_list_nodes
        else:
            return self._recursive_joining(new_list_nodes, siblings,
                                           n_components, max_group)

    def _map_grow_join(self):
        """Map results from grow to join step."""
        mapping = {}
        for node1 in self.mapping_grow.keys():
            for node2 in self.mapping_join.keys():
                if isinstance(node2.identifier, tuple):
                    if node1.identifier == node2.identifier:
                        mapping[node1] = node2
                elif isinstance(node2.identifier, list):
                    if node1.identifier in node2.identifier:
                        mapping[node1] = node2
                else:
                    raise TypeError("Wrong identifier type.")
        return mapping

    def _predict(self, new_data, step="join"):
        """Predict the label for a new observation.

        Parameters
        ----------
        new_data: Functional data
            The new data to predict the cluster.
        step: str, default="join"
            At which step should we predict the label. Should be 'grow' or
            'join'.

        Returns
        -------
        label: int
            The label of the prediction.

        """
        node = self.root_node
        while not node.is_leaf:
            node = node.predict(new_data)

        if step == "grow":
            return self.mapping_grow[node]
        elif step == "join":
            map_grow_join = self._map_grow_join()
            return self.mapping_join[map_grow_join[node]]
        else:
            raise ValueError("Wrong step value.")

    def _predict_proba(self, new_data, step='join'):
        """Predict the probability for each class for a new observation.

        Parameters
        ----------
        new_data: Functional data
            The new data to predict the cluster.
        step: str, default="join"
            At which step should we predict the label. Should be 'grow' or
            'join'.

        Returns
        -------
        proba: dict
            A dictionary containing the probablity to belong to each class for
            the new observation.

        """
        # Compute conditional probabilities
        proba_cond = {self.root_node: 1.0}
        for node in self.tree:
            if not node.is_leaf:
                pred = node.predict_proba(new_data)
                proba_cond[node.left] = pred[0, 0]
                proba_cond[node.right] = pred[0, 1]

        # Compute probabilities to be in each class (after grow step)
        proba_grow = {}
        for leaf in self.get_leaves():
            proba = proba_cond[leaf]
            parent = self.get_parent(leaf)
            while parent is not None:
                proba *= proba_cond[parent]
                parent = self.get_parent(parent)
            proba_grow[leaf] = proba

        if step == "grow":
            return proba_grow
        elif step == "join":
            proba_join = {}
            for node in self.mapping_join:
                proba_node = 0
                if isinstance(node.identifier, tuple):
                    proba_node = proba_grow[node]
                elif isinstance(node.identifier, list):
                    for idx in node.identifier:
                        current_node = self.get_node(idx)
                        proba_node += proba_grow[current_node]
                else:
                    raise TypeError("Wrong identifier type.")
                proba_join[node] = proba_node
            return proba_join
        else:
            raise ValueError("Wrong step value.")
