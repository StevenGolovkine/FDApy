#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for FCUBT.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.functional_data import (
    DenseFunctionalData
)
from FDApy.clustering.fcubt import (
    _Node,
    FCUBT
)


class NodeTest(unittest.TestCase):
    def setUp(self):
        self.argvals = {'input_dim_0': np.array([1, 2, 3, 4, 5])}
        self.values = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        self.data = DenseFunctionalData(self.argvals, self.values)

        self.identifier = (0, 0)
        self.idx_obs = np.arange(self.data.n_obs)
        self.is_root = False
        self.is_leaf = False
        self.normalize = False
        self.node = _Node(
            data=self.data,
            identifier=self.identifier,
            idx_obs=self.idx_obs,
            is_root=self.is_root,
            is_leaf=self.is_leaf,
            normalize=self.normalize
        )

    def test_data(self):
        self.assertEqual(self.node.data, self.data)

        new_data = self.data + self.data
        self.node.data = new_data
        self.assertEqual(self.node.data, new_data)

    def test_identifier(self):
        self.assertEqual(self.node.identifier, self.identifier)

        new_identifier = (1, 1)
        self.node.identifier = new_identifier
        self.assertEqual(self.node.identifier, new_identifier)

    def test_labels_grow(self):
        labels_grow = np.array([1, 2, 3])
        self.node.labels_grow = labels_grow
        np.testing.assert_array_equal(self.node.labels_grow, labels_grow)

    def test_labels_join(self):
        labels_join = np.array([4, 5, 6])
        self.node.labels_join = labels_join
        np.testing.assert_array_equal(self.node.labels_join, labels_join)

    def test_is_root(self):
        self.assertEqual(self.node.is_root, self.is_root)

        new_is_root = True
        self.node.is_root = new_is_root
        self.assertEqual(self.node.is_root, new_is_root)

    def test_is_leaf(self):
        self.assertEqual(self.node.is_leaf, self.is_leaf)

        new_is_leaf = True
        self.node.is_leaf = new_is_leaf
        self.assertEqual(self.node.is_leaf, new_is_leaf)


class FCUBTTest(unittest.TestCase):
    def setUp(self):
        self.root_node = _Node(...)
        self.normalize = False
        self.fcubt = FCUBT(
            root_node=self.root_node,
            normalize=self.normalize
        )
