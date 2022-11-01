#!/usr/bin/python3
# -*-coding:utf8 -*

import numpy as np
import os
import unittest

from FDApy.misc.loader import read_csv


DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestLoaderCSV(unittest.TestCase):
    """Test class for the csv loader."""

    def test_load_dense(self):
        data = read_csv(os.path.join(DATA, 'dense.csv'), index_col=0)
        self.assertEqual(data.n_dim, 1)
        self.assertEqual(data.n_obs, 3)
        self.assertTrue(
            np.allclose(data.argvals['input_dim_0'], np.array([0, 1, 2, 3]))
        )
        self.assertTrue(
            np.allclose(
                data.values,
                np.array(
                    [
                        [4.6, 4.7, 4.2, 3.6],
                        [4.6, 4.1, 4.1, 4.0],
                        [4.7, 4.4, 4.4, 4.0]
                    ]
                )
            )
        )

    def test_load_irregular(self):
        data = read_csv(os.path.join(DATA, 'irregular.csv'), index_col=0)
        self.assertEqual(data.n_dim, 1)
        self.assertEqual(data.n_obs, 3)
        self.assertTrue(
            np.allclose(data.argvals['input_dim_0'][0], np.array([-2, 0, 3]))
        )
        self.assertTrue(
            np.allclose(
                data.values[0],
                np.array([548, 605, 893])
            )
        )
