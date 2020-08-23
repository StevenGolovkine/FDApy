#!/usr/bin/python3.7
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
        self.assertTrue(np.allclose(data.argvals['input_dim_0'],
                                    np.array([0, 1, 2, 3])))

    def test_load_irregular(self):
        data = read_csv(os.path.join(DATA, 'irregular.csv'), index_col=0)
        self.assertEqual(data.n_dim, 1)
        self.assertEqual(data.n_obs, 3)
        self.assertTrue(np.allclose(data.argvals['input_dim_0'][0],
                                    np.array([-2, 0, 3])))


if __name__ == '__main__':
    unittest.main()
