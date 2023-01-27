#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the loader functions.

Written with the help of ChatGPT.

"""
import pandas as pd
import numpy as np
import os
import unittest

from FDApy.representation.functional_data import (
    DenseFunctionalData, IrregularFunctionalData
)
from FDApy.misc.loader import read_csv, _read_csv_dense, _read_csv_irregular


DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestReadCsvDense(unittest.TestCase):
    def test_read_csv_dense(self):
        # create test dataframe
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        argvals = np.array([1, 2])

        # read csv file
        obj = _read_csv_dense(df, argvals)

        # check if object is of type DenseFunctionalData
        self.assertIsInstance(obj, DenseFunctionalData)

        # check if the argvals match
        np.testing.assert_array_equal(obj.argvals['input_dim_0'], argvals)

        # check if the values match
        np.testing.assert_array_equal(obj.values, df.values)


class TestReadCsvIrregular(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, np.nan, 7, 8],
            'col3': [9, 10, 11, 12]
        })
        self.argvals = np.array([0, 1, 2])
        self.expected_argvals = {
            'input_dim_0': {
                0: np.array([0, 1, 2]),
                1: np.array([0, 2]),
                2: np.array([1, 2]),
                3: np.array([0, 1, 2])
            }
        }
        self.expected_values = {
            0: np.array([1, 5, 9]),
            1: np.array([2, 10]),
            2: np.array([7, 11]),
            3: np.array([4, 8, 12])
        }

    def test_read_csv_irregular(self):
        # read csv file
        obj = _read_csv_irregular(self.data, self.argvals)

        # check if object is of type IrregularFunctionalData
        self.assertIsInstance(obj, IrregularFunctionalData)

        # check if the argvals match
        np.testing.assert_equal(obj.argvals, self.expected_argvals)

        # check if the values match
        np.testing.assert_equal(obj.values, self.expected_values)


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
