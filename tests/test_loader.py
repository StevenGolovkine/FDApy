#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the loader functions.

Written with the help of ChatGPT.

"""
import pandas as pd
import numpy as np
import unittest

from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
from FDApy.representation.values import DenseValues, IrregularValues
from FDApy.representation.functional_data import (
    DenseFunctionalData, IrregularFunctionalData
)
from FDApy.misc.loader import read_csv, _read_csv_dense, _read_csv_irregular


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
        self.expected_argvals = IrregularArgvals({
            0: DenseArgvals({'input_dim_0': np.array([0, 1, 2])}),
            1: DenseArgvals({'input_dim_0': np.array([0, 2])}),
            2: DenseArgvals({'input_dim_0': np.array([1, 2])}),
            3: DenseArgvals({'input_dim_0': np.array([0, 1, 2])})
        })
        self.expected_values = IrregularValues({
            0: np.array([1, 5, 9]),
            1: np.array([2, 10]),
            2: np.array([7, 11]),
            3: np.array([4, 8, 12])
        })

    def test_read_csv_irregular(self):
        # read csv file
        obj = _read_csv_irregular(self.data, self.argvals)

        # check if object is of type IrregularFunctionalData
        self.assertIsInstance(obj, IrregularFunctionalData)

        # check if the argvals match
        np.testing.assert_equal(obj.argvals, self.expected_argvals)

        # check if the values match
        np.testing.assert_allclose(obj.values, self.expected_values)


class TestReadCsv(unittest.TestCase):
    def setUp(self):
        # create a test csv file
        self.dense_data = {
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        }
        self.dense_df = pd.DataFrame(self.dense_data)
        self.dense_df.to_csv('dense_test.csv', index=False)

        self.irregular_data = {
            'x': [1, 2, np.nan, 4],
            'y': [2, np.nan, 6, 8]
        }
        self.irregular_df = pd.DataFrame(self.irregular_data)
        self.irregular_df.to_csv('irregular_test.csv', index=False)

    def test_read_csv_dense(self):
        dense_obj = read_csv('dense_test.csv')
        self.assertIsInstance(dense_obj, DenseFunctionalData)
        np.testing.assert_array_equal(dense_obj.argvals['input_dim_0'], [0, 1])
        np.testing.assert_array_equal(dense_obj.values, self.dense_df.values)

    def test_read_csv_irregular(self):
        irregular_obj = read_csv('irregular_test.csv')
        self.assertIsInstance(irregular_obj, IrregularFunctionalData)
        np.testing.assert_array_equal(
            irregular_obj.argvals[0]['input_dim_0'], np.array([0, 1])
        )
        np.testing.assert_array_equal(
            irregular_obj.argvals[1]['input_dim_0'], np.array([0])
        )
        np.testing.assert_array_equal(
            irregular_obj.argvals[2]['input_dim_0'], np.array([1])
        )
        np.testing.assert_array_equal(
            irregular_obj.argvals[3]['input_dim_0'], np.array([0, 1])
        )

        np.testing.assert_array_equal(
            irregular_obj.values[0], np.array([1, 2])
        )
        np.testing.assert_array_equal(
            irregular_obj.values[1], np.array([2])
        )
        np.testing.assert_array_equal(
            irregular_obj.values[2], np.array([6])
        )
        np.testing.assert_array_equal(
            irregular_obj.values[3], np.array([4, 8])
        )

    def tearDown(self):
        # delete test csv files
        import os
        os.remove('dense_test.csv')
        os.remove('irregular_test.csv')
