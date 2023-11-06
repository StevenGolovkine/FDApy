#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for MultivariateFunctionalData.

Written with the help of ChatGPT.

"""
import numpy as np
import pandas as pd
import unittest

from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
from FDApy.representation.values import DenseValues, IrregularValues
from FDApy.representation.functional_data import (
    DenseFunctionalData,
    MultivariateFunctionalData
)

from FDApy.simulation.karhunen import KarhunenLoeve


class MultivariateFunctionalDataTest(unittest.TestCase):
    def setUp(self):
        self.argvals = {'input_dim_0': np.array([1, 2, 3, 4, 5])}
        self.values = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])

        self.fdata1 = DenseFunctionalData(DenseArgvals(self.argvals), DenseValues(self.values))
        self.fdata2 = DenseFunctionalData(DenseArgvals(self.argvals), DenseValues(self.values))
        self.fdata3 = DenseFunctionalData(DenseArgvals(self.argvals), DenseValues(self.values))
        self.multivariate_data = MultivariateFunctionalData([self.fdata1, self.fdata2])

    def test_init(self):
        self.assertEqual(len(self.multivariate_data), 2)
        self.assertIsInstance(self.multivariate_data.data[0], DenseFunctionalData)
        self.assertIsInstance(self.multivariate_data.data[1], DenseFunctionalData)
        
        values = np.array([
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        fdata = DenseFunctionalData(DenseArgvals(self.argvals), DenseValues(values))
        with self.assertRaises(ValueError):
            MultivariateFunctionalData([self.fdata1, fdata])

    def test_repr(self):
        expected_repr = f"Multivariate functional data object with 2 functions of 3 observations."
        actual_repr = repr(self.multivariate_data)
        self.assertEqual(actual_repr, expected_repr)

    def test_getitem(self):
        fdata = self.multivariate_data[0]

        np.testing.assert_array_equal(fdata.data[0].values, self.fdata1[0].values)
        np.testing.assert_array_equal(fdata.data[1].values, self.fdata2[0].values)

    def test_n_obs(self):
        expected_n_obs = 3
        actual_n_obs = self.multivariate_data.n_obs
        self.assertEqual(actual_n_obs, expected_n_obs)

    def test_n_functional(self):
        expected_n_functional = 2
        actual_n_functional = self.multivariate_data.n_functional
        self.assertEqual(actual_n_functional, expected_n_functional)

    def test_n_dimension(self):
        expected_n_dimension = [1, 1]
        actual_n_dimension = self.multivariate_data.n_dimension
        self.assertEqual(actual_n_dimension, expected_n_dimension)

    def test_n_points(self):
        expected_n_points = [(5, ), (5, )]
        actual_n_points = self.multivariate_data.n_points
        self.assertEqual(actual_n_points, expected_n_points)

    def test_append(self):
        res = MultivariateFunctionalData([])

        res.append(self.fdata1)
        np.testing.assert_equal(res.n_functional, 1)

        res.append(self.fdata2)
        np.testing.assert_equal(res.n_functional, 2)

    def test_extend(self):
        self.multivariate_data.extend([self.fdata1, self.fdata3])
        np.testing.assert_equal(self.multivariate_data.n_functional, 4)

    def test_insert(self):
        self.multivariate_data.insert(1, self.fdata3)
        np.testing.assert_equal(self.multivariate_data.n_functional, 3)
        np.testing.assert_equal(self.multivariate_data.data[1], self.fdata3)

    def test_remove(self):
        self.multivariate_data.remove(self.fdata1)
        np.testing.assert_equal(self.multivariate_data.n_functional, 1)
        np.testing.assert_equal(self.multivariate_data.data[0], self.fdata2)

    def test_pop(self):
        popped_data = self.multivariate_data.pop(0)
        np.testing.assert_equal(self.multivariate_data.n_functional, 1)
        np.testing.assert_equal(popped_data, self.fdata1)

    def test_clear(self):
        self.multivariate_data.clear()
        np.testing.assert_equal(self.multivariate_data.n_functional, 0)

    def test_reverse(self):
        self.multivariate_data.reverse()
        np.testing.assert_equal(self.multivariate_data.data[0], self.fdata2)
        np.testing.assert_equal(self.multivariate_data.data[1], self.fdata1)
    
    def test_to_long(self):
        fdata_long = self.multivariate_data.to_long()
        np.testing.assert_array_equal(len(fdata_long), 2)
        self.assertIsInstance(fdata_long[0], pd.DataFrame)
        self.assertIsInstance(fdata_long[1], pd.DataFrame)


class TestSmoothMultivariateFunctionalData(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=50)
        kl.add_noise_and_sparsify(0.05, 0.5)
        
        fdata_1 = kl.data
        fdata_2 = kl.noisy_data
        self.fdata = MultivariateFunctionalData([fdata_1, fdata_2])

    def test_smooth(self):
        fdata_smooth = self.fdata.smooth()

        self.assertIsInstance(fdata_smooth, MultivariateFunctionalData)
        self.assertIsInstance(fdata_smooth.data[0], DenseFunctionalData)
        self.assertIsInstance(fdata_smooth.data[1], DenseFunctionalData)
        np.testing.assert_equal(fdata_smooth.n_functional, 2)

    def test_error_list(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        with self.assertRaises(TypeError):
            self.fdata.smooth(
                points=points,
                kernel_name=['epanechnikov', 'epanechnikov'],
                bandwidth=[0.05, 0.05],
                degree=[1, 1]
            )

    def test_error_length_list(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        with self.assertRaises(ValueError):
            self.fdata.smooth(
                points=[points, points, points],
                kernel_name=['epanechnikov', 'epanechnikov'],
                bandwidth=[0.05, 0.05],
                degree=[1, 1]
            )


class TestMeanhMultivariateFunctionalData(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=50)
        kl.add_noise_and_sparsify(0.05, 0.5)
        
        fdata_1 = kl.data
        fdata_2 = kl.noisy_data
        self.fdata = MultivariateFunctionalData([fdata_1, fdata_2])

    def test_mean(self):
        fdata_smooth = self.fdata.mean()

        self.assertIsInstance(fdata_smooth, MultivariateFunctionalData)
        self.assertIsInstance(fdata_smooth.data[0], DenseFunctionalData)
        self.assertIsInstance(fdata_smooth.data[1], DenseFunctionalData)
        np.testing.assert_equal(fdata_smooth.n_functional, 2)

    def test_error_list(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        with self.assertRaises(TypeError):
            self.fdata.mean(points=points)

    def test_error_length_list(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        with self.assertRaises(ValueError):
            self.fdata.mean(points=[points, points, points])


class TestInnerProductMultivariateFunctionalData(unittest.TestCase):
    def setUp(self) -> None:
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=4)
        kl.add_noise_and_sparsify(0.05, 0.5)

        fdata_1 = kl.data
        fdata_2 = kl.sparse_data
        self.fdata = MultivariateFunctionalData([fdata_1, fdata_2])

    def test_inner_prod(self):
        res = self.fdata.inner_product()
        expected_res = np.array([[ 0.58532546,  0.19442368, -0.04038602,  0.01705178],[ 0.19442368,  0.38395264, -0.45055398,  0.10919059],[-0.04038602, -0.45055398,  0.96833672, -0.07948717],[ 0.01705178,  0.10919059, -0.07948717,  0.18026045]])
        np.testing.assert_array_almost_equal(res, expected_res)


class TestInnerProductMultivariateFunctionalData(unittest.TestCase):
    def setUp(self) -> None:
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=4)
        kl.add_noise_and_sparsify(0.05, 0.5)

        fdata_1 = kl.data
        fdata_2 = kl.sparse_data
        self.fdata = MultivariateFunctionalData([fdata_1, fdata_2])

    def test_norm(self):
        res = self.fdata.norm()
        expected_res = np.array([1.05384959, 0.84700578, 1.37439764, 0.59235447])
        np.testing.assert_array_almost_equal(res, expected_res)
