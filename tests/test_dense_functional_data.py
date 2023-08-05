#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the functional_data.py
file.

Written with the help of ChatGPT.

"""
import numpy as np
import pandas as pd
import unittest

from FDApy.representation.functional_data import (
    FunctionalData,
    DenseFunctionalData,
    IrregularFunctionalData
)
from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
from FDApy.representation.values import DenseValues, IrregularValues

from FDApy.simulation.karhunen import KarhunenLoeve


class TestDenseFunctionalData(unittest.TestCase):
    def setUp(self):
        self.argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 5])})
        self.values = DenseValues(np.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10],[11, 12, 13, 14, 15]]))
        self.func_data = DenseFunctionalData(self.argvals, self.values)

        argvals = IrregularArgvals({
            0: DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4])}),
            1: DenseArgvals({'input_dim_0': np.array([2, 4])})
        })
        values = IrregularValues({0: np.array([1, 6, 9, 4]), 1: np.array([2, 3])})
        self.irreg_data = IrregularFunctionalData(argvals, values)

    def test_repr(self):
        expected_repr = "Functional data object with 3 observations on a 1-dimensional support."
        actual_repr = repr(self.func_data)
        self.assertEqual(actual_repr, expected_repr)

    def test_getitem_dense_functional_data(self):
        data = self.func_data[1]
        expected_argvals = DenseArgvals(self.argvals)
        expected_values = DenseValues(np.array([[6, 7, 8, 9, 10]]))
        np.testing.assert_array_equal(data.argvals, expected_argvals)
        np.testing.assert_array_equal(data.values, expected_values)

    def test_argvals_property(self):
        argvals = self.func_data.argvals
        self.assertEqual(argvals, DenseArgvals(self.argvals))

    def test_argvals_setter(self):
        new_argvals = DenseArgvals({'x': np.linspace(0, 5, 5)})
        self.func_data.argvals = new_argvals
        self.assertEqual(self.func_data._argvals, DenseArgvals(new_argvals))

        expected_argvals_stand = {"x": np.linspace(0, 1, 5)}
        np.testing.assert_array_almost_equal(self.func_data._argvals_stand['x'], expected_argvals_stand['x'])

        with self.assertRaises(TypeError):
            self.func_data.argvals = 0

    def test_argvals_stand_setter(self):
        new_argvals = DenseArgvals({'x': np.linspace(0, 1, 5)})
        self.func_data.argvals_stand = new_argvals
        self.assertEqual(self.func_data._argvals_stand, new_argvals)

        with self.assertRaises(TypeError):
            self.func_data.argvals_stand = 0

    def test_values_property(self):
        dense_values = self.func_data.values
        np.testing.assert_array_equal(dense_values, self.values)

    def test_values_setter(self):
        new_values = DenseValues(np.array([[11, 12, 13, 14, 15]]))
        self.func_data.values = new_values
        np.testing.assert_array_equal(self.func_data.values, new_values)

        with self.assertRaises(TypeError):
            self.func_data.values = 0

    def test_n_points(self):
        expected_result = (5,)
        result = self.func_data.n_points
        np.testing.assert_equal(result, expected_result)

    def test_is_compatible(self):
        DenseFunctionalData._is_compatible(self.func_data, self.func_data)
        self.assertTrue(True)

    def test_non_compatible_type(self):
        with self.assertRaises(TypeError):
            DenseFunctionalData._is_compatible(self.func_data, self.irreg_data)

    def test_non_compatible_nobs(self):
        argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 5])})
        values = DenseValues(np.array([[1, 2, 3, 4, 5]]))
        func_data = DenseFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            DenseFunctionalData._is_compatible(self.func_data, func_data)

    def test_non_compatible_ndim(self):
        argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4]), 'input_dim_1': np.array([5, 6, 7])})
        values = DenseValues(np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],[[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]],[[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]]))
        func_data = DenseFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            DenseFunctionalData._is_compatible(self.func_data, func_data)

    def test_non_compatible_argvals_equality(self):
        argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 6])})
        values = DenseValues(np.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10],[11, 12, 13, 14, 15]]))
        func_data = DenseFunctionalData(argvals, values)
        with self.assertRaises(ValueError):
            DenseFunctionalData._is_compatible(self.func_data, func_data)

    def test_concatenate(self):
        fdata = DenseFunctionalData.concatenate(self.func_data, self.func_data)

        expected_argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 5])})
        expected_values = DenseValues(np.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10],[11, 12, 13, 14, 15], [1, 2, 3, 4, 5],[6, 7, 8, 9, 10],[11, 12, 13, 14, 15]]))
        
        self.assertIsInstance(fdata, DenseFunctionalData)
        self.assertEqual(fdata.argvals, expected_argvals)
        np.testing.assert_allclose(fdata.values, expected_values)

    def test_to_long(self):
        result = self.func_data.to_long()

        expected_dim = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        expected_id = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        expected_values = DenseValues(np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]))

        self.assertTrue(isinstance(result, pd.DataFrame))
        np.testing.assert_array_equal(result['input_dim_0'].values, expected_dim)
        np.testing.assert_array_equal(result['id'].values, expected_id)
        np.testing.assert_array_equal(result['values'].values, expected_values)

    def test_inner_product(self):
        result = self.func_data.inner_product()
        expected = np.array([[42., 102., 162.],[102., 262., 422.],[162., 422., 682.]])
        np.testing.assert_array_almost_equal(result, expected)


class TestPerformComputation(unittest.TestCase):
    def setUp(self):
        self.argvals1 = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 5])})
        self.values1 = DenseValues(np.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10],[11, 12, 13, 14, 15]]))
        self.func_data1 = DenseFunctionalData(self.argvals1, self.values1)

        self.argvals2 = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4, 5])})
        self.values2 = DenseValues(np.array([[6, 7, 8, 9, 10],[11, 12, 13, 14, 15],[1, 2, 3, 4, 5]]))
        self.func_data2 = DenseFunctionalData(self.argvals2, self.values2)

    def test_addition(self):
        result = self.func_data1 + self.func_data2

        expected_values = np.array([[7, 9, 11, 13, 15],[17, 19, 21, 23, 25],[12, 14, 16, 18, 20]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_substraction(self):
        result = self.func_data1 - self.func_data2

        expected_values = np.array([[-5, -5, -5, -5, -5],[-5, -5, -5, -5, -5],[10, 10, 10, 10, 10]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_multiplication(self):
        result = self.func_data1 * self.func_data2

        expected_values = np.array([[6, 14, 24, 36, 50],[66, 84, 104, 126, 150],[11, 24, 39, 56, 75]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_right_multiplication(self):
        result = FunctionalData.__rmul__(self.func_data1, self.func_data2)

        expected_values = np.array([[6, 14, 24, 36, 50],[66, 84, 104, 126, 150],[11, 24, 39, 56, 75]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_equal(result.values, expected_values)

    def test_true_divide(self):
        result = self.func_data1 / self.func_data2

        expected_values = np.array([[0.16666667, 0.28571429, 0.375, 0.44444444, 0.5],[0.54545455, 0.58333333, 0.61538462, 0.64285714, 0.66666667],[11., 6., 4.33333333, 3.5, 3.]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_almost_equal(result.values, expected_values)

    def test_floor_divide(self):
        result = self.func_data1 // self.func_data2

        expected_values = np.array([ [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [11, 6, 4, 3, 3]])
        self.assertEqual(result.argvals, DenseArgvals(self.argvals1))
        np.testing.assert_array_almost_equal(result.values, expected_values)


class TestDenseFunctionalData2D(unittest.TestCase):
    """Test class for the class DenseFunctionalData in two dimension."""

    def setUp(self):
        argvals = DenseArgvals({'input_dim_0': np.array([1, 2, 3, 4]), 'input_dim_1': np.array([5, 6, 7])})

        values = DenseValues(np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]], [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]]))
        self.dense_fd = DenseFunctionalData(argvals, values)

    def test_argvals_stand(self):
        is_equal_dim0 = np.allclose(self.dense_fd.argvals_stand['input_dim_0'], np.array([0., 0.33333333, 0.66666667, 1.]))
        is_equal_dim1 = np.allclose(self.dense_fd.argvals_stand['input_dim_1'], np.array([0., 0.5, 1.]))
        self.assertTrue(is_equal_dim0 and is_equal_dim1)

    def test_n_obs(self):
        self.assertEqual(self.dense_fd.n_obs, 3)

    def test_n_dimension(self):
        self.assertEqual(self.dense_fd.n_dimension, 2)

    def test_subset(self):
        new_dense_fd = self.dense_fd[2]
        self.assertIsInstance(new_dense_fd, DenseFunctionalData)
        self.assertEqual(new_dense_fd.n_obs, 1)
        new_dense_fd = self.dense_fd[1:3]
        self.assertIsInstance(new_dense_fd, DenseFunctionalData)
        self.assertEqual(new_dense_fd.n_obs, 2)

    def test_is_compatible(self):
        DenseFunctionalData._is_compatible(self.dense_fd, self.dense_fd)
        self.assertTrue(True)

    def test_to_long(self):
        result = self.dense_fd.to_long()

        expected_dim_0 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        expected_dim_1 = np.array([5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7])
        expected_id = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        expected_values = DenseValues(np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]))

        self.assertTrue(isinstance(result, pd.DataFrame))
        np.testing.assert_array_equal(result['input_dim_0'].values, expected_dim_0)
        np.testing.assert_array_equal(result['input_dim_1'].values, expected_dim_1)
        np.testing.assert_array_equal(result['id'].values, expected_id)
        np.testing.assert_array_equal(result['values'].values, expected_values)

    def test_mean(self):
        mean_fd = self.dense_fd.mean()
        is_equal = np.allclose(mean_fd.values, np.array([[[3., 4., 5.],[3., 4., 5.],[3., 4., 5.],[3., 4., 5.]]]))
        self.assertTrue(is_equal)


class TestSmoothDense(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5

        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=1)
        kl.add_noise(0.05)
        self.fdata_1d = kl.noisy_data

        kl = KarhunenLoeve(
            basis_name=name, dimension='2D', n_functions=n_functions, argvals=np.linspace(0, 1, 11), random_state=42
        )
        kl.new(n_obs=1)
        kl.add_noise(0.01)
        self.fdata_2d = kl.noisy_data

    def test_smooth_1d(self):
        fdata_smooth = self.fdata_1d.smooth(bandwidth=0.39)

        expected_values = DenseValues([[-0.01606076, -0.03927974, -0.06065247, -0.08064861, -0.0985722 , -0.11473298, -0.12873838, -0.14080426, -0.15110408, -0.15899639, -0.16518377, -0.16975994, -0.17282827, -0.17456334, -0.17516743, -0.17438072, -0.17238781, -0.16916585, -0.1648771 , -0.15955995, -0.15321236, -0.14583429, -0.13743539, -0.12818324, -0.11818117, -0.10732236, -0.09581323, -0.08391604, -0.07171098, -0.05925097, -0.04634749, -0.03312852, -0.01955276, -0.00591993,  0.00780961,  0.02165301,  0.03532213,  0.04888939,  0.06201074,  0.07464752,  0.08733756,  0.10009191,  0.11305935,  0.12596814,  0.13861317,  0.15135273,  0.16403323,  0.17641953,  0.18845338,  0.19979805,  0.21011149,  0.21918813,  0.22666992,  0.23287177,  0.23754429,  0.24084159,  0.24224017,  0.24215612,  0.24053099,  0.23680987,  0.2311225 ,  0.22317144,  0.21296134,  0.20239486,  0.19123498,  0.17965291,  0.16691815,  0.15285536,  0.13727036,  0.12033908,  0.10204399,  0.08203094,  0.06007568,  0.0361132 ,  0.01032256, -0.01737561, -0.04710686, -0.07910889, -0.11332353, -0.14983385, -0.18869856, -0.22996097, -0.2737618 , -0.32018954, -0.36936081, -0.42137813, -0.47639927, -0.53431152, -0.59520363, -0.65906517, -0.72582823, -0.7955359 , -0.86888845, -0.94518609, -1.02528767, -1.10784452, -1.19331881, -1.28199362, -1.37433521, -1.47077994, -1.5719026 ]])
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)

    def test_smooth_default(self):
        fdata_smooth = self.fdata_1d.smooth()

        expected_values = DenseValues([[-0.02429559, -0.04692012, -0.06808662, -0.08733992,  -0.10476349, -0.1201033 , -0.13343605, -0.14495936,  -0.15414689, -0.16142441, -0.16703569, -0.17107945,  -0.17372825, -0.17523312, -0.17537045, -0.17426835,  -0.17187959, -0.1683614 , -0.16379691, -0.15820227,  -0.15157594, -0.14392271, -0.13534245, -0.12595229,  -0.11572434, -0.10476306, -0.09329625, -0.08145468,  -0.06930329, -0.05673707, -0.04379459, -0.03048125,  -0.01698946, -0.0033696 ,  0.01040116,  0.02413118,   0.03780046,  0.05119333,  0.06420193,  0.07674581,   0.08876092,  0.10089324,  0.11315879,  0.12540077,   0.1375912 ,  0.14958905,  0.16156369,  0.17322856,   0.18431632,  0.19486971,  0.20436068,  0.21247915,   0.21925548,  0.22452416,  0.22863351,  0.23118409,   0.23201188,  0.23131111,  0.22901164,  0.22483542,   0.21841862,  0.2097182 ,  0.20024912,  0.19029286,   0.17965316,  0.16825192,  0.15605014,  0.14251845,   0.12750402,  0.11093134,  0.09292191,  0.07337491,   0.05199697,  0.02862551,  0.00326567, -0.02398934,  -0.05322993, -0.08460388, -0.11824667, -0.15415084,  -0.19238895, -0.23301713, -0.27611908, -0.32180786,  -0.37019188, -0.42136926, -0.4754639 , -0.53257764,  -0.5925813 , -0.65556993, -0.72151025, -0.79035115,  -0.86230983, -0.93781306, -1.01645216, -1.09861987,  -1.18328633, -1.27092377, -1.36188424, -1.45665473,  -1.55572334]])
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)

    def test_smooth_2d(self):
        fdata_smooth = self.fdata_2d.smooth(bandwidth=0.38321537573562725)

        expected_values = DenseValues([[[-0.05768065, -0.0325109 , -0.06377169, -0.11902475,  -0.17539351, -0.21983762, -0.22573974, -0.1952416 ,  -0.14636823, -0.10568896, -0.06846036], [-0.01858612,  0.03244905,  0.03064698,  0.01112608,  -0.01460841, -0.04053729, -0.0538999 , -0.05060879,  -0.0388745 , -0.03297848, -0.03478178], [-0.0303675 ,  0.01263408,  0.02215056,  0.02518468,   0.02582411,  0.02440345,  0.02398765,  0.02398748,   0.02317225,  0.00998423, -0.01973678], [-0.06516004, -0.050214  , -0.03416902, -0.01639152,   0.00427802,  0.02776454,  0.04433002,  0.05106246,   0.04656269,  0.02427716, -0.02014263], [-0.08086579, -0.11474459, -0.10810421, -0.08228783,  -0.04735196, -0.00128506,  0.03349477,  0.05161253,   0.04796107,  0.02510488, -0.02057925], [-0.10070974, -0.17777287, -0.18883362, -0.16396666,  -0.11985526, -0.0588545 , -0.00549451,  0.0265492 ,   0.03468447,  0.02015175, -0.01974093], [-0.115502  , -0.21084839, -0.23343946, -0.21371272,  -0.16915681, -0.10150224, -0.03817066, -0.00283201,   0.01216235,  0.0081192 , -0.01957933], [-0.11096745, -0.20223145, -0.2246809 , -0.20963719,  -0.17194025, -0.1109024 , -0.05456805, -0.018249  ,   0.00167824,  0.00171127, -0.01970314], [-0.08745132, -0.15750614, -0.17562421, -0.16628127,  -0.13837972, -0.09199249, -0.05055231, -0.01918915,   0.00227075,  0.00525019, -0.00971076], [-0.03383798, -0.07811742, -0.09357282, -0.09088518,  -0.0769251 , -0.05160711, -0.02848511, -0.01031309,   0.00795577,  0.0168141 ,  0.00998392], [ 0.02666408,  0.02305734,  0.02007509,  0.01847483,   0.01348127,  0.01012247,  0.00818805,  0.00686963,   0.0237614 ,  0.03540203,  0.03374264]]])
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)
    