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


class TestMeanDense(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5

        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=100)
        kl.add_noise(0.01)
        self.fdata_1d = kl.noisy_data

        kl = KarhunenLoeve(
            basis_name=name, dimension='2D', n_functions=n_functions, argvals=np.linspace(0, 1, 11), random_state=42
        )
        kl.new(n_obs=100)
        kl.add_noise(0.01)
        self.fdata_2d = kl.noisy_data

    def test_mean_1d(self):
        mean_estim = self.fdata_1d.mean(smooth=True)

        expected_values = DenseValues([[-2.89510087e-02, -2.84966006e-02, -2.80277897e-02,  -2.75571007e-02, -2.70707065e-02, -2.65946676e-02,  -2.61154844e-02, -2.56446279e-02, -2.51691430e-02,  -2.46795364e-02, -2.41632414e-02, -2.36161209e-02,  -2.30587527e-02, -2.24769509e-02, -2.18733334e-02,  -2.12527572e-02, -2.06218056e-02, -1.99817249e-02,  -1.93322802e-02, -1.86763582e-02, -1.80150675e-02,  -1.73479164e-02, -1.66756597e-02, -1.59979237e-02,  -1.53187348e-02, -1.46377275e-02, -1.39589073e-02,  -1.32807693e-02, -1.25987710e-02, -1.19135027e-02,  -1.12269601e-02, -1.05492184e-02, -9.87721975e-03,  -9.21186608e-03, -8.55931265e-03, -7.91852183e-03,  -7.28695482e-03, -6.67136606e-03, -6.06740413e-03,  -5.47794925e-03, -4.89590572e-03, -4.33320612e-03,  -3.78894543e-03, -3.27508901e-03, -2.77807744e-03,  -2.29359607e-03, -1.84662306e-03, -1.43621396e-03,  -1.06586590e-03, -7.16091014e-04, -3.91791095e-04,  -9.63348214e-05,  1.65023989e-04,  3.91513004e-04,   5.68140224e-04,  7.05506959e-04,  8.11510557e-04,   8.89686175e-04,  9.25040827e-04,  9.21009189e-04,   8.61807439e-04,  7.57249814e-04,  6.39597288e-04,   5.10193398e-04,  3.72799038e-04,  2.06866438e-04,   9.29496642e-06, -2.09345795e-04, -4.60866716e-04,  -7.53112016e-04, -1.09083478e-03, -1.47034691e-03,  -1.89596711e-03, -2.36939488e-03, -2.88716224e-03,  -3.44969251e-03, -4.05568907e-03, -4.71677563e-03,  -5.42461593e-03, -6.18166226e-03, -6.98976418e-03,  -7.84965754e-03, -8.76026281e-03, -9.72051812e-03,  -1.07286755e-02, -1.17850898e-02, -1.28899016e-02,  -1.40493862e-02, -1.52667846e-02, -1.65453683e-02,  -1.78788269e-02, -1.92681721e-02, -2.07367336e-02,  -2.22556961e-02, -2.38214258e-02, -2.54236284e-02,  -2.70829180e-02, -2.88008027e-02, -3.05398148e-02,  -3.22992662e-02, -3.40935325e-02]])
        np.testing.assert_array_almost_equal(mean_estim.values, expected_values)

    def test_smooth_2d(self):
        new_argvals = DenseArgvals({
            'input_dim_0': np.linspace(0, 1, 21),
            'input_dim_1': np.linspace(0, 1, 21)
        })
        fdata_smooth = self.fdata_2d.mean(points=new_argvals, smooth=True)

        expected_values = DenseValues([[[-1.09648983e-02, -1.51424915e-02, -1.84846107e-02,   -2.11174547e-02, -2.35252176e-02, -2.49128945e-02,   -2.55278912e-02, -2.53604866e-02, -2.44490518e-02,   -2.23932346e-02, -2.01508424e-02, -1.74703914e-02,   -1.45793545e-02, -1.17216297e-02, -8.66068286e-03,   -5.56714167e-03, -2.46616213e-03,  9.82305668e-05,    1.98652689e-03,  3.72331775e-03,  5.69114278e-03],  [-8.16705872e-03, -1.18133517e-02, -1.43537236e-02,   -1.66222017e-02, -1.85009051e-02, -1.96076299e-02,   -2.00608225e-02, -1.99139133e-02, -1.89786311e-02,   -1.72456970e-02, -1.54905066e-02, -1.33885673e-02,   -1.10009588e-02, -8.47456692e-03, -5.77415895e-03,   -3.11459994e-03, -7.99660917e-04,  1.05068783e-03,    2.48569240e-03,  3.56733719e-03,  5.32280358e-03],  [-6.85869048e-03, -9.57969546e-03, -1.16139544e-02,   -1.33635777e-02, -1.47953957e-02, -1.56634701e-02,   -1.60006906e-02, -1.57469173e-02, -1.46766869e-02,   -1.30042498e-02, -1.13981181e-02, -9.67076029e-03,   -7.89298464e-03, -5.83275746e-03, -3.70061679e-03,   -1.58495314e-03,  1.52550364e-04,  1.48692728e-03,    2.41278827e-03,  3.30727040e-03,  4.50094966e-03],  [-6.08340642e-03, -8.11594128e-03, -9.72385928e-03,   -1.10433885e-02, -1.21616324e-02, -1.28083377e-02,   -1.28582672e-02, -1.25411842e-02, -1.12253130e-02,   -9.65539081e-03, -8.15785282e-03, -6.76222128e-03,   -5.42061853e-03, -3.76373909e-03, -2.07366212e-03,   -4.13933525e-04,  8.13222838e-04,  1.75102715e-03,    2.19262978e-03,  2.74006254e-03,  3.64072234e-03],  [-5.63346731e-03, -7.09554462e-03, -8.22264309e-03,   -9.24742173e-03, -1.00549864e-02, -1.04574841e-02,   -1.03912915e-02, -9.81121926e-03, -8.63378494e-03,   -7.20342610e-03, -5.86741282e-03, -4.74776718e-03,   -3.58252825e-03, -2.14756342e-03, -6.97349771e-04,    6.95021099e-04,  1.60062604e-03,  2.08504406e-03,    2.18389628e-03,  2.33688718e-03,  2.85382882e-03],  [-5.62265328e-03, -6.49930466e-03, -7.16039889e-03,   -7.80997169e-03, -8.30485705e-03, -8.55143729e-03,   -8.37434472e-03, -7.71577222e-03, -6.52094292e-03,   -5.17863037e-03, -4.12481783e-03, -3.17208795e-03,   -2.11524299e-03, -8.24453470e-04,  4.59773448e-04,    1.60910804e-03,  2.24000501e-03,  2.36298502e-03,    2.25972494e-03,  2.19448756e-03,  2.60034585e-03],  [-5.80595610e-03, -6.12303263e-03, -6.40677681e-03,   -6.77570897e-03, -6.97425268e-03, -6.89048243e-03,   -6.51279888e-03, -5.84952333e-03, -4.81326552e-03,   -3.62199669e-03, -2.72927062e-03, -1.84290744e-03,   -8.14819425e-04,  2.82864417e-04,  1.20170564e-03,    2.09389712e-03,  2.54723363e-03,  2.63083369e-03,    2.42345749e-03,  2.19210207e-03,  2.49208305e-03],  [-5.65480219e-03, -5.59294038e-03, -5.72090519e-03,   -5.93100085e-03, -5.89691445e-03, -5.57443433e-03,   -5.11742967e-03, -4.41494620e-03, -3.44721876e-03,   -2.44563129e-03, -1.67549781e-03, -8.64043217e-04,    1.77947490e-04,  9.71810703e-04,  1.61282351e-03,    2.38021977e-03,  2.73442213e-03,  2.70963842e-03,    2.31917509e-03,  1.89912858e-03,  1.98618228e-03],  [-5.27618326e-03, -5.22829454e-03, -5.21767722e-03,   -5.28578366e-03, -5.24253419e-03, -4.91182235e-03,   -4.44897859e-03, -3.83480506e-03, -2.86832132e-03,   -1.94132859e-03, -1.13973606e-03, -3.35681445e-04,    4.45180399e-04,  1.08444043e-03,  1.70315758e-03,    2.31544112e-03,  2.44525969e-03,  2.30906381e-03,    1.91466954e-03,  1.46490730e-03,  1.27131017e-03],  [-4.56632090e-03, -4.70532501e-03, -4.84070850e-03,   -4.86838877e-03, -4.82719441e-03, -4.69694506e-03,   -4.25055903e-03, -3.55431894e-03, -2.68016735e-03,   -1.81121032e-03, -1.13226373e-03, -4.44975185e-04,    2.46015572e-04,  8.42184388e-04,  1.34176997e-03,    1.70096117e-03,  1.87858434e-03,  1.72721970e-03,    1.47064214e-03,  1.17352669e-03,  7.65312566e-04],  [-3.59397500e-03, -4.03240010e-03, -4.32267315e-03,   -4.41976344e-03, -4.38415894e-03, -4.28546709e-03,   -3.92016187e-03, -3.39837575e-03, -2.88688916e-03,   -2.18630689e-03, -1.39735227e-03, -7.57858342e-04,   -3.36510755e-04,  1.93427948e-04,  5.63503036e-04,    8.43418817e-04,  1.17174669e-03,  1.26961616e-03,    1.09684816e-03,  9.01852418e-04,  5.42596378e-04],  [-2.85840798e-03, -3.51868383e-03, -3.96160849e-03,   -4.09912047e-03, -4.05604589e-03, -4.00219688e-03,   -3.91745645e-03, -3.71634764e-03, -3.37391355e-03,   -2.59705524e-03, -1.93526460e-03, -1.37274235e-03,   -1.03659514e-03, -5.24729450e-04, -3.70098045e-04,   -4.42089606e-05,  3.72107330e-04,  6.80241489e-04,    7.16731208e-04,  6.10696824e-04,  8.46051131e-05],  [-2.31413307e-03, -3.06443577e-03, -3.63289810e-03,   -3.86322321e-03, -4.02055816e-03, -4.06427718e-03,   -4.05393124e-03, -4.04306595e-03, -3.80695308e-03,   -3.25992900e-03, -2.71411992e-03, -2.36339449e-03,   -1.97813048e-03, -1.52227784e-03, -1.20824800e-03,   -8.93200228e-04, -5.62683517e-04, -1.53951914e-04,    1.24120467e-04,  1.47376440e-04, -2.94870148e-04],  [-1.85354141e-03, -2.73413733e-03, -3.38163824e-03,   -3.65415545e-03, -3.94610414e-03, -4.02172936e-03,   -4.13674409e-03, -4.31182135e-03, -4.29155118e-03,   -4.09945717e-03, -3.54160898e-03, -3.46068185e-03,   -2.96438522e-03, -2.37079870e-03, -1.90675907e-03,   -1.89947414e-03, -1.57291598e-03, -9.29077291e-04,   -3.88813329e-04, -9.28576305e-05, -2.27165241e-04],  [-1.34336346e-03, -2.47243820e-03, -3.11819185e-03,   -3.40595535e-03, -3.66751789e-03, -3.92504010e-03,   -4.15191196e-03, -4.40767375e-03, -4.63457505e-03,   -4.60748734e-03, -4.40397171e-03, -4.22087860e-03,   -3.74713037e-03, -3.08712185e-03, -2.78411596e-03,   -2.78881800e-03, -2.35263426e-03, -1.59709239e-03,   -8.21489845e-04, -2.57788221e-04, -1.71271710e-04],  [-7.47246240e-04, -2.07343210e-03, -2.83651237e-03,   -3.16926128e-03, -3.44387005e-03, -3.72577673e-03,   -4.05918192e-03, -4.40470622e-03, -4.88519870e-03,   -5.16045813e-03, -5.33692013e-03, -4.99837321e-03,   -4.48902576e-03, -3.90634812e-03, -3.67966016e-03,   -3.55255938e-03, -3.10650782e-03, -2.31512639e-03,   -1.49803897e-03, -7.68847093e-04, -5.14495039e-04],  [-5.22431502e-04, -1.89617556e-03, -2.78716834e-03,   -3.20736697e-03, -3.47497623e-03, -3.76265996e-03,   -4.02910711e-03, -4.42863317e-03, -5.01788639e-03,   -5.59863741e-03, -5.87772805e-03, -5.59760225e-03,   -5.11309571e-03, -4.73766513e-03, -4.52956735e-03,   -4.33218047e-03, -3.91131273e-03, -3.18323539e-03,   -2.34830012e-03, -1.48445846e-03, -1.01552628e-03],  [-7.46633582e-04, -1.94514158e-03, -2.93199849e-03,   -3.41174909e-03, -3.61463958e-03, -3.86355609e-03,   -4.12162223e-03, -4.49547183e-03, -5.02200604e-03,   -5.71752754e-03, -6.05654359e-03, -5.81367309e-03,   -5.44141329e-03, -5.21339929e-03, -5.12021548e-03,   -5.07513873e-03, -4.64567837e-03, -3.99193478e-03,   -3.11732872e-03, -2.14779844e-03, -1.47157209e-03],  [-1.36562475e-03, -2.16559500e-03, -3.08087686e-03,   -3.55206790e-03, -3.73177706e-03, -3.89497693e-03,   -4.10466480e-03, -4.39654672e-03, -4.83630899e-03,   -5.45897737e-03, -5.81989634e-03, -5.72742337e-03,   -5.44578849e-03, -5.23910468e-03, -5.09311736e-03,   -5.06407475e-03, -4.73269842e-03, -4.19507852e-03,   -3.37181987e-03, -2.43230537e-03, -1.65632301e-03],  [-2.86667391e-03, -2.67011926e-03, -3.47208882e-03,   -3.95225001e-03, -4.00843539e-03, -3.89742966e-03,   -3.85794907e-03, -3.95158504e-03, -4.26519665e-03,   -4.73798219e-03, -5.03912326e-03, -5.03184321e-03,   -4.82811247e-03, -4.61405363e-03, -4.54962843e-03,   -4.50209116e-03, -4.22236250e-03, -3.76034007e-03,   -3.03143583e-03, -2.28237207e-03, -1.61365455e-03],  [-5.73185521e-03, -4.78614511e-03, -4.64510571e-03,   -4.63273609e-03, -4.41528808e-03, -4.00742245e-03,   -3.50642377e-03, -3.18785069e-03, -3.25952733e-03,   -3.43019301e-03, -3.49380548e-03, -3.48319386e-03,   -3.37949245e-03, -3.10048400e-03, -3.20264632e-03,   -3.27282046e-03, -3.14168995e-03, -2.78524746e-03,   -2.33490595e-03, -1.78886882e-03, -8.41166135e-04]]])
        np.testing.assert_array_almost_equal(fdata_smooth.values, expected_values)


class TestCenterDense(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5

        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=10)
        kl.add_noise(0.01)
        self.fdata_1d = kl.noisy_data

        kl = KarhunenLoeve(
            basis_name=name, dimension='2D', n_functions=n_functions, argvals=np.linspace(0, 1, 11), random_state=42
        )
        kl.new(n_obs=10)
        kl.add_noise(0.01)
        self.fdata_2d = kl.noisy_data

    def test_center_1d(self):
        fdata_center = self.fdata_1d.center()

        expected_values = DenseValues([ 0.09486033,  0.05795609, -0.21693676, -0.16425338, -0.23551089, -0.30386333, -0.31432715, -0.17966429, -0.45361599, -0.30378468, -0.59842336, -0.48920438, -0.46117416, -0.4368309 , -0.43877363, -0.44157163, -0.56347882, -0.57935433, -0.448765  , -0.55219987, -0.65633849, -0.63512938, -0.60424434, -0.45067569, -0.47194751, -0.40081153, -0.49423574, -0.41539686, -0.34662515, -0.41639104, -0.31452747, -0.39974496, -0.34193612, -0.31472021, -0.36600409, -0.16668374, -0.23050021, -0.14982844, -0.06990474, -0.03969839,  0.01618227, -0.02607172, -0.02442717,  0.04396772, -0.08295989, -0.02541512,  0.02012964,  0.08522921,  0.2568354 ,  0.15746959,  0.24045551,  0.43743235,  0.30002744,  0.43635092,  0.29481898,  0.39173231,  0.33980103,  0.42176351,  0.55872689,  0.31906955,  0.55029924,  0.54353399,  0.47087264,  0.39369026,  0.55090892,  0.49341502,  0.56942195,  0.54516512,  0.69690797,  0.5033501 ,  0.41221407,  0.51637579,  0.50084557,  0.59154784,  0.51217714,  0.43353228,  0.50939729,  0.20535662,  0.21726424,  0.1413067 ,  0.14331288, -0.01154242,  0.12887491, -0.02234053, -0.21749975, -0.24722099, -0.1946089 , -0.2276154 , -0.2024555 , -0.20678721, -0.55827828, -0.80581957, -1.03288726, -0.91146486, -1.14400929, -1.23476748, -1.39116642, -1.48717114, -1.51608548, -1.76310558, -1.95756873])
        np.testing.assert_array_almost_equal(fdata_center.values[0], expected_values)

        expected_values = DenseValues([0.02072353, 0.18650396, 0.08035557, 0.00709206, 0.13414575, 0.17706249, 0.0775334 , 0.21970148, 0.20040172, 0.3944627 , 0.33091162, 0.32268732, 0.36204651, 0.41492691, 0.47251574, 0.32741173, 0.33816572, 0.56619066, 0.42641317, 0.33646664, 0.53167267, 0.54145552, 0.34861029, 0.52998972, 0.58483138, 0.55183591, 0.57969032, 0.37720821, 0.55101391, 0.48709552, 0.46774817, 0.601039  , 0.65920833, 0.68743702, 0.63568777, 0.48898606, 0.53058539, 0.56701702, 0.49728342, 0.49298242, 0.55695593, 0.46321848, 0.39071749, 0.41673676, 0.51622433, 0.44797147, 0.47680079, 0.50522092, 0.39657341, 0.50603078, 0.46516863, 0.30126768, 0.56704337, 0.3713669 , 0.25503021, 0.31555234, 0.31787633, 0.42550469, 0.3934911 , 0.38804526, 0.48475863, 0.45801275, 0.45688963, 0.46747836, 0.48925165, 0.21786505, 0.39304225, 0.21207872, 0.43406703, 0.43005818, 0.53482208, 0.54842093, 0.44894254, 0.37719829, 0.47382056, 0.37004257, 0.41965584, 0.5147725 , 0.35611557, 0.48751301, 0.4690132 , 0.29936605, 0.48502022, 0.44944407, 0.38086728, 0.28461056, 0.26413196, 0.28449075, 0.37972412, 0.21607213, 0.34043223, 0.15639897, 0.24290942, 0.25587554, 0.46975233, 0.06652941, 0.32907685, 0.17293553, 0.17363069, 0.11738811, 0.12138169])
        np.testing.assert_array_almost_equal(fdata_center.values[9], expected_values)

    def test_center_2d(self):
        fdata_center = self.fdata_2d.center()

        expected_values = DenseValues([[-0.00706571,  0.06140581, -0.25757231, -0.32157638,  -0.51041906, -0.61624207, -0.541792  , -0.23217104,  -0.30070912,  0.03235682, -0.15140494], [-0.04210601,  0.13229547,  0.19446984,  0.15855435,   0.08763887, -0.09355076, -0.13357601,  0.0049195 ,  -0.07350783, -0.14808939, -0.10661364], [-0.09126916,  0.1225272 ,  0.13707298,  0.21896103,   0.11354313,  0.1571509 ,  0.17050731,  0.03511081,   0.07422828, -0.05887118, -0.02686892], [-0.03756098, -0.18062699,  0.00400095, -0.03032467,   0.08986951,  0.18462532,  0.18114731,  0.16736899,   0.04388442, -0.02352476,  0.0038474 ], [-0.17064485, -0.35895575, -0.39037406, -0.29167893,  -0.0375032 , -0.0651889 ,  0.03401201,  0.20063649,   0.00915224,  0.09203582, -0.08290955], [-0.02480404, -0.41378124, -0.44601294, -0.27137679,  -0.39717148, -0.04774822,  0.01376973, -0.03740158,  -0.12081658,  0.02126839, -0.04298182], [ 0.0173447 , -0.3198742 , -0.26618289, -0.40858159,  -0.36688825, -0.12003219, -0.0306473 ,  0.12755587,   0.09141114,  0.0438688 ,  0.15020691], [-0.12249904, -0.30954257, -0.4226641 , -0.34288038,  -0.35596748, -0.0606841 , -0.07935771, -0.1662597 ,  -0.10248244,  0.03568785,  0.08321284], [ 0.20133436,  0.15792942, -0.14162184, -0.27295833,  -0.34220898, -0.0497358 , -0.11997438, -0.05703168,  -0.06439026, -0.0129522 ,  0.10178865], [ 0.02433392, -0.04897771, -0.15331002, -0.21868924,  -0.08933309, -0.03069341,  0.16207065,  0.00494591,   0.0961991 , -0.05056679, -0.12409665], [-0.07864005, -0.05940717,  0.22099298, -0.07957533,   0.08234863, -0.09295291,  0.08840821,  0.032303  ,  -0.01698526, -0.00099382, -0.06032354]])
        np.testing.assert_array_almost_equal(fdata_center.values[0], expected_values)

        expected_values = DenseValues([[-3.55980230e-02,  2.32213599e-01, -5.83379988e-02,   8.15300490e-02,  3.02826244e-01,  3.83297823e-01,   4.00033995e-01,  2.80295781e-01,  8.24816160e-02,   1.36654768e-01,  1.19901579e-01], [ 9.78827660e-03,  1.84473880e-01,  2.16068885e-01,   2.17975942e-01,  3.95867370e-01,  3.42992960e-01,   2.78882084e-01,  3.37647755e-01, -1.49893080e-01,  -1.17640128e-01, -1.11991963e-01], [-3.56499915e-02,  7.15258389e-02,  3.21790407e-02,   1.48814070e-01,  1.96068651e-01,  5.22222048e-01,   4.06865808e-01,  1.09791008e-01,  3.79914729e-02,  -9.41350738e-02, -1.81376573e-01], [-1.10752061e-02, -8.18885873e-03,  1.52540099e-01,   4.67587837e-02,  1.49306616e-01,  2.28212844e-01,   2.07891666e-01,  1.54053446e-01, -7.68093786e-02,   1.03751840e-01, -6.82019447e-02], [ 6.13743855e-02,  7.68906302e-02, -6.99705981e-03,   1.28053268e-01,  2.01886120e-01,  2.10997504e-01,   1.03429888e-01,  2.30541413e-01,  1.40015601e-01,  -1.08770758e-02,  8.54643953e-02], [ 1.89119059e-01,  2.54743413e-01, -2.68870844e-02,   2.04423580e-02,  2.59826286e-01, -1.03267767e-02,  -5.92309391e-02, -4.09703726e-04,  6.83407202e-02,   2.16948877e-03, -5.50889144e-02], [-7.33441677e-02, -3.18193488e-02, -2.47604319e-02,   2.99657195e-01,  9.83595720e-02,  1.43195893e-01,  -9.43542424e-03,  7.38293954e-03, -5.59594328e-02,  -2.57056884e-01, -1.33924307e-01], [-1.84684607e-01, -1.99778439e-01, -8.16852890e-02,   1.73092417e-01,  3.11692753e-02,  1.54187808e-01,   5.77073006e-02,  8.82803242e-02, -8.34916677e-02,   5.72480043e-02,  7.22716055e-02], [-5.60034551e-02,  7.28108625e-02,  7.81522634e-02,  -3.67758784e-02, -9.88686023e-02, -1.10301388e-01,   1.24911069e-02,  3.45783434e-02, -7.76204320e-02,   4.26142255e-03,  5.22443303e-02], [ 7.04481337e-02,  1.67145346e-01,  1.39378560e-01,  -9.35354996e-02,  1.70650198e-01, -5.26958944e-02,   1.02831525e-01,  4.15893728e-03, -4.17192664e-02,  -1.81226297e-01, -2.27587230e-02], [-1.38066695e-01,  1.09845358e-01,  1.59825274e-01,  -7.71277610e-02,  2.86967825e-02, -7.51993398e-02,  -6.75975629e-02,  7.12573723e-02, -5.17529870e-03,   1.77047237e-01, -4.30680144e-02]])
        np.testing.assert_array_almost_equal(fdata_center.values[9], expected_values)



class TestNormDense(unittest.TestCase):
    def setUp(self) -> None:
        name = 'bsplines'
        n_functions = 5

        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=10)
        self.fdata_1d = kl.data

        kl = KarhunenLoeve(
            basis_name=name, dimension='2D', n_functions=n_functions, argvals=np.linspace(0, 1, 11), random_state=42
        )
        kl.new(n_obs=10)
        self.fdata_2d = kl.data

    def test_norm_1d(self):
        results = self.fdata_1d.norm()
        expected_results = np.array([0.53253351, 0.42212112, 0.6709846 , 0.26672898, 0.27440755, 0.37906252, 0.65277413, 0.53998411, 0.2872874 , 0.4934973 ])

        np.testing.assert_almost_equal(results, expected_results)

    def test_norm_1d_stand(self):
        results = self.fdata_1d.norm(use_argvals_stand=True)
        expected_results = np.array([0.53253351, 0.42212112, 0.6709846 , 0.26672898, 0.27440755, 0.37906252, 0.65277413, 0.53998411, 0.2872874 , 0.4934973 ])

        np.testing.assert_almost_equal(results, expected_results)

    def test_norm_1d_sqaured(self):
        results = self.fdata_1d.norm(squared=True)
        expected_results = np.array([0.28359194, 0.17818624, 0.45022033, 0.07114435, 0.07529951, 0.14368839, 0.42611407, 0.29158284, 0.08253405, 0.24353959])

        np.testing.assert_almost_equal(results, expected_results)

    def test_norm_2d(self):
        results = self.fdata_2d.norm()
        expected_results = np.array([0.1565889 , 0.1789413 , 0.23471474, 0.11862412, 0.10131662, 0.11671813, 0.17352746, 0.12817132, 0.10216184, 0.15292851])

        np.testing.assert_almost_equal(results, expected_results)


class TestNormalizeDense(unittest.TestCase):
    def setUp(self) -> None:
        name = 'bsplines'
        n_functions = 5

        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=10)
        self.fdata_1d = kl.data

        kl = KarhunenLoeve(
            basis_name=name, dimension='2D', n_functions=n_functions, argvals=np.linspace(0, 1, 11), random_state=42
        )
        kl.new(n_obs=10)
        self.fdata_2d = kl.data

    def test_normalize_1d(self):
        results = self.fdata_1d.normalize()
        expected_results = 0.21227413

        np.testing.assert_almost_equal(results[1], expected_results)

    def test_normalize_1d_stand(self):
        results = self.fdata_1d.normalize(use_argvals_stand=True)
        expected_results = 0.21227413

        np.testing.assert_almost_equal(results[1], expected_results)

    def test_norm_2d(self):
        results = self.fdata_2d.normalize()
        expected_results = 0.02128688

        np.testing.assert_almost_equal(results[1], expected_results)


class TestCovarianceDense(unittest.TestCase):
    def setUp(self) -> None:
        name = 'bsplines'
        n_functions = 5

        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, argvals=np.linspace(0, 1, 11), random_state=42
        )
        kl.new(n_obs=100)
        kl.add_noise(0.01)
        self.fdata = kl.noisy_data

    def test_default(self):
        self.fdata.covariance()
        results = self.fdata._covariance
        expected_results = DenseValues(np.array([[[ 0.65131383,  0.42540917,  0.23474046,  0.08956508,   -0.02393091, -0.07852248, -0.09937122, -0.1140295 ,   -0.12715368, -0.14651419, -0.17284398],  [ 0.42540917,  0.36558448,  0.29831524,  0.22541171,    0.15235873,  0.0932614 ,  0.04929342,  0.01935932,   -0.00406499, -0.02490258, -0.04308851],  [ 0.23474046,  0.29831524,  0.32455262,  0.30968861,    0.26915847,  0.21990992,  0.16569595,  0.11997717,    0.08822403,  0.05841301,  0.03269999],  [ 0.08956508,  0.22541171,  0.30968861,  0.35448262,    0.34337712,  0.30502616,  0.25874022,  0.20686353,    0.16164431,  0.11931445,  0.0705793 ],  [-0.02393091,  0.15235873,  0.26915847,  0.34337712,    0.36350632,  0.35411815,  0.32830583,  0.28849332,    0.23147628,  0.16123668,  0.07676748],  [-0.07852248,  0.0932614 ,  0.21990992,  0.30502616,    0.35411815,  0.38087224,  0.38448825,  0.36470753,    0.30911065,  0.2131568 ,  0.07881052],  [-0.09937122,  0.04929342,  0.16569595,  0.25874022,    0.32830583,  0.38448825,  0.42538777,  0.43384707,    0.38081103,  0.27932027,  0.11254169],  [-0.1140295 ,  0.01935932,  0.11997717,  0.20686353,    0.28849332,  0.36470753,  0.43384707,  0.47378336,    0.44259647,  0.35753509,  0.20873423],  [-0.12715368, -0.00406499,  0.08822403,  0.16164431,    0.23147628,  0.30911065,  0.38081103,  0.44259647,    0.4658788 ,  0.42779463,  0.32931256],  [-0.14651419, -0.02490258,  0.05841301,  0.11931445,    0.16123668,  0.2131568 ,  0.27932027,  0.35753509,    0.42779463,  0.47529177,  0.48316044],  [-0.17284398, -0.04308851,  0.03269999,  0.0705793 ,    0.07676748,  0.07881052,  0.11254169,  0.20873423,    0.32931256,  0.48316044,  0.65060466]]]))
        expected_noise = 0.07383871598487848

        np.testing.assert_almost_equal(results.values, expected_results)
        np.testing.assert_almost_equal(self.fdata._noise_variance, expected_noise)

    def test_points(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 6)})
        self.fdata.covariance(points=points)

        results = self.fdata._covariance
        expected_results = DenseValues(np.array([[[ 0.65131383,  0.23474046, -0.02393091, -0.09937122,   -0.12715368, -0.17284398],  [ 0.23474046,  0.32455262,  0.26915847,  0.16569595,    0.08822403,  0.03269999],  [-0.02393091,  0.26915847,  0.36350632,  0.32830583,    0.23147628,  0.07676748],  [-0.09937122,  0.16569595,  0.32830583,  0.42538777,    0.38081103,  0.11254169],  [-0.12715368,  0.08822403,  0.23147628,  0.38081103,    0.4658788 ,  0.32931256],  [-0.17284398,  0.03269999,  0.07676748,  0.11254169,    0.32931256,  0.65060466]]]))
        expected_noise = 0.035029758855601834

        np.testing.assert_almost_equal(results.values, expected_results)
        np.testing.assert_almost_equal(self.fdata._noise_variance, expected_noise)

    def test_data2d(self):
        kl = KarhunenLoeve(
            basis_name='bsplines', dimension='2D', n_functions=5, argvals=np.linspace(0, 1, 11), random_state=42
        )
        kl.new(n_obs=10)
        fdata_2d = kl.data

        with self.assertRaises(ValueError):
            fdata_2d.covariance()
