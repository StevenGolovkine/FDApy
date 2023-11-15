#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the utils.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from numpy import AxisError

from FDApy.misc.utils import (
    _cartesian_product,
    _col_mean,
    _col_var,
    _eigh,
    _compute_covariance,
    _get_axis_dimension,
    _get_dict_dimension,
    _get_obs_shape,
    _inner_product,
    _integrate,
    _integration_weights,
    _outer,
    _normalization,
    _row_mean,
    _row_var,
    _shift,
    _standardization,
    _select_number_eigencomponents,
    _compute_eigen,
    _estimate_noise_variance
)


class TestNormalization(unittest.TestCase):
    def test_normalization_default(self):
        x = np.array([0, 5, 10])
        expected = np.array([0., 0.5, 1.])
        result = _normalization(x)
        self.assertTrue(np.allclose(result, expected))

    def test_normalization_with_min_max(self):
        x = np.array([0, 5, 10])
        max_x = 10
        min_x = 0
        expected = np.array([0., 0.5, 1.])
        result = _normalization(x, max_x=max_x, min_x=min_x)
        self.assertTrue(np.allclose(result, expected))

    def test_normalization_with_nan_min_max(self):
        x = np.array([0, 5, 10])
        max_x = None
        min_x = None
        expected = np.array([0., 0.5, 1.])
        result = _normalization(x, max_x=max_x, min_x=min_x)
        self.assertTrue(np.allclose(result, expected))

    def test_normalization_with_zero_range(self):
        x = np.array([5, 5, 5])
        expected = np.array([0., 0., 0.])
        result = _normalization(x)
        self.assertTrue(np.allclose(result, expected))


class TestStandardization(unittest.TestCase):
    def test_standardization(self):
        x = np.array([0, 5, 10])
        expected = np.array([-1.22474487, 0., 1.22474487])
        result = _standardization(x)
        self.assertTrue(np.allclose(result, expected))

    def test_standardization_with_zero_mean_std(self):
        x = np.array([5, 5, 5])
        expected = np.array([0., 0., 0.])
        result = _standardization(x)
        self.assertTrue(np.allclose(result, expected))

    def test_standardization_with_negative_value(self):
        x = np.array([-5, 5, 10])
        expected = np.array([-1.33630621, 0.26726124, 1.06904497])
        result = _standardization(x)
        self.assertTrue(np.allclose(result, expected))


class TestRowMean(unittest.TestCase):
    def test_row_mean(self):
        x = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        expected = np.array([1., 2., 3.])
        result = _row_mean(x)
        self.assertTrue(np.allclose(result, expected))

    def test_row_mean_with_negative_values(self):
        x = np.array(
            [
                [-1., 2., 3.],
                [1., -2., 3.],
                [1., 2., -3.]
            ]
        )
        expected = np.array([0.33333333, 0.66666667, 1.])
        result = _row_mean(x)
        self.assertTrue(np.allclose(result, expected))

    def test_row_mean_with_different_shapes(self):
        x = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        y = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        expected_x = np.mean(x, axis=0)
        expected_y = np.mean(y, axis=0)
        result_x = _row_mean(x)
        result_y = _row_mean(y)
        self.assertTrue(np.allclose(result_x, expected_x))
        self.assertTrue(np.allclose(result_y, expected_y))


class TestRowVar(unittest.TestCase):
    def test_row_var(self):
        x = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        expected = np.array([0., 0., 0.])
        result = _row_var(x)
        self.assertTrue(np.array_equal(expected, result))

    def test_row_var_unequal_values(self):
        x = np.array(
            [
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [10., 11., 12.]
            ]
        )
        expected = np.array([11.25, 11.25, 11.25])
        result = _row_var(x)
        self.assertTrue(np.array_equal(expected, result))

    def test_row_var_input_is_1d_array(self):
        x = np.array([1, 2, 3])
        expected = 0.6666666666666666
        result = _row_var(x)
        self.assertTrue(np.array_equal(expected, result))


class TestColMean(unittest.TestCase):
    def test_col_mean(self):
        x = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        expected = np.array([2., 2., 2., 2.])
        result = _col_mean(x)
        self.assertTrue(np.array_equal(expected, result))

    def test_col_mean_unequal_values(self):
        x = np.array(
            [
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [10., 11., 12.]
            ]
        )
        expected = np.array([2., 5., 8., 11.])
        result = _col_mean(x)
        self.assertTrue(np.array_equal(expected, result))

    def test_col_mean_input_is_1d_array(self):
        x = np.array([1, 2, 3])
        self.assertRaises(AxisError, _col_mean, x)


class TestColVar(unittest.TestCase):
    def test_col_var(self):
        x = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        expected = np.array([0.66666667, 0.66666667, 0.66666667, 0.66666667])
        result = _col_var(x)
        self.assertTrue(np.allclose(expected, result, rtol=1e-6))

    def test_col_var_unequal_values(self):
        x = np.array(
            [
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [10., 11., 12.]
            ]
        )
        expected = np.array([0.66666667, 0.66666667, 0.66666667, 0.66666667])
        result = _col_var(x)
        self.assertTrue(np.allclose(expected, result, rtol=1e-6))

    def test_col_var_input_is_1d_array(self):
        x = np.array([1, 2, 3])
        self.assertRaises(AxisError, _col_var, x)


class TestGetAxisDimension(unittest.TestCase):
    def test_default_axis(self):
        x = np.array([[1, 2], [4, 5], [7, 8]])
        result = _get_axis_dimension(x)
        self.assertEqual(result, 3)

    def test_specified_axis(self):
        x = np.array([[1, 2], [4, 5], [7, 8]])
        result = _get_axis_dimension(x, axis=1)
        self.assertEqual(result, 2)

    def test_empty_array(self):
        x = np.array([])
        result = _get_axis_dimension(x)
        self.assertEqual(result, 0)


class TestGetDictDimension(unittest.TestCase):
    def test_get_dict_dimension(self):
        x = {
            'a': np.array([1, 2, 3]),
            'b': np.array([4, 5])
        }
        self.assertEqual(_get_dict_dimension(x), (3, 2))

    def test_get_dict_dimension_unequal_size(self):
        x = {
            'a': np.array([1, 2, 3]),
            'b': np.array([[4, 5], [6, 7]])
        }
        self.assertEqual(_get_dict_dimension(x), (3, 2))

    def test_get_dict_dimension_same_size(self):
        x = {
            'a': np.array([1, 2, 3]),
            'b': np.array([[4, 5], [6, 7], [8, 9]])
        }
        self.assertEqual(_get_dict_dimension(x), (3, 3))

    def test_get_dict_dimension_multiple_obs(self):
        x = {
            'a': np.array([1, 2, 3]),
            'b': np.array([4, 5]),
            'c': np.array([[1, 2, 3], [4, 5, 6]])
        }
        self.assertEqual(_get_dict_dimension(x), (3, 2, 2))


class TestGetObsShape(unittest.TestCase):
    def test_get_obs_shape(self):
        x = {
            'a': {0: np.array([1, 2, 3]), 1: np.array([4, 5])},
            'b': {0: np.array([1, 2]), 1: np.array([3, 4])}
        }
        obs_0 = _get_obs_shape(x, 0)
        obs_1 = _get_obs_shape(x, 1)

        self.assertEqual(obs_0, (3, 2))
        self.assertEqual(obs_1, (2, 2))


class TestShift(unittest.TestCase):
    def test_shift_positive_num(self):
        x = np.array([1, 2, 3, 4, 5])
        num = 2
        fill_value = np.nan
        expected = np.array([np.nan, np.nan, 1., 2., 3.])
        result = _shift(x, num, fill_value)
        self.assertTrue(np.array_equal(result, expected, equal_nan=True))

    def test_shift_negative_num(self):
        x = np.array([1, 2, 3, 4, 5])
        num = -2
        fill_value = np.nan
        expected = np.array([3., 4., 5., np.nan, np.nan])
        result = _shift(x, num, fill_value)
        self.assertTrue(np.array_equal(result, expected, equal_nan=True))

    def test_shift_zero_num(self):
        x = np.array([1, 2, 3, 4, 5])
        num = 0
        fill_value = np.nan
        expected = np.array([1, 2, 3, 4, 5])
        result = _shift(x, num, fill_value)
        self.assertTrue(np.array_equal(result, expected))


class TestInnerProduct(unittest.TestCase):
    def test_inner_product(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        t = np.linspace(0, 1, 3)
        expected_output = 10.5
        self.assertAlmostEqual(expected_output, _inner_product(x, y, t))

    def test_inner_product_unequal_shape(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6, 7])
        self.assertRaises(ValueError, _inner_product, x, y)

    def test_inner_product_no_t(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        expected_output = 10.5
        self.assertAlmostEqual(expected_output, _inner_product(x, y))
    
    def test_inner_product_2d(self):
        X = np.array([[1, 2], [4, 5], [7, 8]])
        Y = np.array([[4, 5], [7, 8], [1, 2]])
        expected_output = 43.25
        self.assertAlmostEqual(expected_output, _inner_product(X, Y))

    def test_inner_product_3d(self):
        X = np.array([[[1, 2], [4, 5]], [[7, 8], [3, 4]], [[1, 2], [4, 5]]])
        Y = np.array([[[7, 8], [3, 4]], [[1, 2], [4, 5]], [[4, 5], [1, 2]]])
        expected_output = 24.125
        self.assertAlmostEqual(expected_output, _inner_product(X, Y))


class TestOuterFunction(unittest.TestCase):
    def test_outer(self):
        x = np.array([1, 2, 3])
        y = np.array([-1, 2])
        expected_output = np.array([[-1, 2], [-2, 4], [-3, 6]])
        np.testing.assert_array_equal(_outer(x, y), expected_output)

    def test_outer_with_negative_input(self):
        x = np.array([-1, -2, -3])
        y = np.array([-1, -2])
        expected_output = np.array([[1, 2], [2, 4], [3, 6]])
        np.testing.assert_array_equal(_outer(x, y), expected_output)

    def test_outer_with_zero_input(self):
        x = np.array([0, 0, 0])
        y = np.array([0, 0])
        expected_output = np.array([[0, 0], [0, 0], [0, 0]])
        np.testing.assert_array_equal(_outer(x, y), expected_output)


class TestIntegrate(unittest.TestCase):
    def test_integrate_simpson(self):
        X = np.array([1, 2, 4])
        Y = np.array([1, 4, 16])
        result = _integrate(Y, X)
        self.assertEqual(result, 21.0)

    def test_integrate_trapeze(self):
        X = np.array([1, 2, 4])
        Y = np.array([1, 4, 16])
        result = _integrate(Y, X, method='trapz')
        self.assertEqual(result, 22.5)

    def test_integrate_method_error(self):
        X = np.array([1, 2, 4])
        Y = np.array([1, 4, 16])
        with self.assertRaises(ValueError):
            _integrate(Y, X, method='error')


class TestIntegrate2D(unittest.TestCase):
    def test_integrate_simpson(self):
        X = np.array([1, 2, 4])
        Y = np.array([1, 2])
        Z = np.array([[1, 2], [4, 5], [7, 8]])
        result = _integrate(Z, X, Y)
        self.assertEqual(result, 15.75)

    def test_integrate_trapeze(self):
        X = np.array([1, 2, 4])
        Y = np.array([1, 2])
        Z = np.array([[1, 2], [4, 5], [7, 8]])
        result = _integrate(Z, X, Y, method='trapz')
        self.assertEqual(result, 15.0)

    def test_integrate_method_error(self):
        X = np.array([1, 2, 4])
        Y = np.array([1, 2])
        Z = np.array([[1, 2], [4, 5], [7, 8]])
        with self.assertRaises(ValueError):
            _integrate(Z, X, Y, method='error')


class TestIntegrationWeights(unittest.TestCase):
    def test_trapz(self):
        x = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([0.5, 1., 1., 1., 0.5])
        np.testing.assert_array_equal(
            _integration_weights(x, method='trapz'),
            expected_output
        )

    def test_simpson(self):
        x = np.array([1, 2, 3, 4, 5])
        expected_output = np.array(
            [0.33333333, 1.33333333, 0.66666667, 1.33333333, 0.33333333]
        )
        np.testing.assert_array_almost_equal(
            _integration_weights(x, method='simpson'),
            expected_output
        )

    def test_callable_method(self):
        def custom_weights(x: np.ndarray) -> np.ndarray:
            return x

        x = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(
            _integration_weights(x, method=custom_weights),
            expected_output
        )

    def test_invalid_method(self):
        x = np.array([1, 2, 3, 4, 5])
        with self.assertRaises(NotImplementedError) as cm:
            _integration_weights(x, method='invalid_method')
        self.assertEqual(
            str(cm.exception), 'invalid_method not implemented!',
            "NotImplementedError should be raised for invalid method"
        )


class TestSelectNumberComponents(unittest.TestCase):
    def test_integer_input(self):
        eigenvalues = np.array([0.1, 0.2, 0.3, 0.4])
        num_eigen = 2
        result = _select_number_eigencomponents(eigenvalues, num_eigen)
        self.assertEqual(result, num_eigen)

    def test_float_input(self):
        eigenvalues = np.array([0.1, 0.2, 0.3, 0.4])
        percent = 0.5
        result = _select_number_eigencomponents(eigenvalues, percent)
        self.assertEqual(result, 3)

    def test_None_input(self):
        eigenvalues = np.array([0.1, 0.2, 0.3, 0.4])
        result = _select_number_eigencomponents(eigenvalues)
        self.assertEqual(result, len(eigenvalues))

    def test_unexpected_input(self):
        eigenvalues = np.array([0.1, 0.2, 0.3, 0.4])
        percent = 1.75
        with self.assertRaises(ValueError):
            _select_number_eigencomponents(eigenvalues, percent)


class TestComputeCovariance(unittest.TestCase):
    def test_compute_covariance(self):
        eigenvalues = np.array([1, 2, 3], dtype=np.float64)
        eigenfunctions = np.array(
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5]
            ], dtype=np.float64
        )

        expected_covariance = np.array(
            [
                [ 36.,  50.,  64.],
                [ 50.,  70.,  90.],
                [ 64.,  90., 116.]
            ], dtype=np.float64
        )

        result = _compute_covariance(eigenvalues, eigenfunctions)
        np.testing.assert_array_equal(result, expected_covariance)

    def test_input_shape_mismatch(self):
        eigenvalues = np.array([1, 2, 3], dtype=np.float64)
        eigenfunctions = np.array(
            [[1, 2, 3], [2, 3, 4]], dtype=np.float64
        )
        with self.assertRaises(ValueError):
            _compute_covariance(eigenvalues, eigenfunctions)


class TestEigh(unittest.TestCase):
    def test_eigh(self):
        matrix = np.array([[26, 18], [18, 74]])
        eig_val, eig_vec = _eigh(matrix)

        expected_val = np.array([80., 20.])
        expected_vec = np.array([
            [0.31622777, 0.9486833],
            [0.9486833 , 0.31622777]
        ])

        np.testing.assert_array_almost_equal(eig_val, expected_val)
        np.testing.assert_array_almost_equal(np.abs(eig_vec), expected_vec)
        np.testing.assert_array_almost_equal(
            np.matmul(np.matmul(eig_vec, np.diag(eig_val)), eig_vec.T),
            matrix
        )


class TestCartesianProduct(unittest.TestCase):
    def test_single_array(self):
        result = _cartesian_product(np.array([1, 2]))
        expected = np.array([[1], [2]])
        np.testing.assert_array_equal(result, expected)

    def test_multiple_arrays(self):
        result = _cartesian_product(np.array([0, 1]), np.array([1, 2, 3]))
        expected = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])
        np.testing.assert_array_equal(result, expected)

    def test_multiple_arrays_different_sizes(self):
        result = _cartesian_product(
            np.array([0, 1]), np.array([1, 2]), np.array([2, 3])
        )
        expected = np.array([
            [0, 1, 2], [0, 1, 3], [0, 2, 2], [0, 2, 3],
            [1, 1, 2], [1, 1, 3], [1, 2, 2], [1, 2, 3]
        ])
        np.testing.assert_array_equal(result, expected)


class TestComputeEigen(unittest.TestCase):
    def test_computeeigen(self):
        matrix = np.array([[26, 18], [18, 74]])
        eig_val, eig_vec = _compute_eigen(matrix, 1)

        expected_val = np.array([80.])
        expected_vec = np.array([[0.31622777], [0.9486833 ]])

        np.testing.assert_array_almost_equal(eig_val, expected_val)
        np.testing.assert_array_almost_equal(np.abs(eig_vec), expected_vec)


class TestEstimateNoiseVariance(unittest.TestCase):
    def test_estimate_noise_variance(self):
        x = np.array([ 0.60968595,  0.42687124, -0.00694508, -0.24967824, -0.63924524, -0.155665  ,  0.45404848, -0.03835127, -0.30465133, -0.11510283, -0.60347426, -0.35214373, -0.29066131, -0.35554762, -0.17912665, -0.29495087, -0.23515825, -0.54776153, -0.19909394, -0.0385301 , -0.61965528, -0.59832842, -0.09553887, -0.42882287, -0.06147952, -0.46134948, -0.02184155, -0.30052157, -0.25294966,  0.06024915, -0.34776689, -0.45297977, -0.31705891, -0.08850962, -0.51147256,  0.01054835, -0.22216309,  0.18608676, -0.57413947, -0.18228223, -0.35070697, -0.12524521,  0.14817193,  0.08605626,  0.10274285,  0.15687236,  0.27068695,  0.03197991,  0.44725924,  0.46838016,  0.43646785,  0.50402285,  0.47398237,  0.8896721 ,  0.44072296,  0.41565703,  0.33837318,  0.29688513,  0.26844346,  0.36503099,  0.5629462 ,  0.66627843,  0.61341452,  0.4388009 ,  0.81669395,  0.78347266,  0.58234238,  0.46900352,  0.73152057,  0.64161388,  0.26314481,  0.55568595,  0.60984322,  0.32674417,  0.54322836,  0.144386  ,  0.73332652,  0.67426011,  0.29509681,  0.3847471 , -0.28804456, -0.06506755,  0.06566893, -0.17544926,  0.15190063,  0.36145407, -0.43087144, -0.44308208, -0.29677284, -0.08844341, -0.82711264, -0.60910319, -0.37459305, -1.27442565, -1.32106897, -1.26528966, -1.4294778 , -1.28054123, -1.56417686, -2.17799607, -1.83578592])
        
        results = _estimate_noise_variance(x, 2)
        expected_results = 0.05999266
        np.testing.assert_almost_equal(results, expected_results)

        results = _estimate_noise_variance(x, 5)
        expected_results = 0.07193388
        np.testing.assert_almost_equal(results, expected_results)

        with self.assertRaises(ValueError):
            _estimate_noise_variance(x, 15)
