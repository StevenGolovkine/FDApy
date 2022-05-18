#!/usr/bin/python3
# -*-coding:utf8 -*

import numpy as np
import unittest

from FDApy.misc.utils import (
    col_mean_,
    col_var_,
    get_axis_dimension_,
    integrate_,
    integration_weights_,
    outer_,
    range_standardization_,
    row_mean_,
    row_var_,
    shift_
)


class TestUtils(unittest.TestCase):
    """Test class for the functions in utils.py"""

    def test_range_standardization(self):
        X = np.array([0, 5, 10])
        r_ = range_standardization_(X)
        self.assertTrue(
            np.array_equal(r_, np.array([0., 0.5, 1.]))
        )

    def test_row_mean(self):
        X = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        mean_ = row_mean_(X)
        self.assertTrue(
            np.array_equal(mean_, np.array([1., 2., 3.]))
        )

    def test_row_var(self):
        X = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        var_ = row_var_(X)
        self.assertTrue(
            np.array_equal(var_, np.array([0., 0., 0.]))
        )

    def test_col_mean(self):
        X = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        mean_ = col_mean_(X)
        self.assertTrue(
            np.array_equal(mean_, np.array([2., 2., 2., 2.]))
        )

    def test_col_var(self):
        X = np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
        var_ = col_var_(X)
        self.assertTrue(
            np.allclose(var_, np.array([2 / 3, 2 / 3, 2 / 3, 2 / 3]))
        )

    def test_shift_(self):
        X = np.array([1, 2, 3, 4, 5])
        S = shift_(X, num=2, fill_value=0)
        self.assertTrue(np.array_equal(S, np.array([0., 0., 1., 2., 3.])))
        S = shift_(X, num=0, fill_value=0)
        self.assertTrue(np.array_equal(S, np.array([1., 2., 3., 4., 5.])))
        S = shift_(X, num=-2, fill_value=0)
        self.assertTrue(np.array_equal(S, np.array([3., 4., 5., 0., 0.])))

    def test_tensor_product_(self):
        X = np.array([1, 2, 3])
        Y = np.array([-1, 2])
        tens_ = outer_(X, Y)
        self.assertTrue(
            np.array_equal(tens_, np.array([[-1, 2], [-2, 4], [-3, 6]]))
        )

    def test_integrate_(self):
        X = np.array([1, 2, 4])
        Y = np.array([1, 4, 16])
        self.assertEqual(
            integrate_(X, Y), 21.0
        )

    def test_integration_weights_(self):
        X = np.array([1, 2, 3, 4, 5])
        W = integration_weights_(X, method='trapz')
        self.assertTrue(np.array_equal(W, np.array([0.5, 1., 1., 1., 0.5])))
        W = integration_weights_(X, method=lambda x: x)
        self.assertTrue(np.array_equal(W, np.array([1, 2, 3, 4, 5])))

    def test_get_axis_dimension_(self):
        X = np.array([1, 2, 3, 4, 5])
        self.assertEqual(
            get_axis_dimension_(X, 0), 5
        )


class TestUtilsFails(unittest.TestCase):
    """Fail test class for the functions in utils.py"""

    def test_integrate_(self):
        X = np.array([1, 2, 4])
        Y = np.array([1, 4, 16])
        with self.assertRaises(ValueError) as cm:
            integrate_(X, Y, "failed")
        self.assertTrue('Simpsons' in str(cm.exception))

    def test_integration_weights_(self):
        X = np.array([1, 2, 3, 4, 5])
        with self.assertRaises(NotImplementedError) as cm:
            integration_weights_(X, method='failed')
        self.assertTrue('implemented' in str(cm.exception))


if __name__ == '__main__':
    #unittest.main()
    print('COucou')
