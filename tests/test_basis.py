#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for Basis.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.basis import Basis


class TestBasis(unittest.TestCase):
    def setUp(self):
        self.argvals = {'input_dim_0': np.array([0, 0.5, 1])}

    def test_getter(self):
        X = Basis(name='legendre', n_functions=2, argvals=self.argvals)
        np.testing.assert_equal(X.name, 'legendre')
        np.testing.assert_equal(X.norm, False)
        np.testing.assert_equal(X.dimension, '1D')

    def test_argvals(self):
        X = Basis(name='legendre', n_functions=2, argvals=None)
        np.testing.assert_allclose(
            X.argvals['input_dim_0'],
            np.arange(0, 1.01, 0.01)
        )

    def test_basis_legendre(self):
        X = Basis(name='legendre', n_functions=2, argvals=self.argvals)
        np.testing.assert_allclose(
            X.values,
            np.array([[1., 1., 1.], [0., 0.5, 1.]])
        )

    def test_basis_legendre_norm(self):
        X = Basis(name='legendre', n_functions=2, argvals=self.argvals, norm=True)
        np.testing.assert_allclose(
            X.values,
            np.array([[1., 1., 1.], [0., 0.8660254, 1.73205081]])
        )

    def test_basis_wiener(self):
        X = Basis(name='wiener', n_functions=2, argvals=self.argvals)
        np.testing.assert_allclose(
                X.values,
                np.array([[0., 1., 1.41421356],[0., 1., -1.41421356]])
        )

    def test_basis_fourier(self):
        X = Basis(name='fourier', n_functions=2, argvals=self.argvals)
        np.testing.assert_allclose(
            X.values,
            np.array([[1., 1., 1.],[0., 0.47942554, 0.84147098],[1., 0.87758256, 0.54030231]])
        )

    def test_basis_bsplines(self):
        X = Basis(name='bsplines', n_functions=2, argvals=self.argvals, degree=0)
        np.testing.assert_allclose(
            X.values,
            np.array([[1., 0., 0.],[0., 1., 1.]])
        )

    def test_basis_bsplines_with_knots(self):
        X = Basis(name='bsplines', n_functions=2, argvals=self.argvals, degree=0, knots=np.array([0.25, 0.5, 0.75]))
        np.testing.assert_allclose(
            X.values,
            np.array([[1., 0., 0.],[0., 1., 1.]])
        )

    def test_multibasis(self):
        X = Basis(name='legendre', n_functions=2, dimension='2D', argvals=self.argvals)
        np.testing.assert_allclose(
            X.values,
            np.array([[[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]],[[0., 0.5, 1.],[0., 0.5, 1.],[0., 0.5, 1.]],[[0., 0., 0.],[0.5, 0.5, 0.5],[1., 1., 1.]],[[0., 0., 0.],[0., 0.25, 0.5],[0., 0.5, 1.]]])
        )


class TestBasisFails(unittest.TestCase):
    """Fail test class for the functions in basis.py"""

    def setUp(self):
        self.argvals_1d = {
            'input_dim_0': np.array([0, 0.5, 1]),
        }
        self.argvals_2d = {
            'input_dim_0': np.array([0, 0.5, 1]),
            'input_dim_1': np.array([0, 0.5, 1])
        }

    def test_basis(self):
        with self.assertRaises(NotImplementedError) as cm:
            Basis(name='failed', n_functions=2, argvals=self.argvals_1d)
        self.assertTrue('Basis' in str(cm.exception))

    def test_basis_name(self):
        with self.assertRaises(TypeError) as cm:
            Basis(name=0, n_functions=2, argvals=self.argvals_1d)
        self.assertTrue('str' in str(cm.exception))

    def test_basis_2d(self):
        with self.assertRaises(NotImplementedError) as cm:
            Basis(name='legendre', n_functions=2, argvals=self.argvals_2d)
        self.assertTrue('dimensional' in str(cm.exception))

    def test_basis_dim(self):
        with self.assertRaises(ValueError) as cm:
            Basis(name='legendre', n_functions=2, argvals=self.argvals_1d, dimension='3D')
        self.assertTrue('dimension' in str(cm.exception))
