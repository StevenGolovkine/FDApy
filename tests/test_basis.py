#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import unittest

from FDApy.representation.basis import Basis


class TestBasis(unittest.TestCase):
    """Test class for the functions in basis.py
    """

    def setUp(self):
        self.argvals = {'input_dim_0': np.array([0, 0.5, 1])}

    def test_basis_legendre(self):
        X = Basis(name='legendre', n_functions=2, argvals=self.argvals)
        self.assertTrue(np.allclose(X.values,
                                    np.array([[1., 1., 1.], [0., 0.5, 1.]])))

    def test_basis_wiener(self):
        X = Basis(name='wiener', n_functions=2, argvals=self.argvals)
        self.assertTrue(np.allclose(X.values,
                                    np.array([[0., 1., 1.41421356],
                                              [0., 1., -1.41421356]])))

    def test_basis_fourier(self):
        X = Basis(name='fourier', n_functions=2, argvals=self.argvals)
        self.assertTrue(np.allclose(X.values,
                                    np.array([[1., 1., 1.],
                                              [0., 0.47942554, 0.84147098],
                                              [1., 0.87758256, 0.54030231]])))

    def test_basis_bsplines(self):
        X = Basis(name='bsplines', n_functions=2, argvals=self.argvals,
                  degree=0)
        self.assertTrue(np.allclose(X.values,
                                    np.array([[1., 0., 0.], [0., 1., 1.]])))


if __name__ == '__main__':
    unittest.main()
