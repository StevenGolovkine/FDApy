#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for Basis.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.argvals import DenseArgvals
from FDApy.representation.values import DenseValues
from FDApy.representation.basis import Basis, MultivariateBasis


class TestBasis(unittest.TestCase):
    def setUp(self):
        self.argvals = DenseArgvals({"input_dim_0": np.array([0, 0.5, 1])})

        self.argvals_2d = DenseArgvals(
            {"input_dim_0": np.array([0, 0.5, 1]), "input_dim_1": np.array([0, 0.5, 1])}
        )

    def test_getter(self):
        X = Basis(name="legendre", n_functions=2, argvals=self.argvals)
        np.testing.assert_equal(X.name, ("legendre",))
        np.testing.assert_equal(X.n_functions, (2,))
        np.testing.assert_equal(X.is_normalized, False)
        np.testing.assert_equal(X.add_intercept, True)

    def test_argvals(self):
        X = Basis(name="legendre", n_functions=2, argvals=None)
        np.testing.assert_allclose(X.argvals["input_dim_0"], np.arange(0, 1.01, 0.1))

    def test_basis_given(self):
        argvals = DenseArgvals({"input_dim_0": np.array([0, 0.5, 1])})
        values = DenseValues([[1, 2, 3], [4, 5, 6]])
        basis = Basis(name="given", argvals=argvals, values=values)

        np.testing.assert_equal(basis.n_functions, (2,))
        np.testing.assert_equal(basis.values, values)

    def test_basis_legendre(self):
        X = Basis(name="legendre", n_functions=2, argvals=self.argvals)
        np.testing.assert_allclose(
            X.values, np.array([[1.0, 1.0, 1.0], [0.0, 0.5, 1.0]])
        )

    def test_basis_legendre_is_normalized(self):
        X = Basis(
            name="legendre", n_functions=2, argvals=self.argvals, is_normalized=True
        )
        np.testing.assert_allclose(
            X.values, np.array([[1.0, 1.0, 1.0], [0.0, 0.8660254, 1.73205081]])
        )

    def test_basis_wiener(self):
        X = Basis(name="wiener", n_functions=2, argvals=self.argvals)
        np.testing.assert_allclose(
            X.values, np.array([[0.0, 1.0, 1.41421356], [0.0, 1.0, -1.41421356]])
        )

    def test_basis_fourier(self):
        X = Basis(name="fourier", n_functions=2, argvals=self.argvals)
        np.testing.assert_allclose(
            X.values,
            np.array([[1.0, 1.0, 1.0], [-1.41421356, 1.41421356, -1.41421356]]),
        )

    def test_basis_fourier_no_intercept(self):
        X = Basis(
            name="fourier", n_functions=2, argvals=self.argvals, add_intercept=False
        )
        np.testing.assert_allclose(
            X.values,
            np.array(
                [
                    [-1.41421356, 1.41421356, -1.41421356],
                    [-1.731912e-16, 0.000000e00, 1.731912e-16],
                ]
            ),
        )

    def test_basis_bsplines(self):
        X = Basis(name="bsplines", n_functions=2, argvals=self.argvals, degree=1)
        np.testing.assert_allclose(
            X.values, np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])
        )

    def test_multidimensional_basis(self):
        X = Basis(
            name=("legendre", "legendre"), n_functions=(2, 2), argvals=self.argvals_2d
        )
        np.testing.assert_allclose(
            X.values,
            np.array(
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
                    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.25, 0.5], [0.0, 0.5, 1.0]],
                ]
            ),
        )


class TestBasisFails(unittest.TestCase):
    """Fail test class for the functions in basis.py"""

    def setUp(self):
        self.argvals = DenseArgvals({"input_dim_0": np.array([0, 0.5, 1])})

        self.argvals_2d = DenseArgvals(
            {"input_dim_0": np.array([0, 0.5, 1]), "input_dim_1": np.array([0, 0.5, 1])}
        )

    def test_basis(self):
        with self.assertRaises(NotImplementedError) as cm:
            Basis(name="failed", n_functions=2, argvals=self.argvals)
        self.assertTrue("Basis" in str(cm.exception))


class TestMultivariateBasis(unittest.TestCase):
    def setUp(self):
        self.argvals = [
            DenseArgvals({"input_dim_0": np.array([0, 0.5, 1])}),
            DenseArgvals({"input_dim_0": np.array([0, 0.5, 1])}),
        ]

    def test_getter(self):
        X = MultivariateBasis(
            name=["legendre", "legendre"],
            n_functions=[2, 2],
            argvals=self.argvals,
        )
        np.testing.assert_equal(X.name, ["legendre", "legendre"])
        np.testing.assert_equal(X.is_normalized, False)

    def test_argvals(self):
        X = MultivariateBasis(
            name=["legendre", "legendre"],
            n_functions=[2, 2],
            argvals=None,
        )
        np.testing.assert_allclose(
            X.data[0].argvals["input_dim_0"], np.arange(0, 1.1, 0.1)
        )
        np.testing.assert_allclose(
            X.data[1].argvals["input_dim_0"], np.arange(0, 1.1, 0.1)
        )

    def test_multivariate_basis(self):
        basis_name = ["fourier", ("fourier", "fourier")]
        argvals = [
            DenseArgvals({"input_dim_0": np.array([0, 0.5, 1])}),
            DenseArgvals(
                {
                    "input_dim_0": np.array([0, 0.5, 1]),
                    "input_dim_1": np.array([0, 0.5, 1]),
                }
            ),
        ]
        n_functions = [2, (2, 1)]

        basis = MultivariateBasis(
            name=basis_name,
            n_functions=n_functions,
            argvals=argvals,
            is_normalized=False,
            add_intercept=True,
        )

        expected_argvals_0 = {"input_dim_0": np.array([0, 0.5, 1])}
        expected_argvals_1 = {
            "input_dim_0": np.array([0, 0.5, 1]),
            "input_dim_1": np.array([0, 0.5, 1]),
        }

        expected_output_0 = np.array(
            [[1.0, 1.0, 1.0], [-1.41421356, 1.41421356, -1.41421356]]
        )
        expected_output_1 = np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [
                    [-1.41421356, -1.41421356, -1.41421356],
                    [1.41421356, 1.41421356, 1.41421356],
                    [-1.41421356, -1.41421356, -1.41421356],
                ],
            ]
        )

        np.testing.assert_almost_equal(
            basis.data[0].argvals["input_dim_0"], expected_argvals_0["input_dim_0"]
        )
        np.testing.assert_almost_equal(
            basis.data[1].argvals["input_dim_0"], expected_argvals_1["input_dim_0"]
        )

        np.testing.assert_almost_equal(basis.data[0].values, expected_output_0)
        np.testing.assert_almost_equal(basis.data[1].values, expected_output_1)

    def test_multivariate_basis_given(self):
        argvals = [
            DenseArgvals({"input_dim_0": np.array([0, 0.5, 1])}),
            DenseArgvals(
                {
                    "input_dim_0": np.array([0, 0.5, 1]),
                    "input_dim_1": np.array([0, 0.5, 1]),
                }
            ),
        ]
        values = [
            DenseValues([[1, 2, 4], [2, 4, 6]]),
            DenseValues(
                [[[1, 2, 4], [2, 4, 6], [1, 2, 3]], [[1, 2, 4], [2, 4, 6], [1, 2, 3]]]
            ),
        ]
        basis = MultivariateBasis(name="given", argvals=argvals, values=values)

        np.testing.assert_almost_equal(
            basis.data[0].argvals["input_dim_0"], argvals[0]["input_dim_0"]
        )
        np.testing.assert_almost_equal(
            basis.data[1].argvals["input_dim_0"], argvals[1]["input_dim_0"]
        )
        np.testing.assert_almost_equal(
            basis.data[1].argvals["input_dim_1"], argvals[1]["input_dim_1"]
        )

        np.testing.assert_almost_equal(basis.data[0].values, values[0])
        np.testing.assert_almost_equal(basis.data[1].values, values[1])
