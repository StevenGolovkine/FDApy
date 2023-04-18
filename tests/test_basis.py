#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for Basis.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.basis import (
    Basis,
    _simulate_basis_multivariate_weighted,
    _simulate_basis_multivariate_split,
    _simulate_basis_multivariate
)


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

    def test_basis_bsplines_fail(self):
        with self.assertRaises(ValueError) as cm:
            Basis(name='bsplines', n_functions=2, argvals=self.argvals_1d, degree=2)
        self.assertTrue('small' in str(cm.exception))


class TestSimulateBasisMultivariateWeighted(unittest.TestCase):
    def setUp(self) -> None:
        self.basis_name = ['fourier', 'legendre']
        self.argvals = [np.linspace(0, 1, 11), np.linspace(-0.5, 0.5, 11)]
        self.norm = False
        self.n_functions = 3
        self.random_state = np.random.default_rng(42)

    def test_simulate_basis_multivariate_weighted(self):
        output = _simulate_basis_multivariate_weighted(
            self.basis_name, self.argvals, self.n_functions,
            self.norm, self.random_state.uniform
        )
        expected1 = np.array([[0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458], [0.        , 0.0766276 , 0.15248956, 0.22682789, 0.29889983, 0.36798527, 0.43339392, 0.49447224, 0.55060996, 0.60124616, 0.64587491], [0.76755458, 0.76372001, 0.75225459, 0.7332729 , 0.70696459, 0.67359252, 0.63349013, 0.58705813, 0.53476043, 0.47711958, 0.41471151]])
        expected2 = np.array([[ 0.64098359,  0.64098359,  0.64098359,  0.64098359,  0.64098359, 0.64098359,  0.64098359,  0.64098359,  0.64098359,  0.64098359, 0.64098359], [-0.32049179, -0.25639344, -0.19229508, -0.12819672, -0.06409836, 0.        ,  0.06409836,  0.12819672,  0.19229508,  0.25639344, 0.32049179], [-0.08012295, -0.16665573, -0.23395901, -0.28203278, -0.31087704, -0.32049179, -0.31087704, -0.28203278, -0.23395901, -0.16665573, -0.08012295]])
        expected_output = [expected1, expected2]
        for idx in range(len(output)):
            np.testing.assert_array_almost_equal(output[idx], expected_output[idx])


class TestSimulateBasisMultivariateSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.basis_name = 'fourier'
        self.argvals = [np.linspace(0, 1, 11), np.linspace(-0.5, 0.5, 11)]
        self.norm = False
        self.n_functions = 3
        self.random_state = np.random.default_rng(42)

    def test_simulate_basis_multivariate_weighted(self):
        output = _simulate_basis_multivariate_split(
            self.basis_name, self.argvals, self.n_functions,
            self.norm, self.random_state.choice
        )
        expected1 = np.array([[-1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        ], [-0.        , -0.09983342, -0.19866933, -0.29552021, -0.38941834, -0.47942554, -0.56464247, -0.64421769, -0.71735609, -0.78332691, -0.84147098], [-1.        , -0.99500417, -0.98006658, -0.95533649, -0.92106099, -0.87758256, -0.82533561, -0.76484219, -0.69670671, -0.62160997, -0.54030231]])
        expected2 = np.array([[ 1.        ,  1.        ,  1.        ,  1.        ,  1.        , 1.        ,  1.        ,  1.        ,  1.        ,  1.        , 1.        ], [ 0.84147098,  0.89120736,  0.93203909,  0.96355819,  0.98544973, 0.99749499,  0.9995736 ,  0.99166481,  0.97384763,  0.94630009, 0.90929743], [ 0.54030231,  0.45359612,  0.36235775,  0.26749883,  0.16996714, 0.0707372 , -0.02919952, -0.12884449, -0.22720209, -0.32328957,-0.41614684]])
        expected_output = [expected1, expected2]
        for idx in range(len(output)):
            np.testing.assert_array_almost_equal(output[idx], expected_output[idx])


class TestSimulateBasisMultivariate(unittest.TestCase):
    def setUp(self) -> None:
        self.n_components = 2
        self.argvals = [np.linspace(0, 1, 11), np.linspace(-0.5, 0.5, 11)]
        self.n_functions = 3
        self.norm = False
        self.random_state = np.random.default_rng(42)

    def test_simulation_type_split(self):
        simulation_type = 'split'
        name = 'fourier'
        
        output = _simulate_basis_multivariate(
            simulation_type, self.n_components, name, self.argvals,
            self.n_functions, self.norm, rchoice=self.random_state.choice
        )
        expected1 = np.array([[-1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        , -1.        ], [-0.        , -0.09983342, -0.19866933, -0.29552021, -0.38941834, -0.47942554, -0.56464247, -0.64421769, -0.71735609, -0.78332691, -0.84147098], [-1.        , -0.99500417, -0.98006658, -0.95533649, -0.92106099, -0.87758256, -0.82533561, -0.76484219, -0.69670671, -0.62160997, -0.54030231]])
        expected2 = np.array([[ 1.        ,  1.        ,  1.        ,  1.        ,  1.        , 1.        ,  1.        ,  1.        ,  1.        ,  1.        , 1.        ], [ 0.84147098,  0.89120736,  0.93203909,  0.96355819,  0.98544973, 0.99749499,  0.9995736 ,  0.99166481,  0.97384763,  0.94630009, 0.90929743], [ 0.54030231,  0.45359612,  0.36235775,  0.26749883,  0.16996714, 0.0707372 , -0.02919952, -0.12884449, -0.22720209, -0.32328957,-0.41614684]])
        expected_output = [expected1, expected2]
        for idx in range(len(output)):
            np.testing.assert_array_almost_equal(output[idx], expected_output[idx])

    def test_simulation_type_weighted(self):
        simulation_type = 'weighted'
        name = ['fourier', 'legendre']

        output = _simulate_basis_multivariate(
            simulation_type, self.n_components, name, self.argvals, 
            self.n_functions, self.norm, runif=self.random_state.uniform
        )
        expected1 = np.array([[0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458, 0.76755458], [0.        , 0.0766276 , 0.15248956, 0.22682789, 0.29889983, 0.36798527, 0.43339392, 0.49447224, 0.55060996, 0.60124616, 0.64587491], [0.76755458, 0.76372001, 0.75225459, 0.7332729 , 0.70696459, 0.67359252, 0.63349013, 0.58705813, 0.53476043, 0.47711958, 0.41471151]])
        expected2 = np.array([[ 0.64098359,  0.64098359,  0.64098359,  0.64098359,  0.64098359, 0.64098359,  0.64098359,  0.64098359,  0.64098359,  0.64098359, 0.64098359], [-0.32049179, -0.25639344, -0.19229508, -0.12819672, -0.06409836, 0.        ,  0.06409836,  0.12819672,  0.19229508,  0.25639344, 0.32049179], [-0.08012295, -0.16665573, -0.23395901, -0.28203278, -0.31087704, -0.32049179, -0.31087704, -0.28203278, -0.23395901, -0.16665573, -0.08012295]])
        expected_output = [expected1, expected2]
        for idx in range(len(output)):
            np.testing.assert_array_almost_equal(output[idx], expected_output[idx])

    def test_simulation_type_not_implemented(self):
        simulation_type = 'unknown'
        name = 'fourier'
        with self.assertRaises(NotImplementedError):
            _simulate_basis_multivariate(
                simulation_type, self.n_components, name, self.argvals,
                self.n_functions, self.norm
            )

    def test_argvals_length_mismatch(self):
        simulation_type = 'split'
        n_components = 3
        name = 'fourier'
        with self.assertRaises(ValueError):
            _simulate_basis_multivariate(
                simulation_type, n_components, name, self.argvals, self.n_functions
            )

    def test_simulation_type_split_name_failed(self):
        simulation_type = 'split'
        name = ['fourier']
        with self.assertRaises(ValueError):
            _simulate_basis_multivariate(
                simulation_type, self.n_components, name, self.argvals,
                self.n_functions, self.norm, rchoice=self.random_state.choice
            )
    
    def test_simulation_type_weighted_name_failed(self):
        simulation_type = 'weighted'
        name = 'fourier'
        with self.assertRaises(ValueError):
            _simulate_basis_multivariate(
                simulation_type, self.n_components, name, self.argvals,
                self.n_functions, self.norm, rchoice=self.random_state.choice
            )

    def test_simulation_type_weighted_name_list_failed(self):
        simulation_type = 'weighted'
        name = ['fourier']
        with self.assertRaises(ValueError):
            _simulate_basis_multivariate(
                simulation_type, self.n_components, name, self.argvals,
                self.n_functions, self.norm, rchoice=self.random_state.choice
            )
