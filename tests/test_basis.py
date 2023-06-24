#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for Basis.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.basis import (
    Basis,
    MultivariateBasis,
    _simulate_basis_multivariate_weighted,
    _simulate_basis_multivariate_split,
    _simulate_basis_multivariate
)


class TestBasis(unittest.TestCase):
    def setUp(self):
        self.argvals = np.array([0, 0.5, 1])

    def test_getter(self):
        X = Basis(name='legendre', n_functions=2, argvals=self.argvals)
        np.testing.assert_equal(X.name, 'legendre')
        np.testing.assert_equal(X.is_normalized, False)
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

    def test_basis_legendre_is_normalized(self):
        X = Basis(name='legendre', n_functions=2, argvals=self.argvals, is_normalized=True)
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
            np.array([[ 1.        ,  1.        ,  1.        ], [-1.41421356,  1.41421356, -1.41421356]])
        )
    
    def test_basis_fourier_no_intercept(self):
        X = Basis(name='fourier', n_functions=2, argvals=self.argvals, add_intercept=False)
        np.testing.assert_allclose(
            X.values,
            np.array([[-1.41421356,  1.41421356, -1.41421356], [-1.731912e-16,  0.000000e+00,  1.731912e-16]])
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
            np.array([[[0.  , 0.5 , 1.  ],[0.  , 0.5 , 1.  ],[0.  , 0.5 , 1.  ]], [[0.  , 0.  , 0.  ],[0.  , 0.25, 0.5 ],[0.  , 0.5 , 1.  ]]])
        )


class TestBasisFails(unittest.TestCase):
    """Fail test class for the functions in basis.py"""

    def setUp(self):
        self.argvals_1d = np.array([0, 0.5, 1]) 
        self.argvals_2d = [np.array([0, 0.5, 1]), np.array([0, 0.5, 1])]

    def test_basis(self):
        with self.assertRaises(NotImplementedError) as cm:
            Basis(name='failed', n_functions=2, argvals=self.argvals_1d)
        self.assertTrue('Basis' in str(cm.exception))

    def test_basis_name(self):
        with self.assertRaises(TypeError) as cm:
            Basis(name=0, n_functions=2, argvals=self.argvals_1d)
        self.assertTrue('str' in str(cm.exception))

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
        self.is_normalized = False
        self.n_functions = 3
        self.random_state = np.random.default_rng(42)

    def test_simulate_basis_multivariate_weighted(self):
        output = _simulate_basis_multivariate_weighted(
            self.basis_name, self.argvals, self.n_functions,
            self.is_normalized, self.random_state.uniform
        )
        expected1 = np.array([[ 7.67554583e-01,  7.67554583e-01,  7.67554583e-01, 7.67554583e-01,  7.67554583e-01,  7.67554583e-01, 7.67554583e-01,  7.67554583e-01,  7.67554583e-01, 7.67554583e-01,  7.67554583e-01], [-1.08548610e+00, -8.78176703e-01, -3.35433652e-01, 3.35433652e-01,  8.78176703e-01,  1.08548610e+00, 8.78176703e-01,  3.35433652e-01, -3.35433652e-01, -8.78176703e-01, -1.08548610e+00], [-1.32933708e-16, -6.38032722e-01, -1.03235863e+00, -1.03235863e+00, -6.38032722e-01,  0.00000000e+00, 6.38032722e-01,  1.03235863e+00,  1.03235863e+00, 6.38032722e-01,  1.32933708e-16]])
        expected2 = np.array([[ 0.64098359,  0.64098359,  0.64098359,  0.64098359,  0.64098359, 0.64098359,  0.64098359,  0.64098359,  0.64098359,  0.64098359, 0.64098359], [-0.32049179, -0.25639344, -0.19229508, -0.12819672, -0.06409836, 0.        ,  0.06409836,  0.12819672,  0.19229508,  0.25639344, 0.32049179], [-0.08012295, -0.16665573, -0.23395901, -0.28203278, -0.31087704, -0.32049179, -0.31087704, -0.28203278, -0.23395901, -0.16665573, -0.08012295]])
        expected_output = [expected1, expected2]
        for idx in range(len(output)):
            np.testing.assert_array_almost_equal(output[idx], expected_output[idx])


class TestSimulateBasisMultivariateSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.basis_name = 'fourier'
        self.argvals = [np.linspace(0, 1, 11), np.linspace(-0.5, 0.5, 11)]
        self.is_normalized = False
        self.n_functions = 3
        self.random_state = np.random.default_rng(42)

    def test_simulate_basis_multivariate_split(self):
        output = _simulate_basis_multivariate_split(
            self.basis_name, self.argvals, self.n_functions,
            self.is_normalized, self.random_state.choice
        )
        expected1 = np.array([[-7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01], [ 1.00000000e+00,  9.51056516e-01,  8.09016994e-01,   5.87785252e-01,  3.09016994e-01, -6.12323400e-17, -3.09016994e-01, -5.87785252e-01, -8.09016994e-01, -9.51056516e-01, -1.00000000e+00], [ 1.22464680e-16,  3.09016994e-01,  5.87785252e-01,   8.09016994e-01,  9.51056516e-01,  1.00000000e+00,   9.51056516e-01,  8.09016994e-01,  5.87785252e-01,   3.09016994e-01, -0.00000000e+00]])
        expected2 = np.array([[ 7.07106781e-01,  7.07106781e-01,  7.07106781e-01,   7.07106781e-01,  7.07106781e-01,  7.07106781e-01,   7.07106781e-01,  7.07106781e-01,  7.07106781e-01,   7.07106781e-01,  7.07106781e-01], [ 1.00000000e+00,  9.51056516e-01,  8.09016994e-01,   5.87785252e-01,  3.09016994e-01,  6.12323400e-17,  -3.09016994e-01, -5.87785252e-01, -8.09016994e-01,  -9.51056516e-01, -1.00000000e+00], [ 0.00000000e+00,  3.09016994e-01,  5.87785252e-01,   8.09016994e-01,  9.51056516e-01,  1.00000000e+00,   9.51056516e-01,  8.09016994e-01,  5.87785252e-01,   3.09016994e-01,  1.22464680e-16]])
        expected_output = [expected1, expected2]
        for idx in range(len(output)):
            np.testing.assert_array_almost_equal(output[idx], expected_output[idx])


class TestSimulateBasisMultivariate(unittest.TestCase):
    def setUp(self) -> None:
        self.n_components = 2
        self.argvals = [np.linspace(0, 1, 11), np.linspace(-0.5, 0.5, 11)]
        self.n_functions = 3
        self.is_normalized = False
        self.random_state = np.random.default_rng(42)

    def test_simulation_type_split(self):
        simulation_type = 'split'
        name = 'fourier'
        
        output = _simulate_basis_multivariate(
            simulation_type, self.n_components, name, self.argvals,
            self.n_functions, self.is_normalized, rchoice=self.random_state.choice
        )
        expected1 = np.array([[-7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01], [ 1.00000000e+00,  9.51056516e-01,  8.09016994e-01,   5.87785252e-01,  3.09016994e-01, -6.12323400e-17, -3.09016994e-01, -5.87785252e-01, -8.09016994e-01, -9.51056516e-01, -1.00000000e+00], [ 1.22464680e-16,  3.09016994e-01,  5.87785252e-01,   8.09016994e-01,  9.51056516e-01,  1.00000000e+00,   9.51056516e-01,  8.09016994e-01,  5.87785252e-01,   3.09016994e-01, -0.00000000e+00]])
        expected2 = np.array([[ 7.07106781e-01,  7.07106781e-01,  7.07106781e-01,   7.07106781e-01,  7.07106781e-01,  7.07106781e-01,   7.07106781e-01,  7.07106781e-01,  7.07106781e-01,   7.07106781e-01,  7.07106781e-01], [ 1.00000000e+00,  9.51056516e-01,  8.09016994e-01,   5.87785252e-01,  3.09016994e-01,  6.12323400e-17,  -3.09016994e-01, -5.87785252e-01, -8.09016994e-01,  -9.51056516e-01, -1.00000000e+00], [ 0.00000000e+00,  3.09016994e-01,  5.87785252e-01,   8.09016994e-01,  9.51056516e-01,  1.00000000e+00,   9.51056516e-01,  8.09016994e-01,  5.87785252e-01,   3.09016994e-01,  1.22464680e-16]])
        expected_output = [expected1, expected2]
        for idx in range(len(output)):
            np.testing.assert_array_almost_equal(output[idx], expected_output[idx])

    def test_simulation_type_weighted(self):
        simulation_type = 'weighted'
        name = ['fourier', 'legendre']

        output = _simulate_basis_multivariate(
            simulation_type, self.n_components, name, self.argvals, 
            self.n_functions, self.is_normalized, runif=self.random_state.uniform
        )
        expected1 = np.array([[ 7.67554583e-01,  7.67554583e-01,  7.67554583e-01, 7.67554583e-01,  7.67554583e-01,  7.67554583e-01, 7.67554583e-01,  7.67554583e-01,  7.67554583e-01, 7.67554583e-01,  7.67554583e-01], [-1.08548610e+00, -8.78176703e-01, -3.35433652e-01, 3.35433652e-01,  8.78176703e-01,  1.08548610e+00, 8.78176703e-01,  3.35433652e-01, -3.35433652e-01, -8.78176703e-01, -1.08548610e+00], [-1.32933708e-16, -6.38032722e-01, -1.03235863e+00, -1.03235863e+00, -6.38032722e-01,  0.00000000e+00, 6.38032722e-01,  1.03235863e+00,  1.03235863e+00, 6.38032722e-01,  1.32933708e-16]])
        expected2 = np.array([[ 0.64098359,  0.64098359,  0.64098359,  0.64098359,  0.64098359, 0.64098359,  0.64098359,  0.64098359,  0.64098359,  0.64098359, 0.64098359], [-0.32049179, -0.25639344, -0.19229508, -0.12819672, -0.06409836, 0.        ,  0.06409836,  0.12819672,  0.19229508,  0.25639344, 0.32049179], [-0.08012295, -0.16665573, -0.23395901, -0.28203278, -0.31087704, -0.32049179, -0.31087704, -0.28203278, -0.23395901, -0.16665573, -0.08012295]])
        expected_output = [expected1, expected2]
        for idx in range(len(output)):
            np.testing.assert_array_almost_equal(output[idx], expected_output[idx])

    def test_simulation_type_weighted(self):
        simulation_type = 'weighted'
        name = ['fourier', 'legendre']

        output = _simulate_basis_multivariate(
            simulation_type, self.n_components, name, self.argvals, 
            self.n_functions, self.is_normalized, runif=None
        )
        expected1 = np.array([[ 7.07106781e-01,  7.07106781e-01,  7.07106781e-01,  7.07106781e-01,  7.07106781e-01,  7.07106781e-01,  7.07106781e-01,  7.07106781e-01,  7.07106781e-01,  7.07106781e-01,  7.07106781e-01],[-1.00000000e+00, -8.09016994e-01, -3.09016994e-01,  3.09016994e-01,  8.09016994e-01,  1.00000000e+00,  8.09016994e-01,  3.09016994e-01, -3.09016994e-01, -8.09016994e-01, -1.00000000e+00],[-1.22464680e-16, -5.87785252e-01, -9.51056516e-01, -9.51056516e-01, -5.87785252e-01,  0.00000000e+00,  5.87785252e-01,  9.51056516e-01,  9.51056516e-01,  5.87785252e-01,  1.22464680e-16]])
        expected2 = np.array([[ 0.70710678,  0.70710678,  0.70710678,  0.70710678,  0.70710678,  0.70710678,  0.70710678,  0.70710678,  0.70710678,  0.70710678,  0.70710678],[-0.35355339, -0.28284271, -0.21213203, -0.14142136, -0.07071068,  0.        ,  0.07071068,  0.14142136,  0.21213203,  0.28284271,  0.35355339],[-0.08838835, -0.18384776, -0.25809398, -0.31112698, -0.34294679, -0.35355339, -0.34294679, -0.31112698, -0.25809398, -0.18384776, -0.08838835]])
        expected_output = [expected1, expected2]
        for idx in range(len(output)):
            np.testing.assert_array_almost_equal(output[idx], expected_output[idx])

    def test_simulation_type_not_implemented(self):
        simulation_type = 'unknown'
        name = 'fourier'
        with self.assertRaises(NotImplementedError):
            _simulate_basis_multivariate(
                simulation_type, self.n_components, name, self.argvals,
                self.n_functions, self.is_normalized
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
                self.n_functions, self.is_normalized, rchoice=self.random_state.choice
            )
    
    def test_simulation_type_weighted_name_failed(self):
        simulation_type = 'weighted'
        name = 'fourier'
        with self.assertRaises(ValueError):
            _simulate_basis_multivariate(
                simulation_type, self.n_components, name, self.argvals,
                self.n_functions, self.is_normalized, rchoice=self.random_state.choice
            )

    def test_simulation_type_weighted_name_list_failed(self):
        simulation_type = 'weighted'
        name = ['fourier']
        with self.assertRaises(ValueError):
            _simulate_basis_multivariate(
                simulation_type, self.n_components, name, self.argvals,
                self.n_functions, self.is_normalized, rchoice=self.random_state.choice
            )


class TestMultivariateBasis(unittest.TestCase):
    def setUp(self):
        self.argvals = [np.array([0, 0.5, 1]), np.array([0, 0.5, 1])]
        self.n_components = 2

    def test_getter(self):
        X = MultivariateBasis(
            simulation_type='split', n_components=self.n_components,
            name='legendre', n_functions=2, argvals=self.argvals
        )
        np.testing.assert_equal(X.simulation_type, 'split')
        np.testing.assert_equal(X.name, 'legendre')
        np.testing.assert_equal(X.is_normalized, False)
        np.testing.assert_equal(X.dimension, ['1D', '1D'])

        X = MultivariateBasis(
            simulation_type='weighted', n_components=self.n_components,
            name=['legendre', 'fourier'], n_functions=2, argvals=self.argvals
        )
        np.testing.assert_equal(X.simulation_type, 'weighted')
        np.testing.assert_equal(X.name, ['legendre', 'fourier'])
        np.testing.assert_equal(X.is_normalized, False)
        np.testing.assert_equal(X.dimension, ['1D', '1D'])

    def test_setter_fails(self):
        X = MultivariateBasis(
            simulation_type='split', n_components=self.n_components,
            name='legendre', n_functions=2, argvals=self.argvals
        )
        with self.assertRaises(TypeError) as cm:
            X.simulation_type = 0
        self.assertTrue('str' in str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            X.name = 0
        self.assertTrue('str' in str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            X.name = [0]
        self.assertTrue('List[str]' in str(cm.exception))

    def test_argvals(self):
        X = MultivariateBasis(
            simulation_type='split', n_components=self.n_components,
            name='legendre', n_functions=2, argvals=None
        )
        np.testing.assert_allclose(X.data[0].argvals['input_dim_0'], np.arange(0, 1.01, 0.01))
        np.testing.assert_allclose(X.data[1].argvals['input_dim_0'], np.arange(0, 1.01, 0.01))

    def test_multivariate_basis(self):
        n_components = 2
        basis_name = 'fourier'
        argvals = [
            np.linspace(0, 1, 11),
            np.linspace(-0.5, 0.5, 11)
        ]
        n_functions = 3
        dimension = ['1D', '2D']
        random_state = np.random.default_rng(42)

        basis = MultivariateBasis(
            simulation_type='split',
            n_components=n_components,
            name=basis_name,
            n_functions=n_functions,
            dimension=dimension,
            argvals=argvals,
            is_normalized=False,
            rchoice=random_state.choice
        )

        expected_argvals_0 = {'input_dim_0': np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])}
        expected_argvals_1 = {'input_dim_0': np.array([-0.5, -0.4, -0.3, -0.2, -0.1,  0. ,  0.1,  0.2,  0.3,  0.4,  0.5]), 'input_dim_1': np.array([-0.5, -0.4, -0.3, -0.2, -0.1,  0. ,  0.1,  0.2,  0.3,  0.4,  0.5])}

        expected_output_0 = np.array([[-7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01],[ 1.00000000e+00,  9.51056516e-01,  8.09016994e-01,  5.87785252e-01,  3.09016994e-01, -6.12323400e-17, -3.09016994e-01, -5.87785252e-01, -8.09016994e-01, -9.51056516e-01, -1.00000000e+00],[ 1.22464680e-16,  3.09016994e-01,  5.87785252e-01,  8.09016994e-01,  9.51056516e-01,  1.00000000e+00,  9.51056516e-01,  8.09016994e-01,  5.87785252e-01,  3.09016994e-01, -0.00000000e+00]])
        expected_output_1 = np.array([[[ 5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01], [ 5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01], [ 5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01], [ 5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01], [ 5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01], [ 5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01], [ 5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01], [ 5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01], [ 5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01], [ 5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01], [ 5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01,  5.00000000e-01,   5.00000000e-01,  5.00000000e-01]], [[ 7.07106781e-01,  6.72498512e-01,  5.72061403e-01,   4.15626938e-01,  2.18508012e-01,  4.32978028e-17,  -2.18508012e-01, -4.15626938e-01, -5.72061403e-01,  -6.72498512e-01, -7.07106781e-01], [ 7.07106781e-01,  6.72498512e-01,  5.72061403e-01,   4.15626938e-01,  2.18508012e-01,  4.32978028e-17,  -2.18508012e-01, -4.15626938e-01, -5.72061403e-01,  -6.72498512e-01, -7.07106781e-01], [ 7.07106781e-01,  6.72498512e-01,  5.72061403e-01,   4.15626938e-01,  2.18508012e-01,  4.32978028e-17,  -2.18508012e-01, -4.15626938e-01, -5.72061403e-01,  -6.72498512e-01, -7.07106781e-01], [ 7.07106781e-01,  6.72498512e-01,  5.72061403e-01,   4.15626938e-01,  2.18508012e-01,  4.32978028e-17,  -2.18508012e-01, -4.15626938e-01, -5.72061403e-01,  -6.72498512e-01, -7.07106781e-01], [ 7.07106781e-01,  6.72498512e-01,  5.72061403e-01,   4.15626938e-01,  2.18508012e-01,  4.32978028e-17,  -2.18508012e-01, -4.15626938e-01, -5.72061403e-01,  -6.72498512e-01, -7.07106781e-01], [ 7.07106781e-01,  6.72498512e-01,  5.72061403e-01,   4.15626938e-01,  2.18508012e-01,  4.32978028e-17,  -2.18508012e-01, -4.15626938e-01, -5.72061403e-01,  -6.72498512e-01, -7.07106781e-01], [ 7.07106781e-01,  6.72498512e-01,  5.72061403e-01,   4.15626938e-01,  2.18508012e-01,  4.32978028e-17,  -2.18508012e-01, -4.15626938e-01, -5.72061403e-01,  -6.72498512e-01, -7.07106781e-01], [ 7.07106781e-01,  6.72498512e-01,  5.72061403e-01,   4.15626938e-01,  2.18508012e-01,  4.32978028e-17,  -2.18508012e-01, -4.15626938e-01, -5.72061403e-01,  -6.72498512e-01, -7.07106781e-01], [ 7.07106781e-01,  6.72498512e-01,  5.72061403e-01,   4.15626938e-01,  2.18508012e-01,  4.32978028e-17,  -2.18508012e-01, -4.15626938e-01, -5.72061403e-01,  -6.72498512e-01, -7.07106781e-01], [ 7.07106781e-01,  6.72498512e-01,  5.72061403e-01,   4.15626938e-01,  2.18508012e-01,  4.32978028e-17,  -2.18508012e-01, -4.15626938e-01, -5.72061403e-01,  -6.72498512e-01, -7.07106781e-01], [ 7.07106781e-01,  6.72498512e-01,  5.72061403e-01,   4.15626938e-01,  2.18508012e-01,  4.32978028e-17,  -2.18508012e-01, -4.15626938e-01, -5.72061403e-01,  -6.72498512e-01, -7.07106781e-01]], [[ 0.00000000e+00,  2.18508012e-01,  4.15626938e-01,   5.72061403e-01,  6.72498512e-01,  7.07106781e-01,   6.72498512e-01,  5.72061403e-01,  4.15626938e-01,   2.18508012e-01,  8.65956056e-17], [ 0.00000000e+00,  2.18508012e-01,  4.15626938e-01,   5.72061403e-01,  6.72498512e-01,  7.07106781e-01,   6.72498512e-01,  5.72061403e-01,  4.15626938e-01,   2.18508012e-01,  8.65956056e-17], [ 0.00000000e+00,  2.18508012e-01,  4.15626938e-01,   5.72061403e-01,  6.72498512e-01,  7.07106781e-01,   6.72498512e-01,  5.72061403e-01,  4.15626938e-01,   2.18508012e-01,  8.65956056e-17], [ 0.00000000e+00,  2.18508012e-01,  4.15626938e-01,   5.72061403e-01,  6.72498512e-01,  7.07106781e-01,   6.72498512e-01,  5.72061403e-01,  4.15626938e-01,   2.18508012e-01,  8.65956056e-17], [ 0.00000000e+00,  2.18508012e-01,  4.15626938e-01,   5.72061403e-01,  6.72498512e-01,  7.07106781e-01,   6.72498512e-01,  5.72061403e-01,  4.15626938e-01,   2.18508012e-01,  8.65956056e-17], [ 0.00000000e+00,  2.18508012e-01,  4.15626938e-01,   5.72061403e-01,  6.72498512e-01,  7.07106781e-01,   6.72498512e-01,  5.72061403e-01,  4.15626938e-01,   2.18508012e-01,  8.65956056e-17], [ 0.00000000e+00,  2.18508012e-01,  4.15626938e-01,   5.72061403e-01,  6.72498512e-01,  7.07106781e-01,   6.72498512e-01,  5.72061403e-01,  4.15626938e-01,   2.18508012e-01,  8.65956056e-17], [ 0.00000000e+00,  2.18508012e-01,  4.15626938e-01,   5.72061403e-01,  6.72498512e-01,  7.07106781e-01,   6.72498512e-01,  5.72061403e-01,  4.15626938e-01,   2.18508012e-01,  8.65956056e-17], [ 0.00000000e+00,  2.18508012e-01,  4.15626938e-01,   5.72061403e-01,  6.72498512e-01,  7.07106781e-01,   6.72498512e-01,  5.72061403e-01,  4.15626938e-01,   2.18508012e-01,  8.65956056e-17], [ 0.00000000e+00,  2.18508012e-01,  4.15626938e-01,   5.72061403e-01,  6.72498512e-01,  7.07106781e-01,   6.72498512e-01,  5.72061403e-01,  4.15626938e-01,   2.18508012e-01,  8.65956056e-17], [ 0.00000000e+00,  2.18508012e-01,  4.15626938e-01,   5.72061403e-01,  6.72498512e-01,  7.07106781e-01,   6.72498512e-01,  5.72061403e-01,  4.15626938e-01,   2.18508012e-01,  8.65956056e-17]]])
        
        np.testing.assert_almost_equal(basis.data[0].argvals['input_dim_0'], expected_argvals_0['input_dim_0'])
        np.testing.assert_almost_equal(basis.data[1].argvals['input_dim_0'], expected_argvals_1['input_dim_0'])

        np.testing.assert_almost_equal(basis.data[0].values, expected_output_0)
        np.testing.assert_almost_equal(basis.data[1].values, expected_output_1)
