#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the karhunen.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest

from FDApy.representation.basis import Basis, MultivariateBasis
from FDApy.representation.argvals import DenseArgvals
from FDApy.representation.values import DenseValues
from FDApy.representation.functional_data import (
    DenseFunctionalData,
    MultivariateFunctionalData
)
from FDApy.simulation.karhunen import (
    _eigenvalues_linear,
    _eigenvalues_exponential,
    _eigenvalues_quadratic,
    _eigenvalues_inverse,
    _eigenvalues_sqrt,
    _eigenvalues_wiener,
    _simulate_eigenvalues,
    _make_coef,
    _initialize_centers,
    _initialize_clusters_std,
    _compute_data,
    KarhunenLoeve
)


class TestEigenvaluesLinear(unittest.TestCase):
    def test_eigenvalues_exponential_default(self):
        expected_output = np.array(
            [1.0, 0.6666666666666666, 0.3333333333333333]
        )
        result = _eigenvalues_linear()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_linear_n4(self):
        # Test that the function returns the correct values for n=4
        expected = np.array([1., 0.75, 0.5, 0.25])
        result = _eigenvalues_linear(n=4)
        np.testing.assert_allclose(result, expected)
    
    def test_eigenvalues_linear_n1(self):
        # Test that the function returns the correct values for n=1
        expected = np.array([1.0])
        result = _eigenvalues_linear(n=1)
        np.testing.assert_allclose(result, expected)


class TestEigenvaluesExponential(unittest.TestCase):
    def test_eigenvalues_exponential_default(self):
        expected_output = np.array(
            [1., 0.60653066, 0.36787944]
        )
        result = _eigenvalues_exponential()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_exponential_n4(self):
        expected_output = np.array(
            [1., 0.60653066, 0.36787944, 0.22313016]
        )
        result = _eigenvalues_exponential(n=4)
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_exponential_n1(self):
        expected_output = np.array([1.])
        result = _eigenvalues_exponential(n=1)
        np.testing.assert_allclose(result, expected_output)


class TestEigenvaluesQuadratic(unittest.TestCase):
    def test_eigenvalues_quadratic_default(self):
        expected_output = np.array(
            [1., 0.25, 0.11111111]
        )
        result = _eigenvalues_quadratic()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_quadratic_n4(self):
        expected_output = np.array(
            [1., 0.25, 0.11111111, 0.0625]
        )
        result = _eigenvalues_quadratic(n=4)
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_quadratic_n1(self):
        expected_output = np.array([1.])
        result = _eigenvalues_exponential(n=1)
        np.testing.assert_allclose(result, expected_output)


class TestEigenvaluesInverse(unittest.TestCase):
    def test_eigenvalues_inverse_default(self):
        expected_output = np.array(
            [1., 0.5, 0.33333333]
        )
        result = _eigenvalues_inverse()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_inverse_n4(self):
        expected_output = np.array(
            [1., 0.5, 0.33333333, 0.25]
        )
        result = _eigenvalues_inverse(n=4)
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_inverse_n1(self):
        expected_output = np.array([1.])
        result = _eigenvalues_inverse(n=1)
        np.testing.assert_allclose(result, expected_output)


class TestEigenvaluesSqrt(unittest.TestCase):
    def test_eigenvalues_sqrt_default(self):
        expected_output = np.array(
            [1., 0.70710678, 0.57735027]
        )
        result = _eigenvalues_sqrt()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_sqrt_n4(self):
        expected_output = np.array(
            [1., 0.70710678, 0.57735027, 0.5]
        )
        result = _eigenvalues_sqrt(n=4)
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_sqrt_n1(self):
        expected_output = np.array([1.])
        result = _eigenvalues_inverse(n=1)
        np.testing.assert_allclose(result, expected_output)


class TestEigenvaluesWiener(unittest.TestCase):
    def test_eigenvalues_wiener_default(self):
        expected_output = np.array(
            [0.4052847345693511, 0.04503163717437235, 0.016211389382774045]
        )
        result = _eigenvalues_wiener()
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_wiener_n4(self):
        expected_output = np.array(
            [
                0.4052847345693511, 0.04503163717437235,
                0.016211389382774045, 0.008271117032027573
            ]
        )
        result = _eigenvalues_wiener(n=4)
        np.testing.assert_allclose(result, expected_output)

    def test_eigenvalues_wiener_n1(self):
        expected_output = np.array([0.4052847345693511])
        result = _eigenvalues_wiener(n=1)
        np.testing.assert_allclose(result, expected_output)


class TestSimulateEigenvalues(unittest.TestCase):

    def test_linear_eigenvalues(self):
        expected_output = np.array(
            [1.0, 0.6666666666666666, 0.3333333333333333]
        )
        output = _simulate_eigenvalues('linear', n=3)
        np.testing.assert_allclose(output, expected_output)

    def test_exponential_eigenvalues(self):
        expected_output = np.array(
            [1., 0.60653066, 0.36787944]
        )
        output = _simulate_eigenvalues('exponential', n=3)
        np.testing.assert_allclose(output, expected_output)

    def test_quadratic_eigenvalues(self):
        expected_output = np.array(
            [1., 0.25, 0.11111111]
        )
        output = _simulate_eigenvalues('quadratic', n=3)
        np.testing.assert_allclose(output, expected_output)
    
    def test_linear_eigenvalues(self):
        expected_output = np.array(
            [1., 0.5, 0.33333333]
        )
        output = _simulate_eigenvalues('inverse', n=3)
        np.testing.assert_allclose(output, expected_output)
    
    def test_sqrt_eigenvalues(self):
        expected_output = np.array(
            [1., 0.70710678, 0.57735027]
        )
        output = _simulate_eigenvalues('sqrt', n=3)
        np.testing.assert_allclose(output, expected_output)

    def test_wiener_eigenvalues(self):
        expected_output = np.array(
            [0.4052847345693511, 0.04503163717437235, 0.016211389382774045]
        )
        output = _simulate_eigenvalues('wiener', n=3)
        np.testing.assert_allclose(output, expected_output)

    def test_raises_value_error_for_negative_n(self):
        with self.assertRaises(ValueError):
            _simulate_eigenvalues('linear', n=-1)

    def test_raises_not_implemented_error_for_unknown_method(self):
        with self.assertRaises(NotImplementedError):
            _simulate_eigenvalues('unknown', n=3)


class TestMakeCoef(unittest.TestCase):
    def setUp(self):
        self.centers = np.array([[1, 2, 3], [0, 4, 6]])
        self.cluster_std = np.array([[0.5, 0.25, 1],[1, 0.1, 0.5]])
        self.n_obs = 4
        self.n_features = 2
        self.rnorm = np.random.default_rng(42).multivariate_normal

    def test_output_shape(self):
        coefs, labels = _make_coef(
            self.n_obs, self.n_features, self.centers, self.cluster_std
        )
        np.testing.assert_equal(coefs.shape, (self.n_obs, self.n_features))
        np.testing.assert_equal(labels.shape, (self.n_obs,))

    def test_output_labels(self):
        _, labels = _make_coef(
            self.n_obs, self.n_features,
            self.centers, self.cluster_std,
            self.rnorm
        )
        unique_labels = np.unique(labels)
        output = np.array([0, 1, 2])
        np.testing.assert_array_almost_equal(output, unique_labels)

    def test_output_coefs(self):
        coefs, _ = _make_coef(
            self.n_obs, self.n_features,
            self.centers, self.cluster_std,
            self.rnorm
        )
        output = np.array([
            [0.26462019, 0.30471708],
            [1.66507969, 0.7504512 ],
            [1.02448241, 3.58821468],
            [3.1278404 , 5.77638272]
        ])
        np.testing.assert_array_almost_equal(output, coefs)


class TestInitializeCenters(unittest.TestCase):
    def test_centers_initialized_to_zero(self):
        n_features = 3
        n_clusters = 4
        centers = None
        expected_centers = np.zeros((n_features, n_clusters))
        output = _initialize_centers(n_features, n_clusters, centers)
        np.testing.assert_array_equal(output, expected_centers)
    
    def test_centers_initialized_to_provided_value(self):
        n_features = 3
        n_clusters = 4
        centers = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        expected_centers = centers
        output = _initialize_centers(n_features, n_clusters, centers)
        np.testing.assert_array_equal(output, expected_centers)


class TestInitializeClusterStd(unittest.TestCase):
    def test_default(self):
        n_features = 2
        n_clusters = 3
        expected = np.ones((n_features, n_clusters))
        result = _initialize_clusters_std(n_features, n_clusters)
        np.testing.assert_array_equal(result, expected)

    def test_linear(self):
        n_features = 2
        n_clusters = 3
        expected = np.array([
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5]
        ])
        result = _initialize_clusters_std(
            n_features, n_clusters, clusters_std='linear'
        )
        np.testing.assert_allclose(result, expected)

    def test_exponential(self):
        n_features = 2
        n_clusters = 3
        expected = np.array([
            [1., 1., 1.],
            [0.60653066, 0.60653066, 0.60653066]
        ])
        result = _initialize_clusters_std(
            n_features, n_clusters, clusters_std='exponential'
        )
        np.testing.assert_allclose(result, expected)

    def test_wiener(self):
        n_features = 2
        n_clusters = 3
        expected = np.array([
            [0.40528473, 0.40528473, 0.40528473],
            [0.04503164, 0.04503164, 0.04503164]
        ])
        result = _initialize_clusters_std(
            n_features, n_clusters, clusters_std='wiener'
        )
        np.testing.assert_allclose(result, expected)

    def test_custom(self):
        n_features = 2
        n_clusters = 3
        cluster_std = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]
        ])
        expected = cluster_std
        result = _initialize_clusters_std(
            n_features, n_clusters, clusters_std=cluster_std
        )
        np.testing.assert_array_equal(result, expected)


class TestComputeData(unittest.TestCase):
    def setUp(self):
        self.basis1d = Basis(name='legendre', n_functions=2, dimension='1D')
        self.basis2d = Basis(name='legendre', n_functions=2, dimension='2D')

        self.coef_1d = np.array([[0.30471708, -0.73537981]])
        self.coef_2d = np.array([[0.30471708, -0.90065266]])

    def test_dimension_1d(self):
        output = _compute_data(self.basis1d, self.coef_1d)
        expected = np.matmul(self.coef_1d, self.basis1d.values)
        np.testing.assert_array_almost_equal(output.values, expected)

    def test_dimension_2d(self):
        output = _compute_data(self.basis2d, self.coef_2d)
        expected = np.tensordot(self.coef_2d, self.basis2d.values, axes=1)
        np.testing.assert_array_almost_equal(output.values, expected)

    def test_raise_value_error(self):
        argvals = DenseArgvals({'input_dim_0': np.array([0, 1, 2]), 'input_dim_1': np.array([0, 1]), 'input_dim_2': np.array([0, 1])})
        values = DenseValues(np.zeros((1, 3, 2, 2)))
        basis = DenseFunctionalData(argvals, values)  # n_dim = 3

        with self.assertRaises(ValueError):
            _compute_data(basis, self.coef_1d)


class TestCheckBasisNone(unittest.TestCase):
    def setUp(self):
        self.basis = Basis(name='legendre', n_functions=2, dimension='1D')
        self.basis_name = 'fourier'

    def test_raise_error(self):
        with self.assertRaises(ValueError):
            KarhunenLoeve._check_basis_none(self.basis_name, self.basis)

    def test_raise_error_none(self):
        with self.assertRaises(ValueError):
            KarhunenLoeve._check_basis_none(None, None)

    def test_no_raise_error(self):
        KarhunenLoeve._check_basis_none(self.basis_name, None)
        self.assertTrue(True)  # if no error was raised, the test is successful


class TestCheckBasisType(unittest.TestCase):
    def setUp(self):
        self.basis = Basis(name='legendre', n_functions=2, dimension='1D')
        self.basis_multi = MultivariateBasis(
            simulation_type='split',
            n_components=2,
            name='fourier',
            n_functions=3
        )

    def test_raise_error(self):
        with self.assertRaises(ValueError):
            KarhunenLoeve._check_basis_type('error')

    def test_basis_none(self):
        KarhunenLoeve._check_basis_type(None)
        self.assertTrue(True)  # if no error was raised, the test is successful

    def test_basis_basis(self):
        KarhunenLoeve._check_basis_type(self.basis)
        self.assertTrue(True)  # if no error was raised, the test is successful

    def test_basis_list_basis(self):
        KarhunenLoeve._check_basis_type(self.basis_multi)
        self.assertTrue(True)  # if no error was raised, the test is successful


class TestCreateListBasis(unittest.TestCase):
    def setUp(self):
        self.basis_name = ['fourier', 'bsplines']
        self.dimension = ['1D', '2D']
        self.n_functions = 5

    def test_create_basis(self):
        
        basis_list = KarhunenLoeve._create_basis(
            self.basis_name, self.dimension, self.n_functions
        )

        self.assertEqual(len(basis_list), 2)
        self.assertIsInstance(basis_list, MultivariateBasis)
        self.assertEqual(basis_list.name, ['fourier', 'bsplines'])

        self.assertEqual(basis_list.data[0].n_obs, 5)
        self.assertEqual(basis_list.data[0].n_dimension, 1)

        self.assertEqual(basis_list.data[1].n_obs, 5)
        self.assertEqual(basis_list.data[1].n_dimension, 2)

    def test_create_basis_fourier(self):
        n_functions = 6

        basis_list = KarhunenLoeve._create_basis(
            self.basis_name, self.dimension, n_functions
        )

        self.assertEqual(len(basis_list), 2)
        self.assertIsInstance(basis_list, MultivariateBasis)
        self.assertEqual(basis_list.name, ['fourier', 'bsplines'])

        self.assertEqual(basis_list.data[0].n_obs, 6)
        self.assertEqual(basis_list.data[0].n_dimension, 1)

        self.assertEqual(basis_list.data[1].n_obs, 6)
        self.assertEqual(basis_list.data[1].n_dimension, 2)


class TestKarhunenLoeveInit(unittest.TestCase):
    def setUp(self):
        self.basis_name = ['fourier', 'bsplines']
        self.dimension = '1D'
        self.n_functions = 5
        self.basis = Basis('fourier', n_functions=self.n_functions)

    def test_init_with_no_basis_name(self):
        with self.assertRaises(ValueError):
            KarhunenLoeve(
                basis_name=None,
                n_functions=self.n_functions,
                dimension=self.dimension,
                basis=None
            )

    def test_init_with_basis_name_and_basis(self):
        with self.assertRaises(ValueError):
            KarhunenLoeve(
                basis_name=self.basis_name,
                n_functions=self.n_functions,
                dimension=self.dimension,
                basis=self.basis
            )

    def test_init_with_basis_name_as_string(self):
        kl = KarhunenLoeve(
            basis_name=self.basis_name[0],
            n_functions=self.n_functions,
            dimension=self.dimension
        )
        self.assertIsNotNone(kl.basis)
        self.assertIsInstance(kl.basis, Basis)
        self.assertEqual(kl.basis.name, 'fourier')
        self.assertEqual(kl.basis.n_obs, 5)
        self.assertEqual(kl.basis.dimension, '1D')

    def test_init_with_basis_name_as_list(self):
        kl = KarhunenLoeve(
            basis_name=self.basis_name,
            n_functions=self.n_functions,
            dimension=self.dimension
        )
        self.assertIsNotNone(kl.basis)
        self.assertEqual(len(kl.basis), 2)
        self.assertIsInstance(kl.basis, MultivariateBasis)

    def test_init_with_basis_univariate(self):
        kl = KarhunenLoeve(
            basis_name=None,
            n_functions=None,
            dimension=None,
            basis=self.basis
        )
        self.assertIsNotNone(kl.basis)
        self.assertIsInstance(kl.basis, Basis)
        self.assertEqual(kl.basis.name, 'fourier')
        self.assertEqual(kl.basis.n_obs, 5)
        self.assertEqual(kl.basis.dimension, '1D')


class TestKarhunenLoeveNew(unittest.TestCase):
    def setUp(self):
        self.basis_name = ['fourier', 'bsplines']
        self.dimension = '1D'
        self.n_functions = 5
        self.kl = KarhunenLoeve(
            basis_name=self.basis_name,
            n_functions=self.n_functions,
            dimension=self.dimension,
            random_state=42
        )

    def test_new_n_obs(self):
        n_obs = 100
        self.kl.new(n_obs)
        self.assertEqual(self.kl.data.n_obs, n_obs)

    def test_new_labels(self):
        n_clusters = 3
        self.kl.new(10, n_clusters=n_clusters)
        self.assertEqual(len(np.unique(self.kl.labels)), n_clusters)

    def test_new_eigenvalues(self):
        centers = np.zeros((self.n_functions, 1))
        clusters_std = np.ones((self.n_functions, 1))
        self.kl.new(10, 1, centers=centers, clusters_std=clusters_std)
        np.testing.assert_allclose(self.kl.eigenvalues, clusters_std[:, 0])

    def test_new_multivariate(self):
        self.kl.new(10, 1)
        self.assertIsInstance(self.kl.data, MultivariateFunctionalData)

    def test_new_univariate(self):
        kl = KarhunenLoeve(
            basis_name=self.basis_name[0],
            n_functions=self.n_functions,
            dimension=self.dimension
        )
        kl.new(10, 1)
        self.assertIsInstance(kl.data, DenseFunctionalData)
