#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class MFPCA in the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
import warnings

from FDApy.representation.argvals import DenseArgvals
from FDApy.representation.functional_data import MultivariateFunctionalData
from FDApy.preprocessing.dim_reduction.mfpca import MFPCA
from FDApy.simulation.karhunen import KarhunenLoeve


class MFPCATest(unittest.TestCase):
    def test_init(self):
        # Test default initialization
        fpca = MFPCA(n_components=0.99)
        self.assertEqual(fpca.method, "covariance")
        self.assertEqual(fpca.n_components, 0.99)
        self.assertFalse(fpca.normalize)
        self.assertEqual(fpca.weights, None)

        # Test custom initialization
        fpca = MFPCA(method="inner-product", n_components=3, normalize=True)
        self.assertEqual(fpca.method, "inner-product")
        self.assertEqual(fpca.n_components, 3)
        self.assertTrue(fpca.normalize)

    def test_method(self):
        ufpc = MFPCA(n_components=0.99)
        ufpc.method = "inner-product"
        self.assertEqual(ufpc.method, "inner-product")

    def test_n_components(self):
        ufpc = MFPCA(n_components=0.99)
        ufpc.n_components = [4, 3]
        self.assertEqual(ufpc.n_components, [4, 3])

    def test_normalize(self):
        ufpc = MFPCA(n_components=0.99)
        ufpc.normalize = True
        self.assertTrue(ufpc.normalize)


class TestFit(unittest.TestCase):
    def setUp(self):
        kl = KarhunenLoeve(
            basis_name="bsplines",
            n_functions=5,
            argvals=DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)}),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.add_noise(0.01)
        fdata_uni = kl.noisy_data

        kl = KarhunenLoeve(
            basis_name="bsplines",
            n_functions=5,
            argvals=DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)}),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)
        fdata_sparse = kl.sparse_data

        self.fdata = MultivariateFunctionalData([fdata_uni, fdata_sparse])

    def test_fit_error_method(self):
        mfpca = MFPCA(method="error", n_components=3)
        with self.assertRaises(ValueError):
            mfpca.fit(self.fdata)

    def test_fit_covariance(self):
        univariate_expansions = [
            {"method": "UFPCA", "n_components": 15, "method_smoothing": "PS"},
            {"method": "UFPCA", "n_components": 15, "method_smoothing": "PS"},
        ]
        mfpca = MFPCA(
            method="covariance",
            n_components=3,
            univariate_expansions=univariate_expansions,
        )
        mfpca.fit(data=self.fdata, method_smoothing="PS")

        expected_eigenvalues = np.array([0.3456826, 0.04979564, 0.01094198])
        np.testing.assert_array_almost_equal(
            mfpca.eigenvalues, expected_eigenvalues, decimal=3
        )

        expected_eigenfunctions_0 = np.array(
            [
                [0.66258175, -0.01444167, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.00397063, 0.62907167, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.04173367, 0.08928001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(mfpca.eigenfunctions.data[0].coefficients),
            np.abs(expected_eigenfunctions_0),
            decimal=2,
        )

        expected_eigenfunctions_1 = np.array(
            [
                [
                    0.82755887,
                    -0.12543712,
                    -0.03505173,
                    -0.01226347,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.84352585,
                    0.77789592,
                    -0.09958717,
                    -0.08522377,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -2.02305773,
                    -0.76187305,
                    -0.65481286,
                    -0.33083348,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(mfpca.eigenfunctions.data[1].coefficients),
            np.abs(expected_eigenfunctions_1),
            decimal=2,
        )

        self.assertIsNone(mfpca.covariance)

    def test_fit_inner_product(self):
        mfpca = MFPCA(method="inner-product", n_components=0.99, normalize=True)
        mfpca.fit(data=self.fdata, method_smoothing="PS")

        expected_eigenvalues = np.array([1.60366454, 0.25304897, 0.08986414])
        np.testing.assert_array_almost_equal(
            mfpca.eigenvalues, expected_eigenvalues, decimal=2
        )

        expected_eigenfunctions_0 = np.array(
            [
                -0.15606947,
                -0.30886952,
                -0.38658726,
                -0.60512314,
                -0.75482784,
                -0.76376565,
                -0.90559884,
                -0.90712252,
                -0.85680313,
                -0.78332485,
                -0.52981632,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(mfpca.eigenfunctions.data[0].values[0]),
            np.abs(expected_eigenfunctions_0),
            decimal=1,
        )

        expected_eigenfunctions_1 = np.array(
            [
                -0.17475991,
                -0.19292637,
                -0.21041226,
                -0.22641831,
                -0.24219129,
                -0.25755674,
                -0.27363584,
                -0.28856384,
                -0.3028844,
                -0.31743344,
                -0.33232156,
                -0.34777691,
                -0.36222205,
                -0.37704445,
                -0.39219638,
                -0.40801802,
                -0.42300912,
                -0.43736151,
                -0.45210116,
                -0.46671253,
                -0.48105886,
                -0.49550205,
                -0.50993747,
                -0.52370325,
                -0.53734927,
                -0.55113605,
                -0.56465597,
                -0.57821437,
                -0.59153203,
                -0.60426913,
                -0.61724017,
                -0.62978136,
                -0.64228393,
                -0.65456843,
                -0.66670693,
                -0.6784477,
                -0.68998272,
                -0.70116635,
                -0.71234879,
                -0.72303133,
                -0.73355341,
                -0.74392003,
                -0.75382201,
                -0.7632105,
                -0.77261104,
                -0.78189078,
                -0.79042552,
                -0.79896436,
                -0.80716539,
                -0.81476805,
                -0.82230521,
                -0.8293286,
                -0.83621735,
                -0.84242887,
                -0.8482489,
                -0.85417605,
                -0.8585765,
                -0.86368409,
                -0.86835973,
                -0.87284972,
                -0.87697907,
                -0.88019813,
                -0.88320577,
                -0.88561879,
                -0.88794195,
                -0.88965062,
                -0.89118985,
                -0.89224149,
                -0.89229232,
                -0.89294536,
                -0.89268093,
                -0.89217645,
                -0.89107392,
                -0.88970582,
                -0.88763372,
                -0.88564596,
                -0.88272693,
                -0.87929845,
                -0.87570543,
                -0.87170633,
                -0.86709334,
                -0.86208836,
                -0.85661845,
                -0.85082475,
                -0.84444156,
                -0.83754282,
                -0.83041928,
                -0.82241791,
                -0.81438613,
                -0.80556495,
                -0.79618463,
                -0.78671875,
                -0.7764713,
                -0.76525072,
                -0.75405313,
                -0.74242153,
                -0.73108266,
                -0.72056244,
                -0.7098907,
                -0.70397892,
                -0.69774837,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(mfpca.eigenfunctions.data[1].values[0]),
            np.abs(expected_eigenfunctions_1),
            decimal=1,
        )


class TestTransform(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)

        kl = KarhunenLoeve(
            basis_name="bsplines",
            n_functions=5,
            argvals=DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)}),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.add_noise(0.01)
        fdata_uni = kl.noisy_data

        kl = KarhunenLoeve(
            basis_name="bsplines",
            n_functions=5,
            argvals=DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)}),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)
        fdata_sparse = kl.sparse_data
        self.fdata = MultivariateFunctionalData([fdata_uni, fdata_sparse])

        univariate_expansions = [
            {"method": "UFPCA", "n_components": 15, "method_smoothing": "PS"},
            {"method": "UFPCA", "n_components": 15, "method_smoothing": "PS"},
        ]
        mfpca_cov = MFPCA(
            n_components=2,
            method="covariance",
            univariate_expansions=univariate_expansions,
            normalize=True,
        )
        mfpca_cov.fit(self.fdata, method_smoothing="PS")
        self.mfpca_cov = mfpca_cov

        mfpca_inn = MFPCA(n_components=2, method="inner-product", normalize=True)
        mfpca_inn.fit(self.fdata, method_smoothing="PS")
        self.mfpca_inn = mfpca_inn

    def test_error_innpro(self):
        with self.assertRaises(ValueError):
            self.mfpca_cov.transform(self.fdata, method="InnPro")

        with self.assertRaises(ValueError):
            self.mfpca_cov.transform(None, method="InnPro")

    def test_error_unkown_method(self):
        with self.assertRaises(ValueError):
            self.mfpca_cov.transform(self.fdata, method="error")

    def test_data_none(self):
        scores = self.mfpca_cov.transform(None, method="NumInt")
        expected_scores = np.array(
            [
                [-0.50802446, 1.45976142],
                [1.28040069, 1.22835455],
                [-1.00709149, -0.49777253],
                [1.21320319, 1.21258645],
                [-0.54343243, 0.95923373],
                [-0.53905316, -0.07583408],
                [2.37573186, 1.56605796],
                [2.8778799, 1.68259548],
                [1.22127914, 0.43310981],
                [-0.82929794, -0.48834422],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_data_notnone(self):
        scores = self.mfpca_cov.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [-1.24054825, 0.56391971],
                [0.54785412, 0.3324879],
                [-1.73945137, -1.39295822],
                [0.48071786, 0.31676905],
                [-1.27596437, 0.06341355],
                [-1.27157148, -0.9717331],
                [1.64376234, 0.67109989],
                [2.14539948, 0.78685382],
                [0.48878098, -0.46272323],
                [-1.56189765, -1.38442032],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_numint(self):
        scores = self.mfpca_inn.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [-1.25586712, 0.99666744],
                [0.5158972, 0.09481443],
                [-1.61525358, -0.64427102],
                [0.44294513, 0.08325553],
                [-1.2403226, 0.61157485],
                [-1.17194227, -0.38319023],
                [1.55714306, -0.02354195],
                [2.05563155, -0.05652208],
                [0.50108732, -0.63407485],
                [-1.43590675, -0.65361137],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=0
        )

    def test_innpro(self):
        scores = self.mfpca_inn.transform(method="InnPro")
        expected_scores = np.array(
            [
                [-1.08419608, 1.00986994],
                [0.67578999, 0.15011529],
                [-1.44145117, -0.56939632],
                [0.60311901, 0.13901118],
                [-1.06878402, 0.63892167],
                [-1.00090153, -0.3160711],
                [1.70994642, 0.04035405],
                [2.20526721, 0.01015399],
                [0.66090987, -0.55024305],
                [-1.26289861, -0.57616036],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_pace(self):
        scores = self.mfpca_cov.transform(self.fdata, method="PACE")
        expected_scores = np.array(
            [
                [-0.63204461, 1.20357021],
                [1.12957453, 0.44506032],
                [-1.00444295, -0.27300537],
                [1.0124857, 0.37016359],
                [-0.59562873, 0.88937765],
                [-0.55053731, -0.01937044],
                [2.14830046, 0.28750088],
                [2.63564574, 0.27412362],
                [1.09216177, -0.26143997],
                [-0.80298274, -0.29366325],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )


class TestInverseTransform(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)

        kl = KarhunenLoeve(
            basis_name="bsplines",
            n_functions=5,
            argvals=DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)}),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.add_noise(0.01)
        fdata_uni = kl.noisy_data

        kl = KarhunenLoeve(
            basis_name="bsplines",
            n_functions=5,
            argvals=DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)}),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)
        fdata_sparse = kl.sparse_data
        self.fdata = MultivariateFunctionalData([fdata_uni, fdata_sparse])

        mfpca_inn = MFPCA(n_components=2, method="inner-product", normalize=True)
        mfpca_inn.fit(self.fdata, method_smoothing="PS")
        self.mfpca_inn = mfpca_inn
        self.scores = self.mfpca_inn.transform(method="InnPro")

    def test_inverse_scores(self):
        fdata_recons = self.mfpca_inn.inverse_transform(self.scores)

        self.assertIsInstance(fdata_recons, MultivariateFunctionalData)

        expected_values = np.array(
            [
                -0.23438262,
                -0.18201513,
                -0.10779444,
                0.04008788,
                0.1958946,
                0.21827602,
                0.30925696,
                0.41395862,
                0.30934545,
                0.30332572,
                0.11524915,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(fdata_recons.data[0].values[0]), np.abs(expected_values), decimal=0
        )

        expected_values = np.array(
            [
                0.20063226,
                0.21662768,
                0.23272474,
                0.2489671,
                0.26491742,
                0.28034096,
                0.29580769,
                0.30975215,
                0.32274547,
                0.33532984,
                0.34643565,
                0.35645871,
                0.36535022,
                0.3726336,
                0.37926663,
                0.38487262,
                0.38925308,
                0.39263331,
                0.39523921,
                0.39687132,
                0.39760209,
                0.39780013,
                0.39717897,
                0.39595419,
                0.39441839,
                0.39320022,
                0.39220353,
                0.39180062,
                0.39187002,
                0.39310571,
                0.39528642,
                0.39892191,
                0.40377237,
                0.40952845,
                0.41646177,
                0.42402834,
                0.4325966,
                0.44206176,
                0.45179085,
                0.46194331,
                0.47242154,
                0.48309086,
                0.49385161,
                0.50442099,
                0.51511665,
                0.52582326,
                0.53590468,
                0.54597745,
                0.55561782,
                0.56457545,
                0.57300717,
                0.58079275,
                0.58772898,
                0.59385075,
                0.59877553,
                0.60279087,
                0.60527691,
                0.606745,
                0.60660685,
                0.60484931,
                0.60151624,
                0.5963014,
                0.58967509,
                0.58186846,
                0.57326123,
                0.5639239,
                0.55462805,
                0.54584401,
                0.53692095,
                0.52951194,
                0.52271589,
                0.51706182,
                0.51329454,
                0.51057908,
                0.50918997,
                0.50866612,
                0.50887724,
                0.50969773,
                0.5115322,
                0.51389167,
                0.51655444,
                0.51942772,
                0.52257122,
                0.52561157,
                0.5284495,
                0.53093497,
                0.53288323,
                0.53396465,
                0.53443228,
                0.53363177,
                0.53166641,
                0.52839002,
                0.52371964,
                0.51788712,
                0.5112601,
                0.50342707,
                0.49501521,
                0.48617791,
                0.4767204,
                0.46794158,
                0.45880097,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(fdata_recons.data[1].values[0]), np.abs(expected_values), decimal=0
        )
