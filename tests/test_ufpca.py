#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class UFPCA in the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
import warnings


from FDApy.representation.argvals import DenseArgvals
from FDApy.representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData,
)
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.ufpca import UFPCA


class UFPCATest(unittest.TestCase):
    def test_init(self):
        # Test default initialization
        fpca = UFPCA()
        self.assertEqual(fpca.method, "covariance")
        self.assertIsNone(fpca.n_components)
        self.assertFalse(fpca.normalize)
        self.assertEqual(fpca.weights, 1)

        # Test custom initialization
        fpca = UFPCA(method="inner-product", n_components=3, normalize=True)
        self.assertEqual(fpca.method, "inner-product")
        self.assertEqual(fpca.n_components, 3)
        self.assertTrue(fpca.normalize)

    def test_method(self):
        ufpc = UFPCA()
        ufpc.method = "inner-product"
        self.assertEqual(ufpc.method, "inner-product")

    def test_n_components(self):
        ufpc = UFPCA()
        ufpc.n_components = 4
        self.assertEqual(ufpc.n_components, 4)

    def test_normalize(self):
        ufpc = UFPCA()
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
        self.fdata = kl.noisy_data

        kl = KarhunenLoeve(
            basis_name="bsplines",
            n_functions=5,
            argvals=DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)}),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)
        self.fdata_sparse = kl.sparse_data

        kl = KarhunenLoeve(
            basis_name=("bsplines", "bsplines"),
            n_functions=(5, 5),
            argvals=DenseArgvals(
                {
                    "input_dim_0": np.linspace(0, 1, 11),
                    "input_dim_1": np.linspace(0, 1, 11),
                }
            ),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.add_noise(0.01)
        self.fdata_2D = kl.noisy_data

    def test_fit_covariance(self):
        uf = UFPCA(n_components=2, method="covariance")
        uf.fit(self.fdata)

        np.testing.assert_almost_equal(
            self.fdata.argvals["input_dim_0"], uf.mean.argvals["input_dim_0"]
        )

        expected_mean = np.array(
            [
                [
                    0.07913357,
                    0.06488778,
                    0.01913212,
                    -0.0118022,
                    -0.05725637,
                    0.01509972,
                    0.04731989,
                    0.09891403,
                    0.07953968,
                    0.12301846,
                    0.1120006,
                ]
            ]
        )
        np.testing.assert_array_almost_equal(uf.mean.values, expected_mean, decimal=2)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf.covariance, DenseFunctionalData)
        self.assertIsInstance(uf._training_data, DenseFunctionalData)

        expected_eigenvalues = np.array([0.17607021, 0.03293963])
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues, decimal=4
        )

        expected_eigenfunctions = np.array(
            [
                [
                    -0.21410275,
                    -0.43109646,
                    -0.54793253,
                    -0.85844707,
                    -1.07811755,
                    -1.08775624,
                    -1.29401218,
                    -1.30157519,
                    -1.2275912,
                    -1.11744805,
                    -0.75403918,
                ],
                [
                    2.30062061,
                    1.96488036,
                    1.32270592,
                    0.50748242,
                    -0.32758162,
                    -0.28668804,
                    -0.44192569,
                    -0.85648548,
                    -0.11395715,
                    -0.06929625,
                    0.83615138,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(uf.eigenfunctions.values), np.abs(expected_eigenfunctions), decimal=2
        )

        expected_noise = 0.02113118643931334
        np.testing.assert_array_almost_equal(
            uf._noise_variance, expected_noise, decimal=2
        )

    def test_fit_inner_product(self):
        uf = UFPCA(n_components=2, method="inner-product", normalize=True)
        uf.fit(self.fdata_sparse, method_smoothing="PS")

        np.testing.assert_almost_equal(
            self.fdata_sparse.argvals.to_dense()["input_dim_0"],
            uf.mean.argvals["input_dim_0"],
        )

        expected_mean = np.array(
            [
                [
                    0.47824172,
                    0.48763022,
                    0.49641234,
                    0.50449285,
                    0.51177649,
                    0.51816799,
                    0.52357211,
                    0.52789359,
                    0.53103717,
                    0.5329076,
                    0.53340962,
                    0.53247334,
                    0.53013029,
                    0.52643736,
                    0.52145144,
                    0.51522943,
                    0.50782821,
                    0.49930468,
                    0.48971573,
                    0.47911825,
                    0.46756913,
                    0.45515519,
                    0.44208299,
                    0.42858901,
                    0.41490974,
                    0.40128166,
                    0.38794125,
                    0.375125,
                    0.3630694,
                    0.35201091,
                    0.34218604,
                    0.33377554,
                    0.32673723,
                    0.32097322,
                    0.31638561,
                    0.3128765,
                    0.310348,
                    0.3087022,
                    0.30784121,
                    0.30766713,
                    0.30808206,
                    0.30899321,
                    0.31032825,
                    0.31201994,
                    0.31400105,
                    0.31620436,
                    0.31856264,
                    0.32100866,
                    0.32347518,
                    0.32589498,
                    0.32820083,
                    0.3303169,
                    0.332133,
                    0.33353034,
                    0.33439011,
                    0.33459354,
                    0.33402182,
                    0.33255617,
                    0.33007779,
                    0.32646789,
                    0.32160767,
                    0.31545128,
                    0.30824455,
                    0.30030628,
                    0.29195524,
                    0.2835102,
                    0.27528995,
                    0.26761326,
                    0.26079892,
                    0.25516569,
                    0.25103237,
                    0.24864412,
                    0.24795176,
                    0.2488325,
                    0.25116353,
                    0.25482206,
                    0.25968531,
                    0.26563048,
                    0.27253477,
                    0.2802754,
                    0.28872957,
                    0.29776582,
                    0.307218,
                    0.31691129,
                    0.32667088,
                    0.33632194,
                    0.34568966,
                    0.35459923,
                    0.36287581,
                    0.37034461,
                    0.37683079,
                    0.3822062,
                    0.38652933,
                    0.38990532,
                    0.39243931,
                    0.39423644,
                    0.39540187,
                    0.39604073,
                    0.39625816,
                    0.39615932,
                    0.39584935,
                ]
            ]
        )
        np.testing.assert_array_almost_equal(uf.mean.values, expected_mean, decimal=2)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf.covariance, DenseFunctionalData)
        self.assertIsInstance(uf._training_data, IrregularFunctionalData)

        expected_eigenvalues = np.array([0.82820218, 0.11959552])
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues, decimal=2
        )

        expected_eigenfunctions = np.array(
            [
                [
                    -0.24822054,
                    -0.27497311,
                    -0.30025219,
                    -0.32286993,
                    -0.34502022,
                    -0.36643917,
                    -0.38895987,
                    -0.40959307,
                    -0.42922455,
                    -0.44912976,
                    -0.4697034,
                    -0.49120173,
                    -0.51102197,
                    -0.53152587,
                    -0.55252535,
                    -0.5746519,
                    -0.59540438,
                    -0.61513059,
                    -0.63550346,
                    -0.65568555,
                    -0.67543547,
                    -0.69537852,
                    -0.71530355,
                    -0.73419603,
                    -0.75300419,
                    -0.77195146,
                    -0.79048275,
                    -0.80920488,
                    -0.82755438,
                    -0.84489908,
                    -0.86276679,
                    -0.87994463,
                    -0.89712573,
                    -0.91400362,
                    -0.93070131,
                    -0.9467746,
                    -0.96258703,
                    -0.97791219,
                    -0.99320373,
                    -1.00787889,
                    -1.02228843,
                    -1.03638542,
                    -1.04998748,
                    -1.06291934,
                    -1.07580442,
                    -1.08850799,
                    -1.10018452,
                    -1.11191686,
                    -1.12310539,
                    -1.13354473,
                    -1.14385625,
                    -1.15347666,
                    -1.16284074,
                    -1.17137119,
                    -1.17930025,
                    -1.18746411,
                    -1.19316579,
                    -1.20014845,
                    -1.20669793,
                    -1.2128721,
                    -1.21884094,
                    -1.22306265,
                    -1.22722454,
                    -1.23029798,
                    -1.23353763,
                    -1.2359463,
                    -1.2380893,
                    -1.23943434,
                    -1.23926901,
                    -1.24031862,
                    -1.23989761,
                    -1.2393441,
                    -1.2378085,
                    -1.23597642,
                    -1.23293314,
                    -1.23037788,
                    -1.22625964,
                    -1.22147649,
                    -1.21652862,
                    -1.2110365,
                    -1.20467653,
                    -1.19774645,
                    -1.19022403,
                    -1.18223468,
                    -1.17346471,
                    -1.16399492,
                    -1.15427922,
                    -1.14313798,
                    -1.13210531,
                    -1.11984675,
                    -1.10688982,
                    -1.09390573,
                    -1.07977693,
                    -1.06397411,
                    -1.04850234,
                    -1.03252705,
                    -1.01723117,
                    -1.00427347,
                    -0.99120861,
                    -0.98636319,
                    -0.981696,
                ],
                [
                    -2.33412761,
                    -2.32125216,
                    -2.30032179,
                    -2.26729514,
                    -2.22983183,
                    -2.188829,
                    -2.14043133,
                    -2.09411168,
                    -2.04483595,
                    -1.98861696,
                    -1.9344332,
                    -1.87834932,
                    -1.81816443,
                    -1.76137579,
                    -1.6994538,
                    -1.6371766,
                    -1.57389687,
                    -1.50812406,
                    -1.44097269,
                    -1.37346223,
                    -1.30481991,
                    -1.23348687,
                    -1.1634622,
                    -1.09385259,
                    -1.0251496,
                    -0.95420438,
                    -0.88344485,
                    -0.81173139,
                    -0.74250302,
                    -0.67008503,
                    -0.60096408,
                    -0.53015084,
                    -0.46081009,
                    -0.3941477,
                    -0.32686094,
                    -0.26238029,
                    -0.19715119,
                    -0.13117233,
                    -0.0689456,
                    -0.00798027,
                    0.05138948,
                    0.10887529,
                    0.16459832,
                    0.21736148,
                    0.26877893,
                    0.31882384,
                    0.36456604,
                    0.40972114,
                    0.45179167,
                    0.49030593,
                    0.52642429,
                    0.55956225,
                    0.58919508,
                    0.61698571,
                    0.64076186,
                    0.66234643,
                    0.68107928,
                    0.69766031,
                    0.71078449,
                    0.72049954,
                    0.72896128,
                    0.73370834,
                    0.73698973,
                    0.73722067,
                    0.73500901,
                    0.72886064,
                    0.7216885,
                    0.71565297,
                    0.70320674,
                    0.69268486,
                    0.67603242,
                    0.65513367,
                    0.6370544,
                    0.61504577,
                    0.59377306,
                    0.56851749,
                    0.54067148,
                    0.50994522,
                    0.48037257,
                    0.44872171,
                    0.41479137,
                    0.37860101,
                    0.34175121,
                    0.30290841,
                    0.26282261,
                    0.22163402,
                    0.17864185,
                    0.13399221,
                    0.08922322,
                    0.04202225,
                    -0.00520554,
                    -0.05495332,
                    -0.10538649,
                    -0.15670809,
                    -0.20724367,
                    -0.26143531,
                    -0.31556169,
                    -0.37085475,
                    -0.42735571,
                    -0.48240151,
                    -0.53898599,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(uf.eigenfunctions.values), np.abs(expected_eigenfunctions), decimal=2
        )

        expected_noise = 0.0018992206728438432
        np.testing.assert_array_almost_equal(
            uf._noise_variance, expected_noise, decimal=2
        )

    def test_fit_inner_product_2d(self):
        uf = UFPCA(n_components=2, method="inner-product")
        with np.testing.assert_warns(UserWarning):
            uf.fit(self.fdata_2D)

        expected_mean = np.array(
            [
                -0.20949505,
                -0.21953083,
                -0.21452318,
                -0.10253612,
                -0.06442279,
                -0.0663901,
                -0.00569674,
                -0.05482832,
                -0.05683153,
                -0.08004854,
                -0.05969944,
            ]
        )
        np.testing.assert_array_almost_equal(uf.mean.values[0, 1], expected_mean)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf._training_data, DenseFunctionalData)

        expected_eigenvalues = np.array([0.09721722, 0.04854923])
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues, decimal=2
        )

        expected_eigenfunctions = np.array(
            [
                0.23079242,
                0.38356143,
                0.69957733,
                0.99605228,
                1.00411503,
                0.96800581,
                1.00004967,
                0.74455034,
                0.50380433,
                0.27330228,
                0.14752861,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(uf.eigenfunctions.values[0, 1]),
            np.abs(expected_eigenfunctions),
            decimal=2,
        )

        expected_noise = 0.0
        np.testing.assert_array_almost_equal(uf._noise_variance, expected_noise)

    def test_error_method(self):
        uf = UFPCA(n_components=2, method="error")
        with self.assertRaises(NotImplementedError):
            uf.fit(self.fdata)

    def test_error_covariance_2d(self):
        uf = UFPCA(n_components=2, method="covariance")
        with self.assertRaises(ValueError):
            uf.fit(self.fdata_2D)


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
        self.fdata = kl.noisy_data

        kl = KarhunenLoeve(
            basis_name="bsplines",
            n_functions=5,
            argvals=DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)}),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)
        self.fdata_sparse = kl.sparse_data

        uf_dense = UFPCA(n_components=2, method="covariance", normalize=True)
        uf_dense.fit(self.fdata)
        self.uf_dense = uf_dense

        uf_sparse = UFPCA(n_components=2, method="inner-product", normalize=True)
        uf_sparse.fit(self.fdata_sparse, method_smoothing="PS")
        self.uf_sparse = uf_sparse

    def test_error_innpro(self):
        with self.assertRaises(ValueError):
            self.uf_sparse.transform(self.fdata, method="InnPro")

        with self.assertRaises(ValueError):
            self.uf_dense.transform(None, method="InnPro")

    def test_error_unkown_method(self):
        with self.assertRaises(ValueError):
            self.uf_dense.transform(self.fdata, method="error")

    def test_data_none(self):
        scores = self.uf_dense.transform(None, method="NumInt")
        expected_scores = np.array(
            [
                [-0.77377427, -0.7083848],
                [0.46190623, -0.11975884],
                [-1.01722304, 0.44593371],
                [0.40726946, -0.12920261],
                [-0.75156209, -0.52759135],
                [-0.66556736, 0.22942355],
                [1.14016736, -0.02794092],
                [1.58872117, -0.05889241],
                [0.49686958, 0.49864817],
                [-0.88680704, 0.39776551],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_data_notnone(self):
        scores = self.uf_dense.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [-0.87607204, -0.6691707],
                [0.35960847, -0.08054474],
                [-1.1195208, 0.48514781],
                [0.30497169, -0.08998851],
                [-0.85385985, -0.48837725],
                [-0.76786513, 0.26863765],
                [1.03786959, 0.01127318],
                [1.4864234, -0.01967831],
                [0.39457181, 0.53786227],
                [-0.98910481, 0.43697961],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_numint(self):
        scores = self.uf_dense.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [-0.87607204, -0.6691707],
                [0.35960847, -0.08054474],
                [-1.1195208, 0.48514781],
                [0.30497169, -0.08998851],
                [-0.85385985, -0.48837725],
                [-0.76786513, 0.26863765],
                [1.03786959, 0.01127318],
                [1.4864234, -0.01967831],
                [0.39457181, 0.53786227],
                [-0.98910481, 0.43697961],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

        scores = self.uf_sparse.transform(self.fdata_sparse, method="NumInt")
        expected_scores = np.array(
            [
                [-0.89442395, 0.70971822],
                [0.36734562, 0.05634948],
                [-1.15649441, -0.41380784],
                [0.31865209, 0.03148807],
                [-0.8948881, 0.34151457],
                [-0.88233623, -0.2697433],
                [1.15547143, -0.01671868],
                [1.412066, -0.08979359],
                [0.31381151, -0.33038824],
                [-1.03454889, -0.48550538],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_pace(self):
        scores = self.uf_dense.transform(self.fdata, method="PACE")
        expected_scores = np.array(
            [
                [-0.88222364, -0.60121093],
                [0.36987114, -0.10666329],
                [-1.12631358, 0.4741941],
                [0.25920688, -0.04525662],
                [-0.82933481, -0.48121928],
                [-0.75908108, 0.253283],
                [1.02691467, 0.0291445],
                [1.46814402, -0.01557031],
                [0.38008046, 0.52975236],
                [-0.97147338, 0.43265548],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

        scores = self.uf_sparse.transform(self.fdata_sparse, method="PACE")
        expected_scores = np.array(
            [
                [-0.89681266, 0.70437582],
                [0.36778328, 0.05841544],
                [-1.15924411, -0.41778326],
                [0.31231427, 0.02786705],
                [-0.89152932, 0.34329155],
                [-0.88168954, -0.26884701],
                [1.15567213, -0.01909334],
                [1.41186614, -0.09038651],
                [0.31220725, -0.33235442],
                [-1.03374449, -0.48544168],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_innpro(self):
        scores = self.uf_sparse.transform(method="InnPro")
        expected_scores = np.array(
            [
                [-0.76483481, 0.75512297],
                [0.49687213, 0.1032645],
                [-1.0271569, -0.36876321],
                [0.44776006, 0.07805723],
                [-0.76526654, 0.38737671],
                [-0.75269582, -0.22231556],
                [1.2852018, 0.02993049],
                [1.54152294, -0.04286759],
                [0.44307911, -0.28334118],
                [-0.90448198, -0.43646436],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )


class TestInverseTranform(unittest.TestCase):
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
        self.fdata = kl.noisy_data

        kl = KarhunenLoeve(
            basis_name="bsplines",
            n_functions=5,
            argvals=DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)}),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)
        self.fdata_sparse = kl.sparse_data

        kl = KarhunenLoeve(
            basis_name=("bsplines", "bsplines"),
            n_functions=(5, 5),
            argvals=DenseArgvals(
                {
                    "input_dim_0": np.linspace(0, 1, 11),
                    "input_dim_1": np.linspace(0, 1, 11),
                }
            ),
            random_state=42,
        )
        kl.new(n_obs=10)
        kl.add_noise(0.01)
        self.fdata_2D = kl.noisy_data

    def test_inverse_tranform_1D_dense(self):
        uf_dense = UFPCA(n_components=4, method="covariance")
        uf_dense.fit(self.fdata)
        scores_dense = uf_dense.transform(self.fdata)

        fdata_inv = uf_dense.inverse_transform(scores_dense)

        expected_values = np.array(
            [
                -0.48054239,
                -0.09984991,
                0.64309428,
                -0.06983845,
                -0.41097966,
                0.30151375,
                -0.01388041,
                -0.14367584,
                0.63674415,
                0.65409365,
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[:, 0], expected_values, decimal=4
        )

        expected_values = np.array(
            [
                0.2977152,
                -0.27638556,
                0.79490587,
                0.39484557,
                -0.10411061,
                0.39055344,
                -0.38760587,
                -0.48944742,
                0.17673476,
                0.38897018,
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[:, 10], expected_values, decimal=4
        )

    def test_inverse_tranform_1D_sparse(self):
        uf_sparse = UFPCA(n_components=0.95, method="covariance")
        uf_sparse.fit(self.fdata_sparse, method_smoothing="PS")
        scores_sparse = uf_sparse.transform(self.fdata_sparse)

        fdata_inv = uf_sparse.inverse_transform(scores_sparse)

        expected_values = np.array(
            [
                -0.88034793,
                -0.13631771,
                0.45563397,
                -0.22334293,
                -0.32652624,
                0.33553641,
                -0.08239232,
                -0.0216113,
                0.28940678,
                0.59146962,
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[:, 0], expected_values, decimal=4
        )

        expected_values = np.array(
            [
                -0.39781082,
                -0.05055594,
                0.56664499,
                -0.10810642,
                -0.00960558,
                0.44758819,
                -0.1201496,
                -0.11312394,
                0.25176664,
                0.64637646,
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[:, 10], expected_values, decimal=4
        )

    def test_inverse_tranform_2D(self):
        uf_2d = UFPCA(n_components=2, method="inner-product")
        uf_2d.fit(self.fdata_2D)
        scores_2d = uf_2d.transform(method="InnPro")

        fdata_inv = uf_2d.inverse_transform(scores_2d)

        expected_values = np.array(
            [
                [
                    -0.09316572,
                    -0.06726052,
                    -0.04921135,
                    0.00607605,
                    0.04456599,
                    0.01925676,
                    0.02057749,
                    0.02039293,
                    0.00536501,
                    -0.13131239,
                    -0.21273278,
                ],
                [
                    -0.11391025,
                    -0.0996121,
                    -0.01586282,
                    0.16137272,
                    0.19358487,
                    0.18575278,
                    0.24126688,
                    0.11516063,
                    0.05314113,
                    -0.03350556,
                    -0.0518647,
                ],
                [
                    -0.0574131,
                    0.07942835,
                    0.12211008,
                    0.1482357,
                    0.25267756,
                    0.26911108,
                    0.24282615,
                    0.2037064,
                    0.2081606,
                    0.05013568,
                    0.08312889,
                ],
                [
                    -0.05590392,
                    -0.03945514,
                    0.15955264,
                    0.2065392,
                    0.27346512,
                    0.29297388,
                    0.2942224,
                    0.29267663,
                    0.22975122,
                    0.19227882,
                    0.14319313,
                ],
                [
                    -0.03284785,
                    0.11276662,
                    0.13242248,
                    0.21265969,
                    0.23798019,
                    0.33009806,
                    0.38298878,
                    0.32243805,
                    0.29270657,
                    0.30032465,
                    0.26728115,
                ],
                [
                    -0.01600776,
                    0.04389263,
                    0.21842871,
                    0.23650378,
                    0.36793057,
                    0.41636759,
                    0.40813704,
                    0.44362227,
                    0.41012625,
                    0.28283882,
                    0.34404452,
                ],
                [
                    -0.00173176,
                    0.06098764,
                    0.12888656,
                    0.18520956,
                    0.31725701,
                    0.31183466,
                    0.33165889,
                    0.38291657,
                    0.34743178,
                    0.33374205,
                    0.35537755,
                ],
                [
                    0.1136048,
                    -0.00211084,
                    0.14033722,
                    0.17162998,
                    0.17432077,
                    0.21691127,
                    0.21399732,
                    0.32426024,
                    0.35463316,
                    0.33392125,
                    0.40873425,
                ],
                [
                    -0.01446918,
                    0.02415669,
                    -0.05603944,
                    0.09950247,
                    0.09700078,
                    0.19088191,
                    0.1461657,
                    0.09423951,
                    0.28469432,
                    0.3433914,
                    0.24677689,
                ],
                [
                    0.0305257,
                    0.08315056,
                    0.07647659,
                    -0.02112311,
                    0.06717157,
                    0.07836098,
                    0.12051799,
                    0.1762049,
                    0.15529979,
                    0.25569111,
                    0.20925096,
                ],
                [
                    -0.07744773,
                    -0.0719587,
                    -0.01246929,
                    -0.05807006,
                    -0.05867532,
                    -0.02096566,
                    -0.02530286,
                    0.06013469,
                    0.11708264,
                    0.20602768,
                    0.20255632,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[0], expected_values, decimal=4
        )

        expected_values = np.array(
            [
                [
                    -0.28907632,
                    -0.2338714,
                    -0.05686747,
                    -0.03767223,
                    0.01697277,
                    0.04265963,
                    0.27534124,
                    0.21998238,
                    0.24460025,
                    0.32967522,
                    0.33827724,
                ],
                [
                    -0.58520516,
                    -0.46314486,
                    -0.46267608,
                    -0.27060559,
                    -0.15524883,
                    -0.18733295,
                    0.00163684,
                    0.08636223,
                    0.08810669,
                    0.12683489,
                    0.22105962,
                ],
                [
                    -0.65338562,
                    -0.76805166,
                    -0.557185,
                    -0.49004773,
                    -0.32013913,
                    -0.39278131,
                    -0.12294625,
                    0.05870193,
                    0.11215132,
                    0.09559406,
                    0.16200517,
                ],
                [
                    -0.83364548,
                    -0.8791193,
                    -0.79230444,
                    -0.59135793,
                    -0.46737969,
                    -0.41865812,
                    -0.29511322,
                    -0.17000651,
                    -0.02887521,
                    -0.2060102,
                    0.09005959,
                ],
                [
                    -0.91544072,
                    -0.94083272,
                    -0.85383818,
                    -0.7638038,
                    -0.65104678,
                    -0.62609302,
                    -0.49863231,
                    -0.42866583,
                    -0.25275487,
                    -0.25409996,
                    -0.04213041,
                ],
                [
                    -0.96352414,
                    -0.7619857,
                    -0.87118596,
                    -0.89141118,
                    -0.84210716,
                    -0.8787222,
                    -0.57778769,
                    -0.65841581,
                    -0.38850681,
                    -0.33358416,
                    -0.23691252,
                ],
                [
                    -0.87236251,
                    -0.91481191,
                    -0.91629193,
                    -0.90168628,
                    -0.88321237,
                    -0.99917339,
                    -0.66308081,
                    -0.84279092,
                    -0.49106533,
                    -0.30572711,
                    -0.30150271,
                ],
                [
                    -0.69281975,
                    -0.80124834,
                    -0.98407888,
                    -1.09187607,
                    -1.12442553,
                    -1.04321413,
                    -0.98489215,
                    -0.71064326,
                    -0.69881411,
                    -0.50887663,
                    -0.22356372,
                ],
                [
                    -0.72271098,
                    -0.91114529,
                    -0.93805952,
                    -1.13557171,
                    -1.34088431,
                    -1.15207712,
                    -1.19601595,
                    -0.88766059,
                    -0.63776533,
                    -0.41982041,
                    -0.44825719,
                ],
                [
                    -0.89093562,
                    -0.83073555,
                    -1.16312234,
                    -1.04908743,
                    -1.30654536,
                    -1.02218315,
                    -0.93931272,
                    -0.86670183,
                    -0.56584905,
                    -0.29366976,
                    -0.27517221,
                ],
                [
                    -0.75693259,
                    -0.81943855,
                    -0.91915131,
                    -0.96353442,
                    -1.04628425,
                    -0.94573257,
                    -0.94298118,
                    -0.60341201,
                    -0.46945173,
                    -0.22998108,
                    -0.18342412,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[9], expected_values, decimal=4
        )
