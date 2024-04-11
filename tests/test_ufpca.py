#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class UFPCA in the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import pickle
import unittest
import warnings

from pathlib import Path

from FDApy.representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData,
)
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import UFPCA

THIS_DIR = Path(__file__)


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
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_sparse_5_10_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_noisy_5_10_001_2D.pickle"
        with open(fname, "rb") as handle:
            self.fdata_2D = pickle.load(handle)

    def test_fit_covariance(self):
        uf = UFPCA(n_components=2, method="covariance")
        uf.fit(self.fdata)

        np.testing.assert_almost_equal(
            self.fdata.argvals["input_dim_0"], uf.mean.argvals["input_dim_0"]
        )

        expected_mean = np.array(
            [
                [
                    0.07375568,
                    0.04519794,
                    0.0312277,
                    0.02851052,
                    0.03303859,
                    0.04118734,
                    0.05099911,
                    0.06640445,
                    0.08778816,
                    0.11086423,
                    0.127469,
                ]
            ]
        )
        np.testing.assert_array_almost_equal(uf.mean.values, expected_mean, decimal=2)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf.covariance, DenseFunctionalData)
        self.assertIsInstance(uf._training_data, DenseFunctionalData)

        expected_eigenvalues = np.array([0.17348295, 0.03006573])
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues, decimal=4
        )

        expected_eigenfunctions = np.array(
            [
                [
                    -0.26140165,
                    -0.41001957,
                    -0.61706022,
                    -0.82742515,
                    -1.02285102,
                    -1.17460184,
                    -1.25700869,
                    -1.2980176,
                    -1.24819755,
                    -1.08237038,
                    -0.78989346,
                ],
                [
                    -2.49346824,
                    -1.96667331,
                    -1.22552174,
                    -0.49410507,
                    0.06725894,
                    0.47075842,
                    0.59841628,
                    0.57365139,
                    0.38616865,
                    -0.04187332,
                    -0.65309416,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(uf.eigenfunctions.values), np.abs(expected_eigenfunctions), decimal=2
        )

        expected_noise = 0.022070664444049242
        np.testing.assert_array_almost_equal(
            uf._noise_variance, expected_noise, decimal=2
        )

    def test_fit_inner_product(self):
        uf = UFPCA(n_components=2, method="inner-product", normalize=True)
        uf.fit(self.fdata_sparse)

        np.testing.assert_almost_equal(
            self.fdata_sparse.argvals.to_dense()["input_dim_0"],
            uf.mean.argvals["input_dim_0"],
        )

        expected_mean = np.array(
            [
                [
                    0.12434887,
                    0.11908252,
                    0.11421908,
                    0.10947187,
                    0.10488939,
                    0.10070912,
                    0.09664516,
                    0.09253467,
                    0.08845083,
                    0.08452447,
                    0.08075368,
                    0.07721117,
                    0.07387828,
                    0.07066272,
                    0.06758798,
                    0.06467792,
                    0.06196308,
                    0.05943574,
                    0.05708085,
                    0.05488763,
                    0.05283921,
                    0.05092344,
                    0.0491439,
                    0.04749254,
                    0.04593048,
                    0.04442758,
                    0.0430032,
                    0.04167678,
                    0.04043997,
                    0.03928799,
                    0.03823654,
                    0.03728657,
                    0.03643484,
                    0.03570422,
                    0.03506928,
                    0.03451662,
                    0.03405478,
                    0.03362148,
                    0.03317949,
                    0.03274378,
                    0.03235744,
                    0.03198829,
                    0.03173275,
                    0.03167038,
                    0.031703,
                    0.03189464,
                    0.03230836,
                    0.03291306,
                    0.03363299,
                    0.03442316,
                    0.0352868,
                    0.0361709,
                    0.03705142,
                    0.03800144,
                    0.03911824,
                    0.04036386,
                    0.0417444,
                    0.04339252,
                    0.04524003,
                    0.04724243,
                    0.04936164,
                    0.05149668,
                    0.05364945,
                    0.05581475,
                    0.05805643,
                    0.06037518,
                    0.06275827,
                    0.06523721,
                    0.0677535,
                    0.07032589,
                    0.07296403,
                    0.07562365,
                    0.07832671,
                    0.08106611,
                    0.08382972,
                    0.08662793,
                    0.08944839,
                    0.09231851,
                    0.09526583,
                    0.09830031,
                    0.10144258,
                    0.10469438,
                    0.10805781,
                    0.11152237,
                    0.11507758,
                    0.11874588,
                    0.1225301,
                    0.12643069,
                    0.13054871,
                    0.13489754,
                    0.1394041,
                    0.14402535,
                    0.14879677,
                    0.15371587,
                    0.15888,
                    0.16442403,
                    0.17011047,
                    0.17583824,
                    0.18149437,
                    0.18717241,
                    0.19296673,
                ]
            ]
        )
        np.testing.assert_array_almost_equal(uf.mean.values, expected_mean, decimal=2)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf.covariance, DenseFunctionalData)
        self.assertIsInstance(uf._training_data, IrregularFunctionalData)

        expected_eigenvalues = np.array([0.82648629, 0.12032015])
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues, decimal=2
        )

        expected_eigenfunctions = np.array(
            [
                [
                    -0.25804867,
                    -0.27932304,
                    -0.30084046,
                    -0.3221831,
                    -0.34364446,
                    -0.36488276,
                    -0.3859989,
                    -0.40750712,
                    -0.42856828,
                    -0.44972095,
                    -0.47103873,
                    -0.49235777,
                    -0.51359851,
                    -0.53326029,
                    -0.55399432,
                    -0.57554721,
                    -0.59583097,
                    -0.61625451,
                    -0.63638554,
                    -0.65610505,
                    -0.67592675,
                    -0.69588001,
                    -0.71528983,
                    -0.73453819,
                    -0.75374703,
                    -0.77284414,
                    -0.79118691,
                    -0.80953128,
                    -0.82772451,
                    -0.84594621,
                    -0.86365634,
                    -0.88087917,
                    -0.89779426,
                    -0.91470066,
                    -0.93125522,
                    -0.94738404,
                    -0.96321235,
                    -0.97867718,
                    -0.99385677,
                    -1.00868076,
                    -1.02309939,
                    -1.03706258,
                    -1.05050804,
                    -1.06395739,
                    -1.07686452,
                    -1.08932202,
                    -1.10134777,
                    -1.11263768,
                    -1.12377135,
                    -1.13452014,
                    -1.14445591,
                    -1.15424073,
                    -1.16357127,
                    -1.17231391,
                    -1.18064935,
                    -1.18801205,
                    -1.19522835,
                    -1.20188011,
                    -1.20818039,
                    -1.21402569,
                    -1.21919882,
                    -1.22377207,
                    -1.22795861,
                    -1.23113196,
                    -1.2340682,
                    -1.23674334,
                    -1.23871667,
                    -1.23997857,
                    -1.24095032,
                    -1.24104944,
                    -1.24063924,
                    -1.23972967,
                    -1.23816617,
                    -1.23618992,
                    -1.23365891,
                    -1.2303695,
                    -1.22640373,
                    -1.22199948,
                    -1.21679691,
                    -1.21118716,
                    -1.20498422,
                    -1.19804621,
                    -1.19050284,
                    -1.18224781,
                    -1.17330815,
                    -1.16409656,
                    -1.15402972,
                    -1.14329739,
                    -1.13195844,
                    -1.11976981,
                    -1.10715435,
                    -1.09381305,
                    -1.07979774,
                    -1.06516712,
                    -1.04982244,
                    -1.03386764,
                    -1.0169792,
                    -0.99891463,
                    -0.98119388,
                    -0.96290072,
                    -0.94749348,
                ],
                [
                    -2.35986667,
                    -2.32743936,
                    -2.29122284,
                    -2.2532048,
                    -2.21385013,
                    -2.17307299,
                    -2.12997461,
                    -2.08172208,
                    -2.03164931,
                    -1.9801435,
                    -1.92639154,
                    -1.87087781,
                    -1.81312014,
                    -1.75352714,
                    -1.69289624,
                    -1.63042922,
                    -1.56629066,
                    -1.5013473,
                    -1.43577291,
                    -1.36762176,
                    -1.29960101,
                    -1.23132246,
                    -1.16183021,
                    -1.09190707,
                    -1.02166519,
                    -0.95096909,
                    -0.88095822,
                    -0.81037089,
                    -0.73973944,
                    -0.66881974,
                    -0.5995448,
                    -0.53058164,
                    -0.46215076,
                    -0.39404401,
                    -0.32701623,
                    -0.2613193,
                    -0.19654831,
                    -0.13322173,
                    -0.07085346,
                    -0.00989542,
                    0.049018,
                    0.10614599,
                    0.16132101,
                    0.21460657,
                    0.26528085,
                    0.31412784,
                    0.36115335,
                    0.40521231,
                    0.44664466,
                    0.48533466,
                    0.52079694,
                    0.55344693,
                    0.583586,
                    0.61101775,
                    0.6358963,
                    0.65602403,
                    0.6743082,
                    0.68972448,
                    0.70405286,
                    0.7145488,
                    0.72228815,
                    0.72762047,
                    0.7312297,
                    0.73089912,
                    0.72817839,
                    0.72435169,
                    0.71791949,
                    0.70899207,
                    0.69876966,
                    0.68463994,
                    0.66903887,
                    0.65175819,
                    0.63204754,
                    0.61110779,
                    0.58801648,
                    0.56280941,
                    0.53521479,
                    0.50677747,
                    0.47539732,
                    0.44335603,
                    0.40899022,
                    0.37380755,
                    0.3367975,
                    0.29870529,
                    0.25793283,
                    0.21666129,
                    0.17338712,
                    0.12972654,
                    0.08483708,
                    0.03760378,
                    -0.01030375,
                    -0.05913666,
                    -0.10900391,
                    -0.15959443,
                    -0.21218216,
                    -0.26557558,
                    -0.31970588,
                    -0.37384538,
                    -0.42867189,
                    -0.48474826,
                    -0.5414912,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(uf.eigenfunctions.values), np.abs(expected_eigenfunctions), decimal=2
        )

        expected_noise = 0.0002570086309893859
        np.testing.assert_array_almost_equal(
            uf._noise_variance, expected_noise, decimal=2
        )

    def test_fit_inner_product_2d(self):
        uf = UFPCA(n_components=2, method="inner-product")
        with np.testing.assert_warns(UserWarning):
            uf.fit(self.fdata_2D)

        expected_mean = np.array(
            [
                -0.01352835,
                0.00775575,
                0.03081815,
                0.05299966,
                0.07242965,
                0.08581233,
                0.09077358,
                0.08373683,
                0.0692899,
                0.04864807,
                0.02452676,
            ]
        )
        np.testing.assert_array_almost_equal(uf.mean.values[0, 1], expected_mean)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf._training_data, DenseFunctionalData)

        expected_eigenvalues = np.array([0.01612315, 0.01265476])
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues, decimal=2
        )

        expected_eigenfunctions = np.array(
            [
                1.3139411,
                1.87081038,
                1.80461182,
                1.70039748,
                2.47131274,
                2.3632094,
                2.08611778,
                1.61014559,
                0.79028416,
                0.83499408,
                0.44093447,
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

        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_sparse_5_10_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)

        uf_dense = UFPCA(n_components=2, method="covariance", normalize=True)
        uf_dense.fit(self.fdata)
        self.uf_dense = uf_dense

        uf_sparse = UFPCA(n_components=2, method="inner-product", normalize=True)
        uf_sparse.fit(self.fdata_sparse)
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
                [-0.71813626, 0.70199595],
                [0.51083051, 0.09968195],
                [-0.97477608, -0.42831123],
                [0.45342612, 0.13089741],
                [-0.69897548, 0.51259661],
                [-0.62289728, -0.22292284],
                [1.18303862, 0.01290259],
                [1.63508901, 0.04675456],
                [0.53674753, -0.50472232],
                [-0.84384281, -0.39056534],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_data_notnone(self):
        scores = self.uf_dense.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [-0.86730276, 0.66910337],
                [0.36166401, 0.06678937],
                [-1.12394259, -0.46120382],
                [0.30425961, 0.09800482],
                [-0.84814198, 0.47970402],
                [-0.77206379, -0.25581543],
                [1.03387212, -0.01998999],
                [1.48592251, 0.01386198],
                [0.38758102, -0.5376149],
                [-0.99300932, -0.42345793],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_numint(self):
        scores = self.uf_dense.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [-0.86730276, 0.66910337],
                [0.36166401, 0.06678937],
                [-1.12394259, -0.46120382],
                [0.30425961, 0.09800482],
                [-0.84814198, 0.47970402],
                [-0.77206379, -0.25581543],
                [1.03387212, -0.01998999],
                [1.48592251, 0.01386198],
                [0.38758102, -0.5376149],
                [-0.99300932, -0.42345793],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

        scores = self.uf_sparse.transform(self.fdata_sparse, method="NumInt")
        expected_scores = np.array(
            [
                [-0.8932048, 0.69979375],
                [0.36695391, 0.05850807],
                [-1.1550058, -0.41777177],
                [0.31874397, 0.02637056],
                [-0.89458385, 0.33462894],
                [-0.87920991, -0.25359555],
                [1.14092056, -0.01729806],
                [1.40986002, -0.0813921],
                [0.31362985, -0.32632655],
                [-1.0334082, -0.48816478],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_pace(self):
        scores = self.uf_dense.transform(self.fdata, method="PACE")
        expected_scores = np.array(
            [
                [-0.86933171, 0.5950928],
                [0.37310379, 0.09163224],
                [-1.13251564, -0.43804612],
                [0.25554044, 0.05680819],
                [-0.81787147, 0.46727045],
                [-0.7632135, -0.2351465],
                [1.02057667, -0.04169259],
                [1.4643344, 0.00384179],
                [0.36872137, -0.52001518],
                [-0.97586594, -0.41299218],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

        scores = self.uf_sparse.transform(self.fdata_sparse, method="PACE")
        expected_scores = np.array(
            [
                [-0.88881789, 0.70370574],
                [0.36455743, 0.05883119],
                [-1.16473573, -0.42045784],
                [0.30996748, 0.00158097],
                [-0.88005496, 0.34516535],
                [-0.88103995, -0.26969769],
                [1.15239354, -0.02683995],
                [1.41080436, -0.09305173],
                [0.31603538, -0.33594329],
                [-1.03573269, -0.4906334],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_innpro(self):
        scores = self.uf_sparse.transform(method="InnPro")
        expected_scores = np.array(
            [
                [-0.76330068, 0.75746039],
                [0.49783555, 0.10748418],
                [-1.02512119, -0.36659014],
                [0.45063713, 0.07953486],
                [-0.76287314, 0.39410496],
                [-0.75132855, -0.22128188],
                [1.28344831, 0.03107703],
                [1.54039375, -0.04074783],
                [0.44506665, -0.28071931],
                [-0.90323896, -0.43757783],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )


class TestInverseTranform(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)

        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_sparse_5_10_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_noisy_5_10_001_2D.pickle"
        with open(fname, "rb") as handle:
            self.fdata_2D = pickle.load(handle)

    def test_inverse_tranform_1D_dense(self):
        uf_dense = UFPCA(n_components=4, method="covariance")
        uf_dense.fit(self.fdata)
        scores_dense = uf_dense.transform(self.fdata)

        fdata_inv = uf_dense.inverse_transform(scores_dense)

        expected_values = np.array(
            [
                -0.40753127,
                -0.20568612,
                0.61244165,
                -0.1178565,
                -0.41643496,
                0.29597647,
                -0.03981844,
                -0.15375533,
                0.64277527,
                0.66541829,
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[:, 0], expected_values, decimal=1
        )

        expected_values = np.array(
            [
                0.4201324,
                -0.21217503,
                0.75269674,
                0.31538045,
                0.00652527,
                0.31589915,
                -0.31073473,
                -0.50342181,
                0.22993418,
                0.5165931,
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[:, 10], expected_values, decimal=1
        )

    def test_inverse_tranform_1D_sparse(self):
        uf_sparse = UFPCA(n_components=0.95, method="covariance")
        uf_sparse.fit(self.fdata_sparse)
        scores_sparse = uf_sparse.transform(self.fdata_sparse)

        fdata_inv = uf_sparse.inverse_transform(scores_sparse)

        expected_values = np.array(
            [
                -0.56773026,
                -0.10352928,
                0.70392221,
                0.10440785,
                -0.32077949,
                0.40179654,
                -0.12089451,
                -0.09089374,
                0.37721481,
                0.68984205,
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[:, 0], expected_values, decimal=2
        )

        expected_values = np.array(
            [
                -0.5042177,
                -0.12200339,
                0.57338791,
                -0.15786839,
                -0.08524415,
                0.41990794,
                -0.19766755,
                -0.19429954,
                0.22046656,
                0.65822151,
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[:, 10], expected_values, decimal=2
        )

    def test_inverse_tranform_2D(self):
        uf_2d = UFPCA(n_components=2, method="inner-product")
        uf_2d.fit(self.fdata_2D)
        scores_2d = uf_2d.transform(method="InnPro")

        fdata_inv = uf_2d.inverse_transform(scores_2d)

        expected_values = np.array(
            [
                [
                    0.1094295,
                    0.17149375,
                    0.12439368,
                    0.24519342,
                    0.31451756,
                    0.4013249,
                    0.29391825,
                    0.40683293,
                    0.14454507,
                    0.27175398,
                    0.03560844,
                ],
                [
                    -0.06706365,
                    0.08988733,
                    0.20497464,
                    0.37183021,
                    0.33400589,
                    0.30169995,
                    0.25192423,
                    0.43304894,
                    0.23672676,
                    0.01093451,
                    -0.08832987,
                ],
                [
                    -0.27006391,
                    -0.17366062,
                    -0.03617478,
                    0.13577246,
                    0.13800311,
                    0.26104594,
                    0.24802284,
                    0.24124059,
                    0.28397357,
                    0.12708824,
                    0.03075378,
                ],
                [
                    -0.52486996,
                    -0.43698585,
                    -0.27307558,
                    -0.19265109,
                    -0.00157285,
                    0.19190503,
                    0.276462,
                    0.25791863,
                    0.17914546,
                    0.07452619,
                    0.11117705,
                ],
                [
                    -0.72380819,
                    -0.72964671,
                    -0.51173136,
                    -0.28823353,
                    -0.02385011,
                    -0.0932831,
                    0.14677133,
                    0.20644401,
                    0.08537732,
                    0.26585804,
                    -0.02203303,
                ],
                [
                    -0.74809069,
                    -0.65595265,
                    -0.57548245,
                    -0.37885884,
                    -0.36100905,
                    -0.09025652,
                    0.00452398,
                    -0.05091665,
                    -0.06506112,
                    -0.01402704,
                    -0.04163908,
                ],
                [
                    -0.77991596,
                    -0.68051853,
                    -0.42310452,
                    -0.45644278,
                    -0.44693371,
                    -0.19896924,
                    -0.05191562,
                    0.11640772,
                    0.07675789,
                    0.02261674,
                    0.10349693,
                ],
                [
                    -0.66990303,
                    -0.68491039,
                    -0.64830976,
                    -0.43653267,
                    -0.32306582,
                    -0.1045599,
                    -0.11981531,
                    0.01387314,
                    -0.01998826,
                    -0.04605984,
                    0.07389039,
                ],
                [
                    -0.39095078,
                    -0.33928255,
                    -0.39448531,
                    -0.40451518,
                    -0.33298254,
                    -0.07682197,
                    -0.1130773,
                    -0.07986115,
                    -0.07284351,
                    0.03108472,
                    0.09144641,
                ],
                [
                    -0.38955268,
                    -0.37889801,
                    -0.34189724,
                    -0.32494767,
                    -0.1593196,
                    -0.08560623,
                    0.01497331,
                    -0.03227059,
                    0.10465092,
                    -0.01911082,
                    -0.07832927,
                ],
                [
                    -0.33337759,
                    -0.22694003,
                    0.02059245,
                    -0.10484901,
                    -0.09018026,
                    -0.12287236,
                    0.10260031,
                    0.03550698,
                    0.01221164,
                    -0.0224446,
                    -0.00618602,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[0], expected_values, decimal=2
        )

        expected_values = np.array(
            [
                [
                    2.46557033e-01,
                    2.08036217e-01,
                    3.25665733e-01,
                    3.87053023e-01,
                    3.95852169e-01,
                    3.63940526e-01,
                    3.11644408e-01,
                    3.24287443e-01,
                    2.46648220e-01,
                    1.33417178e-01,
                    1.14196306e-01,
                ],
                [
                    1.55046612e-01,
                    2.51001801e-01,
                    2.67391066e-01,
                    2.79063221e-01,
                    3.96872930e-01,
                    3.95365665e-01,
                    3.63431617e-01,
                    2.98767439e-01,
                    1.74748508e-01,
                    1.55700242e-01,
                    7.91641170e-02,
                ],
                [
                    1.56900118e-01,
                    1.99752701e-01,
                    2.46078149e-01,
                    2.52985093e-01,
                    3.16884484e-01,
                    3.07296714e-01,
                    2.68431256e-01,
                    2.10470223e-01,
                    1.78247141e-01,
                    1.46694570e-01,
                    8.87434051e-02,
                ],
                [
                    1.51101267e-01,
                    9.72435406e-02,
                    1.76168689e-01,
                    1.53074050e-01,
                    2.20461763e-01,
                    2.37227278e-01,
                    2.19472686e-01,
                    1.85834298e-01,
                    6.65918131e-02,
                    8.03711161e-02,
                    1.95878554e-02,
                ],
                [
                    5.85398110e-02,
                    8.74080935e-02,
                    1.09269675e-01,
                    1.60359452e-01,
                    1.42784843e-01,
                    1.69247463e-01,
                    1.61961984e-01,
                    1.24104791e-01,
                    1.84054416e-01,
                    6.57782238e-02,
                    2.17726050e-02,
                ],
                [
                    3.61646720e-02,
                    6.53454655e-02,
                    1.05048464e-01,
                    6.96157657e-02,
                    1.51823015e-01,
                    4.06316075e-02,
                    5.06118556e-02,
                    4.82438337e-02,
                    4.92870224e-02,
                    7.41057055e-03,
                    5.31977157e-02,
                ],
                [
                    9.68947202e-03,
                    2.27954268e-02,
                    4.14231743e-02,
                    7.93351793e-02,
                    5.71573518e-02,
                    3.89269050e-02,
                    6.92078887e-02,
                    1.63014839e-02,
                    1.48419602e-02,
                    -1.01900428e-02,
                    -3.97251485e-02,
                ],
                [
                    -1.36467916e-03,
                    1.78053020e-02,
                    2.38074724e-02,
                    4.95924858e-02,
                    3.64512223e-02,
                    8.31779761e-02,
                    4.21520244e-02,
                    2.30610139e-02,
                    -2.44978962e-02,
                    4.68961349e-02,
                    -6.39359755e-03,
                ],
                [
                    9.52907091e-04,
                    4.47539996e-02,
                    -3.41729388e-03,
                    -5.81830079e-03,
                    -1.01128401e-02,
                    -3.46067417e-02,
                    -2.21045298e-02,
                    6.91554076e-03,
                    -3.15681486e-02,
                    -3.81365936e-02,
                    1.28457847e-03,
                ],
                [
                    3.57869752e-02,
                    3.60636358e-03,
                    1.40640970e-02,
                    -2.38799672e-02,
                    1.19970468e-02,
                    1.78593653e-02,
                    3.47300172e-02,
                    4.20069605e-02,
                    -1.37840432e-02,
                    -1.49139541e-02,
                    -4.34809916e-03,
                ],
                [
                    -2.39690276e-02,
                    -2.31422423e-02,
                    2.91163377e-02,
                    2.57262056e-02,
                    1.28703479e-02,
                    3.48063225e-02,
                    -1.93822984e-03,
                    5.47547392e-03,
                    -3.88677840e-04,
                    -4.04628942e-02,
                    -1.12151848e-02,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[9], expected_values, decimal=2
        )
