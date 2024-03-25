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

        expected_eigenvalues = np.array([5.28380137, 0.76922014])
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues, decimal=2
        )

        expected_eigenfunctions = np.array(
            [
                [
                    -0.2578074,
                    -0.27908691,
                    -0.30060415,
                    -0.32198036,
                    -0.34343666,
                    -0.36467848,
                    -0.38580011,
                    -0.40730861,
                    -0.42834317,
                    -0.44951658,
                    -0.47083105,
                    -0.49212604,
                    -0.51331456,
                    -0.53311523,
                    -0.55385421,
                    -0.57527796,
                    -0.59558192,
                    -0.61602855,
                    -0.63622459,
                    -0.65582999,
                    -0.67565267,
                    -0.69567779,
                    -0.715066,
                    -0.73429898,
                    -0.7534854,
                    -0.77257022,
                    -0.7909056,
                    -0.80923993,
                    -0.82742985,
                    -0.84565182,
                    -0.8633381,
                    -0.88056262,
                    -0.89747098,
                    -0.91436418,
                    -0.93091682,
                    -0.9470406,
                    -0.96286309,
                    -0.97832542,
                    -0.993506,
                    -1.00831999,
                    -1.02274004,
                    -1.03670255,
                    -1.05015877,
                    -1.06361051,
                    -1.07649907,
                    -1.08893638,
                    -1.10093631,
                    -1.11229226,
                    -1.123441,
                    -1.13417844,
                    -1.14417235,
                    -1.15394787,
                    -1.16327113,
                    -1.17201041,
                    -1.18037713,
                    -1.1877906,
                    -1.19499162,
                    -1.20163152,
                    -1.20793701,
                    -1.21379986,
                    -1.2189958,
                    -1.22358009,
                    -1.22777609,
                    -1.2309555,
                    -1.23390118,
                    -1.236584,
                    -1.23857075,
                    -1.2398475,
                    -1.24083323,
                    -1.24094566,
                    -1.24054445,
                    -1.2396446,
                    -1.23808475,
                    -1.2361256,
                    -1.23361293,
                    -1.23031861,
                    -1.2263564,
                    -1.22196244,
                    -1.21675477,
                    -1.21115514,
                    -1.20496038,
                    -1.198021,
                    -1.1904776,
                    -1.18222646,
                    -1.17327426,
                    -1.1640479,
                    -1.15395124,
                    -1.1432335,
                    -1.13185811,
                    -1.11964043,
                    -1.10700208,
                    -1.0936518,
                    -1.0795451,
                    -1.06492124,
                    -1.0494966,
                    -1.03347767,
                    -1.01658391,
                    -0.99844886,
                    -0.98067264,
                    -0.96232346,
                    -0.94634513,
                ],
                [
                    -2.36058452,
                    -2.32794805,
                    -2.29177115,
                    -2.25376233,
                    -2.21432699,
                    -2.17339354,
                    -2.13014277,
                    -2.08181714,
                    -2.0316542,
                    -1.9800323,
                    -1.92620813,
                    -1.87069162,
                    -1.81293982,
                    -1.7532874,
                    -1.69253335,
                    -1.6300078,
                    -1.56595821,
                    -1.50114721,
                    -1.43557233,
                    -1.36702523,
                    -1.29895322,
                    -1.2308727,
                    -1.16130475,
                    -1.09132205,
                    -1.02104189,
                    -0.9503335,
                    -0.88031609,
                    -0.80971815,
                    -0.73906999,
                    -0.66809553,
                    -0.59879204,
                    -0.52985867,
                    -0.46140108,
                    -0.39328673,
                    -0.3262529,
                    -0.26055294,
                    -0.19577268,
                    -0.13243576,
                    -0.07007526,
                    -0.00909633,
                    0.04982605,
                    0.1069604,
                    0.16211476,
                    0.2154325,
                    0.26615784,
                    0.3150132,
                    0.36206132,
                    0.40600221,
                    0.44743932,
                    0.48613041,
                    0.52157238,
                    0.55420033,
                    0.58433943,
                    0.61178187,
                    0.63658811,
                    0.65671998,
                    0.67501125,
                    0.69043967,
                    0.70462677,
                    0.71517537,
                    0.72286798,
                    0.72816472,
                    0.73173279,
                    0.73139465,
                    0.72864068,
                    0.7247944,
                    0.71833222,
                    0.70936997,
                    0.6991103,
                    0.68493799,
                    0.66929659,
                    0.65199798,
                    0.63226995,
                    0.61130356,
                    0.58816258,
                    0.56293225,
                    0.53533156,
                    0.50687081,
                    0.47546178,
                    0.44338564,
                    0.40892256,
                    0.37373576,
                    0.33672513,
                    0.29865538,
                    0.25781701,
                    0.21649587,
                    0.17311658,
                    0.12948781,
                    0.08463455,
                    0.03729659,
                    -0.01067262,
                    -0.05951121,
                    -0.10952561,
                    -0.16014535,
                    -0.21299237,
                    -0.2664211,
                    -0.320499,
                    -0.37474011,
                    -0.42963983,
                    -0.48574594,
                    -0.54247383,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(uf.eigenfunctions.values), np.abs(expected_eigenfunctions), decimal=2
        )

        expected_noise = 0.008553043015890941
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
                [-1.64421421, 1.56853553],
                [1.09649423, 0.22629428],
                [-2.21610859, -0.951685],
                [0.96801524, 0.2949224],
                [-1.60091261, 1.1470657],
                [-1.43127658, -0.49318871],
                [2.59511428, 0.03275203],
                [3.60314587, 0.10832757],
                [1.15399333, -1.12182232],
                [-1.92410043, -0.8670596],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_data_notnone(self):
        scores = self.uf_dense.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [-1.93425629, 1.49130805],
                [0.80645215, 0.1490668],
                [-2.50615067, -1.02891249],
                [0.67797316, 0.21769491],
                [-1.89095469, 1.06983821],
                [-1.72131866, -0.57041619],
                [2.3050722, -0.04447545],
                [3.31310379, 0.03110009],
                [0.86395125, -1.1990498],
                [-2.21414251, -0.94428708],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_numint(self):
        scores = self.uf_dense.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [-1.93425629, 1.49130805],
                [0.80645215, 0.1490668],
                [-2.50615067, -1.02891249],
                [0.67797316, 0.21769491],
                [-1.89095469, 1.06983821],
                [-1.72131866, -0.57041619],
                [2.3050722, -0.04447545],
                [3.31310379, 0.03110009],
                [0.86395125, -1.1990498],
                [-2.21414251, -0.94428708],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

        scores = self.uf_sparse.transform(self.fdata_sparse, method="NumInt")
        expected_scores = np.array(
            [
                [-2.25810039, 1.76977377],
                [0.9275964, 0.14770703],
                [-2.91964257, -1.05593689],
                [0.80570652, 0.06577182],
                [-2.26145304, 0.84712699],
                [-2.22247276, -0.6405693],
                [2.88412986, -0.04453739],
                [3.56396845, -0.20677979],
                [0.79287282, -0.82557932],
                [-2.61222714, -1.2336566],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_pace(self):
        scores = self.uf_dense.transform(self.fdata, method="PACE")
        expected_scores = np.array(
            [
                [-1.93899413, 1.32706428],
                [0.83200798, 0.20451005],
                [-2.5253765, -0.97756639],
                [0.56945161, 0.12597842],
                [-1.82370757, 1.04257981],
                [-1.70169819, -0.52452527],
                [2.27562252, -0.09293715],
                [3.26523933, 0.00864074],
                [0.82207418, -1.16024722],
                [-2.17604368, -0.92121085],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

        scores = self.uf_sparse.transform(self.fdata_sparse, method="PACE")
        expected_scores = np.array(
            [
                [-2.24734585e00, 1.77857250e00],
                [9.21988297e-01, 1.48858591e-01],
                [-2.94590867e00, -1.06371000e00],
                [7.83835091e-01, 3.17037548e-03],
                [-2.22538679e00, 8.72811492e-01],
                [-2.22833366e00, -6.82003161e-01],
                [2.91433258e00, -6.75658838e-02],
                [3.56780782e00, -2.34912076e-01],
                [7.99024953e-01, -8.49464542e-01],
                [-2.61970028e00, -1.24064258e00],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_innpro(self):
        scores = self.uf_sparse.transform(method="InnPro")
        expected_scores = np.array(
            [
                [-1.94074407, 1.91480394],
                [1.24833135, 0.27229558],
                [-2.6024512, -0.92678885],
                [1.12887058, 0.20051392],
                [-1.93933264, 0.99709652],
                [-1.9101899, -0.55956026],
                [3.23454221, 0.07879998],
                [3.8841787, -0.10277679],
                [1.11472293, -0.71006512],
                [-2.29445915, -1.10641985],
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
