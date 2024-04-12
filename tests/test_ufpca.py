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
                    -0.26565157,
                    -0.28654435,
                    -0.30732559,
                    -0.32798638,
                    -0.34852751,
                    -0.36894862,
                    -0.38923274,
                    -0.40937847,
                    -0.42938064,
                    -0.4492351,
                    -0.46893784,
                    -0.48847862,
                    -0.50785281,
                    -0.52705364,
                    -0.54607605,
                    -0.56491515,
                    -0.5835664,
                    -0.6020246,
                    -0.62028308,
                    -0.63833641,
                    -0.65617827,
                    -0.67380294,
                    -0.6912065,
                    -0.70838415,
                    -0.72533113,
                    -0.74204286,
                    -0.75851312,
                    -0.77473848,
                    -0.7907156,
                    -0.8064399,
                    -0.82191026,
                    -0.83712637,
                    -0.85208321,
                    -0.86678422,
                    -0.88122861,
                    -0.89541797,
                    -0.90935898,
                    -0.92305673,
                    -0.93652172,
                    -0.94976995,
                    -0.96277478,
                    -0.97542334,
                    -0.98770899,
                    -0.99961799,
                    -1.01113465,
                    -1.02225608,
                    -1.03298143,
                    -1.04329782,
                    -1.05319157,
                    -1.06265512,
                    -1.07168617,
                    -1.08027215,
                    -1.08841409,
                    -1.09609738,
                    -1.10331625,
                    -1.11007072,
                    -1.11635345,
                    -1.12216335,
                    -1.12749493,
                    -1.13233311,
                    -1.13667104,
                    -1.1404978,
                    -1.14404692,
                    -1.14737769,
                    -1.15045275,
                    -1.15324327,
                    -1.15572826,
                    -1.1578833,
                    -1.15968816,
                    -1.16114353,
                    -1.16223486,
                    -1.16295393,
                    -1.16329058,
                    -1.16323881,
                    -1.16279202,
                    -1.16194725,
                    -1.16069716,
                    -1.15903548,
                    -1.1569609,
                    -1.15446904,
                    -1.15155629,
                    -1.14821881,
                    -1.14445415,
                    -1.14025924,
                    -1.13563147,
                    -1.13056911,
                    -1.12506894,
                    -1.11912915,
                    -1.11274739,
                    -1.10592337,
                    -1.09865711,
                    -1.09094479,
                    -1.08278665,
                    -1.07417727,
                    -1.06511326,
                    -1.05558973,
                    -1.04560963,
                    -1.03518069,
                    -1.02428883,
                    -1.01294345,
                    -1.00115576,
                ],
                [
                    -2.49124953,
                    -2.43368581,
                    -2.3753665,
                    -2.3163261,
                    -2.2566104,
                    -2.19630305,
                    -2.13546735,
                    -2.07416162,
                    -2.01245736,
                    -1.95042006,
                    -1.88808764,
                    -1.82555803,
                    -1.76290469,
                    -1.70019129,
                    -1.63748535,
                    -1.57485465,
                    -1.51236705,
                    -1.45008984,
                    -1.38809537,
                    -1.32644936,
                    -1.26522014,
                    -1.20447033,
                    -1.14426175,
                    -1.08465371,
                    -1.02569875,
                    -0.9674536,
                    -0.9099698,
                    -0.85329786,
                    -0.79748488,
                    -0.74257807,
                    -0.6886151,
                    -0.63563048,
                    -0.583656,
                    -0.53270523,
                    -0.4827991,
                    -0.4339481,
                    -0.38614561,
                    -0.33938895,
                    -0.29365592,
                    -0.24890268,
                    -0.20513328,
                    -0.1625338,
                    -0.12118176,
                    -0.08113923,
                    -0.04248082,
                    -0.00524478,
                    0.03050471,
                    0.06469851,
                    0.09729653,
                    0.12823854,
                    0.15749337,
                    0.18502598,
                    0.21079043,
                    0.23475596,
                    0.25690584,
                    0.27718721,
                    0.29558449,
                    0.31212319,
                    0.32677677,
                    0.33948174,
                    0.35023208,
                    0.35902166,
                    0.36663377,
                    0.37324565,
                    0.37872132,
                    0.38294093,
                    0.38580915,
                    0.38722737,
                    0.38712787,
                    0.38550665,
                    0.38230467,
                    0.37751562,
                    0.37111026,
                    0.36308077,
                    0.35341672,
                    0.34212144,
                    0.32919494,
                    0.31463796,
                    0.29846613,
                    0.28068996,
                    0.26132521,
                    0.24038813,
                    0.217897,
                    0.19387256,
                    0.16833929,
                    0.14132138,
                    0.11284405,
                    0.08293322,
                    0.05161142,
                    0.01890954,
                    -0.01512417,
                    -0.05048228,
                    -0.08713823,
                    -0.12506553,
                    -0.16424239,
                    -0.20464781,
                    -0.24624501,
                    -0.28896573,
                    -0.33283525,
                    -0.37780328,
                    -0.42379269,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(uf.eigenfunctions.values), np.abs(expected_eigenfunctions), decimal=2
        )

        expected_noise = 0.0013317231007740737
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
