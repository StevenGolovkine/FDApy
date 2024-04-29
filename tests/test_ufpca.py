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
from FDApy.preprocessing.dim_reduction.ufpca import UFPCA

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

        expected_eigenvalues = np.array([0.91755711, 0.13357827])
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

        expected_noise = 0.0014784661534153342
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

        expected_eigenvalues = np.array([0.01612375, 0.012662])
        np.testing.assert_array_almost_equal(
            uf.eigenvalues, expected_eigenvalues, decimal=2
        )

        expected_eigenfunctions = np.array(
            [
                1.52073378,
                1.66682625,
                1.81305859,
                1.90064879,
                1.92269403,
                1.83969782,
                1.6498485,
                1.39139402,
                1.08412894,
                0.77796513,
                0.50532293,
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
                [-0.84961868, 0.52266213],
                [0.35158542, 0.11840634],
                [-1.11557215, -0.55978978],
                [0.29235801, 0.12926122],
                [-0.84397875, 0.13565325],
                [-0.84315237, -0.3835477],
                [1.09042898, 0.1690216],
                [1.34674072, 0.15702849],
                [0.29198374, -0.23862421],
                [-0.99395738, -0.6287577],
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
                [-1.07040075, 0.78609478],
                [0.3708355, 0.06056965],
                [-1.15617377, -0.40063039],
                [0.29850589, 0.08098738],
                [-0.97070503, 0.3451164],
                [-0.87228797, -0.27915615],
                [1.21639805, -0.05278968],
                [1.49879089, -0.11846076],
                [0.38207405, -0.34499154],
                [-0.99263135, -0.50533816],
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
                    0.08628559,
                    0.16630144,
                    0.24342878,
                    0.29216678,
                    0.32812967,
                    0.33761358,
                    0.32983935,
                    0.28647565,
                    0.22478143,
                    0.13463809,
                    0.00535322,
                ],
                [
                    -0.10521953,
                    -0.00203116,
                    0.09449094,
                    0.17491122,
                    0.24208555,
                    0.2837958,
                    0.29323158,
                    0.26233923,
                    0.20929874,
                    0.12950891,
                    0.01737025,
                ],
                [
                    -0.30187431,
                    -0.192011,
                    -0.07889221,
                    0.0264817,
                    0.12028076,
                    0.19257083,
                    0.22337676,
                    0.21688556,
                    0.18090308,
                    0.1135388,
                    0.01837875,
                ],
                [
                    -0.46970765,
                    -0.35763812,
                    -0.23733889,
                    -0.11888651,
                    -0.00757131,
                    0.08371944,
                    0.13952351,
                    0.15669768,
                    0.13969974,
                    0.08944235,
                    0.01993622,
                ],
                [
                    -0.59781707,
                    -0.49261394,
                    -0.37298412,
                    -0.24628739,
                    -0.12168247,
                    -0.01848792,
                    0.05543223,
                    0.09228385,
                    0.09669612,
                    0.06914308,
                    0.029137,
                ],
                [
                    -0.65925074,
                    -0.56921603,
                    -0.46395451,
                    -0.34362466,
                    -0.21500605,
                    -0.10172552,
                    -0.01944534,
                    0.02924784,
                    0.0529318,
                    0.05196739,
                    0.04085681,
                ],
                [
                    -0.65193727,
                    -0.57906499,
                    -0.48907102,
                    -0.38126947,
                    -0.26473788,
                    -0.15402657,
                    -0.07103228,
                    -0.01365344,
                    0.01935875,
                    0.03131384,
                    0.03845882,
                ],
                [
                    -0.59111176,
                    -0.53162091,
                    -0.45439808,
                    -0.36247913,
                    -0.26484088,
                    -0.16939693,
                    -0.09129545,
                    -0.03246046,
                    0.00103945,
                    0.01554371,
                    0.02942308,
                ],
                [
                    -0.49814388,
                    -0.44782983,
                    -0.38406125,
                    -0.30697914,
                    -0.2282144,
                    -0.15212896,
                    -0.08414261,
                    -0.03446832,
                    -0.00491415,
                    0.00832927,
                    0.02499033,
                ],
                [
                    -0.38020742,
                    -0.33500514,
                    -0.28506156,
                    -0.22143053,
                    -0.16063661,
                    -0.105682,
                    -0.05913206,
                    -0.02317417,
                    -0.00353432,
                    0.00505654,
                    0.01515176,
                ],
                [
                    -0.26211094,
                    -0.20009551,
                    -0.15822904,
                    -0.11080573,
                    -0.06829196,
                    -0.03720733,
                    -0.01566865,
                    0.00519518,
                    0.01122661,
                    0.00314246,
                    -0.0144254,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[0], expected_values, decimal=2
        )

        expected_values = np.array(
            [
                [
                    2.22728027e-01,
                    2.68275080e-01,
                    3.15564963e-01,
                    3.48210420e-01,
                    3.67975016e-01,
                    3.68177358e-01,
                    3.44100765e-01,
                    2.98690902e-01,
                    2.41370878e-01,
                    1.76438489e-01,
                    1.00961906e-01,
                ],
                [
                    1.81085982e-01,
                    2.23274304e-01,
                    2.67053472e-01,
                    3.01990483e-01,
                    3.25436271e-01,
                    3.28765351e-01,
                    3.09261110e-01,
                    2.68188872e-01,
                    2.13029564e-01,
                    1.51317557e-01,
                    8.97624022e-02,
                ],
                [
                    1.44082423e-01,
                    1.76493440e-01,
                    2.13507479e-01,
                    2.46186144e-01,
                    2.67101902e-01,
                    2.71272743e-01,
                    2.56445113e-01,
                    2.24030791e-01,
                    1.76684969e-01,
                    1.21145579e-01,
                    6.68361012e-02,
                ],
                [
                    1.09374202e-01,
                    1.32753289e-01,
                    1.62253662e-01,
                    1.89721785e-01,
                    2.07959026e-01,
                    2.12261438e-01,
                    1.99969194e-01,
                    1.73258092e-01,
                    1.35250671e-01,
                    8.86842166e-02,
                    4.45113844e-02,
                ],
                [
                    7.68355804e-02,
                    9.41680279e-02,
                    1.15985554e-01,
                    1.36325940e-01,
                    1.52747937e-01,
                    1.57584440e-01,
                    1.45771222e-01,
                    1.22338708e-01,
                    9.30007634e-02,
                    6.01267871e-02,
                    2.80569530e-02,
                ],
                [
                    4.95773605e-02,
                    6.21320069e-02,
                    7.64610660e-02,
                    8.92966916e-02,
                    1.00968901e-01,
                    1.04690476e-01,
                    9.49111708e-02,
                    7.75391859e-02,
                    5.56330661e-02,
                    3.33221016e-02,
                    1.24965776e-02,
                ],
                [
                    3.15745766e-02,
                    3.91965548e-02,
                    4.77271782e-02,
                    5.55434643e-02,
                    6.07246972e-02,
                    6.38305078e-02,
                    5.80967678e-02,
                    4.47363289e-02,
                    2.88064667e-02,
                    1.39538195e-02,
                    4.60392733e-03,
                ],
                [
                    1.82858425e-02,
                    2.27690252e-02,
                    2.92605168e-02,
                    3.37098990e-02,
                    3.73434534e-02,
                    3.73968354e-02,
                    3.24317810e-02,
                    2.31001571e-02,
                    1.21476633e-02,
                    1.03096776e-03,
                    -6.21440213e-04,
                ],
                [
                    8.69301110e-03,
                    9.95705512e-03,
                    1.39236706e-02,
                    2.02430551e-02,
                    2.32242546e-02,
                    2.10140546e-02,
                    1.62326049e-02,
                    9.23236495e-03,
                    1.96272699e-04,
                    -8.10129251e-03,
                    -6.18748208e-03,
                ],
                [
                    2.03482749e-03,
                    3.51358190e-03,
                    5.69172483e-03,
                    1.07299575e-02,
                    1.38713051e-02,
                    1.23430533e-02,
                    8.34467935e-03,
                    4.18613694e-03,
                    -4.39038358e-03,
                    -1.10493280e-02,
                    -1.04378612e-02,
                ],
                [
                    -7.43337391e-03,
                    -3.64815548e-04,
                    4.85164361e-03,
                    9.82258649e-03,
                    1.38201723e-02,
                    1.37120674e-02,
                    1.03541819e-02,
                    7.22306350e-03,
                    -4.69483908e-03,
                    -1.72165311e-02,
                    -2.37099295e-02,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(
            fdata_inv.values[9], expected_values, decimal=2
        )
