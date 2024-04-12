#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class MFPCA in the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import pickle
import unittest
import warnings

from pathlib import Path

from FDApy.representation.functional_data import MultivariateFunctionalData
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import MFPCA

THIS_DIR = Path(__file__)


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
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_sparse_5_10_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_sparse])

    def test_fit_error_method(self):
        mfpca = MFPCA(method="error", n_components=[0.95, 0.99], normalize=True)
        with self.assertRaises(NotImplementedError):
            mfpca.fit(self.fdata)

    def test_fit_covariance(self):
        mfpca = MFPCA(method="covariance", n_components=[0.95, 3], normalize=True)
        mfpca.fit(data=self.fdata)

        expected_eigenvalues = np.array(
            [
                1.78133863e00,
                2.85754552e-01,
                5.71406498e-02,
                3.90640800e-03,
                1.39986051e-03,
            ]
        )
        np.testing.assert_array_almost_equal(
            mfpca.eigenvalues, expected_eigenvalues, decimal=3
        )

        expected_eigenfunctions_0 = np.array(
            [
                0.17757411,
                0.2828946,
                0.42949634,
                0.57835542,
                0.71636339,
                0.82347509,
                0.88148227,
                0.91016916,
                0.87486334,
                0.75779305,
                0.55165734,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(mfpca.eigenfunctions.data[0].values[0]),
            np.abs(expected_eigenfunctions_0),
            decimal=2,
        )

        expected_eigenfunctions_1 = np.array(
            [
                0.21310205,
                0.24824758,
                0.27079077,
                0.28185648,
                0.28602236,
                0.28778952,
                0.29078421,
                0.29591964,
                0.30233557,
                0.30994681,
                0.31879701,
                0.32862705,
                0.33836315,
                0.35007093,
                0.3640333,
                0.37891935,
                0.3951793,
                0.41316612,
                0.43299058,
                0.45284257,
                0.47194979,
                0.49084483,
                0.50965421,
                0.52647777,
                0.54189143,
                0.55720627,
                0.57301159,
                0.58778556,
                0.60077967,
                0.6139826,
                0.62776982,
                0.64130531,
                0.65408496,
                0.66517465,
                0.67527274,
                0.68366953,
                0.69102732,
                0.69778453,
                0.70376832,
                0.70806417,
                0.71105725,
                0.7142657,
                0.71826179,
                0.72307899,
                0.73056653,
                0.74131775,
                0.75303167,
                0.76572266,
                0.77866301,
                0.7913789,
                0.80524551,
                0.81929954,
                0.83271648,
                0.84582109,
                0.85623936,
                0.86309011,
                0.8679418,
                0.87214307,
                0.87446663,
                0.8745462,
                0.87365288,
                0.87253035,
                0.86945916,
                0.86549027,
                0.86244617,
                0.86191869,
                0.86351771,
                0.86722701,
                0.87555693,
                0.88633575,
                0.89721359,
                0.90697645,
                0.91455656,
                0.92041797,
                0.92427819,
                0.92669364,
                0.92642735,
                0.92269909,
                0.91606644,
                0.90692148,
                0.89679041,
                0.88663746,
                0.87817136,
                0.87186145,
                0.86723602,
                0.86196497,
                0.85432351,
                0.8448153,
                0.83434232,
                0.82336799,
                0.81102675,
                0.7971638,
                0.78114694,
                0.76196026,
                0.73756479,
                0.70651142,
                0.66729537,
                0.61848721,
                0.5617167,
                0.50097342,
                0.43753327,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(mfpca.eigenfunctions.data[1].values[0]),
            np.abs(expected_eigenfunctions_1),
            decimal=2,
        )

        self.assertIsNone(mfpca.covariance)

    def test_fit_inner_product(self):
        mfpca = MFPCA(method="inner-product", n_components=0.99, normalize=True)
        mfpca.fit(data=self.fdata)

        expected_eigenvalues = np.array([1.60178227, 0.25340301, 0.0900525])
        np.testing.assert_array_almost_equal(
            mfpca.eigenvalues, expected_eigenvalues, decimal=2
        )

        expected_eigenfunctions_0 = np.array(
            [
                -0.16098404,
                -0.30107645,
                -0.42954078,
                -0.53949143,
                -0.62511116,
                -0.68485645,
                -0.73651596,
                -0.76747476,
                -0.76788101,
                -0.73153452,
                -0.65565047,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(mfpca.eigenfunctions.data[0].values[0]),
            np.abs(expected_eigenfunctions_0),
            decimal=1,
        )

        expected_eigenfunctions_1 = np.array(
            [
                -0.18483015,
                -0.19979107,
                -0.21469477,
                -0.22950673,
                -0.24428413,
                -0.25903177,
                -0.27367655,
                -0.28820747,
                -0.30260352,
                -0.31691177,
                -0.33115345,
                -0.34529475,
                -0.35934331,
                -0.37329826,
                -0.38714188,
                -0.4008514,
                -0.41441776,
                -0.42784654,
                -0.44113299,
                -0.45427335,
                -0.46724673,
                -0.48003386,
                -0.49264975,
                -0.50510153,
                -0.51738713,
                -0.52950174,
                -0.5414148,
                -0.55314169,
                -0.56468241,
                -0.57600833,
                -0.58714362,
                -0.59813474,
                -0.60894327,
                -0.61953449,
                -0.62994851,
                -0.64016533,
                -0.65016665,
                -0.6599837,
                -0.66957865,
                -0.67896324,
                -0.68818076,
                -0.69724482,
                -0.70613979,
                -0.71474247,
                -0.72303482,
                -0.73106047,
                -0.73880753,
                -0.74621187,
                -0.75323262,
                -0.75988171,
                -0.76623127,
                -0.77224549,
                -0.77786085,
                -0.78318847,
                -0.78824325,
                -0.7930093,
                -0.79748545,
                -0.80166557,
                -0.80551452,
                -0.80898411,
                -0.8122561,
                -0.81540322,
                -0.81831872,
                -0.82099331,
                -0.82346638,
                -0.82571275,
                -0.82768485,
                -0.82936972,
                -0.83078788,
                -0.83202616,
                -0.83311994,
                -0.83399059,
                -0.83457459,
                -0.83489543,
                -0.83496014,
                -0.83475384,
                -0.83425928,
                -0.83345222,
                -0.83235,
                -0.83094176,
                -0.82921551,
                -0.82717417,
                -0.82481416,
                -0.82212949,
                -0.81913213,
                -0.81582371,
                -0.81217753,
                -0.80820039,
                -0.80392197,
                -0.7993379,
                -0.79447457,
                -0.789262,
                -0.78366421,
                -0.77773587,
                -0.77142386,
                -0.76477249,
                -0.75785649,
                -0.75074337,
                -0.74331815,
                -0.73547625,
                -0.72734096,
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

        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_sparse_5_10_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_sparse])

        mfpca_cov = MFPCA(n_components=[2, 2], method="covariance", normalize=True)
        mfpca_cov.fit(self.fdata)
        self.mfpca_cov = mfpca_cov

        mfpca_inn = MFPCA(n_components=2, method="inner-product", normalize=True)
        mfpca_inn.fit(self.fdata)
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
                [1.0111459, 1.02657591, 0.08934055, -0.00390411],
                [-0.74719645, 0.15568283, 0.01984571, -0.01595666],
                [1.37016116, -0.56207689, 0.02112094, -0.02988672],
                [-0.67889633, 0.14852816, -0.03501114, -0.02775543],
                [1.00232072, 0.64422207, -0.04903482, 0.01457348],
                [0.93162256, -0.30035794, 0.00603123, 0.0371346],
                [-1.7688418, 0.04060232, 0.01535549, -0.0779793],
                [-2.27653986, 0.02399045, -0.04750734, 0.0595043],
                [-0.73125516, -0.55390327, 0.13973369, 0.02194336],
                [1.19548704, -0.58098918, -0.05581468, -0.01519743],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_data_notnone(self):
        scores = self.mfpca_cov.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [1.24465970e00, 9.60887606e-01, 6.77698358e-02, 1.61054857e-02],
                [-5.13686118e-01, 9.00092840e-02, -1.70895099e-03, 4.04914227e-03],
                [1.60367561e00, -6.27761207e-01, -4.45359685e-04, -9.87659120e-03],
                [-4.46542237e-01, 9.08335425e-02, -4.78444021e-02, -9.09674013e-03],
                [1.23576995e00, 5.78449798e-01, -7.07016856e-02, 3.45220844e-02],
                [1.16454917e00, -3.61912116e-01, -1.10199933e-02, 5.64578684e-02],
                [-1.53728047e00, -2.38163747e-02, -4.92786479e-03, -5.99163234e-02],
                [-2.04322600e00, -4.18840485e-02, -6.92944612e-02, 7.93230561e-02],
                [-4.97744624e-01, -6.19543913e-01, 1.18215271e-01, 4.19484784e-02],
                [1.42893113e00, -6.46824267e-01, -7.75510158e-02, 4.74784080e-03],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_numint(self):
        scores = self.mfpca_inn.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [-1.1376663, 0.58417021],
                [0.48243942, 0.20879993],
                [-1.52197399, -0.9462889],
                [0.38412958, 0.28888895],
                [-1.11322532, 0.15130436],
                [-1.09334108, -0.63685117],
                [1.42847731, 0.37881989],
                [1.89220351, 0.50553255],
                [0.43223318, -0.38321874],
                [-1.33924529, -0.96401581],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=0
        )

    def test_innpro(self):
        scores = self.mfpca_inn.transform(method="InnPro")
        expected_scores = np.array(
            [
                [-1.07193063, 1.01367552],
                [0.68725499, 0.15538091],
                [-1.42896651, -0.56507026],
                [0.61588108, 0.14282348],
                [-1.05600416, 0.64516147],
                [-0.988859, -0.31257721],
                [1.71976261, 0.04364545],
                [2.21514592, 0.01394986],
                [0.67318003, -0.54532433],
                [-1.25083976, -0.5737856],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_pace(self):
        scores = self.mfpca_cov.transform(self.fdata, method="PACE")
        expected_scores = np.array(
            [
                [1.04732279, 1.03372031, 0.09161056, 0.00418627],
                [-0.71113053, 0.16277339, 0.02205012, -0.00797352],
                [1.40635874, -0.5550222, 0.02329331, -0.02177366],
                [-0.64269932, 0.15483285, -0.03366476, -0.01962276],
                [1.03849777, 0.65133725, -0.046797, 0.02266481],
                [0.96785148, -0.29373421, 0.00773077, 0.04528995],
                [-1.73272365, 0.0476329, 0.01749681, -0.06994339],
                [-2.24053401, 0.03109405, -0.04529198, 0.06742824],
                [-0.69518696, -0.54684134, 0.1419067, 0.0299295],
                [1.23168424, -0.57392198, -0.05362856, -0.00708507],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )


class TestInverseTransform(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)

        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_sparse_5_10_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_sparse])

        mfpca_inn = MFPCA(n_components=2, method="inner-product", normalize=True)
        mfpca_inn.fit(self.fdata)
        self.mfpca_inn = mfpca_inn
        self.scores = self.mfpca_inn.transform(method="InnPro")

    def test_inverse_scores(self):
        fdata_recons = self.mfpca_inn.inverse_transform(self.scores)

        self.assertIsInstance(fdata_recons, MultivariateFunctionalData)

        expected_values = np.array(
            [
                -0.67238479,
                -0.44451676,
                -0.17702189,
                0.07227995,
                0.30066195,
                0.4772673,
                0.57724859,
                0.58423426,
                0.5316852,
                0.40722064,
                0.1805144,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(fdata_recons.data[0].values[0]), np.abs(expected_values), decimal=0
        )

        expected_values = np.array(
            [
                -0.52510908,
                -0.51254333,
                -0.49841429,
                -0.48365781,
                -0.4682471,
                -0.45203241,
                -0.43500804,
                -0.41637608,
                -0.39731491,
                -0.37758688,
                -0.35696816,
                -0.33562819,
                -0.31342919,
                -0.29099217,
                -0.26776443,
                -0.24360851,
                -0.21916854,
                -0.19428085,
                -0.16905828,
                -0.14301849,
                -0.11687397,
                -0.09053436,
                -0.06385684,
                -0.03698469,
                -0.00996234,
                0.01720152,
                0.04393632,
                0.07093343,
                0.0979822,
                0.12523227,
                0.15181085,
                0.17819824,
                0.20441024,
                0.23062562,
                0.25646022,
                0.2817863,
                0.30677641,
                0.33118753,
                0.355185,
                0.37862081,
                0.40128739,
                0.42321767,
                0.44443121,
                0.46526068,
                0.48515926,
                0.50445403,
                0.52320631,
                0.54094677,
                0.55793064,
                0.57397069,
                0.58876292,
                0.60255499,
                0.61541325,
                0.62730478,
                0.63841735,
                0.64773253,
                0.65655707,
                0.66457206,
                0.67235268,
                0.67888252,
                0.68442952,
                0.6889805,
                0.6928896,
                0.69525845,
                0.69687132,
                0.69812223,
                0.69840507,
                0.69773967,
                0.69661509,
                0.6940268,
                0.69087025,
                0.68706606,
                0.68236173,
                0.67716613,
                0.67112966,
                0.66424152,
                0.65643714,
                0.64828453,
                0.63906061,
                0.62957591,
                0.61929707,
                0.60866654,
                0.59742072,
                0.58573655,
                0.57309723,
                0.56034083,
                0.54681557,
                0.5331631,
                0.51917014,
                0.50440975,
                0.48949702,
                0.47425372,
                0.45858717,
                0.44275872,
                0.42629262,
                0.40986786,
                0.39314604,
                0.37609039,
                0.3588859,
                0.3411941,
                0.32424847,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(fdata_recons.data[1].values[0]), np.abs(expected_values), decimal=0
        )
