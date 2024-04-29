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
from FDApy.preprocessing.dim_reduction.mfpca import MFPCA

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
                1.88149247e00,
                3.00014097e-01,
                6.34325079e-02,
                4.13197838e-03,
                1.47151277e-03,
            ]
        )
        np.testing.assert_array_almost_equal(
            mfpca.eigenvalues, expected_eigenvalues, decimal=3
        )

        expected_eigenfunctions_0 = np.array(
            [
                0.1730797,
                0.27548914,
                0.41803811,
                0.56278465,
                0.69699408,
                0.80116146,
                0.85758515,
                0.88549964,
                0.85117092,
                0.73731704,
                0.53682735,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(mfpca.eigenfunctions.data[0].values[0]),
            np.abs(expected_eigenfunctions_0),
            decimal=2,
        )

        expected_eigenfunctions_1 = np.array(
            [
                0.21874433,
                0.25477417,
                0.27788683,
                0.28923583,
                0.29351343,
                0.29533249,
                0.29840915,
                0.30367821,
                0.3102571,
                0.31805845,
                0.3271274,
                0.33719892,
                0.34717263,
                0.35916707,
                0.37347244,
                0.38872425,
                0.40538471,
                0.42381569,
                0.44413051,
                0.46447453,
                0.48405655,
                0.50342211,
                0.52270043,
                0.53994321,
                0.55574148,
                0.57143955,
                0.58764127,
                0.60278631,
                0.61610706,
                0.62964196,
                0.64377543,
                0.65765015,
                0.67074906,
                0.68211439,
                0.69246242,
                0.70106613,
                0.70860507,
                0.71552886,
                0.72166011,
                0.72606108,
                0.7291263,
                0.73241059,
                0.73650105,
                0.74143332,
                0.74910308,
                0.76011897,
                0.7721226,
                0.78512938,
                0.79839321,
                0.81142791,
                0.8256429,
                0.84005102,
                0.85380679,
                0.86724174,
                0.8779224,
                0.88494547,
                0.88991762,
                0.89422178,
                0.89660059,
                0.89667821,
                0.89575785,
                0.89460219,
                0.89144899,
                0.88737498,
                0.88424868,
                0.88370177,
                0.88533459,
                0.88913118,
                0.8976657,
                0.90871219,
                0.91986131,
                0.92986862,
                0.93763819,
                0.94364562,
                0.94760238,
                0.95007825,
                0.94980417,
                0.94598106,
                0.93918066,
                0.9298044,
                0.91941684,
                0.90900551,
                0.90032277,
                0.8938508,
                0.88910499,
                0.88369689,
                0.87585916,
                0.86610695,
                0.85536471,
                0.84410894,
                0.83145192,
                0.81723544,
                0.80081171,
                0.78113978,
                0.75612892,
                0.72429378,
                0.68409183,
                0.63405675,
                0.57585938,
                0.51358872,
                0.44855077,
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

        expected_eigenvalues = np.array([1.69276847, 0.26647388, 0.09580169])
        np.testing.assert_array_almost_equal(
            mfpca.eigenvalues, expected_eigenvalues, decimal=2
        )

        expected_eigenfunctions_0 = np.array(
            [
                -0.15706472,
                -0.29308894,
                -0.41786522,
                -0.52468317,
                -0.60788717,
                -0.66596213,
                -0.71618648,
                -0.74631852,
                -0.74677879,
                -0.71152992,
                -0.63785911,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(mfpca.eigenfunctions.data[0].values[0]),
            np.abs(expected_eigenfunctions_0),
            decimal=1,
        )

        expected_eigenfunctions_1 = np.array(
            [
                -0.19005596,
                -0.20551661,
                -0.22089927,
                -0.23619744,
                -0.25141136,
                -0.26654067,
                -0.28157263,
                -0.29650598,
                -0.31133672,
                -0.32606159,
                -0.3406777,
                -0.35517715,
                -0.36955633,
                -0.38381001,
                -0.39793424,
                -0.41192521,
                -0.42577938,
                -0.43949269,
                -0.45306002,
                -0.46647717,
                -0.47973928,
                -0.49284192,
                -0.50578205,
                -0.51855592,
                -0.53115989,
                -0.54359039,
                -0.55584262,
                -0.56791392,
                -0.57980166,
                -0.59150231,
                -0.60301491,
                -0.61433912,
                -0.62547113,
                -0.6364133,
                -0.64716498,
                -0.65772729,
                -0.66810505,
                -0.67830194,
                -0.68832568,
                -0.69818798,
                -0.70786946,
                -0.7172875,
                -0.72643702,
                -0.73530765,
                -0.7438877,
                -0.75217494,
                -0.76016845,
                -0.76785862,
                -0.77523529,
                -0.78229253,
                -0.78902858,
                -0.79543397,
                -0.80150947,
                -0.80724432,
                -0.81263411,
                -0.81767843,
                -0.82237203,
                -0.82671404,
                -0.83070041,
                -0.83432016,
                -0.83756797,
                -0.84043571,
                -0.84309677,
                -0.84559528,
                -0.8479034,
                -0.84999961,
                -0.85186824,
                -0.85349116,
                -0.85485333,
                -0.85595502,
                -0.85678555,
                -0.85733883,
                -0.85760732,
                -0.85758653,
                -0.85727165,
                -0.85666048,
                -0.85574761,
                -0.85452841,
                -0.85300199,
                -0.85116514,
                -0.84901527,
                -0.84654959,
                -0.84376639,
                -0.84066347,
                -0.83723899,
                -0.83349176,
                -0.82941948,
                -0.82502089,
                -0.82029429,
                -0.81523953,
                -0.80985685,
                -0.80414343,
                -0.79809962,
                -0.79172145,
                -0.78500652,
                -0.77795128,
                -0.77055806,
                -0.7628328,
                -0.75476501,
                -0.74636162,
                -0.73763138,
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
