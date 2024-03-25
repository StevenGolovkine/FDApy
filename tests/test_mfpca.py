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
                3.50806382e-01,
                5.64526459e-02,
                1.10401418e-02,
                7.77583361e-04,
                2.73560071e-04,
            ]
        )
        np.testing.assert_array_almost_equal(
            mfpca.eigenvalues, expected_eigenvalues, decimal=3
        )

        expected_eigenfunctions_0 = np.array(
            [
                0.1792514,
                0.28585526,
                0.43374633,
                0.58388951,
                0.72322982,
                0.83177516,
                0.89072049,
                0.91984303,
                0.88410617,
                0.76567561,
                0.55726369,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(mfpca.eigenfunctions.data[0].values[0]),
            np.abs(expected_eigenfunctions_0),
            decimal=2,
        )

        expected_eigenfunctions_1 = np.array(
            [
                0.21049475,
                0.24506742,
                0.2671824,
                0.2780416,
                0.28221162,
                0.28412025,
                0.28731851,
                0.29263519,
                0.29918781,
                0.30688353,
                0.3157637,
                0.32558927,
                0.33529007,
                0.34690359,
                0.36073004,
                0.37545497,
                0.39151317,
                0.40925313,
                0.42878593,
                0.44832184,
                0.46709625,
                0.48564531,
                0.50410778,
                0.52060926,
                0.53573933,
                0.55080814,
                0.56641085,
                0.58103257,
                0.59392776,
                0.6070569,
                0.62077076,
                0.63422032,
                0.64690444,
                0.65790658,
                0.66792239,
                0.67627116,
                0.68360897,
                0.69038133,
                0.69639843,
                0.70073495,
                0.70377377,
                0.70701186,
                0.71103439,
                0.71586666,
                0.72332678,
                0.7340163,
                0.74565997,
                0.75826682,
                0.7711172,
                0.7837418,
                0.79750991,
                0.8114653,
                0.82478214,
                0.83778575,
                0.84811707,
                0.85490017,
                0.85969388,
                0.86383083,
                0.86609776,
                0.86613884,
                0.86521406,
                0.86406265,
                0.86099267,
                0.85705021,
                0.85403637,
                0.85352294,
                0.85511162,
                0.85879413,
                0.86703763,
                0.87768859,
                0.88843738,
                0.89808533,
                0.90557731,
                0.91137292,
                0.91520356,
                0.91762434,
                0.91740924,
                0.91377426,
                0.90727267,
                0.89827989,
                0.8883028,
                0.87829374,
                0.8699551,
                0.86376157,
                0.85923262,
                0.85404633,
                0.84648721,
                0.83705371,
                0.82664029,
                0.81569539,
                0.80336006,
                0.7894739,
                0.77341311,
                0.75417058,
                0.72974863,
                0.69872966,
                0.65962858,
                0.61101157,
                0.5544647,
                0.49388184,
                0.43053637,
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

        expected_eigenvalues = np.array([9.13689059, 1.42898364, 0.52052483])
        np.testing.assert_array_almost_equal(
            mfpca.eigenvalues, expected_eigenvalues, decimal=2
        )

        expected_eigenfunctions_0 = np.array(
            [
                -0.14732401,
                -0.27406383,
                -0.40542518,
                -0.52600214,
                -0.63343339,
                -0.71755315,
                -0.76850624,
                -0.76992701,
                -0.73821502,
                -0.66391496,
                -0.53747605,
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(mfpca.eigenfunctions.data[0].values[0]),
            np.abs(expected_eigenfunctions_0),
            decimal=1,
        )

        expected_eigenfunctions_1 = np.array(
            [
                -0.19153583,
                -0.20766733,
                -0.22399033,
                -0.24021785,
                -0.25650976,
                -0.27265127,
                -0.28871007,
                -0.30507694,
                -0.32109725,
                -0.33722587,
                -0.35347133,
                -0.36970988,
                -0.38587602,
                -0.40102592,
                -0.41687004,
                -0.43322792,
                -0.4487678,
                -0.46441808,
                -0.47988306,
                -0.49490661,
                -0.51009973,
                -0.52545108,
                -0.5403252,
                -0.55508669,
                -0.56981572,
                -0.58446708,
                -0.59855782,
                -0.61264812,
                -0.62663133,
                -0.64063784,
                -0.65423823,
                -0.66749246,
                -0.68050726,
                -0.69351055,
                -0.70625406,
                -0.71867176,
                -0.73085745,
                -0.74276899,
                -0.75446358,
                -0.76587929,
                -0.7769909,
                -0.78775302,
                -0.79812484,
                -0.80849201,
                -0.81842835,
                -0.82801882,
                -0.83727196,
                -0.84602681,
                -0.85462268,
                -0.86289973,
                -0.87061008,
                -0.87814184,
                -0.88532743,
                -0.89206344,
                -0.89851075,
                -0.9042136,
                -0.90976208,
                -0.91487644,
                -0.91974347,
                -0.92425703,
                -0.9282581,
                -0.93179121,
                -0.93502439,
                -0.93747637,
                -0.93974686,
                -0.94181531,
                -0.94334698,
                -0.94433539,
                -0.94509987,
                -0.94518865,
                -0.94488938,
                -0.94420768,
                -0.9430156,
                -0.9415181,
                -0.93959164,
                -0.93707079,
                -0.93403825,
                -0.93067432,
                -0.92668668,
                -0.92239934,
                -0.91765472,
                -0.91234396,
                -0.90657399,
                -0.90026009,
                -0.89340959,
                -0.88634948,
                -0.87862782,
                -0.87043731,
                -0.86174457,
                -0.85240387,
                -0.84274382,
                -0.83254277,
                -0.82177105,
                -0.81060405,
                -0.79882525,
                -0.7865962,
                -0.77371359,
                -0.75986408,
                -0.74629872,
                -0.73229305,
                -0.72002322,
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
                [2.47342401e00, 2.44061802e00, 3.98171223e-01, 1.76930020e-01],
                [-1.71249980e00, 3.81313640e-01, 7.74390058e-02, -1.58452534e-01],
                [3.32665051e00, -1.32298453e00, -3.81886179e-02, 1.76941756e-01],
                [-1.55157398e00, 3.53686239e-01, -5.58238660e-02, -1.78379572e-01],
                [2.45466240e00, 1.52149684e00, -1.11174638e-03, 2.12248243e-01],
                [2.29092807e00, -7.05362672e-01, -3.54325117e-02, 2.59134808e-01],
                [-4.15154499e00, 1.12361873e-01, 4.82298792e-02, -4.82815834e-01],
                [-5.34331378e00, 6.67755842e-02, -1.13948305e-01, -2.48030410e-01],
                [-1.66804115e00, -1.28176277e00, 2.39920173e-01, -5.68290167e-02],
                [2.91252060e00, -1.37582012e00, -2.27113417e-01, 1.77103488e-01],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_data_notnone(self):
        scores = self.mfpca_cov.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [2.96354997, 2.27364409, 0.33074992, 0.25566839],
                [-1.22235486, 0.2144104, 0.01009783, -0.07969514],
                [3.81677391, -1.48993907, -0.10558821, 0.25567757],
                [-1.06462338, 0.20865779, -0.09866679, -0.10280958],
                [2.94463121, 1.35434414, -0.06873936, 0.29082929],
                [2.77945508, -0.86086868, -0.09000803, 0.33627792],
                [-3.66657521, -0.05126168, -0.01559466, -0.40923573],
                [-4.85375434, -0.10072599, -0.18198197, -0.16985917],
                [-1.17789011, -1.44856849, 0.1726889, 0.02193452],
                [3.40245599, -1.54316748, -0.29496112, 0.255651],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=1
        )

    def test_numint(self):
        scores = self.mfpca_inn.transform(self.fdata, method="NumInt")
        expected_scores = np.array(
            [
                [-2.88261621, 2.00971038],
                [1.20135204, 0.30268344],
                [-3.76579672, -1.76214561],
                [1.03041901, 0.27894675],
                [-2.86724943, 1.06644491],
                [-2.753039, -1.0648562],
                [3.60802175, 0.24660656],
                [4.73937143, 0.29756411],
                [1.1231306, -1.25945258],
                [-3.34702489, -1.79452666],
            ]
        )
        np.testing.assert_array_almost_equal(
            np.abs(scores), np.abs(expected_scores), decimal=0
        )

    def test_innpro(self):
        scores = self.mfpca_inn.transform(method="InnPro")
        expected_scores = np.array(
            [
                [-2.57587537, 2.43106607],
                [1.62549417, 0.36763129],
                [-3.43076581, -1.33562776],
                [1.45640966, 0.32870131],
                [-2.54180981, 1.5129186],
                [-2.39740383, -0.74700828],
                [4.11015145, 0.10354331],
                [5.25135535, 0.01114447],
                [1.57312796, -1.25945359],
                [-3.00784073, -1.38153627],
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
