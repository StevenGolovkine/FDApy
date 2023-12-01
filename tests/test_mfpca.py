#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class MFPCA in the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
import warnings

from FDApy.representation.functional_data import MultivariateFunctionalData
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import (
    MFPCA
)


class MFPCATest(unittest.TestCase):
    def test_init(self):
        # Test default initialization
        fpca = MFPCA()
        self.assertEqual(fpca.method, 'covariance')
        self.assertIsNone(fpca.n_components)
        self.assertFalse(fpca.normalize)
        self.assertEqual(fpca.weights, None)

        # Test custom initialization
        fpca = MFPCA(method='inner-product', n_components=3, normalize=True)
        self.assertEqual(fpca.method, 'inner-product')
        self.assertEqual(fpca.n_components, 3)
        self.assertTrue(fpca.normalize)

    def test_method(self):
        ufpc = MFPCA()
        ufpc.method = 'inner-product'
        self.assertEqual(ufpc.method, 'inner-product')

    def test_n_components(self):
        ufpc = MFPCA()
        ufpc.n_components = [4, 3]
        self.assertEqual(ufpc.n_components, [4, 3])

    def test_normalize(self):
        ufpc = MFPCA()
        ufpc.normalize = True
        self.assertTrue(ufpc.normalize)


class TestFit(unittest.TestCase):
    def setUp(self):
        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)

        fdata_uni = kl.data
        fdata_sparse = kl.sparse_data
        self.fdata = MultivariateFunctionalData([fdata_uni, fdata_sparse])

    def test_fit_error_method(self):
        mfpca = MFPCA(method='error', n_components=[0.95, 0.99], normalize=True)
        with self.assertRaises(NotImplementedError):
            mfpca.fit(self.fdata)

    def test_fit_covariance(self):
        mfpca = MFPCA(method='covariance', n_components=[0.95, 0.99], normalize=True)
        mfpca.fit(data=self.fdata)

        expected_eigenvalues = np.array([2.01623143e-01, 1.76702288e-01, 3.46774709e-02, 2.50964812e-03, 1.66237881e-04, 2.22946578e-05, 1.55005859e-05, 8.08648708e-07])
        np.testing.assert_array_almost_equal(mfpca.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions_0 = np.array([-0.11384354, -0.15540465, -0.19555623, -0.23430321, -0.27167061, -0.30770165, -0.3423404 , -0.3756506 , -0.40760077, -0.43822493, -0.46743402, -0.49518976, -0.52141292, -0.54599046, -0.56876553, -0.58959108, -0.6085579 , -0.62600365, -0.64199156, -0.65658409, -0.66984306, -0.68182966, -0.69260415, -0.70222544, -0.71075282, -0.71824723, -0.72476987, -0.73038183, -0.73514406, -0.73911738, -0.74236253, -0.74494019, -0.74691098, -0.74833554, -0.74927452, -0.74978861, -0.74993858, -0.74978571, -0.74939235, -0.74882228, -0.74814076, -0.74741411, -0.74670891, -0.74609091, -0.74562367, -0.74536715, -0.74537616, -0.74569896, -0.74637594, -0.74743856, -0.7489085 , -0.75079713, -0.75310534, -0.75582371, -0.75893298, -0.76240487, -0.76620318, -0.77028507, -0.77460251, -0.77910381, -0.78373507, -0.78844149, -0.79316856, -0.7978628 , -0.80247222, -0.80694627, -0.81123551, -0.81529101, -0.81906406, -0.82250602, -0.82556835, -0.82820254, -0.83036007, -0.8319923 , -0.83305042, -0.83348532, -0.83324751, -0.83228706, -0.83055447, -0.82800528, -0.82459692, -0.82028222, -0.81501098, -0.80873061, -0.80138678, -0.79292323, -0.78305323, -0.77167855, -0.75892191, -0.74493238, -0.72983365, -0.71372496, -0.69666892, -0.67878211, -0.66005974, -0.64056211, -0.6202515 , -0.59921607, -0.57743616, -0.55491237, -0.53166489])
        np.testing.assert_array_almost_equal(np.abs(mfpca.eigenfunctions.data[0].values[0]), np.abs(expected_eigenfunctions_0))

        expected_eigenfunctions_1 = np.array([-0.58362154, -0.63056133, -0.662449  , -0.67516354, -0.67289354, -0.66612502, -0.66456253, -0.66588295, -0.66505954, -0.66130437, -0.654613  , -0.64663106, -0.63416285, -0.61762043, -0.60505368, -0.59747743, -0.59112319, -0.5858056 , -0.58823402, -0.58775903, -0.58498215, -0.58286996, -0.58531769, -0.59132295, -0.59711273, -0.60294711, -0.61292456, -0.61989885, -0.62120529, -0.61820264, -0.61700454, -0.6179755 , -0.61514701, -0.60885573, -0.59709343, -0.58339917, -0.57421088, -0.57059264, -0.57289732, -0.57605699, -0.58059892, -0.58585304, -0.59233705, -0.59804271, -0.60318338, -0.61317578, -0.62670841, -0.6389082 , -0.64635662, -0.65184288, -0.65789529, -0.66236162, -0.66493084, -0.66553414, -0.66218477, -0.65727404, -0.64695659, -0.63336283, -0.61893406, -0.60871737, -0.60097208, -0.59778286, -0.60209991, -0.61281844, -0.63131958, -0.64990508, -0.66374109, -0.67539166, -0.68339117, -0.68937627, -0.69155601, -0.69371608, -0.69775107, -0.70353297, -0.71019668, -0.7170622 , -0.72409176, -0.73283683, -0.74398859, -0.75385615, -0.75960814, -0.76284773, -0.76519231, -0.76658102, -0.76938983, -0.77612967, -0.78820928, -0.80508717, -0.82306603, -0.84332841, -0.86581264, -0.89068377, -0.91618098, -0.94394251, -0.9732098 , -1.01007963, -1.06093297, -1.11822927, -1.17946735, -1.23240015, -1.26027117])
        np.testing.assert_array_almost_equal(np.abs(mfpca.eigenfunctions.data[1].values[0]), np.abs(expected_eigenfunctions_1))

    def test_fit_inner_product(self):
        mfpca = MFPCA(method='inner-product', n_components=0.99, normalize=True)
        mfpca.fit(data=self.fdata)

        expected_eigenvalues = np.array([5.81661382, 4.66140815, 1.17135333, 0.2752354 ])
        np.testing.assert_array_almost_equal(mfpca.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions_0 = np.array([ 0.95291165,  0.89564791,  0.83977688,  0.785273  ,  0.73211058,  0.68026376,  0.62970649,  0.5804126 ,  0.53235569,  0.48550921,  0.4398464 ,  0.39534028,  0.35196366,  0.30968905,  0.26848872,  0.22833479,  0.18919943,  0.15105502,  0.11387434,  0.0776308 ,  0.04229859,  0.00785277, -0.02573053, -0.05847412, -0.09039964, -0.12152758, -0.15187735, -0.18146728, -0.21031486, -0.23843683, -0.2658495 , -0.29256912, -0.31861229, -0.34399658, -0.36874138, -0.39286889, -0.41640557, -0.43938407, -0.46184583, -0.48384485, -0.50523642, -0.52545961, -0.5445235 , -0.56243469, -0.57919723, -0.59481266, -0.60927994, -0.62259543, -0.6347529 , -0.64574352, -0.65555583, -0.66417573, -0.67158652, -0.67776885, -0.68270076, -0.68635768, -0.68871244, -0.68973527, -0.68939386, -0.68765334, -0.6844763 , -0.67982286, -0.6744941 , -0.6686899 , -0.66231801, -0.655303  , -0.64758158, -0.63909914, -0.62980731, -0.61966218, -0.60862287, -0.59665054, -0.58370764, -0.56975722, -0.55476256, -0.53868675, -0.52149248, -0.50314179, -0.48359601, -0.46281562, -0.44076026, -0.41738872, -0.39265902, -0.36652847, -0.33895381, -0.30989138, -0.27929726, -0.24712745, -0.21333811, -0.17788565, -0.14072689, -0.10181905, -0.06111973, -0.01858683,  0.02582143,  0.0721465 ,  0.12042959,  0.1707116 ,  0.22303316,  0.27743468,  0.33395629])
        np.testing.assert_array_almost_equal(np.abs(mfpca.eigenfunctions.data[0].values[0]), np.abs(expected_eigenfunctions_0))

        expected_eigenfunctions_1 = np.array([ 1.19402528,  1.12581035,  1.05862136,  0.99331525,  0.92962648,  0.86681029,  0.80540216,  0.7456192 ,  0.68739479,  0.63019594,  0.57374473,  0.51817946,  0.4635616 ,  0.41002789,  0.35770495,  0.30669376,  0.25700357,  0.20865093,  0.16154101,  0.11560034,  0.07085556,  0.02733228, -0.01505494, -0.05638883, -0.09665062, -0.13587261, -0.17402688, -0.21118244, -0.24737413, -0.28252503, -0.3167998 , -0.35042044, -0.38322848, -0.41505126, -0.44594561, -0.47591227, -0.50500601, -0.53332601, -0.56086632, -0.58767891, -0.61384599, -0.6395004 , -0.66440777, -0.68768028, -0.70910254, -0.72874301, -0.74678156, -0.76326941, -0.77776352, -0.79042139, -0.80151735, -0.81076387, -0.81808212, -0.82366341, -0.82780941, -0.83073806, -0.83278878, -0.8342926 , -0.83463565, -0.83337876, -0.83110255, -0.82803255, -0.82400623, -0.81914145, -0.81355364, -0.80704227, -0.79957444, -0.79109815, -0.78145753, -0.77087256, -0.759584  , -0.74738282, -0.73393982, -0.71917072, -0.70303744, -0.68548873, -0.66646528, -0.64591172, -0.6238615 , -0.60024565, -0.57500067, -0.54813913, -0.51964564, -0.48943034, -0.45748802, -0.42391772, -0.38853214, -0.35116314, -0.31209243, -0.27126803, -0.22853284, -0.18330967, -0.13535887, -0.08495511, -0.03188824,  0.02336574,  0.08060379,  0.13951809,  0.20078964,  0.26471271,  0.33080149])
        np.testing.assert_array_almost_equal(np.abs(mfpca.eigenfunctions.data[1].values[0]), np.abs(expected_eigenfunctions_1))


class TestTransform(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)

        fdata_uni = kl.data
        fdata_sparse = kl.sparse_data
        self.fdata = MultivariateFunctionalData([fdata_uni, fdata_sparse])

        mfpca_cov = MFPCA(n_components=[2, 2], method='covariance', normalize=True)
        mfpca_cov.fit(self.fdata)
        self.mfpca_cov = mfpca_cov
        
        mfpca_inn = MFPCA(n_components=2, method='inner-product', normalize=True)
        mfpca_inn.fit(self.fdata)
        self.mfpca_inn = mfpca_inn

    def test_error_innpro(self):
        with self.assertRaises(ValueError):
            self.mfpca_cov.transform(self.fdata, method='InnPro')

        with self.assertRaises(ValueError):
            self.mfpca_cov.transform(None, method='InnPro')

    def test_error_unkown_method(self):
        with self.assertRaises(ValueError):
            self.mfpca_cov.transform(self.fdata, method='error')
    
    def test_data_none(self):
        scores = self.mfpca_cov.transform(None, method='NumInt')
        expected_scores = np.array([[ 0.95753043,  2.67570818,  0.28459838, -0.02256917],[ 2.79515671,  2.53812271,  0.42915128,  0.0364238 ],[-4.20655814, -1.1109952 , -0.40367335, -0.29026642],[ 0.57921642,  1.11346464, -0.06183846, -0.02530755],[ 0.37330027,  1.95009793,  0.21190821, -0.14963979],[-2.06518133,  0.63958928, -0.1271601 , -0.26717931],[ 1.89462225, -3.61084519, -0.18833857,  0.42579222],[ 2.62881597, -2.55320304, -0.07945005,  0.36701679],[-0.14982394, -1.41598382, -0.15245475,  0.10330174],[-2.8983734 , -0.07105038, -0.19464333, -0.22883083]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))

    def test_data_notnone(self):
        scores = self.mfpca_cov.transform(self.fdata, method='NumInt')
        expected_scores = np.array([[ 0.33910714,  2.32144149,  0.21584025, -0.00775407],[ 2.17670773,  2.18386248,  0.3603851 ,  0.05121032],[-4.82500604, -1.46530483, -0.47248196, -0.27543601],[-0.02302468,  0.80627075, -0.08100833, -0.03691257],[-0.24512652,  1.59595955,  0.14325991, -0.13493928],[-2.67543872,  0.31006755, -0.17003225, -0.26655722],[ 1.273019  , -3.96845772, -0.26170151,  0.44067017],[ 2.01064377, -2.90704725, -0.14770679,  0.38168942],[-0.76822501, -1.77001555, -0.22099653,  0.11793261],[-3.51677755, -0.42521749, -0.26330459, -0.21408512]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))

    def test_numint(self):
        scores = self.mfpca_inn.transform(self.fdata, method='NumInt')
        expected_scores = np.array([[-1.21640419, -1.30420282],[-0.45964864, -3.19354221],[-1.0860186 ,  5.26480494],[-0.56416797, -0.52730895],[-1.14328856, -0.57885154],[-1.38384028,  2.22884568],[ 3.18237149,  1.09983748],[ 2.7567867 , -0.27065239],[ 0.8882041 ,  1.72581606],[-1.17147325,  3.52179092]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))
    
    def test_innpro(self):
        scores = self.mfpca_inn.transform(method='InnPro')
        expected_scores = np.array([[-1.67302776, -2.5203345 ],[-0.08610058, -3.72131113],[-2.19617915,  3.69635949],[-0.54965729, -1.39795227],[-1.35116198, -1.61606303],[-2.06348873,  0.93224925],[ 4.54742773,  1.29749921],[ 4.21298438, -0.02568143],[ 1.08996581,  1.11768554],[-2.12941088,  2.09340916]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))
