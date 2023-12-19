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
        fpca = MFPCA(n_components=0.99)
        self.assertEqual(fpca.method, 'covariance')
        self.assertEqual(fpca.n_components, 0.99)
        self.assertFalse(fpca.normalize)
        self.assertEqual(fpca.weights, None)

        # Test custom initialization
        fpca = MFPCA(method='inner-product', n_components=3, normalize=True)
        self.assertEqual(fpca.method, 'inner-product')
        self.assertEqual(fpca.n_components, 3)
        self.assertTrue(fpca.normalize)

    def test_method(self):
        ufpc = MFPCA(n_components=0.99)
        ufpc.method = 'inner-product'
        self.assertEqual(ufpc.method, 'inner-product')

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
        mfpca = MFPCA(method='covariance', n_components=[0.95, 3], normalize=True)
        mfpca.fit(data=self.fdata)

        expected_eigenvalues = np.array([2.25664124e-01, 1.81435936e-01, 4.36168322e-02, 2.08246603e-04, 4.93736729e-05, 7.89461415e-07])
        np.testing.assert_array_almost_equal(mfpca.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions_0 = np.array([ 1.21559194,  1.12371158,  1.0346471 ,  0.94832999,  0.86468463,  0.78363532,  0.70514152,  0.62912163,  0.55553357,  0.48430829,  0.41543017,  0.34886607,  0.28460258,  0.22264996,  0.1630634 ,  0.10592974,  0.05122761, -0.00128053, -0.05164335, -0.09991251, -0.14614233, -0.19039058, -0.23271924, -0.27319624, -0.31188665, -0.34884966, -0.38414418, -0.4178297 , -0.44996612, -0.48061353, -0.50983214, -0.53768221, -0.56422398, -0.58951768, -0.61362352, -0.63660167, -0.65851238, -0.67941658, -0.69937664, -0.71845686, -0.73672332, -0.75424324, -0.77108382, -0.78731067, -0.80298599, -0.81816662, -0.83290208, -0.84723262, -0.86118762, -0.87478416, -0.88802591, -0.90090247, -0.91338912, -0.92544705, -0.93702398, -0.94805516, -0.95846483, -0.96816789, -0.97707176, -0.98507846, -0.99208651, -0.99799281, -1.00269423, -1.00608873, -1.00807614, -1.00855816, -1.00743803, -1.00461971, -1.00000738, -0.99350531, -0.98501784, -0.9744494 , -0.96170441, -0.94668732, -0.92930261, -0.9094548 , -0.8870486 , -0.86198907, -0.83418119, -0.80352639, -0.76992392, -0.73327698, -0.69349146, -0.6504755 , -0.60413874, -0.55439232, -0.50080913, -0.44331307, -0.38207194, -0.31723272, -0.24887539, -0.17703934, -0.10172141, -0.02296905,  0.05929773,  0.14509316,  0.23452996,  0.3276086 ,  0.42442668,  0.52507238,  0.62961817])
        np.testing.assert_array_almost_equal(np.abs(mfpca.eigenfunctions.data[0].values[0]), np.abs(expected_eigenfunctions_0))

        expected_eigenfunctions_1 = np.array([ 1.36046051,  1.19883029,  1.06503362,  0.94957068,  0.8442768 ,  0.74491317,  0.65025673,  0.56107259,  0.47902569,  0.40391226,  0.33589168,  0.27389962,  0.21889788,  0.16821048,  0.120448  ,  0.07631407,  0.03498687, -0.00717834, -0.05069379, -0.09283572, -0.1319451 , -0.16809856, -0.20548496, -0.24144478, -0.27503372, -0.31083614, -0.35088115, -0.39184437, -0.43101178, -0.46935422, -0.50613708, -0.53814711, -0.5659214 , -0.58996409, -0.61079615, -0.62838941, -0.64348819, -0.65648463, -0.66716328, -0.67522858, -0.6821016 , -0.69003762, -0.69988574, -0.71179032, -0.72767602, -0.7481839 , -0.77088216, -0.79547176, -0.82093626, -0.84634357, -0.8729173 , -0.89958789, -0.92504903, -0.94932002, -0.96965626, -0.98487255, -0.99728342, -1.00819098, -1.01578402, -1.01866933, -1.01875349, -1.01764251, -1.01276532, -1.00545795, -0.99904988, -0.99541811, -0.99391848, -0.9937609 , -0.99679374, -1.00038347, -1.00162939, -0.99903662, -0.99270291, -0.98301286, -0.97010415, -0.9555259 , -0.93708224, -0.91297711, -0.88410622, -0.84970789, -0.80854893, -0.76103937, -0.70761065, -0.64721527, -0.58325013, -0.51715763, -0.44863746, -0.37912588, -0.30973091, -0.24095477, -0.17267237, -0.10509659, -0.03943168,  0.02360579,  0.08394963,  0.14061536,  0.19397611,  0.24562648,  0.29616055,  0.3469799 ,  0.40758071])
        np.testing.assert_array_almost_equal(np.abs(mfpca.eigenfunctions.data[1].values[0]), np.abs(expected_eigenfunctions_1))

        self.assertIsNone(mfpca.covariance)

    def test_fit_inner_product(self):
        mfpca = MFPCA(method='inner-product', n_components=0.99, normalize=True)
        mfpca.fit(data=self.fdata)

        expected_eigenvalues = np.array([5.81661382, 4.66140815, 1.17135333, 0.2752354 ])
        np.testing.assert_array_almost_equal(mfpca.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions_0 = np.array([ 1.13353822,  1.05026785,  0.96903127,  0.88979834,  0.81224653,  0.73734331,  0.66503451,  0.59526597,  0.52798352,  0.463133  ,  0.40066023,  0.34051104,  0.28263128,  0.22696677,  0.17346335,  0.12206685,  0.07272309,  0.0253779 , -0.0200229 , -0.06353347, -0.10520803, -0.14510074, -0.18326581, -0.21975745, -0.25462987, -0.28793729, -0.31973395, -0.35007408, -0.37901193, -0.40660177, -0.43289788, -0.45795455, -0.48182608, -0.5045668 , -0.52623104, -0.54687318, -0.5665476 , -0.58530868, -0.60321063, -0.62030756, -0.63665347, -0.65230233, -0.66730816, -0.68172502, -0.69560711, -0.70900879, -0.72198442, -0.73458837, -0.74687312, -0.75888024, -0.77063199, -0.78212616, -0.79333439, -0.80420389, -0.81466253, -0.82462728, -0.83401321, -0.8427354 , -0.85070896, -0.85784879, -0.8640696 , -0.86928594, -0.87341232, -0.87636323, -0.87805325, -0.87839713, -0.87730983, -0.87470638, -0.8705018 , -0.86461112, -0.85694934, -0.84743149, -0.83597258, -0.82248765, -0.80689171, -0.78909979, -0.76902693, -0.74658816, -0.72169852, -0.69427305, -0.6642268 , -0.63147483, -0.59593218, -0.55751391, -0.51613507, -0.47171073, -0.42415595, -0.37338579, -0.3193153 , -0.26185954, -0.20093357, -0.13645245, -0.06833122,  0.00351506,  0.07917133,  0.15872256,  0.2422537 ,  0.32984969,  0.4200259 ,  0.51327849,  0.60965476])
        np.testing.assert_array_almost_equal(np.abs(mfpca.eigenfunctions.data[0].values[0]), np.abs(expected_eigenfunctions_0))

        expected_eigenfunctions_1 = np.array([ 1.41374022,  1.31031092,  1.20878993,  1.11046257,  1.01314019,  0.92031465,  0.83078653,  0.74288504,  0.6593535 ,  0.57808368,  0.49876497,  0.42126892,  0.34620625,  0.28659309,  0.21972702,  0.14769207,  0.08761613,  0.02926417, -0.02667013, -0.07925123, -0.13135867, -0.18215481, -0.22895885, -0.27387608, -0.31804907, -0.36051338, -0.39894872, -0.43636696, -0.47202533, -0.50708654, -0.5411677 , -0.57201223, -0.60101454, -0.62958916, -0.65664811, -0.68238197, -0.7068804 , -0.73045339, -0.75294822, -0.77418951, -0.79432491, -0.81393339, -0.83245174, -0.85067969, -0.86814008, -0.88502439, -0.90140545, -0.91651471, -0.93184005, -0.94697608, -0.96156793, -0.97586099, -0.99000176, -1.00378323, -1.0170384 , -1.0288877 , -1.04063777, -1.05123936, -1.06146986, -1.07063983, -1.07844595, -1.08509264, -1.09045172, -1.09352374, -1.09533573, -1.09630214, -1.09498589, -1.0915244 , -1.08693108, -1.07946707, -1.06928339, -1.05732553, -1.04268601, -1.02654042, -1.00817762, -0.98543263, -0.95990463, -0.93215768, -0.90067921, -0.86665924, -0.82960587, -0.7884076 , -0.7441522 , -0.69551612, -0.64336995, -0.58917127, -0.53021142, -0.466455  , -0.39946788, -0.32694335, -0.25135389, -0.17096679, -0.08610915,  0.00323891,  0.09677678,  0.19466857,  0.29991449,  0.41374778,  0.52525211,  0.64065792,  0.7198611 ])
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

    def test_error_pace(self):
        with self.assertRaises(ValueError):
            self.mfpca_cov.transform(self.fdata, method='PACE')
    
    def test_data_none(self):
        scores = self.mfpca_cov.transform(None, method='NumInt')
        expected_scores = np.array([[-1.53552641e+00, -2.72380203e+00,  4.41053298e-01,  6.16888163e-02],[-2.80254083e-03, -3.65442509e+00,  3.73541592e-01, -1.31036695e-02],[-2.29107486e+00,  3.57649777e+00, -3.41509771e-01,  3.19624595e-01],[-4.09201776e-01, -1.16263743e+00, -1.03957758e-01, -1.85953370e-02],[-1.31354769e+00, -1.67787168e+00,  2.28000350e-01,  1.29112020e-01],[-2.06976263e+00,  9.51201217e-01, -1.52761341e-01,  2.58541056e-01],[ 4.45750159e+00,  1.31528572e+00, -1.81176929e-01, -4.84145064e-01],[ 4.15061196e+00,  1.32705567e-01, -1.48918161e-01, -4.25258363e-01],[ 1.03903754e+00,  1.12201466e+00, -1.48570176e-01, -8.57121480e-02],[-2.18044794e+00,  1.97826557e+00, -1.61852177e-01,  2.81073329e-01]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))

    def test_data_notnone(self):
        scores = self.mfpca_cov.transform(self.fdata, method='NumInt')
        expected_scores = np.array([[-1.72252071, -2.07243311,  0.39525212,  0.06035322],[-0.1897398 , -3.00305669,  0.32773102, -0.01449607],[-2.47796185,  4.22789691, -0.38735958,  0.31818729],[-0.63848039, -0.55214658, -0.10185786,  0.0153669 ],[-1.50057435, -1.02660598,  0.1823072 ,  0.12779156],[-2.27954361,  1.58130651, -0.17350817,  0.2763534 ],[ 4.27242394,  1.97182066, -0.23243613, -0.48653411],[ 3.96399575,  0.78326316, -0.19398016, -0.42710372],[ 0.85172244,  1.77314204, -0.19407598, -0.08676864],[-2.36759065,  2.62935934, -0.2073546 ,  0.27983989]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))

    def test_numint(self):
        scores = self.mfpca_inn.transform(self.fdata, method='NumInt')
        expected_scores = np.array([[-1.88301632, -1.86633235],[-0.28926982, -3.03643315],[-2.35966314,  4.38902692],[-0.6289921 , -0.57724058],[-1.56003964, -0.93948264],[-2.21859155,  1.65357491],[ 4.31802918,  1.89536276],[ 4.01260678,  0.63381267],[ 0.89760033,  1.78828706],[-2.30765807,  2.77753358]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))
    
    def test_innpro(self):
        scores = self.mfpca_inn.transform(method='InnPro')
        expected_scores = np.array([[-1.67302776, -2.5203345 ],[-0.08610058, -3.72131113],[-2.19617915,  3.69635949],[-0.54965729, -1.39795227],[-1.35116198, -1.61606303],[-2.06348873,  0.93224925],[ 4.54742773,  1.29749921],[ 4.21298438, -0.02568143],[ 1.08996581,  1.11768554],[-2.12941088,  2.09340916]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))


class TestInverseTransform(unittest.TestCase):
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

        mfpca_inn = MFPCA(n_components=2, method='inner-product', normalize=True)
        mfpca_inn.fit(self.fdata)
        self.mfpca_inn = mfpca_inn
        self.scores = self.mfpca_inn.transform(method='InnPro')

    def test_inverse_scores(self):
        fdata_recons = self.mfpca_inn.inverse_transform(self.scores)
        
        self.assertIsInstance(fdata_recons, MultivariateFunctionalData)

        expected_values = np.array([-0.83105578, -0.79440929, -0.75829631, -0.72271025, -0.68754173, -0.65313341, -0.61947438, -0.58655375, -0.55436062, -0.52288409, -0.49211328, -0.46203729, -0.43264524, -0.40392626, -0.37586948, -0.34846407, -0.32169932, -0.29556464, -0.27004964, -0.24514419, -0.22083848, -0.19712304, -0.17398881, -0.15142718, -0.12943003, -0.10798978, -0.0870994 , -0.0667525 , -0.04694333, -0.02766685, -0.00891879,  0.00930433,  0.02700503,  0.04418489,  0.06084429,  0.07698236,  0.09259666,  0.10768291,  0.12223437,  0.13624133,  0.14972946,  0.16280097,  0.17545684,  0.18769754,  0.19952312,  0.21093319,  0.22192677,  0.23250235,  0.2426571 ,  0.25238347,  0.26166599,  0.2704793 ,  0.27878751,  0.28654483,  0.29369757,  0.30018727,  0.30595423,  0.31093814,  0.31507815,  0.3183127 ,  0.32057947,  0.32181549,  0.3221044 ,  0.32143719,  0.31975484,  0.31699954,  0.31311437,  0.30804292,  0.30172905,  0.29411673,  0.28514996,  0.2747727 ,  0.26292877,  0.24956185,  0.23461541,  0.21803272,  0.19975681,  0.17973046,  0.15789621,  0.13419636,  0.108573  ,  0.08096797,  0.05132296,  0.01957949, -0.01432107, -0.05043739, -0.08882823, -0.12955232, -0.17266835, -0.21823493, -0.26631056, -0.31695362, -0.3702224 , -0.42617508, -0.48486975, -0.54636441, -0.61071696, -0.67798522, -0.74728135, -0.81891579, -0.89292707])
        np.testing.assert_array_almost_equal(np.abs(fdata_recons.data[0].values[0]), np.abs(expected_values))

        expected_values = np.array([-0.79789383, -0.76559351, -0.7323349 , -0.69953111, -0.66665591, -0.63558463, -0.60450798, -0.57278669, -0.54212179, -0.51184651, -0.48213836, -0.45228213, -0.42264664, -0.39912331, -0.37208933, -0.34195348, -0.31617043, -0.29121154, -0.26681665, -0.24168725, -0.217273  , -0.19409853, -0.17123311, -0.1489922 , -0.12663493, -0.10485696, -0.08458792, -0.06423161, -0.04455056, -0.02503361, -0.00584847,  0.01198085,  0.02942697,  0.04677419,  0.0634414 ,  0.07962748,  0.09518661,  0.11022864,  0.12484025,  0.1387643 ,  0.15202548,  0.16488034,  0.17721603,  0.18945463,  0.20125762,  0.21280683,  0.22400979,  0.23452771,  0.24467872,  0.25445705,  0.26357094,  0.27209896,  0.28027117,  0.28799164,  0.29521615,  0.301109  ,  0.30652978,  0.31084344,  0.31484986,  0.31757198,  0.31934857,  0.32037682,  0.32061305,  0.31914854,  0.31680198,  0.3139521 ,  0.30954748,  0.30378393,  0.2973323 ,  0.28903122,  0.27928236,  0.26836638,  0.25563784,  0.24204426,  0.22704286,  0.20949961,  0.19009389,  0.16934701,  0.14629914,  0.12176443,  0.09518504,  0.06661558,  0.03614325,  0.00347034, -0.03178363, -0.06845681, -0.10787591, -0.14967759, -0.1932495 , -0.2400992 , -0.28881089, -0.3402141 , -0.39442055, -0.45102845, -0.5103814 , -0.57234658, -0.63825716, -0.70734251, -0.77715014, -0.84979869, -0.90751596])
        np.testing.assert_array_almost_equal(np.abs(fdata_recons.data[1].values[0]), np.abs(expected_values))
