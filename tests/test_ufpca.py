#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class UFPCA in the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
import warnings

from FDApy.representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData
)
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import UFPCA


class UFPCATest(unittest.TestCase):
    def test_init(self):
        # Test default initialization
        fpca = UFPCA()
        self.assertEqual(fpca.method, 'covariance')
        self.assertIsNone(fpca.n_components)
        self.assertFalse(fpca.normalize)
        self.assertEqual(fpca.weights, 1)

        # Test custom initialization
        fpca = UFPCA(method='inner-product', n_components=3, normalize=True)
        self.assertEqual(fpca.method, 'inner-product')
        self.assertEqual(fpca.n_components, 3)
        self.assertTrue(fpca.normalize)

    def test_method(self):
        ufpc = UFPCA()
        ufpc.method = 'inner-product'
        self.assertEqual(ufpc.method, 'inner-product')

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
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)
        kl.add_noise(0.05)

        self.fdata_uni = kl.noisy_data
        self.fdata_sparse = kl.sparse_data

    def test_fit_covariance(self):
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.fdata_uni)

        np.testing.assert_almost_equal(self.fdata_uni.argvals['input_dim_0'], uf.mean.argvals['input_dim_0'])

        expected_mean = np.array([[ 0.29045424,  0.27942843,  0.26896556,  0.25900549,   0.24912338,  0.23943513,  0.23017978,  0.22129709,   0.21272427,  0.20442406,  0.19625977,  0.18827046,   0.18051492,  0.17300266,  0.16567253,  0.15854528,   0.15170074,  0.14515442,  0.13889422,  0.13292373,   0.12723889,  0.12183088,  0.11672198,  0.11191495,   0.10740251,  0.10312256,  0.0990593 ,  0.09525083,   0.0916412 ,  0.08829702,  0.08522189,  0.08235639,   0.0797102 ,  0.07729729,  0.07502554,  0.07288924,   0.07088434,  0.06894849,  0.06704019,  0.06518787,   0.06353725,  0.0622083 ,  0.06114953,  0.06021326,   0.05958728,  0.05917813,  0.05893681,  0.05870529,   0.0585545 ,  0.05855088,  0.0586438 ,  0.05874717,   0.05885579,  0.05898676,  0.05921496,  0.05952665,   0.05982694,  0.06017409,  0.06063384,  0.06097649,   0.06123333,  0.0613199 ,  0.06146798,  0.06170739,   0.06197647,  0.06219188,  0.06237853,  0.06259421,   0.0628608 ,  0.06316329,  0.0634289 ,  0.06355304,   0.06354819,  0.06345794,  0.06328595,  0.06294231,   0.06245048,  0.06182061,  0.06104625,  0.06011864,   0.05894954,  0.05751931,  0.05581091,  0.05382092,   0.05151727,  0.04886673,  0.04587648,  0.04255237,   0.03891879,  0.03498621,  0.03068485,  0.02604904,   0.02116039,  0.01604088,  0.01043464,  0.00419863,  -0.0024695 , -0.00948138, -0.01692398, -0.02498189,  -0.03333946]])
        np.testing.assert_array_almost_equal(uf.mean.values, expected_mean)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf.covariance, DenseFunctionalData)
        self.assertIsInstance(uf._training_data, DenseFunctionalData)

        expected_eigenvalues = np.array([0.11991524, 0.08933699])
        np.testing.assert_array_almost_equal(uf.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions = np.array([[-1.63014726e+00, -1.45499104e+00, -1.28377829e+00,  -1.12502195e+00, -9.83616243e-01, -8.57834339e-01,  -7.43849772e-01, -6.39871878e-01, -5.45262854e-01,  -4.56458398e-01, -3.72573923e-01, -2.92525563e-01,  -2.13926218e-01, -1.34353279e-01, -5.51191480e-02,   2.08536784e-02,  9.62058739e-02,  1.74156012e-01,   2.53808738e-01,  3.31405540e-01,  4.06894963e-01,   4.84332065e-01,  5.60084372e-01,  6.30492068e-01,   6.94510389e-01,  7.50353230e-01,  7.95635187e-01,   8.30110532e-01,  8.55294738e-01,  8.73126543e-01,   8.87666941e-01,  9.00916564e-01,  9.13971461e-01,   9.28329747e-01,  9.43906067e-01,  9.59474940e-01,   9.77486528e-01,  1.00063752e+00,  1.02678217e+00,   1.05327991e+00,  1.07993020e+00,  1.10487329e+00,   1.12709093e+00,  1.14704665e+00,  1.16659015e+00,   1.18529637e+00,  1.20011411e+00,  1.21179441e+00,   1.22094450e+00,  1.22987724e+00,  1.24013533e+00,   1.24926239e+00,  1.25930915e+00,  1.27163855e+00,   1.28244803e+00,  1.29044256e+00,  1.29562928e+00,   1.30118269e+00,  1.30823162e+00,  1.31608438e+00,   1.32397155e+00,  1.32971754e+00,  1.33409385e+00,   1.33754100e+00,  1.34345389e+00,  1.35136076e+00,   1.35849548e+00,  1.36653356e+00,  1.37890436e+00,   1.39284758e+00,  1.40312884e+00,  1.40928418e+00,   1.40870182e+00,  1.40036408e+00,  1.38687220e+00,   1.36869719e+00,  1.34718816e+00,  1.31816357e+00,   1.28115221e+00,  1.23448744e+00,  1.17784237e+00,   1.11671526e+00,  1.05227487e+00,  9.84819396e-01,   9.17886913e-01,  8.50568831e-01,  7.79240521e-01,   7.02970900e-01,  6.23433906e-01,  5.44338500e-01,   4.65254553e-01,  3.86399749e-01,  3.03775997e-01,   2.15980154e-01,  1.23099628e-01,  2.67271292e-02,  -7.27176351e-02, -1.76405871e-01, -2.83869292e-01,  -3.94441066e-01, -5.23599231e-01], [-2.47689816e+00, -2.36165807e+00, -2.26639987e+00,  -2.19128594e+00, -2.13006212e+00, -2.07686320e+00,  -2.02353134e+00, -1.96860976e+00, -1.91058771e+00,  -1.84779069e+00, -1.78284129e+00, -1.72075837e+00,  -1.65884388e+00, -1.59423696e+00, -1.52622875e+00,  -1.45873634e+00, -1.39178056e+00, -1.32532697e+00,  -1.25769300e+00, -1.18811873e+00, -1.12062631e+00,  -1.05614527e+00, -9.94026753e-01, -9.31728838e-01,  -8.72459454e-01, -8.16292888e-01, -7.66157592e-01,  -7.23619963e-01, -6.84899564e-01, -6.51790533e-01,  -6.21936070e-01, -5.94334840e-01, -5.70559379e-01,  -5.51368599e-01, -5.34324304e-01, -5.16621213e-01,  -4.97177510e-01, -4.71839420e-01, -4.41780469e-01,  -4.09133080e-01, -3.72170646e-01, -3.28886175e-01,  -2.81712761e-01, -2.33711244e-01, -1.82942649e-01,  -1.26684102e-01, -7.13134295e-02, -2.23915632e-02,   2.02975433e-02,  5.48460500e-02,  7.90621918e-02,   8.78033544e-02,  8.36820494e-02,  6.97674105e-02,   5.24276121e-02,  3.32769748e-02,  1.45998289e-02,   1.24047284e-03, -1.20598315e-02, -2.35851498e-02,  -2.98274184e-02, -3.13973961e-02, -2.75963867e-02,  -2.46036750e-02, -2.62834009e-02, -3.06186089e-02,  -3.78862249e-02, -4.55129659e-02, -5.24225838e-02,  -6.08565882e-02, -7.54484440e-02, -9.56518775e-02,  -1.17197325e-01, -1.40235687e-01, -1.68182462e-01,  -2.03240740e-01, -2.41681372e-01, -2.82690505e-01,  -3.26777735e-01, -3.71696035e-01, -4.17228707e-01,  -4.61442045e-01, -5.06680538e-01, -5.54707021e-01,  -6.05382650e-01, -6.58888029e-01, -7.14314195e-01,  -7.73146499e-01, -8.33471513e-01, -8.97596510e-01,  -9.67371782e-01, -1.03856293e+00, -1.10813628e+00,  -1.17590917e+00, -1.24460177e+00, -1.31890754e+00,  -1.39882668e+00, -1.48824860e+00, -1.58911229e+00,  -1.70409942e+00, -1.83291058e+00]])
        np.testing.assert_array_almost_equal(np.abs(uf.eigenfunctions.values), np.abs(expected_eigenfunctions))

        expected_noise = 0.051559586753176664
        np.testing.assert_array_almost_equal(uf._noise_variance, expected_noise)

    def test_fit_inner_product(self):
        uf = UFPCA(n_components=2, method='inner-product', normalize=True)
        uf.fit(self.fdata_sparse)

        np.testing.assert_almost_equal(self.fdata_sparse.argvals.to_dense()['input_dim_0'], uf.mean.argvals['input_dim_0'])

        expected_mean = np.array([[ 0.28376655,  0.27226067,  0.26151169,  0.25122347,   0.24151566,  0.23249356,  0.2237986 ,  0.21527625,   0.20701826,  0.19913702,  0.19159585,  0.1844392 ,   0.17766052,  0.17118638,  0.16501919,  0.15916035,   0.1536228 ,  0.14839666,  0.14346798,  0.13882251,   0.13443622,  0.13029322,  0.12639942,  0.12274674,   0.11929506,  0.11601014,  0.1129029 ,  0.10998983,   0.10724221,  0.10463146,  0.10217847,  0.09990654,   0.0978084 ,  0.09587624,  0.09409314,  0.09244486,   0.09090257,  0.08941525,  0.08794176,  0.08649046,   0.08508513,  0.08368808,  0.08249575,  0.08169721,   0.08115533,  0.08075489,  0.08040056,  0.08009846,   0.07992895,  0.07981205,  0.07967847,  0.07956885,   0.07946866,  0.07935828,  0.07921175,  0.07907922,   0.07889339,  0.07851242,  0.07798815,  0.07735139,   0.07669992,  0.07609985,  0.07550173,  0.07487512,   0.0742502 ,  0.07360217,  0.07290657,  0.07218689,   0.07140302,  0.07060384,  0.06979056,  0.06888127,   0.06785737,  0.06669848,  0.06537975,  0.06389539,   0.06221831,  0.06034092,  0.05828097,  0.05602698,   0.0535707 ,  0.05089959,  0.04800349,  0.04486199,   0.04145401,  0.03778645,  0.03383956,  0.02959964,   0.02513171,  0.02041768,  0.01539678,  0.00996362,   0.00408526, -0.00220905, -0.00887934, -0.01581486,  -0.02315817, -0.03100882, -0.03959173, -0.04895527,  -0.05899069]])
        np.testing.assert_array_almost_equal(uf.mean.values, expected_mean)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf.covariance, DenseFunctionalData)
        self.assertIsInstance(uf._training_data, IrregularFunctionalData)

        expected_eigenvalues = np.array([3.54661948, 2.84364594])
        np.testing.assert_array_almost_equal(uf.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions = np.array([[ 1.82484292,  1.69186576,  1.5613364 ,  1.43490649,   1.30977342,  1.19043475,  1.0753156 ,  0.96228953,   0.85487456,  0.75036167,  0.64836968,  0.54871363,   0.45218711,  0.37552176,  0.2895316 ,  0.19689345,   0.1196305 ,  0.04458742, -0.02734514, -0.0950031 ,  -0.1620252 , -0.22734071, -0.28754856, -0.34532666,  -0.40214836, -0.45676793, -0.50620715, -0.55434194,  -0.60021128, -0.64531269, -0.68915136, -0.72882227,  -0.76613106, -0.8028847 , -0.83768717, -0.87078561,  -0.90229102, -0.93260297, -0.96152964, -0.98884177,  -1.01472905, -1.03993663, -1.06373985, -1.0871633 ,  -1.10959609, -1.13128749, -1.15233081, -1.17173959,  -1.19141611, -1.21084515, -1.22957209, -1.24790892,  -1.26604631, -1.28371937, -1.30071657, -1.31589974,  -1.33095077, -1.34452319, -1.35762184, -1.36934864,  -1.37932385, -1.38780791, -1.39463692, -1.39852694,  -1.40079698, -1.40198077, -1.40023118, -1.39572438,  -1.3897652 , -1.38012053, -1.36698013, -1.35156142,  -1.33269872, -1.3119045 , -1.28826533, -1.25899543,  -1.22615256, -1.19046063, -1.14998028, -1.10623378,  -1.05859447, -1.00563619, -0.94875313, -0.88625137,  -0.81924175, -0.74959489, -0.6738367 , -0.5919206 ,  -0.50586294, -0.41270031, -0.31560381, -0.21235088,  -0.10335351,  0.01140301,  0.13153438,  0.25725143,   0.39241732,  0.53857778,  0.68176546,  0.82995867,   0.93159492], [ 2.02139464,  1.97882449,  1.93405103,  1.88929125,   1.84520419,  1.8047213 ,  1.76238027,  1.7172929 ,   1.67242264,  1.62791497,  1.58447451,  1.54019536,   1.49564235,  1.45703657,  1.41509154,  1.36925688,   1.32715364,  1.2869818 ,  1.24731939,  1.20344963,   1.16210334,  1.12408615,  1.08432021,  1.04553446,   1.00637405,  0.96799693,  0.93121255,  0.89388898,   0.85757447,  0.82165185,  0.78635884,  0.75289615,   0.71952028,  0.68657871,  0.65481699,  0.62376278,   0.59384407,  0.56484443,  0.53624922,  0.50870079,   0.48232179,  0.45673022,  0.43236545,  0.40921686,   0.38747636,  0.36641973,  0.34610092,  0.3268875 ,   0.30934659,  0.29290821,  0.27793221,  0.26451645,   0.25205417,  0.24059363,  0.23004452,  0.22229834,   0.21571577,  0.21086034,  0.20616853,  0.20405793,   0.20354439,  0.20442007,  0.20655446,  0.21170153,   0.21839762,  0.22579875,  0.23572665,  0.24779193,   0.26074311,  0.27668569,  0.29457208,  0.31412499,   0.33642913,  0.35961648,  0.38461398,  0.41293419,   0.44383527,  0.47623262,  0.51172899,  0.5489071 ,   0.58896229,  0.63075355,  0.67491581,  0.72114616,   0.77124981,  0.82305351,  0.87811496,  0.93529542,   0.99429214,  1.05756212,  1.12283644,  1.1907861 ,   1.26192796,  1.33513273,  1.41178728,  1.4916955 ,   1.57585078,  1.66057614,  1.74727144,  1.83675047,   1.91063043]])
        np.testing.assert_array_almost_equal(np.abs(uf.eigenfunctions.values), np.abs(expected_eigenfunctions))

        expected_noise = 0.07236562113039097
        np.testing.assert_array_almost_equal(uf._noise_variance, expected_noise)

    def test_fit_inner_product_2d(self):
        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, dimension='2D', random_state=42
        )
        kl.new(n_obs=10)
        fdata = kl.data

        uf = UFPCA(n_components=2, method='inner-product')
        with np.testing.assert_warns(UserWarning):
            uf.fit(fdata)

        expected_mean = np.array([ 0.01931551,  0.02921439,  0.03871832,  0.04784962,  0.05662887,  0.06507482,  0.07319529,  0.08100521,  0.08851074,  0.09572043,  0.10263512,  0.10925907,  0.1155975 ,  0.12166136,  0.12747401,  0.13307525,  0.13840131,  0.14321654,  0.14753644,  0.15137652,  0.15475228,  0.15767923,  0.16017289,  0.16224875,  0.16392234,  0.16520915,  0.16612469,  0.16668448,  0.16690402,  0.16679882,  0.16638439,  0.16567623,  0.16468985,  0.16344077,  0.16194449,  0.16021651,  0.15827235,  0.15612744,  0.15379712,  0.15129653,  0.1486406 ,  0.14584395,  0.14292085,  0.13988517,  0.13675034,  0.13352929,  0.13023441,  0.12687751,  0.1234698 ,  0.12002184,  0.11654355,  0.11304417,  0.10953228,  0.10601578,  0.10250193,  0.09899734,  0.09550803,  0.09203942,  0.08859641,  0.08518338,  0.08180429,  0.07846269,  0.07516177,  0.07190444,  0.06869336,  0.06553103,  0.06241982,  0.05936205,  0.05636003,  0.05341608,  0.05053251,  0.04771162,  0.04495574,  0.04226718,  0.03964824,  0.03710124,  0.0346285 ,  0.03223232,  0.02991502,  0.02767891,  0.02552631,  0.02345952,  0.02148085,  0.01959263,  0.01779716,  0.01609676,  0.01444709,  0.0128357 ,  0.01127366,  0.00977023,  0.00833211,  0.00696422,  0.00567024,  0.00445352,  0.00331634,  0.00226106,  0.00128934,  0.00040323, -0.00039582, -0.00110645, -0.00172746])
        np.testing.assert_array_almost_equal(uf.mean.values[0, 1], expected_mean)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf._training_data, DenseFunctionalData)

        expected_eigenvalues = np.array([0.01098397, 0.0048397 ])
        np.testing.assert_array_almost_equal(uf.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions = np.array([ 0.19501408,  0.37830686,  0.55475685,  0.72452616,  0.88776485,  1.04461189,  1.1951709 ,  1.33956013,  1.47788332,  1.61025054,  1.7367911 ,  1.85767475,  1.97315048,  2.08360788,  2.18965596,  2.29210288,  2.38997858,  2.47974092,  2.56160814,  2.63579847,  2.70253016,  2.76202146,  2.81449061,  2.86015589,  2.89923556,  2.93194789,  2.95851118,  2.97914373,  2.99406383,  3.00348982,  3.00764003,  3.0067328 ,  3.00098649,  2.99061948,  2.97585013,  2.95689681,  2.93397779,  2.90731072,  2.87711183,  2.84359513,  2.80697164,  2.76744875,  2.72522966,  2.68051284,  2.63349166,  2.58435404,  2.5332821 ,  2.48045201,  2.42603375,  2.370191  ,  2.31308102,  2.25485466,  2.19565624,  2.13562366,  2.07488841,  2.01357567,  1.95180445,  1.88968778,  1.82733291,  1.76484166,  1.70231069,  1.63983199,  1.57749334,  1.51537886,  1.45356972,  1.39214487,  1.33118182,  1.27075745,  1.21094858,  1.151832  ,  1.0934845 ,  1.03598288,  0.97940394,  0.92382445,  0.86932122,  0.81597101,  0.76385062,  0.71303682,  0.6636064 ,  0.61563613,  0.56920279,  0.52438316,  0.48125402,  0.43989215,  0.40037433,  0.36277734,  0.32644466,  0.2911647 ,  0.25712963,  0.2245117 ,  0.19343287,  0.16397404,  0.13618535,  0.11011235,  0.08576731,  0.06317089,  0.04232394,  0.02324749,  0.00594495, -0.00957741, -0.02331022])
        np.testing.assert_array_almost_equal(np.abs(uf.eigenfunctions.values[0, 1]), np.abs(expected_eigenfunctions))

        expected_noise = 0.0
        np.testing.assert_array_almost_equal(uf._noise_variance, expected_noise)

    def test_error_method(self):
        uf = UFPCA(n_components=2, method='error')
        with self.assertRaises(NotImplementedError):
            uf.fit(self.fdata_uni)

    def test_error_covariance_2d(self):
        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, dimension='2D'
        )
        kl.new(n_obs=10)

        uf = UFPCA(n_components=2, method='covariance')
        with self.assertRaises(ValueError):
            uf.fit(kl.data)


class TestTransform(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=10)
        kl.sparsify(0.8, 0.05)

        self.fdata_uni = kl.data
        self.fdata_sparse = kl.sparse_data

        uf_dense = UFPCA(n_components=2, method='covariance', normalize=True)
        uf_dense.fit(self.fdata_uni)
        self.uf_dense = uf_dense
        
        uf_sparse = UFPCA(n_components=2, method='inner-product', normalize=True)
        uf_sparse.fit(self.fdata_sparse)
        self.uf_sparse = uf_sparse

    def test_error_innpro(self):
        with self.assertRaises(ValueError):
            self.uf_sparse.transform(self.fdata_uni, method='InnPro')

        with self.assertRaises(ValueError):
            self.uf_dense.transform(None, method='InnPro')

    def test_error_unkown_method(self):
        with self.assertRaises(ValueError):
            self.uf_dense.transform(self.fdata_uni, method='error')
    
    def test_data_none(self):
        scores = self.uf_dense.transform(None, method='NumInt')
        expected_scores = np.array([[ 1.03857683,  1.61120636],[ 0.02698638,  2.33010653],[ 1.38748226, -2.28995058],[ 0.33836467,  0.88750734],[ 0.84348845,  1.02930045],[ 1.29442399, -0.56477517],[-2.83064835, -0.8054215 ],[-2.63650019,  0.01381634],[-0.67908482, -0.68846887],[ 1.34022021, -1.28508822]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))

    def test_data_notnone(self):
        scores = self.uf_dense.transform(self.fdata_uni, method='NumInt')
        expected_scores = np.array([[ 1.16668875,  1.1781701 ],[ 0.1550983 ,  1.89707026],[ 1.51559417, -2.72298685],[ 0.46647658,  0.45447107],[ 0.97160037,  0.59626419],[ 1.42253591, -0.99781144],[-2.70253643, -1.23845777],[-2.50838828, -0.41921993],[-0.55097291, -1.12150514],[ 1.46833212, -1.71812449]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))

    def test_numint(self):
        scores = self.uf_dense.transform(self.fdata_uni, method='NumInt')
        expected_scores = np.array([[ 1.16668875,  1.1781701 ],[ 0.1550983 ,  1.89707026],[ 1.51559417, -2.72298685],[ 0.46647658,  0.45447107],[ 0.97160037,  0.59626419],[ 1.42253591, -0.99781144],[-2.70253643, -1.23845777],[-2.50838828, -0.41921993],[-0.55097291, -1.12150514],[ 1.46833212, -1.71812449]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))
    
        scores = self.uf_sparse.transform(self.fdata_sparse, method='NumInt')
        expected_scores = np.array([[-1.47448809, -1.43914632],[-0.2451687 , -2.36855161],[-1.81009102,  3.44062355],[-0.43170432, -0.37565379],[-1.22147414, -0.72201493],[-1.70518328,  1.31885992],[ 3.37004344,  1.43175512],[ 3.13354181,  0.47121724],[ 0.71058316,  1.38986738],[-1.78064819,  2.18141344]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))

    def test_pace(self):
        scores = self.uf_dense.transform(self.fdata_uni, method='PACE')
        expected_scores = np.array([[ 1.1406154 ,  1.17401501], [ 0.17205472,  1.91685814], [ 1.51264055, -2.71803916], [ 0.48106745,  0.4663187 ], [ 0.95334311,  0.57517276], [ 1.42649532, -0.99910737], [-2.71632551, -1.24535746], [-2.50739818, -0.41797038], [-0.5432319 , -1.10295379], [ 1.47464713, -1.70293616]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))
    
        scores = self.uf_sparse.transform(self.fdata_sparse, method='PACE')
        expected_scores = np.array([[-1.43636235, -1.5555992 ],[-0.28494179, -2.36035226],[-1.86811635,  3.37421264],[-0.48768007, -0.44165171],[-1.1848935 , -0.7100151 ],[-1.76289829,  1.31187716],[ 3.4886555 ,  1.6149282 ],[ 3.1437263 ,  0.48403803],[ 0.71227321,  1.38118874],[-1.77206206,  2.19828456]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))

    def test_innpro(self):
        scores = self.uf_sparse.transform(method='InnPro')
        expected_scores = np.array([[-1.3121429 , -1.94493404],[-0.09384769, -2.91629487],[-1.69314081,  2.89618655],[-0.43382027, -1.09186274],[-1.05862142, -1.25499434],[-1.60518627,  0.73376555],[ 3.56531104,  0.99841448],[ 3.2889    , -0.0419241 ],[ 0.85839191,  0.86656285],[-1.64872606,  1.64496551]])
        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))


class TestInverseTranform(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=10)
        kl.add_noise(0.05)
        kl.sparsify(0.95, 0.01)

        self.fdata_uni = kl.noisy_data
        self.fdata_sparse = kl.sparse_data

        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, argvals=np.linspace(0, 1, 21),
            dimension='2D', random_state=42
        )
        kl.new(n_obs=10)
        self.fdata = kl.data

    def test_inverse_tranform_1D_dense(self):
        uf_dense = UFPCA(n_components=4, method='covariance')
        uf_dense.fit(self.fdata_uni)
        scores_dense = uf_dense.transform(self.fdata_uni)

        fdata_inv = uf_dense.inverse_transform(scores_dense)
        
        expected_values = np.array([ 0.28386669, -1.37945681,  1.02263568, -0.77402862, -0.4972872 , -0.55773764,  2.00609315,  1.26005907,  0.8043036 ,  0.30744544])
        np.testing.assert_array_almost_equal(fdata_inv.values[:, 0], expected_values)

        expected_values = np.array([-0.27129124, -0.65766802,  0.75434737, -0.29719701, -0.2303506 ,  0.04830468,  0.83039506,  0.54284052,  0.40990883,  0.45802911])
        np.testing.assert_array_almost_equal(fdata_inv.values[:, 10], expected_values)
    
    def test_inverse_tranform_1D_sparse(self):
        uf_sparse = UFPCA(n_components=0.95, method='covariance')
        uf_sparse.fit(self.fdata_sparse)
        scores_sparse = uf_sparse.transform(self.fdata_sparse)

        fdata_inv = uf_sparse.inverse_transform(scores_sparse)
        
        expected_values = np.array([ 0.26667399, -1.21682852,  0.92784872, -0.96427364, -0.40289253, -0.39454108,  2.05518414,  0.8992658 ,  0.88881886,  0.4031194 ])
        np.testing.assert_array_almost_equal(fdata_inv.values[:, 0], expected_values)

        expected_values = np.array([-0.2474047 , -0.67959052,  0.78886515, -0.30766829, -0.23187046,  0.09402671,  0.92377868,  0.44696334,  0.52153204,  0.44421386])
        np.testing.assert_array_almost_equal(fdata_inv.values[:, 10], expected_values)
    
    def test_inverse_tranform_2D(self):
        uf_2d = UFPCA(n_components=2, method='inner-product')
        uf_2d.fit(self.fdata)
        scores_2d = uf_2d.transform(method='InnPro')

        fdata_inv = uf_2d.inverse_transform(scores_2d)
        
        expected_values = np.array([[-9.40549977e-02, -1.55483054e-01, -1.97161450e-01,  -2.22586724e-01, -2.34734095e-01, -2.37137912e-01,  -2.33629413e-01, -2.15164022e-01, -1.87197349e-01,  -1.54438803e-01, -1.20753640e-01, -8.91101537e-02,  -6.15620675e-02, -3.92661644e-02, -2.25352192e-02,  -1.09359326e-02, -4.51804915e-03, -2.59102302e-03,  -3.84083675e-03, -6.80912512e-03, -1.00919803e-02], [-6.92792236e-02, -1.17954339e-01, -1.48850063e-01,  -1.65011826e-01, -1.69450705e-01, -1.65556737e-01,  -1.56524449e-01, -1.36147108e-01, -1.09308914e-01,  -8.01999555e-02, -5.21861538e-02, -2.77329048e-02,  -8.37962944e-03,  5.23475353e-03,  1.32953824e-02,   1.66814080e-02,  1.57829402e-02,  1.15522080e-02,   5.29247572e-03, -1.66339998e-03, -8.11816440e-03], [-5.05017285e-02, -8.93200527e-02, -1.12301271e-01,  -1.21994515e-01, -1.21204084e-01, -1.13190643e-01,  -1.00057723e-01, -7.82802215e-02, -5.23033247e-02,  -2.59139588e-02, -2.09231490e-03,  1.70775599e-02,   3.04375607e-02,  3.77256998e-02,  3.94866430e-02,   3.69232421e-02,  3.08299559e-02,  2.21733783e-02,   1.22150567e-02,  2.19805070e-03, -6.76998202e-03], [-3.84343745e-02, -6.94687271e-02, -8.66925386e-02,  -9.20616754e-02, -8.79873709e-02, -7.75550759e-02,  -6.14929757e-02, -3.87014405e-02, -1.33050222e-02,   1.12034476e-02,  3.21234818e-02,  4.76445537e-02,   5.68786301e-02,  5.98276049e-02,  5.72856576e-02,   5.06786332e-02,  4.11245890e-02,  2.94518968e-02,   1.68916178e-02,  4.62715685e-03, -6.25811385e-03], [-3.18287979e-02, -5.72845685e-02, -7.06052561e-02,  -7.33076338e-02, -6.74142276e-02, -5.50950730e-02,  -3.74373732e-02, -1.42728709e-02,  1.05089965e-02,   3.36245737e-02,  5.25661582e-02,  6.57057104e-02,   7.23294109e-02,  7.26030699e-02,  6.74684102e-02,   5.84728979e-02,  4.68363802e-02,  3.34393019e-02,   1.93814393e-02,  5.80597192e-03, -6.19499418e-03], [-2.84067375e-02, -5.06237149e-02, -6.16734073e-02,  -6.29535215e-02, -5.57431899e-02, -4.24212855e-02,  -2.42869236e-02, -1.34623289e-03,  2.26939328e-02,   4.47038986e-02,  6.23075587e-02,  7.39922763e-02,   7.91455077e-02,  7.80181588e-02,  7.16146769e-02,   6.15166627e-02,  4.89340963e-02,  3.47801673e-02,   2.01778704e-02,  6.16902445e-03, -6.14942996e-03], [-2.55215983e-02, -4.70448610e-02, -5.74654013e-02,  -5.83305750e-02, -5.10570957e-02, -3.79809698e-02,  -2.03823164e-02,  1.75562446e-03,  2.48668016e-02,   4.59440019e-02,  6.27078259e-02,  7.37118190e-02,   7.83775095e-02,  7.69593618e-02,  7.04396436e-02,   6.03594500e-02,  4.79239061e-02,  3.40337379e-02,   1.97828605e-02,  6.18479480e-03, -5.70729194e-03], [-2.34499959e-02, -4.71044627e-02, -5.92758881e-02,  -6.16770866e-02, -5.58699382e-02, -4.41917746e-02,  -2.79042564e-02, -6.87242773e-03,  1.54193641e-02,   3.60261611e-02,  5.27037559e-02,  6.40089486e-02,   6.93329639e-02,  6.88680351e-02,  6.35071485e-02,   5.46826804e-02,  4.35835021e-02,  3.10727826e-02,   1.81751899e-02,  5.84565951e-03, -4.93812845e-03], [-2.18647694e-02, -4.90208927e-02, -6.41663879e-02,  -6.91607607e-02, -6.56892543e-02, -5.60989639e-02,  -4.16262399e-02, -2.19692533e-02, -5.45048394e-04,   1.97456743e-02,  3.66715395e-02,  4.87654364e-02,   5.53561653e-02,  5.65367984e-02,  5.30697459e-02,   4.62337707e-02,  3.71860262e-02,  2.67327260e-02,   1.58018273e-02,  5.26803033e-03, -3.98777044e-03], [-2.05509973e-02, -5.13088409e-02, -6.96491422e-02,  -7.75221742e-02, -7.66849832e-02, -6.94692268e-02,  -5.70698140e-02, -3.90136533e-02, -1.86280502e-02,   1.24749570e-03,  1.84047426e-02,  3.13556529e-02,   3.93620438e-02,  4.24059471e-02,  4.11006711e-02,   3.65473460e-02,  2.98521264e-02,  2.17498258e-02,   1.30556431e-02,  4.55017105e-03, -2.99428196e-03], [-1.93564883e-02, -5.28244162e-02, -7.37943171e-02,  -8.42222987e-02, -8.58632263e-02, -8.09825330e-02,  -7.07097764e-02, -5.44322215e-02, -3.53382217e-02,  -1.61634751e-02,  9.41541969e-04,  1.44937321e-02,   2.37047641e-02,  2.84536976e-02,  2.92048357e-02,   2.68751561e-02,  2.24983302e-02,  1.67278212e-02,   1.02597456e-02,  3.77347472e-03, -2.07099884e-03], [-1.81534506e-02, -5.27897392e-02, -7.52948166e-02,  -8.75327727e-02, -9.11726428e-02, -8.83492907e-02,  -8.01000833e-02, -6.57273273e-02, -4.82158473e-02,  -3.01398143e-02, -1.35424450e-02,  1.38556556e-04,   1.00958539e-02,  1.61276353e-02,  1.85630660e-02,   1.81419853e-02,  1.58044654e-02,  1.21169324e-02,   7.65787405e-03,  3.00400130e-03, -1.29438171e-03], [-1.68257236e-02, -5.08011439e-02, -7.34878890e-02,  -8.65659923e-02, -9.15399156e-02, -9.03493984e-02,  -8.39158165e-02, -7.15197738e-02, -5.58744920e-02,  -3.93307123e-02, -2.37698831e-02, -1.05379130e-02,  -4.23086755e-04,  6.32186228e-03,  9.91351545e-03,   1.09309545e-02,  1.02027999e-02,  8.20686012e-03,   5.41119749e-03,  2.29299237e-03, -6.99963228e-04], [-1.52815621e-02, -4.68209799e-02, -6.83335186e-02,  -8.12450153e-02, -8.68342784e-02, -8.67931836e-02,  -8.19112050e-02, -7.15061285e-02, -5.79594153e-02,  -4.33428289e-02, -2.93259934e-02, -1.71189671e-02,  -7.45309763e-03, -6.00155847e-04,  3.57024615e-03,   5.49821265e-03,  5.88889676e-03,  5.13378339e-03,   3.60151599e-03,  1.67636001e-03, -2.86393249e-04], [-1.34919629e-02, -4.11530080e-02, -6.03495954e-02,  -7.22132721e-02, -7.77611037e-02, -7.84042312e-02,  -7.47937068e-02, -6.63307326e-02, -5.50228277e-02,  -4.26228686e-02, -3.05513827e-02, -1.98481356e-02,  -1.11559910e-02, -4.73704179e-03, -5.21014514e-04,   1.81702276e-03,  2.85418894e-03,  2.90135535e-03,   2.24086404e-03,  1.17314722e-03, -2.75856048e-05], [-1.15401529e-02, -3.43972856e-02, -5.05047063e-02,  -6.06870407e-02, -6.56881430e-02, -6.66275294e-02,  -6.40188100e-02, -5.73778476e-02, -4.83218721e-02,  -3.82678709e-02, -2.83692784e-02, -1.94763492e-02,  -1.21229475e-02, -6.53975022e-03, -2.69386766e-03,  -3.52153992e-04,  9.37606575e-04,  1.41405530e-03,   1.28717398e-03,  7.84249626e-04,  1.10319003e-04], [-8.10368747e-03, -2.60318674e-02, -3.89610574e-02,  -4.73690836e-02, -5.15957478e-02, -5.25800073e-02,  -5.08075518e-02, -4.59119936e-02, -3.91132554e-02,  -3.14834613e-02, -2.38986923e-02, -1.70086753e-02,  -1.12266781e-02, -6.73960874e-03, -3.53832216e-03,  -1.46586439e-03, -1.90851631e-04,  4.50990127e-04,   6.03526916e-04,  4.38993698e-04,  1.08672260e-04], [-3.37074286e-03, -1.66124594e-02, -2.63895222e-02,  -3.28871443e-02, -3.62612263e-02, -3.71444042e-02,  -3.60862873e-02, -3.28528788e-02, -2.82721358e-02,  -2.30730898e-02, -1.78535758e-02, -1.30599852e-02,  -8.98051652e-03, -5.75192173e-03, -3.37974967e-03,  -1.77061709e-03, -6.98065580e-04, -7.50647373e-05,   1.83688885e-04,  1.94709973e-04,  6.47707011e-05], [ 2.00375909e-03, -7.00799360e-03, -1.37941669e-02,  -1.83320108e-02, -2.07205432e-02, -2.14416945e-02,  -2.09926144e-02, -1.92929195e-02, -1.68024123e-02,  -1.39252329e-02, -1.09940245e-02, -8.26023892e-03,  -5.89090489e-03, -3.97185921e-03, -2.51743943e-03,  -1.48631740e-03, -7.33043324e-04, -2.53605503e-04,  -1.13546520e-05,  5.96854363e-05,  2.71614626e-05], [ 7.19446773e-03,  1.86470555e-03, -2.16275500e-03,  -4.74411332e-03, -6.05090525e-03, -6.50700190e-03,  -6.58481140e-03, -6.25837086e-03, -5.65856058e-03,  -4.89656466e-03, -4.06465327e-03, -3.23739779e-03,  -2.47207713e-03, -1.80827364e-03, -1.26665766e-03,  -8.48203936e-04, -4.73501822e-04, -2.03498998e-04,  -4.86469081e-05,  1.19566065e-05,  1.14057103e-05], [ 1.13376540e-02,  9.16358043e-03,  7.61760640e-03,   6.91758118e-03,  6.70761380e-03,  6.55717957e-03,   6.03232321e-03,  5.19871721e-03,  4.19826606e-03,   3.16560425e-03,  2.21116730e-03,  1.40906694e-03,   7.93047388e-04,  3.60524283e-04,  8.47078172e-05,  -6.84667216e-05, -7.34153133e-05, -3.19571495e-05,   4.41797697e-06,  1.83029472e-05,  1.30316544e-05]])
        np.testing.assert_array_almost_equal(fdata_inv.values[0], expected_values)

        expected_values = np.array([[ 2.74722416e-02,  1.03095145e-01,  1.67819354e-01,   2.21706650e-01,  2.64905913e-01,  2.98269564e-01,   3.23693113e-01,  3.35806862e-01,  3.35990776e-01,   3.25735087e-01,  3.06638246e-01,  2.80402582e-01,   2.48832838e-01,  2.13837425e-01,  1.77431739e-01,   1.41740354e-01,  1.05585754e-01,  7.05084746e-02,   3.82235847e-02,  1.02558123e-02, -1.22315897e-02], [ 2.40118407e-02,  9.74432715e-02,  1.60851364e-01,   2.14411853e-01,  2.58294483e-01,  2.93116804e-01,   3.19967144e-01,  3.34180421e-01,  3.36705915e-01,   3.28655674e-01,  3.11333234e-01,  2.86249922e-01,   2.55130263e-01,  2.19906371e-01,  1.82700898e-01,   1.45798428e-01,  1.08741259e-01,  7.29434620e-02,   4.00098697e-02,  1.13330361e-02, -1.18706923e-02], [ 1.97785454e-02,  9.04713194e-02,  1.51947542e-01,   2.04362683e-01,  2.47919543e-01,  2.83184436e-01,   3.10434686e-01,  3.25806616e-01,  3.29922656e-01,   3.23605970e-01,  3.07926201e-01,  2.84226922e-01,   2.54134748e-01,  2.19549987e-01,  1.82618731e-01,   1.45687171e-01,  1.08852380e-01,  7.32715281e-02,   4.04809250e-02,  1.17964338e-02, -1.15332190e-02], [ 1.53125264e-02,  8.25962481e-02,  1.41519566e-01,   1.92113906e-01,  2.34580341e-01,  2.69506803e-01,   2.96284552e-01,  3.12011350e-01,  3.17068575e-01,   3.12061451e-01,  2.97873300e-01,  2.75698422e-01,   2.47052997e-01,  2.13763888e-01,  1.77935558e-01,   1.41895842e-01,  1.06147575e-01,  7.15738450e-02,   3.96505161e-02,  1.16300841e-02, -1.12345320e-02], [ 1.14511635e-02,  7.44101788e-02,  1.29964449e-01,   1.77990394e-01,  2.18676912e-01,  2.52133178e-01,   2.77787249e-01,  2.93271724e-01,  2.98798746e-01,   2.94812918e-01,  2.82049112e-01,  2.61567481e-01,   2.34765017e-01,  2.03363709e-01,  1.69375465e-01,   1.35044321e-01,  1.00994263e-01,  6.81067355e-02,   3.77114712e-02,  1.09825448e-02, -1.08598861e-02], [ 8.84018995e-03,  6.65900850e-02,  1.17974006e-01,   1.62778593e-01,  2.00720176e-01,  2.31931906e-01,   2.55988168e-01,  2.70792390e-01,  2.76443043e-01,   2.73267180e-01,  2.61879306e-01,  2.43217085e-01,   2.18553163e-01,  1.89483172e-01,  1.57889964e-01,   1.25884750e-01,  9.40915223e-02,  6.33711322e-02,   3.50284600e-02,  1.01174687e-02, -1.02233933e-02], [ 7.88739144e-03,  5.97030407e-02,  1.05966481e-01,   1.46460252e-01,  1.80896032e-01,  2.09354358e-01,   2.31418421e-01,  2.45173443e-01,  2.50653147e-01,   2.48105027e-01,  2.38048163e-01,  2.21308571e-01,   1.99030936e-01,  1.72666809e-01,  1.43939233e-01,   1.14784931e-01,  8.58074878e-02,  5.78116074e-02,   3.19981726e-02,  9.33836285e-03, -9.13017951e-03], [ 7.19500303e-03,  5.28502577e-02,  9.36396457e-02,   1.29373342e-01,  1.59789453e-01,  1.84954035e-01,   2.04486243e-01,  2.16675695e-01,  2.21548549e-01,   2.19318617e-01,  2.10440544e-01,  1.95642546e-01,   1.75937307e-01,  1.52611088e-01,  1.27191004e-01,   1.01391834e-01,  7.57675787e-02,  5.10415511e-02,   2.82788140e-02,  8.34004981e-03, -7.86554858e-03], [ 6.66635300e-03,  4.61466361e-02,  8.13541722e-02,   1.12145030e-01,  1.38302387e-01,  1.59901838e-01,   1.76615330e-01,  1.86948212e-01,  1.90954722e-01,   1.88844289e-01,  1.81027763e-01,  1.68146102e-01,   1.51079934e-01,  1.30940056e-01,  1.09038798e-01,   8.68437543e-02,  6.48410046e-02,  4.36577016e-02,   2.42058200e-02,  7.22046302e-03, -6.53157777e-03], [ 6.21062769e-03,  3.96935190e-02,  6.94315538e-02,   9.53331909e-02,  1.17234629e-01,  1.35232901e-01,   1.49060974e-01,  1.57442023e-01,  1.60475442e-01,   1.58381287e-01,  1.51538394e-01,  1.40508230e-01,   1.26042847e-01,  1.09077012e-01,  9.07044427e-02,   7.21396464e-02,  5.37912536e-02,  3.61842070e-02,   2.00727058e-02,  6.06206227e-03, -5.22534203e-03], [ 5.75527793e-03,  3.35794204e-02,  5.81430822e-02,   7.94042091e-02,  9.72513648e-02,  1.11805169e-01,   1.22860884e-01,  1.29353957e-01,  1.31431074e-01,   1.29326470e-01,  1.23391676e-01,  1.14114289e-01,   1.02124244e-01,  8.81876437e-02,  7.31880738e-02,   5.80968363e-02,  4.32427400e-02,  2.90495659e-02,   1.61193051e-02,  4.93208710e-03, -4.02649307e-03], [ 5.25484203e-03,  2.78809982e-02,  4.77031219e-02,   6.47190878e-02,  7.88628277e-02,  9.02735654e-02,   9.88046927e-02,  1.03591955e-01,  1.04820585e-01,   1.02734203e-01,  9.76566200e-02,  9.00057429e-02,   8.02982347e-02,  6.91449621e-02,  5.72371757e-02,   4.53246879e-02,  3.36599362e-02,  2.25722039e-02,   1.25245779e-02,  3.88324934e-03, -2.98833907e-03], [ 4.69381477e-03,  2.26633466e-02,  3.82668615e-02,   5.15288192e-02,  6.24175178e-02,  7.10813777e-02,   7.74238083e-02,  8.07635327e-02,  8.13089103e-02,   7.93030003e-02,  7.50383804e-02,  6.88667344e-02,   6.12020698e-02,  5.25176026e-02,  4.33362529e-02,   3.42158132e-02,  2.53404080e-02,  1.69556648e-02,   9.40422085e-03,  2.95398279e-03, -2.13483907e-03], [ 4.08364096e-03,  1.79796711e-02,  2.99325830e-02,   3.99790544e-02,  4.81090445e-02,  5.44689519e-02,   5.90016585e-02,  6.11874188e-02,  6.12396724e-02,   5.93889340e-02,  5.58919082e-02,  5.10375082e-02,   4.51488820e-02,  3.85814794e-02,  3.17171081e-02,   2.49548107e-02,  1.84217364e-02,  1.22933953e-02,   6.81305830e-03,  2.16822474e-03, -1.46354324e-03], [ 3.45392910e-03,  1.38704149e-02,  2.27485250e-02,   3.01241486e-02,  3.59966082e-02,  4.04996465e-02,   4.36042773e-02,  4.49282989e-02,  4.46732118e-02,   4.30457797e-02,  4.02628526e-02,  3.65546585e-02,   3.21659191e-02,  2.73548141e-02,  2.23897539e-02,   1.75445525e-02,  1.29023441e-02,  8.58312960e-03,   4.75220245e-03,  1.53470630e-03, -9.54511246e-04], [ 2.84022490e-03,  1.03626407e-02,  1.67239603e-02,   2.19493790e-02,  2.60374588e-02,  2.91012397e-02,   3.11294688e-02,  3.18529752e-02,  3.14482815e-02,   3.00902896e-02,  2.79541558e-02,  2.52166753e-02,   2.20567576e-02,  1.86550148e-02,  1.51924324e-02,   1.18484905e-02,  8.67485612e-03,  5.74995172e-03,   3.18081188e-03,  1.04668999e-03, -5.82746794e-04], [ 1.71433437e-03,  6.92190639e-03,  1.13304305e-02,   1.49393381e-02,  1.76610591e-02,  1.96079154e-02,   2.08220475e-02,  2.11381934e-02,  2.06991768e-02,   1.96427920e-02,  1.81017424e-02,  1.62037368e-02,   1.40715313e-02,  1.18229334e-02,  9.57074750e-03,   7.42283796e-03,  5.39598124e-03,  3.52720780e-03,   1.88668201e-03,  5.55299959e-04, -4.31993781e-04], [ 4.07386857e-04,  3.80979377e-03,  6.68577306e-03,   9.00606462e-03,  1.07038085e-02,  1.18215417e-02,   1.24483768e-02,  1.25194759e-02,  1.21381384e-02,   1.14018654e-02,  1.04013293e-02,  9.21979934e-03,   7.93295352e-03,  6.60909208e-03,  5.30974209e-03,   4.09069197e-03,  2.92842466e-03,  1.84108572e-03,   8.98926633e-04,  1.51601437e-04, -3.80495672e-04], [-8.35811898e-04,  1.19903237e-03,  2.90476040e-03,   4.23270610e-03,  5.14509099e-03,  5.68802321e-03,   5.92828015e-03,  5.88957272e-03,  5.63327187e-03,   5.21606443e-03,  4.68886564e-03,  4.09616655e-03,   3.47581319e-03,  2.85923098e-03,  2.27209459e-03,   1.73542345e-03,  1.18997450e-03,  6.70900455e-04,   2.21023521e-04, -1.26723263e-04, -3.56959497e-04], [-1.84496482e-03, -8.18737986e-04,  1.26725649e-05,   5.91845362e-04,  9.22471052e-04,  1.06074161e-03,   1.07978133e-03,  1.03399044e-03,  9.43720470e-04,   8.26842545e-04,  6.98391106e-04,  5.70361449e-04,   4.51631646e-04,  3.48021678e-04,  2.62498055e-04,   1.95541643e-04,  7.93605455e-05, -5.62244801e-05,  -1.83332612e-04, -2.78216610e-04, -3.25537176e-04], [-2.50115888e-03, -2.20454355e-03, -2.00950748e-03,  -1.97789874e-03, -2.05455831e-03, -2.16880034e-03,  -2.24061550e-03, -2.22391765e-03, -2.13501815e-03,  -1.99017873e-03, -1.80460594e-03, -1.59175711e-03,  -1.36312722e-03, -1.12850324e-03, -8.96672580e-04,  -6.76415374e-04, -5.27346687e-04, -4.31273915e-04,  -3.69050425e-04, -3.22045087e-04, -2.72915881e-04]])
        np.testing.assert_array_almost_equal(fdata_inv.values[9], expected_values)
