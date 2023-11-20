#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the class UFPCA in the fpca.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
import warnings

from FDApy.representation.argvals import DenseArgvals
from FDApy.representation.values import DenseValues
from FDApy.representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData
)
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import (
    UFPCA
)


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

        expected_eigenfunctions = np.array([[ 1.54234035,  1.45461587,  1.36820796,  1.28421951,   1.20231272,  1.12153204,  1.04256142,  0.96567772,   0.89079472,  0.81723053,  0.74462799,  0.67316341,   0.60291742,  0.53406782,  0.46677732,  0.40117411,   0.33726895,  0.27508326,  0.21449611,  0.15541334,   0.09786872,  0.04189539, -0.01261593, -0.06577094,  -0.11754504, -0.16797938, -0.217038  , -0.26481032,  -0.311341  , -0.35653154, -0.40059412, -0.44381341,  -0.48598567, -0.52688953, -0.5665981 , -0.60511235,  -0.64250273, -0.67889667, -0.71428805, -0.74874379,  -0.78236923, -0.81533492, -0.84733768, -0.87723909,  -0.90476289, -0.92999342, -0.95316288, -0.97433819,  -0.99295186, -1.0092072 , -1.02345326, -1.03532156,  -1.04471303, -1.05187114, -1.0571822 , -1.06092752,  -1.06354439, -1.06545684, -1.06587542, -1.0642383 ,  -1.06129197, -1.05732507, -1.05212868, -1.04585409,  -1.03864877, -1.03025525, -1.02063213, -1.00971257,  -0.99729653, -0.98366745, -0.9691341 , -0.95342753,  -0.93612528, -0.9171188 , -0.89635936, -0.87378118,  -0.849308  , -0.8228687 , -0.79450608, -0.76413179,  -0.73166479, -0.69712118, -0.66048109, -0.62162947,  -0.58055957, -0.53739869, -0.49190694, -0.44386926,  -0.39364709, -0.34117348, -0.28624662, -0.22812619,  -0.16650422, -0.10173428, -0.03354899,  0.03744123,   0.11097687,  0.18666534,  0.26538087,  0.34750016,   0.43239947], [ 1.99261533,  1.95163423,  1.9100175 ,  1.86844644,   1.82741642,  1.78679811,  1.74626018,  1.70567015,   1.66490714,  1.62429295,  1.58382456,  1.5433463 ,   1.50309111,  1.46343431,  1.42443381,  1.38590525,   1.34775392,  1.31004786,  1.27289678,  1.23634295,   1.20035155,  1.16498395,  1.13028288,  1.09631238,   1.0631635 ,  1.03089151,  0.99955518,  0.96905562,   0.93941915,  0.91065074,  0.8826341 ,  0.85535049,   0.82890248,  0.80335703,  0.77864197,  0.75479719,   0.73186203,  0.70975553,  0.68829694,  0.66744335,   0.64723838,  0.62763606,  0.60888291,  0.59106716,   0.57415799,  0.55878037,  0.54503153,  0.53278232,   0.52187124,  0.51213769,  0.50393909,  0.49724614,   0.49180848,  0.48784428,  0.48553361,  0.48440794,   0.48425485,  0.48549028,  0.48850695,  0.49319407,   0.49877077,  0.50495311,  0.51199343,  0.51977491,   0.52837773,  0.53805838,  0.5487811 ,  0.56062243,   0.57354792,  0.5871945 ,  0.6015516 ,  0.61681901,   0.63314874,  0.65072898,  0.66956878,  0.68970664,   0.71120999,  0.73416101,  0.75855959,  0.78444227,   0.81181397,  0.84070293,  0.87115728,  0.90318437,   0.93679523,  0.97196811,  1.00871085,  1.04695526,   1.08664399,  1.12779555,  1.17033449,  1.21456125,   1.26080167,  1.3086292 ,  1.35779889,  1.40813291,   1.4596142 ,  1.51235882,  1.56711081,  1.62421027,   1.68260963]])
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


# class TestPace(unittest.TestCase):
#     def setUp(self):
#         argvals = np.linspace(0, 1, 10)
#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals, 
#             n_functions=5, random_state=42
#         )
#         kl.new(n_obs=50)
#         self.data = kl.data

#     def test_pace(self):
#         self.data.covariance()
        
#         uf = UFPCA(n_components=2, method='covariance')
#         uf.fit(self.data)

#         scores = uf._pace(self.data)

#         expected_scores = np.array([
#             [ 1.46015886e+00,  2.04695739e+00],
#             [ 4.94950452e-01,  1.78515078e-01],
#             [ 2.15517571e-01, -1.99545738e-01],
#             [ 4.73664501e-01, -1.56381155e-01],
#             [ 7.73468093e-01,  2.56786248e-01]
#         ])
#         np.testing.assert_array_almost_equal(
#             np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#         )


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

    # def test_error_innpro(self):
    #     uf = UFPCA(n_components=2, method='covariance')
    #     uf.fit(self.fdata_uni)
    #     with self.assertRaises(ValueError):
    #         uf.transform(None, method='InnPro')

    #     uf = UFPCA(n_components=2, method='inner-product')
    #     uf.fit(self.fdata_uni)
    #     with self.assertRaises(ValueError):
    #         uf.transform(self.fdata_uni, method='InnPro')

    def test_error_unkown_method(self):
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.fdata_uni)

        with self.assertRaises(ValueError):
            uf.transform(self.fdata_uni, method='error')

#     def test_pace(self):
#         self.data.covariance()

#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data)

#         scores = uf.transform(self.data, method='PACE')
#         expected_scores = np.array([
#             [-1.35951225, -1.83425839],
#             [-0.39430398,  0.03418355],
#             [-0.11487114,  0.41224429],
#             [-0.37301803,  0.36907972],
#             [-0.67282157, -0.0440876 ]
#         ])
#         np.testing.assert_array_almost_equal(
#             np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#         )

#     def test_numint(self):
#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data)

#         scores = uf.transform(self.data, method='NumInt')
#         expected_scores = np.array([
#             [-1.32124942, -1.7905831 ],
#             [-0.54539202, -0.13829588],
#             [-0.01131015,  0.53047647],
#             [-0.39974495,  0.33857587],
#             [-0.68649162, -0.05968698]
#         ])
#         np.testing.assert_array_almost_equal(
#             np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#         )

    # def test_innpro(self):
    #     uf = UFPCA(n_components=2, method='inner-product')
    #     uf.fit(self.data_uni)

    #     scores = uf.transform(self.data, method='InnPro')
    #     expected_scores = np.array([
    #         [-1.32124942, -1.7905831 ],
    #         [-0.54539202, -0.13829588],
    #         [-0.01131015,  0.53047647],
    #         [-0.39974495,  0.33857587],
    #         [-0.68649162, -0.05968698]
    #     ])
    #     np.testing.assert_array_almost_equal(
    #         np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
    #     )

#     # def test_normalize(self):
#     #     uf = UFPCA(n_components=2, method='inner-product', normalize=True)
#     #     uf.fit(self.data)

#     #     scores = uf.transform(self.data, method='InnPro')
#     #     expected_scores = np.array([
#     #         [-0.30298673, -0.41061355],
#     #         [-0.1250684 , -0.03171378],
#     #         [-0.00259362,  0.12164799],
#     #         [-0.09166885,  0.07764166],
#     #         [-0.15742512, -0.01368732]
#     #     ])
#     #     np.testing.assert_array_almost_equal(
#     #         np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#     #     )


# class TestInverseTranform(unittest.TestCase):
#     def setUp(self):
#         warnings.simplefilter('ignore', category=UserWarning)

#         argvals = np.linspace(0, 1, 10)
#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals,
#             n_functions=5, random_state=42
#         )
#         kl.new(n_obs=50)
#         self.data_1d = kl.data

#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals,
#             n_functions=5, dimension='2D', random_state=42
#         )
#         kl.new(n_obs=50)
#         self.data_2d = kl.data

#     def test_inverse_tranform_1D(self):
#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data_1d)
#         scores = uf.transform(self.data_1d)

#         expected_data = uf.weights * np.dot(scores, uf.eigenfunctions.values)
#         data = uf.inverse_transform(scores)

#         np.testing.assert_array_almost_equal(
#             data.values,
#             expected_data + uf.mean.values
#         )
    
#     def test_inverse_tranform_2D(self):
#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data_2d)
#         scores = uf.transform(self.data_2d)

#         expected_data = uf.weights * np.einsum(
#             'ij,jkl->ikl',
#             scores,
#             uf.eigenfunctions.values
#         )
#         data = uf.inverse_transform(scores)

#         np.testing.assert_array_almost_equal(
#             data.values,
#             expected_data + uf.mean.values
#         )

#     def test_error(self):
#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data_1d)
#         scores = uf.transform(self.data_1d)

#         argvals = {
#             'input_dim_0': np.array([3, 4, 3]),
#             'input_dim_1': np.array([5, 6]),
#             'input_dim_2': np.array([1, 2, 4])
#         }
#         values = np.array([
#             [
#                 [[1, 2, 3], [1, 2, 3]],
#                 [[5, 6, 7], [5, 6, 7]],
#                 [[3, 4, 5], [3, 4, 5]]
#             ]
#         ])
#         uf._eigenfunctions = DenseFunctionalData(DenseArgvals(argvals), DenseValues(values))

#         with self.assertRaises(ValueError):
#             uf.inverse_transform(scores)
