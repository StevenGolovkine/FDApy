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
from FDApy.representation.functional_data import DenseFunctionalData
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

        self.fdata_uni = kl.data
        self.fdata_sparse = kl.sparse_data

    def test_fit_covariance(self):
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.fdata_uni)

        np.testing.assert_almost_equal(self.fdata_uni.argvals['input_dim_0'], uf.mean.argvals['input_dim_0'])

        expected_mean = np.array([ 0.25140901,  0.24368371,  0.23613364,  0.22875992,   0.22156365,  0.21454593,  0.20770789,  0.20105062,   0.19457524,  0.18828286,  0.18217458,  0.17625151,   0.17051474,  0.16496535,  0.15960444,  0.15443302,   0.14945205,  0.1446623 ,  0.14006437,  0.13565859,   0.13144498,  0.12742319,  0.12359247,  0.11995162,   0.11649893,  0.11323215,  0.11014846,  0.1072444 ,   0.10451587,  0.10195804,  0.09956531,  0.09733124,   0.09524849,  0.09330869,  0.0915023 ,  0.08981849,   0.08824486,  0.08676714,  0.0853688 ,  0.08403038,   0.08276789,  0.08167392,  0.0807398 ,  0.0799563 ,   0.07931366,  0.07880153,  0.07840899,  0.07812456,   0.07793619,  0.07783122,  0.07779644,  0.07781805,   0.07788167,  0.07797232,  0.07807447,  0.07817198,   0.07824816,  0.07828572,  0.07826684,  0.0781731 ,   0.07798553,  0.07768462,  0.07739751,  0.07715871,   0.07695264,  0.07676479,  0.07658136,  0.07638902,   0.07617468,  0.0759254 ,  0.07562824,  0.07527024,   0.0748383 ,  0.07431915,  0.07369935,  0.07296523,   0.07210287,  0.07109813,  0.06993659,  0.06860361,   0.06708432,  0.06536363,  0.06342625,  0.06125675,   0.05883955,  0.056159  ,  0.0531994 ,  0.04994505,   0.0463803 ,  0.04248959,  0.03825745,  0.03366854,   0.02870763,  0.02335957,  0.01760933,  0.01144195,   0.00484258, -0.00220354, -0.0097111 , -0.01769469,  -0.02616881])
        np.testing.assert_array_almost_equal(uf.mean.values[0], expected_mean)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf.covariance, DenseFunctionalData)

    def test_fit_inner_product(self):
        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(self.fdata_sparse)

        np.testing.assert_almost_equal(self.fdata_sparse.argvals.to_dense()['input_dim_0'], uf.mean.argvals['input_dim_0'])

        expected_mean = np.array([ 0.28376655,  0.27226067,  0.26151169,  0.25122347,0.24151566,  0.23249356,  0.2237986 ,  0.21527625,0.20701826,  0.19913702,  0.19159585,  0.1844392 ,0.17766052,  0.17118638,  0.16501919,  0.15916035,0.1536228 ,  0.14839666,  0.14346798,  0.13882251,0.13443622,  0.13029322,  0.12639942,  0.12274674,0.11929506,  0.11601014,  0.1129029 ,  0.10998983,0.10724221,  0.10463146,  0.10217847,  0.09990654,0.0978084 ,  0.09587624,  0.09409314,  0.09244486,0.09090257,  0.08941525,  0.08794176,  0.08649046,0.08508513,  0.08368808,  0.08249575,  0.08169721,0.08115533,  0.08075489,  0.08040056,  0.08009846,0.07992895,  0.07981205,  0.07967847,  0.07956885,0.07946866,  0.07935828,  0.07921175,  0.07907922,0.07889339,  0.07851242,  0.07798815,  0.07735139,0.07669992,  0.07609985,  0.07550173,  0.07487512,0.0742502 ,  0.07360217,  0.07290657,  0.07218689,0.07140302,  0.07060384,  0.06979056,  0.06888127,0.06785737,  0.06669848,  0.06537975,  0.06389539,0.06221831,  0.06034092,  0.05828097,  0.05602698,0.0535707 ,  0.05089959,  0.04800349,  0.04486199,0.04145401,  0.03778645,  0.03383956,  0.02959964,0.02513171,  0.02041768,  0.01539678,  0.00996362,0.00408526, -0.00220905, -0.00887934, -0.01581486,-0.02315817, -0.03100882, -0.03959173, -0.04895527,-0.05899069])
        np.testing.assert_array_almost_equal(uf.mean.values[0], expected_mean)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf.covariance, DenseFunctionalData)

    def test_fit_normalize(self):
        uf = UFPCA(n_components=2, method='covariance', normalize=True)
        uf.fit(self.fdata_uni)

        np.testing.assert_almost_equal(self.fdata_uni.argvals['input_dim_0'], uf.mean.argvals['input_dim_0'])

        expected_mean = np.array([ 1.18436009,  1.14796703,  1.11239951,  1.07766273,   1.04376189,  1.0107022 ,  0.97848891,  0.94712725,   0.91662246,  0.88697975,  0.85820433,  0.8303014 ,   0.80327609,  0.77713356,  0.75187888,  0.72751691,   0.70405208,  0.68148811,  0.65982777,  0.63907263,   0.61922277,  0.60027657,  0.5822305 ,  0.56507886,   0.5488136 ,  0.53342416,  0.51889721,  0.50521652,   0.49236272,  0.48031306,  0.46904117,  0.45851674,   0.44870513,  0.43956692,  0.43105723,  0.42312499,   0.41571178,  0.40875044,  0.40216299,  0.39585782,   0.38991038,  0.38475681,  0.38035627,  0.37666532,   0.37363791,  0.37122531,  0.3693761 ,  0.36803619,   0.36714877,  0.36665428,  0.36649045,  0.36659226,   0.36689194,  0.367319  ,  0.3678002 ,  0.36825956,   0.36861843,  0.36879541,  0.36870645,  0.36826483,   0.36738122,  0.36596366,  0.36461114,  0.36348616,   0.36251537,  0.36163044,  0.36076635,  0.35986024,   0.35885051,  0.35767615,  0.3562763 ,  0.35458979,   0.35255495,  0.35010932,  0.34718952,  0.34373115,   0.33966868,  0.33493542,  0.32946354,  0.32318403,   0.31602683,  0.30792083,  0.29879407,  0.28857378,   0.27718662,  0.26455884,  0.25061649,  0.23528562,   0.21849249,  0.20016377,  0.18022663,  0.15860879,   0.13523847,  0.11004436,  0.08295559,  0.05390175,   0.02281286, -0.01038065, -0.04574793, -0.08335772,  -0.12327839])
        np.testing.assert_array_almost_equal(uf.mean.values[0], expected_mean)

        self.assertIsInstance(uf.eigenvalues, np.ndarray)
        self.assertIsInstance(uf.eigenfunctions, DenseFunctionalData)
        self.assertIsInstance(uf.covariance, DenseFunctionalData)

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


class TestFitCovariance(unittest.TestCase):
    def setUp(self) -> None:
        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=100)
        kl.sparsify(0.8, 0.05)

        self.fdata_uni = kl.data
        self.fdata_sparse = kl.sparse_data

    def test_fit_covariance_dense(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 31)})
        mean = self.fdata_uni.mean()

        ufpca = UFPCA(n_components=3)
        ufpca._fit_covariance(data=self.fdata_uni, points=points, mean=mean)

        expected_eigenvalues = np.array([0.249208  , 0.11510566, 0.05382122])
        np.testing.assert_array_almost_equal(ufpca.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions = np.array([[-2.03050399e-01, -1.28816058e-02,  1.58461253e-01,   3.12084925e-01,  4.48608916e-01,  5.67813128e-01,   6.71393644e-01,  7.61050939e-01,  8.38270483e-01,   9.04603116e-01,  9.61523904e-01,  1.01051016e+00,   1.05313115e+00,  1.09096996e+00,  1.12560006e+00,   1.15825499e+00,  1.18934515e+00,  1.21840081e+00,   1.24416558e+00,  1.26500126e+00,  1.27926904e+00,   1.28536490e+00,  1.28175573e+00,  1.26691694e+00,   1.23924699e+00,  1.19723682e+00,  1.13896279e+00,   1.06237725e+00,  9.69200082e-01,  8.60231831e-01,   7.35341051e-01], [-1.87465891e+00, -1.78530124e+00, -1.70093552e+00,  -1.61868936e+00, -1.53396349e+00, -1.44138716e+00,  -1.34249113e+00, -1.23838079e+00, -1.12938083e+00,  -1.01595434e+00, -8.98522110e-01, -7.77518446e-01,  -6.53351887e-01, -5.26421711e-01, -3.97158188e-01,  -2.66068751e-01, -1.33996886e-01, -2.09931192e-03,   1.28241691e-01,  2.55449294e-01,  3.77955334e-01,   4.94221661e-01,  6.02713920e-01,  7.01919576e-01,   7.90260404e-01,  8.66282936e-01,  9.27798870e-01,   9.72197049e-01,  1.00370793e+00,  1.02566344e+00,   1.03976094e+00], [ 1.75152846e+00,  1.48624714e+00,  1.23319139e+00,   9.90945002e-01,  7.57849945e-01,  5.33284961e-01,   3.20364665e-01,  1.21078648e-01, -6.34034473e-02,  -2.31694338e-01, -3.82397003e-01, -5.14149040e-01,  -6.25545611e-01, -7.15138244e-01, -7.81363353e-01,  -8.22434472e-01, -8.36519252e-01, -8.21777519e-01,  -7.76141442e-01, -6.97478613e-01, -5.83564682e-01,  -4.31959834e-01, -2.40317262e-01, -6.36913765e-03,   2.72342877e-01,  5.98631433e-01,  9.74050370e-01,   1.39697189e+00,  1.86981071e+00,  2.39976001e+00,   2.99455896e+00]])
        np.testing.assert_array_almost_equal(np.abs(ufpca.eigenfunctions.values), np.abs(expected_eigenfunctions))

        expected_covariance = np.array([ 0.52599863,  0.48580502,  0.44767562,  0.41090333,  0.37440838,  0.33703812,  0.2993508 ,  0.26172727,  0.22432305,  0.1873393 ,  0.1509695 ,  0.11540714,  0.08084351,  0.04747144,  0.01549939, -0.01482944, -0.04319645, -0.06921523, -0.09243229, -0.11234775, -0.12845617, -0.14024115, -0.14719473, -0.14881979, -0.14459023, -0.13397751, -0.11640148, -0.09144919, -0.05980236, -0.02157285,  0.02351   ])
        np.testing.assert_almost_equal(ufpca._covariance.values[0, 1], expected_covariance)

    def test_fit_covariance_irregular(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 31)})
        mean = self.fdata_sparse.mean()

        ufpca = UFPCA(n_components=3)
        ufpca._fit_covariance(data=self.fdata_sparse, points=points, mean=mean)

        expected_eigenvalues = np.array([0.25039568, 0.11548423, 0.05415895])
        np.testing.assert_array_almost_equal(ufpca.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions = np.array([[-0.22004529, -0.03219429,  0.13773589,  0.30001494,   0.46207611,  0.60961836,  0.73063454,  0.81031355,   0.85748449,  0.89850535,  0.94699771,  1.01080893,   1.06807786,  1.10111334,  1.12061952,  1.14621192,   1.18029694,  1.2156647 ,  1.24588578,  1.2699721 ,   1.28388024,  1.29036547,  1.28494763,  1.25210825,   1.20593077,  1.16052985,  1.10085329,  1.02966196,   0.95531412,  0.88651111,  0.85587579], [-1.98403232, -1.77704242, -1.66479685, -1.58372222,  -1.51019657, -1.43646396, -1.34599018, -1.23768187,  -1.10293329, -0.9648364 , -0.83998325, -0.73885463,  -0.65867079, -0.56506005, -0.44591305, -0.29288539,  -0.15283413, -0.02262808,  0.10788183,  0.23879689,   0.37646003,  0.50975427,  0.62639165,  0.72027683,   0.79483645,  0.86458321,  0.93778743,  0.99963232,   1.04445634,  1.07390297,  1.06067932], [ 1.73045808,  1.35721894,  1.1459679 ,  0.96243124,   0.78403516,  0.59366248,  0.36960302,  0.16265274,  -0.01558696, -0.18087274, -0.34024768, -0.47811362,  -0.57875294, -0.65694524, -0.70426965, -0.75096739,  -0.78398543, -0.79080569, -0.76423205, -0.70816079,  -0.64538609, -0.56468348, -0.41711984, -0.16919133,   0.15823433,  0.53777091,  0.95793989,  1.41576884,   1.911816  ,  2.49024978,  3.07602527]])
        np.testing.assert_array_almost_equal(np.abs(ufpca.eigenfunctions.values), np.abs(expected_eigenfunctions))

        expected_covariance = np.array([ 0.53613597,  0.46470795,  0.42477489,  0.39333758,  0.36382909,  0.33351483,  0.2975026 ,  0.25942122,  0.2182862 ,  0.1774658 ,  0.13973754,  0.10833557,  0.084021  ,  0.0587963 ,  0.03070905, -0.00433419, -0.03577734, -0.06328471, -0.08835832, -0.11129737, -0.13504648, -0.15652137, -0.16956731, -0.17034561, -0.16120691, -0.14725633, -0.13091351, -0.10937848, -0.08151572, -0.04448603,  0.00153244])
        np.testing.assert_almost_equal(ufpca._covariance.values[0, 1], expected_covariance)


class TestFitInnerProduct(unittest.TestCase):
    def setUp(self) -> None:
        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=100)
        kl.sparsify(0.8, 0.05)

        self.fdata_uni = kl.data
        self.fdata_sparse = kl.sparse_data

    def test_fit_inner_product_dense(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 31)})

        ufpca = UFPCA(n_components=3)
        ufpca._fit_inner_product(data=self.fdata_uni, points=points)

        expected_eigenvalues = np.array([0.24674571, 0.11342926, 0.05243551])
        np.testing.assert_array_almost_equal(ufpca.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions = np.array([[-0.00275609, -0.0975696 , -0.18789855, -0.2737856 ,  -0.35521487, -0.43212877, -0.50446215, -0.57218024,  -0.63528169, -0.69379329, -0.74776521, -0.79727242,  -0.84242709, -0.88341181, -0.92055454, -0.95449908,  -0.9827774 , -1.0079017 , -1.02920872, -1.04635211,  -1.05913174, -1.06741874, -1.07112207, -1.07017602,  -1.06453854, -1.05419574, -1.03916699, -1.01950178,  -0.99524303, -0.96639322, -0.93289674], [ 1.91721275,  1.81467618,  1.71027358,  1.60418202,   1.49660152,  1.38776656,  1.27793243,  1.16734867,   1.0562572 ,  0.94489765,  0.83351346,  0.72235621,   0.61168735,  0.50177547,  0.39288663,  0.28525946,   0.17859798,  0.07171819, -0.0349663 , -0.14110505,  -0.24638019, -0.3504928 , -0.45315651, -0.55409382,  -0.65303338, -0.74970711, -0.84384732, -0.93518617,  -1.02346553, -1.10843713, -1.18984447], [ 1.53371438,  1.34634882,  1.16923661,  1.00309854,   0.84877966,  0.70722037,  0.57933974,  0.46590638,   0.36753681,  0.28471566,  0.21780608,  0.1670326 ,   0.13242263,  0.11367354,  0.10987331,  0.11888097,   0.14788218,  0.18714164,  0.23992393,  0.30820319,   0.3932924 ,  0.49612949,  0.61741672,  0.75768842,   0.91734021,  1.09663424,  1.29569294,  1.51449848,   1.75295909,  2.01100205,  2.2886535 ]])
        np.testing.assert_array_almost_equal(np.abs(ufpca.eigenfunctions.values), np.abs(expected_eigenfunctions))

        expected_covariance = np.array([ 0.50297525,  0.47092466,  0.43910587,  0.40760721,  0.37652915,  0.34598495,  0.31609048,  0.28695051,  0.25865835,  0.23129817,  0.20494693,  0.17967406,  0.15553802,  0.13257713,  0.11078966,  0.09008909,  0.07086235,  0.05223894,  0.03451849,  0.01790417,  0.00254928, -0.01142154, -0.02390196, -0.03479871, -0.04402904, -0.05151958, -0.05720612, -0.0610336 , -0.06295434, -0.0629223 , -0.06088417])
        np.testing.assert_almost_equal(ufpca._covariance.values[0, 1], expected_covariance)

    def test_fit_inner_product_irregular(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 31)})

        ufpca = UFPCA(n_components=3)
        ufpca._fit_inner_product(data=self.fdata_sparse, points=points)

        expected_eigenvalues = np.array([0.24652452, 0.11380499, 0.05214754])
        np.testing.assert_array_almost_equal(ufpca.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions = np.array([[-0.00763577, -0.10131652, -0.1904903 , -0.27524873,  -0.35560874, -0.43153016, -0.5029752 , -0.56992937,  -0.6324013 , -0.69041988, -0.74403092, -0.79329878,  -0.83832193, -0.87927072, -0.91645886, -0.95050648,  -0.9789815 , -1.004372  , -1.02603908, -1.04365034,  -1.05701143, -1.06599537, -1.07051141, -1.07049121,  -1.06588327, -1.05665395, -1.04279261, -1.02431416,  -1.00124428, -0.97356749, -0.9412121 ], [ 1.94482633,  1.83925955,  1.73201797,  1.62331514,   1.51337843,  1.40245992,  1.29080343,  1.17863541,   1.06617172,  0.95362491,  0.84121347,  0.72916818,   0.61773011,  0.50713961,  0.39763064,  0.28941319,   0.18228597,  0.07506575, -0.03182718, -0.13805268,  -0.24330725, -0.34730757, -0.44978734, -0.5504939 ,  -0.64918299, -0.74561377, -0.83954636, -0.93074101,  -1.01895597, -1.10393989, -1.18541014], [ 1.52327203,  1.33452929,  1.15726065,  0.99198154,   0.83936171,  0.70018476,  0.5751579 ,  0.46485976,   0.36974946,  0.29017957,  0.22640326,  0.17856005,   0.14660915,  0.13017823,  0.12828568,  0.1387289 ,   0.16885198,  0.20932826,  0.26324167,  0.33239959,   0.41796094,  0.52072581,  0.64126967,  0.78001277,   0.93726081,  1.11322211,  1.30801337,  1.52164675,   1.75403753,  2.00510999,  2.27489347]])
        np.testing.assert_array_almost_equal(np.abs(ufpca.eigenfunctions.values), np.abs(expected_eigenfunctions))

        expected_covariance = np.array([ 0.51328397,  0.48039185,  0.44783512,  0.41569661,  0.38407097,  0.35306448,  0.32277644,  0.29329418,  0.26469503,  0.23704875,  0.21041983,  0.18486789,  0.16044302,  0.13717387,  0.11504893,  0.09397436,  0.07435838,  0.0553664 ,  0.03728504,  0.02030299,  0.00455957, -0.00983341, -0.02278242, -0.03420704, -0.04403616, -0.0522057 , -0.0586576 , -0.06334046, -0.06620892, -0.06721604, -0.06630239])
        np.testing.assert_almost_equal(ufpca._covariance.values[0, 1], expected_covariance)

    def test_fit_inner_product_2d(self):
        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, dimension='2D', random_state=42
        )
        kl.new(n_obs=10)
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 31), 'input_dim_1': np.linspace(0, 1, 31)})
        
        ufpca = UFPCA(n_components=3)
        with self.assertWarns(UserWarning):
            ufpca._fit_inner_product(data=kl.data, points=points)
        
        expected_eigenvalues = np.array([0.01098398, 0.00483969, 0.00231953])
        np.testing.assert_array_almost_equal(ufpca.eigenvalues, expected_eigenvalues)

        expected_eigenfunctions = np.array([[ 0.42525949,  0.8563933 ,  1.24209706,  1.58137448,   1.8738059 ,  2.12014364,  2.3237942 ,  2.49401279,   2.62500056,  2.69999727,  2.72337215,  2.69998249,   2.63526753,  2.53537672,  2.40678279,  2.25565403,   2.087718  ,  1.90824747,  1.72207012,  1.53356877,   1.34659027,  1.16442531,  0.98989878,  0.82543794,   0.66952929,  0.52182158,  0.38507543,  0.26071132,   0.14891143,  0.04996946, -0.03670302], [ 1.8875952 ,  2.50134585,  3.00941326,  3.41707957,   3.72477528,  3.93929528,  4.06828618,  4.13129903,   4.11275127,  3.97525065,  3.7385874 ,  3.42375935,   3.05294705,  2.64932483,  2.23525304,  1.82987486,   1.44853705,  1.10271916,  0.80009723,  0.54476472,   0.33771591,  0.17741514,  0.06028504, -0.01878353,  -0.06337308, -0.07670991, -0.06358558, -0.03031326,   0.01627209,  0.07084066,  0.12830395], [ 0.22279502,  0.51794216,  0.70035575,  0.76856927,   0.73006728,  0.59267277,  0.37588295,  0.11057594,  -0.20200597, -0.57195243, -0.9749799 , -1.38715099,  -1.78566352, -2.14930357, -2.45965356, -2.70254902,  -2.86850934, -2.95282805, -2.95549127, -2.88089493,  -2.73715962, -2.53525531, -2.28813478, -2.00983483,  -1.71580958, -1.41357273, -1.11156381, -0.81803242,  -0.53789908, -0.27650938, -0.03559819]])
        np.testing.assert_array_almost_equal(np.abs(ufpca.eigenfunctions.values[:, 0]), np.abs(expected_eigenfunctions))



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


# class TestNumericalIntegration(unittest.TestCase):
#     def setUp(self):
#         warnings.simplefilter('ignore', category=UserWarning)

#         argvals = np.linspace(0, 1, 10)
#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals, 
#             n_functions=5, random_state=42
#         )
#         kl.new(n_obs=50)
#         self.data = kl.data

#     def test_numerical_integration(self):
#         self.data.covariance()
        
#         uf = UFPCA(n_components=2, method='covariance')
#         uf.fit(self.data)

#         scores = uf._numerical_integration(self.data)

#         expected_scores = np.array([
#             [ 1.42086765,  2.00210923],
#             [ 0.64501025,  0.34982201],
#             [ 0.11092837, -0.31895034],
#             [ 0.49936318, -0.12704974],
#             [ 0.78610985,  0.27121312]
#         ])
#         np.testing.assert_array_almost_equal(
#             np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#         )

#     def test_numerical_integration_2d(self):
#         argvals = np.linspace(0, 1, 10)
#         kl = KarhunenLoeve(
#             basis_name='fourier', argvals=argvals, n_functions=5,
#             dimension='2D', random_state=42
#         )
#         kl.new(n_obs=50)
#         data = kl.data

#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(data)

#         scores = uf._numerical_integration(data)

#         expected_scores = np.array([
#             [-1.42086765, -2.00210923],
#             [-0.64501025, -0.34982201],
#             [-0.11092837,  0.31895034],
#             [-0.49936318,  0.12704974],
#             [-0.78610985, -0.27121312]
#         ])
#         np.testing.assert_array_almost_equal(
#             np.abs(scores[:5, :]), np.abs(expected_scores), decimal=4
#         )

#     def test_value_error(self):
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
#         data = DenseFunctionalData(DenseArgvals(argvals), DenseValues(values))

#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data)

#         with self.assertRaises(ValueError):
#             uf._numerical_integration(data)


class TestTransform(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        argvals = np.linspace(0, 1, 10)
        kl = KarhunenLoeve(
            basis_name='fourier', argvals=argvals, 
            n_functions=5, random_state=42
        )
        kl.new(n_obs=50)
        self.data = kl.data

    def test_error_innpro(self):
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.data)

        with self.assertRaises(ValueError):
            uf.transform(self.data, method='InnPro')

    def test_error_unkown_method(self):
        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.data)

        with self.assertRaises(ValueError):
            uf.transform(self.data, method='error')

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

#     def test_innpro(self):
#         uf = UFPCA(n_components=2, method='inner-product')
#         uf.fit(self.data)

#         scores = uf.transform(self.data, method='InnPro')
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
