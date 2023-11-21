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
    UFPCA,
    _fit_covariance,
    _fit_inner_product,
    _transform_numerical_integration_dense,
    _transform_numerical_integration_irregular,
    _transform_pace_dense,
    _transform_pace_irregular,
    _transform_innpro
)


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
        results = _fit_covariance(data=self.fdata_uni, points=points, n_components=3)

        expected_eigenvalues = np.array([0.249208  , 0.11510566, 0.05382122])
        np.testing.assert_array_almost_equal(results['eigenvalues'], expected_eigenvalues)

        expected_eigenfunctions = np.array([[-2.03050399e-01, -1.28816058e-02,  1.58461253e-01,   3.12084925e-01,  4.48608916e-01,  5.67813128e-01,   6.71393644e-01,  7.61050939e-01,  8.38270483e-01,   9.04603116e-01,  9.61523904e-01,  1.01051016e+00,   1.05313115e+00,  1.09096996e+00,  1.12560006e+00,   1.15825499e+00,  1.18934515e+00,  1.21840081e+00,   1.24416558e+00,  1.26500126e+00,  1.27926904e+00,   1.28536490e+00,  1.28175573e+00,  1.26691694e+00,   1.23924699e+00,  1.19723682e+00,  1.13896279e+00,   1.06237725e+00,  9.69200082e-01,  8.60231831e-01,   7.35341051e-01], [-1.87465891e+00, -1.78530124e+00, -1.70093552e+00,  -1.61868936e+00, -1.53396349e+00, -1.44138716e+00,  -1.34249113e+00, -1.23838079e+00, -1.12938083e+00,  -1.01595434e+00, -8.98522110e-01, -7.77518446e-01,  -6.53351887e-01, -5.26421711e-01, -3.97158188e-01,  -2.66068751e-01, -1.33996886e-01, -2.09931192e-03,   1.28241691e-01,  2.55449294e-01,  3.77955334e-01,   4.94221661e-01,  6.02713920e-01,  7.01919576e-01,   7.90260404e-01,  8.66282936e-01,  9.27798870e-01,   9.72197049e-01,  1.00370793e+00,  1.02566344e+00,   1.03976094e+00], [ 1.75152846e+00,  1.48624714e+00,  1.23319139e+00,   9.90945002e-01,  7.57849945e-01,  5.33284961e-01,   3.20364665e-01,  1.21078648e-01, -6.34034473e-02,  -2.31694338e-01, -3.82397003e-01, -5.14149040e-01,  -6.25545611e-01, -7.15138244e-01, -7.81363353e-01,  -8.22434472e-01, -8.36519252e-01, -8.21777519e-01,  -7.76141442e-01, -6.97478613e-01, -5.83564682e-01,  -4.31959834e-01, -2.40317262e-01, -6.36913765e-03,   2.72342877e-01,  5.98631433e-01,  9.74050370e-01,   1.39697189e+00,  1.86981071e+00,  2.39976001e+00,   2.99455896e+00]])
        np.testing.assert_array_almost_equal(np.abs(results['eigenfunctions'].values), np.abs(expected_eigenfunctions))

        expected_noise = 0.014204431460944762
        np.testing.assert_almost_equal(results['noise_variance_cov'], expected_noise)

    def test_fit_covariance_irregular(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 31)})
        results = _fit_covariance(data=self.fdata_sparse, points=points, n_components=3)

        expected_eigenvalues = np.array([0.25039568, 0.11548423, 0.05415895])
        np.testing.assert_array_almost_equal(results['eigenvalues'], expected_eigenvalues)

        expected_eigenfunctions = np.array([[-0.22004529, -0.03219429,  0.13773589,  0.30001494,   0.46207611,  0.60961836,  0.73063454,  0.81031355,   0.85748449,  0.89850535,  0.94699771,  1.01080893,   1.06807786,  1.10111334,  1.12061952,  1.14621192,   1.18029694,  1.2156647 ,  1.24588578,  1.2699721 ,   1.28388024,  1.29036547,  1.28494763,  1.25210825,   1.20593077,  1.16052985,  1.10085329,  1.02966196,   0.95531412,  0.88651111,  0.85587579], [-1.98403232, -1.77704242, -1.66479685, -1.58372222,  -1.51019657, -1.43646396, -1.34599018, -1.23768187,  -1.10293329, -0.9648364 , -0.83998325, -0.73885463,  -0.65867079, -0.56506005, -0.44591305, -0.29288539,  -0.15283413, -0.02262808,  0.10788183,  0.23879689,   0.37646003,  0.50975427,  0.62639165,  0.72027683,   0.79483645,  0.86458321,  0.93778743,  0.99963232,   1.04445634,  1.07390297,  1.06067932], [ 1.73045808,  1.35721894,  1.1459679 ,  0.96243124,   0.78403516,  0.59366248,  0.36960302,  0.16265274,  -0.01558696, -0.18087274, -0.34024768, -0.47811362,  -0.57875294, -0.65694524, -0.70426965, -0.75096739,  -0.78398543, -0.79080569, -0.76423205, -0.70816079,  -0.64538609, -0.56468348, -0.41711984, -0.16919133,   0.15823433,  0.53777091,  0.95793989,  1.41576884,   1.911816  ,  2.49024978,  3.07602527]])
        np.testing.assert_array_almost_equal(np.abs(results['eigenfunctions'].values), np.abs(expected_eigenfunctions))

        expected_noise = 0.013971444244734173
        np.testing.assert_almost_equal(results['noise_variance_cov'], expected_noise)


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
        results = _fit_inner_product(data=self.fdata_uni, points=points, n_components=3)

        expected_eigenvalues = np.array([0.24673014, 0.11341369, 0.05241994])
        np.testing.assert_array_almost_equal(results['eigenvalues'], expected_eigenvalues)

        expected_eigenfunctions = np.array([[ 0.07904403, -0.0522243 , -0.17323811, -0.28449633,  -0.38683601, -0.48103603, -0.56782871, -0.64797133,  -0.72209437, -0.79084309, -0.85477101, -0.91443988,  -0.97072477, -1.02068763, -1.06308277, -1.0981388 ,  -1.12598681, -1.14643027, -1.15918243, -1.16548869,  -1.16836975, -1.16689301, -1.16044931, -1.14844347,  -1.13019022, -1.10502436, -1.07215333, -1.03081311,  -0.98019199, -0.9194178 , -0.84797022], [ 1.88221147,  1.7958967 ,  1.70528191,  1.61049742,   1.51182059,  1.40950285,  1.30378747,  1.19498129,   1.0834208 ,  0.96955626,  0.85390742,  0.73711456,   0.61997967,  0.50177327,  0.38270144,  0.26364263,   0.14564359,  0.02984988, -0.08266956, -0.19211056,  -0.30051224, -0.40675023, -0.50987583, -0.60901362,  -0.70327257, -0.79182691, -0.87380164, -0.94837712,  -1.01470598, -1.0718885 , -1.11933824], [ 1.6787004 ,  1.44522845,  1.2211007 ,  1.0074812 ,   0.80511713,  0.61484592,  0.4376114 ,  0.27422139,   0.12562026, -0.00760725, -0.12495849, -0.22622666,  -0.31282335, -0.37906112, -0.42034759, -0.43506199,  -0.42153695, -0.37766269, -0.30170482, -0.19823803,  -0.07771983,  0.06458031,  0.23119328,  0.42402016,   0.64478711,  0.89455225,  1.17454398,  1.48564976,   1.82892525,  2.20565128,  2.61589876]])
        np.testing.assert_array_almost_equal(np.abs(results['eigenfunctions'].values), np.abs(expected_eigenfunctions))

        expected_eigenvectors = np.array([[-0.008733  , -0.0030051 , -0.1741724 ],[ 0.04219662, -0.04267833, -0.12641441],[-0.12108065,  0.06025756,  0.0943915 ],[-0.0088194 , -0.06732271, -0.00186014],[-0.01954053, -0.01493807, -0.09659307],[-0.07602117, -0.00119646,  0.00662709],[ 0.06174976,  0.09383498,  0.1993483 ],[ 0.0743865 ,  0.04751714,  0.15987125],[-0.00844423,  0.06093655,  0.08744061],[-0.09233449,  0.0587303 ,  0.0147853 ],[ 0.08214717,  0.0623118 ,  0.0459721 ],[-0.04111606, -0.03299329, -0.01772025],[-0.03840731, -0.16234148, -0.03435818],[-0.01319255, -0.0057966 ,  0.05709688],[ 0.07032875, -0.19492882, -0.01097621],[-0.02659881,  0.00597957,  0.01271511],[ 0.09544327,  0.0269896 , -0.05507715],[-0.02612558, -0.03358465,  0.07178676],[ 0.06910697,  0.08655483, -0.08191197],[ 0.11580859, -0.18118686, -0.10648156],[-0.05191057,  0.07555372, -0.07798331],[ 0.05107248, -0.05283492, -0.13513753],[ 0.10289094,  0.0760332 ,  0.05857227],[-0.09748488, -0.05444236, -0.04148754],[-0.11580292, -0.11324362,  0.01259335],[ 0.06390819,  0.15776016, -0.0024374 ],[ 0.07271434, -0.0510693 , -0.17024385],[-0.245899  , -0.18258888,  0.16383405],[ 0.15774371, -0.06771167,  0.05670714],[-0.03336886, -0.10899231,  0.0049511 ],[ 0.16382707, -0.09865221,  0.0664193 ],[-0.01245946,  0.19222919, -0.04165432],[-0.05484743, -0.06948657, -0.07528548],[-0.04378362,  0.0295348 , -0.06717049],[ 0.1152519 ,  0.06876932, -0.12340069],[-0.07533277,  0.12357182,  0.03105574],[-0.07291166,  0.03772542,  0.13680281],[ 0.08763994,  0.06825273,  0.28354455],[-0.04314766,  0.07600723,  0.02954068],[ 0.00472009, -0.00675604, -0.04149717],[-0.0516362 ,  0.17621988, -0.15315217],[ 0.16069783, -0.04721407,  0.0270464 ],[ 0.08171064, -0.01289092,  0.02699787],[-0.0762568 ,  0.15560226, -0.05832435],[ 0.04281682,  0.0404213 , -0.01060694],[-0.14968107,  0.27157739, -0.09438429],[-0.03812314, -0.13460153, -0.14962479],[ 0.17543729, -0.14449541, -0.07102004],[ 0.13786597, -0.11923821,  0.21232715],[-0.19201353, -0.16896278, -0.24088401],[-0.04267361,  0.07809812, -0.11694755],[ 0.03947559,  0.12374036, -0.05156695],[ 0.01933876, -0.02599894,  0.02024273],[-0.03676911,  0.07850294,  0.04660694],[-0.16831218, -0.14008614,  0.10865772],[ 0.01062076, -0.0990011 ,  0.103197  ],[ 0.01643114, -0.05393901, -0.10169906],[ 0.00827094,  0.00661658, -0.03329161],[ 0.07196644,  0.11553371, -0.10237994],[ 0.10200702,  0.07022124,  0.05494097],[ 0.02298408,  0.02264207,  0.05837924],[-0.13074488, -0.13757273,  0.03680965],[-0.16969617,  0.09560838,  0.06460298],[ 0.00739114, -0.01519597, -0.07752542],[-0.15391309, -0.02887034,  0.15381423],[ 0.14394465, -0.01547485,  0.00297279],[ 0.06062948,  0.15479084,  0.12153651],[-0.06729462, -0.17039177, -0.02180165],[-0.06801167, -0.17974878,  0.01169363],[ 0.06036134, -0.15345737, -0.01581285],[-0.0821605 ,  0.01142586,  0.14453963],[ 0.02788152,  0.07011333, -0.00876139],[-0.00459663,  0.07771313, -0.0619854 ],[ 0.03061428, -0.01575465,  0.1985644 ],[-0.16626106,  0.07016121,  0.01624366],[ 0.01781897,  0.05128193,  0.19213513],[ 0.14797004,  0.16614177, -0.05687796],[-0.02310504,  0.03680472, -0.18263189],[-0.00740353, -0.17959672, -0.04244708],[-0.05914212,  0.07729665, -0.05607531],[-0.04294082,  0.01587882,  0.0014898 ],[ 0.11105764,  0.10102984, -0.16272543],[ 0.10290305, -0.28270517, -0.06850217],[-0.1498238 ,  0.07413551,  0.07118855],[ 0.17512663,  0.01189553, -0.04194826],[-0.05372605,  0.1024595 , -0.06303869],[ 0.07879285, -0.00282842,  0.02639226],[ 0.11103174, -0.0882354 ,  0.05433985],[-0.10295865, -0.00816072, -0.04727131],[-0.09245344, -0.06339003, -0.02768822],[-0.24676993, -0.00992003, -0.07835909],[-0.01863479,  0.06313668,  0.05060027],[ 0.26912758,  0.0322884 , -0.00582812],[-0.13396673, -0.01017556, -0.18631445],[ 0.00173354, -0.03987268,  0.09115262],[-0.0457993 , -0.05354499,  0.25821317],[-0.09885875,  0.10451185,  0.00982951],[ 0.04144853, -0.0526841 ,  0.03110231],[-0.02675141,  0.0836424 ,  0.0541262 ],[ 0.24128846,  0.03199564, -0.05867297]])
        np.testing.assert_array_almost_equal(np.abs(results['eigenvectors']), np.abs(expected_eigenvectors))

    def test_fit_inner_product_sparse(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 31)})
        results = _fit_inner_product(data=self.fdata_sparse, points=points, n_components=3)

        expected_eigenvalues = np.array([0.24649833, 0.1137788 , 0.05212135])
        np.testing.assert_array_almost_equal(results['eigenvalues'], expected_eigenvalues)

        expected_eigenfunctions = np.array([[ 0.07533891, -0.05619051, -0.17644861, -0.28672377,  -0.38758051, -0.48029271, -0.5659127 , -0.64524327,  -0.7186441 , -0.78643419, -0.84901944, -0.9069788 ,  -0.96088738, -1.01082382, -1.05337508, -1.08895896,  -1.11781581, -1.13876994, -1.15183038, -1.16066492,  -1.16543837, -1.1653951 , -1.16008703, -1.14932339,  -1.13206235, -1.10750077, -1.07523862, -1.03519957,  -0.98614602, -0.92654413, -0.85633575], [ 1.91118095,  1.8235003 ,  1.72915817,  1.62998416,   1.52759431,  1.42239701,  1.31398813,  1.20246727,   1.08844195,  0.97256878,  0.85556665,  0.73767466,   0.6191739 ,  0.50033081,  0.38139968,  0.26292983,   0.14694774,  0.03271004, -0.0781068 , -0.18791104,  -0.29654861, -0.40281551, -0.50588768, -0.60524179,  -0.69937535, -0.78735861, -0.86883606, -0.94417649,  -1.01153601, -1.06800463, -1.1143639 ], [ 1.68635585,  1.44472994,  1.21433748,  0.99464347,   0.78796845,  0.59615319,  0.41900592,  0.25660406,   0.11073031, -0.01869151, -0.12962772, -0.22143279,  -0.29549009, -0.35317331, -0.38798106, -0.3977845 ,  -0.37985565, -0.33081366, -0.25067491, -0.15191422,  -0.03367279,  0.10664281,  0.27092455,  0.45975293,   0.675847  ,  0.9212359 ,  1.19599221,  1.49824614,   1.82945843,  2.19039061,  2.58269295]])
        np.testing.assert_array_almost_equal(np.abs(results['eigenfunctions'].values), np.abs(expected_eigenfunctions))

        expected_eigenvectors = np.array([[-0.00910154, -0.0038095 , -0.16806664],[ 0.04188371, -0.04489693, -0.127812  ],[-0.12093194,  0.06142492,  0.09580927],[-0.00880084, -0.0679463 , -0.00161218],[-0.01965687, -0.01520458, -0.09570006],[-0.0759594 , -0.00137035,  0.00544256],[ 0.0620322 ,  0.09656572,  0.20116714],[ 0.0744255 ,  0.04925143,  0.16147601],[-0.00824469,  0.06171693,  0.08811699],[-0.09233473,  0.05875602,  0.01419238],[ 0.08222939,  0.06231011,  0.04525034],[-0.04117416, -0.03303263, -0.02050286],[-0.03840118, -0.16323449, -0.03565668],[-0.0132027 , -0.00423823,  0.06191324],[ 0.07023947, -0.19500853, -0.00876886],[-0.0266523 ,  0.00667516,  0.01653176],[ 0.09519145,  0.02618695, -0.05323836],[-0.02601748, -0.03276029,  0.07376482],[ 0.06896085,  0.08596398, -0.07927654],[ 0.11572843, -0.18197602, -0.10398798],[-0.05214294,  0.07476651, -0.07743923],[ 0.05099614, -0.0534987 , -0.13085668],[ 0.10315477,  0.07598124,  0.05542574],[-0.09755299, -0.05472997, -0.03823515],[-0.11588619, -0.1135988 ,  0.01272473],[ 0.06413411,  0.15707466, -0.005557  ],[ 0.07219447, -0.05237664, -0.16669149],[-0.24561044, -0.18084733,  0.16437534],[ 0.15796153, -0.06754167,  0.06138227],[-0.03329528, -0.1083646 ,  0.00723027],[ 0.16399742, -0.09807323,  0.06819644],[-0.01254217,  0.19244956, -0.04015588],[-0.05501956, -0.06983538, -0.07926593],[-0.04390541,  0.0284476 , -0.07066709],[ 0.1150203 ,  0.06673271, -0.12735365],[-0.07536898,  0.1232789 ,  0.02661248],[-0.07265834,  0.03947412,  0.14038681],[ 0.08862453,  0.07116714,  0.27936892],[-0.0430622 ,  0.07663683,  0.03598326],[ 0.00459786, -0.00798013, -0.04413678],[-0.05184541,  0.17445591, -0.15150624],[ 0.16073187, -0.04746694,  0.0279024 ],[ 0.08189295, -0.012636  ,  0.0298262 ],[-0.07632752,  0.15523769, -0.05847339],[ 0.04280534,  0.04017421, -0.01424986],[-0.14947517,  0.2707431 , -0.10094521],[-0.03853431, -0.13664771, -0.1461977 ],[ 0.17508176, -0.14410874, -0.06870476],[ 0.13818627, -0.11826561,  0.20827457],[-0.19276672, -0.17082213, -0.23286756],[-0.04297143,  0.07677438, -0.12113148],[ 0.03923105,  0.12331554, -0.05033617],[ 0.01917497, -0.02616612,  0.01980409],[-0.03676262,  0.07720601,  0.04476263],[-0.16821277, -0.13889844,  0.11063537],[ 0.01053336, -0.0977658 ,  0.10816885],[ 0.0166158 , -0.05448327, -0.09676366],[ 0.00827632,  0.00610776, -0.03503116],[ 0.07187156,  0.1143625 , -0.11155926],[ 0.1027196 ,  0.07136934,  0.04517632],[ 0.02304559,  0.02440453,  0.06339945],[-0.13026987, -0.13707402,  0.03528901],[-0.16967526,  0.09666435,  0.06529201],[ 0.00713871, -0.0160347 , -0.0747457 ],[-0.153492  , -0.02679045,  0.15405003],[ 0.14380134, -0.01432254,  0.0060298 ],[ 0.06080167,  0.15586558,  0.12016316],[-0.06738288, -0.17050316, -0.01980021],[-0.06808093, -0.18065212,  0.00945695],[ 0.0606755 , -0.1535798 , -0.02327429],[-0.08190185,  0.01258368,  0.14448977],[ 0.02796407,  0.06997873, -0.01131863],[-0.00435295,  0.0771695 , -0.06572173],[ 0.03078261, -0.01241169,  0.20427098],[-0.16616611,  0.0699705 ,  0.0143266 ],[ 0.01797659,  0.05513533,  0.20106261],[ 0.14751536,  0.16521037, -0.06045591],[-0.02347156,  0.0349634 , -0.18306873],[-0.00761143, -0.1804825 , -0.04505696],[-0.05923603,  0.07660733, -0.05485609],[-0.04274248,  0.01576211, -0.00113258],[ 0.11078537,  0.09919916, -0.16727984],[ 0.10242227, -0.28304468, -0.06361624],[-0.14949208,  0.07509735,  0.06701257],[ 0.17509616,  0.01160184, -0.04250359],[-0.05405429,  0.10099983, -0.06870384],[ 0.07931972, -0.00254788,  0.01875676],[ 0.11100283, -0.08818269,  0.05397712],[-0.10280561, -0.0079465 , -0.04624373],[-0.0926423 , -0.06419747, -0.02980353],[-0.24714032, -0.01025713, -0.07591281],[-0.01863226,  0.06316445,  0.04936387],[ 0.26907242,  0.03152324, -0.00699652],[-0.13399902, -0.01154447, -0.18431173],[ 0.0018933 , -0.03892686,  0.09295881],[-0.04559156, -0.05206798,  0.25439481],[-0.09885444,  0.10459513,  0.0084989 ],[ 0.04147133, -0.05236415,  0.02986451],[-0.02679958,  0.08462287,  0.05660264],[ 0.24121921,  0.03192949, -0.05681752]])
        np.testing.assert_array_almost_equal(np.abs(results['eigenvectors']), np.abs(expected_eigenvectors))

    def test_fit_inner_product_2d(self):
        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, dimension='2D', random_state=42
        )
        kl.new(n_obs=10)
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 31), 'input_dim_1': np.linspace(0, 1, 31)})

        results = _fit_inner_product(data=kl.data, points=points, n_components=3, noise_variance=0)
        
        expected_eigenvalues = np.array([0.01098398, 0.00483969, 0.00231953])
        np.testing.assert_array_almost_equal(results['eigenvalues'], expected_eigenvalues)

        expected_eigenfunctions = np.array([[ 0.20682534,  0.80205269,  1.32401771,  1.77736403,   2.16785037,  2.50693936,  2.76414249,  2.93894447,   3.03937124,  3.0736448 ,  3.0505235 ,  2.97813705,   2.86472845,  2.71859179,  2.54712263,  2.35709488,   2.15463232,  1.94488013,  1.73225279,  1.52044945,   1.31264028,  1.11172977,  0.92052612,  0.74195516,   0.57886066,  0.43401157,  0.30851769,  0.19731575,   0.10419423,  0.03083209, -0.02241254], [ 0.99597781,  2.18332111,  3.16457109,  3.96994193,   4.61912391,  5.13707662,  5.41772855,  5.4667867 ,   5.32036842,  5.01437707,  4.58633489,  4.0727424 ,   3.50958386,  2.9311167 ,  2.36726977,  1.84200507,   1.37238423,  0.96949738,  0.63829619,  0.37665047,   0.17894333,  0.03784088, -0.05560035, -0.10961338,  -0.13197779, -0.13140834, -0.11419909, -0.07910182,  -0.03487937,  0.00928537,  0.0454779 ], [ 0.04220196,  0.60013476,  0.96691291,  1.14514616,   1.15670488,  1.04742964,  0.78734604,  0.39588172,  -0.09424799, -0.64990737, -1.23704651, -1.82229057,  -2.37314786, -2.85904495, -3.25446343, -3.5418506 ,  -3.71091555, -3.75847199, -3.68962564, -3.51672169,  -3.25661037, -2.92922635, -2.5551155 , -2.15443417,  -1.74754097, -1.35468918, -0.99310414, -0.66338123,  -0.3778083 , -0.14374666,  0.03540871]])
        np.testing.assert_array_almost_equal(np.abs(results['eigenfunctions'].values[:, 0]), np.abs(expected_eigenfunctions))

        expected_eigenvectors = np.array([[-0.16023502, -0.25916387,  0.69352094],[-0.59327901, -0.18169182, -0.15757975],[ 0.58892225, -0.14372699, -0.10404258],[-0.28024854, -0.28946203, -0.44447702],[-0.01298703, -0.09790341,  0.42423096],[ 0.21479235, -0.19854235, -0.15823092],[ 0.05588307,  0.66067552,  0.14747419],[-0.18637344,  0.51507827, -0.13311371],[ 0.03540505,  0.14717772, -0.1778395 ],[ 0.33400626, -0.15137129, -0.09853385]])
        np.testing.assert_array_almost_equal(np.abs(results['eigenvectors']), np.abs(expected_eigenvectors))


class TestTransformNumericalIntegrationDense(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=10)
        kl.add_noise(0.05)
        self.fdata_uni = kl.noisy_data

        uf = UFPCA(n_components=2, method='covariance')
        uf.fit(self.fdata_uni)
        self.uf_eigen = uf.eigenfunctions

    def test_numerical_integration(self):
        scores_dense = _transform_numerical_integration_dense(self.fdata_uni, self.uf_eigen)
        expected_scores = np.array([[ 0.23185638,  0.27092884],[ 0.02684859,  0.41977244],[ 0.35129452, -0.57867029],[ 0.04697217,  0.06655563],[ 0.20541785,  0.12994119],[ 0.2936377 , -0.18557023],[-0.59218129, -0.25986862],[-0.55898506, -0.14784151],[-0.09181077, -0.19276825],[ 0.323464  , -0.3623928 ]])
        np.testing.assert_array_almost_equal(np.abs(scores_dense), np.abs(expected_scores))

    def test_numerical_integration_2d(self):
        kl = KarhunenLoeve(basis_name='bsplines', n_functions=5, argvals=np.linspace(0, 1, 10), dimension='2D', random_state=42)
        kl.new(n_obs=10)
        fdata = kl.data

        uf = UFPCA(n_components=2, method='inner-product')
        uf.fit(fdata)

        scores = _transform_numerical_integration_dense(fdata, uf.eigenfunctions)
        expected_scores = np.array([[-0.02274272, -0.01767752],[-0.14201805, -0.03527876],[ 0.20025465,  0.01227318],[-0.0490446 , -0.04197494],[ 0.00975116, -0.01904456],[ 0.08457573, -0.02113259],[ 0.04962573,  0.0915368 ],[-0.01979976,  0.0609952 ],[ 0.04741364,  0.03495292],[ 0.12644882,  0.00416015]])

        np.testing.assert_array_almost_equal(np.abs(scores), np.abs(expected_scores))


class TestTransformNumericalIntegrationIrregular(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=10)
        kl.add_noise(0.05)
        kl.sparsify(0.95, 0.01)

        self.fdata_sparse = kl.sparse_data

        uf = UFPCA(n_components=0.95, method='covariance')
        uf.fit(self.fdata_sparse)
        self.uf_eigen = uf.eigenfunctions

    def test_numerical_integration(self):
        scores_sparse = _transform_numerical_integration_irregular(self.fdata_sparse, self.uf_eigen)
        expected_scores = np.array([[ 0.22203175,  0.17646635,  0.20582599],[ 0.08804525,  0.411517  , -0.0441252 ],[ 0.13957335, -0.67481938,  0.03599057],[ 0.08157136,  0.08586396, -0.18263835],[ 0.17731011,  0.07096538,  0.00388039],[ 0.19391082, -0.27504909, -0.09204208],[-0.4688101 , -0.15005271,  0.24150274],[-0.39397066,  0.05064411,  0.08163727],[-0.13128561, -0.22303583,  0.11581405],[ 0.16903147, -0.45101713,  0.05664777]])
        np.testing.assert_array_almost_equal(np.abs(scores_sparse), np.abs(expected_scores))


class TestTransformPACE(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=10)
        kl.add_noise_and_sparsify(0.05, 0.9, 0.05)
        self.fdata_uni = kl.noisy_data
        self.fdata_sparse = kl.sparse_data

        uf_dense = UFPCA(n_components=2, method='covariance')
        uf_dense.fit(self.fdata_uni)
        self.uf_dense = uf_dense

        uf_sparse = UFPCA(n_components=2, method='covariance')
        uf_sparse.fit(self.fdata_sparse)
        self.uf_sparse = uf_sparse

    def test_pace_dense(self):
        scores_dense = scores_dense = _transform_pace_dense(
            self.fdata_uni, self.uf_dense.eigenfunctions, self.uf_dense.eigenvalues,
            self.uf_dense.covariance, self.uf_dense._noise_variance
        )
        expected_scores = np.array([[ 0.2250209 ,  0.26640435], [ 0.03071671,  0.42263802], [ 0.34750068, -0.57693324], [ 0.04796388,  0.06556203], [ 0.20086595,  0.12434758], [ 0.29359884, -0.18454036], [-0.5901051 , -0.25663729], [-0.55689136, -0.14721497], [-0.08779404, -0.18558396], [ 0.32294073, -0.35842484]])
        np.testing.assert_array_almost_equal(np.abs(scores_dense), np.abs(expected_scores))

    def test_pace_irregular(self):
        scores_sparse = _transform_pace_irregular(
            self.fdata_sparse, self.uf_sparse.eigenfunctions,
            self.uf_sparse.eigenvalues, self.uf_sparse.covariance,
            self.uf_sparse._noise_variance
        )
        expected_scores = np.array([[ 0.21468274,  0.17152223],[ 0.08052812,  0.42301138],[ 0.29872919, -0.62039464],[ 0.05720657,  0.0820881 ],[ 0.22212744,  0.12684133],[ 0.26609097, -0.20669733],[-0.62263044, -0.21029513],[-0.56512957, -0.10453311],[-0.10389419, -0.1884054 ],[ 0.28613813, -0.39222563]])
        np.testing.assert_array_almost_equal(np.abs(scores_sparse), np.abs(expected_scores))


class TestTransformInnPro(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)

        kl = KarhunenLoeve(
            basis_name='bsplines', n_functions=5, random_state=42
        )
        kl.new(n_obs=10)
        kl.add_noise_and_sparsify(0.05, 0.9, 0.05)
        self.fdata_uni = kl.noisy_data
        self.fdata_sparse = kl.sparse_data

        uf_dense = UFPCA(n_components=2, method='inner-product')
        uf_dense.fit(self.fdata_uni)
        self.uf_dense = uf_dense

        uf_sparse = UFPCA(n_components=2, method='inner-product')
        uf_sparse.fit(self.fdata_sparse)
        self.uf_sparse = uf_sparse

    def test_pace_dense(self):
        scores_dense = scores_dense = _transform_innpro(
            self.fdata_uni, self.uf_dense._eigenvectors, self.uf_dense.eigenvalues
        )
        expected_scores = np.array([[-0.20008122, -0.35917685],[-0.0060345 , -0.50948171],[-0.32688088,  0.49534243],[-0.02326554, -0.15936427],[-0.17266196, -0.21488918],[-0.28291378,  0.09745108],[ 0.6209799 ,  0.17771336],[ 0.57871673,  0.05379105],[ 0.11103202,  0.1062688 ],[-0.29411576,  0.27046191]])
        np.testing.assert_array_almost_equal(np.abs(scores_dense), np.abs(expected_scores))

    def test_pace_irregular(self):
        scores_sparse = _transform_innpro(
            self.fdata_sparse, self.uf_sparse._eigenvectors, self.uf_sparse.eigenvalues
        )
        expected_scores = np.array([[-0.20302867, -0.36076181],[-0.03589738, -0.51099149],[-0.30730618,  0.53774483],[-0.03362226, -0.14896928],[-0.20057172, -0.22435715],[-0.26726809,  0.10645834],[ 0.62510127,  0.15013469],[ 0.57992695,  0.0435626 ],[ 0.10344107,  0.10811265],[-0.29378279,  0.26613948]])
        np.testing.assert_array_almost_equal(np.abs(scores_sparse), np.abs(expected_scores))
