#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for MultivariateFunctionalData.

Written with the help of ChatGPT.

"""
import numpy as np
import pandas as pd
import unittest

from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
from FDApy.representation.values import DenseValues, IrregularValues
from FDApy.representation.functional_data import (
    DenseFunctionalData,
    MultivariateFunctionalData
)

from FDApy.simulation.karhunen import KarhunenLoeve


class MultivariateFunctionalDataTest(unittest.TestCase):
    def setUp(self):
        self.argvals = {'input_dim_0': np.array([1, 2, 3, 4, 5])}
        self.values = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])

        self.fdata1 = DenseFunctionalData(DenseArgvals(self.argvals), DenseValues(self.values))
        self.fdata2 = DenseFunctionalData(DenseArgvals(self.argvals), DenseValues(self.values))
        self.fdata3 = DenseFunctionalData(DenseArgvals(self.argvals), DenseValues(self.values))
        self.multivariate_data = MultivariateFunctionalData([self.fdata1, self.fdata2])

    def test_init(self):
        self.assertEqual(len(self.multivariate_data), 2)
        self.assertIsInstance(self.multivariate_data.data[0], DenseFunctionalData)
        self.assertIsInstance(self.multivariate_data.data[1], DenseFunctionalData)
        
        values = np.array([
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        fdata = DenseFunctionalData(DenseArgvals(self.argvals), DenseValues(values))
        with self.assertRaises(ValueError):
            MultivariateFunctionalData([self.fdata1, fdata])

    def test_repr(self):
        expected_repr = f"Multivariate functional data object with 2 functions of 3 observations."
        actual_repr = repr(self.multivariate_data)
        self.assertEqual(actual_repr, expected_repr)

    def test_getitem(self):
        fdata = self.multivariate_data[0]

        np.testing.assert_array_equal(fdata.data[0].values, self.fdata1[0].values)
        np.testing.assert_array_equal(fdata.data[1].values, self.fdata2[0].values)

    def test_n_obs(self):
        expected_n_obs = 3
        actual_n_obs = self.multivariate_data.n_obs
        self.assertEqual(actual_n_obs, expected_n_obs)

    def test_n_functional(self):
        expected_n_functional = 2
        actual_n_functional = self.multivariate_data.n_functional
        self.assertEqual(actual_n_functional, expected_n_functional)

    def test_n_dimension(self):
        expected_n_dimension = [1, 1]
        actual_n_dimension = self.multivariate_data.n_dimension
        self.assertEqual(actual_n_dimension, expected_n_dimension)

    def test_n_points(self):
        expected_n_points = [(5, ), (5, )]
        actual_n_points = self.multivariate_data.n_points
        self.assertEqual(actual_n_points, expected_n_points)

    def test_append(self):
        res = MultivariateFunctionalData([])

        res.append(self.fdata1)
        np.testing.assert_equal(res.n_functional, 1)

        res.append(self.fdata2)
        np.testing.assert_equal(res.n_functional, 2)

    def test_extend(self):
        self.multivariate_data.extend([self.fdata1, self.fdata3])
        np.testing.assert_equal(self.multivariate_data.n_functional, 4)

    def test_insert(self):
        self.multivariate_data.insert(1, self.fdata3)
        np.testing.assert_equal(self.multivariate_data.n_functional, 3)
        np.testing.assert_equal(self.multivariate_data.data[1], self.fdata3)

    def test_remove(self):
        self.multivariate_data.remove(self.fdata1)
        np.testing.assert_equal(self.multivariate_data.n_functional, 1)
        np.testing.assert_equal(self.multivariate_data.data[0], self.fdata2)

    def test_pop(self):
        popped_data = self.multivariate_data.pop(0)
        np.testing.assert_equal(self.multivariate_data.n_functional, 1)
        np.testing.assert_equal(popped_data, self.fdata1)

    def test_clear(self):
        self.multivariate_data.clear()
        np.testing.assert_equal(self.multivariate_data.n_functional, 0)

    def test_reverse(self):
        self.multivariate_data.reverse()
        np.testing.assert_equal(self.multivariate_data.data[0], self.fdata2)
        np.testing.assert_equal(self.multivariate_data.data[1], self.fdata1)
    
    def test_to_long(self):
        fdata_long = self.multivariate_data.to_long()
        np.testing.assert_array_equal(len(fdata_long), 2)
        self.assertIsInstance(fdata_long[0], pd.DataFrame)
        self.assertIsInstance(fdata_long[1], pd.DataFrame)


class TestSmoothMultivariateFunctionalData(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=50)
        kl.add_noise_and_sparsify(0.05, 0.5)
        
        fdata_1 = kl.data
        fdata_2 = kl.noisy_data
        self.fdata = MultivariateFunctionalData([fdata_1, fdata_2])

    def test_smooth(self):
        fdata_smooth = self.fdata.smooth()

        self.assertIsInstance(fdata_smooth, MultivariateFunctionalData)
        self.assertIsInstance(fdata_smooth.data[0], DenseFunctionalData)
        self.assertIsInstance(fdata_smooth.data[1], DenseFunctionalData)
        np.testing.assert_equal(fdata_smooth.n_functional, 2)

    def test_error_list(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        with self.assertRaises(TypeError):
            self.fdata.smooth(
                points=points,
                kernel_name=['epanechnikov', 'epanechnikov'],
                bandwidth=[0.05, 0.05],
                degree=[1, 1]
            )

    def test_error_length_list(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        with self.assertRaises(ValueError):
            self.fdata.smooth(
                points=[points, points, points],
                kernel_name=['epanechnikov', 'epanechnikov'],
                bandwidth=[0.05, 0.05],
                degree=[1, 1]
            )


class TestMeanhMultivariateFunctionalData(unittest.TestCase):
    def setUp(self):
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=50)
        kl.add_noise_and_sparsify(0.05, 0.5)
        
        fdata_1 = kl.data
        fdata_2 = kl.noisy_data
        self.fdata = MultivariateFunctionalData([fdata_1, fdata_2])

    def test_mean(self):
        fdata_smooth = self.fdata.mean()

        self.assertIsInstance(fdata_smooth, MultivariateFunctionalData)
        self.assertIsInstance(fdata_smooth.data[0], DenseFunctionalData)
        self.assertIsInstance(fdata_smooth.data[1], DenseFunctionalData)
        np.testing.assert_equal(fdata_smooth.n_functional, 2)

    def test_error_list(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        with self.assertRaises(TypeError):
            self.fdata.mean(points=points)

    def test_error_length_list(self):
        points = DenseArgvals({'input_dim_0': np.linspace(0, 1, 11)})
        with self.assertRaises(ValueError):
            self.fdata.mean(points=[points, points, points])


class TestInnerProductMultivariateFunctionalData(unittest.TestCase):
    def setUp(self) -> None:
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=4)
        kl.add_noise_and_sparsify(0.05, 0.5)

        fdata_1 = kl.data
        fdata_2 = kl.sparse_data
        self.fdata = MultivariateFunctionalData([fdata_1, fdata_2])

    def test_inner_prod(self):
        res = self.fdata.inner_product()
        expected_res = np.array([[ 0.58532546,  0.19442368, -0.04038602,  0.01705178],[ 0.19442368,  0.38395264, -0.45055398,  0.10919059],[-0.04038602, -0.45055398,  0.96833672, -0.07948717],[ 0.01705178,  0.10919059, -0.07948717,  0.18026045]])
        np.testing.assert_array_almost_equal(res, expected_res)


class TestInnerProductMultivariateFunctionalData(unittest.TestCase):
    def setUp(self) -> None:
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=4)
        kl.add_noise_and_sparsify(0.05, 0.5)

        fdata_1 = kl.data
        fdata_2 = kl.sparse_data
        self.fdata = MultivariateFunctionalData([fdata_1, fdata_2])

    def test_norm(self):
        res = self.fdata.norm()
        expected_res = np.array([1.05384959, 0.84700578, 1.37439764, 0.59235447])
        np.testing.assert_array_almost_equal(res, expected_res)


class TestNormalizeMultivariateFunctionalData(unittest.TestCase):
    def setUp(self) -> None:
        name = 'bsplines'
        n_functions = 5
        kl = KarhunenLoeve(
            basis_name=name, n_functions=n_functions, random_state=42
        )
        kl.new(n_obs=4)
        kl.add_noise_and_sparsify(0.05, 0.5)

        fdata_1 = kl.data
        fdata_2 = kl.sparse_data
        self.fdata = MultivariateFunctionalData([fdata_1, fdata_2])

    def test_normalize(self):
        res, weights = self.fdata.normalize()

        expected_weight = np.array([0.20365764, 0.19388443])
        np.testing.assert_array_almost_equal(weights, expected_weight)

        np.testing.assert_equal(res.n_functional, 2)
        
        expected_values = np.array([[ 1.49622218,  1.11315141,  0.75585471,  0.42370986,   0.11609462, -0.16761321, -0.42803585, -0.66579554,  -0.88151448, -1.0758149 , -1.24931903, -1.40264907,  -1.53642725, -1.65127579, -1.74781692, -1.82667284,  -1.88846579, -1.93381799, -1.96335164, -1.97768898,  -1.97745223, -1.9632636 , -1.93574531, -1.8955196 ,  -1.84320867, -1.77943474, -1.70482005, -1.6199868 ,  -1.52555722, -1.42215353, -1.31039795, -1.19091271,  -1.06432001, -0.93124208, -0.79230115, -0.64811943,  -0.49931914, -0.34652251, -0.19035175, -0.03142909,   0.12962326,  0.29218308,  0.45562813,  0.61933621,   0.78268509,  0.94505256,  1.10581638,  1.26435435,   1.42004424,  1.57226382,  1.72039089,  1.86380531,   2.00189537,  2.13405142,  2.25966385,  2.37812301,   2.48881929,  2.59114305,  2.68448467,  2.76823451,   2.84178294,  2.90452035,  2.95583709,  2.99512353,   3.02177006,  3.03516703,  3.03470482,  3.01977381,   2.98976435,  2.94406683,  2.88207161,  2.80316906,   2.70674956,  2.59220347,  2.45892117,  2.30629303,   2.13370941,  1.94056069,  1.72623724,  1.49012942,   1.23162762,  0.9501222 ,  0.64500353,  0.31566198,  -0.03851207, -0.41812826, -0.82379622, -1.25612557,  -1.71572595, -2.20320698, -2.71917829, -3.26424951,  -3.83903028, -4.44413021, -5.08015895, -5.74772611,  -6.44744133, -7.17991424, -7.94575447, -8.74557164,  -9.57997539], [-6.39396342, -5.98232444, -5.58972096, -5.2157198 ,  -4.85988778, -4.52179171, -4.20099842, -3.8970747 ,  -3.60958739, -3.3381033 , -3.08218924, -2.84141203,  -2.61533848, -2.40353542, -2.20556965, -2.02100799,  -1.84941726, -1.69036428, -1.54341586, -1.40813882,  -1.28409996, -1.17086612, -1.0680041 , -0.97508072,  -0.8916628 , -0.81731715, -0.75161059, -0.69410993,  -0.644382  , -0.6019936 , -0.56651155, -0.53750267,  -0.51453377, -0.49717168, -0.4849832 , -0.47753515,  -0.47439435, -0.47512762, -0.47930176, -0.4864836 ,  -0.49623995, -0.50813763, -0.52174345, -0.53662424,  -0.55234679, -0.56847794, -0.5845845 , -0.60023328,  -0.6149911 , -0.62842478, -0.64010112, -0.64970519,  -0.65739493, -0.66344654, -0.66813622, -0.67174014,  -0.67453452, -0.67679553, -0.67879937, -0.68082223,  -0.6831403 , -0.68602978, -0.68976685, -0.69462771,  -0.70088855, -0.70882556, -0.71871493, -0.73083285,  -0.74545552, -0.76285913, -0.78331986, -0.80711392,  -0.83451749, -0.86580676, -0.90125792, -0.94114718,  -0.98575071, -1.03534471, -1.09020537, -1.15060889,  -1.21683145, -1.28914925, -1.36783848, -1.45317533,  -1.54543599, -1.64489665, -1.75183351, -1.86652276,  -1.98924058, -2.12026317, -2.25986673, -2.40832744,  -2.56592149, -2.73292508, -2.9096144 , -3.09626564,  -3.29315499, -3.50055865, -3.7187528 , -3.94801363,  -4.18861735], [ 4.31802102,  4.28661581,  4.2523785 ,  4.21547349,   4.17606518,  4.13431796,  4.09039624,  4.04446441,   3.99668687,  3.94722802,  3.89625225,  3.84392396,   3.79040756,  3.73586743,  3.68046797,  3.62437359,   3.56774868,  3.51075764,  3.45356486,  3.39633475,   3.3392317 ,  3.28242011,  3.22606437,  3.17032889,   3.11537806,  3.06137628,  3.00848794,  2.95687745,   2.90670921,  2.8581476 ,  2.81135703,  2.7665019 ,   2.7237466 ,  2.68325553,  2.64519309,  2.60972368,   2.57701169,  2.54722152,  2.52051757,  2.49706424,   2.47702592,  2.46056702,  2.44785192,  2.43904503,   2.43431075,  2.43381347,  2.43771759,  2.44618751,   2.45938762,  2.47748233,  2.50063603,  2.52892155,   2.56204544,  2.59962268,  2.64126826,  2.68659715,   2.73522434,  2.78676482,  2.84083356,  2.89704554,   2.95501575,  3.01435917,  3.07469079,  3.13562557,   3.19677851,  3.25776459,  3.31819879,  3.37769609,   3.43587148,  3.49233993,  3.54671643,  3.59861596,   3.64765351,  3.69344404,  3.73560256,  3.77374403,   3.80748344,  3.83643578,  3.86021602,  3.87843915,   3.89072014,  3.89667399,  3.89591567,  3.88806017,   3.87272246,  3.84951753,  3.81806037,  3.77796594,   3.72884925,  3.67032526,  3.60200896,  3.52351533,   3.43445936,  3.33445602,  3.2231203 ,  3.10006718,   2.96491164,  2.81726866,  2.65675324,  2.48298034,   2.29556495], [-4.21929891, -3.86854444, -3.53945219, -3.23138976,  -2.94372479, -2.67582489, -2.42705769, -2.1967908 ,  -1.98439185, -1.78922846, -1.61066825, -1.44807883,  -1.30082783, -1.16828287, -1.04981157, -0.94478156,  -0.85256044, -0.77251585, -0.7040154 , -0.64642672,  -0.59911742, -0.56145513, -0.53280746, -0.51254204,  -0.50002649, -0.49462842, -0.49571547, -0.50265525,  -0.51481537, -0.53156347, -0.55226716, -0.57629406,  -0.6030118 , -0.63178799, -0.66199025, -0.69298621,  -0.72414349, -0.7548297 , -0.78441247, -0.81225942,  -0.83773817, -0.86021634, -0.87906155, -0.89364142,  -0.90332358, -0.90747564, -0.90546522, -0.89665994,  -0.88042743, -0.85613531, -0.82315119, -0.78105173,  -0.73024971, -0.67136696, -0.60502528, -0.53184649,  -0.45245242, -0.36746486, -0.27750565, -0.1831966 ,  -0.08515952,  0.01598377,  0.11961145,  0.22510171,   0.33183273,  0.4391827 ,  0.54652979,  0.6532522 ,   0.7587281 ,  0.86233568,  0.96345312,  1.06145861,   1.15573033,  1.24564646,  1.33058519,  1.4099247 ,   1.48304318,  1.54931881,  1.60812977,  1.65885424,   1.70087042,  1.73355647,  1.7562906 ,  1.76845097,   1.76941578,  1.75856321,  1.73527144,  1.69891866,   1.64888304,  1.58454278,  1.50527606,  1.41046105,   1.29947595,  1.17169894,  1.0265082 ,  0.86328191,   0.68139826,  0.48023544,  0.25917162,  0.01758499,  -0.24514627]])
        np.testing.assert_array_almost_equal(res.data[0].values, expected_values)
        expected_values = np.array([ 1.3584412 ,  2.20391159,  0.26684921, -0.37204357,  0.16429984, -1.94207116, -2.2046438 , -2.67303592, -1.12551113, -0.61670761, -1.21993587, -1.43580713, -1.7233163 , -1.616918  , -0.78573731, -1.44377352, -1.4159017 , -1.3747195 , -0.84180992,  1.36012511, -0.07931144,  0.66629818,  1.64239541,  1.9076571 ,  0.7593733 ,  0.79485713,  2.48112806,  0.33586305,  1.04240194,  2.53782715,  2.12152399,  2.28753089,  1.79492849,  3.60745374,  3.79473761,  2.83088629,  2.35500009,  2.63093076,  0.75359332,  0.71579549,  2.49641462, -0.67613604, -0.83308437, -2.35522448, -3.15461153, -6.33588495, -9.46246192])
        np.testing.assert_array_almost_equal(res.data[1].values[0], expected_values)
