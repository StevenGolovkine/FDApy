#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for MultivariateFunctionalData.

Written with the help of ChatGPT.

"""
import numpy as np
import pandas as pd
import pickle
import unittest

from pathlib import Path

from FDApy.representation.argvals import DenseArgvals
from FDApy.representation.values import DenseValues
from FDApy.representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData,
    MultivariateFunctionalData,
)

from FDApy.simulation.karhunen import KarhunenLoeve

THIS_DIR = Path(__file__)


class MultivariateFunctionalDataTest(unittest.TestCase):
    def setUp(self):
        self.argvals = {"input_dim_0": np.array([1, 2, 3, 4, 5])}
        self.values = np.array(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
        )

        self.fdata1 = DenseFunctionalData(
            DenseArgvals(self.argvals), DenseValues(self.values)
        )
        self.fdata2 = DenseFunctionalData(
            DenseArgvals(self.argvals), DenseValues(self.values)
        )
        self.fdata3 = DenseFunctionalData(
            DenseArgvals(self.argvals), DenseValues(self.values)
        )
        self.multivariate_data = MultivariateFunctionalData([self.fdata1, self.fdata2])

    def test_init(self):
        self.assertEqual(len(self.multivariate_data), 2)
        self.assertIsInstance(self.multivariate_data.data[0], DenseFunctionalData)
        self.assertIsInstance(self.multivariate_data.data[1], DenseFunctionalData)

        values = np.array(
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
        )
        fdata = DenseFunctionalData(DenseArgvals(self.argvals), DenseValues(values))
        with self.assertRaises(ValueError):
            MultivariateFunctionalData([self.fdata1, fdata])

    def test_repr(self):
        expected_repr = (
            f"Multivariate functional data object with 2 functions of 3 observations."
        )
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
        expected_n_points = [(5,), (5,)]
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


class TestNoisevariance(unittest.TestCase):
    def setUp(self) -> None:
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_sparse_5_10_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_sparse])

    def test_noise_variance(self):
        res = self.fdata.noise_variance(order=2)
        expected_res = [0.022610151507812086, 0.0002675118281909649]
        np.testing.assert_almost_equal(res, expected_res)


class TestSmoothMultivariateFunctionalData(unittest.TestCase):
    def setUp(self):
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_uni])

    def test_smooth(self):
        fdata_smooth = self.fdata.smooth(method='LP')

        self.assertIsInstance(fdata_smooth, MultivariateFunctionalData)
        self.assertIsInstance(fdata_smooth.data[0], DenseFunctionalData)
        self.assertIsInstance(fdata_smooth.data[1], DenseFunctionalData)
        np.testing.assert_equal(fdata_smooth.n_functional, 2)

    def test_error_list(self):
        points = DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)})
        with self.assertRaises(TypeError):
            self.fdata.smooth(
                points=points,
                method='LP',
                kernel_name=["epanechnikov", "epanechnikov"],
                bandwidth=[0.05, 0.05],
                degree=[1, 1],
            )

    def test_error_length_list(self):
        points = DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)})
        with self.assertRaises(ValueError):
            self.fdata.smooth(
                points=[points, points, points],
                kernel_name=["epanechnikov", "epanechnikov"],
                bandwidth=[0.05, 0.05],
                degree=[1, 1],
            )


class TestMeanhMultivariateFunctionalData(unittest.TestCase):
    def setUp(self):
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_uni])

    def test_mean(self):
        fdata_smooth = self.fdata.mean()

        self.assertIsInstance(fdata_smooth, MultivariateFunctionalData)
        self.assertIsInstance(fdata_smooth.data[0], DenseFunctionalData)
        self.assertIsInstance(fdata_smooth.data[1], DenseFunctionalData)
        np.testing.assert_equal(fdata_smooth.n_functional, 2)

    def test_error_list(self):
        points = DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)})
        with self.assertRaises(TypeError):
            self.fdata.mean(points=points)

    def test_error_length_list(self):
        points = DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)})
        with self.assertRaises(ValueError):
            self.fdata.mean(points=[points, points, points])


class TestCenterMultivariateFunctionalData(unittest.TestCase):
    def setUp(self):
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_uni])

    def test_center_1d(self):
        fdata_center = self.fdata.center()

        self.assertIsInstance(fdata_center, MultivariateFunctionalData)
        np.testing.assert_equal(fdata_center.n_functional, 2)

    def test_center_1d_with_given_mean(self):
        precomputed_mean = self.fdata.mean()
        fdata_center = self.fdata.center(mean=precomputed_mean)

        self.assertIsInstance(fdata_center, MultivariateFunctionalData)
        np.testing.assert_equal(fdata_center.n_functional, 2)


class TestNormMultivariateFunctionalData(unittest.TestCase):
    def setUp(self) -> None:
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_uni])

    def test_norm(self):
        res = self.fdata.norm()
        expected_res = np.array(
            [
                1.05258869,
                0.3708455,
                1.11546354,
                0.5636047,
                0.90327287,
                0.75064308,
                0.94101483,
                1.33889354,
                0.65137428,
                0.9804458,
            ]
        )
        np.testing.assert_array_almost_equal(res, expected_res)


class TestNormalizeMultivariateFunctionalData(unittest.TestCase):
    def setUp(self) -> None:
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_uni])

    def test_normalize(self):
        res, weights = self.fdata.normalize()

        expected_weight = np.array([0.20115047, 0.20115047])
        np.testing.assert_array_almost_equal(weights, expected_weight)

        np.testing.assert_equal(res.n_functional, 2)

        expected_values = np.array(
            [
                [
                    -2.42877589e00,
                    -1.75674625e00,
                    -1.85796607e00,
                    -1.08675732e-01,
                    1.05864161e00,
                    2.08720919e00,
                    3.08371103e00,
                    4.40221457e00,
                    3.23103598e00,
                    3.65031808e00,
                    1.28589723e00,
                ],
                [
                    -1.08375875e00,
                    -5.15201233e-01,
                    -2.52086603e-01,
                    -3.01815287e-01,
                    -4.33221232e-01,
                    -1.12947466e00,
                    -1.19639944e00,
                    -4.77331792e-01,
                    -9.37914118e-01,
                    -1.49230785e00,
                    -1.58791861e00,
                ],
                [
                    2.90406563e00,
                    3.15909349e00,
                    2.50695968e00,
                    2.37186810e00,
                    1.57164796e00,
                    1.87610774e00,
                    2.43514634e00,
                    2.51080947e00,
                    3.50678918e00,
                    3.50144205e00,
                    3.99756502e00,
                ],
                [
                    -4.74118253e-01,
                    -1.14453775e00,
                    -8.54335751e-01,
                    -1.92935786e00,
                    -2.11501843e00,
                    -1.90560603e00,
                    -1.39384903e00,
                    -3.62778367e-01,
                    3.37372830e-01,
                    1.17755958e00,
                    2.03592343e00,
                ],
                [
                    -2.23583853e00,
                    -1.21580668e00,
                    -2.08911737e-03,
                    1.33692887e00,
                    3.00811826e00,
                    2.90944622e00,
                    3.13617155e00,
                    3.45663905e00,
                    1.81653967e00,
                    1.41390720e00,
                    -3.18231931e-01,
                ],
                [
                    1.67311291e00,
                    1.56354537e00,
                    1.96915795e00,
                    2.54892023e00,
                    1.20784081e00,
                    2.21017978e00,
                    2.07263170e00,
                    1.65220865e00,
                    1.24506050e00,
                    2.02973246e00,
                    1.76443358e00,
                ],
                [
                    1.18775538e-01,
                    -1.08004835e00,
                    -1.00787855e00,
                    -2.36063071e00,
                    -3.01699789e00,
                    -2.61959222e00,
                    -2.80854728e00,
                    -2.41368682e00,
                    -2.73134301e00,
                    -2.83521323e00,
                    -1.88364168e00,
                ],
                [
                    -7.29331162e-01,
                    -1.38246052e00,
                    -2.33941106e00,
                    -2.76625964e00,
                    -3.82095489e00,
                    -3.24629614e00,
                    -3.94724586e00,
                    -4.66661716e00,
                    -4.33435443e00,
                    -3.33452035e00,
                    -2.47295186e00,
                ],
                [
                    2.85723090e00,
                    2.53318834e00,
                    4.33948213e-01,
                    -1.07346447e00,
                    -2.27981421e00,
                    -1.43018668e00,
                    -1.93327851e00,
                    -1.40668751e00,
                    -1.00483220e00,
                    -2.24203782e-01,
                    8.44649215e-01,
                ],
                [
                    3.33268618e00,
                    3.06480638e00,
                    2.35483620e00,
                    1.69575146e00,
                    1.97331318e00,
                    1.99888088e00,
                    2.90412200e00,
                    2.22264466e00,
                    2.82588332e00,
                    2.22902909e00,
                    1.90227626e00,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(res.data[0].values, expected_values)
        expected_values = np.array(
            [
                [
                    -2.42877589e00,
                    -1.75674625e00,
                    -1.85796607e00,
                    -1.08675732e-01,
                    1.05864161e00,
                    2.08720919e00,
                    3.08371103e00,
                    4.40221457e00,
                    3.23103598e00,
                    3.65031808e00,
                    1.28589723e00,
                ],
                [
                    -1.08375875e00,
                    -5.15201233e-01,
                    -2.52086603e-01,
                    -3.01815287e-01,
                    -4.33221232e-01,
                    -1.12947466e00,
                    -1.19639944e00,
                    -4.77331792e-01,
                    -9.37914118e-01,
                    -1.49230785e00,
                    -1.58791861e00,
                ],
                [
                    2.90406563e00,
                    3.15909349e00,
                    2.50695968e00,
                    2.37186810e00,
                    1.57164796e00,
                    1.87610774e00,
                    2.43514634e00,
                    2.51080947e00,
                    3.50678918e00,
                    3.50144205e00,
                    3.99756502e00,
                ],
                [
                    -4.74118253e-01,
                    -1.14453775e00,
                    -8.54335751e-01,
                    -1.92935786e00,
                    -2.11501843e00,
                    -1.90560603e00,
                    -1.39384903e00,
                    -3.62778367e-01,
                    3.37372830e-01,
                    1.17755958e00,
                    2.03592343e00,
                ],
                [
                    -2.23583853e00,
                    -1.21580668e00,
                    -2.08911737e-03,
                    1.33692887e00,
                    3.00811826e00,
                    2.90944622e00,
                    3.13617155e00,
                    3.45663905e00,
                    1.81653967e00,
                    1.41390720e00,
                    -3.18231931e-01,
                ],
                [
                    1.67311291e00,
                    1.56354537e00,
                    1.96915795e00,
                    2.54892023e00,
                    1.20784081e00,
                    2.21017978e00,
                    2.07263170e00,
                    1.65220865e00,
                    1.24506050e00,
                    2.02973246e00,
                    1.76443358e00,
                ],
                [
                    1.18775538e-01,
                    -1.08004835e00,
                    -1.00787855e00,
                    -2.36063071e00,
                    -3.01699789e00,
                    -2.61959222e00,
                    -2.80854728e00,
                    -2.41368682e00,
                    -2.73134301e00,
                    -2.83521323e00,
                    -1.88364168e00,
                ],
                [
                    -7.29331162e-01,
                    -1.38246052e00,
                    -2.33941106e00,
                    -2.76625964e00,
                    -3.82095489e00,
                    -3.24629614e00,
                    -3.94724586e00,
                    -4.66661716e00,
                    -4.33435443e00,
                    -3.33452035e00,
                    -2.47295186e00,
                ],
                [
                    2.85723090e00,
                    2.53318834e00,
                    4.33948213e-01,
                    -1.07346447e00,
                    -2.27981421e00,
                    -1.43018668e00,
                    -1.93327851e00,
                    -1.40668751e00,
                    -1.00483220e00,
                    -2.24203782e-01,
                    8.44649215e-01,
                ],
                [
                    3.33268618e00,
                    3.06480638e00,
                    2.35483620e00,
                    1.69575146e00,
                    1.97331318e00,
                    1.99888088e00,
                    2.90412200e00,
                    2.22264466e00,
                    2.82588332e00,
                    2.22902909e00,
                    1.90227626e00,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(res.data[1].values, expected_values)

    def test_normalize_with_given_weights(self):
        res, weights = self.fdata.normalize(weights=np.array([1, 1]))

        expected_weight = np.array([1, 1])
        np.testing.assert_array_almost_equal(weights, expected_weight)

        np.testing.assert_equal(res.n_functional, 2)


class TestInnerProductMultivariateFunctionalData(unittest.TestCase):
    def setUp(self) -> None:
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_uni])

    def test_inner_prod(self):
        res = self.fdata.inner_product(noise_variance=np.array([0, 0]))
        expected_res = np.array(
            [
                [
                    0.47349529,
                    -0.13685186,
                    0.19119084,
                    -0.03992746,
                    0.33384288,
                    0.11100958,
                    -0.34205956,
                    -0.47111704,
                    -0.26865753,
                    0.1448863,
                ],
                [
                    -0.13685186,
                    0.12450628,
                    -0.22608606,
                    0.05969545,
                    -0.09813655,
                    -0.13059145,
                    0.23567236,
                    0.31842272,
                    0.06843584,
                    -0.18454769,
                ],
                [
                    0.19119084,
                    -0.22608606,
                    0.48887238,
                    -0.16196905,
                    0.17461587,
                    0.28840398,
                    -0.47155034,
                    -0.65736206,
                    -0.10689362,
                    0.40955172,
                ],
                [
                    -0.03992746,
                    0.05969545,
                    -0.16196905,
                    0.17612865,
                    -0.15066836,
                    -0.14634096,
                    0.20094949,
                    0.27361664,
                    0.09435212,
                    -0.18174139,
                ],
                [
                    0.33384288,
                    -0.09813655,
                    0.17461587,
                    -0.15066836,
                    0.35205208,
                    0.15040087,
                    -0.3301055,
                    -0.45477844,
                    -0.27455165,
                    0.17201809,
                ],
                [
                    0.11100958,
                    -0.13059145,
                    0.28840398,
                    -0.14634096,
                    0.15040087,
                    0.20536565,
                    -0.30116113,
                    -0.41335325,
                    -0.09709361,
                    0.25418339,
                ],
                [
                    -0.34205956,
                    0.23567236,
                    -0.47155034,
                    0.20094949,
                    -0.3301055,
                    -0.30116113,
                    0.55853624,
                    0.75736446,
                    0.24419536,
                    -0.40056556,
                ],
                [
                    -0.47111704,
                    0.31842272,
                    -0.65736206,
                    0.27361664,
                    -0.45477844,
                    -0.41335325,
                    0.75736446,
                    1.05971458,
                    0.32831889,
                    -0.56709246,
                ],
                [
                    -0.26865753,
                    0.06843584,
                    -0.10689362,
                    0.09435212,
                    -0.27455165,
                    -0.09709361,
                    0.24419536,
                    0.32831889,
                    0.2357352,
                    -0.09771088,
                ],
                [
                    0.1448863,
                    -0.18454769,
                    0.40955172,
                    -0.18174139,
                    0.17201809,
                    0.25418339,
                    -0.40056556,
                    -0.56709246,
                    -0.09771088,
                    0.37342934,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(res, expected_res)

    def test_inner_prod_with_unknow_variance(self):
        res = self.fdata.inner_product()
        expected_res = np.array(
            [
                [
                    0.42827499,
                    -0.13685186,
                    0.19119084,
                    -0.03992746,
                    0.33384288,
                    0.11100958,
                    -0.34205956,
                    -0.47111704,
                    -0.26865753,
                    0.1448863,
                ],
                [
                    -0.13685186,
                    0.07928598,
                    -0.22608606,
                    0.05969545,
                    -0.09813655,
                    -0.13059145,
                    0.23567236,
                    0.31842272,
                    0.06843584,
                    -0.18454769,
                ],
                [
                    0.19119084,
                    -0.22608606,
                    0.44365208,
                    -0.16196905,
                    0.17461587,
                    0.28840398,
                    -0.47155034,
                    -0.65736206,
                    -0.10689362,
                    0.40955172,
                ],
                [
                    -0.03992746,
                    0.05969545,
                    -0.16196905,
                    0.13090835,
                    -0.15066836,
                    -0.14634096,
                    0.20094949,
                    0.27361664,
                    0.09435212,
                    -0.18174139,
                ],
                [
                    0.33384288,
                    -0.09813655,
                    0.17461587,
                    -0.15066836,
                    0.30683178,
                    0.15040087,
                    -0.3301055,
                    -0.45477844,
                    -0.27455165,
                    0.17201809,
                ],
                [
                    0.11100958,
                    -0.13059145,
                    0.28840398,
                    -0.14634096,
                    0.15040087,
                    0.16014534,
                    -0.30116113,
                    -0.41335325,
                    -0.09709361,
                    0.25418339,
                ],
                [
                    -0.34205956,
                    0.23567236,
                    -0.47155034,
                    0.20094949,
                    -0.3301055,
                    -0.30116113,
                    0.51331594,
                    0.75736446,
                    0.24419536,
                    -0.40056556,
                ],
                [
                    -0.47111704,
                    0.31842272,
                    -0.65736206,
                    0.27361664,
                    -0.45477844,
                    -0.41335325,
                    0.75736446,
                    1.01449428,
                    0.32831889,
                    -0.56709246,
                ],
                [
                    -0.26865753,
                    0.06843584,
                    -0.10689362,
                    0.09435212,
                    -0.27455165,
                    -0.09709361,
                    0.24419536,
                    0.32831889,
                    0.1905149,
                    -0.09771088,
                ],
                [
                    0.1448863,
                    -0.18454769,
                    0.40955172,
                    -0.18174139,
                    0.17201809,
                    0.25418339,
                    -0.40056556,
                    -0.56709246,
                    -0.09771088,
                    0.32820903,
                ],
            ]
        )
        np.testing.assert_array_almost_equal(res, expected_res)


class TestCovarianceMultivariateFunctionalData(unittest.TestCase):
    def setUp(self) -> None:
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_uni])

    def test_error_list_cov(self):
        points = DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)})
        with self.assertRaises(TypeError):
            self.fdata.covariance(points=points)

    def test_error_length_list_cov(self):
        points = DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)})
        with self.assertRaises(ValueError):
            self.fdata.covariance(points=[points, points, points])

    def test_covariance(self):
        res = self.fdata.covariance()

        np.testing.assert_equal(res.n_functional, 2)

        expected_values = np.array(
            [
                0.16515663,
                0.1462883,
                0.11637048,
                0.08890965,
                0.07124684,
                0.05672914,
                0.0534345,
                0.05765955,
                0.06480845,
                0.07809725,
                0.09294409,
            ]
        )
        np.testing.assert_array_almost_equal(
            self.fdata._covariance.data[0].values[0, 1], expected_values
        )
        expected_values = np.array(
            [
                0.16515663,
                0.1462883,
                0.11637048,
                0.08890965,
                0.07124684,
                0.05672914,
                0.0534345,
                0.05765955,
                0.06480845,
                0.07809725,
                0.09294409,
            ]
        )
        np.testing.assert_array_almost_equal(
            self.fdata._covariance.data[1].values[0, 1], expected_values
        )

        expected_noise = [0, 0]
        np.testing.assert_almost_equal(self.fdata._noise_variance_cov, expected_noise)

    def test_covariance_points(self):
        points = DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)})
        res = self.fdata.covariance(points=[points, points])

        np.testing.assert_equal(res.n_functional, 2)

        expected_values = np.array(
            [
                0.16515663,
                0.1462883,
                0.11637048,
                0.08890965,
                0.07124684,
                0.05672914,
                0.0534345,
                0.05765955,
                0.06480845,
                0.07809725,
                0.09294409,
            ]
        )
        np.testing.assert_array_almost_equal(
            self.fdata._covariance.data[0].values[0, 1], expected_values
        )
        expected_values = np.array(
            [
                0.16515663,
                0.1462883,
                0.11637048,
                0.08890965,
                0.07124684,
                0.05672914,
                0.0534345,
                0.05765955,
                0.06480845,
                0.07809725,
                0.09294409,
            ]
        )
        np.testing.assert_array_almost_equal(
            self.fdata._covariance.data[1].values[0, 1], expected_values
        )

        expected_noise = [0, 0]
        np.testing.assert_almost_equal(self.fdata._noise_variance_cov, expected_noise)


class TestConcatenateMultivariateFunctionalData(unittest.TestCase):
    def setUp(self):
        fname = THIS_DIR.parent / "data/data_noisy_5_10_001.pickle"
        with open(fname, "rb") as handle:
            self.fdata_uni = pickle.load(handle)

        fname = THIS_DIR.parent / "data/data_sparse_5_10_08.pickle"
        with open(fname, "rb") as handle:
            self.fdata_sparse = pickle.load(handle)
        self.fdata = MultivariateFunctionalData([self.fdata_uni, self.fdata_sparse])

    def test_concatenate(self):
        res = MultivariateFunctionalData.concatenate(self.fdata, self.fdata)

        self.assertIsInstance(res, MultivariateFunctionalData)
        np.testing.assert_equal(res.n_functional, 2)
        self.assertIsInstance(res.data[0], DenseFunctionalData)
        self.assertIsInstance(res.data[1], IrregularFunctionalData)
        np.testing.assert_equal(res.data[0].n_obs, 20)
        np.testing.assert_equal(res.data[1].n_obs, 20)

    def test_concatenate_error(self):
        new = MultivariateFunctionalData(
            [self.fdata_uni, self.fdata_sparse, self.fdata_uni]
        )
        with self.assertRaises(ValueError):
            MultivariateFunctionalData.concatenate(self.fdata, new)
