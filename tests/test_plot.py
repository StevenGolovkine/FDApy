#!/usr/bin/python3
# -*-coding:utf8 -*
"""Module that contains unit tests for the functions of the plot.py file.

Written with the help of ChatGPT.

"""
import numpy as np
import unittest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
from FDApy.representation.values import DenseValues, IrregularValues
from FDApy.representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData,
    MultivariateFunctionalData
)
from FDApy.visualization.plot import (
    _init_ax,
    _plot_1d,
    _plot_2d,
    plot,
    plot_multivariate
)


class TestInitAx(unittest.TestCase):
    def test_init_ax_default(self):
        ax = _init_ax()
        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.name, 'rectilinear')

    def test_init_ax_3d(self):
        ax = _init_ax(projection='3d')
        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.name, '3d')

    def test_init_ax_custom(self):
        ax = plt.axes(projection='polar')
        ax = _init_ax(ax=ax)
        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.name, 'polar')


class TestPlot1D(unittest.TestCase):
    def setUp(self):
        self.arg_den = np.linspace(0, 1, 100)
        self.val_den = np.sin(2 * np.pi * self.arg_den)
        self.data_dense = DenseFunctionalData(
            DenseArgvals({'input_dim_0': self.arg_den}),
            DenseValues(self.val_den[np.newaxis])
        )

        self.arg_irr = np.array([0, 1, 2, 3, 4])
        self.val_irr = np.array([1, 2, 3, 4, 5])
        self.data_irreg = IrregularFunctionalData(
            IrregularArgvals({0: DenseArgvals({'input_dim_0': self.arg_irr})}),
            IrregularValues({0: self.val_irr})
        )

        self.labels = np.array([0])

    def test_plot_1d_error(self):
        with self.assertRaises(TypeError):
            _plot_1d(data=np.array([1, 2, 3]), labels=np.array([0]))

    def test_plot_1d_dense(self):
        # Call the function to plot the object
        _, ax = plt.subplots()
        ax = _plot_1d(self.data_dense, labels=self.labels, colors=None, ax=ax)

        # Generate the expected plot
        _, ax_expected = plt.subplots()
        ax_expected.plot(self.arg_den, self.val_den, c='b')

        # Compare the generated plot with the expected plot
        np.testing.assert_array_equal(
            ax.get_lines()[0].get_xdata(),
            ax_expected.get_lines()[0].get_xdata()
        )
        np.testing.assert_array_equal(
            ax.get_lines()[0].get_ydata(),
            ax_expected.get_lines()[0].get_ydata()
        )
        np.testing.assert_array_equal(
            ax.get_lines()[0].get_color(), mpl.cm.jet(self.labels[0])
        )

    def test_plot_1d_irregular(self):
        # Call the function with the data and labels
        _, ax = plt.subplots()
        ax = _plot_1d(self.data_irreg, labels=self.labels, ax=ax)

        # Check if the plot has the correct number of curves
        np.testing.assert_equal(len(ax.lines), 1)

        # Check if the curves have the correct data and colors
        np.testing.assert_array_equal(
            ax.get_lines()[0].get_xdata(), self.arg_irr
        )
        np.testing.assert_array_equal(
            ax.get_lines()[0].get_ydata(), self.val_irr
        )
        np.testing.assert_array_equal(
            ax.get_lines()[0].get_color(), mpl.cm.jet(self.labels[0])
        )

    def test_plot_1d_with_colors(self):
        # Plot the data with specified colors
        _, ax = plt.subplots()
        _plot_1d(self.data_dense, self.labels, colors='r', ax=ax)

        np.testing.assert_equal(ax.get_lines()[0].get_color(), 'r')


class TestPlot2D(unittest.TestCase):
    def setUp(self):
        self.arg_den = {
            'input_dim_0': np.array([1, 2, 3, 4]),
            'input_dim_1': np.array([5, 6, 7])
        }
        self.val_den = np.array([
            [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
            [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]]
        ])
        self.data_dense = DenseFunctionalData(
            DenseArgvals(self.arg_den), 
            DenseValues(self.val_den)
        )

        self.arg_irr = {0: DenseArgvals(
            {
                'input_dim_0': np.array([1, 2, 3, 4]),
                'input_dim_1': np.array([5, 6, 7])
            }
        )}
        self.val_irr = {
            0: np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]])
        }
        self.data_irreg = IrregularFunctionalData(
            IrregularArgvals(self.arg_irr),
            IrregularValues(self.val_irr)
        )

        self.labels = np.array([0, 1])

    def test_plot_2d_error_type(self):
        with self.assertRaises(TypeError):
            _plot_2d(data=np.array([1, 2, 3]), labels=np.array([0]))

    def test_plot_2d_error_irregular(self):
        with self.assertRaises(NotImplementedError):
            _plot_2d(data=self.data_irreg, labels=np.array([0]))

    def test_plot_2d_dense_unique(self):
        # Call the function to plot the object
        ax = _init_ax(projection='rectilinear')
        ax = _plot_2d(self.data_dense[0], labels=self.labels[0], ax=ax)

        # Generate the expected plot
        ax_expected = _init_ax(projection='rectilinear')
        ax_expected.contourf(
            self.arg_den['input_dim_1'],
            self.arg_den['input_dim_0'],
            self.val_den[0]
        )

        self.assertIsInstance(ax, Axes)
        for idx in np.arange(len(ax.collections)):
            # For the vertices
            for idx_sub in np.arange(len(ax.collections[idx].get_paths())):
                np.testing.assert_array_equal(
                    ax.collections[idx].get_paths()[idx_sub].vertices,
                    ax_expected.collections[idx].get_paths()[idx_sub].vertices
                )

    def test_plot_2d_dense_multiple(self):
        ax = _init_ax(projection='3d')
        ax = _plot_2d(
            self.data_dense,
            labels=self.labels, ax=ax, colors=['r', 'y']
        )

        # Generate the expected plot
        ax_expected = _init_ax(projection='3d')
        x, y = np.meshgrid(
            self.data_dense.argvals['input_dim_0'],
            self.data_dense.argvals['input_dim_1'],
            indexing='ij'
        )
        ax_expected.plot_surface(x, y, self.data_dense.values[0], color='r')
        ax_expected.plot_surface(x, y, self.data_dense.values[1], color='y')

        self.assertIsInstance(ax, Axes)
        for idx in np.arange(len(ax.collections)):
            # For the vertices
            for idx_sub in np.arange(len(ax.collections[idx].get_paths())):
                np.testing.assert_array_equal(
                    ax.collections[idx].get_paths()[idx_sub].vertices,
                    ax_expected.collections[idx].get_paths()[idx_sub].vertices
                )
            # For the colors
            np.testing.assert_equal(
                ax.collections[idx]._facecolors,
                ax_expected.collections[idx]._facecolors
            )


class TestPlot(unittest.TestCase):
    def setUp(self):
        self.arg_den_1d = np.linspace(0, 1, 100)
        self.val_den_1d = np.sin(2 * np.pi * self.arg_den_1d)
        self.data_dense_1d = DenseFunctionalData(
            DenseArgvals({'input_dim_0': self.arg_den_1d}),
            DenseValues(self.val_den_1d[np.newaxis])
        )

        self.arg_den_2d = {'input_dim_0': np.array([1, 2, 3, 4]), 'input_dim_1': np.array([5, 6, 7])}
        self.val_den_2d = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]]])
        self.data_dense_2d = DenseFunctionalData(
            DenseArgvals(self.arg_den_2d),
            DenseValues(self.val_den_2d)
        )

    def test_plot_error(self):
        arg_den_3d = {'input_dim_0': np.array([1, 2]), 'input_dim_1': np.array([5, 6]), 'input_dim_2': np.array([3, 4])}
        val_den_3d = np.array([[[[1, 2], [1, 2]], [[1, 2], [1, 2]]], [[[5, 6], [5, 6]], [[5, 6], [5, 6]]]])
        data_dense_3d = DenseFunctionalData(
            DenseArgvals(arg_den_3d),
            DenseValues(val_den_3d)
        )
        with self.assertRaises(ValueError):
            plot(data_dense_3d)

    def test_plot_1d(self):
        ax = plot(self.data_dense_1d, labels=np.array([0]))
        np.testing.assert_array_equal(ax.get_lines()[0].get_xdata(), self.arg_den_1d)
        np.testing.assert_array_equal(ax.get_lines()[0].get_ydata(), self.val_den_1d)

    def test_plot_2d_nobs(self):
        ax = plot(self.data_dense_2d, labels=np.array([0, 0]))
        
        # Generate the expected plot
        ax_expected = _init_ax(projection='3d')
        x, y = np.meshgrid(
            self.data_dense_2d.argvals['input_dim_0'],
            self.data_dense_2d.argvals['input_dim_1'],
            indexing='ij'
        )
        ax_expected.plot_surface(x, y, self.data_dense_2d.values[0])
        ax_expected.plot_surface(x, y, self.data_dense_2d.values[1])

        self.assertIsInstance(ax, Axes)
        for idx in np.arange(len(ax.collections)):
            # For the vertices
            for idx_sub in np.arange(len(ax.collections[idx].get_paths())):
                np.testing.assert_array_equal(
                    ax.collections[idx].get_paths()[idx_sub].vertices,
                    ax_expected.collections[idx].get_paths()[idx_sub].vertices
                )
    
    def test_plot_2d_unique_obs(self):
        ax = plot(self.data_dense_2d[0], labels=np.array([0]))
        
        # Generate the expected plot
        ax_expected = _init_ax(projection='rectilinear')
        ax_expected.contourf(
            self.arg_den_2d['input_dim_1'],
            self.arg_den_2d['input_dim_0'],
            self.val_den_2d[0]
        )

        self.assertIsInstance(ax, Axes)
        for idx in np.arange(len(ax.collections)):
            # For the vertices
            for idx_sub in np.arange(len(ax.collections[idx].get_paths())):
                np.testing.assert_array_equal(
                    ax.collections[idx].get_paths()[idx_sub].vertices,
                    ax_expected.collections[idx].get_paths()[idx_sub].vertices
                )

    def test_plot_label_none(self):
        ax = plot(self.data_dense_2d, labels=None)
    
        np.testing.assert_array_almost_equal(
            ax.collections[0]._facecolors,
            np.array([[0., 0., 0.20125634, 1.],[0., 0., 0.20125634, 1.],[0., 0., 0.20125634, 1.],[0., 0., 0.20125634, 1.],[0., 0., 0.20125634, 1.],[0., 0., 0.20125634, 1.]])
        )
        np.testing.assert_array_almost_equal(
            ax.collections[1]._facecolors,
            np.array([[0.20125634, 0., 0., 1.],[0.20125634, 0., 0., 1.],[0.20125634, 0., 0., 1.],[0.20125634, 0., 0., 1.],[0.20125634, 0., 0., 1.],[0.20125634, 0., 0., 1.]])
        )


class TestPlotMultivariate(unittest.TestCase):
    def setUp(self):
        self.arg_den_1d = np.linspace(0, 1, 100)
        self.val_den_1d = np.stack([np.sin(2 * np.pi * self.arg_den_1d),np.sin(-2 * np.pi * self.arg_den_1d)])
        self.data_dense_1d = DenseFunctionalData(
            DenseArgvals({'input_dim_0': self.arg_den_1d}),
            DenseValues(self.val_den_1d)
        )

        self.arg_den_2d = {'input_dim_0': np.array([1, 2, 3, 4]), 'input_dim_1': np.array([5, 6, 7])}
        self.val_den_2d = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], [[5, 6, 7], [5, 6, 7], [5, 6, 7], [5, 6, 7]]])
        self.data_dense_2d = DenseFunctionalData(
            DenseArgvals(self.arg_den_2d),
            DenseValues(self.val_den_2d)
        )

        self.data = MultivariateFunctionalData([self.data_dense_1d, self.data_dense_2d])

    def test_plot_multivariate(self):
        ax = plot_multivariate(self.data)

        # Generate the expected plot for 1D
        _, ax_expected_1d = plt.subplots()
        ax_expected_1d.plot(self.arg_den_1d, self.val_den_1d[0,:])
        ax_expected_1d.plot(self.arg_den_1d, self.val_den_1d[1,:])

        np.testing.assert_array_equal(
            ax[0].get_lines()[0].get_xdata(),
            ax_expected_1d.get_lines()[0].get_xdata()
        )
        np.testing.assert_array_equal(
            ax[0].get_lines()[0].get_ydata(),
            ax_expected_1d.get_lines()[0].get_ydata()
        )
        np.testing.assert_array_equal(
            ax[0].get_lines()[1].get_xdata(),
            ax_expected_1d.get_lines()[1].get_xdata()
        )
        np.testing.assert_array_equal(
            ax[0].get_lines()[1].get_ydata(),
            ax_expected_1d.get_lines()[1].get_ydata()
        )

        # Generate the expected plot for 2D
        ax_expected_2d = _init_ax(projection='rectilinear')
        ax_expected_2d.contourf(
            self.arg_den_2d['input_dim_1'],
            self.arg_den_2d['input_dim_0'],
            self.val_den_2d[0]
        )

        self.assertIsInstance(ax[1], Axes)
        for idx in np.arange(len(ax[1].collections)):
            # For the vertices
            for idx_sub in np.arange(len(ax[1].collections[idx].get_paths())):
                np.testing.assert_array_equal(
                    ax[1].collections[idx].get_paths()[idx_sub].vertices,
                    ax_expected_2d.collections[idx].get_paths()[idx_sub].vertices
                )

        # Get the title
        np.testing.assert_equal(ax[0].title.get_text(), '')
        np.testing.assert_equal(ax[1].title.get_text(), '')

    def test_plot_multivariate_error(self):
        arg_den_3d = {'input_dim_0': np.array([1, 2]), 'input_dim_1': np.array([5, 6]), 'input_dim_2': np.array([3, 4])}
        val_den_3d = np.array([[[[1, 2], [1, 2]], [[1, 2], [1, 2]]], [[[5, 6], [5, 6]], [[5, 6], [5, 6]]]])
        data_dense_3d = DenseFunctionalData(
            DenseArgvals(arg_den_3d),
            DenseValues(val_den_3d)
        )

        data = MultivariateFunctionalData([self.data_dense_1d, data_dense_3d])
        with self.assertRaises(ValueError):
            plot_multivariate(data)
