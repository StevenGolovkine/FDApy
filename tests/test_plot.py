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

from FDApy.representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData,
    MultivariateFunctionalData
)
from FDApy.visualization.plot import (
    _init_ax,
    _plot_1d
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
            {'input_dim_0': self.arg_den}, self.val_den[np.newaxis]
        )

        self.arg_irr = np.array([0, 1, 2, 3, 4])
        self.val_irr = np.array([1, 2, 3, 4, 5])
        self.data_irreg = IrregularFunctionalData(
            {'input_dim_0': {0: self.arg_irr}}, {0: self.val_irr}
        )
        
        self.labels = np.array([0])
        
    def test_plot_1d_error(self):
        with self.assertRaises(TypeError):
            _plot_1d(data=np.array([1, 2, 3]))

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

    # def test_plot_1d_default_colors():
    #     # Create an IrregularFunctionalData object
    #     x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    #     y = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]])
    #     data = IrregularFunctionalData(x, y)

    #     # Create labels
    #     labels = np.array([0, 1, 0, 1, 0])

    #     # Call the plotting function with default colors
    #     fig, ax = plt.subplots()
    #     _plot_1d(data, labels, ax=ax)

    #     # Check that the colors used match the default colormap
    #     expected_colors = plt.cm.jet(np.linspace(0, 1, len(np.unique(labels))))
    #     assert_array_equal(ax.collections[0].get_facecolors(), expected_colors[labels])

    # def test_plot_1d_with_colors():
    #     # Create some example data
    #     n_obs = 5
    #     n_values = 20
    #     x = np.random.uniform(low=0, high=1, size=(n_obs, n_values))
    #     y = np.random.uniform(low=-1, high=1, size=(n_obs, n_values))
    #     data = IrregularFunctionalData(x, y)

    #     # Specify colors for each observation
    #     labels = np.array([0, 1, 1, 2, 2])
    #     colors = np.array(['red', 'green', 'blue'])

    #     # Plot the data with specified colors
    #     fig, ax = plt.subplots()
    #     _plot_1d(data, labels, colors=colors, ax=ax)

    #     # Check that the colors are as expected
    #     for i, l in enumerate(labels):
    #         assert ax.get_lines()[i].get_color() == colors[l]

    # def test_plot_1d_ax_default():
    #     # create a DenseFunctionalData object
    #     x = np.linspace(0, 1, 101)
    #     y = np.sin(2 * np.pi * x)
    #     data = DenseFunctionalData({'input_dim_0': x}, y)

    #     # call the _plot_1d function with default ax argument
    #     ax = _plot_1d(data, np.zeros_like(y))

    #     # check that ax is not None
    #     assert ax is not None

    #     # check that ax is an instance of Axes
    #     assert isinstance(ax, Axes)

    # def test_plot_1d_with_provided_ax():
    #     # Create some test data
    #     x = np.linspace(0, 1, 11)
    #     y = np.random.randn(11, 5)
    #     labels = np.array([0, 1, 2, 3, 4])

    #     # Create a new figure and axes object
    #     fig, ax = plt.subplots()

    #     # Call the plotting function with the provided axes object
    #     _plot_1d(DenseFunctionalData(x, y), labels=labels, ax=ax)

    #     # Make sure the data was plotted correctly
    #     assert_array_equal(ax.lines[0].get_xdata(), x)
    #     assert_array_equal(ax.lines[0].get_ydata(), y[:, 0])
    #     assert_array_equal(ax.lines[1].get_xdata(), x)
    #     assert_array_equal(ax.lines[1].get_ydata(), y[:, 1])
    #     assert_array_equal(ax.lines[2].get_xdata(), x)
    #     assert_array_equal(ax.lines[2].get_ydata(), y[:, 2])
    #     assert_array_equal(ax.lines[3].get_xdata(), x)
    #     assert_array_equal(ax.lines[3].get_ydata(), y[:, 3])
    #     assert_array_equal(ax.lines[4].get_xdata(), x)
    #     assert_array_equal(ax.lines[4].get_ydata(), y[:, 4])

    # def test_plot_1d_no_labels(self):
    #     # create a DenseFunctionalData object with random data
    #     x = np.linspace(0, 1, num=10)
    #     y = np.random.randn(5, 10)
    #     fd = DenseFunctionalData(x, y)

    #     # create a figure and call the function without labels
    #     fig, ax = plt.subplots()
    #     _ = _plot_1d(fd, ax=ax)

    #     # check that the plot has the expected number of lines
    #     lines = ax.get_lines()
    #     assert len(lines) == fd.n_samples

    #     # check that the plot doesn't have a legend
    #     assert ax.get_legend() is None

    # def test_plot_1d_with_labels(self):
    #     # create a DenseFunctionalData object
    #     data = DenseFunctionalData(
    #         values=np.array([
    #             [1, 2, 3],
    #             [2, 3, 4],
    #             [3, 4, 5],
    #             [4, 5, 6],
    #         ]),
    #         argvals={'input_dim_0': np.array([0, 1, 2])}
    #     )
    #     # define labels
    #     labels = np.array([0, 1, 0, 1])

    #     # plot the data with labels
    #     fig, ax = plt.subplots()
    #     plot_1d(data, labels=labels, ax=ax)

    #     # check that the plot has the correct number of lines
    #     assert len(ax.lines) == len(np.unique(labels))

    #     # check that each line has the correct color
    #     for line, label in zip(ax.lines, labels):
    #         assert line.get_color() == mpl.cm.jet(label / (len(np.unique(labels)) - 1))

    #     # check that the legend is correct
    #     handles, labels = ax.get_legend_handles_labels()
    #     assert len(handles) == len(np.unique(labels))
    #     for handle, label in zip(handles, labels):
    #         assert handle.get_color() == mpl.cm.jet(label / (len(np.unique(labels)) - 1))
