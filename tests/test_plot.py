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
