#!/usr/bin/python3.7
# -*-coding:utf8 -*
"""
Module for the plotting of functional data.

This module is used to create plot of diverse type of functional data.
Currently, there is an implementation for univariate and irregular functional
data. The multivariate case is also treated as list of univariate and/or
irregular functional data.
"""
import matplotlib.pyplot as plt
import numpy as np

from .univariate_functional import UnivariateFunctionalData
from .irregular_functional import IrregularFunctionalData
from .multivariate_functional import MultivariateFunctionalData


def plot(data, main="", xlab="", ylab=""):
    """
    Plot function.

    Generic plot function for univariate, irregular and multivariate
    functional data.

    Parameters
    ----------
    data : UnivariateFunctionalData, IrregularFunctionalData or
    MultivariateFunctionalData
        The object to plot.
    main : str
        Title of the graph.
    xlab : str or list of str
        Label of the X axis.
    ylab : str or list of str
        Label of the Y axis.

    Returns
    -------
    fig, ax : elements for plotting using matplotlib

    """
    if isinstance(data, UnivariateFunctionalData):
        fig, ax = _plot_univariate(data, main, xlab, ylab)
    elif isinstance(data, IrregularFunctionalData):
        fig, ax = _plot_irregular(data, main, xlab, ylab)
    elif isinstance(data, MultivariateFunctionalData):
        fig, ax = _plot_multivariate(data, main, xlab, ylab)
    else:
        raise ValueError(
            """Data has to be elements of UnivariateFunctionalData or
            IrregularFunctionalData or MultivariateFunctionalData!""")

    return fig, ax


def _plot_univariate(data, main="", xlab="", ylab=""):
    """
    Plot univariate functional data.

    This function is used to plot an instance of univariate functional data.

    Parameters
    ----------
    data : UnivariateFunctionalData.
        The object to plot.
    main : str
        Title of the graph.
    xlab : str
        Label of the X axis.
    ylab : str
        Label of the Y axis.

    Returns
    -------
    fig, ax : elements for ploting using matplotlib

    """
    fig, ax = plt.subplots(1, 1)

    ax.set_title(main)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if data.nObs() == 1 and data.dimension() == 2:
        p = ax.contour(
            data.argvals[0],
            data.argvals[1],
            np.squeeze(data.values).T)
        plt.clabel(p, inline=1)
    else:
        for obs in data.values:
            ax.plot(data.argvals[0], obs)

    return fig, ax


def _plot_irregular(data, main="", xlab="", ylab=""):
    """
    Plot irregular functional data.

    This function is used to plot an instance of irregular functional data.

    Parameters
    ----------
    data : IrregularFunctionalData.
        The object to plot.
    main : str
        Title of the graph.
    xlab : str
        Label of the X axis.
    ylab : str
        Label of the Y axis.

    Returns
    -------
    fig, ax : elements for ploting using matplotlib

    """
    fig, ax = plt.subplots(1, 1)

    ax.set_title(main)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if np.mean(data.nObsPoint()) > 20:
        for fd in data:
            ax.plot(np.array(fd.argvals[0]), fd.values[0])
    else:
        for fd in data:
            ax.scatter(np.array(fd.argvals[0]), fd.values[0])

    return fig, ax


def _plot_multivariate(data, main="", xlab="", ylab=""):
    """
    Plot multivariate functional data.

    This function is used to plot an instance of multivariate functional data.

    Parameters
    ----------
    data : MultivariateFunctionalData.
        The object to plot.
    main : str or list of str
        Title of the graph.
    xlab : str or list of str
        Label of the X axis.
    ylab : str or list of str
        Label of the Y axis.

    Returns
    -------
    figs, axes : elements for ploting using matplotlib

    """
    nFunc = data.nFunctions()

    if isinstance(main, list) and len(main) != nFunc:
        raise ValueError(
            'The parameter `main` has not the right length!')
    if isinstance(xlab, list) and len(xlab) != nFunc:
        raise ValueError(
            'The parameter `xlab` has not the right length!')
    if isinstance(ylab, list) and len(ylab) != nFunc:
        raise ValueError(
            'The parameter `ylab` has not the right length!')

    if isinstance(main, str):
        main = [main for i in range(nFunc)]
    if isinstance(xlab, str):
        xlab = [xlab for i in range(nFunc)]
    if isinstance(ylab, str):
        ylab = [ylab for i in range(nFunc)]

    figs = []
    axes = []

    for i, func in enumerate(data):

        if isinstance(func, UnivariateFunctionalData):
            fig, ax = _plot_univariate(func, main[i], xlab[i], ylab[i])
        elif isinstance(func, IrregularFunctionalData):
            fig, ax = _plot_irregular(func, main[i], xlab[i], ylab[i])
        else:
            raise ValueError(
                """Data has to be elements of UnivariateFunctionalData
                or IrregularFunctionalData!""")
        figs.append(fig)
        axes.append(ax)
    return figs, axes
