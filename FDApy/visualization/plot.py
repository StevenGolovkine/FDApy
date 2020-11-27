#!/usr/bin/python3.7
# -*-coding:utf8 -*
"""
Module for the plotting of functional data.

This module is used to create plot of diverse type of functional data.
Currently, there is an implementation for dense and irregular functional
data.
"""
import matplotlib.pyplot as plt
import numpy as np

from ..representation.functional_data import (DenseFunctionalData,
                                              IrregularFunctionalData)


def _init_ax(ax=None, projection='rectilinear'):
    """Initialize axes."""
    if ax is None:
        ax = plt.gca(projection=projection)
    return ax


def plot(data, ax=None, **plt_kwargs):
    """Plot function.

    Generic plot function for DenseFunctionalData and IrregularFunctionalData
    objects.

    Parameters
    ----------
    data: UnivariateFunctionalData, IrregularFunctionalData
        The object to plot.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes object onto which the objects are plotted.
    **plt_kwargs:
        Keywords plotting arguments

    Returns
    -------
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes objects onto the plot is done.

    """
    if data.n_dim == 1:
        ax = _init_ax(ax, projection='rectilinear')
        ax = _plot_1d(data, ax, **plt_kwargs)
    elif data.n_dim == 2:
        if data.n_obs == 1:
            ax = _init_ax(ax, projection='rectilinear')
        else:
            ax = _init_ax(ax, projection='3d')
        ax = _plot_2d(data, ax, **plt_kwargs)
    else:
        raise ValueError(f"Can not plot functions of dimension {data.n_dim},"
                         " limited to dimension 2.")
    return ax


def _plot_1d(data, ax=None, **plt_kwargs):
    """Plot one dimensional functional data.

    This function is used to plot an instance of functional data in 1D.

    Parameters
    ----------
    data: UnivariateFunctionalData or IrregularFunctionalData
        The object to plot.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes object onto which the objects are plotted.
    **plt_kwargs:
        Keywords plotting arguments

    Returns
    -------
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes objects onto the plot is done.

    """
    if isinstance(data, DenseFunctionalData):
        for obs in data.values:
            ax.plot(data.argvals['input_dim_0'], obs, **plt_kwargs)
    elif isinstance(data, IrregularFunctionalData):
        for argval, value in zip(data.argvals['input_dim_0'].values(),
                                 data.values.values()):
            ax.scatter(argval, value, **plt_kwargs)
    else:
        raise TypeError('Data type not recognized!')
    return ax


def _plot_2d(data, ax=None, **plt_kwargs):
    """Plot two dimensional functional data.

    This function is used to plot an instance of functional data in 2D.

    Parameters
    ----------
    data: IrregularFunctionalData
        The object to plot.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes object onto which the objects are plotted.
    **plt_kwargs:
        Keywords plotting arguments

    Returns
    -------
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes objects onto the plot is done.

    """
    if isinstance(data, DenseFunctionalData):
        if data.n_obs == 1:
            cs = ax.contourf(data.argvals['input_dim_0'],
                             data.argvals['input_dim_1'],
                             data.values.squeeze(),
                             **plt_kwargs)
            plt.colorbar(cs)
        else:
            x, y = np.meshgrid(data.argvals['input_dim_0'],
                               data.argvals['input_dim_1'],
                               indexing='ij')
            for obs in data.values:
                ax.plot_surface(x, y, obs, **plt.kwargs)
    elif isinstance(data, IrregularFunctionalData):
        raise NotImplementedError("Currently 2d irregular functional data"
                                  " plotting is not implemented.")
    else:
        raise TypeError("Data type not recognized!")
    return ax
