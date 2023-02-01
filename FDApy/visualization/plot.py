#!/usr/bin/env python
# -*-coding:utf8 -*
"""
Module for the plotting of functional data.

This module is used to create plot of diverse type of functional data.
Currently, there is an implementation for dense and irregular functional
data.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from matplotlib.axes import Axes
from typing import Optional, Union

from ..representation.functional_data import (
    DenseFunctionalData, IrregularFunctionalData, MultivariateFunctionalData
)


#############################################################################
# Utility functions
def _init_ax(
    ax: Optional[Axes] = None,
    projection: str = 'rectilinear'
) -> Axes:
    """Initialize axes."""
    if ax is None:
        ax = plt.axes(projection=projection)
    return ax


#############################################################################
# Utility functions
def _plot_1d(
    data: Union[DenseFunctionalData, IrregularFunctionalData],
    labels: npt.NDArray,
    ax: Axes = None,
    **plt_kwargs
) -> Axes:
    """Plot one dimensional functional data.

    This function is used to plot an instance of functional data in 1D.

    Parameters
    ----------
    data: UnivariateFunctionalData or IrregularFunctionalData
        The object to plot.
    labels: np.array, default=None
        The labels of each curve.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes object onto which the objects are plotted.
    **plt_kwargs:
        Keywords plotting arguments

    Returns
    -------
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes objects onto the plot is done.

    """
    COLORS = mpl.cm.jet(np.linspace(0, 1, len(np.unique(labels))))
    if isinstance(data, DenseFunctionalData):
        for obs, l in zip(data.values, labels):
            ax.plot(
                data.argvals['input_dim_0'], obs, c=COLORS[l], **plt_kwargs
            )
    elif isinstance(data, IrregularFunctionalData):
        for argval, value, l in zip(
            data.argvals['input_dim_0'].values(),
            data.values.values(),
            labels
        ):
            ax.plot(
                argval, value, c=COLORS[l], **plt_kwargs
            )
            ax.scatter(argval, value, c=[COLORS[l]], **plt_kwargs)
    else:
        raise TypeError('Data type not recognized!')
    return ax


def _plot_2d(
    data: Union[DenseFunctionalData, IrregularFunctionalData],
    labels: npt.NDArray,
    ax: Axes = None,
    **plt_kwargs
) -> Axes:
    """Plot two dimensional functional data.

    This function is used to plot an instance of functional data in 2D.

    Parameters
    ----------
    data: IrregularFunctionalData
        The object to plot.
    labels: np.array, default=None
        The labels of each curve.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes object onto which the objects are plotted.
    **plt_kwargs:
        Keywords plotting arguments

    Returns
    -------
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes objects onto the plot is done.

    """
    COLORS = mpl.cm.jet(np.linspace(0, 1, len(np.unique(labels))))
    if isinstance(data, DenseFunctionalData):
        if data.n_obs == 1:
            cs = ax.contourf(
                data.argvals['input_dim_0'],
                data.argvals['input_dim_1'],
                data.values.squeeze(),
                **plt_kwargs
            )
            plt.colorbar(cs)
        else:
            x, y = np.meshgrid(
                data.argvals['input_dim_0'],
                data.argvals['input_dim_1'],
                indexing='ij'
            )
            for obs, l in zip(data.values, labels):
                ax.plot_surface(x, y, obs, color=COLORS[l], **plt_kwargs)
    elif isinstance(data, IrregularFunctionalData):
        raise NotImplementedError(
            "Currently 2d irregular functional data"
            " plotting is not implemented."
        )
    else:
        raise TypeError("Data type not recognized!")
    return ax


#############################################################################
# Plotting functions
def plot(
    data: Union[DenseFunctionalData, IrregularFunctionalData],
    labels: Optional[npt.NDArray] = None,
    ax: Axes = None,
    **plt_kwargs
) -> Axes:
    """Plot function for univariate functional data.

    Generic plot function for DenseFunctionalData and IrregularFunctionalData
    objects.

    Parameters
    ----------
    data: UnivariateFunctionalData, IrregularFunctionalData
        The object to plot.
    labels: np.array, default=None
        The labels of each curve.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes object onto which the objects are plotted.
    **plt_kwargs:
        Keywords plotting arguments

    Returns
    -------
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes objects onto the plot is done.

    """
    if labels is None:
        labels = np.arange(data.n_obs)
    if data.n_dim == 1:
        ax = _init_ax(ax, projection='rectilinear')
        ax = _plot_1d(data, labels, ax, **plt_kwargs)
    elif data.n_dim == 2:
        if data.n_obs == 1:
            ax = _init_ax(ax, projection='rectilinear')
        else:
            ax = _init_ax(ax, projection='3d')
        ax = _plot_2d(data, labels, ax, **plt_kwargs)
    else:
        raise ValueError(
            f"Can not plot functions of dimension {data.n_dim},"
            " limited to dimension 2."
        )
    return ax


def plot_multivariate(
    data: MultivariateFunctionalData,
    labels: Optional[npt.NDArray] = None,
    ax: Axes = None,
    **plt_kwargs
):
    """Plot function for multivariate functional data.

    Generic plot function for MultivariateFunctionalData objects.

    Parameters
    ----------
    data: MultivariateFunctional
        The object to plot.
    labels: np.array, default=None
        The labels of each curve.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes object onto which the objects are plotted.
    **plt_kwargs:
        Keywords plotting arguments

    Returns
    -------
    axes: list of matplotlib.axes._subplots.AxesSubplot
        Axes objects onto the plot is done.

    """
    # Get parameters
    ncols = plt_kwargs.get("ncols", 2)
    nrows = data.n_functional // ncols + (data.n_functional % ncols > 0)

    # Set spacing of the plots
    plt.subplots_adjust(wspace=0.7, hspace=0.2)

    axes = []
    for n, data in enumerate(data):
        ax = plt.subplot(nrows, ncols, n + 1)
        if data.n_dim == 1:
            axes.append(plot(data, labels=labels, ax=ax))
        elif data.n_dim == 2:
            axes.append(plot(data[0], labels=labels, ax=ax))
        else:
            raise ValueError(
                f"Can not plot functions of dimension {data.n_dim},"
                " limited to dimension 2."
            )
    return axes
