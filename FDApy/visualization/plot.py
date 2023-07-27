#!/usr/bin/env python
# -*-coding:utf8 -*
"""Module for the plotting of functional data.

This module is used to create plot of diverse type of functional data.
Currently, there is an implementation for dense and irregular functional
data.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from matplotlib.axes import Axes
from typing import Optional, List, Union

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
    labels: npt.NDArray[np.float64],
    colors: Optional[npt.NDArray[np.float64]] = None,
    ax: Optional[Axes] = None,
    **plt_kwargs
) -> Axes:
    """Plot one dimensional functional data.

    This function is used to plot an instance of functional data in 1D.

    Parameters
    ----------
    data: Union[DenseFunctionalData, IrregularFunctionalData]
        The object to plot.
    labels: npt.NDArray[np.float64], default=None
        The labels of each curve.
    colors: Optional[npt.NDArray[np.float64]], default=None
        Colors used for the plot. If `colors` is `None`, it uses the `jet`
        colormaps from the `matplotlib` library by default.
    ax: Optional[Axes]
        Axes object onto which the objects are plotted.
    **plt_kwargs
        Keywords plotting arguments

    Returns
    -------
    Axes
        Axes objects onto the plot is done.

    """
    if colors is None:
        colors = mpl.cm.jet(np.linspace(0, 1, len(np.unique(labels))))

    if isinstance(data, DenseFunctionalData):
        for obs, l in zip(data.values, labels):
            ax.plot(
                data.argvals['input_dim_0'], obs, c=colors[l], **plt_kwargs
            )
    elif isinstance(data, IrregularFunctionalData):
        for argval, value, l in zip(
            data.argvals.values(),
            data.values.values(),
            labels
        ):
            ax.plot(
                argval['input_dim_0'], value, c=colors[l], **plt_kwargs
            )
            ax.scatter(
                argval['input_dim_0'], value, c=[colors[l]], **plt_kwargs
            )
    else:
        raise TypeError('Data type not recognized!')
    return ax


def _plot_2d(
    data: Union[DenseFunctionalData, IrregularFunctionalData],
    labels: npt.NDArray[np.float64],
    colors: Optional[npt.NDArray[np.float64]] = None,
    ax: Optional[Axes] = None,
    **plt_kwargs
) -> Axes:
    """Plot two dimensional functional data.

    This function is used to plot an instance of functional data in 2D.

    Parameters
    ----------
    data: Union[DenseFunctionalData, IrregularFunctionalData]
        The object to plot.
    labels: npt.NDArray[np.float64], default=None
        The labels of each curve.
    colors: Optional[npt.NDArray[np.float64]], default=None
        Colors used for the plot. If `colors` is `None`, it uses the `jet`
        colormaps from the `matplotlib` library by default.
    ax: Optional[Axes]
        Axes object onto which the objects are plotted.
    **plt_kwargs:
        Keywords plotting arguments

    Returns
    -------
    Axes
        Axes objects onto the plot is done.

    """
    if colors is None:
        colors = mpl.cm.jet(np.linspace(0, 1, len(np.unique(labels))))

    if isinstance(data, DenseFunctionalData):
        if data.n_obs == 1:
            cs = ax.contourf(
                data.argvals['input_dim_1'],
                data.argvals['input_dim_0'],
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
                ax.plot_surface(x, y, obs, color=colors[l], **plt_kwargs)
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
    labels: Optional[npt.NDArray[np.float64]] = None,
    colors: Optional[npt.NDArray[np.float64]] = None,
    ax: Optional[Axes] = None,
    **plt_kwargs
) -> Axes:
    """Plot function for univariate functional data.

    Generic plot function for DenseFunctionalData and IrregularFunctionalData
    objects.

    Parameters
    ----------
    data: Union[DenseFunctionalData, IrregularFunctionalData]
        The object to plot.
    labels: Optional[npt.NDArray[np.float64]], default=None
        The labels of each curve.
    colors: Optional[npt.NDArray[np.float64]], default=None
        Colors used for the plot. If `colors` is `None`, it uses the `jet`
        colormaps from the `matplotlib` library by default.
    ax: Optional[Axes]
        Axes object onto which the objects are plotted.
    **plt_kwargs:
        Keywords plotting arguments

    Returns
    -------
    Axes
        Axes objects onto the plot is done.

    """
    if labels is None:
        labels = np.arange(data.n_obs)
    if data.n_dimension == 1:
        ax = _init_ax(ax, projection='rectilinear')
        ax = _plot_1d(data, labels, colors, ax, **plt_kwargs)
    elif data.n_dimension == 2:
        if data.n_obs == 1:
            ax = _init_ax(ax, projection='rectilinear')
        else:
            ax = _init_ax(ax, projection='3d')
        ax = _plot_2d(data, labels, colors, ax, **plt_kwargs)
    else:
        raise ValueError(
            f"Can not plot functions of dimension {data.n_dimension},"
            " limited to dimension 2."
        )
    return ax


def plot_multivariate(
    data: MultivariateFunctionalData,
    labels: Optional[npt.NDArray[np.float64]] = None,
    titles: Optional[List[str]] = None,
    colors: Optional[npt.NDArray[np.float64]] = None,
    ax: Optional[Axes] = None,
    **plt_kwargs
) -> List[Axes]:
    """Plot function for multivariate functional data.

    Generic plot function for MultivariateFunctionalData objects.

    Parameters
    ----------
    data: MultivariateFunctional
        The object to plot.
    labels: Optional[npt.NDArray[np.float64]], default=None
        The labels of each curve.
    titles: Optional[List[np.str_]], default=None
        Titles of the subfigure.
    colors: Optional[npt.NDArray[np.float64]], default=None
        Colors used for the plot. If `colors` is `None`, it uses the `jet`
        colormaps from the `matplotlib` library by default.
    ax: Optional[Axes]
        Axes object onto which the objects are plotted.
    **plt_kwargs:
        Keywords plotting arguments

    Returns
    -------
    List[Axes]
        Axes objects onto the plot is done.

    """
    # Get parameters
    ncols = plt_kwargs.get("ncols", 2)
    nrows = data.n_functional // ncols + (data.n_functional % ncols > 0)

    if titles is None:
        titles = data.n_functional * [""]

    # Set spacing of the plots
    plt.subplots_adjust(wspace=0.7, hspace=0.2)

    axes = []
    for n, data in enumerate(data.data):
        ax = plt.subplot(nrows, ncols, n + 1)
        ax.set_title(titles[n])
        if data.n_dimension == 1:
            axes.append(plot(data, labels=labels, colors=colors, ax=ax))
        elif data.n_dimension == 2:
            axes.append(plot(data[0], labels=labels, colors=colors, ax=ax))
        else:
            raise ValueError(
                f"Can not plot functions of dimension {data.n_dimension},"
                " limited to dimension 2."
            )
    return axes
