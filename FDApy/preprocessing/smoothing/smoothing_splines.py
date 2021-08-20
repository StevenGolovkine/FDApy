#!/usr/bin/env python
# -*-coding:utf8 -*

"""Module for SmoothingSplines classes.

This module is used to perform smoothing spline. It is a wrapper around the
package csaps (https://csaps.readthedocs.io/en/latest/index.html).
"""
from csaps import csaps


###############################################################################
# Class SmoothingSplines


class SmoothingSpline():
    r"""Smoothing Spline.

    This class implements cubic smoothing splines algortihm proposed by Carl
    de Boor in his book "A Practical Guide to Splines". It is a wrapper around
    the python package `csaps`.

    The smoothing spline :math:`f` minimizes

    :math:`p\sum_{j = 1}^n w_j\lvert y_j - f(x_j) \rvert^2 +
    (1 - p)\int \lambda(t)\lvert D^2f(t)\rvert^2dt`.

    The smoothing parameter :math:`p` should be in :math:`[0, 1]`.

    Parameters
    ----------
    smooth: float, default=None
        The smoothing factor values. Can be a list for multidimensional
        smoothing. Should be in the range :math:`[0, 1]`. If None, the
        smoothing parameter will be computed automatically.

    References
    ----------
    * C. de Boor, A Practical Guide to Splines, Springer-Verlad, 1978.
    * csaps, https://github.com/espdev/csaps

    """

    def __init__(self, smooth=None):
        """Initialize SmoothingSpline object."""
        self.smooth = smooth

    @property
    def smooth(self):
        return self._smooth

    @smooth.setter
    def smooth(self, new_smooth):
        self._smooth = new_smooth

    def fit(self, x, y):
        """Fit smoothing spline.

        Parameters
        ----------
        x: array-like
            Training data, input array.
        y: array-like
            Target values

        Returns
        -------
        self: returns an instance of self.

        """
        self.x = x
        self.y = y
        self.model = csaps(x, y, smooth=self.smooth)
        return self

    def predict(self, x):
        """Predict using smoothing splines.

        Parameters
        ----------
        x: array-like
            Data

        Returns
        -------
        y_pred: array-like
            Return predicted values.

        """
        return self.model(x)

    def fit_predict(self, x, y, x_pred=None):
        """Fit the model using `x` and predict on `x_pred`.

        Parameters
        ----------
        x: array-like
            Training data, input array.
        y: array-like
            Target values
        x_pred: array-like
            Data to predict

        Returns
        -------
        y_pred: array-like
            Return predicted values.

        """
        if x_pred is None:
            x_pred = x
        return self.fit(x, y).predict(x_pred)
