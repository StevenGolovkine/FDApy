#!/usr/bin/env python
# -*-coding:utf8 -*

"""Module for LocalPolynomial classes.

This module is used to fit local polynomial regression.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import Union

from sklearn.preprocessing import PolynomialFeatures


##############################################################################
# Inner functions for the LocalPolynomial class.

def _gaussian(
    t: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the gaussian density with mean 0 and standard deviation 1.

    Parameters
    ----------
    t: array-like, shape = (n_samples,)
        Array at which computes the gaussian density

    Returns
    -------
    kernel: array-like, shape = (n_samples,)

    """
    return np.exp(- t**2 / 2) / np.sqrt(2 * np.pi)  # type: ignore


def _epanechnikov(
    t: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the Epanechnikov kernel.

    Parameters
    ----------
    t: array-like, shape = (n_samples,)
        Array on which computes the Epanechnikov kernel

    Returns
    -------
    kernel: array-like, shape = (n_samples,)

    References
    ----------
    Hastie, Tibshirani and Friedman, Elements of Statistical Learning, 2009,
    equation 6.4

    """
    kernel = np.zeros(t.shape)
    idx = np.where(np.abs(t) <= 1)
    kernel[idx] = 0.75 * (1 - t[idx]**2)
    return kernel


def _tri_cube(
    t: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the tri-cube kernel.

    Parameters
    ----------
    t: array-like, shape = (n_samples,)
        Array on which computes the tri-cube kernel

    Returns
    -------
    kernel: array-like, shape = (n_samples,)

    References
    ----------
    Hastie, Tibshirani and Friedman, Elements of Statistical Learning, 2009,
    equation 6.6

    """
    kernel = np.zeros(t.shape)
    idx = np.where(np.abs(t) < 1)
    kernel[idx] = (1 - np.abs(t[idx])**3)**3
    return kernel


def _bi_square(
    t: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the bi-square kernel.

    Parameters
    ----------
    t: array-like, shape = (n_samples,)
        Array on which computes the bi-square kernel

    Returns
    -------
    kernel: array-like, shape = (n_samples,)

    References
    ----------
    Cleveland, Robust Locally Weighted Regression and Smoothing Scatterplots,
    1979, p.831

    """
    kernel = np.zeros(t.shape)
    idx = np.where(np.abs(t) < 1)
    kernel[idx] = (1 - t[idx]**2)**2
    return kernel


def _compute_kernel(
    x: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    h: Union[float, npt.NDArray[np.float64]],
    kernel_name: str = 'gaussian'
) -> npt.NDArray[np.float64]:
    """Compute kernel at point norm(x - x0) / h.

    Parameters
    ----------
    x: array-like, shape = (n_dim, n_samples)
        Training data.
    x0: float-array, shape = (n_dim, )
        Number around which compute the kernel.
    h: float or float-array, shape = (n_samples, )
        Bandwidth to control the importance of points far from x0.
    kernel_name : string, default='gaussian'
        Kernel name used.

    Returns
    -------
    kernel: array-like , shape = (n_samples, )

    References
    ----------
    Hastie, Tibshirani and Friedman, Elements of Statistical Learning, 2009,
    equation 6.13

    """
    if not np.iterable(x0):
        x0 = np.asarray([x0])
    if x.ndim != np.size(x0):
        raise ValueError('x and x0 do not have the same dimension!')

    t = np.sqrt(np.sum(np.power(x - x0[:, np.newaxis], 2), axis=0)) / h

    if kernel_name == 'gaussian':
        kernel = _gaussian(t)
    elif kernel_name == 'epanechnikov':
        kernel = _epanechnikov(t)
    elif kernel_name == 'tricube':
        kernel = _tri_cube(t)
    elif kernel_name == 'bisquare':
        kernel = _bi_square(t)
    else:
        raise NotImplementedError(f'The kernel {kernel_name} is not'
                                  ' implemented')
    return kernel


def _loc_poly(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    design_matrix: npt.NDArray[np.float64],
    design_matrix_x0: npt.NDArray[np.float64],
    kernel_name: str = 'epanechnikov',
    h: float = 0.05
) -> float:
    r"""Local polynomial regression for one point.

    Let :math:`(x_1, Y_1), ..., (x_n, Y_n)` be a random sample of bivariate
    data. Assume the following model: :math:`Y_i = f(x_i) + e_i`. We would
    like to estimate the unknown regression function
    :math:`f(x) = E[Y | X = x]`. We approximate :math:`f(x)` using Taylor
    series.

    Parameters
    ----------
    x: array-like, shape = (n_dim, n_samples)
        Input array.
    y: array-like, shape = (n_samples, )
        1-D input array such that :math:`y = f(x) + e`.
    x0: array-like, shape = (n_dim, )
        1-D array on which estimate the function f(x).
    design_matrix: array-like, shape = (n_sample, degree + 1)
        Design matrix of the matrix x.
    design_matrix_x0: array-like, shape = (n_dim, degree + 1)
        Design matrix of the observation point x0.
    kernel_name: string, default='epanechnikov'
        Kernel name used as weight.
    h: float or float-array, default=0.05
        Bandwidth for the kernel trick.

    Returns
    -------
    y0_pred: float
        Prediction of y0, which is f(x0).

    References
    ----------
    Zhang and Chen, Statistical Inferences for functional data, The Annals of
    Statistics, 1052-1079, No. 3, Vol. 35, 2007.

    """
    # Compute kernel.
    kernel = _compute_kernel(x=x, x0=x0, h=h, kernel_name=kernel_name)

    # Compute the estimation of f (and derivatives) at x0.
    temp = np.dot(design_matrix.T, np.diag(kernel))
    beta = np.dot(np.linalg.pinv(np.dot(temp, design_matrix)), np.dot(temp, y))

    return np.dot(design_matrix_x0, beta)  # type: ignore

#############################################################################
# Class LocalPolynomial


class LocalPolynomial():
    r"""Local polynomial regression.

    Let :math:`(x_1, Y_1), ..., (x_n, Y_n)` be a random sample of bivariate
    data. For all :math:`i, x_i` belongs to :math:`\mathbb{R}^d` and
    :math:`Y_i` in :math:`\mathbb{R}`. Assume the following model:
    :math:`Y_i = f(x_i) + e_i`. We would like to estimate the unknown
    regression function :math:`f(x) = E[Y | X = x]`. We approximate :math`f(x)`
    using Taylor series.

    Parameters
    ----------
    kernel: string, default="gaussian"
        Kernel name used as weight (default = 'gaussian').
    bandwidth: float, default=0.05
        Strictly positive. Control the size of the associated neighborhood.
    degree: integer, default=2
        Degree of the local polynomial to fit. If degree = 0, we fit the local
        constant estimator (equivalent to the Nadaraya-Watson estimator). If
        degree = 1, we fit the local linear estimator.

    References
    ----------
    * Zhang and Chen, Statistical Inferences for functional data, The Annals of
    Statistics, 1052-1079, No. 3, Vol. 35, 2007.
    * https://github.com/arokem/lowess/blob/master/lowess/lowess.py

    """

    def __init__(
        self,
        kernel_name: str = "gaussian",
        bandwidth: float = 0.05,
        degree: int = 2
    ) -> None:
        """Initialize LocalPolynomial object."""
        # TODO: Add test on parameters.
        self.kernel_name = kernel_name
        self.bandwidth = bandwidth
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree)

    @property
    def kernel_name(self) -> str:
        """Getter for `kernel_name`."""
        return self._kernel_name

    @kernel_name.setter
    def kernel_name(self, new_kernel_name: str) -> None:
        self._kernel_name = new_kernel_name

    @property
    def bandwidth(self) -> float:
        """Getter for `bandwidth`."""
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, new_bandwidth: float) -> None:
        self._bandwidth = new_bandwidth

    @property
    def degree(self) -> float:
        """Getter for `degree`."""
        return self._degree

    @degree.setter
    def degree(self, new_degree: int) -> None:
        self._degree = new_degree
        self._poly_features = PolynomialFeatures(degree=new_degree)

    def fit(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64]
    ) -> LocalPolynomial:
        """Fit local polynomial regression.

        Parameters
        ----------
        x: array-like, shape = (n_dim, n_samples)
            Training data, input array.
        y: array-like, shape = (n_samples, )
            Target values, 1-D input array

        Returns
        -------
        self: returns an instance of self.

        """
        # TODO: Add tests on the parameters.
        self.x = x
        self.y = y

        x0 = np.unique(self.x, axis=0)
        if not np.iterable(self.bandwidth):
            bandwidth = np.repeat(self.bandwidth, np.size(x0) // np.ndim(x0))

        design_matrix = self.poly_features.\
            fit_transform(np.array(self.x, ndmin=2).T)
        design_matrix_x0 = self.poly_features.\
            fit_transform(np.array(x0, ndmin=2).T)

        self.X_fit_ = np.array([_loc_poly(self.x, self.y, i, design_matrix, j,
                                          self.kernel_name, h)
                                for (i, j, h) in zip(x0.T,
                                                     design_matrix_x0,
                                                     bandwidth)])
        return self

    def predict(
        self,
        x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Predict using local polynomial regression.

        Parameters
        ----------
        x: array-like, shape = (n_dim, n_samples)
            Data

        Returns
        -------
        y_pred: array-like, shape = (n_samples,)
            Return predicted values.

        """
        if isinstance(x, (int, float, np.int_, np.float_)):
            x = [x]

        if not np.iterable(self.bandwidth):
            bandwidth = np.repeat(self.bandwidth, np.size(x) // np.ndim(x))

        design_matrix = self.poly_features.\
            fit_transform(np.array(self.x, ndmin=2).T)
        design_matrix_x0 = self.poly_features.\
            fit_transform(np.array(x, ndmin=2).T)

        y_pred = np.array([_loc_poly(self.x, self.y, i, design_matrix, j,
                                     self.kernel_name, h)
                           for (i, j, h) in zip(x.T,
                                                design_matrix_x0,
                                                bandwidth)])

        return y_pred

    def fit_predict(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        x_pred: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Fit the model using `x` and predict on `x_pred`.

        Parameters
        ----------
        x: array-like, shape = (n_dim, n_samples)
            Training data, input array
        y: array-like, shape = (n_sample, )
            Target values, 1-D input array
        x_pred: array-like, shape = (n_dim, n_samples2)
            Data to predict

        Returns
        -------
        y_pred: array-like, shape = (n_samples2,)
            Return predicted values

        """
        self.x = x
        self.y = y

        if isinstance(x_pred, (int, float, np.int_, np.float_)):
            x_pred = [x_pred]

        if not np.iterable(self.bandwidth):
            bandwidth = np.repeat(self.bandwidth,
                                  np.size(x_pred) // np.ndim(x_pred))

        design_matrix = self.poly_features.\
            fit_transform(np.array(self.x, ndmin=2).T)
        design_matrix_x0 = self.poly_features.\
            fit_transform(np.array(x_pred, ndmin=2).T)

        y_pred = np.array([_loc_poly(self.x, self.y, i, design_matrix, j,
                                     self.kernel_name, h)
                           for (i, j, h) in zip(x_pred.T,
                                                design_matrix_x0,
                                                bandwidth)])

        return y_pred
