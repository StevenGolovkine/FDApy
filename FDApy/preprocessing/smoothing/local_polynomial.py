#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Local Polynomials
-----------------

TODO: Rename most of the parameters. Propose somehting to estimate the
bandwith automatically. Parallelize the compuation. Update the references.

"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import Callable, Union

from sklearn.preprocessing import PolynomialFeatures


##############################################################################
# Inner functions for the LocalPolynomial class.

def _gaussian(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Compute the Gaussian density with mean 0 and standard deviation 1.

    The Gaussian density is given by

    .. math::
        K(x) = \frac{\exp(-x^2/2)}{\sqrt{2 \pi}}.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape = (n_samples,)
        Array at which computes the gaussian density.

    Returns
    -------
    npt.NDArray[np.float64], shape = (n_samples,)
        Values of the kernel.

    """
    return np.exp(- x**2 / 2) / np.sqrt(2 * np.pi)  # type: ignore


def _epanechnikov(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Compute the Epanechnikov kernel.

    The Epanechnikov kernel is given, in [1]_ equation 6.4, by

    .. math::
        K(x) = \frac{3}{4}(1 - x^2)  \text{if } |x| \leq 1.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape = (n_samples,)
        Array on which computes the Epanechnikov kernel.

    Returns
    -------
    npt.NDArray[np.float64], shape = (n_samples,)
        Values of the kernel.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.

    """
    kernel = np.zeros(x.shape)
    idx = np.where(np.abs(x) <= 1)
    kernel[idx] = 0.75 * (1 - x[idx]**2)
    return kernel


def _tri_cube(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Compute the tri-cube kernel.

    The tri-cube kernel is given, in [1]_ equation 6.6, by

    .. math::
        K(x) = (1 - |x|^3)^3  \text{if } |x| \leq 1.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape = (n_samples,)
        Array on which computes the tri-cube kernel

    Returns
    -------
    npt.NDArray[np.float64], shape = (n_samples,)
        Values of the kernel.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.

    """
    kernel = np.zeros(x.shape)
    idx = np.where(np.abs(x) < 1)
    kernel[idx] = (1 - np.abs(x[idx])**3)**3
    return kernel


def _bi_square(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Compute the bi-square kernel.

    The bi-square kernel is given, in [1]_, by

    .. math::
        K(x) = (1 - x^2)^2  \text{if } |x| \leq 1.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape = (n_samples,)
        Array on which computes the bi-square kernel

    Returns
    -------
    npt.NDArray[np.float64], shape = (n_samples,)
        Values of the kernel.

    References
    ----------
    ..[1] Cleveland W. (1979) Robust Locally Weighted Regression and Smoothing
        Scatterplots. Journal of the American Statistical Association,
        74(368): 829--836.

    """
    kernel = np.zeros(x.shape)
    idx = np.where(np.abs(x) < 1)
    kernel[idx] = (1 - x[idx]**2)**2
    return kernel


def _kernel(
    name: np.str_
) -> Callable:
    """Map between kernel names and functions.

    Parameters
    ----------
    name: np.str_
        Name of the kernel.

    Returns
    -------
    Callable
        The kernel function.

    """
    if name == 'gaussian':
        return _gaussian
    elif name == 'epanechnikov':
        return _epanechnikov
    elif name == 'tricube':
        return _tri_cube
    elif name == 'bisquare':
        return _bi_square
    else:
        raise NotImplementedError(
            f'The kernel {name} is not implemented'
        )


def _compute_kernel(
    x: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    bandwidth: Union[np.float64, npt.NDArray[np.float64]],
    kernel_name: np.str_ = 'gaussian'
) -> npt.NDArray[np.float64]:
    """Compute the kernel at a given point.

    The kernel is computed at points ||x - x0|| / h defined in [1]_ equation
    6.13. The norm used is the Euclidean norm.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape = (n_dim, n_samples)
        Training data.
    x0: npt.NDArray[np.float64], shape = (n_dim, )
        Number around which compute the kernel.
    bandwidth: Union[np.float64, npt.NDArray[np.float64]], shape = (n_samples,)
        Bandwidth to control the importance of points far from x0.
    kernel_name : np.str_, default='gaussian'
        Kernel name used.

    Returns
    -------
    npt.NDArray[np.float64] , shape = (n_samples,)
        Values of the kernel.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.

    """
    if not np.iterable(x0):
        x0 = np.asarray([x0])
    if x.ndim != np.size(x0):
        raise ValueError('x and x0 do not have the same dimension!')

    t = np.sqrt(np.sum(np.power(x - x0[:, np.newaxis], 2), axis=0)) / bandwidth

    kernel = _kernel(kernel_name)
    return kernel(t)


def _loc_poly(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    design_matrix: npt.NDArray[np.float64],
    design_matrix_x0: npt.NDArray[np.float64],
    kernel_name: np.str_ = 'epanechnikov',
    bandwidth: Union[np.float64, npt.NDArray[np.float64]] = 0.05
) -> np.float64:
    r"""Local polynomial regression for one point.

    Let :math:`(x_1, Y_1), ..., (x_n, Y_n)` be a random sample of bivariate
    data. Assume the following model: :math:`Y_i = f(x_i) + e_i`. We would
    like to estimate the unknown regression function
    :math:`f(x) = E[Y | X = x]`. We approximate :math:`f(x)` using Taylor
    series.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape = (n_dim, n_samples)
        Input array.
    y: npt.NDArray[np.float64], shape = (n_samples, )
        1-D input array such that :math:`y = f(x) + e`.
    x0: npt.NDArray[np.float64], shape = (n_dim, )
        1-D array on which estimate the function f(x).
    design_matrix: npt.NDArray[np.float64], shape = (n_sample, degree + 1)
        Design matrix of the matrix x.
    design_matrix_x0: npt.NDArray[np.float64], shape = (n_dim, degree + 1)
        Design matrix of the observation point x0.
    kernel_name: np.str_, default='epanechnikov'
        Kernel name used as weight.
    bandwidth: Union[np.float64, npt.NDArray[np.float64]], default=0.05
        Bandwidth for the kernel trick.

    Returns
    -------
    np.float64
        Prediction of y0, which is f(x0).

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.
    .. [2] Zhang, J.-T. and Jianwei C. (2007) Statistical Inferences for
        Functional Data, The Annals of Statistics, 35(3), 1052--1079.

    """
    # Compute kernel.
    kernel = _compute_kernel(
        x=x, x0=x0, bandwidth=bandwidth, kernel_name=kernel_name
    )

    # Compute the estimation of f (and derivatives) at x0.
    temp = np.dot(design_matrix.T, np.diag(kernel))
    beta = np.dot(np.linalg.pinv(np.dot(temp, design_matrix)), np.dot(temp, y))

    return np.dot(design_matrix_x0, beta)  # type: ignore


#############################################################################
# Class LocalPolynomial

class LocalPolynomial():
    r"""Local Polynomial Regression.

    This module implements Local Polynomial Regression over different
    dimensional domain [2]_, [3]_. The idea of local regression is to fit a
    (simple) different model separetely at each query point :math:`x_0`. Using
    only the observations close to :math:`x_0`, the resulting estimated
    function is smooth in the definition domain. Selecting observations close
    to :math:`x_0` is achieved via a weighted (kernel) function which assigned
    a weight to each observation based on its (euclidean) distance from the
    query point.

    Different kernels are defined (`gaussian`, `epanechnikov`, `tricube`,
    `bisquare`). Each of them has slightly different properties. Kernels are
    indexed by a parameter (`bandwith`) that controls the width of the
    neighborhood of :math:`x_0`. Note that the bandwidth can be adaptive and
    depend on :math:`x_0`.

    The degree of smoothing functions is controled using the `degree`
    parameter. A degree of 0 corresponds to locally constant, a degree of 1 to
    locally linear and a degree of 2 to locally quadratic, etc. High degrees
    can cause overfitting.

    The implementation is adapted from [4]_.

    Parameters
    ----------
    kernel_name: np.str_, default="gaussian"
        Kernel name used as weight (`gaussian`, `epanechnikov`, `tricube`,
        `bisquare`).
    bandwidth: np.float64, default=0.05
        Strictly positive. Control the size of the associated neighborhood.
    degree: np.int64, default=1
        Degree of the local polynomial to fit. If `degree = 0`, we fit the
        local constant estimator (equivalent to the Nadaraya-Watson estimator).
        If `degree = 1`, we fit the local linear estimator. If `degree = 2`, we
        fit the local quadratic estimator.
    robust: np.bool_
        Whether to apply the robustification procedure from [1]_, page 831.

    Attributes
    ----------
    kernel: Callable
        Function associated to the kernel name.
    poly_features: PolynomialFeatures
        An instance of `sklearn.preprocessing.PolynomialFeatures` used to
        create design matrices.

    References
    ----------
    .. [1] Cleveland W. (1979) Robust Locally Weighted Regression and Smoothing
        Scatterplots. Journal of the American Statistical Association,
        74(368): 829--836.
    .. [2] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.
    .. [3] Zhang, J.-T. and Jianwei C. (2007) Statistical Inferences for
        Functional Data, The Annals of Statistics, 35(3), 1052--1079.
    .. [4] https://github.com/arokem/lowess/blob/master/lowess/lowess.py

    """

    def __init__(
        self,
        kernel_name: np.str_ = "gaussian",
        bandwidth: np.float64 = 0.05,
        degree: np.int64 = 1,
        robust: np.bool_ = False
    ) -> None:
        """Initialize LocalPolynomial object."""
        self.kernel_name = kernel_name
        self.bandwidth = bandwidth
        self.degree = degree
        self.robust = robust

    @property
    def kernel_name(self) -> np.str_:
        """Getter for `kernel_name`."""
        return self._kernel_name

    @kernel_name.setter
    def kernel_name(self, new_kernel_name: np.str_) -> None:
        self._kernel_name = new_kernel_name
        self._kernel = _kernel(new_kernel_name)

    @property
    def bandwidth(self) -> np.float64:
        """Getter for `bandwidth`."""
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, new_bandwidth: np.float64) -> None:
        if new_bandwidth <= 0:
            raise ValueError('Bandwidth parameter must be strictly positive.')
        self._bandwidth = new_bandwidth

    @property
    def degree(self) -> np.float64:
        """Getter for `degree`."""
        return self._degree

    @degree.setter
    def degree(self, new_degree: np.int64) -> None:
        if new_degree < 0:
            raise ValueError('Degree parameter must be positive.')
        self._degree = new_degree
        self._poly_features = PolynomialFeatures(degree=new_degree)

    @property
    def robust(self) -> np.bool_:
        """Getter for `robust`."""
        return self._robust

    @robust.setter
    def robust(self, new_robust: np.bool_) -> None:
        self._robust = new_robust

    @property
    def kernel(self) -> Callable:
        """Getter for `kernel`."""
        return self._kernel

    @property
    def poly_features(self) -> PolynomialFeatures:
        """Getter for `poly_features`."""
        return self._poly_features

    def fit(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64]
    ) -> LocalPolynomial:
        """Fit local polynomial regression.

        Parameters
        ----------
        x: npt.NDArray[np.float64], shape = (n_dim, n_samples)
            Training data, input array.
        y: npt.NDArray[np.float64], shape = (n_samples,)
            Target values, 1-D input array

        Returns
        -------
        LocalPolynomial
            Returns an instance of self.

        TODO: Change this, it should not return that.

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

        x_fit = [
            _loc_poly(self.x, self.y, i, design_matrix, j, self.kernel_name, h)
            for (i, j, h) in zip(x0.T, design_matrix_x0, bandwidth)
        ]
        self.X_fit_ = np.array(x_fit)
        return self

    def predict(
        self,
        x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Predict using local polynomial regression.

        Parameters
        ----------
        x: npt.NDArray[np.float64], shape = (n_dim, n_samples)
            Data

        Returns
        -------
        npt.NDArray[np.float64], shape = (n_samples,)
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

        y_pred = [
            _loc_poly(self.x, self.y, i, design_matrix, j, self.kernel_name, h)
            for (i, j, h) in zip(x.T, design_matrix_x0, bandwidth)
        ]
        return np.array(y_pred)

    def fit_predict(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        x_pred: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Fit the model using `x` and predict on `x_pred`.

        Parameters
        ----------
        x: npt.NDArray[np.float64], shape = (n_dim, n_samples)
            Training data, input array
        y: npt.NDArray[np.float64], shape = (n_sample,)
            Target values, 1-D input array
        x_pred: npt.NDArray[np.float64], shape = (n_dim, n_samples2)
            Data to predict

        Returns
        -------
        npt.NDArray[np.float64], shape = (n_samples2,)
            Return predicted values.

        """
        self.x = x
        self.y = y

        if isinstance(x_pred, (int, float, np.int_, np.float_)):
            x_pred = [x_pred]

        if not np.iterable(self.bandwidth):
            bandwidth = np.repeat(
                self.bandwidth, np.size(x_pred) // np.ndim(x_pred)
            )

        design_matrix = self.poly_features.\
            fit_transform(np.array(self.x, ndmin=2).T)
        design_matrix_x0 = self.poly_features.\
            fit_transform(np.array(x_pred, ndmin=2).T)

        y_pred = [
            _loc_poly(self.x, self.y, i, design_matrix, j, self.kernel_name, h)
            for (i, j, h) in zip(x_pred.T, design_matrix_x0, bandwidth)
        ]

        return np.array(y_pred)
