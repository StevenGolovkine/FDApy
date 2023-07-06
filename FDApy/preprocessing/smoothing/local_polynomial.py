#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Local Polynomials
-----------------

"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import Callable, Optional, Union

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
    return np.exp(- np.square(x) / 2) / np.sqrt(2 * np.pi)


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
    kernel[idx] = 0.75 * (1 - np.square(x[idx]))
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
        Array on which computes the tri-cube kernel.

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
    kernel[idx] = np.power(1 - np.power(np.abs(x[idx]), 3), 3)
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
        Array on which computes the bi-square kernel.

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
    kernel[idx] = np.square(1 - np.square(x[idx]))
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
    x0: Union[float, npt.NDArray[np.float64]],
    bandwidth: float,
    kernel: Callable = _epanechnikov
) -> npt.NDArray[np.float64]:
    r"""Compute the weights at a particular query point.

    The weights are defined using a kernel function and are computed at points
    :math:`\lvert\lvert x - x0\rvert\rvert / \lambda` defined in [1]_, equation
    6.13. The used norm is the Euclidean norm. The function allows
    multidimensional inputs. The arguments `x`, `x0` must have a common
    dimension.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape = (n_samples, n_dim)
        Training data.
    x0: Union[float, npt.NDArray[np.float64]], shape = (n_dim,)
        Query point. For one-dimensional smoothing, `x0` must be passed as a
        `float`. For higher-dimensional smoothing, `x0` must be passed as
        a `npt.NDArray[np.float64]]`.
    bandwidth: float
        Width of the neighborhood of `x0`.
    kernel: Callable, default=_epanechnikov
        Kernel function to used.

    Returns
    -------
    npt.NDArray[np.float64] , shape=(n_samples,)
        Values of the kernel.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.

    """
    if x.ndim == 1:
        xx = np.abs(x - x0)
    else:
        xx = np.linalg.norm(x - x0, axis=1)
    return kernel(xx / bandwidth)


def _local_regression(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    dmat: npt.NDArray[np.float64],
    dmat_x0: npt.NDArray[np.float64],
    bandwidth: float = 0.05,
    kernel: Callable = _epanechnikov
) -> float:
    r"""Local polynomial regression for one point.

    This function estimates the regression function :math:`f(x)` over the
    domain `\mathbb{R}^p` at a particular query point `x_0`. A generalized
    version of the equation (6.8) in [1]_ is implemented. It allows
    multidimensional inputs for `x` and `x_0`.

    Parameters
    ----------
    y: npt.NDArray[np.float64], shape = (n_samples,)
        Target values.
    x: npt.NDArray[np.float64], shape = (n_samples, n_dim)
        Training data.
    x0: Union[float, npt.NDArray[np.float64]], shape = (n_dim,)
        Query point. For one-dimensional smoothing, `x0` must be passed as a
        `float`. For higher-dimensional smoothing, `x0` must be passed as
        a `npt.NDArray[np.float64]]`.
    dmat: npt.NDArray[np.float64], shape = (n_sample, n_features)
        Design matrix for the training data `x`. The dimension `n_features` is
        related to the degree of the fitted polynomials. It includes intercept
        and interaction in the case of multidimensional inputs.
    dmat_x0: npt.NDArray[np.float64], shape = (n_dim, n_features)
        Design matrix for the query points `x_0`. The dimension `n_features`
        must be the same as the design matrix of `x`.
    bandwidth: float
        Width of the neighborhood of `x0`.
    kernel: Callable, default=_epanechnikov
        Kernel function to used.

    Returns
    -------
    float
        Estimation of `f(x_0)`.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.

    """
    kernel_values = _compute_kernel(
        x=x, x0=x0, bandwidth=bandwidth, kernel=kernel
    )
    temp = dmat.T * kernel_values
    beta = np.linalg.lstsq(
        np.dot(temp, dmat), np.dot(temp, y), rcond=1e-10
    )[0]

    return np.dot(dmat_x0, beta)


#############################################################################
# Class LocalPolynomial

class LocalPolynomial():
    r"""Local Polynomial Regression.

    This module implements Local Polynomial Regression over different
    dimensional domain [2]_. The idea of local regression is to fit a
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

    The implementation is adapted from [3]_.

    Parameters
    ----------
    kernel_name: np.str_, default="gaussian"
        Kernel name used as weight (`gaussian`, `epanechnikov`, `tricube`,
        `bisquare`).
    bandwidth: float, default=0.05
        Strictly positive. Control the size of the associated neighborhood.
    degree: int, default=1
        Degree of the local polynomial to fit. If ``degree = 0``, we fit the
        local constant estimator (equivalent to the Nadaraya-Watson estimator).
        If ``degree = 1``, we fit the local linear estimator. If
        ``degree = 2``, we fit the local quadratic estimator.
    robust: bool, default=False
        Whether to apply the robustification procedure from [1]_, page 831.

    Attributes
    ----------
    kernel: Callable
        Function associated to the kernel name.
    poly_features: PolynomialFeatures
        An instance of ``sklearn.preprocessing.PolynomialFeatures`` used to
        create design matrices. It includes an intercept and interactions for
        multidimensional inputs.

    Notes
    -----
    This methods is *memory-based* and thus require no training; all the work
    is performed at evaluation time [2]_. For now, no ``fit`` function is
    necessary and only a ``predict`` is implemented.

    References
    ----------
    .. [1] Cleveland W. (1979) Robust Locally Weighted Regression and Smoothing
        Scatterplots. Journal of the American Statistical Association,
        74(368): 829--836.
    .. [2] Hastie, T., Tibshirani, R., Friedman, J. (2009) The Elements of
        Statistical Learning: Data Mining, Inference, and Prediction,
        Second Edition, Springer Series in Statistics.
    .. [3] https://github.com/arokem/lowess/blob/master/lowess/lowess.py

    TODO
    ----
    Add robustification

    """

    def __init__(
        self,
        kernel_name: np.str_ = "gaussian",
        bandwidth: float = 0.05,
        degree: int = 1,
        robust: bool = False
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
    def bandwidth(self) -> float:
        """Getter for `bandwidth`."""
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, new_bandwidth: float) -> None:
        if new_bandwidth <= 0:
            raise ValueError('Bandwidth parameter must be strictly positive.')
        self._bandwidth = new_bandwidth

    @property
    def degree(self) -> float:
        """Getter for `degree`."""
        return self._degree

    @degree.setter
    def degree(self, new_degree: int) -> None:
        if new_degree < 0:
            raise ValueError('Degree parameter must be positive.')
        self._degree = new_degree
        self._poly_features = PolynomialFeatures(degree=new_degree)

    @property
    def robust(self) -> bool:
        """Getter for `robust`."""
        return self._robust

    @robust.setter
    def robust(self, new_robust: bool) -> None:
        self._robust = new_robust

    @property
    def kernel(self) -> Callable:
        """Getter for `kernel`."""
        return self._kernel

    @property
    def poly_features(self) -> PolynomialFeatures:
        """Getter for `poly_features`."""
        return self._poly_features

    def predict(
        self,
        y: npt.NDArray[np.float64],
        x: npt.NDArray[np.float64],
        x_new: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        """Predict using local polynomial regression.

        Parameters
        ----------
        y: npt.NDArray[np.float64], shape = (n_samples,)
            Target values.
        x: npt.NDArray[np.float64], shape = (n_samples, n_dim)
            Training data.
        x_new: Optional[npt.NDArray[np.float64]], default=None
            Query points at which estimates the function. If ``None``, the
            (unique) training data are used as query points. The shape of the
            array must be (n_points, n_dim).

        Returns
        -------
        npt.NDArray[np.float64], shape = (n_samples,)
            Return predicted values.

        Notes
        -----
        Be careful that, for two-dimensional and higher-dimensional data, not
        passing a ``x_new`` argument may result to something unexpected as for
        now, the function ``np.unique`` re-order the columns of the data. To be
        sure of the results, please provide a ``x_new`` argument.

        Examples
        --------
        For one-dimensional data.

        >>> n_points = 101
        >>> x = np.linspace(0, 1, n_points)
        >>> y = np.sin(x) + np.random.normal(0, 0.05, n_points)
        >>> x_new = np.linspace(0, 1, 11)

        >>> lp = LocalPolynomial(
        ...     kernel_name='epanechnikov', bandwidth=0.3, degree=1
        ... )
        >>> lp.predict(y=y, x=x, x_new=x_new)

        For two-dimensional data.

        >>> n_points = 51
        >>> pts = np.linspace(0, 1, n_points)
        >>> xx, yy = np.meshgrid(pts, pts, indexing='ij')
        >>> x = np.column_stack([xx.flatten(), yy.flatten()])
        >>> eps = np.random.normal(0, 0.1, len(x))
        >>> y = np.sin(x[:, 0]) * np.cos(x[:, 1]) + eps

        >>> lp = LocalPolynomial(
        ...     kernel_name='epanechnikov', bandwidth=0.3, degree=2
        ... )
        >>> lp.predict(y=y, x=x, x_new=x_new)

        """
        if x_new is None:
            x_new = np.unique(x, axis=0)

        if x.ndim == 1:  # because PolynomialFeatures wants 2d arrays.
            x = x.reshape(-1, 1)
        if x_new.ndim == 1:
            x_new = x_new.reshape(-1, 1)

        dmat_sampling = self.poly_features.fit_transform(x)
        dmat_query = self.poly_features.fit_transform(x_new)

        y_pred = np.zeros(x_new.shape[0])
        for idx, (pts, dmat) in enumerate(zip(x_new, dmat_query)):
            y_pred[idx] = _local_regression(
                y, x, pts, dmat_sampling, dmat, self.bandwidth, self.kernel
            )
        return y_pred
