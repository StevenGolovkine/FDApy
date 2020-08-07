#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""Module for LocalPolynomial classes.

This module is used fit local polynomial regression.
"""
import numpy as np

from sklearn.preprocessing import PolynomialFeatures


##############################################################################
# Inner functions for the LocalPolynomial class.

def _gaussian(t):
    """Compute the gaussian density with mean 0 and stadard deviation 1.

    Parameters
    ----------
    t : array-like, shape = (n_samples,)
        Array at which computes the gaussian density

    Returns
    -------
    K : array-like, shape = (n_samples,)

    """
    return np.exp(- t**2 / 2) / np.sqrt(2 * np.pi)


def _epanechnikov(t):
    """Compute the Epanechnikov kernel.

    Parameters
    ----------
    t : array-like, shape = (n_samples,)
        Array on which computes the Epanechnikov kernel

    Returns
    -------
    K : array-like, shape = (n_samples,)

    References
    ----------
    Hastie, Tibshirani and Friedman, Elements of Statistical Learning, 2009,
    equation 6.4

    """
    K = np.zeros(t.shape)
    idx = np.where(np.abs(t) <= 1)
    K[idx] = 0.75 * (1 - t[idx]**2)
    return K


def _tri_cube(t):
    """Compute the tri-cube kernel.

    Parameters
    ----------
    t : array-like, shape = (n_samples,)
        Array on which computes the tri-cube kernel

    Returns
    -------
    K : array-like, shape = (n_samples,)

    References
    ----------
    Hastie, Tibshirani and Friedman, Elements of Statistical Learning, 2009,
    equation 6.6

    """
    K = np.zeros(t.shape)
    idx = np.where(np.abs(t) < 1)
    K[idx] = (1 - np.abs(t[idx])**3)**3
    return K


def _bi_square(t):
    """Compute the bi-square kernel.

    Parameters
    ----------
    t : array-like, shape = (n_samples,)
        Array on which computes the bi-square kernel

    Returns
    -------
    K : array-like, shape = (n_samples,)

    References
    ----------
    Cleveland, Robust Locally Weighted Regression and Smoothing Scatterplots,
    1979, p.831

    """
    K = np.zeros(t.shape)
    idx = np.where(np.abs(t) < 1)
    K[idx] = (1 - t[idx]**2)**2
    return K


def _compute_kernel(x, x0, h, kernel='gaussian'):
    """Compute kernel at point norm(x - x0) / h.

    Parameters
    ----------
    x : array-like, shape = (n_dim, n_samples)
        Training data.
    x0 : float-array, shape = (n_dim, )
        Number around which compute the kernel.
    h : float or float-array, shape = (n_samples, )
        Bandwidth to control the importance of points far from x0.
    kernel : string, default='gaussian'
        Kernel name used.

    Returns
    -------
    K : array-like , shape = (n_samples, )

    References
    ----------
    Hastie, Tibshirani and Friedman, Elements of Statistical Learning, 2009,
    equation 6.13

    """
    if not np.iterable(x0):
        x0 = np.asarray([x0])
    if x.ndim != np.size(x0):
        raise ValueError("""
        x and x0 do not have the same dimension!""")

    t = np.sqrt(np.sum(np.power(x - x0[:, np.newaxis], 2), axis=0)) / h

    if kernel == 'gaussian':
        K = _gaussian(t)
    elif kernel == 'epanechnikov':
        K = _epanechnikov(t)
    elif kernel == 'tricube':
        K = _tri_cube(t)
    elif kernel == 'bisquare':
        K = _bi_square(t)
    else:
        raise ValueError(''.join[
            'The kernel `', kernel, '` is not implemented!'])

    return K


def _loc_poly(x, y, x0, B, B0,
              kernel='epanechnikov', h=0.05):
    """Local polynomial regression for one point.

    Let (x_1, Y_1), ..., (x_n, Y_n) be a random sample of bivariate data.
    Assume the following model: Y_i = f(x_i) + e_i. We would like to estimate
    the unknown regression function f(x) = E[Y | X = x]. We approximate f(x)
    using Taylor series.

    Parameters
    ----------
    x : array-like, shape = (n_dim, n_samples)
        Input array.
    y : array-like, shape = (n_samples, )
        1-D input array such that y = f(x) + e.
    x0 : array-like, shape = (n_dim, )
        1-D array on which estimate the function f(x).
    B : array-like, shape = (n_sample, degree + 1)
        Design matrix of the matrix x.
    B0 : array-like, shape = (n_dim, degree + 1)
        Design matrix of the observation point x0.
    kernel : string, default='epanechnikov'
        Kernel name used as weight.
    h : float or float-array, default=0.05
        Bandwidth for the kernel trick.

    Returns
    -------
    y0_pred : float
        Prediction of y0, which is f(x0).

    References
    ----------
    Zhang and Chen, Statistical Inferences for functional data, The Annals of
    Statistics, 1052-1079, No. 3, Vol. 35, 2007.

    """
    # Compute kernel.
    K = _compute_kernel(x=x, x0=x0, h=h, kernel=kernel)

    # Compute the estimation of f (and derivatives) at x0.
    BtW = np.dot(B.T, np.diag(K))
    beta = np.dot(np.linalg.pinv(np.dot(BtW, B)), np.dot(BtW, y))

    return np.dot(B0, beta)

#############################################################################
# Class LocalPolynomial


class LocalPolynomial():
    """Local polynomial regression.

    Let (x_1, Y_1), ..., (x_n, Y_n) be a random sample of bivariate data.
    For all i, x_i belongs to R^d and Y_i in R. Assume the following model:
    Y_i = f(x_i) + e_i. We would like to estimate the unknown regression
    function f(x) = E[Y | X = x]. We approximate f(x) using Taylor series.

    Parameters
    ----------
    kernel : string, default="gaussian"
        Kernel name used as weight (default = 'gaussian').
    bandwidth : float, default=0.05
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

    def __init__(self, kernel="gaussian", bandwidth=0.05, degree=2):
        """Initialize LocalPolynomial object."""
        # TODO: Add test on parameters.
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree)

    @property
    def kernel(self):
        """Getter for `kernel`."""
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel):
        self._kernel = new_kernel

    @property
    def bandwidth(self):
        """Getter for `bandwidth`."""
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, new_bandwidth):
        self._bandwidth = new_bandwidth

    @property
    def degree(self):
        """Getter for `degree`."""
        return self._degree

    @degree.setter
    def degree(self, new_degree):
        self._degree = new_degree
        self._poly_features = PolynomialFeatures(degree=new_degree)

    def fit(self, X, y):
        """Fit local polynomial regression.

        Parameters
        ----------
        X : array-like, shape = (n_dim, n_samples)
            Training data, input array.
        y : array-like, shape = (n_samples, )
            Target values, 1-D input array

        Returns
        -------
        self : returns an instance of self.

        """
        # TODO: Add tests on the parameters.
        self.X = X
        self.Y = y

        x0 = np.unique(self.X, axis=0)
        if not np.iterable(self.bandwidth):
            bandwidth = np.repeat(self.bandwidth, np.size(x0) / np.ndim(x0))

        design_matrix = self.poly_features.\
            fit_transform(np.array(self.X, ndmin=2).T)
        design_matrix_x0 = self.poly_features.\
            fit_transform(np.array(x0, ndmin=2).T)

        self.X_fit_ = np.array([_loc_poly(self.X, self.Y, i, design_matrix, j,
                                          self.kernel, h)
                                for (i, j, h) in zip(x0.T,
                                                     design_matrix_x0,
                                                     bandwidth)])
        return self

    def predict(self, X):
        """Predict using local polynomial regression.

        Parameters
        ----------
        X : array-like, shape = (n_dim, n_samples)

        Returns
        -------
        y_pred : array-like, shape = (n_samples,)
            Return predicted values.

        """
        if isinstance(X, (int, float, np.int_, np.float_)):
            X = [X]

        if not np.iterable(self.bandwidth):
            bandwidth = np.repeat(self.bandwidth, np.size(X) / np.ndim(X))

        design_matrix = self.poly_features.\
            fit_transform(np.array(self.X, ndmin=2).T)
        design_matrix_x0 = self.poly_features.\
            fit_transform(np.array(X, ndmin=2).T)

        y_pred = np.array([_loc_poly(self.X, self.Y, i, design_matrix, j,
                                     self.kernel, h)
                           for (i, j, h) in zip(X.T,
                                                design_matrix_x0,
                                                bandwidth)])

        return y_pred

    def fit_predict(self, X, y, X_pred=None):
        """Fit the model using X and predict on X_pred.

        Parameters
        ----------
        X : array-like, shape = (n_dim, n_samples)
            Training data, input array
        y : array-like, shape = (n_sample, )
            Target values, 1-D input array
        X_pred : array-like, shape = (n_dim, n_samples2)

        Returns
        -------
        y_pred : array-like, shape = (n_samples2,)
            Return predicted values

        """
        self.X = X
        self.Y = y

        if isinstance(X_pred, (int, float, np.int_, np.float_)):
            X_pred = [X_pred]

        if not np.iterable(self.bandwidth):
            bandwidth = np.repeat(self.bandwidth,
                                  np.size(X_pred) / np.ndim(X_pred))

        design_matrix = self.poly_features.\
            fit_transform(np.array(self.X, ndmin=2).T)
        design_matrix_x0 = self.poly_features.\
            fit_transform(np.array(X_pred, ndmin=2).T)

        y_pred = np.array([_loc_poly(self.X, self.Y, i, design_matrix, j,
                                     self.kernel, h)
                           for (i, j, h) in zip(X_pred.T,
                                                design_matrix_x0,
                                                bandwidth)])

        return y_pred
