#!/usr/bin/python3.7
# -*-coding:utf8 -*
"""Module that contains utility functions.

This module is used to define diverse helper functions. These functions are
designed to standardize, manipulate and do computation on array.
"""
import numpy as np
import scipy

from sklearn.preprocessing import StandardScaler


#############################################################################
# Standardization functions
#############################################################################
def rangeStandardization_(X):
    r"""Transform a vector [a, b] into a vector [0, 1].

    This function standardizes a vector by applying the following
    transformation to the vector :math:`X`:
    ..math:: X_{norm} = \frac{X - \min{X}}{\max{X} - \min{X}}

    Parameters
    ----------
    X : array-like, shape = (n_features, )
        Data

    Returns
    -------
    range_ : array_like, shape = (n_features)

    Example
    -------
    >>>rangeStandardization_(np.array([0, 5, 10]))
    array([0., 0.5, 1.])

    """
    range_ = (X - np.min(X)) / (np.max(X) - np.min(X))
    return range_


def rowMean_(X):
    """Compute the mean of an array with respect to the rows.

    This function computes the mean of an array with respect to the rows.

    Parameters
    ----------
    X : array-like, shape = (n_obs, n_features)
        Data

    Returns
    -------
    mean_ : array-like, shape = (n_features,)

    Example
    -------
    >>>rowMean_(
        np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]))
    array([1., 2., 3.])

    """
    scaler = StandardScaler()
    return scaler.fit(X).mean_


def rowVar_(X):
    """Compute the variance of an array with respect to the rows.

    This function computes the variance of the row of an array.

    Parameters
    ----------
    X : array-like, shape = (n_obs, n_features)
        Data

    Returns
    -------
    var_ : array-like, shape = (n_features,)

    Example
    -------
    >>>rowVar_(
        np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]))
    array([0., 0., 0.])

    """
    scaler = StandardScaler()
    return scaler.fit(X).var_


def colMean_(X):
    """Compute the mean of an array with respect to the columns.

    This function computes the mean of an array with respect to the columns.

    Parameters
    ----------
    X : array-like, shape = (n_obs, n_features)
        Data

    Returns
    -------
    mean_ : array-like, shape = (n_obs,)

    Example
    -------
    >>>colMean_(
        np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]))
    array([2., 2., 2., 2.])

    """
    scaler = StandardScaler()
    return scaler.fit(X.T).mean_


def colVar_(X):
    """Compute the variance of an array with respect to the columns.

    This function computes the variance of the column of an array.

    Parameters
    ----------
    X : array-like, shape = (n_obs, n_features)
        Data

    Returns
    -------
    var_ : array-like, shape = (n_obs,)

    Example:
    >>>colVar_(
        np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]))
    array([0.66666667, 0.66666667, 0.66666667, 0.66666667])

    """
    scaler = StandardScaler()
    return scaler.fit(X.T).var_


############################################################################
# Array manipulation functions.
############################################################################


def shift_(X, num, fill_value=np.nan):
    """Shift an array.

    This function shifts an array :math:`X` by a number :math:`num`.

    Parameters
    ----------
    X : array-like ,shape = (n_obs, n_features)
        Input array
    num : int
        The number of columns to shift.
    fill_value : float or np.nan
        The value with one fill the array.

    Returns
    -------
    res : array-like, shape = (n_obs, n_features)
        The shift array.

    Example
    -------
    >>>shift_(np.array([1, 2, 3, 4, 5]), num=2, fill_value=np.nan)
    array([nan, nan, 1, 2, 3])

    References
    ----------
    * https://stackoverflow.com/
    questions/30399534/shift-elements-in-a-numpy-array/42642326

    """
    res = np.empty_like(X)
    if num > 0:
        res[:num] = fill_value
        res[num:] = X[:-num]
    elif num < 0:
        res[num:] = fill_value
        res[:num] = X[-num:]
    else:
        res = X
    return res


##############################################################################
# Array computation
##############################################################################

def tensorProduct_(X, Y):
    """Compute the tensor product of two vectors.

    This function computes the tensor product of two vectors.

    Parameters
    ----------
    X : array-like, shape = (n_obs1,)
        First input vector
    Y : array-like, shape = (n_obs2,)
        Second input vector

    Returns
    -------
    res : ndarray, shape = (n_obs1, n_obs2)

    Example
    -------
    >>>X = np.array([1, 2, 3])
    >>>Y = np.array([-1, 2])
    >>>tensorProduct_(X, Y)
    array([[-1, 2], [-2, 4], [-3, 6]])

    """
    return np.outer(X, Y)


def integrate_(X, Y, method='simpson'):
    """Compute an estimate of the integral.

    This function computes an esmitation of the integral of :math:`Y` over the
    domain :math:`X`.

    Parameters
    ----------
    X : array-like, shape = (n_features,)
        Domain for the integration, it has to be ordered.
    Y : array-like, shape = (n_features,)
        Observations
    method : str, default = 'simpson'
        The method used to integrated. Currently, only the Simpsons method
        is implemented.

    Returns
    -------
    res : int
        Estimation of the integration of Y over X.

    Example
    -------
    >>>X = np.array([1, 2, 4])
    >>>Y = np.array([1, 4, 16])
    >>>integrate_(X, Y)
    21.0

    """
    if method != 'simpson':
        raise ValueError('Only the Simpsons method is implemented!')
    return scipy.integrate.simps(Y, X)


def integrationWeights_(X, method='trapz'):
    """Computation integration weights.

    Compute weights for numerical integration over the domain `X` given
    the method `method`.

    Parameters
    ----------
    X : array-like, shape = (n_points,)
        Domain on which compute the weights.
    method : str or callable, default = 'trapz'
            The method to compute the weights.

    Returns
    -------
    W : array-like, shape = (n_points,)
        The weights

    Example
    -------
    >>>integrationWeights_(np.array([1, 2, 3, 4, 5]), method='trapz')
    array([0.5, 1., 1., 1., 0.5])

    Notes
    -----
    TODO :
    * Add other methods: Simpson, midpoints, ...
    * Add tests

    References
    ----------
    * https://en.wikipedia.org/wiki/Trapezoidal_rule

    """
    L = len(X)
    if method == 'trapz':
        W = 0.5 * np.concatenate([[X[1] - X[0]],
                                  X[2:] - X[:(L - 2)],
                                  [X[L - 1] - X[L - 2]]])
    elif callable(method):
        W = method(X)
    else:
        raise ValueError("Method not implemented!")

    return W
