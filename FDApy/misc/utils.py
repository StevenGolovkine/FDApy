#!/usr/bin/env python
# -*-coding:utf8 -*
"""Module that contains utility functions.

This module is used to define diverse helper functions. These functions are
designed to standardize, manipulate and do computation on array.
"""
import itertools
import numpy as np
import numpy.typing as npt
import scipy

from typing import Dict, Tuple

from sklearn.preprocessing import StandardScaler


#############################################################################
# Standardization functions
#############################################################################

def range_standardization_(
    x: npt.NDArray[np.float64],
    max_x: float = np.nan,
    min_x: float = np.nan
) -> npt.NDArray[np.float64]:
    r"""Transform a vector [a, b] into a vector [0, 1].

    This function standardizes a vector by applying the following
    transformation to the vector :math:`X`:
    ..math:: X_{norm} = \frac{X - \min{X}}{\max{X} - \min{X}}

    Parameters
    ----------
    x: array-like, shape = (n_features, )
        Data
    max_x: float, default=None
        Maximum value
    min_x: float, default=None
        Minimum value

    Returns
    -------
    range_: array_like, shape = (n_features)

    Example
    -------
    >>> range_standardization_(np.array([0, 5, 10]))
    array([0., 0.5, 1.])

    """
    if (np.isnan(max_x)) and (np.isnan(min_x)):
        max_x = np.amax(x)
        min_x = np.amin(x)
    range_ = (x - min_x) / (max_x - min_x)
    return range_


def row_mean_(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the mean of an array with respect to the rows.

    This function computes the mean of an array with respect to the rows.

    Parameters
    ----------
    x: array-like, shape = (n_obs, n_features)
        Data

    Returns
    -------
    mean_: array-like, shape = (n_features,)

    Example
    -------
    >>> row_mean_(
        np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]))
    array([1., 2., 3.])

    """
    scaler = StandardScaler()
    return scaler.fit(x).mean_  # type: ignore


def row_var_(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the variance of an array with respect to the rows.

    This function computes the variance of the row of an array.

    Parameters
    ----------
    x: array-like, shape = (n_obs, n_features)
        Data

    Returns
    -------
    var_: array-like, shape = (n_features,)

    Example
    -------
    >>>row_var_(
        np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]))
    array([0., 0., 0.])

    """
    scaler = StandardScaler()
    return scaler.fit(x).var_  # type: ignore


def col_mean_(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the mean of an array with respect to the columns.

    This function computes the mean of an array with respect to the columns.

    Parameters
    ----------
    x: array-like, shape = (n_obs, n_features)
        Data

    Returns
    -------
    mean_: array-like, shape = (n_obs,)

    Example
    -------
    >>> col_mean_(np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]))
    array([2., 2., 2., 2.])

    """
    scaler = StandardScaler()
    return scaler.fit(x.T).mean_  # type: ignore


def col_var_(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the variance of an array with respect to the columns.

    This function computes the variance of the column of an array.

    Parameters
    ----------
    x: array-like, shape = (n_obs, n_features)
        Data

    Returns
    -------
    var_: array-like, shape = (n_obs,)

    Example:
    >>> col_var_(
        np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]))
    array([0.66666667, 0.66666667, 0.66666667, 0.66666667])

    """
    scaler = StandardScaler()
    return scaler.fit(x.T).var_  # type: ignore


############################################################################
# Array manipulation functions.
############################################################################

def get_axis_dimension_(
    x: npt.NDArray[np.float64],
    axis: int = 0
) -> int:
    """Get the dimension of an array :math:`X` along the `axis`."""
    return x.shape[axis]


def get_dict_dimension_(
    x: Dict[str, npt.NDArray[np.float64]]
) -> Tuple[int, ...]:
    """Return the shape of `X` defined as a dict of np.ndarray."""
    return tuple(i.shape[0] for i in x.values())


def get_obs_shape_(
    x: Dict[str, Dict[int, npt.NDArray[np.float64]]],
    obs: int
) -> Tuple[int, ...]:
    """Return the shape of `obs` if `X` is a nested dict."""
    shapes = tuple(dim[obs].shape for _, dim in x.items())
    return tuple(itertools.chain.from_iterable(shapes))


def shift_(
    x: npt.NDArray[np.float64],
    num: int,
    fill_value: float = np.nan
) -> npt.NDArray[np.float64]:
    """Shift an array.

    This function shifts an array :math:`X` by a number :math:`num`.

    Parameters
    ----------
    x: array-like ,shape = (n_obs, n_features)
        Input array
    num: int
        The number of columns to shift.
    fill_value: float or np.nan
        The value with one fill the array.

    Returns
    -------
    res: array-like, shape = (n_obs, n_features)
        The shift array.

    Example
    -------
    >>> shift_(np.array([1, 2, 3, 4, 5]), num=2, fill_value=np.nan)
    array([nan, nan, 1, 2, 3])

    References
    ----------
    * https://stackoverflow.com/
    questions/30399534/shift-elements-in-a-numpy-array/42642326

    """
    res = np.empty_like(x, dtype=float)
    if num > 0:
        res[:num] = fill_value
        res[num:] = x[:-num]
    elif num < 0:
        res[num:] = fill_value
        res[:num] = x[-num:]
    else:
        res = x
    return res


##############################################################################
# Array computation
##############################################################################

def outer_(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the tensor product of two vectors.

    This function computes the tensor product of two vectors.

    Parameters
    ----------
    x: array-like, shape = (n_obs1,)
        First input vector
    y: array-like, shape = (n_obs2,)
        Second input vector

    Returns
    -------
    res : ndarray, shape = (n_obs1, n_obs2)

    Example
    -------
    >>> X = np.array([1, 2, 3])
    >>> Y = np.array([-1, 2])
    >>> tensorProduct_(X, Y)
    array([[-1, 2], [-2, 4], [-3, 6]])

    """
    return np.outer(x, y)


def integrate_(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    method: str = 'simpson'
) -> float:
    """Compute an estimate of the integral.

    This function computes an estimation of the integral of :math:`Y` over the
    domain :math:`X`.

    Parameters
    ----------
    x: array-like, shape = (n_features,)
        Domain for the integration, it has to be ordered.
    y: array-like, shape = (n_features,)
        Observations
    method : str, default = 'simpson'
        The method used to integrated. Currently, only the Simpsons method
        is implemented.

    Returns
    -------
    res : float
        Estimation of the integration of Y over X.

    Example
    -------
    >>> X = np.array([1, 2, 4])
    >>> Y = np.array([1, 4, 16])
    >>> integrate_(X, Y)
    21.0

    """
    if method != 'simpson':
        raise ValueError('Only the Simpsons method is implemented!')
    return scipy.integrate.simps(y, x)  # type: ignore


def integration_weights_(
    x: npt.NDArray[np.float64],
    method: str = 'trapz'
) -> npt.NDArray[np.float64]:
    """Compute integration weights.

    Compute weights for numerical integration over the domain `X` given
    the method `method`.

    Parameters
    ----------
    x: array-like, shape = (n_points,)
        Domain on which compute the weights.
    method: str or callable, default = 'trapz'
            The method to compute the weights.

    Returns
    -------
    w: array-like, shape = (n_points,)
        The weights

    Example
    -------
    >>> integrationWeights_(np.array([1, 2, 3, 4, 5]), method='trapz')
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
    if method == 'trapz':
        w = 0.5 * np.concatenate(
            (
                np.array([x[1] - x[0]]),
                x[2:] - x[:(len(x) - 2)],
                np.array([x[len(x) - 1] - x[len(x) - 2]])
            ), axis=None
        )
    elif callable(method):
        w = method(x)
    else:
        raise NotImplementedError("Method not implemented!")
    return w  # type: ignore
