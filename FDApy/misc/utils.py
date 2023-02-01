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

from typing import Dict, Optional, Tuple


#############################################################################
# Standardization functions
#############################################################################
def _normalization(
    x: npt.NDArray[np.float64],
    max_x: float = np.nan,
    min_x: float = np.nan
) -> npt.NDArray[np.float64]:
    r"""Normalize a vector :math:`[a, b]` into a vector :math:`[0, 1]`.

    This function standardizes a vector by applying the following
    transformation to the vector :math:`X`:

    ..math:: X_{norm} = \frac{X - \min{X}}{\max{X} - \min{X}}.

    Parameters
    ----------
    x: np.ndarray, shape=(n_obs,)
        Vector of data
    max_x: float, default=None
        Maximum value
    min_x: float, default=None
        Minimum value

    Returns
    -------
    np.array, shape = (n_obs,)
        Vector of standardized data.

    Example
    -------
    _normalization(np.array([0, 5, 10]))
    > array([0., 0.5, 1.])

    """
    if (np.isnan(max_x)) and (np.isnan(min_x)):
        max_x = np.amax(x)
        min_x = np.amin(x)
    if max_x == min_x:
        return np.zeros_like(x)
    else:
        return (x - min_x) / (max_x - min_x)


def _standardization(
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Standardize a vector :math:`[a, b]`.

    This function standardizes a vector by applying the following
    transformation to the vector :math:`X`:

    ..math:: X_{norm} = \frac{X - mean(X)}{sd(X)}.

    Parameters
    ----------
    x: np.ndarray, shape=(n_obs,)
        Vector of data

    Returns
    -------
    np.ndarray, shape = (n_obs,)
        Vector of standardized data.

    Example
    -------
    _standardization(np.array([0, 5, 10]))
    > array([-1.22474487, 0., 1.22474487])

    """
    if np.std(x) == 0:
        return np.zeros_like(x)
    else:
        return (x - np.mean(x)) / np.std(x)


def _row_mean(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the mean of an array with respect to the rows.

    This function computes the mean of an array with respect to the rows.

    Parameters
    ----------
    x: np.ndarray, shape = (n_obs, n_features)
        Matrix of data

    Returns
    -------
    np.ndarray, shape = (n_features,)
        Vector of means

    Example
    -------
    _row_mean(
        np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
    )
    > array([1., 2., 3.])

    """
    return np.mean(x, axis=0)


def _row_var(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the variance of an array with respect to the rows.

    This function computes the variance of the rows of an array.

    Parameters
    ----------
    x: np.ndarray, shape = (n_obs, n_features)
        Matrix of data

    Returns
    -------
    np.ndarray, shape = (n_features,)
        Vector of variances

    Example
    -------
    _row_var(
        np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
    )
    > array([0., 0., 0.])

    """
    return x.var(axis=0)


def _col_mean(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the mean of an array with respect to the columns.

    This function computes the mean of an array with respect to the columns.

    Parameters
    ----------
    x: np.ndarray, shape = (n_obs, n_features)
        Matrix of data

    Returns
    -------
    np.ndarray, shape = (n_obs,)
        Vector of means

    Example
    -------
    _col_mean(
        np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
    )
    > array([2., 2., 2., 2.])

    """
    return x.mean(axis=1)


def _col_var(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the variance of an array with respect to the columns.

    This function computes the variance of the column of an array.

    Parameters
    ----------
    x: np.ndarray, shape = (n_obs, n_features)
        Matrix of data

    Returns
    -------
    np.ndarray, shape = (n_obs,)
        Vector of variances

    Example
    -------
    _col_var(
        np.array(
            [
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]
            ]
        )
    )
    > array([0.66666667, 0.66666667, 0.66666667, 0.66666667])

    """
    return x.var(axis=1)


############################################################################
# Array manipulation functions.
############################################################################

def _get_axis_dimension(
    x: npt.NDArray[np.float64],
    axis: int = 0
) -> int:
    """Get the dimension of an array :math:`X` along the `axis`.

    Parameters
    ----------
    x: np.ndarray[np.float64], shape=(n_obs, n_features)
        Matrix of data
    axis: int, default=0
        Integer value that represents the axis along which the dimension of the
        array is to be computed. The default value is 0.

    Returns
    -------
    int
        Dimension of the array along the specified axis.

    Example
    -------
    x = np.array([[1, 2], [4, 5], [7, 8]])
    _get_axis_dimension(x, 0)
    > 3
    _get_axis_dimension(x, 1)
    > 2

    """
    return x.shape[axis]


def _get_dict_dimension(
    x: Dict[str, npt.NDArray[np.float64]]
) -> Tuple[int, ...]:
    """Return the shape of an object defined as a dict of np.ndarray.

    Parameters
    ----------
    x: dict
        Dictionary containing keys as string and values as numpy array.

    Returns
    -------
    tuple
        Tuple containing the shape of the arrays defined in the dictionary.

    Example
    -------
    x = {
        'a': np.array([1, 2, 3]),
        'b': np.array([4, 5])
    }
    _get_dict_dimension(x)
    > (3, 2)

    """
    return tuple(el.shape[0] for el in x.values())


def _get_obs_shape(
    x: Dict[str, Dict[int, npt.NDArray[np.float64]]],
    obs: int
) -> Tuple[int, ...]:
    """Get the shape of `obs` if `X` is a nested dict.

    Parameters
    ----------
    x: dict
        Nested dictionary containing the data, where the first level of keys
        are strings and the second level of keys are integers representing the
        observation number.
    obs: int
        Observation number for which to get the shape.

    Returns
    -------
    tuple
        Tuple containing the shape of the `obs`-th observation.

    Example
    -------
    x = {
        'a': {0: np.array([1, 2, 3]), 1: np.array([4, 5])},
        'b': {0: np.array([1, 2]), 1: np.array([3, 4])}
    }
    _get_obs_shape(x, 0)
    > (3, 2)
    _get_obs_shape(x, 1)
    > (2, 2)

    """
    shapes = tuple(el[obs].shape for el in x.values())
    return tuple(itertools.chain.from_iterable(shapes))


def _shift(
    x: npt.NDArray[np.float64],
    num: int = 0,
    fill_value: float = np.nan
) -> npt.NDArray[np.float64]:
    """Shift an array.

    This function shifts an array :math:`X` by a number :math:`num`.

    Parameters
    ----------
    x: np.ndarray, shape=(n_obs, n_features)
        Matrix of data
    num: int, default=0
        The number of columns to shift.
    fill_value: float or np.nan
        The value with one fill the array.

    Returns
    -------
    np.ndarray, shape = (n_obs, n_features)
        The shift array.

    Example
    -------
    _shift(np.array([1, 2, 3, 4, 5]), num=2, fill_value=np.nan)
    > array([nan, nan, 1., 2., 3.])

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

def _inner_product(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    axis: Optional[npt.NDArray[np.float64]] = None
) -> float:
    r"""Compute the inner product between two curves.

    This function computes the inner product between two curves. The inner
    product is defined as

    .. math::
        \langle x, y \rangle = \int_{\mathcal{T}} x(t)y(t)dt, t \in \mathcal{T}

    where :math:`\mathcal{T}` is a one-dimensional domain.

    Parameters
    ----------
    x: np.ndarray
        First curve considered.
    y: np.ndarray
        Second curve considered.
    axis: np.ndarray, default=None
        Domain of integration. If ``axis`` is ``None``, the domain is set to be
        a regular grid on :math:`[0, 1]` with ``len(x)`` number of points.

    Returns
    -------
    float
        The inner product between ``x`` and ``y``.

    Example
    -------
    _inner_product(np.array([1, 2, 3]), np.array([4, 5, 6]))
    > 10.5

    """
    if x.shape != y.shape:
        raise ValueError("Arguments x and y do not have the same shape.")
    if axis is None:
        axis = np.linspace(0, 1, x.shape[0])
    return np.trapz(x=axis, y=x * y)


def _inner_product_2d(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    primary_axis: Optional[npt.NDArray[np.float64]] = None,
    secondary_axis: Optional[npt.NDArray[np.float64]] = None
) -> float:
    r"""Compute the inner product between two surfaces.

    This function computes the inner product between two surfaces. The inner
    product is defined as

    .. math::
        \langle x, y \rangle = \int_{\mathcal{T}} x(t)y(t)dt =
        \int_{\mathcal{T}_1}\int_{\mathcal{T}_2} x(t_1t_2)y(t_1t_2)dt_1 dt_2,
        t = (t_1, t_2) \in \mathcal{T} = \mathcal{T}_1 \times \mathcal{T}_2

    where :math:`\mathcal{T}` is a 2-dimensional domain.

    Parameters
    ----------
    x: np.ndarray
        First surface considered.
    y: np.ndarray
        Second surface considered.
    primary_axis: np.ndarray, default=None
        Domain of integration for the primary axis. If ``primary_axis`` is
        ``None``, the domain is set to be a regular grid on :math:`[0, 1]` with
        ``len(x)`` number of points.
    secondary_axis: np.ndarray, default=None
        Domain of integration for the secondary axis. If ``secondary_axis`` is
        ``None``, the domain is set to be a regular grid on :math:`[0, 1]` with
        ``len(x)`` number of points.

    Returns
    -------
    float
        The inner product between ``x`` and ``y``.

    Example
    -------
    _inner_product_2d(
        np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        np.array([[4, 5, 6], [1, 2, 3], [4, 5, 6]])
    )
    > 10.5

    """
    if x.shape != y.shape:
        raise ValueError("Arguments x and y do not have the same shape.")
    if primary_axis is None:
        primary_axis = np.linspace(0, 1, x.shape[0])
    if secondary_axis is None:
        secondary_axis = np.linspace(0, 1, x.shape[1])
    return np.trapz(x=secondary_axis, y=np.trapz(x=primary_axis, y=x * y))


def _outer(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the tensor product of two vectors.

    This function computes the tensor product of two vectors.

    Parameters
    ----------
    x: np.ndarray, shape=(n_obs1,)
        First input vector
    y: np.ndarray, shape=(n_obs2,)
        Second input vector

    Returns
    -------
    np.ndarray, shape=(n_obs1, n_obs2)
        Tensor product between ``x`` and ``y``.

    Example
    -------
    X = np.array([1, 2, 3])
    Y = np.array([-1, 2])
    _outer(X, Y)
    > array([[-1, 2], [-2, 4], [-3, 6]])

    """
    return np.outer(x, y)


def _integrate(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    method: str = 'simpson'
) -> float:
    """Compute an estimate of the integral.

    This function computes an estimation of the integral of :math:`Y` over the
    domain :math:`X`.

    Parameters
    ----------
    x: np.ndarray, shape = (n_features,)
        Domain for the integration, it has to be ordered.
    y: np.ndarray, shape = (n_features,)
        Observations
    method : str, {'simpson', 'trapz'}, default = 'simpson'
        The method used to integrated.

    Returns
    -------
    float
        Estimation of the integration of Y over X.

    Example
    -------
    >>> X = np.array([1, 2, 4])
    >>> Y = np.array([1, 4, 16])
    >>> _integrate(X, Y)
    21.0

    """
    if method == 'simpson':
        return scipy.integrate.simps(x=x, y=y)  # type: ignore
    elif method == 'trapz':
        return np.trapz(x=x, y=y)
    else:
        raise ValueError(f'{method} not implemented!')


def _integration_weights(
    x: npt.NDArray[np.float64],
    method: str = 'trapz'
) -> npt.NDArray[np.float64]:
    """Compute integration weights.

    Compute weights for numerical integration over the domain `X` given
    the method `method`.

    Parameters
    ----------
    x: np.ndarray, shape = (n_points,)
        Domain on which compute the weights.
    method: str or callable, default = 'trapz'
        The method to compute the weights.

    Returns
    -------
    np.ndarray, shape = (n_points,)
        The integration weights

    Example
    -------
    _integration_weights(np.array([1, 2, 3, 4, 5]), method='trapz')
    > array([0.5, 1., 1., 1., 0.5])
    _integration_weights(np.array([1, 2, 3, 4, 5]), method='simpson')
    > array([0.33333333, 1.33333333, 0.66666667, 1.33333333, 0.33333333])

    References
    ----------
    * https://en.wikipedia.org/wiki/Trapezoidal_rule
    * https://en.wikipedia.org/wiki/Simpson%27s_rule

    """
    if method == 'trapz':
        weights = 0.5 * np.concatenate(
            (
                np.array([x[1] - x[0]]),
                2 * (x[1:(len(x) - 1)] - x[:(len(x) - 2)]),
                np.array([x[len(x) - 1] - x[len(x) - 2]])
            ), axis=None
        )
    elif method == 'simpson':
        weights = np.concatenate(
            (
                np.array([x[1] - x[0]]),
                [
                    4 * h if idx % 2 == 0 else 2 * h
                    for idx, h in enumerate(
                        x[1:(len(x) - 1)] - x[:(len(x) - 2)]
                    )
                ],
                np.array([x[len(x) - 1] - x[len(x) - 2]])
            ), axis=None
        ) / 3
    elif callable(method):
        weights = method(x)
    else:
        raise NotImplementedError(f"{method} not implemented!")
    return weights  # type: ignore
