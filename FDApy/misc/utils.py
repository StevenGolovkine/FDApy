#!/usr/bin/env python
# -*-coding:utf8 -*
"""
Utility functions
-----------------

"""
import itertools
import numpy as np
import numpy.typing as npt
import scipy

from numpy.linalg import eigh

from typing import Callable, Dict, Optional, Tuple, Union


#############################################################################
# Standardization functions
#############################################################################
def _normalization(
    x: npt.NDArray[np.float64],
    max_x: Optional[np.float64] = None,
    min_x: Optional[np.float64] = None
) -> npt.NDArray[np.float64]:
    r"""Normalize a vector :math:`[a, b]` into a vector :math:`[0, 1]`.

    This function standardizes a vector by applying the following
    transformation to the vector :math:`X`:

    ..math:: X_{norm} = \frac{X - \min{X}}{\max{X} - \min{X}}.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape=(n_obs,)
        Vector of data
    max_x: Optional[np.float64], default=None
        Maximum value
    min_x: Optional[np.float64], default=None
        Minimum value

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs,)
        Vector of standardized data.

    Example
    -------
    >>> _normalization(np.array([0, 5, 10]))
    array([0., 0.5, 1.])

    """
    if (max_x is None) and (min_x is None):
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
    x: npt.NDArray[np.float64], shape=(n_obs,)
        Vector of data

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs,)
        Vector of standardized data.

    Example
    -------
    >>> _standardization(np.array([0, 5, 10]))
    array([-1.22474487, 0., 1.22474487])

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
    x: npt.NDArray[np.float64], shape=(n_obs, n_features)
        Matrix of data

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_features,)
        Vector of means

    Example
    -------
    >>> _row_mean(
    ...     np.array(
    ...         [
    ...             [1., 2., 3.],
    ...             [1., 2., 3.],
    ...             [1., 2., 3.],
    ...             [1., 2., 3.]
    ...         ]
    ...     )
    ... )
    array([1., 2., 3.])

    """
    return np.mean(x, axis=0)


def _row_var(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the variance of an array with respect to the rows.

    This function computes the variance of the rows of an array.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape=(n_obs, n_features)
        Matrix of data

    Returns
    -------
    npt.NDArray[np.float64], shape = (n_features,)
        Vector of variances

    Example
    -------
    >>> _row_var(
    ...     np.array(
    ...         [
    ...             [1., 2., 3.],
    ...             [1., 2., 3.],
    ...             [1., 2., 3.],
    ...             [1., 2., 3.]
    ...         ]
    ...     )
    ... )
    array([0., 0., 0.])

    """
    return x.var(axis=0)


def _col_mean(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the mean of an array with respect to the columns.

    This function computes the mean of an array with respect to the columns.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape=(n_obs, n_features)
        Matrix of data

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs,)
        Vector of means

    Example
    -------
    >>> _col_mean(
    ...     np.array(
    ...         [
    ...             [1., 2., 3.],
    ...             [1., 2., 3.],
    ...             [1., 2., 3.],
    ...             [1., 2., 3.]
    ...         ]
    ...     )
    ... )
    array([2., 2., 2., 2.])

    """
    return x.mean(axis=1)


def _col_var(
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the variance of an array with respect to the columns.

    This function computes the variance of the column of an array.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape=(n_obs, n_features)
        Matrix of data

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs,)
        Vector of variances

    Example
    -------
    >>> _col_var(
    ...     np.array(
    ...         [
    ...             [1., 2., 3.],
    ...             [1., 2., 3.],
    ...             [1., 2., 3.],
    ...             [1., 2., 3.]
    ...         ]
    ...     )
    ... )
    array([0.66666667, 0.66666667, 0.66666667, 0.66666667])

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
    x: npt.NDArray[np.float64], shape=(n_obs, n_features)
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
    >>> x = np.array([[1, 2], [4, 5], [7, 8]])
    >>> _get_axis_dimension(x, 0)
    3
    >>> _get_axis_dimension(x, 1)
    2

    """
    return x.shape[axis]


def _get_dict_dimension(
    x: Dict[np.str_, npt.NDArray[np.float64]]
) -> Tuple[int, ...]:
    """Return the shape of an object defined as a dict of np.ndarray.

    Parameters
    ----------
    x: Dict[np.str_, npt.NDArray[np.float64]]
        Dictionary containing keys as string and values as numpy array.

    Returns
    -------
    Tuple[int, ...]
        Tuple containing the shape of the arrays defined in the dictionary.

    Example
    -------
    >>> x = {
    ...     'a': np.array([1, 2, 3]),
    ...     'b': np.array([4, 5])
    ... }
    >>> _get_dict_dimension(x)
    (3, 2)

    """
    return tuple(el.shape[0] for el in x.values())


def _get_obs_shape(
    x: Dict[np.str_, Dict[int, npt.NDArray[np.float64]]],
    obs: int
) -> Tuple[int, ...]:
    """Get the shape of `obs` if `X` is a nested dict.

    Parameters
    ----------
    x: Dict[np.str_, Dict[int, npt.NDArray[np.float64]]]
        Nested dictionary containing the data, where the first level of keys
        are strings and the second level of keys are integers representing the
        observation number.
    obs: int
        Observation number for which to get the shape.

    Returns
    -------
    Tuple[int, ...]
        Tuple containing the shape of the `obs`-th observation.

    Example
    -------
    >>> x = {
    ...     'a': {0: np.array([1, 2, 3]), 1: np.array([4, 5])},
    ...     'b': {0: np.array([1, 2]), 1: np.array([3, 4])}
    ... }
    >>> _get_obs_shape(x, 0)
    (3, 2)
    >>> _get_obs_shape(x, 1)
    (2, 2)

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
    x: npt.NDArray[np.float64], shape=(n_obs, n_features)
        Matrix of data
    num: int, default=0
        The number of columns to shift.
    fill_value: float, default=np.nan
        The value with one fill the array.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs, n_features)
        The shift array.

    Example
    -------
    >>> _shift(np.array([1, 2, 3, 4, 5]), num=2, fill_value=np.nan)
    array([nan, nan, 1., 2., 3.])

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

def _integrate(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    method: np.str_ = 'simpson'
) -> float:
    r"""Compute an estimate of the integral of 1-dimensional curve.

    This function computes an estimation of the integral of :math:`y(x)` over
    the domain :math:`x`:

    .. math:: \int y(x)dx

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape=(n_features,)
        Domain for the integration, it has to be ordered.
    y: npt.NDArray[np.float64], shape=(n_features,)
        Observations
    method: np.str_, {'simpson', 'trapz'}, default = 'simpson'
        The method used to integrated.

    Returns
    -------
    float
        Estimation of the integration of :math:`y(x)`.

    Example
    -------
    >>> X = np.array([1, 2, 4])
    >>> Y = np.array([1, 4, 16])
    >>> _integrate(Y, X)
    21.0

    """
    if method == 'simpson':
        return scipy.integrate.simps(x=x, y=y)  # type: ignore
    elif method == 'trapz':
        return np.trapz(x=x, y=y)
    else:
        raise ValueError(f'{method} not implemented!')


def _integrate_2d(
    z: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    method: np.str_ = 'simpson'
) -> float:
    r"""Compute an estimate of the integral of 2- dimensional surface.

    This function computes an estimation of the integral of :math:`z(x, y)`
    over the domain :math:`x \times y`.

    Parameters
    ----------
    z: npt.NDArray[np.float64], shape=(n_features_1, n_features_2,)
        Domain for the integration, it has to be ordered.
    x: npt.NDArray[np.float64], shape=(n_features_1,)
        First domain for the integration, it has to be ordered.
    y: npt.NDArray[np.float64], shape=(n_features_2,)
        Second domain for the integration, it has to be ordered.
    method: np.str_, {'simpson', 'trapz'}, default = 'simpson'
        The method used to integrated.

    Returns
    -------
    float
        Estimation of the integration of :math:`z(x, y)`.

    Example
    -------
    >>> X = np.array([1, 2, 4])
    >>> Y = np.array([1, 2])
    >>> Z = np.array([[1, 2], [4, 5], [7, 8]])
    >>> _integrate_2d(Z, X, Y)
    15.75

    """
    if method == 'simpson':
        temp = scipy.integrate.simps(x=x, y=z, axis=0)
        return scipy.integrate.simps(x=y, y=temp)  # type: ignore
    elif method == 'trapz':
        temp = np.trapz(x=x, y=z, axis=0)
        return np.trapz(x=y, y=temp)
    else:
        raise ValueError(f'{method} not implemented!')


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
    x: npt.NDArray[np.float64]
        First curve considered.
    y: npt.NDArray[np.float64]
        Second curve considered.
    axis: Optional[npt.NDArray[np.float64]], default=None
        Domain of integration. If ``axis`` is ``None``, the domain is set to be
        a regular grid on :math:`[0, 1]` with ``len(x)`` number of points.

    Returns
    -------
    float
        The inner product between ``x`` and ``y``.

    Example
    -------
    >>> _inner_product(np.array([1, 2, 3]), np.array([4, 5, 6]))
    10.5

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
    x: npt.NDArray[np.float64]
        First surface considered.
    y: npt.NDArray[np.float64]
        Second surface considered.
    primary_axis: Optional[npt.NDArray[np.float64]], default=None
        Domain of integration for the primary axis. If ``primary_axis`` is
        ``None``, the domain is set to be a regular grid on :math:`[0, 1]` with
        ``len(x)`` number of points.
    secondary_axis: Optional[npt.NDArray[np.float64]], default=None
        Domain of integration for the secondary axis. If ``secondary_axis`` is
        ``None``, the domain is set to be a regular grid on :math:`[0, 1]` with
        ``len(x)`` number of points.

    Returns
    -------
    float
        The inner product between ``x`` and ``y``.

    Example
    -------
    >>> _inner_product_2d(
    ...     np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
    ...     np.array([[4, 5, 6], [1, 2, 3], [4, 5, 6]])
    ... )
    10.5

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
    x: npt.NDArray[np.float64], shape=(n_obs1,)
        First input vector
    y: npt.NDArray[np.float64], shape=(n_obs2,)
        Second input vector

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_obs1, n_obs2)
        Tensor product between ``x`` and ``y``.

    Example
    -------
    >>> X = np.array([1, 2, 3])
    >>> Y = np.array([-1, 2])
    >>> _outer(X, Y)
    array([[-1, 2], [-2, 4], [-3, 6]])

    """
    return np.outer(x, y)


def _integration_weights(
    x: npt.NDArray[np.float64],
    method: Union[np.str_, Callable] = 'trapz'
) -> npt.NDArray[np.float64]:
    """Compute integration weights.

    Compute weights for numerical integration over the domain `X` given
    the method `method`.

    Parameters
    ----------
    x: npt.NDArray[np.float64], shape=(n_points,)
        Domain on which compute the weights.
    method: Union[np.str_, Callable], default='trapz'
        The method to compute the weights.

    Returns
    -------
    npt.NDArray[np.float64], shape=(n_points,)
        The integration weights.

    Example
    -------
    >>> _integration_weights(np.array([1, 2, 3, 4, 5]), method='trapz')
    array([0.5, 1., 1., 1., 0.5])
    >>> _integration_weights(np.array([1, 2, 3, 4, 5]), method='simpson')
    array([0.33333333, 1.33333333, 0.66666667, 1.33333333, 0.33333333])

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


def _select_number_eigencomponents(
    eigenvalues: npt.NDArray[np.float64],
    percentage: Optional[Union[np.float64, np.int64]] = None
) -> int:
    """Select the number of eigencomponents.

    Parameters
    ----------
    eigenvalues: npt.NDArray[np.float64]
        An estimation of the eigenvalues.
    percentage: Optional[Union[np.float64, np.int]], default=None
        Number of components to keep. If `percentage` is `None`, all
        components are kept, ``percentage == len(eigenvalues)``.
        If `percentage` is an integer, `percentage` components are kept. If
        `0 < percentage < 1`, select the number of components such that the
        amount of variance that needs to be explained is greater than the
        percentage specified by `percentage`.

    Returns
    -------
    int
        Number of eigenvalues to retain.

    """
    if isinstance(percentage, int):
        return percentage
    elif isinstance(percentage, float) and (percentage < 1):
        var_explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        return np.sum(var_explained < percentage) + 1
    elif percentage is None:
        return len(eigenvalues)
    else:
        raise ValueError('The `percentage` parameter is not correct.')


def _compute_covariance(
    eigenvalues: npt.NDArray[np.float64],
    eigenfunctions: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the covariance matrix using Mercer's theorem.

    Parameters
    ----------
    eigenvalues: npt.NDArray[np.float64], shape=(n_components,)
        The singular values corresponding to each of selected components.
    eigenfunctions: npt.NDArray[np.float64], shape=(n_components, n_points)
        An array representing the eigenfunctions.

    Returns
    -------
    npt.NDArray[np.float64]
        An estimation of the covariance using Mercer's theorem.

    References
    ----------
    - Mercer, J. (1909), "Functions of positive and negative type and their
    connection with the theory of integral equations", Philosophical
    Transactions of the Royal Society A, 209 (441-458): 415-446.

    """
    temp = np.dot(np.transpose(eigenfunctions), np.diag(eigenvalues))
    return np.dot(temp, eigenfunctions)


##############################################################################
# Array computation
##############################################################################
def _eigh(
    matrix: npt.NDArray[np.float64],
    UPLO: np.str_ = 'L'  # noqa
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Return the eigenvalues and eigenvectors of a real symmetrix matrix.

    Returns two objects, a 1-D array containing the eigenvalues of `matrix`,
    and a 2-D square array of the corresponding eigenvectors (in columns).
    This function overrides ``numpy.linalg.eigh`` by returning the eigenvalues
    in descending order.

    Parameters
    ----------
    matrix: npt.NDArray[np.float64], shape=(M, M)
        Hermitian or real symmetric matrices whose eigenvalues and eigenvectors
        are to be computed.
    UPLO: np.str_, {'L', 'U'}, default='L'
        Specifies whether the calculation is done with the lower triangular
        part of a ('L', default) or the upper triangular part ('U').
        Irrespective of this value only the real parts of the diagonal will be
        considered in the computation to preserve the notion of a Hermitian
        matrix. It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray]:
        The first element represents the eigenvalues in descending order, each
        repeated according to its multiplicity. The second element represents
        the normalized eigenvectors. The column ``v[:, i]`` of the eigenvectors
        matrix corresponds to the eigenvalue ``w[i]``.

    """
    eigenvalues, eigenvectors = eigh(matrix, UPLO)
    return eigenvalues[::-1], np.fliplr(eigenvectors)
