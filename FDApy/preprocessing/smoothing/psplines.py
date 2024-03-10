#!/usr/bin/env python
# -*-coding:utf8 -*

"""
P-splines
---------

"""
import numpy as np
import numpy.typing as npt

from scipy.linalg import solve_triangular

from typing import Dict, Optional, List

from ...representation.basis import _basis_bsplines


########################################################################################
# Utils
def _row_tensor(
    x: npt.NDArray[np.float64], y: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray[np.float64]:
    """
    Compute the row-wise tensor product of two 2D arrays.

    The row-wise tensor product of two 2D arrays `x` and `y` is a 2D array `z` such that
    each row of `z` is the Kronecker product of the corresponding row of `x` and `y`.
    If `y` is not provided, it defaults to `x`. Note that `x` and `y` must have the
    same number of rows.

    Parameters
    ----------
    x: npt.NDArray[np.float64]
        A 2D array of shape `(m, n)`.
    y: Optional[npt.NDArray[np.float64]], optional
        A 2D array of shape `(m, q)`. If not provided, it defaults to `x`.

    Returns
    -------
    npt.NDArray[np.float64]
        A 2D array of shape `(m, n*q)` or `(m, n*n)` if `y` is not provided.

    Examples
    --------
    >>> x = np.array([[1, 2], [3, 4]])
    >>> y = np.array([[5, 6, 7], [7, 8, 9]])
    >>> _row_tensor(x, y)
    array([
        [ 5.,  6.,  7., 10., 12., 14.],
        [21., 24., 27., 28., 32., 36.]
    ])
    >>> _row_tensor(x)
    array([
        [ 1.,  2.,  2.,  4.],
        [ 9., 12., 12., 16.]
    ])

    Notes
    -----
    This function is adapted from [1]_.

    References
    ----------
    ..[1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized Linear Array
        Models with Applications to Multidimensional Smoothing. Journal of the Royal
        Statistical Society. Series B (Statistical Methodology) 68, pp.259--280.

    """
    if y is None:
        y = x
    if x.shape[0] != y.shape[0]:
        raise ValueError("`x` and `y` must have the same number of rows.")
    onex = np.ones((1, x.shape[1]))
    oney = np.ones((1, y.shape[1]))
    return np.kron(x, oney) * np.kron(onex, y)


def _h_transform(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute the H-transform of a nD array `y` with respect to a 2D array `x`.

    The H-transform of a nD array `y` with respect to a 2D array `x` is a nD array `z`.
    The H-transform generalizes the pre-multiplication of vectors and matrices by a
    matrix.

    Parameters
    ----------
    x: npt.NDArray[np.float64]
        A 2D array of shape `(n, m)`.
    y: npt.NDArray[np.float64]
        A nD array of shape `(m, n1, n2, ..., nk)`.

    Returns
    -------
    npt.NDArray[np.float64]
        A nD array of shape `(n, n1, n2, ..., nk)`.

    Notes
    -----
    This function is adapted from [1]_.

    Examples
    --------
    >>> x = np.array([[1, 2, 3]])
    >>> y = np.array([[1, 2], [3, 4], [5, 6]])
    >>> _h_transform(x, y)
    array([[22, 28]])

    References
    ----------
    ..[1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized Linear Array
        Models with Applications to Multidimensional Smoothing. Journal of the Royal
        Statistical Society. Series B (Statistical Methodology) 68, pp.259--280.

    """
    if x.shape[1] != y.shape[0]:
        raise ValueError(
            "The second dimension of `x` must be equal to the first dimension of `y`."
        )
    y_dim = y.shape
    y_reshape = y.reshape(y_dim[0], np.prod(y_dim[1:]))
    xy_product = x @ y_reshape
    return xy_product.reshape((xy_product.shape[0], *y_dim[1:]))


def _rotate(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Rotate the axes of a multi-dimensional array to the right.

    The rotation of a nD array moves the first axis of a multi-dimensional array to the
    last position. This is equivalent to rotating the axes of the array to the right.
    It generalizes the transpose operation to nD arrays.

    Parameters
    ----------
    x: npt.NDArray[np.float64]
        A multi-dimensional array of shape `(n1, n2, ..., nk)`.

    Returns
    -------
    npt.NDArray[np.float64]
        A multi-dimensional array of shape `(n2, ..., nk, n1)`.

    Notes
    -----
    This function is adapted from [1]_.

    Examples
    --------
    >>> x = np.array([[[1, 2], [3, 4], [5, 6]], [[5, 6], [7, 8], [9, 0]]])
    >>> _rotate(x)
    array([
        [[1, 5],[2, 6]],
        [[3, 7],[4, 8]],
        [[5, 9],[6, 0]]
    ])

    References
    ----------
    ..[1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized Linear Array
        Models with Applications to Multidimensional Smoothing. Journal of the Royal
        Statistical Society. Series B (Statistical Methodology) 68, pp.259--280.

    """
    return np.moveaxis(x, 0, -1)


def _rotated_h_transform(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute the rotated H-transform of a nD array `y` with respect to a 2D array `x`.

    The rotated H-transform of a nD array `y` with respect to a 2D array `x` is a nD
    array `z`. This function performed a H-transform of `x` and `y`, and then rotate
    the results.

    Parameters
    ----------
    x: npt.NDArray[np.float64]
        A 2D array of shape `(n, m)`.
    y: npt.NDArray[np.float64]
        A nD array of shape `(m, n1, n2, ..., nk)`.

    Notes
    -----
    This function is adapted from [1]_.

    Returns
    -------
    npt.NDArray[np.float64]
        A nD array of shape `(n1, n2, ..., nk, m)`.

    Examples
    --------
    >>> x = np.array([[1, 2, 3]])
    >>> y = np.array([[1, 2], [3, 4], [5, 6]])
    >>> _rotated_h_transform(x, y)
    array([
        [22],
        [28]
    ])

    References
    ----------
    ..[1] Currie, I. D., Durban, M., Eilers, P. H. C. (2006), Generalized Linear Array
        Models with Applications to Multidimensional Smoothing. Journal of the Royal
        Statistical Society. Series B (Statistical Methodology) 68, pp.259--280.

    """
    return _rotate(_h_transform(x, y))


def _create_permutation(p: int, k: int) -> npt.NDArray[np.float64]:
    """
    Create a permutation array for a given number of factors and levels.

    This function creates a permutation array for a given number of factors `p` and
    levels `k`. The resulting array is a 1D array of shape `(k*p,)` that contains the
    indices of all possible combinations of `p` factors with `k` levels each.

    Parameters
    ----------
    p: int
        The number of factors.
    k: int
        The number of levels.

    Returns
    -------
    npt.NDArray[np.float64]
        A 1D array of shape `(k*p,)` that contains the indices of all possible
        combinations of `p` factors with `k` levels each.

    Examples
    --------
    >>> np.tile(np.arange(3), 2)
    array([0, 1, 2, 0, 1, 2])
    >>> _create_permutation(3, 2)
    array([0, 3, 1, 4, 2, 5])
    >>> np.repeat(np.arange(3), 2)
    array([0, 0, 1, 1, 2, 2])
    >>> _create_permutation(2, 3)
    array([0, 2, 4, 1, 3, 5])

    """
    a = np.arange(0, k)
    b = np.arange(0, p)
    m = np.add.outer(a * p, b)
    return m.flatten("F")


def _tensor_product_penalties(
    penalties: list[npt.NDArray[np.float64]],
) -> list[npt.NDArray[np.float64]]:
    """
    Compute the tensor product of a list of penalty matrices.

    The `_tensor_product_penalties` function computes the tensor product of a list of
    penalty matrices. The resulting list contains the tensor product of each penalty
    matrix with itself and with the identity matrices of the same size as the other
    penalty matrices. If a penalty matrix is square, its tensor product with itself is
    symmetrized by taking the mean of the matrix and its transpose.

    Parameters
    ----------
    penalties: list[npt.NDArray[np.float64]]
        A list of penalty matrices.

    Returns
    -------
    list[npt.NDArray[np.float64]]
        A list of tensor product matrices.

    Notes
    -----
    This function is adapted from the function `tensor.prod.penalties` in [1]_.

    Examples
    --------
    >>> penalties = [
        np.array([
            [ 1., -1.,  0.],
            [-1.,  2., -1.],
            [ 0., -1.,  2.]
        ]),
        np.array([
            [ 1., -1.],
            [-1.,  2.]
        ])
    ]
    >>> _tensor_product_penalties(penalties)
    [
         array([
             [ 1.,  0., -1., -0.,  0.,  0.],
             [ 0.,  1., -0., -1.,  0.,  0.],
             [-1., -0.,  2.,  0., -1., -0.],
             [-0., -1.,  0.,  2., -0., -1.],
             [ 0.,  0., -1., -0.,  2.,  0.],
             [ 0.,  0., -0., -1.,  0.,  2.]
         ]),
         array([
             [ 1., -1.,  0., -0.,  0., -0.],
             [-1.,  2., -0.,  0., -0.,  0.],
             [ 0., -0.,  1., -1.,  0., -0.],
             [-0.,  0., -1.,  2., -0.,  0.],
             [ 0., -0.,  0., -0.,  1., -1.],
             [-0.,  0., -0.,  0., -1.,  2.]
         ]
     )]

    References
    ----------
    .. [1] Wood, S. (2023). mgcv: Mixed GAM Computation Vehicle with Automatic
        Smoothness Estimation.

    """
    n_penalties = len(penalties)
    eyes = [np.eye(penalty.shape[1]) for penalty in penalties]

    if n_penalties == 1:
        return penalties[0]
    else:
        tensors_list = []
        for idx in range(n_penalties):
            left = penalties[0] if idx == 0 else eyes[0]
            for j in range(1, n_penalties):
                right = penalties[j] if idx == j else eyes[j]
                left = np.kron(left, right)
            # Make sure the matrix is symmetric
            if left.shape[0] == left.shape[1]:
                left = (left + left.T) / 2
            tensors_list.append(left)
        return tensors_list


########################################################################################
# Inner functions for the PSplines class.
def _fit_one_dimensional(
    data: npt.NDArray[np.float64],
    basis: npt.NDArray[np.float64],
    sample_weights: Optional[npt.NDArray[np.float64]] = None,
    penalty: float = 1.0,
    order_penalty: int = 2,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Fit a one-dimensional P-splines model to the given data.

    The `_fit_one_dimensional` function fits a one-dimensional P-splines model to the
    given data using a basis matrix and an optional penalty matrix. The function returns
    a dictionary containing the fitted values, the estimated coefficients, and the hat
    matrix.

    Parameters
    ----------
    data : npt.NDArray[np.float64]
        A one-dimensional array of shape `(n_obs,)` containing the response variable
        values.
    basis : npt.NDArray[np.float64]
        A two-dimensional array of shape `(n_obs, n_basis)` containing the basis matrix.
    sample_weights : Optional[npt.NDArray[np.float64]], optional
        A one-dimensional array of shape `(n_obs,)` containing the weights for each
        observation. If not provided, all observations are assumed to have equal weight.
    penalty : float, optional
        The penalty parameter for the P-splines model.
    order_penalty : int, optional
        The order of the penalty difference matrix.

    Returns
    -------
    Dict[str, npt.NDArray[np.float64]]
        A dictionary containing the following keys:
        - 'y_hat': A one-dimensional array of shape `(n_obs,)` containing the fitted
        values.
        - 'beta_hat': A one-dimensional array of shape `(n_basis,)` containing the
        estimated coefficients.
        - 'hat_matrix': A one-dimensional array of shape `(n_obs,)` containing the
        diagonal of the hat matrix.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> basis = np.array([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [1, 4, 9, 16, 25]])
    >>> _fit_one_dimensional(data, basis)
    {
        'y_hat': array([1., 2., 3., 4., 5.]),
        'beta_hat': array([1., 1., 1.]),
        'hat_matrix': array([0.2, 0.2, 0.2, 0.2, 0.2])
    }

    """
    # Get parameters.
    n_obs = len(data)
    n_basis = basis.shape[1]

    # Construct the penalty.
    pen_mat = np.diff(np.eye(n_basis), n=order_penalty, axis=0)

    # Build the different part of the model.
    if sample_weights is None:
        sample_weights = np.ones(n_obs)
    weight_mat = np.diag(sample_weights)

    bwb_mat = basis.T @ weight_mat @ basis
    pen_mat = penalty * pen_mat.T @ pen_mat
    bwy_mat = basis.T @ weight_mat @ data

    # Fit the model
    inv_mat = np.linalg.pinv(bwb_mat + pen_mat)
    beta_hat = inv_mat @ bwy_mat
    y_hat = basis @ beta_hat

    # Compute the hat matrix
    hat_matrix = np.diag(basis @ inv_mat @ basis.T @ weight_mat)

    return {"y_hat": y_hat, "beta_hat": beta_hat, "hat_matrix": hat_matrix}


def _fit_n_dimensional(
    data: npt.NDArray[np.float64],
    basis_list: List[npt.NDArray[np.float64]],
    sample_weights: Optional[npt.NDArray[np.float64]] = None,
    penalty: tuple[float, ...] = 1.0,
    order_penalty: int = 2,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Fit an N-dimensional P-splines model to the given data.

    The `_fit_n_dimensional` function fits an N-dimensional P-splines model to the given
    data using a list of basis matrices and an optional penalty matrix. The function
    returns a dictionary containing the fitted values, the estimated coefficients, and
    the hat matrix.

    Parameters
    ----------
    data : npt.NDArray[np.float64]
        An N-dimensional array of shape `(n1, n2, ..., nk)` containing the response
        variable values.
    basis_list : List[npt.NDArray[np.float64]]
        A list of two-dimensional arrays of shape `(n1, m1), (n2, m2), ..., (nk, mk)`
        containing the basis matrices for each dimension.
    sample_weights : Optional[npt.NDArray[np.float64]], optional
        An N-dimensional array of shape `(n1, n2, ..., nk)` containing the weights for
        each observation. If not provided, all observations are assumed to have equal
        weight.
    penalty : tuple[float, ...], optional
        A tuple of penalty parameters for each dimension.
    order_penalty : int, optional
        The order of the penalty difference matrix.

    Returns
    -------
    Dict[str, npt.NDArray[np.float64]]
        A dictionary containing the following keys:
        - 'y_hat': An N-dimensional array of shape `(n1, n2, ..., nk)` containing the
        fitted values.
        - 'beta_hat': An N-dimensional array of shape `(m1, m2, ..., mk)` containing the
        estimated coefficients.
        - 'hat_matrix': A zero value.

    Examples
    --------
    >>> data = np.array([[1, 2], [3, 4]])
    >>> basis_list = [np.array([[1, 1], [1, 2]]), np.array([[1, 1], [2, 3]])]
    >>> _fit_n_dimensional(data, basis_list)
    {
        'y_hat': array([[1., 2.], [3., 4.]]),
        'beta_hat': array([[1., 1.], [1., 1.]]),
        'hat_matrix': 0
    }

    """
    n = tuple(basis.shape[1] for basis in basis_list)
    RT = [_row_tensor(basis) for basis in basis_list]

    XWX = _rotated_h_transform(RT[0].T, sample_weights)
    for idx in np.arange(1, len(RT)):
        XWX = _rotated_h_transform(RT[idx].T, XWX)
    XWX = (
        XWX.reshape(np.repeat(n, 2))
        .transpose(_create_permutation(2, len(n)))
        .reshape((np.prod(n), np.prod(n)))
    )

    # Penalty
    E = [np.eye(i) for i in n]
    D = [np.diff(i, n=1, axis=0) for i in E]
    DD = [d.T @ d for d in D]
    PP = _tensor_product_penalties(DD)

    P = np.sum([l * P for (l, P) in zip(penalty, PP)], axis=0)

    # Last part of the equation
    R = _rotated_h_transform(basis_list[0].T, data * sample_weights)
    for idx in np.arange(1, len(basis_list)):
        R = _rotated_h_transform(basis_list[idx].T, R)
    R = R.reshape(np.prod(n))

    # Fit
    fit = np.linalg.lstsq(XWX + P, R, rcond=None)
    A = fit[0].reshape(n)
    Zhat = _rotated_h_transform(basis_list[0], A)
    for idx in np.arange(1, len(basis_list)):
        Zhat = _rotated_h_transform(basis_list[idx], Zhat)

    # Compute the H matrix
    hat_matrix = 0
    return {"y_hat": Zhat, "beta_hat": A, "hat_matrix": hat_matrix}


########################################################################################
# class PSplines


class PSplines:
    r"""P-Splines Smoothing.

    Parameters
    ----------
    n_segments: int, defualt=10
        The number of evenly spaced segments.
    degree: int, default=3
        The number of the degree of the basis.
    order_penalty: int, default=2
        The number of the order of the difference penalty.
    order_derivative: int, default=0
        The order of the derivative to compute.

    Attributes
    ----------
    y_hat: npt.NDArray[np.float64]
    beta_hat: npt.NDArray[np.float64]
    parameters: dict

    Notes
    -----
    This code is adapted from _[2].

    References
    ----------
    .. [1] Eilers, P., Marx, B.D., (2021) Practical Smoothing: The Joys of
        P-splines. Cambridge University Press, Cambridge.
    .. [2] Eilers, P., Marx, B., Li, B., Gampe, J., Rodriguez-Alvarez, M.X., (2023)
        JOPS: Practical Smoothing with P-Splines.

    """

    def __init__(
        self,
        n_segments: int = 10,
        degree: int = 3,
        order_penalty: int = 2,
        order_derivative: int = 0,
    ) -> None:
        """Initializa PSplines object."""
        self.n_segments = n_segments
        self.degree = degree
        self.order_penalty = order_penalty
        self.order_derivative = order_derivative

    def fit(
        self,
        y: npt.NDArray[np.float64],
        x: npt.NDArray[np.float64],
        sample_weights: npt.NDArray[np.float64] = None,
        penalty: float = 1.0,
    ) -> None:
        """Fit the model.

        Parameters
        ----------
        y: npt.NDArray[np.float64], shape = (n_samples,)
            Target values.
        x: npt.NDArray[np.float64], shape = (n_samples,)
            Training data.
        sample_weights: npt.NDArray[np.float64], shape = (n_samples,)
            Indiviudal weights for each sample.
        penalty: float, default=1.0
            The (positive) number for the tuning parameter for the penalty.

        Returns
        -------
        self: object
            Fitted estimator.

        Examples
        --------
        x = np.linspace(0, 4, 100)
        y = 0.5 * np.sin(x**2) + np.random.normal(loc=0, scale=0.05, size=len(x))
        PSplines(n_segments=50).fit(y, x, penalty=0.05)

        """
        m = len(x)
        basis_mat = _basis_bsplines(
            argvals=x,
            n_functions=self.n_segments + self.degree,
            degree=self.degree,
            domain_min=np.min(x),
            domain_max=np.max(x),
        ).T

        if len(y.shape) == 1:
            res = _fit_one_dimensional(
                data=y,
                basis=basis_mat,
                sample_weights=sample_weights,
                penalty=penalty,
                order_penalty=2,
            )
        else:
            res = _fit_n_dimensional(
                data=y,
                basis_list=basis_mat,
                sample_weights=sample_weights,
                penalty=penalty,
                order_penalty=2,
            )

        y_hat = res["y_hat"]
        beta_hat = res["beta_hat"]
        hat_mat = res["hat_matrix"]
        # Cross-validation and dispersion
        r = (y - y_hat) / (1 - hat_mat)
        cv = np.sqrt(np.mean(np.power(r, 2)))
        ed = np.sum(hat_mat)
        sigma = np.sqrt(np.sum(np.power(y - y_hat, 2)) / (m - ed))
        ed_resid = m - ed

        if self.order_derivative > 0:
            basis_mat_der = _basis_bsplines(
                argvals=x,
                n_functions=self.n_segments + self.degree - self.order_derivative,
                degree=self.degree - self.order_derivative,
                domain_min=np.min(x),
                domain_max=np.max(x),
            ).T
            num = np.diff(beta_hat, n=self.order_derivative)
            deno = ((np.max(x) - np.min(x)) / self.n_segments) ** self.order_derivative
            alpha_der = num / deno
            y_hat = basis_mat_der @ alpha_der

        # Export results
        self.y_hat = y_hat
        self.beta_hat = beta_hat
        self.parameters = {"sigma": sigma, "cv": cv, "effdim": ed, "ed_resid": ed_resid}
        return self

    def predict(self, x: Optional[npt.NDArray[np.float64]] = None) -> None:
        """Predict using the model.

        Parameters
        ----------
        x: npt.NDArray[np.float64], shape = (n_samples,)
            New samples.

        Returns
        -------
        npt.NDArray[np.float64], shape = (n_samples,)
            Return predicted values.

        """
        if x is None:
            return self.y_hat

        basis_mat = _basis_bsplines(
            argvals=x,
            n_functions=self.n_segments + self.degree,
            degree=self.degree,
            domain_min=np.min(x),
            domain_max=np.max(x),
        ).T
        y_estim = basis_mat @ self.beta_hat

        # SE bands on a grid using QR
        l_mat = solve_triangular(self.parameters["R"].T, basis_mat.T, lower=True)
        v2 = self.parameters["sigma"] ** 2 * np.sum(l_mat * l_mat, axis=0)
        se_eta = np.sqrt(v2)
        return y_estim, se_eta
