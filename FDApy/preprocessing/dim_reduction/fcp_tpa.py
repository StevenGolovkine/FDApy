#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Functional Canonical Polyadic-Tensor Power Algorithm
----------------------------------------------------

"""
import numpy as np
import numpy.typing as npt
import warnings

from typing import Dict, List, Tuple

from numpy.linalg import norm
from scipy.optimize import minimize_scalar

from ...representation.argvals import DenseArgvals
from ...representation.values import DenseValues
from ...representation.functional_data import DenseFunctionalData
from ...misc.utils import _eigh


##############################################################################
# Utility functions

def _initialize_vectors(
    shape: Tuple[int, int, int]
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    r"""Init u, v and w in the FCP-TPA algorithm.

    Parameters
    ----------
    shape: Tuple[int, int, int]
        Shape of the dataset. It should be in the format :math:`(N, M_1, M_2)`.

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray, npt.NDArray]
        A tuple containing normed vectors for the initialization of the FCP-TPA
        algorithm.

    """
    vectors = [np.random.uniform(-1, 1, dimension) for dimension in shape]
    return tuple(vector / norm(vector) for vector in vectors)


def _initalize_output(
    shape: Tuple[int, int, int],
    n_components: int
) -> Tuple[npt.NDArray, List[npt.NDArray]]:
    """Init coefficients and u, v and w eigenvectors matrices.

    Parameters
    ----------
    shape: Tuple[int, int, int]
        Shape of the dataset. It should be in the format :math:`(N, M_1, M_2)`.
    n_components: int
        Number of components to retain.

    Returns
    -------
    Tuple[npt.NDArray, List[npt.NDArray]]
        A tuple containing initialized matrices for the results of the FCP-TPA
        algorithm.

    """
    coefficients = np.zeros(n_components)
    matrices = [np.zeros((dimension, n_components)) for dimension in shape]
    return coefficients, matrices


def _eigendecomposition_penalty_matrices(
    penalty_matrices: Dict[str, npt.NDArray[np.float64]]
) -> Dict[str, Tuple[npt.NDArray, npt.NDArray]]:
    """Compute eigendecomposition of penalty matrices in the FCP-TPA algorithm.

    Parameters
    ----------
    penalty_matrices: Dict[str, npt.NDArray[np.float64]]
        A dictionary with entries :math:`v` and :math:`w`, containing a
        roughness penalty matrix for each direction of the image. The algorithm
        does not induce smoothness along observations.

    Returns
    -------
    Dict[str, Tuple[npt.NDArray, npt.NDArray]]
        A dictionary where each entry contains the eigenvalues and eigenvectors
        of the penalty matrix for each dimension of the image. The eigenvalues
        are sorted in descending order.

    """
    return {name: _eigh(matrix) for name, matrix in penalty_matrices.items()}


def _gcv(
    alpha: float,
    dimension_length: int,
    vector: npt.NDArray[np.float64],
    smoother: float,
    rayleigh: npt.NDArray[np.float64]
) -> float:
    r"""Generalized cross-validation for the FCP-TPA algortihm.

    This function calculates the generalized cross-validation criterion for the
    smoothing parameters alpha that is used in the FCP-TPA algorithm [1]_. The
    code is adapted from [2]_. It corresponds to Equations (19) and (20) in
    [3]_.

    Parameters
    ----------
    alpha: float
        The current value of the smoothing parameter. It corresponds to
        :math:`\alpha_u` and :math:`\alpha_v` in Equations (19) and (20) in
        [3]_.
    dimension_length: int
        The length of the dimension, for which the smoothing parameter is to
        be optimized. It corresponds to :math:`m` and :math:`n` in Equations
        (19) and (20) in [3]_.
    vector: npt.NDArray[np.float64], shape=(dimension_length,)
        Solutions to the least square problem. It corresponds to
        :math:`Xv / ||v||^2` and :math:`X^\top u / ||u||^2` in Equations (19)
        and (20) in [3]_.
    smoother: float
        Nornalization parameter. It corresponds to :math:`S_u` and :math:`S_v`
        in Equations (19) and (20) in [3]_.
    rayleigh: npt.NDArray[np.float64], shape=(dimension_length,)
        A vector containing the eigenvalues of the penalty matrix corresponding
        to the current image direction. It corresponds to the Rayleight
        quotients :math:`\mathcal{R}_u(u)` and :math:`\mathcal{R}_v(v)` in
        Equations (19) and (20) in [3]_.

    Returns
    -------
    float
        The value of the GCV criterion.

    References
    ----------
    .. [1] Allen G. (2013), Multi-way Functional Principal Components Analysis,
        IEEE International Workshop on Computational Advances in Multi-Sensor
        Adaptive Processing.
    .. [2] Happ-Kurz C. (2020) Object-Oriented Software for Functional Data,
        Journal of Statistical Software, 93(5): 1--38.
    .. [3] Huang J. Z., Shen H. and Buja A. (2009) The Analysis of Two-Way
        Functional Data Using Two-Way Regularized Singular Value Decomposition,
        Journal of the American Statistical Association, 104(488): 1609--1620.

    """
    shrinking = smoother / (1 + alpha * rayleigh)
    num = np.sum(np.power(((1 - shrinking) * vector), 2)) / dimension_length
    deno = np.power((1 - np.sum(shrinking) / dimension_length), 2)
    return num / deno


def _find_optimal_alpha(
    alpha_range: Tuple[float, float],
    data: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    alpha: float,
    penalty_matrix: npt.NDArray[np.float64],
    eigencomponents: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    formula: str
) -> float:
    r"""Find the optimal smoothing parameters in FCP-TPA using GCV.

    This function find the optimal smoothing parameters :math:`\alpha_v` (or
    :math:`\alpha_w`) for the two image directions (v and w) in the FCP_TPA
    algorithm [1]_ based on generalized cross-validation, which is nested in
    the tensor power algorithm. Given a range of possible values of
    :math:`\alpha_v` (or :math:`\alpha_w`, respectively), the minimum is found
    by optimizing the GCV criterion using the function ``minimize_scalar`` from
    the module ``scipy.optimize``. The code is adapted from [2]_.

    Parameters
    ----------
    alpha_range: Tuple[float, float]
        A tuple with two elements, containing the minimal and maximal
        values for the smoothing parameter that is to be optimized. It
        corresponds to minimal and maximal values of :math:`\alpha_u` and
        :math:`\alpha_v` in Equations (19) and (20) in [3]_.
    data: npt.NDArray[np.float64], shape=(n_obs, m_1, m_2)
        The tensor containing the data of dimension
        :math:`n_{obs} \times m_1 \times m_2`. It corresponds to
        :math:`\hat{\mathcal{X}}` in Algorithm in [1]_.
    u: npt.NDArray[np.float64], shape=(n_obs,)
        The current value of the eigenvectors :math:`u_k` (not
        normalized) of dimensions :math:`n_{obs}`. It corresponds to
        :math:`u_k` in Algorithm in [1]_.
    v: npt.NDArray[np.float64]
        The current value of the eigenvectors :math:`v_k` (or :math:`w_k`) (not
        normalized) of dimensions :math:`m_1` (or :math:`m_2`). It corresponds
        to :math:`v_k` (or :math:`w_k`) in Algorithm in [1]_.
    alpha: float
        The current value of the smoothing parameter for the other image
        direction (:math:`\alpha_w` if the optimization is performed with
        respect to the vector :math:`v_k` and :math:`\alpha_v` if the
        optimization is performed with respect to the vector :math:`w_k`),
        which is kept as fixed. It corresponds to :math:`\alpha_u` and
        :math:`\alpha_v` in Equations (19) and (20) in [3]_.
    penalty_matrix: npt.NDArray[np.float64], shape=(m, m)
        A matrix of dimension :math:`m \times m`, the penalty matrix for the
        other image direction. It corresponds to :math:`\Omega_v` and
        :math:`\Omega_u` in Equations (17) and (18) in [3]_.
    eigencomponents: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        A tuple containing the eigenvalues and eigenvectors of the penalty
        matrix for the image direction for which the optimal smoothing
        parameter is to be found. The shape of the eigenvalues array is
        :math:`m` and the shape of the eigenvectors array is
        :math:`m \times m`. The eigenvalues corresponds to the Rayleight
        quotients :math:`\mathcal{R}_u(u)` and :math:`\mathcal{R}_v(v)` in
        Equations (19) and (20) in [3]_.
    formula: str
        The formula to be passed to the ``np.einsum`` function regarding the
        direction to optimize.

    Returns
    -------
    float
        The optimal smoothing parameter :math:`\alpha` found by optimizing the
        GCV criterion within the given range of possible values.

    Notes
    -----
    The code has been tested for the version 1.10.0 of ``scipy``.

    References
    ----------
    .. [1] Allen G. (2013), Multi-way Functional Principal Components Analysis,
        IEEE International Workshop on Computational Advances in Multi-Sensor
        Adaptive Processing.
    .. [2] Happ-Kurz C. (2020) Object-Oriented Software for Functional Data,
        Journal of Statistical Software, 93(5): 1--38.
    .. [3] Huang J. Z., Shen H. and Buja A. (2009) The Analysis of Two-Way
        Functional Data Using Two-Way Regularized Singular Value Decomposition,
        Journal of the American Statistical Association, 104(488): 1609--1620.

    """
    eigenvalues, eigenvectors = eigencomponents

    temp = np.einsum(formula, u, v, data)
    vector = np.dot(eigenvectors.T, temp) / (norm(u) * norm(v))
    v_w_v = np.dot(v.T, np.dot(penalty_matrix, v))
    smoother = 1 / (1 + alpha * v_w_v / norm(v))

    results = minimize_scalar(
        _gcv,
        args=(len(eigenvalues), vector, smoother, eigenvalues),
        bounds=alpha_range
    )
    return results.x


def _compute_denominator(
    a: npt.NDArray,
    alpha: float,
    penalty_matrix: npt.NDArray
) -> float:
    r"""Compute denominator of equations (17) and (18) in [1]_.

    This function computes the denominator of equations (17) and (18) in [1]_,
    which is, for a vector :math:`a`:

    .. math::

        a^{\top} (\mathbf{I} + \alpha_{a}\Omega_a) a.

    References
    ----------
    .. [1] Huang J. Z., Shen H. and Buja A. (2009) The Analysis of Two-Way
        Functional Data Using Two-Way Regularized Singular Value Decomposition,
        Journal of the American Statistical Association, 104(488): 1609--1620.

    """
    return np.dot(a.T, a + alpha * np.dot(penalty_matrix, a))


def _update_vector(
    data: npt.NDArray[np.float64],
    vectors: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    penalty_matrix: npt.NDArray,
    alpha: float,
    denominator: float,
    formula: str
) -> npt.NDArray:
    r"""Update individual vector in FCP-TPA.

    This function is used to compute the step (2.a.i), (2.a.ii) and (2.a.iii)
    in the FCP-TPA algortihm in [1]_. The vectors :math:`u, v` and :math:`w`
    are computed using equations (17) and (18) in [2]_.

    Parameters
    ----------
    data: npt.NDArray[np.float64], shape=(n_obs, m_1, m_2)
        Data as an array of shape :math:`(n_obs, m_1, m_2)`.
    vectors: Tuple[npt.NDArray, npt.NDArray, npt.NDArray]
        Vectors. The first element of the tuple is the vector to update, the
        other two are the remaining dimensions of the tensor.
    penalty_matrix: npt.NDArray[np.float64]
        A roughness penalty matrix for the direction of the image to update.
    alpha: float
        The smoothing parameter for the dimension to be updated.
    denominator: float
        The denominator of equations (17) and (18) in [2]_.
    formula: str
        The formula to be passed to the ``np.einsum`` function regarding the
        direction to optimize.

    Returns
    -------
    npt.NDArray
        The updated vector.

    References
    ----------
    .. [1] Allen G. (2013), Multi-way Functional Principal Components Analysis,
        IEEE International Workshop on Computational Advances in Multi-Sensor
        Adaptive Processing.
    .. [2] Huang J. Z., Shen H. and Buja A. (2009) The Analysis of Two-Way
        Functional Data Using Two-Way Regularized Singular Value Decomposition,
        Journal of the American Statistical Association, 104(488): 1609--1620.

    """
    a = np.eye(len(vectors[0])) + alpha * penalty_matrix
    b = np.einsum(formula, vectors[1], vectors[2], data)
    return np.linalg.solve(a, b) / denominator


def _update_components(
    data: npt.NDArray[np.float64],
    vectors: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    penalty_matrices: Dict[str, npt.NDArray[np.float64]],
    alphas: Dict[str, Tuple[float, float]],
    alpha_range: Dict[str, Tuple[float, float]],
    eigens: Dict[str, Tuple[npt.NDArray, npt.NDArray]]
) -> Tuple[Tuple[npt.NDArray], Dict[str, float]]:
    r"""Update the components in FCP-TPA.

    This function corresponds to one pass of the step (2.a) in the FCP-TPA
    algorithm in [1]_. The vectors :math:`u, v` and :math:`w` are computed
    using equations (17) and (18) in [2]_, and the smoothing parameters
    :math:`\alpha_u` and :math:`\alpha_v` are updated using the GCV criteria
    defined in equations (19) and (20) in [2]_.

    Parameters
    ----------
    data: npt.NDArray[np.float64], shape=(n_obs, m_1, m_2)
        Data as an array of shape :math:`(n_obs, m_1, m_2)`.
    vectors: Tuple[npt.NDArray, npt.NDArray, npt.NDArray]
        Vectors
    penalty_matrices: Dict[str, npt.NDArray[np.float64]]
        A dictionary with entries :math:`v` and :math:`w`, containing a
        roughness penalty matrix for each direction of the image. The
        algorithm does not induce smoothness along observations.
    alphas: Dict[str, Tuple[float, float]]
        A dictionary with entries :math:`v` and :math:`w`, containing the
        smoothing parameters for both dimension of the images.
    alpha_range: Dict[str, Tuple[float, float]]
        A dictionary with entries :math:`v` and :math:`w`, containing the
        range of smoothness parameters :math:`\alpha_{v_k}, \alpha_{w_k}`
        as a tuple.
    eigens: Dict[str, Tuple[npt.NDArray, npt.NDArray]]
        Eigendecomposition of the penalty matrices.

    Returns
    -------
    Tuple[Tuple[npt.NDArray], Dict[str, float]]
        The updated parameters.

    References
    ----------
    .. [1] Allen G. (2013), Multi-way Functional Principal Components Analysis,
        IEEE International Workshop on Computational Advances in Multi-Sensor
        Adaptive Processing.
    .. [2] Huang J. Z., Shen H. and Buja A. (2009) The Analysis of Two-Way
        Functional Data Using Two-Way Regularized Singular Value Decomposition,
        Journal of the American Statistical Association, 104(488): 1609--1620.

    """
    u, v, w = vectors

    # Update u
    v_cross = _compute_denominator(v, alphas['v'], penalty_matrices['v'])
    w_cross = _compute_denominator(w, alphas['w'], penalty_matrices['w'])
    u = _update_vector(
        data, (u, v, w),
        penalty_matrix=0, alpha=0,  # No smoothing for u.
        denominator=v_cross * w_cross, formula='i, j, kij -> k'
    )

    # Update v
    u_cross = _compute_denominator(u, 0, 0)
    v = _update_vector(
        data, (v, u, w),
        penalty_matrix=penalty_matrices['v'], alpha=alphas['v'],
        denominator=u_cross * w_cross, formula='i, j, ikj -> k'
    )

    # Update alpha_v
    alpha_v = _find_optimal_alpha(
        alpha_range=alpha_range['v'],
        data=data, u=u, v=w,
        alpha=alphas['w'], penalty_matrix=penalty_matrices['w'],
        eigencomponents=eigens['v'], formula='i, j, ikj -> k'
    )

    # Update w
    v_cross = _compute_denominator(v, alpha_v, penalty_matrices['v'])
    w = _update_vector(
        data, (w, u, v),
        alpha=alphas['w'], penalty_matrix=penalty_matrices['w'],
        denominator=u_cross * v_cross, formula='i, j, ijk -> k'
    )

    # Update alpha_w
    alpha_w = _find_optimal_alpha(
        alpha_range=alpha_range['w'],
        data=data, u=u, v=v,
        alpha=alpha_v, penalty_matrix=penalty_matrices['v'],
        eigencomponents=eigens['w'], formula='i, j, ijk -> k'
    )

    alphas = {'v': alpha_v, 'w': alpha_w}
    vectors = (u, v, w)
    return vectors, alphas


##############################################################################
# Class FCPTPA

class FCPTPA():
    r"""Functional Canonical Polyadic - Tensor Power Algorithm (FCP-TPA).

    This module implements the Functional CP-TPA algorithm [1]_. This method
    computes an eigendecomposition of image observations, which can be
    interpreted as functions on a two-dimensional domain. We assume :math:`N`
    observations of 2D images with dimension :math:`M_1 \times M_2`. The
    results are given in a CANDECOMP/PARAFRAC (CP) model format

    .. math::

        X = \sum_{k = 1}^K c_k \cdot u_k \circ v_k \circ w_k

    where :math:`\circ` stands for the outer product, :math:`c_k` is a
    coefficient (scalar) and :math:`u_k, v_k, w_k` are eigenvectors for
    each direction of the tensor. In this representation, the outer product
    :math:`v_k \circ w_k` can be regarded as the :math:`k`-th eigenimage,
    while :math:`d_k \cdot u_k` represents the vector of individual scores for
    this eigenimage and each observation.

    The smoothness of the eigenvectors :math:`v_k, w_k` is induced by
    penalty matrices for both image directions, that are weighted by
    smoothing parameters :math:`\alpha_{v_k}, \alpha_{w_k}`. The
    eigenvectors :math:`u_k` are not smoothed, hence the algorithm does not
    induce smoothness along observations.

    Optimal smoothing parameters are found via a nested generalized cross
    validation [4]_. In each iteration of the TPA (tensor power algorithm),
    the GCV criterion is optimized via ``scipy.optimize`` on the intervals
    specified via ``alpha_range``.

    The FCP-TPA algorithm is an iterative algorithm. Convergence is assumed if
    the relative difference between the actual and the previous values are all
    below the tolerance level ``tolerance``. The tolerance level is increased
    automatically, if the algorithm has not converged after ``max_iteration``
    steps and if ``adapt_tolerance = TRUE``. If the algorithm did not converge
    after ``max_iteration`` steps steps, the function throws a warning. The
    code is adapted from [2]_ and [3]_.

    Parameters
    ----------
    n_components: int, default=5
        Number of components to be calculated.
    normalize: bool, default=False
        Should the results be normalied?

    Attributes
    ----------
    eigenvalues: npt.NDArray[np.float64], shape=(n_components,)
        The singular values corresponding to each of selected components.
    eigenfunctions: DenseFunctionalData
        Principal axes in feature space, representing the directions of
        maximum variance in the data.

    References
    ----------
    .. [1] Allen G. (2013) Multi-way Functional Principal Components Analysis,
        IEEE International Workshop on Computational Advances in Multi-Sensor
        Adaptive Processing.
    .. [2] Happ C. and Greven S. (2018) Multivariate Functional Principal
        Component Analysis for Data Observed on Different (Dimensional)
        Domains, Journal of the American Statistical Association, 113(522),
        649--659, DOI: 10.1080/01621459.2016.1273115.
    .. [3] Happ-Kurz C. (2020) Object-Oriented Software for Functional Data,
        Journal of Statistical Software, 93(5): 1--38.
    .. [4] Huang J. Z., Shen H. and Buja A. (2009) The Analysis of Two-Way
        Functional Data Using Two-Way Regularized Singular Value Decomposition,
        Journal of the American Statistical Association, 104(488): 1609--1620.

    """

    def __init__(
        self,
        n_components: int = 5,
        normalize: bool = False
    ) -> None:
        """Initialize FCPTPA object."""
        self.n_components = n_components
        self.normalize = normalize

    @property
    def n_components(self) -> int:
        """Getter for `n_components`."""
        return self._n_components

    @n_components.setter
    def n_components(self, new_n_components: int) -> None:
        self._n_components = new_n_components

    @property
    def normalize(self) -> bool:
        """Getter for `normalize`."""
        return self._normalize

    @normalize.setter
    def normalize(self, new_normalize: bool) -> None:
        self._normalize = new_normalize

    @property
    def eigenvalues(self) -> npt.NDArray[np.float64]:
        """Getter for `eigenvalues`."""
        return self._eigenvalues

    @property
    def eigenfunctions(self) -> DenseFunctionalData:
        """Getter for `eigenfunctions`."""
        return self._eigenfunctions

    def fit(
        self,
        data: DenseFunctionalData,
        penalty_matrices: Dict[str, npt.NDArray[np.float64]],
        alpha_range: Dict[str, Tuple[float, float]],
        tolerance: float = 1e-4,
        max_iteration: int = 15,
        adapt_tolerance: bool = True,
        verbose: bool = False
    ) -> None:
        r"""Fit the model on data.

        This function is used to fit a model on the data.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data used to estimate the eigencoponents. The dimension of
            its value parameter is :math:`N \times M_1 \times M_2`.
        penalty_matrices: Dict[str, npt.NDArray[np.float64]]
            A dictionary with entries :math:`v` and :math:`w`, containing a
            roughness penalty matrix for each direction of the image. The
            algorithm does not induce smoothness along observations.
        alpha_range: Dict[str, Tuple[float, float]]
            A dictionary with entries :math:`v` and :math:`w`, containing the
            range of smoothness parameters :math:`\alpha_{v_k}, \alpha_{w_k}`
            as a tuple.
        tolerance: float, default=1e-4
            A numeric value, giving the tolerance for relative error values
            in the algorithm. It is automatically multiplyed by 10 after
            ``max_iter`` steps, if ``adapt_tol = True``.
        max_iteration: int, default=15
            An integer, the maximal iteration steps. Can be doubled, if
            ``adapt_tol = True``.
        adapt_tolerance: bool, default=True
            If True, the tolerance is adapted (multiply by 10), if the
            algorithm has not converged after ``max_iter`` steps and another
            ``max_iter`` steps are allowed with the increased tolerance.
        verbose: bool, default=False
            If True, computational details are given on the standard output
            during the computation. Here for debug purpose.

        Examples
        --------
        Simulate some data.

        >>> kl = KarhunenLoeve(
        ...     basis_name='bsplines',
        ...     n_functions=5,
        ...     dimension='2D',
        ...     argvals={'input_dim_0': np.linspace(0, 1, 101)},
        ...     random_state=42
        ... )
        >>> kl.new(n_obs=50)
        >>> data = kl.data

        Define some parameters.

        >>> n_points = data.n_points
        >>> mat_v = np.diff(np.identity(n_points['input_dim_0']))
        >>> mat_w = np.diff(np.identity(n_points['input_dim_1']))

        Fit the FCP-TPA algorithm.

        >>> fcptpa = FCPTPA(n_components=10)
        >>> fcptpa.fit(
        ...     data,
        ...     penalty_matrices={
        ...         'v': np.dot(mat_v, mat_v.T),
        ...         'w': np.dot(mat_w, mat_w.T)
        ...     },
        ...     alpha_range={
        ...         'v': (1e-2, 1e2),
        ...         'w': (1e-2, 1e2)
        ...     },
        ...     tolerance=1e-4,
        ...     max_iteration=15,
        ...     adapt_tolerance=True
        ... )

        """
        # Get parameters
        values = data.values
        dimension = values.shape

        # Initialization
        vectors = _initialize_vectors(dimension)
        alphas = {idx: minimum for idx, (minimum, _) in alpha_range.items()}

        # Eigendecomposition of penalty matrix
        eigens = _eigendecomposition_penalty_matrices(penalty_matrices)

        # Initialization of the output
        coefficients, matrices = _initalize_output(
            dimension, self.n_components
        )

        # Loop over the number of wanted components
        for n_component in range(self.n_components):
            if verbose:
                print(f"\nComponents = {n_component}\n")

            # Initialize old versions
            vectors_old = tuple([np.zeros_like(vector) for vector in vectors])
            tolerance_old = tolerance

            # Number of iterations
            n_iter = 0

            # Repeat until convergence (defined by tolerance)
            while (
                any(
                    norm(vector - vector_old) / norm(vector) > tolerance
                    for vector, vector_old in zip(vectors, vectors_old)
                )
            ):
                vectors_old = vectors

                # Update components
                vectors, alphas = _update_components(
                    values,
                    vectors,
                    penalty_matrices,
                    alphas,
                    alpha_range,
                    eigens
                )

                n_iter = n_iter + 1
                if n_iter > max_iteration:
                    if adapt_tolerance and (n_iter < 2 * max_iteration):
                        tolerance = 10 * tolerance
                    else:
                        vectors_old = vectors
                        warnings.warn((
                            f'FCP-TPA algorithm did not converge; iteration '
                            f'for the component {n_component} stopped.'
                        ), UserWarning)

            if verbose:
                print(
                    f'Absolute error:\n'
                    f'u: {norm(vectors[0] - vectors_old[0])}, '
                    f'v: {norm(vectors[1] - vectors_old[1])}, '
                    f'w: {norm(vectors[2] - vectors_old[2])}, '
                    f'alpha_v: {alphas["v"]}, '
                    f'alpha_w: {alphas["w"]}.'
                )

            # Reset tolerance if necessary
            if adapt_tolerance and (n_iter >= max_iteration):
                tolerance = tolerance_old

            # Scale vector to have norm one
            vectors = tuple(vector / norm(vector) for vector in vectors)

            # Calculate results
            coefficients[n_component] = np.einsum(
                'ijk, i, j, k -> ...', values, *vectors
            )
            # Assign the vectors to the right matrix
            for matrix, vector in zip(matrices, vectors):
                matrix[:, n_component] = vector

            # Update the values
            values = values - (
                coefficients[n_component] *
                np.einsum('i, j, k -> ijk', *vectors)
            )

        # Save the results
        eigenimages = np.einsum('ik, jk -> kij', *matrices[1:])

        self._scores = np.einsum('j, ij -> ij', coefficients, matrices[0])
        self._eigenvalues = np.var(self._scores, axis=0)
        self._eigenfunctions = DenseFunctionalData(
            DenseArgvals(data.argvals), DenseValues(eigenimages)
        )

        if self.normalize:
            norm_data = self._eigenfunctions.norm(squared=False)
            new_eigenimages = (
                self._eigenfunctions.values / norm_data[:, None, None]
            )
            new_scores = self._scores * norm_data

            self._eigenvalues = self._eigenvalues * np.power(norm_data, 2)
            self._eigenfunctions.values = new_eigenimages
            self._scores = new_scores

    def transform(
        self,
        data: DenseFunctionalData,
        method: str = 'NumInt'
    ) -> npt.NDArray[np.float64]:
        """Apply dimension reduction to the data.

        Parameters
        ----------
        data: DenseFunctionalData
            Functional data object to be transformed. It has to be
            2-dimensional data.
        method: str, {'NumInt', 'FCPTPA'}
            Not used. To be compliant with other methods.

        Returns
        -------
        npt.NDArray[np.float64], shape=(n_obs, n_components)
            An array representing the projection of the data onto the basis of
            functions defined by the eigenimages.

        Examples
        --------
        Using the model fitted using the ``fit`` function.

        >>> scores = fcptpa.transform(data, 'NumInt')

        """
        if method == 'NumInt':
            if self.normalize:
                n_points = np.prod(data.n_points)
            else:
                n_points = 1
            return np.einsum(
                'ikl, jkl -> ij',
                data.values,
                self.eigenfunctions.values
            ) / n_points
        elif method == 'FCPTPA':
            return self._scores
        else:
            raise ValueError(
                f"Method {method} not implemented."
            )

    def inverse_transform(
        self,
        scores: npt.NDArray[np.float64]
    ) -> DenseFunctionalData:
        """Transform the data back to its original space.

        Return a DenseFunctionalData whose transform would be ``scores``.

        Parameters
        ----------
        scores: npt.NDArray[np.float64], shape=(n_obs, n_components)
            A set of coefficients to generate new data, where ``n_obs`` is the
            number of observations and ``n_components`` is the number of
            components.

        Returns
        -------
        DenseFunctionalData
            The transformation of the scores into the original space.

        Examples
        --------
        Using the model fitted using the ``fit`` function.

        >>> data_f = fcptpa.inverse_transform(scores)

        """
        argvals = self.eigenfunctions.argvals
        values = np.einsum(
            'ij, jkl -> ikl',
            scores,
            self.eigenfunctions.values
        )
        return DenseFunctionalData(DenseArgvals(argvals), DenseValues(values))
