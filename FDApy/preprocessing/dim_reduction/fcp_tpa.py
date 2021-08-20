#!/usr/bin/env python
# -*-coding:utf8 -*

"""Mdoule for FCPTPA class.

This module is used to implement the Functional CP-TPA algorithm [1]. This
method computes an eigendecomposition of image observations, which can be
interpreted as functions on a two-dimensional domain.

References
----------
[1] - G. I. Allen, "Multi-way Functional Principal Components Analysis", IEEE
International Workshop on Computational Advances in Multi-Sensor Adaptive
Processing, 2013.
[2] - C. Happ, S. Greven (2018): Multivariate Functional Principal Component
Analysis for Data Observed on Different (Dimensional) Domains. Journal of the
American Statistical Association, 113(522): 649-659 .
[3] - C. Happ-Kurz (2020): Object-Oriented Software for Functional Data.
Journal of Statistical Software, 93(5): 1-38 .

"""
import numpy as np
import warnings

from numpy.linalg import norm
from scipy.optimize import minimize

from ...representation.functional_data import DenseFunctionalData


##############################################################################
# Utility functions

def gcv(alpha, n, z, eta, lamb):
    r"""Generalized cross-validation for the FCP-TPA algortihm.

    This function calculates the generalized cross-validation criterion for the
    smoothing parameters alpha that is used in the FCP-TPA algorithm.

    Parameters
    ----------
    alpha: float
        The current value of the smoothing parameter.
    n: integer
        The length of the dimension, for which the smoothing parameter is to
        be optimized.
    z: np.ndarray
        A vector of length :math:`n`.
    eta: float
        A value.
    lamb: np.ndarray
        A vector of length :math:`n`, containing the eigenvalues of the
        penalty matrix corresponding to the current image direction.

    Returns
    -------
    gcv: float
        The value of the GCV criterion.

    References
    ----------
    - G. I. Allen, "Multi-way Functional Principal Components Analysis", IEEE
    International Workshop on Computational Advances in Multi-Sensor Adaptive
    Processing, 2013.

    """
    eig = 1 / (1 + alpha * lamb)
    res = np.sum((z * (1 - eta * eig))**2) / n
    return res / (1 - eta / n * np.sum(eig))**2


def find_opt_alpha(alpha_range, data, u, v, alpha, penal_mat, eival, dim):
    r"""Find the optimal smoothing parameters in FCP-TPA using GCV.

    Parameters
    ----------
    alpha_range: np.ndarray
        A numeric vector with two elements, containing the minimal and maximal
        values for the smoothing parameter that is to be optimized.
    data: np.ndarray
        The tensor containing the data of dimension
        :math:`N \times S_1 \times S_2`
    u: np.ndarray
        First eigenvectors
    v: np.ndarray
        Second eigenvectors
    alpha: float
        The current value of the smoothing parameter for the eigenvector v.
    penal_mat: np.ndarray
        The penalty matrix corresponding to the direction given by v.
    eival: tuple of np.array
        The eigenvectors and eigenvalues of the penalty matrix for the image
        direction for which the optimal smoothing parameter is to be found.
    dim: integer
        The direction to optimize. Should be 2 or 3.

    Returns
    -------
    alpha: float
        The optimal smoothing parameter found by optimizing the GCV criterion
        within the given range of possible values.

    References
    ----------
    - G. I. Allen (2013), "Multi-way Functional Principal Components Analysis",
    IEEE International Workshop on Computational Advances in Multi-Sensor
    Adaptive Processing.
    - J. Z. Huang, H. Shen and A. Buja (2009), "The Analysis of Two-Way
    Functional Data Using Two-Way Regularized Singular Value Decomposition".
    Journal of the American Statistical Association, Vol. 104, No. 488,
    1609 -- 1620.

    """
    evec, lamb = eival
    if dim == 2:
        b = np.einsum('i, j, ikj', u, v, data)
    elif dim == 3:
        b = np.einsum('i, j, ijk', u, v, data)
    else:
        raise ValueError(f"The direction can not be {dim}.")

    z = np.dot(evec.T, b) / (norm(u) * norm(v))
    vv = np.dot(v.T, np.dot(penal_mat, v))
    eta = 1 / (1 + alpha * vv / norm(v))

    res = minimize(gcv, x0=min(alpha_range), args=(len(lamb), z, eta, lamb),
                   bounds=[(min(alpha_range), max(alpha_range))]).x
    return res


##############################################################################
# Class FCPTPA

class FCPTPA():
    """Functional Canonical Polyadic - Tensor Power Algorithm (FCP-TPA).

    Implement the Functional CP-TPA algorithm. This method computes an
    eigendecomposition of image observations, which can be interpreted as
    functions on a two-dimensional domain.

    Parameters
    ----------
    n_components: int, default=None
        Number of components to be calculated.

    """

    def __init__(self, n_components=None):
        """Initialize FCPTPA object."""
        self.n_components = n_components

    def fit(self, data, penal_mat, alpha_range, tol=1e-4, max_iter=15,
            adapt_tol=True, verbose=False):
        r"""Fit the model on data.

        This function is used to fit a model on the data.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data. The dimension of its value parameter is
            :math:`N \times S_1 \times S_2`.
        penal_mat: dict
            A dictionary with entries :math:`v` and :math:`w`, containing a
            roughness penalty matrix for each direction of the image.
        alpha_range: dict
            A dictionary of length 2 with entries :math:`v` and :math:`w`,
            containing the range of smoothness parameters to test for each
            direction.
        tol: float, default=1e-4
            A numeric value, giving the tolerance for relative error values
            in the algorithm. It is automatically multiplued by 10 after
            `max_iter` steps, if `adapt_tol = True`.
        max_iter: int, default=15
            An integer, the maximal iteration steps. Can be doubled, if
            `adapt_tol = True`.
        adapt_tol: bool, default=True
            If True, the tolerance is adapted (multiply by 10), if the
            algorithm has not converged after `max_iter` steps and another
            `max_iter` steps are allowed with the increased tolerance.
        verbose: bool, default=False
            If True, computational details are given on the standard output
            during the computation.

        Example
        -------
        >>> penal_mat = dict(v = np.array(...),
                                 w = np.array(...))
        >>> alpha_range = dict(v = np.array([1e-4, 1e4]),
                               w = np.array([1e-4, 1e4]))

        """

        # Get the values and dimension
        values = data.values
        dim = values.shape

        # Initialization vectors
        u = np.random.uniform(low=-1, high=1, size=dim[0])
        u = u / norm(u)
        v = np.random.uniform(low=-1, high=1, size=dim[1])
        v = v / norm(v)
        w = np.random.uniform(low=-1, high=1, size=dim[2])
        w = w / norm(w)

        # Initialization smoothing parameters
        alpha_v = min(alpha_range['v'])
        alpha_w = min(alpha_range['w'])

        # Eigendecomposition of penalty matrix
        eigen_v = np.linalg.eigh(penal_mat['v'])
        eigen_w = np.linalg.eigh(penal_mat['w'])

        # Initialization of diagonal matrices
        iden_v = np.identity(dim[1])
        iden_w = np.identity(dim[2])

        # Initialization of the output
        coef = np.zeros(self.n_components)
        mat_u = np.zeros((dim[0], self.n_components))
        mat_v = np.zeros((dim[1], self.n_components))
        mat_w = np.zeros((dim[2], self.n_components))

        # Loop over the number of wanted components
        for k in range(self.n_components):
            if verbose:
                print(f"\nk = {k}\n")

            # Initialize old versions
            u_old = np.zeros_like(u)
            v_old = np.zeros_like(v)
            w_old = np.zeros_like(w)
            tol_old = tol

            # Number of iterations
            it = 0

            # Repeat until convergence
            while (
                (norm(u - u_old) / norm(u) > tol) or
                (norm(v - v_old) / norm(v) > tol) or
                (norm(w - w_old) / norm(w) > tol)
            ):
                # Update u
                u_old = u
                v_cross = np.dot(v.T, v + alpha_v * np.dot(penal_mat['v'], v))
                w_cross = np.dot(w.T, w + alpha_w * np.dot(penal_mat['w'], w))
                u = np.einsum('i, j, kij', v, w, values) / (v_cross * w_cross)

                # Update v
                v_old = v
                u_cross = np.dot(u.T, u)
                a = iden_v + alpha_v * penal_mat['v']
                b = np.einsum('i, j, ikj', u, w, values)
                v = np.linalg.solve(a, b) / (u_cross * w_cross)

                # Update alpha_v
                alpha_v = find_opt_alpha(alpha_range=alpha_range['v'],
                                         data=values,
                                         u=u, v=w,
                                         alpha=alpha_w,
                                         penal_mat=penal_mat['w'],
                                         eival=eigen_v,
                                         dim=2)
                # Update w
                w_old = w
                v_cross = np.dot(v.T, v + alpha_v * np.dot(penal_mat['v'], v))
                a = iden_w + alpha_w * penal_mat['w']
                b = np.einsum('i, j, ijk', u, v, values)
                w = np.linalg.solve(a, b) / (u_cross * v_cross)

                # Update alpha_w
                alpha_w = find_opt_alpha(alpha_range=alpha_range['w'],
                                         data=values,
                                         u=u, v=v,
                                         alpha=alpha_v,
                                         penal_mat=penal_mat['v'],
                                         eival=eigen_w,
                                         dim=3)

                it = it + 1

                if it > max_iter:
                    if adapt_tol and (it < 2 * max_iter):
                        tol = 10 * tol
                    else:
                        u_old, v_old, w_old = u, v, w
                        warnings.warn(f"FCP-TPA algortihm did not converge; "
                                      f"iteration {k} stopped.")

            if verbose:
                print(f"Absolute error:\n"
                      f"u: {norm(u - u_old)}, "
                      f"v: {norm(v - v_old)}, "
                      f"w: {norm(w - w_old)}, "
                      f"alpha_v: {alpha_v}, "
                      f"alpha_w: {alpha_w}.")

            # Reset tolerance if necessary
            if adapt_tol and (it >= max_iter):
                tol = tol_old

            # Scale vector to have norm one
            u = u / norm(u)
            v = v / norm(v)
            w = w / norm(w)

            # Calculate results
            coef[k] = np.einsum('i, j, k, ijk', u, v, w, values)
            mat_u[:, k] = u
            mat_v[:, k] = v
            mat_w[:, k] = w

            # Update the values
            values = values - coef[k] * np.multiply.outer(u, np.outer(v, w))

        # Save the results
        eigenimages = np.einsum('ik, jk -> kij', mat_v, mat_w)
        self.eigenvalues = coef
        self.scores = mat_u
        self.eigenfunctions = DenseFunctionalData(data.argvals,
                                                  eigenimages)

    def transform(self, data, method=None):
        """Apply dimension reduction to the data.

        Parameters
        ----------
        data: DenseFunctionalData
            Functional data object to be transformed.
        method: None
            Not used. To be compliant with other methods.

        Returns
        -------
        scores: np.array, shape=(n_obs, n_components)
            An array representing the projection of the data onto the basis of
            functions defined by the eigenimages.

        """
        return np.einsum('ikl, jkl', data.values, self.eigenfunctions.values)

    def inverse_transform(self, scores):
        """Transform the data back to its original space.

        Return a DenseFunctionalData whose transform would be `scores`.

        Parameters
        ----------
        scores: np.ndarray, shape=(n_obs, n_components)
            New_data, where `n_obs` is the number of observations and
            `n_components` is the number of components.

        Returns
        -------
        data: DenseFunctionalData object
            The transformation of the scores into the original space.

        """
        argvals = self.eigenfunctions.argvals
        values = np.einsum('ij, jkl', scores, self.eigenfunctions.values)
        return DenseFunctionalData(argvals, values)
