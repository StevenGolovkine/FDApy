#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Module for the definition for Functional Linear Mixed Models.

This module is used to implements algorithms FLMM. It is used to model the
variance containing within functional data.
"""
import itertools
import numpy as np

from typing import Dict, Optional, List, Union

from ..representation.functional_data import (
    FunctionalData,
    DenseFunctionalData,
    IrregularFunctionalData
)
from ..misc.utils import integration_weights_

###############################################################################
# Checkers for parameters


###############################################################################
# Class FLMM


class FLMM():
    """A class defining Functional Linear Mixed Model.

    Estimation of functional linear mixed models (FLMMs) for functional data
    based on functional principal components analysis (FPCA).

    Parameters
    ----------
    n_components: list of {int, float, None}, default=None
        Number of components to keep for each grouping factors in data.

        If `n_components` is an integer, we keep `n_components`.

        If `0 < n_components < 1`, we select the number of components such that
        the amount of explained variance is greater than the percentage
        specified by `n_components`.
    smooth: str
        Method to used for the smoothing of the covariance surfaces.

    Attributes
    ----------
    sigma2: float
        Estimated measurement error variance :math:`sigma^2`.
    total_var: float
        Total average variance of the curves.
    explained_var: float
        Level of variance explained by the selected functional principal
        components (+ error variance).
    error_var: float
        Variance of the error.
    nu: Dictionary of np.ndarray
        Dictionary of array containing the estimated eigenvalues.
    xi: List of np.ndarray
        List of array containing the predicted random basis weights.
    phi: List of DenseFunctionalData
        List of FD containing the functional principal components kept per
        grouping factors, with the smooth errors.
    random_effects: List of DenseFunctionalData
        List of FD containing the estimated random effects.

    References
    ----------
    * Greven, S., Cederbaul, J., and Shou, H. (2016). Principal component-based
        functional linear mixed models.
    * JonaCRC. (2017). JonaCRC/denseFLMM: First release (v0.1.0). Zenodo.
        https://doi.org/10.5281/zenodo.322651

    """

    def __init__(
        self,
        n_components: List[Union[int, float, None]] = None,
        smooth: Optional[str] = None
    ) -> None:
        """Initialize FLMM object."""
        self.n_components = n_components
        self.smooth = smooth

    def fit(
        self,
        data: FunctionalData,
        n_factors: int,
        n_levels: List[int],
        group_list: Dict[int, Dict[int, np.ndarray]],
        **kwargs
    ) -> None:
        """Fit the model on data.

        Parameters
        ----------
        data: FunctionalData
            Training data.
        n_factors: int
            Number of grouping factors not used for the estimation of the error
            variance.
        n_levels: List of int
            List containing the number of levels for each grouping factors.
        group_list: Dictionary
            Dictionary of design matrices.

        Keyword Args
        ------------

        """
        self._fit(data, n_factors, n_levels, group_list)

    def _fit(
        self,
        data: FunctionalData,
        n_factors: int,
        n_levels: List[int],
        group_list: Dict[int, Dict[int, np.ndarray]],
        **kwargs
    ) -> None:
        """Dispatch ot the right submethod depending on the input."""
        if isinstance(data, DenseFunctionalData):
            self._fit_dense(data, n_factors, n_levels, group_list)
        elif isinstance(data, IrregularFunctionalData):
            self._fit_irregular(data)
        else:
            raise TypeError('FLMM supports DenseFunctionalData and '
                            'IrregularFunctionalData objects!')

    def _fit_dense(
        self,
        data: DenseFunctionalData,
        n_factors: int,
        n_levels: List[int],
        group_list: Dict[int, Dict[int, np.ndarray]]
    ) -> None:
        r"""Functional Linear Mixte Model for Dense Functional Data.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data.
        n_factors: int
            Number of grouping factors not used for the estimation of the error
            variance.
        n_levels: List of int
            List containing the number of levels for each grouping factors.
        group_list: Dictionary
            Dictionary of design matrices.

        References
        ----------
        * JonaCRC. (2017). JonaCRC/denseFLMM: First release (v0.1.0). Zenodo.
        https://doi.org/10.5281/zenodo.322651

        Notes
        -----
        TODO: Add smoothing of covariance surfaces.
        TODO: Add the case of group-specific errors

        """
        # Step 0: Get different parameters.
        argvals = data.argvals["input_dim_0"]
        n_points = data.n_points['input_dim_0']
        n_groups = len(group_list)
        rho = [len(effect) for effect in group_list.values()]
        sq2 = np.sum(np.power(rho, 2))
        crho2 = np.cumsum(np.power(rho, 2))
        # rowvec = np.repeat(argvals, n_points)
        interv = argvals[1] - argvals[0]
        norm = 1 / np.sqrt(interv)

        # Step 1:  Center the data.
        # values = data.values

        # Step 2: Estimate the covariance surfaces.
        gcyc = np.repeat(np.arange(n_groups), np.power(rho, 2))
        qcyc = np.concatenate([np.repeat(np.arange(x), x) for x in rho])
        pcyc = np.concatenate([np.tile(np.arange(x), x) for x in rho])

        xtx = [
            xtx_entry_(
                group_list[gcyc[i]][pcyc[i]],
                group_list[gcyc[j]][pcyc[j]],
                group_list[gcyc[i]][qcyc[i]],
                group_list[gcyc[j]][qcyc[j]]
            ) for i, j in itertools.combinations_with_replacement(
                np.arange(sq2), 2)
        ]
        xtx_mat = np.zeros((n_groups, n_groups))
        xtx_mat[np.triu_indices(n_groups)] = xtx
        xtx_mat = xtx_mat + xtx_mat.T - np.diag(np.diag(xtx_mat))

        xty = [
            xty_entry_(
                group_list[gcyc[i]][pcyc[i]],
                group_list[gcyc[i]][qcyc[i]],
                data.values
            ) for i in np.arange(sq2)
        ]
        xty_mat = np.stack(xty)

        cov_mat = np.linalg.solve(xtx_mat, xty_mat)

        # Step 3: Smooth the covariances.
        if n_groups == n_factors + 1:
            diagos = np.diag(cov_mat[sq2 - 1, ].reshape(n_points, n_points))
        else:
            diagos = {}
            for i, k in enumerate(crho2[np.arange(n_factors, n_groups)]):
                diagos[i] = np.diag(cov_mat[k, ].reshape(n_points, n_points))

        if self.smooth is not None:
            print("Smooth the matrices")
        cov = {idx: k.reshape((n_points, n_points))
               for idx, k in enumerate(cov_mat)}
        cov = {idx: (k + k.T) / 2 for idx, k in cov.items()}

        # Step 4: Estimate noise variance.
        var_hat = np.diag(cov[n_groups - 1])
        cov_diag = diagos

        ll = argvals[len(argvals) - 1] - argvals[0]
        lower = np.sum(~(argvals >= (argvals[0] + 0.25 * ll)))
        upper = np.sum((argvals <= (argvals[len(argvals) - 1] - 0.25 * ll)))
        weights = integration_weights_(argvals[lower:upper], method='trapz')
        nume = np.dot(weights, (var_hat - cov_diag)[lower:upper])
        var_noise = np.maximum(nume / argvals[upper] - argvals[lower], 0)

        # Step 5: Estimate eigenfunctions/eigenvalues.
        eig_dict = {idx: np.linalg.eigh(k) for idx, k in cov.items()}
        eig_dict = {idx: (np.flipud(val), np.fliplr(vec))
                    for idx, (val, vec) in eig_dict.items()}
        nu_hat = {idx: val * interv for idx, (val, vec) in eig_dict.items()}

        total_variance = (
            var_noise + np.sum([nu * (nu > 0) for nu in nu_hat.values()])
        )

        if isinstance(self.n_components, float):
            explained_var = var_noise / total_variance
            n_comp = np.zeros(n_groups, dtype='int')
            while explained_var < self.n_components:
                maxg = np.argmax(
                    [nu[n_comp[idx]] for idx, nu in nu_hat.items()]
                )
                n_comp[maxg] = n_comp[maxg] + 1
                idx = np.int32(n_comp[maxg] - 1)
                explained_var = (
                    explained_var + (nu_hat[maxg][idx] / total_variance)
                )

        phis_estim = {
            idx: vec[:, np.arange(n_comp[idx], dtype='int')] * norm
            for idx, (val, vec) in eig_dict.items()
        }
        nu_hat = {
            idx: nu[np.arange(n_comp[idx], dtype='int')]
            for idx, nu in nu_hat.items()
        }

        # Step 6: Recompute error variance
        eigval_sum = np.sum(np.concatenate([nu for nu in nu_hat.values()]))
        explained_variance = (var_noise + eigval_sum) / total_variance
        if n_groups == n_factors + 1:
            new_diag = np.diag(
                np.dot(
                    phis_estim[n_groups - 1],
                    np.dot(
                        np.diag(nu_hat[n_groups - 1]),
                        phis_estim[n_groups - 1].T
                    )
                )
            )
            sigma2_hat = np.max(np.mean(diagos - new_diag), 0)
        else:
            sigma2_hat = var_noise

        # Step 7: Predict basis weights
        zty = np.concatenate([
            zty_entry_(data.values, z[0], phi.T)
            for (idx, z), (_, phi) in zip(group_list.items(),
                                          phis_estim.items())
        ])

        ztz_dict = {
            (i, j): ztz_entry_(
                group_list[i][0], group_list[j][0],
                phis_estim[i].T, phis_estim[j].T
            ) for (i, j) in itertools.product(np.arange(n_groups), repeat=2)
        }
        ztz = np.hstack(
            [
                np.vstack([ztz_dict[(i, j)] for i in np.arange(n_groups)])
                for j in np.arange(n_groups)
            ]
        )

        n_eff = n_levels * n_comp
        cin = np.hstack([0, np.cumsum(n_eff)])
        d_inv = np.diag(
            np.repeat(
                np.hstack([1 / nu for nu in nu_hat.values()]),
                repeats=np.repeat(n_levels, repeats=n_comp)
            )
        )
        b_hat = np.linalg.solve(ztz + sigma2_hat * d_inv, zty)
        xi_hat = [
            b_hat[cin[g] + np.arange(n_eff[g])] for g in np.arange(n_groups)
        ]
        xi_hat = [
            xi_hat[g].reshape((n_levels[g], n_comp[g]), order='F')
            for g in np.arange(n_groups)
        ]

        # Step 8: Compute random effects
        rand_effects = [
            DenseFunctionalData(data.argvals, np.matmul(xi, phi.T))
            for xi, phi in zip(xi_hat, phis_estim.values())
        ]

        self.sigma2 = sigma2_hat
        self.total_var = total_variance
        self.explained_var = explained_variance
        self.error_var = var_noise
        self.nu = nu_hat
        self.xi = xi_hat
        self.random_effects = rand_effects
        self.phi = [
            DenseFunctionalData(data.argvals, phi.T)
            for phi in phis_estim.values()
        ]

    def _fit_irregular(
        self,
        data: IrregularFunctionalData
    ) -> None:
        """Functional Linear Mixte Model for Irregular Functional Data.

        Parameters
        ----------
        data: IrregularFunctionalData
            Training data

        References
        ----------
        * https://github.com/JonaCRC/sparseFLMM/

        Notes
        -----
        TODO: Implement the method.

        """
        pass


def xtx_entry_(x1, x2, z1, z2):
    """Compute xtx entry."""
    mat_a = x1.T.dot(x2)
    mat_b = z1.T.dot(z2)
    return np.einsum('ij,ji->', mat_a, mat_b.T)


def xty_entry_(x1, x2, y):
    """Compute xty entry."""
    mat_a = x1.T.dot(y)
    mat_b = x2.T.dot(y)
    return mat_a.T.dot(mat_b).flatten()


def zty_entry_(y, z, phi):
    """Compute zty entry."""
    return np.matmul(phi, np.matmul(y.T, z)).flatten()


def ztz_entry_(z1, z2, phi1, phi2):
    """Compute ztz entry."""
    phitphi = np.matmul(phi1, phi2.T)
    ztz = np.matmul(z1.T, z2)
    return np.kron(phitphi, ztz)
