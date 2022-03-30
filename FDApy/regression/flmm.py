#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Module for the definition for Functional Linear Mixed Models.

This module is used to implements algorithms FLMM. It is used to model the
variance containing within functional data.
"""
import itertools
import numpy as np

from typing import Optional, List, Union

from ...representation.functional_data import (DenseFunctionalData,
                                               IrregularFunctionalData)
from ...misc.utils import integration_weights_

###############################################################################
# Checkers for parameters


###############################################################################
# Class FLMM


class FLMM():
    """A class defining Functional Linear Mixed Model.

    Parameters
    ----------

    Attributes
    ----------
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
        **kwargs
    ) -> None:
        """Fit the model on data.

        Parameters
        ----------
        data: FunctionalData
            Training data.

        Keyword Args
        ------------
        method: str, default=None
            Smoothing method for the covariance.
        """
        self.parameters = {
            'method': kwargs.get('method', None)
        }
        self._fit(data)

    def _fit(
        self,
        data: FunctionalData
    ) -> None:
        """Dispatch ot the right submethod depending on the input."""
        if isinstance(data, DenseFunctionalData):
            self._fit_dense(data)
        else if:
            self._fit_irregular(data)
        else:
            raise TypeError('FLMM supports DenseFunctionalData and '
                            'IrregularFunctionalData objects!')

    def _fit_dense(
        self,
        data: DenseFunctionalData,
        group_list: List
    ) -> None:
        r"""Functional Linear Mixte Model for Dense Functional Data.

        Parameters
        ----------
        data: DenseFunctionalData
            Training data.
        group_list: List
            List of length :math:`H`.

        References
        ----------
        * JonaCRC. (2017). JonaCRC/denseFLMM: First release (v0.1.0). Zenodo.
        https://doi.org/10.5281/zenodo.322651

        Notes
        -----
        TODO: Modify G.

        """
        # Step 0: Get different parameters.
        argvals = data.argvals["input_dim_0"]
        n_points = data.n_points['input_dim_0']
        n_obs = data.n_obs
        n_groups = len(group_list)
        rhovec = [len(effect) for effect in group_list.values()]
        sq2 = np.sum(np.power(rhovec, 2))
        cum_rhovec2 = np.cumsum(np.power(rhovec, 2))
        rowvec = np.repeat(argvals, n_points)
        colvec = np.tile(argvals, n_points)
        interv = argvals[1] - argvals[0]
        norm = 1 / np.sqrt(interv)

        # Step 1:  Center the data.
        values = data.values  # It is assumed that the data is centered for now

        # Step 2: Estimate the covariances.
        gcyc = np.repeat(np.arange(n_groups), np.power(rhovec, 2))
        qcyc = np.concatenate([np.repeat(np.arange(x), x) for x in rhovec])
        pcyc = np.concatenate([np.tile(np.arange(x), x) for x in rhovec])

        xtx = [
            xtx_entry(
                n_groups[gcyc[i]][pcyc[i]],
                n_groups[gcyc[j]][pcyc[j]],
                n_groups[gcyc[i]][qcyc[i]],
                n_groups[gcyc[j]][qcyc[j]]
            ) for i, j in itertools.combinations_with_replacement(
                np.arange(sq2), 2)
        ]
        xtx_mat = np.zeros((n_groups, n_groups))
        xtx_mat[np.triu_indices(n_groups)] = xtx
        xtx_mat = xtx_mat + xtx_mat.T - np.diag(np.diag(xtx_mat))

        xty = [
            xty_entry(
                n_groups[gcyc[i]][pcyc[i]],
                n_groups[gcyc[i]][qcyc[i]],
                Y
            ) for i in np.arange(sq2)
        ]
        xty_mat = np.stack(xty)

        cov_mat = np.linalg.solve(xtx_mat, xty_mat)

        # Step 3: Smooth the covariances.
        if n_groups == G + 1:
            diagos = np.diag(cov_mat[sq2 - 1, ].reshape(n_points, n_points))
        else:
            diagos = {}
            for idx, k in enumerate(cum_rhovec2[np.arange(G, n_groups)]):
                diagos[idx] = np.diag(cov_mat[k, ].reshape(n_points, n_points))

        if smooth is not None:
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
            NPC = np.zeros(n_groups, dtype='int')
            while explained_var < self.n_components:
                maxg = np.argmax([nu[NPC[idx]] for idx, nu in nu_hat.items()])
                NPC[maxg] = NPC[maxg] + 1
                idx = np.int32(NPC[maxg] - 1)
                explained_var = (
                    explained_var + (nu_hat[maxg][idx] / total_variance)
                )

        phis_estim = {
            idx: vec[:, np.arange(NPC[idx], dtype='int')] * norm
            for idx, (val, vec) in eig_dict.items()
        }
        nu_hat = {
            idx: nu[np.arange(NPC[idx], dtype='int')]
            for idx, nu in nu_hat.items()
        }

        # Step 6: Recompute error variance
        eigval_sum = np.sum(np.concatenate([nu for nu in nu_hat.values()]))
        explained_variance = (var_noise + eigval_sum) / total_variance
        if n_groups == G + 1:
            new_diag = np.diag(
                np.dot(
                    phis_estim[n_groups - 1],
                    np.dot(
                        np.diag(nu_hat[n_groups - 1]),
                        phis_estim[n_groups - 1].T)
                    )
                )
            sigma2_hat = np.max(np.mean(diagos - new_diag), 0)
        else:
            sigma2_hat = var_noise

        # Step 7: Predict basis weights
        zty = np.concatenate([
            zty_entry(Y, Z[0], phi.T)
            for (idx, Z), (_, phi) in zip(group_list.items(),
                                          phis_estim.items())
        ])

        ztz_dict = {
            (i, j): ztz_entry(
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

        n_eff = Lvec * NPC
        cin = np.hstack([0, np.cumsum(n_eff)])
        Dinv = np.diag(
            np.repeat(
                np.hstack([1 / nu for nu in nu_hat.values()]),
                repeats=np.repeat(Lvec, repeats=NPC)
            )
        )
        b_hat = np.linalg.solve(ZtZ + sigma2_hat * Dinv, ZtY)
        xi_hat = [
            b_hat[cin[g] + np.arange(n_eff[g])] for g in np.arange(n_groups)
        ]
        xi_hat = [
            xi_hat[g].reshape((Lvec[g], NPCs[g]), order='F')
            for g in np.arange(n_groups)
        ]

        self.sigma2 = sigma2_hat
        self.xi = xi_hat

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


def xtx_entry(X1, X2, Z1, Z2):
    A = X1.T.dot(X2)
    B = Z1.T.dot(Z2)
    return np.einsum('ij,ji->', A, B.T)


def xty_entry(X1, X2, Y):
    A = X1.T.dot(Y)
    B = X2.T.dot(Y)
    return A.T.dot(B).flatten()


def zty_entry(Y, Z, phi):
    return np.matmul(phi, np.matmul(Y.T, Z)).flatten()


def ztz_entry(Z1, Z2, phi1, phi2):
    phitphi = np.matmul(phi1, phi2.T)
    ZtZ = np.matmul(Z1.T, Z2)
    return np.kron(phitphi, ZtZ)
