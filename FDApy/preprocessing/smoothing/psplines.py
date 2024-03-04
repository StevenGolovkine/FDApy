#!/usr/bin/env python
# -*-coding:utf8 -*

"""
P-splines
---------

"""
import numpy as np
import numpy.typing as npt

from scipy.linalg import solve_triangular

from typing import Optional

from ...representation.basis import _basis_bsplines


########################################################################################
# Inner functions for the PSplines class.


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

        # Construct penalty stuff
        n = basis_mat.shape[1]
        pen_mat = np.sqrt(penalty) * np.diff(np.eye(n), n=self.order_penalty, axis=0)
        nix = np.zeros(n - self.order_penalty)

        if sample_weights is None:
            sample_weights = np.ones(m)
        new_basis_mat = np.vstack([basis_mat, pen_mat])
        new_y = np.concatenate([y, nix])
        new_weights_mat = np.diag(np.concatenate([sample_weights, nix + 1]))

        fit = np.linalg.lstsq(new_weights_mat @ new_basis_mat, new_y, rcond=None)
        beta_hat = fit[0]
        y_hat = basis_mat @ beta_hat

        q_mat, r_mat = np.linalg.qr(new_basis_mat)
        hat_mat = np.sum(np.power(q_mat, 2), axis=1)[:m]

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
        self.parameters = {
            "sigma": sigma,
            "cv": cv,
            "effdim": ed,
            "ed_resid": ed_resid,
            "R": r_mat,
        }
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
