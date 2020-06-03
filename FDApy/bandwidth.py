#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""Module for Bandwidth class.

This module is used to estimate the bandwidth parameter that is necessary in
the case of kernel regression.
"""
import numpy as np

from .src.sigma import estimate_sigma


##############################################################################
# Estimation of mu
def estimate_mu(data):
    """Perform an estimation of the mean number of sampling points.

    Parameters
    ----------
    data: FunctionalData
        An element of the class IrregularFunctionalData
    """
    return np.mean([len(i) for i in data.argvals])


##############################################################################
# Estimation of H0
def estimate_H0(data, t0, k0, sigma=None):
    """Perform an estimation of :math:`H_0`.

    Parameters
    ----------
    data: FunctionalData
        An element of the class IrregularFunctionalData
    t0: float
        Time to estimate H0
    k0: int
        Considered neighborhood
    sigma: float, default=None
        Estimation of the standard deviation of the noise

    Returns
    -------
    An estimation of H0.

    """
    def theta(v, k, idx):
        return (v[idx + 2 * k - 1] - v[idx + k])**2

    first_part = 0
    second_part = 0
    two_log_two = 2 * np.log(2)
    if sigma is None:
        idxs = [np.min(np.argsort(abs(argval - t0))
                       [np.arange(8 * k0 - 6)])
                for argval in data.argvals]
        a = np.mean([theta(obs, 4 * k0 - 3, idx)
                    for obs, idx in zip(data.values, idxs)])
        b = np.mean([theta(obs, 2 * k0 - 2, idx)
                    for obs, idx in zip(data.values, idxs)])
        c = np.mean([theta(obs, k0, idx)
                    for obs, idx in zip(data.values, idxs)])
        if (a - b > 0) and (b - c > 0) and (a - 2 * b + c > 0):
            first_part = np.log(a - b)
            second_part = np.log(b - c)
    else:  # Case where sigma is known
        idxs = [np.min(np.argsort(abs(argval - t0))
                       [np.arange(4 * k0 - 2)])
                for argval in data.argvals]
        a = np.mean([theta(obs, 2 * k0 - 1, idx)
                    for obs, idx in zip(data.values, idxs)])
        b = np.mean([theta(obs, k0, idx)
                    for obs, idx in zip(data.values, idxs)])
        if (a - 2 * np.power(sigma, 2) > 0) and\
           (b - 2 * np.power(sigma, 2)) and\
           (a - b > 0):
            first_part = np.log(a - 2 * np.power(sigma, 2))
            second_part = np.log(b - np.power(sigma, 2))
    return (first_part - second_part) / two_log_two


def estimate_H0_list(data, t0, k0, sigma=None):
    """Perform an estimation of :math:`H_0` in case t0 is a list.

    Parameters
    ----------
    data: FunctionalData
    t0: list of float
        Times to estimate H0
    k0: int
        Neighborhood to consider
    sigma: float, default:None
        An estimation of the standard deviation of the noise

    """
    return [estimate_H0(data, i, k0, sigma) for i in t0]


##############################################################################
# Estimation of L0
def estimate_L0(data, t0, k0, H0, sigma=None, density=False):
    """Perform an estimation of :math:`L_0`.

    Parameters
    ----------
    data: FunctionalData

    """
    def theta(v, k, idx):
        return (v[idx + 2 * k - 1] - v[idx + k])**2

    def eta(v, k, idx, H):
        return (v[idx + 2 * k - 1] - v[idx + k])**(2 * H)

    mu_hat = estimate_mu(data)

    nume = 1
    deno = 1
    if density is False:  # Case where the density is not known
        if sigma is None:  # Subcase where sigma is not known
            idxs = [np.min(np.argsort(abs(argval - t0))
                           [np.arange(4 * k0 - 2)])
                    for argval in data.argvals]
            a = np.mean([theta(obs, 2 * k0 - 1, idx)
                        for obs, idx in zip(data.values, idxs)])
            b = np.mean([theta(obs, k0, idx)
                        for obs, idx in zip(data.values, idxs)])
            c = np.mean([eta(obs, 2 * k0 - 1, idx, H0)
                        for obs, idx in zip(data.argvals, idxs)])
            d = np.mean([eta(obs, k0, idx, H0)
                        for obs, idx in zip(data.argvals, idxs)])
            if (a - b > 0) and (c - d > 0):
                nume = a - b
                deno = c - d
        else:  # Subcase where sigma is known
            idxs = [np.min(np.argsort(abs(argval - t0))
                           [np.arange(2 * k0)])
                    for argval in data.argvals]
            a = np.mean([theta(obs, k0, idx)
                        for obs, idx in zip(data.values, idxs)])
            b = np.mean([eta(obs, k0, idx, H0)
                        for obs, idx in zip(data.argvals, idxs)])
            if (a - 2 * np.power(sigma, 2)) and (b > 0):
                nume = a - 2 * np.power(sigma, 2)
                deno = b
    else:  # Case where the density is known (only the uniform case)
        if sigma is None:  # Subcase where sigma is not known
            idxs = [np.min(np.argsort(abs(argval - t0))
                           [np.arange(4 * k0 - 2)])
                    for argval in data.argvals]
            a = np.mean([theta(obs, 2 * k0 - 1, idx)
                        for obs, idx in zip(data.values, idxs)])
            b = np.mean([theta(obs, k0, idx)
                        for obs, idx in zip(data.values, idxs)])
            if (a - b > 0):
                nume = a - b
            deno = (np.power(2, 2 * H0) - 1) *\
                np.power((k0 - 1) / (mu_hat + 1), 2 * H0)
        else:  # Subcase where sigma is known
            idxs = [np.min(np.argsort(abs(argval - t0))
                           [np.arange(2 * k0)])
                    for argval in data.argvals]
            a = np.mean([theta(obs, k0, idx)
                        for obs, idx in zip(data.values, idxs)])
            if (a - 2 * np.power(sigma, 2)):
                nume = a - 2 * np.power(sigma, 2)
            deno = np.power((k0 - 1) / (mu_hat + 1), 2 * H0)

    return np.power(nume / deno, 0.5)


def estimate_L0_list(data, t0, k0, H0, sigma=None, density=False):
    """Perform an estimation of :math:`L_0` in case t0 is a list.

    Parameters
    ----------
    data: FunctionalData
        An object that represents functional data
    t0: list of float
        Times to estimate H0
    k0: int
        Neighborhood to consider
    H0: list of float
        An estimation of H0 at t0
    sigma: float, default:None
        An estimation of the standard deviation of the noise
    density: boolean, default=False
        Do the density of the sampling points follow a uniform distribution

    """
    return [estimate_L0(data, i, k0, j, sigma, density)
            for (i, j) in zip(t0, H0)]


##############################################################################
# Estimation of the bandwidth
def estimate_bandwidth(data, H0, L0, sigma, K="epanechnikov"):
    """Perform an estimation of the bandwidth."""
    # Set kernel constants
    if K == "epanechnikov":
        K_norm2 = 0.6
        K_phi = 3 / ((H0 + 1) * (H0 + 3))
    else:
        K_norm2 = 1
        K_phi = 1 / (H0 + 1)

    # Estimate the bandwidth
    nume = sigma**2 * K_norm2 * np.math.factorial(np.floor(H0))
    deno = H0 * L0 * K_phi
    frac = nume / deno

    return [(frac / len(i))**(1 / (2 * H0 + 1)) for i in data.argvals]


def estimate_bandwidth_list(data, H0, L0, sigma, K="epanechnikov"):
    """Perform an estimation of the bandwidth.

    Perform an estimation of the bandwidth in case :math:`H_0` and
    :math:`L_0` are lists.
    """
    return [estimate_bandwidth(data, i, j, sigma, K) for (i, j) in zip(H0, L0)]


##############################################################################
# Class Bandwidth

class Bandwidth(object):
    """An object to define a bandwidth for the smoothing.

    Parameters
    ----------
    t0 : list of float
        A list of sampling points
    k0 : list of int
        A list of neighborhood
    H : list of float
        A list of Hurst parameters
    L : list of float
        A list of constant

    Attributes
    ----------
    b : list of float
        Estimation of the bandwidth for each t0

    """

    def __init__(self, t0=0.5, k0=2):
        """Initialize Bandwidth object.

        Parameters
        ----------
        t0 : list of float
            A list of sampling points
        k0 : list of int
            A list of neighborhood

        """
        self.t0 = t0
        self.k0 = k0

    def __repr__(self):
        """Override print function."""
        res = "Bandwidth:\n" +\
            "\tSampling points: " + str(self.t0) + "\n" +\
            "\tHurst estimates: " + str(self.H) + "\n" +\
            "\tConstant estimates: " + str(self.L) + "\n" +\
            "\tBandwidth estimates: " + str(self.b) + "."
        return res

    def estimate_H(self, data, sigma=None):
        """Perform an estimation of :math:`H_0`."""
        if isinstance(self.t0, list):
            self.H = estimate_H0_list(data, self.t0, self.k0, sigma)
        else:
            self.H = estimate_H0(data, self.t0, self.k0, sigma)

    def estimate_L(self, data, H0=None, sigma=None, density=False):
        """Perform an estimation of :math:`L_0`."""
        # Estimate parameters
        if (not hasattr(self, 'H')) or (H0 is None):
            self.estimate_H(data, sigma)

        # Set parameters
        H = H0 if H0 is not None else self.H

        if isinstance(self.t0, list):
            self.L = estimate_L0_list(data, self.t0, self.k0, H,
                                      sigma, density)
        else:
            self.L = estimate_L0(data, self.t0, self.k0, H,
                                 sigma, density)

    def estimate_bandwidth(self, data, H0=None, L0=None, sigma=None):
        """Perform an estimation of the bandwidth.

        Parameters
        ----------
        data: FunctionalData
        H0: list of float, default=None
            An estimation of H0
        L0: list of float, default=None
            An estimation of L0
        sigma: float, default:None
            An estimation of the standard deviation of the noise

        """
        # Estimate parameters
        if (not hasattr(self, 'H')) or (H0 is None):
            self.estimate_H(data, sigma)

        if (not hasattr(self, 'L')) or (L0 is None and H0 is None):
            self.estimate_L(data, self.H, sigma)

        #  Set parameters
        H = H0 if H0 is not None else self.H
        L = L0 if L0 is not None else self.L
        s = sigma if sigma is not None else estimate_sigma(data.values)

        if isinstance(self.t0, list):
            self.b = estimate_bandwidth_list(data, H, L, s)
        else:
            self.b = estimate_bandwidth(data, H, L, s)
