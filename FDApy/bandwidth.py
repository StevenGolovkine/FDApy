#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np


def estimate_H0(data, t0, k0, sigma=None):
    """Perform an estimation of :math:`H_0`.

    Parameters
    ----------
    data: FunctionalData
    """
    def theta(v, k, idx):
        return np.power(v[idx + 2 * k - 1] - v[idx + k], 2)
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
                       [np.arange(8 * k0 - 6)])
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
    """
    return [estimate_H0(data, i, k0, sigma) for i in t0]


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
        self.t0 = t0
        self.k0 = k0

    def __repr__(self):
        res = "Bandwidth:\n" +\
            "\tSampling points: " + str(self.t0) + "\n" +\
            "\tHurst estimates: " + str(self.H) + "\n" +\
            "\tConstant estimates: " + str(self.L) + "\n" +\
            "\tBandwidth estimates: " + str(self.b) + "."
        return res

    def estimate_H(self, data, sigma=None):
        """Perform an estimation of :math:`H_0`.

        Parameters
        ----------

        """
        if isinstance(self.t0, list):
            self.H = estimate_H0_list(data, self.t0, self.k0, sigma)
        else:
            self.H = estimate_H0(data, self.t0, self.k0, sigma)
