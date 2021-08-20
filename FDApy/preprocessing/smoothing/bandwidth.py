#!/usr/bin/env python
# -*-coding:utf8 -*

"""Module for Bandwidth class.

This module is used to estimate the bandwidth parameter that is necessary in
the case of kernel regression.
"""
import numpy as np

from typing import NamedTuple

from ...src.sigma import estimate_sigma


##############################################################################
# Misc functions
def theta_(v, k, idx):
    """Estimate theta.

    Parameters
    ----------
    v: np.ndarray or list
        A vector
    k: int
        An integer
    idx: int
        An integer

    Returns
    -------
    res: float

    """
    return (v[idx + 2 * k - 1] - v[idx + k])**2


def eta_(v, k, idx, hurst):
    """Estimate eta.

    Parameters
    ----------
    v: np.ndarray or list
        A vector
    k: int
        An integer
    idx: int
        An integer
    hurst: float
        A float

    Returns
    -------
    res: float

    """
    return (v[idx + 2 * k - 1] - v[idx + k])**(2 * hurst)


def indices_(data, t0, ranges):
    """Get indices.

    Parameters
    ----------
    data: dict_values
        Values dictionary from an IrregularFunctionalData object.
    t0: float
        A float
    ranges: int
        An integer

    Returns
    -------
    res: list of int

    """
    return [np.min(np.argsort(abs(argval - t0))[np.arange(ranges)])
            for argval in data]


def mean_theta_(data, idxs, ranges):
    """Compute mean theta.

    Parameters
    ----------
    data: dict_values
        Values dictionary from an IrregularFunctionalData object.
    idxs: list of int
        A list of integer
    ranges: int
        An integer

    Returns
    -------
    res: float

    """
    return np.mean([theta_(obs, ranges, idx) for obs, idx in zip(data, idxs)])


def mean_eta_(data, idxs, ranges, hurst):
    """Compute mean eta.

    Parameters
    ----------
    data: dict_values
        Values dictionary from an IrregularFunctionalData object.
    idxs: list of int
        A list of integer
    ranges: int
        An integer
    hurst: float
        A float

    Returns
    -------
    res: float

    """
    return np.mean([eta_(obs, ranges, idx, hurst)
                    for obs, idx in zip(data, idxs)])


##############################################################################
# Estimation of Hurst parameters
def estimate_hurst_(argvals, values, t0, k0, sigma=None):
    """Perform an estimation of the Hurst parameter.

    This function performs an estimation of the Hurst coefficients, which is
    commonly denoted as :math:`H_0`.

    Parameters
    ----------
    argvals: dict_values
        The values of the argvals dictionary from an IrregularFunctionalData
        object.
    values: dict_values
        The values of the values dictionary from an IrregularFunctionalData
        object.
    t0: float
        Time at which the estimation of the Hurst coefficient is performed.
    k0: int
        Considered neighborhood.
    sigma: float, default=None
        Estimation of the standard deviation of the noise.

    Returns
    -------
    hurst: float
        An estimation of the Hurst parameter at :math:`t_0`.

    """
    first_part = np.log(4)
    second_part = 0
    if sigma is None:
        idxs = indices_(argvals, t0, 8 * k0 - 6)
        a = mean_theta_(values, idxs, 4 * k0 - 3)
        b = mean_theta_(values, idxs, 2 * k0 - 1)
        c = mean_theta_(values, idxs, k0)
        if (a - b > 0) and (b - c > 0) and (a - 2 * b + c > 0):
            first_part = np.log(a - b)
            second_part = np.log(b - c)
    else:  # Case where sigma is known
        idxs = indices_(argvals, t0, 4 * k0 - 2)
        a = mean_theta_(values, idxs, 2 * k0 - 1)
        b = mean_theta_(values, idxs, k0)
        if min(a, b) > (2 * np.power(sigma, 2)):
            first_part = np.log(a - 2 * np.power(sigma, 2))
            second_part = np.log(b - np.power(sigma, 2))
    return (first_part - second_part) / np.log(4)


def estimate_hurst_list_(argvals, values, t0, k0, sigma=None):
    """Perform an estimation of the Hurst coefficient along a list.

    This function performs an estimation of the Hurst coefficient :math:`H_0`
    at different :math:`t_0`.

    Parameters
    ----------
    argvals: dict_values
        The values of the argvals dictionary from an IrregularFunctionalData
        object.
    values: dict_values
        The values of the values dictionary from an IrregularFunctionalData
        object.
    t0: list of float
        Times where the esimtation of :math:`H_0` is done.
    k0: list of int
        Neighborhood of :math:`t_0` to consider.
    sigma: float, default:None
        An estimation of the standard deviation of the noise.

    Returns
    -------
    hurst: list of float
        An estimation of the Hurst parameter along a list of :math:`t_0`. It
        has the same length than the list of :math:`t_0`.

    """
    if len(t0) != len(k0):
        raise ValueError('t0 and k0 do not have the same length.')
    return [estimate_hurst_(argvals, values, i, j, sigma)
            for (i, j) in zip(t0, k0)]


##############################################################################
# Estimation of L0
def estimate_constant_(argvals, values, t0, k0, hurst, sigma=None):
    """Perform an estimation of the Lipschitz constant.

    This function performs an estimation of the Lipschitz constant, which is
    commonly denoted as :math:`L_0`.

    Parameters
    ----------
    argvals: dict_values
        The values of the argvals dictionary from an IrregularFunctionalData
        object.
    values: dict_values
        The values of the values dictionary from an IrregularFunctionalData
        object.
    t0: float
        Time at which the estimation of the Lipschitz constant is performed.
    k0: int
        Considered neighborhood.
    hurst: float
        An estimation of the Hurst parameter.
    sigma: float, default=None
        Estimation of the standard deviation of the noise.

    Returns
    -------
    constant: float
        An estimation of the Lipschitz constant.

    """
    nume = 1
    deno = 1
    if sigma is None:  # Subcase where sigma is not known
        idxs = indices_(argvals, t0, 4 * k0 - 2)
        a = mean_theta_(values, idxs, 2 * k0 - 1)
        b = mean_theta_(values, idxs, k0)
        c = mean_eta_(argvals, idxs, 2 * k0 - 1, hurst)
        d = mean_eta_(argvals, idxs, k0, hurst)
        if (a - b > 0) and (c - d > 0):
            nume = a - b
            deno = c - d
    else:  # Subcase where sigma is known
        idxs = indices_(argvals, t0, 2 * k0)
        a = mean_theta_(values, idxs, k0)
        b = mean_eta_(argvals, idxs, k0, hurst)
        if (a - 2 * np.power(sigma, 2)) and (b > 0):
            nume = a - 2 * np.power(sigma, 2)
            deno = b
    return np.power(nume / deno, 0.5)


def estimate_constant_list_(argvals, values, t0, k0, hurst, sigma=None):
    """Perform an estimation of the Lipschitz constant along a list.

    This function performs an estimation of the Lipschitz constant :math:`L_0`
    at different :math:`t_0`.

    Parameters
    ----------
    argvals: dict_values
        The values of the argvals dictionary from an IrregularFunctionalData
        object.
    values: dict_values
        The values of the values dictionary from an IrregularFunctionalData
        object.
    t0: list of float
        Time at which the estimation of the Lipschitz constant is performed.
    k0: list of int
        Neighborhood of :math:`t_0` to consider.
    hurst: list of float
        An estimation of the Hurst parameter at each :math:`t_0`.
    sigma: float, default:None
        An estimation of the standard deviation of the noise.

    Returns
    -------
    constant: list of float
        An estimation of the Lipschitz constant along a list of :math:`t_0`. It
        has the same length than the list of :math:`t_0`.

    """
    if len(t0) != len(k0):
        raise ValueError('t0 and k0 do not have the same length.')
    if len(t0) != len(hurst):
        raise ValueError('t0 and hurst do not have the same length.')
    if len(k0) != len(hurst):
        raise ValueError('k0 and hurst do not have the same length.')
    return [estimate_constant_(argvals, values, i, j, k, sigma)
            for (i, j, k) in zip(t0, k0, hurst)]


##############################################################################
# Estimation of the bandwidth
def estimate_bandwidth_(argvals, hurst, constant, sigma,
                        kernel="epanechnikov"):
    """Perform an estimation of the bandwidth.

    This function performs an estimation of the bandwidth, which is
    commonly denoted as :math:`b`.

    Parameters
    ----------
    argvals: dict_values
        The values of the argvals dictionary from an IrregularFunctionalData
        object.
    hurst: float
        An estimation of the Hurst parameter.
    constant: float
        An estimation of the Lipschitz constant.
    sigma: float
        An estimation of the standard deviation of the noise.
    kernel: str, default="epanechnikov"
        The kernel used for the estimation. Should be "epanechnikov" or
        "uniform".

    Returns
    -------
    bandwidth: list of float
        An estimation of the bandwidth.

    """
    # if kernel == "epanechnikov":
    #     k_norm2 = 0.6
    #     k_phi = 3 / ((hurst + 1) * (hurst + 3))
    # elif kernel == "uniform":
    #     k_norm2 = 1
    #     k_phi = 1 / (hurst + 1)
    # else:
    #     raise NotImplementedError('Kernel not implemented.')

    nume = sigma**2 * np.math.factorial(np.floor(hurst))**2  # * k_norm2
    deno = 2 * hurst * constant**2  # * k_phi
    frac = nume / deno

    return [(frac / len(obs))**(1 / (2 * hurst + 1)) for obs in argvals]


def estimate_bandwidth_list_(argvals, hurst, constant, sigma,
                             kernel="epanechnikov"):
    r"""Perform an estimation of the bandwidth along a list.

    Perform an estimation of the bandwidth in case :math:`H_0`, :math:`L_0`
    :math:`\sigma` are lists.

    Parameters
    ----------
    argvals: dict_values
        The values of the argvals dictionary from an IrregularFunctionalData
        object.
    hurst: list of float
        An estimation of the Hurst parameter.
    constant: list of float
        An estimation of the Lipschitz constant.
    sigma: float
        An estimation of the standard deviation of the noise.
    kernel: str, default="epanechnikov"
        The kernel used for the estimation. Should be "epanechnikov" or
        "uniform".

    Returns
    -------
    bandwidth: list of float
        An estimation of the bandwidth along a list of :math:`t_0`.

    """
    return [estimate_bandwidth_(argvals, i, j, sigma, kernel)
            for (i, j) in zip(hurst, constant)]


##############################################################################
# Class BandwithResult

class BandwidthResult(NamedTuple):
    """An object containing the Bandwidth."""

    points: list
    neighborhood: list
    hurst_coefficient: list
    constants: list
    sigma: float
    bandwidths: list

    def __repr__(self) -> str:
        """Override print function."""
        return (f"Bandwidth:\n"
                f"\t Sampling points: {str(self.points)}.\n"
                f"\t Neighborhood: {str(self.neighborhood)}.\n"
                f"\t Hurst estimates: {str(self.hurst_coefficient)}.\n"
                f"\t Constant estimates: {str(self.constants)}.\n"
                f"\t Bandwidth esimtates: {str(self.bandwidths)}.")


##############################################################################
# Class Bandwidth

class Bandwidth(object):
    """A module for the compuation of the bandwidth for the smoothing.

    This module is used to compute the bandwidths for a given dataset and will
    be used to perform smoothing. This cbandwidth computation is defined in the
    article *add ref*.

    Attributes
    ----------
    bandwidth : BandwidthResult object
        Estimation of the bandwidths for each points.

    """

    def __init__(
        self,
        points: list = 0.5,
        neighborhood: list = 2,
        kernel: str = "epanechnikov"
    ) -> None:
        """Initialize Bandwidth object.

        Parameters
        ----------
        points: list of float
            A list of sampling points at which the estimation should be done.
        neighborhood: list of int
            A list of neighborhood
        kernel: str, default="epanechnikov"
            The kernel used to compute the bandwidth.

        """
        self.points = points
        self.neighborhood = neighborhood
        self.kernel = kernel

    def __str__(self):
        """Override __str__ functions."""
        return (f'Bandwidth(points={self.points}, neighborhood='
                f'{self.neighborhood})')

    def __repr__(self):
        """Override print function."""
        return self.__str__()

    def __call__(
        self,
        data,
        hurst: list = None,
        constants: list = None,
        sigma: float = None
    ) -> BandwidthResult:
        """Compute the bandwidths.

        Parameters
        ----------
        data: IrregularFunctionalData
            An element of the class IrregularFunctionalData.
        hurst: list of float
            An estimate of the Hurst coefficient.
        constants: list of float
            An estimate of the Lipschitz constant
        sigma: float
            An estimate of the standard deviation of the noise.

        Returns
        -------
        res: BandwidthResult
            Return an instance of BandwidthResult.

        """
        if data.n_dim > 1:
            raise NotImplementedError("Bandwidth computation is only"
                                      " implement for one dimensional data.")

        argvals = data.argvals['input_dim_0'].values()
        values = data.values.values()

        if hurst is None:
            hurst = self.estimate_hurst(argvals, values)
        if constants is None:
            constants = self.estimate_constant(argvals, values, hurst)
        if sigma is None:
            sigma = estimate_sigma(values)

        bandwidth = self.estimate_bandwidth(argvals, hurst, constants, sigma,
                                            self.kernel)
        return BandwidthResult(self.points, self.neighborhood,
                               hurst, constants, sigma, bandwidth)

    def estimate_hurst(self, argvals, values, sigma=None):
        """Perform an estimation of the Hurst coeffifient.

        Parameters
        ----------
        argvals: dict_values
            The values of the argvals dictionary from an
            IrregularFunctionalData object.
        values: dict_values
            The values of the values dictionary from an IrregularFunctionalData
            object.
        sigma: float, default:None
            An estimation of the standard deviation of the noise.

        Returns
        -------
        hurst: list of float or float
            An estimation of the Hurst parameter along a list of :math:`t_0`.
            It has the same length than the list of :math:`t_0`.

        """
        if isinstance(self.points, list):
            return estimate_hurst_list_(argvals, values, self.points,
                                        self.neighborhood, sigma)
        else:
            return estimate_hurst_(argvals, values, self.points,
                                   self.neighborhood, sigma)

    def estimate_constant(self, argvals, values, hurst, sigma=None):
        """Perform an estimation of the Lipschitz constant.

        Parameters
        ----------
        argvals: dict_values
            The values of the argvals dictionary from an
            IrregularFunctionalData object.
        values: dict_values
            The values of the values dictionary from an IrregularFunctionalData
            object.
        hurst: list of float
            An estimation of the Hurst parameter at each :math:`t_0`.
        sigma: float, default:None
            An estimation of the standard deviation of the noise.

        Returns
        -------
        constant: list of float or float
            An estimation of the Lipschitz constant along a list of
            :math:`t_0`. It has the same length than the list of :math:`t_0`.

        """
        if isinstance(self.points, list):
            return estimate_constant_list_(argvals, values, self.points,
                                           self.neighborhood, hurst, sigma)
        else:
            return estimate_constant_(argvals, values, self.points,
                                      self.neighborhood, hurst, sigma)

    def estimate_bandwidth(self, argvals, hurst, constants, sigma,
                           kernel="epanechnikov"):
        """Perform an estimation of the bandwidth.

        Parameters
        ----------
        argvals: dict_values
            The values of the argvals dictionary from an
            IrregularFunctionalData object.
        hurst: list of float
            An estimation of the Hurst parameter.
        constants: list of float
            An estimation of the Lipschitz constant.
        sigma: list of float
            An estimation of the standard deviation of the noise.
        kernel: str, default="epanechnikov"
            The kernel used for the estimation. Should be "epanechnikov" or
            "uniform".

        Returns
        -------
        bandwidth: list of float
            An estimation of the bandwidth along a list of :math:`t_0`.

        """
        if isinstance(self.points, list):
            return estimate_bandwidth_list_(argvals, hurst, constants,
                                            sigma, kernel)
        else:
            return estimate_bandwidth_(argvals, hurst, constants, sigma,
                                       kernel)
