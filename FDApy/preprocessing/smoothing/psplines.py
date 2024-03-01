#!/usr/bin/env python
# -*-coding:utf8 -*

"""
P-splines
---------

"""
import numpy as np
import numpy.typing as npt


########################################################################################
# Inner functions for the PSplines class.


def _tpower(x, knots, p):
    res = np.zeros((len(x), len(knots)))
    for idx, knot in enumerate(knots):
        res[:, idx] = np.power(x - knot, p) * (x >= knot)
    return res


########################################################################################
# class PSplines


class PSplines:
    r"""P-Splines Smoothing.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----

    References
    ----------
    
    """

    def __init__(
        self
    ) -> None:
        """Initializa PSplines object."""
        pass

    def fit(
        self
    ) -> None:
        """Fit the model."""
        pass

    def predict(
        self
    ) -> None:
        """Predict using the model."""
        pass
