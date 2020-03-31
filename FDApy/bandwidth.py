#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import scipy


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
    def __init__(self):
        pass

    def __repr__(self):
        res = "Bandwidth:\n" +\
            "\tSampling points: " + str(self.t0) + "\n" +\
            "\tHurst estimates: " + str(self.H) + "\n" +\
            "\tConstant estimates: " + str(self.L) + "\n" +\
            "\tBandwidth estiates: " + str(self.b) + "."
        return res
