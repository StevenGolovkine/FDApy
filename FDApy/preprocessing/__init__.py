"""
Preprocessing module.

This module is used to manage the different preprocessing step of functional
data. It includes functions about dimension reduction and smoothing.
"""

from .dim_reduction.fpca import UFPCA, MFPCA
from .smoothing.bandwidth import Bandwidth
from .smoothing.local_polynomial import LocalPolynomial
