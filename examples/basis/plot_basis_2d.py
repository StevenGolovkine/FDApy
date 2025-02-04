"""
Two-dimensional Basis
=====================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation import Basis
from FDApy.representation import DenseArgvals
from FDApy.visualization import plot

###############################################################################
# It is possible to define multi-dimensional basis functions using the :class:`~FDApy.representation.Basis` object. Multidimensional basis functions are defined as a tensor product of unidimensional basis. To create a multidimensional basis functions, we need a tuple of names and a tuple of number of functions. The sampling points are defined as a :class:`~FDApy.representation.DenseArgvals` object where each entry corresponds to the sampling points of one input dimension.

# Parameters
name = ("fourier", "fourier")
n_functions = (5, 5)
argvals = DenseArgvals(
    {"input_dim_0": np.linspace(0, 1, 101), "input_dim_1": np.linspace(0, 1, 51)}
)

###############################################################################
# Here, we show the basis functions for the tensor product of two Fourier basis. The basis functions consist of the tensor product of sine and cosine functions with a frequency that increases with the number of the function.
basis = Basis(name=name, n_functions=n_functions, argvals=argvals)

_ = plot(basis)
