"""
Two-dimensional Basis
=====================

"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation import Basis
from FDApy.representation import DenseArgvals
from FDApy.visualization import plot

# Parameters
name = ("fourier", "fourier")
n_functions = (5, 5)
argvals = DenseArgvals(
    {"input_dim_0": np.linspace(0, 1, 101), "input_dim_1": np.linspace(0, 1, 51)}
)

###############################################################################
basis = Basis(name=name, n_functions=n_functions, argvals=argvals)

_ = plot(basis)


###############################################################################
basis = Basis(name=name, n_functions=n_functions, argvals=argvals, add_intercept=False)

_ = plot(basis)
