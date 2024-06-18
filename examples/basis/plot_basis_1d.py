"""
One-dimensional Basis
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
n_functions = 5
argvals = DenseArgvals({'input_dim_0': np.linspace(0, 1, 101)})


###############################################################################
# Fourier
basis = Basis(
    name="fourier", n_functions=n_functions, argvals=argvals
)

_ = plot(basis)

###############################################################################
# B-splines
basis = Basis(
    name="bsplines", n_functions=n_functions, argvals=argvals
)

_ = plot(basis)

###############################################################################
# Wiener
basis = Basis(
    name="wiener", n_functions=n_functions, argvals=argvals
)

_ = plot(basis)
