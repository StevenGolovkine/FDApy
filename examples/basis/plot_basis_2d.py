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
from FDApy.visualization import plot

# Parameters
name = "fourier"
n_functions = 5
argvals = np.linspace(0, 1, 101)
dimension = "2D"

###############################################################################
basis = Basis(name=name, n_functions=n_functions, argvals=argvals, dimension=dimension)

_ = plot(basis)


###############################################################################
basis = Basis(
    name=name,
    n_functions=n_functions,
    argvals=argvals,
    dimension=dimension,
    add_intercept=False,
)

_ = plot(basis)
