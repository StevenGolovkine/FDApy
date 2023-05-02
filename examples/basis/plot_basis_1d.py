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

from FDApy.representation.basis import Basis
from FDApy.visualization.plot import plot

# Parameters
n_functions = 5
argvals = np.linspace(0, 1, 101)
dimension = '1D'

###############################################################################
# Fourier
basis = Basis(
    name='fourier', n_functions=n_functions,
    argvals=argvals, dimension=dimension
)

_ = plot(basis)

###############################################################################
# B-splines
basis = Basis(
    name='bsplines', n_functions=n_functions,
    argvals=argvals, dimension=dimension
)

_ = plot(basis)

###############################################################################
# Wiener
basis = Basis(
    name='wiener', n_functions=n_functions,
    argvals=argvals, dimension=dimension
)

_ = plot(basis)
