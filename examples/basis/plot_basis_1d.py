"""
Basis
=====

"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation.basis import Basis
from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.visualization.plot import plot

# Parameters
name = 'fourier'
n_functions = 5
argvals = np.linspace(0, 1, 101)
dimension = '1D'

###############################################################################
basis = Basis(
    name=name, n_functions=n_functions, argvals=argvals, dimension=dimension
)

_ = plot(basis)