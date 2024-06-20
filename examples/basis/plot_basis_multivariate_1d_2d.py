"""
Multivariate Basis of multi-dimensional data
============================================

"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation import MultivariateBasis
from FDApy.representation import DenseArgvals
from FDApy.visualization import plot_multivariate

# Parameters
basis_name = ["fourier", ("legendre", "legendre")]
argvals = [
    DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)}),
    DenseArgvals(
        {"input_dim_0": np.linspace(0, 0.5, 11), "input_dim_1": np.linspace(0, 1, 11)}
    ),
]
n_functions = [9, (3, 3)]

###############################################################################
basis = MultivariateBasis(name=basis_name, n_functions=n_functions, argvals=argvals)

_ = plot_multivariate(basis)
