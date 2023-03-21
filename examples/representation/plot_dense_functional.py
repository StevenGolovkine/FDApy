"""
Representation of univariate functional data
============================================

Examples of representation of univariate functional data.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.visualization.plot import plot


###############################################################################
# For unidimensional functional data
# ----------------------------------
#
# 
argvals = np.linspace(0, np.pi, num=100)
X = np.array([
    np.sin(2 * np.pi * argvals),
    np.cos(2 * np.pi * argvals)
])

fdata = DenseFunctionalData(
    argvals={'input_dim_0': argvals},
    values=X
)

_ = plot(fdata)


###############################################################################
# For two-dimensional functional data
# -----------------------------------
#
#
argvals = np.linspace(0, np.pi, num=100)
X = np.array([
    np.outer(np.sin(argvals), np.cos(argvals)),
    np.outer(np.sin(-argvals), np.cos(argvals))
])

fdata = DenseFunctionalData(
    argvals={'input_dim_0': argvals, 'input_dim_1': argvals},
    values=X
)

_ = plot(fdata)
