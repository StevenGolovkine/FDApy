"""
Representation of multivariate functional data
==============================================

Examples of representation of multivariate functional data.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData,
    MultivariateFunctionalData
)
from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
from FDApy.representation.values import DenseValues, IrregularValues
from FDApy.visualization.plot import plot_multivariate


###############################################################################
# The representation of multivariate functional data
# --------------------------------------------------
# Multivariate functional data are defined as a list of univariate functional
# data. The univariate functional data can be of whatever dimension (curves,
# surfaces, ...) and dense or irregular. There is no restriction on the number
# of elements in the list but each univariate element must have the same number
# of observations.

###############################################################################
# First example
# -------------
# First, we will define two univariate unidimensional dense functional data.
# By putting them as a list, we can define a ``MultivariateFunctionalData``
# object.
argvals = np.linspace(0, np.pi, num=100)
X = np.array([
    np.sin(2 * np.pi * argvals),
    np.cos(2 * np.pi * argvals)
])
fdata_first = DenseFunctionalData(
    argvals=DenseArgvals({'input_dim_0': argvals}),
    values=DenseValues(X)
)

argvals = np.linspace(0, 1, num=50)
X = np.array([
    np.exp(-argvals),
    np.log(1 + argvals)
])
fdata_second = DenseFunctionalData(
    argvals=DenseArgvals({'input_dim_0': argvals}),
    values=DenseValues(X)
)

fdata = MultivariateFunctionalData([fdata_first, fdata_second])

_ = plot_multivariate(fdata)


###############################################################################
# Second exmaple
# --------------
# Second, we will define a univariate unidimensional irregular functional data
# and a univariate two-dimensional dense functional data. By putting them as a
# list, we can define a ``MultivariateFunctionalData`` object.
argvals = IrregularArgvals({
    0: DenseArgvals({
        'input_dim_0': np.linspace(0, 1, num=20),
        'input_dim_1': np.linspace(0, 1, num=20)
    }),
    1: DenseArgvals({
        'input_dim_0': np.linspace(0.2, 0.8, num=15),
        'input_dim_1': np.linspace(0.2, 0.8, num=15)
    })
})
X = IrregularValues({
    0: np.outer(
        np.sin(argvals[0]['input_dim_0']),
        np.cos(argvals[0]['input_dim_1'])
    ),
    1: np.outer(
        np.sin(-argvals[1]['input_dim_0']),
        np.cos(argvals[1]['input_dim_1'])
    )
})

fdata_first = IrregularFunctionalData(
    argvals=argvals,
    values=X
)

argvals = np.linspace(0, np.pi, num=100)
X = np.array([
    np.outer(np.sin(argvals), np.cos(argvals)),
    np.outer(np.sin(-argvals), np.cos(argvals))
])
fdata_second = DenseFunctionalData(
    argvals=DenseArgvals({'input_dim_0': argvals, 'input_dim_1': argvals}),
    values=DenseValues(X)
)

fdata = MultivariateFunctionalData([fdata_first, fdata_second])

# Note: For 2D data, only the first observation is plotted.
