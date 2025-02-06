"""
Representation of multivariate functional data
==============================================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy import (
    DenseFunctionalData,
    IrregularFunctionalData,
    MultivariateFunctionalData,
)
from FDApy.representation import DenseArgvals, IrregularArgvals
from FDApy.representation import DenseValues, IrregularValues
from FDApy.visualization import plot_multivariate


###############################################################################
# Multivariate functional data are defined as a list of univariate functional data and are represented with a :class:`~FDApy.representation.MultivariateFunctionalData` object. The univariate functional data can be of whatever dimension (curves, surfaces, ...) and dense, irregular or defined using a basis of function. There is no restriction on the number of elements in the list but each univariate element must have the same number of observations. It is possible to mix unidimensional and multidimensional functional data in the same list.

###############################################################################
# First example
# -------------
# First, we will define two univariate unidimensional dense functional data. Creating a list of these two objects, we can define a :class:`~FDApy.representation.MultivariateFunctionalData` object. We consider two observations of the multivariate functional data. The first feature is sampled on a hundred points between :math:`0` and :math:`\pi` and the second feature is sampled on fifty points between :math:`0` and :math:`1`.

argvals = np.linspace(0, np.pi, num=100)
X = np.array([np.sin(2 * np.pi * argvals), np.cos(2 * np.pi * argvals)])
fdata_first = DenseFunctionalData(
    argvals=DenseArgvals({"input_dim_0": argvals}), values=DenseValues(X)
)

argvals = np.linspace(0, 1, num=50)
X = np.array([np.exp(-argvals), np.log(1 + argvals)])
fdata_second = DenseFunctionalData(
    argvals=DenseArgvals({"input_dim_0": argvals}), values=DenseValues(X)
)

fdata = MultivariateFunctionalData([fdata_first, fdata_second])

_ = plot_multivariate(fdata)


###############################################################################
# Second exmaple
# --------------
# Second, we will define a multivariate functional data with one univariate dense functional data and one univariate irregular functional data. Both univariate functional data are two-dimensional. We consider two observations of the multivariate functional data. For the first feature, the first observation is sampled on a grid of :math:`20 \times 20` sampling points and the second observation is sampled on a grid of :math:`15 \times 15` sampling points. For the second feature, the observations are sampled on a hundred points between :math:`0` and :math:`\pi` for each dimension.

argvals = IrregularArgvals(
    {
        0: DenseArgvals(
            {
                "input_dim_0": np.linspace(0, 1, num=20),
                "input_dim_1": np.linspace(0, 1, num=20),
            }
        ),
        1: DenseArgvals(
            {
                "input_dim_0": np.linspace(0.2, 0.8, num=15),
                "input_dim_1": np.linspace(0.2, 0.8, num=15),
            }
        ),
    }
)
X = IrregularValues(
    {
        0: np.outer(
            np.sin(argvals[0]["input_dim_0"]), np.cos(argvals[0]["input_dim_1"])
        ),
        1: np.outer(
            np.sin(-argvals[1]["input_dim_0"]), np.cos(argvals[1]["input_dim_1"])
        ),
    }
)

fdata_first = IrregularFunctionalData(argvals=argvals, values=X)

argvals = np.linspace(0, np.pi, num=100)
X = np.array(
    [
        np.outer(np.sin(argvals), np.cos(argvals)),
        np.outer(np.sin(-argvals), np.cos(argvals)),
    ]
)
fdata_second = DenseFunctionalData(
    argvals=DenseArgvals({"input_dim_0": argvals, "input_dim_1": argvals}),
    values=DenseValues(X),
)

fdata = MultivariateFunctionalData([fdata_first, fdata_second])

_ = plot_multivariate(fdata)
