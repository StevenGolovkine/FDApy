"""
Representation of univariate and irregular functional data
==========================================================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy import IrregularFunctionalData
from FDApy.representation import DenseArgvals, IrregularArgvals, IrregularValues
from FDApy.visualization import plot

###############################################################################
# In this section, we are showing the building blocks of the representation of univariate and irregular functional data. To define a :class:`~FDApy.representation.IrregularFunctionalData` object, we need a set of :class:`~FDApy.representation.IrregularArgvals` (the sampling points of the curves) and a set of :class:`~FDApy.representation.IrregularValues` (the observed points of the curves). The sampling points of the data are defined as a dictionary where each entry corresponds to an observation. Each entry of the dictionary corresponds to the sampling points of one observation and is represented as a :class:`~FDApy.representation.DenseArgvals`. The values of the functional data are defined in a dictionary where each entry represents an observation as an :class:`~numpy.ndarray`. Each entry should have the same dimension has the corresponding entry in the :class:`~FDApy.representation.IrregularArgvals` dictionary.

###############################################################################
# For unidimensional functional data
# ----------------------------------
# First, we will define unidimensional irregular functional data. We represent two observations of a functional data irregularly sampled on a set of points. The first observation in sampled on :math:`20` points between :math:`0` and :math:`1` and the second observation is sampled on :math:`15` points between :math:`0.2` and :math:`0.8`. The values of the functional data is a dictionary where each entry corresponds to the observed values of one observation.

argvals = IrregularArgvals(
    {
        0: DenseArgvals({"input_dim_0": np.linspace(0, 1, num=20)}),
        1: DenseArgvals({"input_dim_0": np.linspace(0.2, 0.8, num=15)}),
    }
)
X = IrregularValues(
    {
        0: np.sin(2 * np.pi * argvals[0]["input_dim_0"]),
        1: np.cos(2 * np.pi * argvals[1]["input_dim_0"]),
    }
)

fdata = IrregularFunctionalData(argvals=argvals, values=X)

_ = plot(fdata)


###############################################################################
# For two-dimensional functional data
# -----------------------------------
# Second, we will consider two-dimensional functional data where the observations are not sampled on the same grid. We represent two observations of a functional data irregularly sampled on a set of points. The first observation is sampled on a grid of :math:`20 \times 20` sampling points and the second observation is sampled on a grid of :math:`15 \times 15` sampling points. The values of the functional data is a dictionary where each entry corresponds to the observed values of one observation.

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

fdata = IrregularFunctionalData(argvals=argvals, values=X)

_ = plot(fdata)

###############################################################################
# For higher-dimensional functional data
# --------------------------------------
# It is possible to define functional data with more than two dimensions. All you have to do is to add more dimension in the argvals and another dimension in the values array. However, no plotting function is available for data with more than two dimensions.
