"""
Representation of univariate and dense functional data
======================================================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy import DenseFunctionalData
from FDApy.representation import DenseArgvals, DenseValues
from FDApy.visualization import plot

###############################################################################
# In this section, we are showing the building blocks of the representation of univariate and dense functional data. To define a :class:`~FDApy.representation.DenseFunctionalData` object, we need a set of :class:`~FDApy.representation.DenseArgvals` (the sampling points of the curves) and a set of :class:`~FDApy.representation.DenseValues` (the observed points of the curves). The sampling points of the functional data are defined as a dictionary where each entry is a one-dimensional numpy :class:`~numpy.ndarray` that represents an input dimension (one entry corresponds to curves, two entries correspond to surface, ...). The shape of the array of the first dimension would be :math:`(m_1,)`, the shape of the array of the second dimension would be :math:`(m_2,)` and so on. Curves will thus be sampled on :math:`m_1` points, surface will be sampled on :math:`m_1 \times m_2`, etc. The values of the functional data are defined as an :class:`~numpy.ndarray`. The shape of the array is :math:`(n, m_1, m_2, \dots)` where :math:`n` is the number of curves in the sample.

###############################################################################
# For unidimensional functional data
# ----------------------------------
# First, we will consider unidimensional dense functional data. We represent two observations of a functional data regularly sampled on a hundred points between :math:`0` and :math:`\pi`. The shape of the array of the values is :math:`(2, 100)`. The first dimension corresponds to the number of curves and the second dimension corresponds to the input dimension.

argvals = np.linspace(0, np.pi, num=100)
X = np.array([np.sin(2 * np.pi * argvals), np.cos(2 * np.pi * argvals)])

fdata = DenseFunctionalData(
    argvals=DenseArgvals({"input_dim_0": argvals}), values=DenseValues(X)
)

_ = plot(fdata)


###############################################################################
# For two-dimensional functional data
# -----------------------------------
# Second, we will consider two-dimensional dense functional data. We represent two observations of a functional data regularly sampled on a hundred points between :math:`0` and :math:`\pi` for each dimension. The shape of the array of the values is :math:`(2, 100, 100)`. The first dimension corresponds to the number of curves, the second dimension corresponds to the first input dimension and the third dimension corresponds to the second input dimension.

argvals = np.linspace(0, np.pi, num=100)
X = np.array(
    [
        np.outer(np.sin(argvals), np.cos(argvals)),
        np.outer(np.sin(-argvals), np.cos(argvals)),
    ]
)

fdata = DenseFunctionalData(
    argvals=DenseArgvals({"input_dim_0": argvals, "input_dim_1": argvals}),
    values=DenseValues(X),
)

_ = plot(fdata)

###############################################################################
# For higher-dimensional functional data
# --------------------------------------
# It is possible to define functional data with more than two dimensions. All you have to do is to add more entries in the dictionary of the argvals and another dimension in the values array. However, no plotting function is available for data with more than two dimensions.
