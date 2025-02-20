"""
Smoothing of dense two-dimensional functional data
==================================================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation import DenseArgvals
from FDApy.simulation import KarhunenLoeve
from FDApy.visualization import plot

###############################################################################
# The package includes different smoothing methods to smooth functional data. In this section, we are showing the building blocks of the smoothing of dense two-dimensional functional data. First, we simulate functional data using the Karhunen-Loève decomposition using B-splines basis functions. We then add some noise to the simulation.

# Set general parameters
rng = 42
n_obs = 4

# Parameters of the basis
name = ("bsplines", "bsplines")
n_functions = (5, 5)

argvals = DenseArgvals({
    "input_dim_0": np.linspace(0, 1, 51),
    "input_dim_1": np.linspace(0, 1, 51)
})


kl = KarhunenLoeve(
    basis_name=name, argvals=argvals, n_functions=n_functions, random_state=rng
)
kl.new(n_obs=n_obs)
data = kl.data

# Add some noise to the simulation.
kl.add_noise(0.05)

###############################################################################
# Smoothing two-dimensional functional data is similar to smoothing one-dimensional functional data. The main difference is that the smoothing is done in two dimensions. In this example, we will smooth the noisy data using the :func:`~FDApy.representation.DenseFunctionalData.smooth` function. This function allows to smooth the data using different methods such as local polynomials and P-splines. In this example, we will use the local polynomials smoothing method with an Epanechnikov kernel and a bandwidth of :math:`0.5`. We plot the smoothed data.

# Smooth the data
points = DenseArgvals({
    "input_dim_0": np.linspace(0, 1, 11),
    "input_dim_1": np.linspace(0, 1, 11)
})
kernel_name = "epanechnikov"
bandwidth = 0.5
degree = 1

data_smooth = kl.noisy_data.smooth(
    points=points,
    method="LP",
    kernel_name=kernel_name,
    bandwidth=bandwidth,
    degree=degree,
)

_ = plot(data_smooth)
