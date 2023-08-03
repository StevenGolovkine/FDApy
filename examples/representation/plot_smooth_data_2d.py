"""
Smoothing of dense two-dimensional functional data
==================================================

Examples of smoothing of univariate and dense functional data.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.representation.functional_data import MultivariateFunctionalData
from FDApy.representation.argvals import DenseArgvals
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.visualization.plot import plot_multivariate

# Set general parameters
rng = 42
n_obs = 4

# Parameters of the basis
name = 'bsplines'
n_functions = 5


argvals = np.linspace(0, 1, 51)
kl = KarhunenLoeve(
    basis_name=name, argvals=argvals, n_functions=n_functions,
    dimension='2D', random_state=rng
)
kl.new(n_obs=n_obs)
data = kl.data

# Add some noise to the simulation.
kl.add_noise(0.05)

# Smooth the data
points = DenseArgvals({
    'input_dim_0': np.linspace(0, 1, 11),
    'input_dim_1': np.linspace(0, 1, 11)
})
kernel_name = "epanechnikov"
bandwidth = 0.2
degree = 1

data_smooth = kl.noisy_data.smooth(
    points=points, kernel_name=kernel_name, bandwidth=bandwidth, degree=degree
)


_ = plot_multivariate(
    MultivariateFunctionalData([data[0], data_smooth[0], kl.noisy_data[0]]),
    titles=['True', 'Smooth', 'Noisy'],
    ncols=3
)
