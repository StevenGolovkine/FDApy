"""
Smoothing of dense two-dimensional functional data
==================================================

Examples of smoothing of univariate and dense functional data.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation import DenseArgvals
from FDApy.simulation import KarhunenLoeve
from FDApy.visualization import plot

# Set general parameters
rng = 42
n_obs = 4

# Parameters of the basis
name = ('bsplines', 'bsplines')
n_functions = (5, 5)

argvals = DenseArgvals({
    'input_dim_0': np.linspace(0, 1, 51),
    'input_dim_1': np.linspace(0, 1, 51)
})


kl = KarhunenLoeve(
    basis_name=name, argvals=argvals, n_functions=n_functions, random_state=rng
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
bandwidth = 0.5
degree = 1

data_smooth = kl.noisy_data.smooth(
    points=points, method='LP', kernel_name=kernel_name, bandwidth=bandwidth, degree=degree
)

_ = plot(data_smooth)