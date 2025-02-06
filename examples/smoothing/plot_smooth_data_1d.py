"""
Smoothing of dense one-dimensional functional data
==================================================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.simulation import KarhunenLoeve
from FDApy.representation import DenseArgvals
from FDApy.visualization import plot

###############################################################################
# The package includes different smoothing methods to smooth functional data. In this section, we are showing the building blocks of the smoothing of dense unidimensional functional data. First, we simulate functional data using the Karhunen-Loève decomposition using B-splines basis functions. We then add some noise to the simulation.


# Set general parameters
rng = 42
n_obs = 4

# Parameters of the basis
name = "bsplines"
n_functions = 5
points = DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)})

# Simulate data
kl = KarhunenLoeve(
    basis_name=name,
    n_functions=n_functions,
    argvals=points,
    random_state=rng
)
kl.new(n_obs=n_obs)

# Add some noise to the simulation.
kl.add_noise(0.05)


##############################################################################
# We will smooth the noisy data using the :func:`~FDApy.representation.DenseFunctionalData.smooth` function. This function allows to smooth the data using different methods such as local polynomials and P-splines. In this example, we will use the local polynomials smoothing method with an Epanechnikov kernel and a bandwidth of :math:`0.1`. In the plot, the red line represents the true data, the grey line represents the noisy data and the blue line represents the smoothed data.

# Smooth the data
kernel_name = "epanechnikov"
bandwidth = 0.1
degree = 1

fdata_smooth = kl.noisy_data.smooth(
    points=points,
    method="LP",
    kernel_name=kernel_name,
    bandwidth=bandwidth,
    degree=degree,
)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for idx, ax in enumerate(axes.flat):
   plot(kl.noisy_data[idx], colors="k", alpha=0.2, ax=ax)
   plot(kl.data[idx], colors="r", ax=ax)
   plot(fdata_smooth[idx], colors="b", ax=ax)
   ax.set_title(f"Observation {idx + 1}")

plt.show()
