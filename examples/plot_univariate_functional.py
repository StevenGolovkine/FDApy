"""
Smooth Univariate Functional data
=================================

This notebook shows how to smooth univariate functional data.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

import numpy as np
import matplotlib.pyplot as plt

from FDApy.representation.simulation import Brownian
from FDApy.visualization.plot import plot


###############################################################################
# Generate some data as Fractional Brownian motion.
#

# Simulate some fractional brownian motions.
brownian = Brownian(name='standard')
brownian.new(n_obs=1000, argvals=np.linspace(0, 1, 301))
brownian.add_noise(0.05)

# Plot some simulations
_ = plot(brownian.noisy_data)

###############################################################################
# Now, we will smooth the data according the methodology from (add ref).
#

# Smooth the data
data_smooth = brownian.noisy_data.smooth(points=0.5, neighborhood=14)

# Plot of the smoothing data
_ = plot(data_smooth)

###############################################################################
# In order to look more precisely at the smoothing results, let's plot one
# individual curve, along its noisy and smoothed version.
#

# Plot individual curves
idx = 5
fig, ax = plt.subplots(1, 1)
ax.scatter(brownian.noisy_data.argvals['input_dim_0'],
           brownian.noisy_data.values[idx, :],
           alpha=0.5, label='Noisy')
ax.plot(brownian.data.argvals['input_dim_0'],
        brownian.data.values[idx, :],
        color='red', label='True')
ax.plot(data_smooth.argvals['input_dim_0'],
        data_smooth.values[idx, :],
        color='green', label='Smooth')
ax.set_xlabel('Sampling points')
ax.legend()
