"""
Smooth Univariate Functional data
=================================

This notebook shows how to smooth univariate functional data.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt

from FDApy.basis import Brownian
from FDApy.plot import plot


###############################################################################
# Generate some data as Fractional Brownian motion.
#

# Simulate some fractional brownian motions.
sim = Brownian(N=1000, M=300, brownian_type='fractional')
data = sim.new(x0=0, H=0.5)
data_noisy = data.add_noise(0.05)

# Plot some simulations
fig, ax = plot(data_noisy,
               main='Fractional Brownian motion',
               xlab='Sampling points')

###############################################################################
# Now, we will smooth the data according the methodology from (add ref).
#

# Smooth the data
data_smooth = data_noisy.smooth(t0=0.5, k0=14)

# Plot of the smoothing data
fig, ax = plot(data_smooth,
               main='Fractional Brownian motion smoothed',
               xlab='Sampling points')

###############################################################################
# In order to look more precisely at the smoothing results, let's plot one
# individual curve, along its noisy and smoothed version.
#

# Plot individual curves
idx = 5
fig, ax = plt.subplots(1, 1)
ax.scatter(data_noisy.argvals[0],
           data_noisy.values[idx, :],
           alpha=0.5, label='Noisy')
ax.plot(data.argvals[0], data.values[idx, :],
        color='red', label='True')
ax.plot(data_smooth.argvals[idx], data_smooth.values[idx],
        color='green', label='Smooth')
ax.set_xlabel('Sampling points')
ax.legend()
