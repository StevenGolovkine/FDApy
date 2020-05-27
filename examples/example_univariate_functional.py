"""
Smooth Univariate Functional data
==========================

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
sim.new(x0=0, hurst=0.5)
sim.add_noise(0.05)

# Plot some simulations
fig, ax = plot(sim.noisy_obs_,
               main='Fractional Brownian motion',
               xlab='Sampling points')
plt.show()

###############################################################################
# Now, we will smooth the data according the methodology from (add ref).
#

# Smooth the data
sim_smooth = sim.noisy_obs_.smooth(t0=0.5, k0=14)

# Plot of the smoothing data
fig, ax = plot(sim_smooth,
               main='Fractional Brownian motion smoothed',
               xlab='Sampling points')
plt.show()

###############################################################################
# In order to look more precisely at the smoothing results, let's plot one
# individual curve, along its noisy and smoothed version.
#

# Plot individual curves
idx = 5
fig, ax = plt.subplots(1, 1)
ax.scatter(sim.noisy_obs_.argvals[0],
           sim.noisy_obs_.values[idx, :],
           alpha=0.5, label='Noisy')
ax.plot(sim.obs_.argvals[0], sim.obs_.values[idx, :],
        color='red', label='True')
ax.plot(sim_smooth.argvals[idx], sim_smooth.values[idx],
        color='green', label='Smooth')
ax.set_xlabel('Sampling points')
ax.legend()

plt.show()
