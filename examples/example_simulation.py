"""
Functional data simulation
==========================

This notebook shows how to simulate functional data according with different
basis.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

from FDApy.basis import Basis, Brownian, basis_legendre, basis_wiener
from FDApy.plot import plot

###############################################################################
# We will define a Legendre polynomial basis using the method
# :func:`~FDApy.basis.basis_legendre`.
#

argvals = np.linspace(-1, 1, 1000)
LP = basis_legendre(K=5, argvals=argvals, norm=True)

# Plot the basis
fig, ax = plot(LP, main='Legendre basis', xlab='Sampling points')
plt.show()

###############################################################################
# Next, we will define a Wiener basis using the method
# :func:`~FDApy.basis.basis_wiener`.
#

argvals = np.linspace(-1, 1, 1000)
WP = basis_wiener(K=5, argvals=argvals, norm=True)

# Plot the basis
fig, ax = plot(WP, main='Wiener basis', xlab='Sampling points')
plt.show()

###############################################################################
# Now, we will simulate some curves data according to diverse basis with
# different eigenvalues decay.
#

# Legendre basis and exponential eigenvalues decay
sim = Basis(N=100, M=50, basis_name='legendre', K=5,
            eigenvalues='exponential', norm=True)
sim.new()

# Plot some simulations
fig, ax = plot(sim.obs_, main='Simulation', xlab='Sampling points')
plt.show()

# Legendre basis and linear eigenvalues decay
sim = Basis(N=100, M=50, basis_name='legendre', K=5,
            eigenvalues='linear', norm=True)
sim.new()

# Plot some simulations
fig, ax = plot(sim.obs_, main='Simulation', xlab='Sampling points')
plt.show()

# Wiener basis and Wiener eigenvalues decay
sim = Basis(N=100, M=50, basis_name='wiener', K=5,
            eigenvalues='wiener', norm=True)
sim.new()

# Plot some simulations
fig, ax = plot(sim.obs_, main='Simulation', xlab='Sampling points')
plt.show()

# Wiener basis and user-set eigenvalues
sim = Basis(N=100, M=50, basis_name='wiener', K=3,
            eigenvalues=[100, 25, 5], norm=True)
sim.new()

# Plot some simulations
fig, ax = plot(sim.obs_, main='Simulation', xlab='Sampling points')
plt.show()

###############################################################################
# We can also add some noise to the data.
#
# First, we consider homoscedastic noise. Thus, we add realizations of the
# random variable :math:`\varepsilon \sim \mathcal{N}(0, \sigma^2)` to the
# data.
#

# Add some noise to the simulation.
sim.add_noise(5)

# Plot the noisy simulations
fig, ax = plot(sim.noisy_obs_,
               main='Noisy simulation',
               xlab='Sampling points')
plt.show()

###############################################################################
# Second, we may add heteroscedatic noise to the data. In this case, the
# quantity added to the data is defined as realisations of the random variable
# :math:`\varepsilon \sim \mathcal{N}(0, \sigma^2(t))`.
#

# Add some heteroscedastic noise to the simulation
sim.add_noise(sd_function=lambda x: np.sqrt(np.abs(x) + 1))

# Plot the heteroscedastic noisy simulations
fig, ax = plot(sim.noisy_obs_,
               main='Noisy heteroscedastic simulation',
               xlab='Sampling points')
plt.show()

###############################################################################
# We can also simulate Brownian motion and some of processes derived from it,
# such as Geometric Brownian motion and Fractional Brownian motion.
#

# Simulate some standard brownian motions.
sim = Brownian(N=100, M=50, brownian_type='standard')
sim.new(x0=0)

# Plot some simulations
fig, ax = plot(sim.obs_,
               main='Standard Brownian motion',
               xlab='Sampling points')
plt.show()

# Simulate some geometric brownian motions.
sim = Brownian(N=100, M=50, brownian_type='geometric')
sim.new(x0=1, mu=5, sigma=1)

# Plot some simulations
fig, ax = plot(sim.obs_,
               main='Geometric Brownian motion',
               xlab='Sampling points')
plt.show()

# Simulate some fractional brownian motions.
sim = Brownian(N=100, M=50, brownian_type='fractional')
sim.new(hurst=0.7)

# Plot some simulations
fig, ax = plot(sim.obs_,
               main='Fractional Brownian motion',
               xlab='Sampling points')
plt.show()
