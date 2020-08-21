"""
Functional data simulation
==========================

This notebook shows how to simulate functional data according with different
basis.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

import numpy as np

from FDApy.representation.simulation import Brownian, KarhunenLoeve
from FDApy.visualization.plot import plot


###############################################################################
# Now, we will simulate some curves data using the Karhunen-Loève expansion.
#
kl = KarhunenLoeve(name='bsplines', n_functions=5)
kl.new(n_obs=100, argvals=np.linspace(0, 1, 301))

_ = plot(kl.data)

###############################################################################
# We can also add some noise to the data.
#
# First, we consider homoscedastic noise. Thus, we add realizations of the
# random variable :math:`\varepsilon \sim \mathcal{N}(0, \sigma^2)` to the
# data.
#

# Add some noise to the simulation.
kl.add_noise(0.05)

# Plot the noisy simulations
_ = plot(kl.noisy_data)


###############################################################################
# We can also simulate Brownian motion and some of processes derived from it,
# such as Geometric Brownian motion and Fractional Brownian motion.
#

###############################################################################
# Simulate some standard brownian motions.
brownian = Brownian(name='standard')
brownian.new(n_obs=100, argvals=np.linspace(0, 1, 301))

# Plot some simulations
_ = plot(brownian.data)

###############################################################################
# Simulate some geometric brownian motions.
brownian_geo = Brownian(name='geometric')
brownian_geo.new(n_obs=100, argvals=np.linspace(0, 1, 301))

# Plot some simulations
_ = plot(brownian_geo.data)

###############################################################################
# Simulate some fractional brownian motions.
brownian_frac = Brownian(name='fractional')
brownian_frac.new(n_obs=100, argvals=np.linspace(0, 1, 301), hurst=0.7)

# Plot some simulations
_ = plot(brownian_frac.data)
