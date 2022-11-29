"""
Simulation of functional data
==========================

Examples of simulation of functional data and the effect of adding noise and sparsification.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.visualization.plot import plot

# Set general parameters
rng = 42
n_obs = 10

# Parameters of the basis
name = 'bsplines'
n_functions = 25

###############################################################################
# Simulation of functional data
# -----------------------------
#
# We simulate :math:`N = 10` curves on the one-dimensional observation grid
# :math:`\{0, 0.01, 0.02, \cdots, 1\}`, based on the first
# :math:`K = 25` B-splines basis functions on :math:`[0, 1]` and the variance of
# the scores random variables equal to :math:`1`.
kl = KarhunenLoeve(
    name=name, n_functions=n_functions, random_state=rng
)
kl.new(n_obs=n_obs)

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
# Sparsification
# --------------
#

# Sparsify the data
kl.sparsify(percentage=0.9, epsilon=0.05)

_ = plot(kl.sparse_data)
