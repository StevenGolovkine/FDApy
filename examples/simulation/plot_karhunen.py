"""
Simulation using Karhunen-Loève decomposition
=============================================

Examples of simulation using the decomposition of Karhunen-Loève decomposition.
"""

###############################################################################
# A Brownian motion is a real valued continuous-time random process.
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.visualization.plot import plot

# Set general parameters
rng = 42
n_obs = 10
argvals = np.arange(0, 1.01, 0.01)

# Parameters of the basis
name = 'fourier'
n_functions = 25

###############################################################################
# Curves simulation
# ------------------------
#
kl = KarhunenLoeve(name=name, n_functions=n_functions, random_state=rng)
kl.new(n_obs=n_obs)

_ = plot(kl.data)
