"""
Simulation of clusters
======================

Examples of simulation of clusters of functional data.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.visualization.plot import plot

# Set general parameters
rng = 42
n_obs = 20

# Define the random state
random_state = np.random.default_rng(rng)

# Parameters of the basis
name = 'fourier'
n_functions = 25

# Parameters of the clusters
n_clusters = 2
mean = np.array([0, 0])
covariance = np.array([[1, -0.6], [-0.6, 1]])
centers = random_state.multivariate_normal(mean, covariance, size=n_functions)

###############################################################################
# We simulate :math:`N = 20` curves on the one-dimensional observation grid
# :math:`\{0, 0.01, 0.02, \cdots, 1\}`, based on the first
# :math:`K = 25` Fourier basis functions on :math:`[0, 1]`. The clusters are 
# defined through the coefficients in the Karhunen-Loève decomposition. The
# centers of the clusters are generated as Gaussian random variables with
# parameters defined by `mean` and `covariance`. We also consider an
# exponential decreasing of the eigenvalues.
kl = KarhunenLoeve(
    name=name, n_functions=n_functions, random_state=rng
)
kl.new(
    n_obs=n_obs,
    n_clusters=n_clusters,
    centers=centers,
    cluster_std='exponential'
)

_ = plot(kl.data, kl.labels)
