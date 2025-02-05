"""
Simulation of clusters of univariate functional data
====================================================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation import DenseArgvals
from FDApy.simulation import KarhunenLoeve
from FDApy.visualization import plot

###############################################################################
# The package provides a class to simulate clusters of univariate functional data based on the Karhunen-Loève decomposition. The class :class:`FDApy.simulation.KarhunenLoeve` allows to simulate functional data based on the truncated Karhunen-Loève representation of a functional process.

###############################################################################
# We simulate :math:`N = 20` curves on the one-dimensional observation grid :math:`\{0, 0.01, 0.02, \cdots, 1\}`, based on the first :math:`K = 25` Fourier basis functions on :math:`[0, 1]`. The clusters are defined through the coefficients in the Karhunen-Loève decomposition and parametrize using the `centers` parameter. The centers of the clusters are generated as Gaussian random variables with parameters defined by a `mean` and a `covariance`. We also consider an exponential decreasing of the eigenvalues.


# Set general parameters
rng = 42
n_obs = 20

# Define the random state
random_state = np.random.default_rng(rng)

# Parameters of the basis
name = "fourier"
n_functions = 25
argvals = DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)})

# Parameters of the clusters
n_clusters = 2
mean = np.array([0, 0])
covariance = np.array([[1, -0.6], [-0.6, 1]])
centers = random_state.multivariate_normal(mean, covariance, size=n_functions)

kl = KarhunenLoeve(
    basis_name=name, argvals=argvals, n_functions=n_functions, random_state=rng
)
kl.new(n_obs=n_obs, n_clusters=n_clusters, centers=centers, cluster_std="exponential")

_ = plot(kl.data, kl.labels)
