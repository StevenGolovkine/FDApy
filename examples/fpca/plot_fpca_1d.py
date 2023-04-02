"""
FPCA of 1-dimensional data
--------------------------

Example of functional principal components analysis of 1-dimensional data.
"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import UFPCA
from FDApy.visualization.plot import plot

# Set general parameters
rng = 42
n_obs = 50
idx = 5
colors = np.array([[0.5, 0, 0, 1]])

# Parameters of the basis
name = 'fourier'
n_functions = 25

###############################################################################
# We simulate :math:`N = 50` curves on the one-dimensional observation grid
# :math:`\{0, 0.01, 0.02, \cdots, 1\}`, based on the first :math:`K = 25`
# Fourier basis functions on :math:`[0, 1]` and the variance of the scores
# random variables equal to :math:`1`.
kl = KarhunenLoeve(
    basis_name=name, n_functions=n_functions, random_state=rng
)
kl.new(n_obs=n_obs)
data = kl.data

_ = plot(data)

###############################################################################
# Covariance decomposition
# ------------------------
#
# Perform univariate FPCA with a predefined number of components using a
# decomposition of the covariance operator.
ufpca = UFPCA(n_components=3, method='covariance')
ufpca.fit(data)

# Plot the eigenfunctions
_ = plot(ufpca.eigenfunctions)

###############################################################################
# Estimate the scores -- projection of the curves onto the eigenfunctions -- by
# numerical integration.
scores = ufpca.transform(data, method='NumInt')

# Plot of the scores
_ = plt.scatter(scores[:, 0], scores[:, 1])

###############################################################################
# Reconstruct the curves using the scores.
data_recons = ufpca.inverse_transform(scores)

###############################################################################
# Plot an example of the curve reconstruction
ax = plot(data[idx], label='True')
plot(data_recons[idx], colors=colors, ax=ax, label='Reconstruction')
plt.legend()
plt.show()

###############################################################################
# Inner-product matrix decomposition
# ----------------------------------
#
# Perform univariate FPCA with an estimation of the number of components by the
# percentage of variance explained using a decomposition of the inner-product
# matrix.
ufpca = UFPCA(n_components=0.99, method='inner-product')
ufpca.fit(data)

# Plot the eigenfunctions
_ = plot(ufpca.eigenfunctions)

###############################################################################
# Estimate the scores -- projection of the curves onto the eigenfunctions --
# using the eigenvectors from the decomposition of the inner-product matrix.
scores = ufpca.transform(data, method='InnPro')

# Plot of the scores
_ = plt.scatter(scores[:, 0], scores[:, 1])

###############################################################################
# Reconstruct the curves using the scores.
data_recons = ufpca.inverse_transform(scores)

###############################################################################
# Plot an example of the curve reconstruction
ax = plot(data[idx], label='True')
plot(data_recons[idx], colors=colors, ax=ax, label='Reconstruction')
plt.legend()
plt.show()
