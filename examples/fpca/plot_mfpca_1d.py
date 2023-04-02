"""
MFPCA of 1-dimensional data
===========================

Example of multivariate functional principal components analysis of
1-dimensional data.
"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import MFPCA
from FDApy.visualization.plot import plot_multivariate

# Set general parameters
rng = 42
n_obs = 50
idx = 5
colors = np.array([[0.5, 0, 0, 1]])


# Parameters of the basis
name = ['bsplines', 'fourier']
n_functions = 5

###############################################################################
# We simulate :math:`N = 50` curves of a 2-dimensional process. The first
# component of the process is defined on the one-dimensional observation grid
# :math:`\{0, 0.01, 0.02, \cdots, 1\}`, based on the first :math:`K = 5`
# B-splines basis functions on :math:`[0, 1]` and the variance of the scores
# random variables equal to :math:`1`. The second component of the process is
# defined on the one-dimensional observation grid
# :math:`\{0, 0.01, 0.02, \cdots, 1\}`, based on the first :math:`K = 5`
# Fourier basis functions on :math:`[0, 1]` and the variance of the scores
# random variables equal to :math:`1`.
kl = KarhunenLoeve(
    basis_name=name, n_functions=n_functions, random_state=rng
)
kl.new(n_obs=n_obs)
data = kl.data

_ = plot_multivariate(data)

###############################################################################
# Covariance decomposition
# ------------------------
#
# Perform multivariate FPCA with an estimation of the number of components by
# the percentage of variance explained using a decomposition of the covariance
# operator.
mfpca = MFPCA(n_components=[0.99, 0.99], method='covariance')
mfpca.fit(data)

# Plot the eigenfunctions
_ = plot_multivariate(mfpca.eigenfunctions)

###############################################################################
# Estimate the scores -- projection of the curves onto the eigenfunctions -- by
# numerical integration.
scores = mfpca.transform(data, method='NumInt')

# Plot of the scores
_ = plt.scatter(scores[:, 0], scores[:, 1])

###############################################################################
# Reconstruct the curves using the scores.
data_recons = mfpca.inverse_transform(scores)

ax = plot_multivariate(data.get_obs(idx))
ax = plot_multivariate(data_recons.get_obs(idx), colors=colors, ax=ax)
plt.legend(['True', 'Reconstruction'])
plt.show()

###############################################################################
# Inner-product matrix decomposition
# ----------------------------------
#
# Perform univariate FPCA with an estimation of the number of components by the
# percentage of variance explained using a decomposition of the inner-product
# matrix.
mfpca = MFPCA(n_components=0.99, method='inner-product')
mfpca.fit(data)

# Plot the eigenfunctions
_ = plot_multivariate(mfpca.eigenfunctions)

###############################################################################
# Estimate the scores -- projection of the curves onto the eigenfunctions --
# using the eigenvectors from the decomposition of the inner-product matrix.
scores = mfpca.transform(data, method='InnPro')

# Plot of the scores
_ = plt.scatter(scores[:, 0], scores[:, 1])

###############################################################################
# Reconstruct the curves using the scores.
data_recons = mfpca.inverse_transform(scores)

ax = plot_multivariate(data.get_obs(idx))
ax = plot_multivariate(data_recons.get_obs(idx), colors=colors, ax=ax)
plt.legend(['True', 'Reconstruction'])
plt.show()
