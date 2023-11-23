"""
FPCA of 2-dimensional data
--------------------------

Example of functional principal components analysis of 2-dimensional data.
"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.representation.functional_data import MultivariateFunctionalData
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import UFPCA
from FDApy.visualization.plot import plot, plot_multivariate

# Set general parameters
rng = 42
n_obs = 50
idx = 5
colors = np.array([[0.5, 0, 0, 1]])

# Parameters of the basis
name = 'bsplines'
n_functions = 5

###############################################################################
# We simulate :math:`N = 50` images on the two-dimensional observation grid
# :math:`\{0, 0.01, 0.02, \cdots, 1\} \times \{0, 0.01, 0.02, \cdots, 1\}`,
# based on the tensor product of the first :math:`K = 5` B-splines
# basis functions on :math:`[0, 1] \times [0, 1]` and the variance of
# the scores random variables equal to :math:`1`.
kl = KarhunenLoeve(
    basis_name=name, n_functions=n_functions, dimension='2D', random_state=rng
)
kl.new(n_obs=n_obs)
data = kl.data

_ = plot(data)

###############################################################################
# Inner-product matrix decomposition
# ----------------------------------
#
# Perform univariate FPCA with an estimation of the number of components by the
# percentage of variance explained using a decomposition of the inner-product
# matrix.
ufpca = UFPCA(n_components=0.99, method='inner-product')
ufpca.fit(data)

###############################################################################
# Estimate the scores -- projection of the curves onto the eigenfunctions --
# using the eigenvectors from the decomposition of the inner-product matrix.
# numerical integration.
scores = ufpca.transform(method='InnPro')

# Plot of the scores
_ = plt.scatter(scores[:, 0], scores[:, 1])

###############################################################################
# Reconstruct the curves using the scores.
data_recons = ufpca.inverse_transform(scores)

###############################################################################
# Plot an example of the curve reconstruction
data_multi = MultivariateFunctionalData([data[idx], data_recons[idx]])
_ = plot_multivariate(data_multi, titles=['True', 'Reconstruction'])
