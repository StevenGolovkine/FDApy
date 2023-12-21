"""
MFPCA of 1-dimensional sparse data
----------------------------------

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
from FDApy.visualization.plot import plot, plot_multivariate

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
kl.add_noise_and_sparsify(noise_variance=0.05, percentage=0.5, epsilon=0.05)
data = kl.sparse_data

_ = plot_multivariate(data)


##############################################################################
# Covariance decomposition
# ------------------------

# Perform multivariate FPCA with an estimation of the number of components by
# the percentage of variance explained using a decomposition of the covariance
# operator.
mfpca_cov = MFPCA(n_components=[3, 3], method='covariance')
mfpca_cov.fit(data)

# Plot the eigenfunctions
_ = plot_multivariate(mfpca_cov.eigenfunctions)


###############################################################################
# Estimate the scores -- projection of the curves onto the eigenfunctions.
scores_numint = mfpca_cov.transform(data, method='NumInt')

# Plot of the scores
_ = plt.scatter(scores_numint[:, 0], scores_numint[:, 1])


###############################################################################
# Reconstruct the curves using the scores.
data_recons_numint = mfpca_cov.inverse_transform(scores_numint)

###############################################################################
# Inner-product matrix decomposition
# ----------------------------------
#
# Perform univariate FPCA with an estimation of the number of components by the
# percentage of variance explained using a decomposition of the inner-product
# matrix.
mfpca_innpro = MFPCA(n_components=3, method='inner-product')
mfpca_innpro.fit(data)

# Plot the eigenfunctions
_ = plot_multivariate(mfpca_innpro.eigenfunctions)

###############################################################################
# Estimate the scores -- projection of the curves onto the eigenfunctions --
# using the eigenvectors from the decomposition of the inner-product matrix.
scores_innpro = mfpca_innpro.transform(method='InnPro')

# Plot of the scores
_ = plt.scatter(scores_innpro[:, 0], scores_innpro[:, 1])


###############################################################################
# Reconstruct the curves using the scores.
data_recons_innpro = mfpca_innpro.inverse_transform(scores_innpro)


###############################################################################
# Comparison of the methods.
colors_numint = np.array([[0.9, 0, 0, 1]])
colors_pace = np.array([[0, 0.9, 0, 1]])
colors_innpro = np.array([[0.9, 0, 0.9, 1]])

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16,16))
for idx_plot, idx in enumerate(np.random.choice(n_obs, 5)):
    for idx_data, (dd, dd_numint, dd_innpro) in enumerate(zip(kl.data.data, data_recons_numint.data, data_recons_innpro.data)):
        axes[idx_plot, idx_data] = plot(dd[idx], ax=axes[idx_plot, idx_data])
        axes[idx_plot, idx_data] = plot(dd_numint[idx], colors=colors_numint, ax=axes[idx_plot, idx_data])
        axes[idx_plot, idx_data] = plot(dd_innpro[idx], colors=colors_innpro, ax=axes[idx_plot, idx_data])

    #temp_ax = axes.flatten()[idx_plot]
    #temp_ax = plot_multivariate(kl.data[idx], ax=temp_ax, label='True')
    #plot_multivariate(data_recons_numint[idx], colors=colors_numint, ax=temp_ax, label='Reconstruction NumInt')
    #plot_multivariate(data_recons_innpro[idx], colors=colors_innpro, ax=temp_ax, label='Reconstruction InnPro')
    #temp_ax.legend()
plt.show()
