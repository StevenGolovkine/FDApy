"""
MFPCA of 2-dimensional data
===========================

Example of multivariate functional principal components analysis of
2-dimensional data.
"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.representation import DenseArgvals
from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction import MFPCA
from FDApy.visualization import plot, plot_multivariate

# Set general parameters
rng = 42
n_obs = 50
idx = 5


# Parameters of the basis
name = [('bsplines', 'bsplines'), ('fourier', 'fourier')]
n_functions = [(5, 5), (5, 5)]
argvals = [
    DenseArgvals({
        'input_dim_0': np.linspace(0, 1, 21),
        'input_dim_1': np.linspace(0, 1, 21)
    }),
    DenseArgvals({
        'input_dim_0': np.linspace(0, 1, 21),
        'input_dim_1': np.linspace(0, 1, 21)
    })
]


###############################################################################
# We simulate :math:`N = 50` curves of a 2-dimensional process. The first
# component of the process is defined on the two-dimensional observation grid
# :math:`\{0, 0.05, 0.1, \cdots, 1\} \times \{0, 0.05, 0.1, \cdots, 1\}`,
# based on the tensor product of the first :math:`K = 5` B-splines
# basis functions on :math:`[0, 1] \times [0, 1]` and the variance of
# the scores random variables equal to :math:`1`. The second component of the
# process is defined on the two-dimensional observation grid
# :math:`\{0, 0.05, 0.1, \cdots, 1\} \times \{0, 0.05, 0.01, \cdots, 1\}`,
# based on the tensor product of the first :math:`K = 5` Fourier
# basis functions on :math:`[0, 1] \times [0, 1]` and the variance of
# the scores random variables equal to :math:`1`.
kl = KarhunenLoeve(
    basis_name=name,
    n_functions=n_functions,
    argvals=argvals,
    random_state=rng
)
kl.new(n_obs=50)
data = kl.data

_ = plot_multivariate(data)


###############################################################################
# Covariance decomposition
# ------------------------
#
# Perform multivariate FPCA with a prespecified number of components using the
# decomposition of the covariance operator. The decomposition of the covariance
# operator is based on the FCP-TPA algorithm, which is an iterative algorithm.
# The number of components has thus to be prespecified.
univariate_expansions = [
    {
        'method': 'FCPTPA',
        'n_components': 20
    },
    {
        'method': 'FCPTPA',
        'n_components': 20
    }
]
mfpca_cov = MFPCA(
    n_components=5, method='covariance',
    univariate_expansions=univariate_expansions
)
mfpca_cov.fit(data, method_smoothing='PS')


###############################################################################
# Estimate the scores -- projection of the curves onto the eigenfunctions -- by
# numerical integration.
scores_cov = mfpca_cov.transform(data, method="NumInt")

# Plot of the scores
_ = plt.scatter(scores_cov[:, 0], scores_cov[:, 1])


###############################################################################
# Reconstruct the curves using the scores.
data_recons_cov = mfpca_cov.inverse_transform(scores_cov)


###############################################################################
# Inner-product matrix decomposition
# ----------------------------------
#
# Perform multivariate FPCA with an estimation of the number of components by
# the percentage of variance explained using a decomposition of the
# inner-product matrix.
mfpca_innpro = MFPCA(n_components=5, method="inner-product")
mfpca_innpro.fit(data, method_smoothing='PS')


###############################################################################
# Estimate the scores -- projection of the curves onto the eigenfunctions --
# using the eigenvectors from the decomposition of the inner-product matrix.
scores_innpro = mfpca_innpro.transform(method="InnPro")

# Plot of the scores
_ = plt.scatter(scores_innpro[:, 0], scores_innpro[:, 1])


###############################################################################
# Reconstruct the surfaces using the scores.
data_recons_innpro = mfpca_innpro.inverse_transform(scores_innpro)


###############################################################################
# Plot an example of the curve reconstruction
# -------------------------------------------
indexes = np.random.choice(n_obs, 5)

# For the first component
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(16, 16))
for idx_plot, idx in enumerate(indexes):
    axes[idx_plot, 0] = plot(data.data[0][idx], ax=axes[idx_plot, 0])
    axes[idx_plot, 0].set_title("True")

    axes[idx_plot, 1] = plot(data_recons_cov.data[0][idx], ax=axes[idx_plot, 1])
    axes[idx_plot, 1].set_title("FCPTPA")

    axes[idx_plot, 2] = plot(data_recons_innpro.data[0][idx], ax=axes[idx_plot, 2])
    axes[idx_plot, 2].set_title("InnPro")

plt.show()


# For the second component
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(16, 16))
for idx_plot, idx in enumerate(indexes):
    axes[idx_plot, 0] = plot(data.data[1][idx], ax=axes[idx_plot, 0])
    axes[idx_plot, 0].set_title("True")

    axes[idx_plot, 1] = plot(data_recons_cov.data[1][idx], ax=axes[idx_plot, 1])
    axes[idx_plot, 1].set_title("FCPTPA")

    axes[idx_plot, 2] = plot(data_recons_innpro.data[1][idx], ax=axes[idx_plot, 2])
    axes[idx_plot, 2].set_title("InnPro")

plt.show()
