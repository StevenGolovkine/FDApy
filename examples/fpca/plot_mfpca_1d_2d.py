"""
MFPCA of 1- and 2-dimensional data
==================================

Example of multivariate functional principal components analysis of a
combinaison of 1- and 2-dimensional data.
"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.simulation import KarhunenLoeve
from FDApy.preprocessing import MFPCA
from FDApy.visualization import plot

# Set general parameters
rng = 42
n_obs = 50

# Parameters of the basis
name = ["bsplines", "fourier"]
n_functions = 5
dimension = ["1D", "2D"]
argvals = [np.linspace(0, 1, 101), np.linspace(0, 1, 21)]


###############################################################################
# We simulate :math:`N = 50` curves of a 2-dimensional process. The first
# component of the process is defined on the one-dimensional observation grid
# :math:`\{0, 0.01, 0.02, \cdots, 1\}`, based on the first :math:`K = 5`
# B-splines basis functions on :math:`[0, 1]` and the variance of the scores
# random variables equal to :math:`1`. The second component of the
# process is defined on the two-dimensional observation grid
# :math:`\{0, 0.05, 0.1, \cdots, 1\} \times \{0, 0.05, 0.1, \cdots, 1\}`,
# based on the tensor product of the first :math:`K = 5` Fourier
# basis functions on :math:`[0, 1] \times [0, 1]` and the variance of
# the scores random variables equal to :math:`1`.
kl = KarhunenLoeve(
    basis_name=name,
    n_functions=n_functions,
    argvals=argvals,
    dimension=dimension,
    add_intercept=False,
    random_state=rng,
)
kl.new(n_obs=50)
data = kl.data


# Plot of the data
fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1)
ax = plot(data.data[0], ax=ax)
ax.set_title("First component")

ax = fig.add_subplot(1, 2, 2, projection="3d")
ax = plot(data.data[1], ax=ax)
ax.set_title("Second component")

plt.show()


###############################################################################
# Covariance decomposition
# ------------------------
#
# Perform multivariate FPCA with an estimation of the variance explained for
# the first component and a prespecified number of components for the second
# component using the decomposition of the covariance operator. The
# decomposition of the covariance operator is based on the FCP-TPA algorithm
# for 2-dimensional data, which is an iterative algorithm. The number of
# components has thus to be prespecified.
mfpca_cov = MFPCA(n_components=[0.95, 5], method="covariance")
mfpca_cov.fit(data)


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
mfpca_innpro = MFPCA(n_components=0.95, method="inner-product")
mfpca_innpro.fit(data)


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
## Plot an example of the curve reconstruction
indexes = np.random.choice(n_obs, 5)

colors_numint = np.array([[0.9, 0, 0, 1]])
colors_pace = np.array([[0, 0.9, 0, 1]])
colors_innpro = np.array([[0.9, 0, 0.9, 1]])

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(16, 16))
for idx_plot, idx in enumerate(indexes):
    plot(data.data[0][idx], ax=axes[idx_plot, 0], label="True")
    plot(
        data_recons_cov.data[0][idx],
        colors=colors_numint,
        ax=axes[idx_plot, 0],
        label="Reconstruction NumInt",
    )
    plot(
        data_recons_innpro.data[0][idx],
        colors=colors_innpro,
        ax=axes[idx_plot, 0],
        label="Reconstruction InnPro",
    )
    axes[idx_plot, 0].legend()

    axes[idx_plot, 1] = plot(data.data[1][idx], ax=axes[idx_plot, 1])
    axes[idx_plot, 1].set_title("True")

    axes[idx_plot, 2] = plot(data_recons_cov.data[1][idx], ax=axes[idx_plot, 2])
    axes[idx_plot, 2].set_title("FCPTPA")

    axes[idx_plot, 3] = plot(data_recons_innpro.data[1][idx], ax=axes[idx_plot, 3])
    axes[idx_plot, 3].set_title("InnPro")

plt.show()
