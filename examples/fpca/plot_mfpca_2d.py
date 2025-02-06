"""
MFPCA of 2-dimensional data
===========================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.representation import DenseArgvals
from FDApy.simulation import KarhunenLoeve
from FDApy.preprocessing import MFPCA
from FDApy.visualization import plot, plot_multivariate

###############################################################################
# In this section, we are showing how to perform a multivariate functional principal component analysis on two-dimensional data using the :class:`~FDApy.preprocessing.MFPCA` class. We will compare two methods to perform the dimension reduction: the FCP-TPA and the decomposition of the inner-product matrix. We will use the first :math:`K = 5` principal components to reconstruct the curves.


# Set general parameters
rng = 42
n_obs = 50
idx = 5


# Parameters of the basis
name = [("bsplines", "bsplines"), ("fourier", "fourier")]
n_functions = [(5, 5), (5, 5)]
argvals = [
    DenseArgvals(
        {"input_dim_0": np.linspace(0, 1, 21), "input_dim_1": np.linspace(0, 1, 21)}
    ),
    DenseArgvals(
        {"input_dim_0": np.linspace(0, 1, 21), "input_dim_1": np.linspace(0, 1, 21)}
    ),
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
    basis_name=name, n_functions=n_functions, argvals=argvals, random_state=rng
)
kl.new(n_obs=50)
data = kl.data

_ = plot_multivariate(data)


###############################################################################
# Estimation of the eigencomponents
# ---------------------------------
#
# The :class:`~FDApy.preprocessing.MFPCA` class requires two parameters: the number of components to estimate and the method to use. The method parameter can be either `covariance` or `inner-product`. The first method estimates the eigenfunctions by decomposing the covariance operator, while the second method estimates the eigenfunctions by decomposing the inner-product matrix. In the case of a decomposition of the covariance operator, the method also requires the univariate expansions to estimate the eigenfunctions of each component. Here, we use the FCP-TPA to estimate the eigenfunctions of each component.

# First, we perform a multivariate FPCA using a decomposition of the covariance operator.
univariate_expansions = [
    {"method": "FCPTPA", "n_components": 20},
    {"method": "FCPTPA", "n_components": 20},
]
mfpca_cov = MFPCA(
    n_components=5, method="covariance", univariate_expansions=univariate_expansions
)
mfpca_cov.fit(data, method_smoothing="PS")

###############################################################################
#

# Second, we perform a multivariate FPCA using a decomposition of the inner-product matrix.
mfpca_innpro = MFPCA(n_components=5, method="inner-product")
mfpca_innpro.fit(data, method_smoothing="PS")


###############################################################################
# Estimation of the scores
# ------------------------
#
# Once the eigenfunctions are estimated, we can compute the scores using numerical integration or the eigenvectors from the decomposition of the inner-product matrix. Note that, when using the eigenvectors from the decomposition of the inner-product matrix, new data can not be passed as argument of the :func:`~FDApy.preprocessing.MFPCA.transform` method because the estimation is performed using the eigenvectors of the inner-product matrix.
scores_cov = mfpca_cov.transform(data, method="NumInt")
scores_innpro = mfpca_innpro.transform(method="InnPro")

# Plot of the scores
_ = plt.scatter(scores_cov[:, 0], scores_cov[:, 1], label="FCPTPA")
_ = plt.scatter(scores_innpro[:, 0], scores_innpro[:, 1], label="InnPro")
plt.legend()
plt.show()


###############################################################################
# Comparison of the methods
# -------------------------
#
# Finally, we compare the reconstruction of the curves using the first :math:`K = 5` principal components.
data_recons_cov = mfpca_cov.inverse_transform(scores_cov)
data_recons_innpro = mfpca_innpro.inverse_transform(scores_innpro)


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
