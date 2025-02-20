"""
MFPCA of 1- and 2-dimensional data
==================================

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
# In this section, we are showing how to perform a multivariate functional principal component analysis on one-dimensional and two-dimensional data using the :class:`~FDApy.preprocessing.MFPCA` class. We will compare two methods to perform the dimension reduction: the decomposition of the covariance operator and the decomposition of the inner-product matrix. We will use :math:`0.9\%` of the variance explained in the data to reconstruct the curves.

# Set general parameters
rng = 42
n_obs = 50
idx = 5

# Parameters of the basis
name = ["bsplines", ("fourier", "fourier")]
n_functions = [9, (3, 3)]
argvals = [
    DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)}),
    DenseArgvals(
        {"input_dim_0": np.linspace(0, 1, 21), "input_dim_1": np.linspace(0, 1, 21)}
    ),
]

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
    basis_name=name, n_functions=n_functions, argvals=argvals, random_state=rng
)
kl.new(n_obs=50)
data = kl.data

_ = plot_multivariate(data)


###############################################################################
# Estimation of the eigencomponents
# ---------------------------------
#
# The :class:`~FDApy.preprocessing.MFPCA` class requires two parameters: the number of components to estimate and the method to use. The method parameter can be either `covariance` or `inner-product`. The first method estimates the eigenfunctions by decomposing the covariance operator, while the second method estimates the eigenfunctions by decomposing the inner-product matrix. In the case of a decomposition of the covariance operator, the method also requires the univariate expansions to estimate the eigenfunctions of each component. Here, we use the univariate functional principal component analysis with penalized splines to estimate the eigenfunctions of the first component and the FCP-TPA to estimate the eigenfunctions of the second component.

# First, we perform a multivariate FPCA using a decomposition of the covariance operator.
univariate_expansions = [
    {"method": "UFPCA", "n_components": 15, "method_smoothing": "PS"},
    {"method": "FCPTPA", "n_components": 20},
]
mfpca_cov = MFPCA(
    n_components=0.9, method="covariance", univariate_expansions=univariate_expansions
)
mfpca_cov.fit(data)

###############################################################################
#

# Second, we perform a multivariate FPCA using a decomposition of the inner-product matrix.
mfpca_innpro = MFPCA(n_components=0.95, method="inner-product")
mfpca_innpro.fit(data)


###############################################################################
# Estimation of the scores
# ------------------------
#
# Once the eigenfunctions are estimated, we can compute the scores using numerical integration or the eigenvectors from the decomposition of the inner-product matrix. Note that, when using the eigenvectors from the decomposition of the inner-product matrix, new data can not be passed as argument of the :func:`~FDApy.preprocessing.MFPCA.transform` method because the estimation is performed using the eigenvectors of the inner-product matrix.

scores_cov = mfpca_cov.transform(data, method="NumInt")
scores_innpro = mfpca_innpro.transform(method="InnPro")

# Plot of the scores
_ = plt.scatter(scores_cov[:, 0], scores_cov[:, 1], label="NumInt")
_ = plt.scatter(scores_innpro[:, 0], scores_innpro[:, 1], label="InnPro")
plt.legend()
plt.show()


###############################################################################
# Comparison of the methods
# -------------------------
#
# Finally, we compare the methods by reconstructing the curves using :math:`0.9\%` of the variance explained.
data_recons_cov = mfpca_cov.inverse_transform(scores_cov)
data_recons_innpro = mfpca_innpro.inverse_transform(scores_innpro)


###############################################################################
#
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
