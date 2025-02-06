"""
FPCA of 1-dimensional data
--------------------------

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.representation import DenseArgvals
from FDApy.simulation import KarhunenLoeve
from FDApy.preprocessing import UFPCA
from FDApy.visualization import plot

###############################################################################
# In this section, we are showing how to perform a functional principal component on one-dimensional data using the :class:`~FDApy.preprocessing.UFPCA` class. We will compare two methods to perform the dimension reduction: the decomposition of the covariance operator and the decomposition of the inner-product matrix. We will use the first :math:`K = 5` principal components to reconstruct the curves.


# Set general parameters
rng = 42
n_obs = 50

# Parameters of the basis
name = "fourier"
n_functions = 25
argvals = DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)})


###############################################################################
# We simulate :math:`N = 50` curves on the one-dimensional observation grid :math:`\{0, 0.01, 0.02, \cdots, 1\}`, based on the first :math:`K = 25` Fourier basis functions on :math:`[0, 1]` and the variance of the scores random variables decreasing exponentially.

kl = KarhunenLoeve(
    n_functions=n_functions, basis_name=name, argvals=argvals, random_state=rng
)
kl.new(n_obs=n_obs, clusters_std="exponential")
kl.add_noise(noise_variance=0.05)
data = kl.noisy_data

_ = plot(data)
plt.show()

###############################################################################
# Estimation of the eigencomponents
# ---------------------------------
#
# The :class:`~FDApy.preprocessing.UFPCA` class requires two parameters: the number of components to estimate and the method to use. The method parameter can be either `covariance` or `inner-product`. The first method estimates the eigenfunctions by decomposing the covariance operator, while the second method estimates the eigenfunctions by decomposing the inner-product matrix.

# First, we perform a univariate FPCA using a decomposition of the covariance operator.
ufpca_cov = UFPCA(n_components=5, method="covariance")
ufpca_cov.fit(data)

# Plot the eigenfunctions using the decomposition of the covariance operator.
_ = plot(ufpca_cov.eigenfunctions)
plt.show()

###############################################################################
#

# Second, we perform a univariate FPCA using a decomposition of the inner-product matrix.
ufpca_innpro = UFPCA(n_components=5, method="inner-product")
ufpca_innpro.fit(data)

# Plot the eigenfunctions using the decomposition of the inner-product matrix.
_ = plot(ufpca_innpro.eigenfunctions)
plt.show()

###############################################################################
# Estimation of the scores
# ------------------------
#
# Once the eigenfunctions are estimated, we can compute the scores using numerical integration, the PACE algorithm or the eigenvectors from the decomposition of the inner-product matrix. The :func:`~FDApy.preprocessing.UFPCA.transform` method requires the data as argument and the method to use. The method parameter can be either `NumInt`, `PACE` or `InnPro`. Note that, when using the eigenvectors from the decomposition of the inner-product matrix, new data can not be passed as argument of the :func:`~FDApy.preprocessing.UFPCA.transform` method because the estimation is performed using the eigenvectors of the inner-product matrix.

scores_numint = ufpca_cov.transform(data, method="NumInt")
scores_pace = ufpca_cov.transform(data, method="PACE")
scores_innpro = ufpca_innpro.transform(method="InnPro")

# Plot of the scores
plt.scatter(scores_numint[:, 0], scores_numint[:, 1], label="NumInt")
plt.scatter(scores_pace[:, 0], scores_pace[:, 1], label="PACE")
plt.scatter(scores_innpro[:, 0], scores_innpro[:, 1], label="InnPro")
plt.legend()
plt.show()


###############################################################################
# Comparison of the methods
# -------------------------
#
# Finally, we compare the methods by reconstructing the curves using the first :math:`K = 5` principal components. We plot a sample of curves and their reconstruction.

data_recons_numint = ufpca_cov.inverse_transform(scores_numint)
data_recons_pace = ufpca_cov.inverse_transform(scores_pace)
data_recons_innpro = ufpca_innpro.inverse_transform(scores_innpro)

colors_numint = np.array([[0.9, 0, 0, 1]])
colors_pace = np.array([[0, 0.9, 0, 1]])
colors_innpro = np.array([[0.9, 0, 0.9, 1]])

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 16))
for idx_plot, idx in enumerate(np.random.choice(n_obs, 10)):
    temp_ax = axes.flatten()[idx_plot]
    temp_ax = plot(kl.data[idx], ax=temp_ax, label="True")
    plot(
        data_recons_numint[idx],
        colors=colors_numint,
        ax=temp_ax,
        label="Reconstruction NumInt",
    )
    plot(
        data_recons_pace[idx],
        colors=colors_pace,
        ax=temp_ax,
        label="Reconstruction PACE",
    )
    plot(
        data_recons_innpro[idx],
        colors=colors_innpro,
        ax=temp_ax,
        label="Reconstruction InnPro",
    )
    temp_ax.legend()
plt.show()
