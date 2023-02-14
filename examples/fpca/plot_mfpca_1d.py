"""
MFPCA of 1-dimensional data
--------------------------

Example of multivariate functional principal components analysis of
1-dimensional data.
"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np
import matplotlib.pyplot as plt

from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.representation.functional_data import MultivariateFunctionalData
from FDApy.preprocessing.dim_reduction.fpca import MFPCA
from FDApy.visualization.plot import plot, plot_multivariate



# With simulated data
kl = KarhunenLoeve(basis_name='bsplines', n_functions=5)
kl.new(n_obs=50)

kl_2 = KarhunenLoeve(basis_name='fourier', n_functions=5)
kl_2.new(n_obs=50)

data = MultivariateFunctionalData([kl.data, kl_2.data])


plot_multivariate(data)
plt.show()


plot_multivariate(data.mean())
plt.show()


# Perform multivariate FPCA
mfpca_cov = MFPCA(n_components=[0.95, 0.95], method='covariance')
mfpca_cov.fit(data)

mfpca_cov.eigenvalues

plot_multivariate(mfpca_cov.eigenfunctions)
plt.show()


# Perform multivariate FPCA
mfpca_inn = MFPCA(n_components=0.99, method='inner-product')
mfpca_inn.fit(data)


mfpca_inn.eigenvalues


plot_multivariate(mfpca_inn.eigenfunctions)
plt.show()


# ## Estimate the scores
scores = mfpca_cov.transform(data, method='NumInt')


scores_ = mfpca_inn.transform(data, method='InnPro')



plt.scatter(scores[:, 0], scores[:, 1])
plt.scatter(scores_[:, 0], scores_[:, 1])
plt.show()


# ## Transform the data back to the original space


data_f_cov = mfpca_cov.inverse_transform(scores)



data_f_inn = mfpca_inn.inverse_transform(scores_)


idx = 5
colors = np.array([[0.5, 0, 0, 1]])
colors_2 = np.array([[0, 0.5, 0, 1]])

ax = plot_multivariate(MultivariateFunctionalData([data[0][idx], data[1][idx]]))
ax = plot_multivariate(MultivariateFunctionalData([data_f_cov[0][idx], data_f_cov[1][idx]]), colors=colors, ax=ax)
ax = plot_multivariate(MultivariateFunctionalData([data_f_inn[0][idx], data_f_inn[1][idx]]), colors=colors_2, ax=ax)
