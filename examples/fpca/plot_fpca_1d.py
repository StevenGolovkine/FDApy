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

from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import UFPCA
from FDApy.visualization.plot import plot


# With simulated data
kl = KarhunenLoeve(basis_name='fourier', n_functions=25)
kl.new(n_obs=50)
data = kl.data

_ = plot(data)


# Perform univariate FPCA
ufpca = UFPCA(n_components=0.99, method='covariance')
ufpca.fit(data)

# Plot the eigenfunctions
_ = plot(ufpca.eigenfunctions)

# Estimation of the scores
scores = ufpca.transform(data, method='NumInt')

# Plot of the scores
plt.scatter(scores[:, 0], scores[:, 1])

# Tranform the data back to its original space.
data_f = ufpca.inverse_transform(scores)

# Plot an example of the curve reconstruction
idx = 5
plot(data[idx])
plt.plot(data_f.argvals['input_dim_0'], data_f.values[idx], c='red')
plt.show()
