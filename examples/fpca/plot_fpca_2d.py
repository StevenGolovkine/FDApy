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

from FDApy.simulation.karhunen import KarhunenLoeve
from FDApy.preprocessing.dim_reduction.fpca import UFPCA
from FDApy.visualization.plot import plot


# With simulated data
kl = KarhunenLoeve(basis_name='bsplines', n_functions=5, dimension='2D')
kl.new(n_obs=50)
data = kl.data

_ = plot(data)


# Perform univariate FPCA
ufpca = UFPCA(n_components=0.95, method='inner-product')
ufpca.fit(data)

scores = ufpca.transform(data, method='InnPro')

plt.scatter(scores[:, 0], scores[:, 1])

data_f = ufpca.inverse_transform(scores)

idx = 1
_ = plot(data[idx])
_ = plot(data_f[idx])
