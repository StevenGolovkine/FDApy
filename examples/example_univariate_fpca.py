"""
Univariate Functional Principal Components Analysis
==========================

This notebook shows how to perform an univariate functional principal
components analysis on an example dataset.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from FDApy.univariate_functional import UnivariateFunctionalData
from FDApy.fpca import UFPCA
from FDApy.plot import plot


###############################################################################
# Load the data into Pandas dataframe
temperature = pd.read_csv('./data/canadian_temperature_daily.csv',
                          index_col=0)

# Create univariate functional data for the daily temperature data.
argvals = pd.factorize(temperature.columns)[0]
values = np.array(temperature) / 4
dailyTemp = UnivariateFunctionalData(argvals, values)

###############################################################################
# Perform a univariate functional PCA and explore the results.

# Perform a univariate FPCA on dailyTemp.
fpca = UFPCA(n_components=0.99)
fpca.fit(dailyTemp, method='GAM', kernel='gaussian', bandwidth=50)

# Plot the results of the FPCA (eigenfunctions)
for idx, eigenfunction in enumerate(fpca.eigenfunctions):
    plt.plot(eigenfunction, label=' '.join(['Eigenfunction', str(idx + 1)]))
plt.xlab('Days')
plt.title('Eigenfunctions')
plt.legend()
plt.show()

###############################################################################
# Compute the scores of the dailyTemp data into the eigenfunctions basis using
# numerical integration.

# Compute scores
dailyTemp_proj = fpca.transform(dailyTemp, method='NumInt')

# Plot the projection of the data onto the eigenfunctions
pd.plotting.scatter_matrix(pd.DataFrame(dailyTemp_proj), diagonal='kde')
plt.show()

###############################################################################
# Then, we can test if the reconstruction of the data is good.

# Test if the reconstruction is good.
dailyTemp_reconst = fpca.inverse_transform(dailyTemp_proj)

# Plot the reconstructed curves
fig, ax = plot(dailyTemp_reconst,
               main='Daily temperature',
               xlab='Day',
               ylab='Temperature')
plt.show()
