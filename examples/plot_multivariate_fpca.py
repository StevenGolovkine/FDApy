"""
Multivariate Functional Principal Components Analysis
=====================================================

This notebook shows how to perform an multivariate functional principal
components analysis on an example dataset.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import pandas as pd

from FDApy.representation import MultivariateFunctionalData
from FDApy.preprocessing.dim_reduction.fpca import MFPCA
from FDApy.visualization.plot import plot
from FDApy.misc.loader import read_csv

###############################################################################
# Load the data as DenseFunctionalData.
precipitation = read_csv('./data/canadian_precipitation_monthly.csv',
                         index_col=0)
temperature = read_csv('./data/canadian_temperature_daily.csv', index_col=0)

# Create multivariate functional data for the Canadian weather data.
canadWeather = MultivariateFunctionalData([precipitation, temperature])

###############################################################################
# Perform a multivariate functional PCA and explore the results.

# Perform multivariate FPCA
mfpca = MFPCA(n_components=[0.99, 0.95])
mfpca.fit(canadWeather, method='NumInt')

# Plot the results of the FPCA (eigenfunctions)
fig, (ax1, ax2) = plt.subplots(1, 2)
_ = plot(mfpca.basis[0], ax=ax1)
_ = plot(mfpca.basis[1], ax=ax2)

###############################################################################
# Compute the scores of the dailyTemp data into the eigenfunctions basis using
# numerical integration.

# Compute the scores
canadWeather_proj = mfpca.transform()

# Plot the projection of the data onto the eigenfunctions
_ = pd.plotting.scatter_matrix(pd.DataFrame(canadWeather_proj), diagonal='kde')

###############################################################################
# Then, we can test if the reconstruction of the data is good.

# Test if the reconstruction is good.
canadWheather_reconst = mfpca.inverse_transform(canadWeather_proj)

# Plot the reconstructed curves
fig, (ax1, ax2) = plt.subplots(1, 2)
_ = plot(canadWheather_reconst[0], ax=ax1)
_ = plot(canadWheather_reconst[1], ax=ax2)
