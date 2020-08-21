"""
Univariate Functional Principal Components Analysis
===================================================

This notebook shows how to perform an univariate functional principal
components analysis on an example dataset.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

import pandas as pd

from FDApy.preprocessing.dim_reduction.fpca import UFPCA
from FDApy.visualization.plot import plot
from FDApy.misc.loader import read_csv


###############################################################################
# Load the data into Pandas dataframe
temperature = read_csv('./data/canadian_temperature_daily.csv', index_col=0)

###############################################################################
# Perform a univariate functional PCA and explore the results.

# Perform a univariate FPCA on dailyTemp.
fpca = UFPCA(n_components=0.99)
fpca.fit(temperature)

# Plot the results of the FPCA (eigenfunctions)
_ = plot(fpca.eigenfunctions)

###############################################################################
# Compute the scores of the dailyTemp data into the eigenfunctions basis using
# numerical integration.

# Compute scores
temperature_proj = fpca.transform(temperature, method='NumInt')

# Plot the projection of the data onto the eigenfunctions
_ = pd.plotting.scatter_matrix(pd.DataFrame(temperature_proj), diagonal='kde')

###############################################################################
# Then, we can test if the reconstruction of the data is good.

# Test if the reconstruction is good.
temperature_reconst = fpca.inverse_transform(temperature_proj)

# Plot the reconstructed curves
_ = plot(temperature_reconst)
