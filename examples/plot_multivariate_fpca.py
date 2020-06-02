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
import numpy as np
import pandas as pd

from FDApy.univariate_functional import UnivariateFunctionalData
from FDApy.multivariate_functional import MultivariateFunctionalData
from FDApy.fpca import MFPCA
from FDApy.plot import plot

###############################################################################
# Load the data into Pandas dataframe
precipitation = pd.read_csv('./data/canadian_precipitation_monthly.csv',
                            index_col=0)
temperature = pd.read_csv('./data/canadian_temperature_daily.csv',
                          index_col=0)

###############################################################################
# Create univariate functional data for the precipitation and temperature
# dataset. Then, we will combine them to form a multivariate functional
# dataset.

# Create univariate functional data for the precipitation data
argvals = pd.factorize(precipitation.columns)[0]
values = np.array(precipitation)
monthlyPrec = UnivariateFunctionalData(argvals, values)

# Create univariate functional data for the daily temperature data.
argvals = pd.factorize(temperature.columns)[0]
values = np.array(temperature) / 4
dailyTemp = UnivariateFunctionalData(argvals, values)

# Create multivariate functional data for the Canadian weather data.
canadWeather = MultivariateFunctionalData([dailyTemp, monthlyPrec])

###############################################################################
print(monthlyPrec.argvals)


###############################################################################
# Perform a multivariate functional PCA and explore the results.

# Perform multivariate FPCA
mfpca = MFPCA(n_components=[0.99, 0.95], method='NumInt')
mfpca.fit(canadWeather)

# Plot the results of the FPCA (eigenfunctions)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(mfpca.basis_[0])
plt.title('Eigenfunctions for dailyTemp')
plt.subplot(1, 2, 2)
plt.plot(mfpca.basis_[1])
plt.title('Eigenfunctions for monthlyPrec')
plt.tight_layout()
plt.show()

###############################################################################
# Compute the scores of the dailyTemp data into the eigenfunctions basis using
# numerical integration.

# Compute the scores
canadWeather_proj = mfpca.transform(canadWeather)

# Plot the projection of the data onto the eigenfunctions
pd.plotting.scatter_matrix(pd.DataFrame(canadWeather_proj), diagonal='kde')
plt.show()

###############################################################################
# Then, we can test if the reconstruction of the data is good.

# Test if the reconstruction is good.
canadWheather_reconst = mfpca.inverse_transform(canadWeather_proj)

# Plot the reconstructed curves
fig, ax = plot(canadWheather_reconst,
               main=['Daily temperature', 'Monthly precipitation'],
               xlab=['Day', 'Month'],
               ylab=['Temperature', 'Precipitation'])
plt.show()
