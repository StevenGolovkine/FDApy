"""
Canadian weather analysis
=========================

This notebook shows how to deal with univariate and multivariate functional
data by analyzing the canadian weather dataset.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from FDApy.univariate_functional import UnivariateFunctionalData
from FDApy.multivariate_functional import MultivariateFunctionalData
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
# Print out an univariate functional and a multivariate functional data object.

# Print univariate functional data
print(dailyTemp)

# Print multivariate functional data
print(canadWeather)

###############################################################################
# We can plot the data.

# Plot the multivariate functional data
fig, ax = plot(canadWeather,
               main=['Daily temperature', 'Monthly precipitation'],
               xlab=['Day', 'Month'],
               ylab=['Temperature', 'Precipitation'])
plt.show()

###############################################################################
# The attributs of the univariate functional data classes can easily be
# accessed.

# Accessing the argvals of the object
print(monthlyPrec.argvals)

# Get the number of observations for the object
monthlyPrec.nObs()

# Retrieve the number of sampling points for the object
monthlyPrec.nObsPoint()

# Dimension of the domain of observations
monthlyPrec.dimension()

# Extract observations from an univariate functional data object
print(monthlyPrec[3:6])

###############################################################################
# In a same way, the attributs of the multivariate functional data classes
# can also be easily accessed.

# Number of sampling points for the object
canadWeather.nObsPoint()

# Extract functions from MultivariateFunctionalData
print(canadWeather[0])

###############################################################################
# Compute the mean function for an univariate functional data object.

# Mean function of the monthly precipitation
monthlyPrec.mean()

# Plot the mean function of the monthly precipation
fig, ax = plot(monthlyPrec.mean_,
               main='Mean monthly precipitation',
               xlab='Month',
               ylab='Precipitation (mm)')
plt.show()

###############################################################################
# Compute the covariance surface for an univariate functional data object.

# Covariance function of the monthly precipitation
monthlyPrec.covariance()

# Plot the covariance function of the monthly precipitation
fig, ax = plot(monthlyPrec.covariance_,
               main='Covariance monthly precipitation',
               xlab='Month',
               ylab='Month')
plt.show()

###############################################################################
# We can also compute a smoothed estimate of the mean function and the
# covariance surface.

# Smoothing covariance of the daily temperature
dailyTemp.covariance(smooth=True, method='GAM', bandwidth=20)

# Plot the smooth covariance function of the daily temperature
fig, ax = plot(dailyTemp.covariance_,
               main='Covariance daily temperature',
               xlab='Day',
               ylab='Day')

###############################################################################
# Instead of directly computing an estimation of the mean and covariance by
# smoothing, we can smooth all the curve in an individual way.

# Smooth the data
dailyTempSmooth = dailyTemp.smooth(t0=200, k0=17,
                                   points=dailyTemp.argvals[0],
                                   kernel='gaussian')

# Plot the smooth data
plot(dailyTempSmooth)
plt.show()
