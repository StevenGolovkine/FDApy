"""
Canadian weather analysis
=========================

This notebook shows how to deal with univariate and multivariate functional
data by analyzing the canadian weather dataset.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

from FDApy import MultivariateFunctionalData
from FDApy.misc.loader import read_csv
from FDApy.plot import plot

###############################################################################
# Load the data as DenseFunctionalData.
precipitation = read_csv('./data/canadian_precipitation_monthly.csv',
                         index_col=0)
temperature = read_csv('./data/canadian_temperature_daily.csv', index_col=0)

# Create multivariate functional data for the Canadian weather data.
canadWeather = MultivariateFunctionalData([precipitation, temperature])

###############################################################################
# Print out an univariate functional data object.

# Print univariate functional data
print(temperature)

###############################################################################
# Print out a multivariate functional data object.

# Print multivariate functional data
print(canadWeather)

###############################################################################
# We can plot the data.

# Plot the multivariate functional data
fig, ax = plot(canadWeather,
               main=['Daily temperature', 'Monthly precipitation'],
               xlab=['Day', 'Month'],
               ylab=['Temperature', 'Precipitation'])

###############################################################################
# The attributs of the univariate functional data classes can easily be
# accessed.

###############################################################################
# The sampling points of the data can easily be accessed.

# Accessing the argvals of the object
print(precipitation.argvals)

###############################################################################
# The number of observations within the data are obtained using the function
# :func:`~FDApy.univariate_functional.UnivariateFunctional.n_obs`.

# Get the number of observations for the object
print(precipitation.n_obs)

###############################################################################
# The number of sampling points per observation is given by the function
# :func:`~FDApy.univariate_functional.UnivariateFunctional.n_points`.

# Retrieve the number of sampling points for the object
print(precipitation.n_points)

###############################################################################
# The dimension of the data is given by the function
# :func:`~FDApy.univariate_functional.UnivariateFunctional.n_dim`.

# Get the dimension of the domain of the observations
print(precipitation.n_dim)

###############################################################################
# The extraction of observations is also easily done.

# Extract observations from the object
print(precipitation[3:6])

###############################################################################
# In a same way, the attributs of the multivariate functional data classes
# can also be easily accessed.

# Number of observations for the object
canadWeather.n_obs

# Extract functions from MultivariateFunctionalData
print(canadWeather[0])

###############################################################################
# Compute the mean function for an univariate functional data object.

# Mean function of the monthly precipitation
precipitation_mean = precipitation.mean()

# Plot the mean function of the monthly precipation
fig, ax = plot(precipitation_mean,
               main='Mean monthly precipitation',
               xlab='Month',
               ylab='Precipitation (mm)')

###############################################################################
# Compute the covariance surface for an univariate functional data object.

# Covariance function of the monthly precipitation
precipitation_covariance = precipitation.covariance()

# Plot the covariance function of the monthly precipitation
fig, ax = plot(precipitation_covariance,
               main='Covariance monthly precipitation',
               xlab='Month',
               ylab='Month')

###############################################################################
# We can also compute a smoothed estimate of the mean function and the
# covariance surface.

# Smoothing covariance of the daily temperature
temperature.covariance(smooth=True, method='GAM', bandwidth=20)

# Plot the smooth covariance function of the daily temperature
fig, ax = plot(temperature.covariance_,
               main='Covariance daily temperature',
               xlab='Day',
               ylab='Day')

###############################################################################
# Instead of directly computing an estimation of the mean and covariance by
# smoothing, we can smooth all the curve in an individual way.

# Smooth the data
temperature_smooth = temperature.smooth(t0=200, k0=17,
                                        points=temperature.argvals[0],
                                        kernel='gaussian')

# Plot the smooth data
fig, ax = plot(temperature_smooth,
               main='Daily temperature',
               xlab='Day',
               ylab='Temperature')
