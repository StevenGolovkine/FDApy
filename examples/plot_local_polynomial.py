"""
Local Polynomial Regression
===========================

This notebook shows how to perform a local polynomial regression on
one and two-dimensional data.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

from FDApy.preprocessing.smoothing.local_polynomial import LocalPolynomial


###############################################################################
# We generate some one-dimensional data to perform local polynomial smoothing.
#

X = np.random.normal(0, 1, 100)
Y = 2 * np.sin(X) + np.random.normal(0, 0.25, 100)
Y_true = 2 * np.sin(np.linspace(-2, 2, 200))

###############################################################################
# Fit local polynomials
#

# Fit local polynomials
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=2, degree=2)
lp.fit(X, Y)

# Plot the results
plt.scatter(X, Y, alpha=0.5, color='blue', label='Noisy')
plt.scatter(np.sort(X), lp.X_fit_, color='red', label='Estimated')
plt.plot(np.linspace(-2, 2, 200), Y_true, 'green', label='True')
plt.legend()

###############################################################################
# Estimate the curve on a regular grid.

# Estimation on a grid
y_pred = lp.predict(np.linspace(-2, 2, 500))

# Plot the results
plt.scatter(X, Y, alpha=0.5, color='blue', label='noisy')
plt.scatter(np.linspace(-2, 2, 500), y_pred, color='red', label='Prediction')
plt.plot(np.linspace(-2, 2, 200), Y_true, color='green', label='True')
plt.legend()

###############################################################################
# We will now do the same using two-dimensional data.
#

X = np.random.randn(2, 100)
Y = -1 * np.sin(X[0]) + 0.5 * np.cos(X[1]) + 0.2 * np.random.randn(100)
X0 = np.mgrid[-10:10:1, -10:10:1] / 10
X0 = np.vstack([X0[0].ravel(), X0[1].ravel()])

###############################################################################
# Fit local polynomials
#

# Fit local polynomials
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=2, degree=1)
lp.fit(X, Y)

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_ = ax.scatter(X[0], X[1], Y)
_ = ax.scatter(X[0], X[1], lp.X_fit_, color='red')

###############################################################################
# Estimate the curve on a regular surface.

# Estimation on a grid
y_pred = lp.predict(X0)

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_ = ax.scatter(X[0], X[1], Y)
_ = ax.scatter(X0[0], X0[1], y_pred, color='red')
