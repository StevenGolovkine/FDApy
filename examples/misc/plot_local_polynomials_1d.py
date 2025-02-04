"""
Smoothing of 1D data using local polynomial regression
======================================================

Examples of smoothing of one-dimensional data using local polynomial
regression.
"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.preprocessing import LocalPolynomial

# Set general parameters
rng = 42
rnorm = np.random.default_rng(rng).standard_normal
n_points = 101

# Simulate data
x = rnorm(n_points)
y = np.cos(x) + 0.2 * rnorm(n_points)
x_new = np.linspace(-1, 1, 51)

###############################################################################
# Assess the influence of the degree of the polynomials.

# Fit local polynomial regression with degree 0
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.5, degree=0)
y_pred_0 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with degree 1
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.5, degree=1)
y_pred_1 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with degree 2
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.5, degree=2)
y_pred_2 = lp.predict(y=y, x=x, x_new=x_new)


###############################################################################
plt.scatter(x, y, c="grey", alpha=0.2)
plt.plot(np.sort(x), np.cos(np.sort(x)), c="k", label="True")
plt.plot(x_new, y_pred_0, c="r", label="Degree 0")
plt.plot(x_new, y_pred_1, c="g", label="Degree 1")
plt.plot(x_new, y_pred_2, c="y", label="Degree 2")
plt.legend()
plt.show()


###############################################################################
# Assess the influence of the bandwith :math:`\lambda`.

# Fit local polynomial regression with bandwidth 0.2
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.2, degree=1)
y_pred_0 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with bandwidth 0.5
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.5, degree=1)
y_pred_1 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with bandwidth 0.8
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.8, degree=1)
y_pred_2 = lp.predict(y=y, x=x, x_new=x_new)


###############################################################################
plt.scatter(x, y, c="grey", alpha=0.2)
plt.plot(np.sort(x), np.cos(np.sort(x)), c="k", label="True")
plt.plot(x_new, y_pred_0, c="r", label="$\lambda = 0.2$")
plt.plot(x_new, y_pred_1, c="g", label="$\lambda = 0.5$")
plt.plot(x_new, y_pred_2, c="y", label="$\lambda = 0.8$")
plt.legend()
plt.show()
