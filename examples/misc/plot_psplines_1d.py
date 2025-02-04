"""
Smoothing of 1D data using P-Splines
====================================

Examples of smoothing of one-dimensional data using P-Splines.
"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.preprocessing.smoothing.psplines import PSplines

# Set general parameters
rng = 42
rnorm = np.random.default_rng(rng).standard_normal
n_points = 101

# Simulate data
x = np.sort(rnorm(n_points))
y = np.cos(x) + 0.2 * rnorm(n_points)
x_new = np.linspace(-2, 2, 51)

###############################################################################
# Assess the influence of the degree of the B-Splines basis.
#

# Fit P-Splines regression with degree 0
ps = PSplines(n_segments=20, degree=0)
ps.fit(y, x, penalty=5)
y_pred_0 = ps.predict(x_new)

# Fit P-Splines regression with degree 1
ps = PSplines(n_segments=20, degree=1)
ps.fit(y, x, penalty=5)
y_pred_1 = ps.predict(x_new)

# Fit P-Splines regression with degree 2
ps = PSplines(n_segments=20, degree=2)
ps.fit(y, x, penalty=5)
y_pred_2 = ps.predict(x_new)

###############################################################################
plt.scatter(x, y, c="grey", alpha=0.2)
plt.plot(np.sort(x), np.cos(np.sort(x)), c="k", label="True")
plt.plot(x_new, y_pred_0, c="r", label="Degree 0")
plt.plot(x_new, y_pred_1, c="g", label="Degree 1")
plt.plot(x_new, y_pred_2, c="y", label="Degree 2")
plt.legend()
plt.show()

###############################################################################
# Assess the influence of the penalty :math:`\lambda`.

# Fit P-Splines regression with penalty=10
ps = PSplines(n_segments=20, degree=3)
ps.fit(y, x, penalty=10)
y_pred_0 = ps.predict(x_new)

# Fit P-Splines regression with penalty=1
ps = PSplines(n_segments=20, degree=3)
ps.fit(y, x, penalty=1)
y_pred_1 = ps.predict(x_new)

# Fit P-Splines regression with penalty=0.1
ps = PSplines(n_segments=20, degree=3)
ps.fit(y, x, penalty=0.1)
y_pred_2 = ps.predict(x_new)

###############################################################################
plt.scatter(x, y, c="grey", alpha=0.2)
plt.plot(np.sort(x), np.cos(np.sort(x)), c="k", label="True")
plt.plot(x_new, y_pred_0, c="r", label="$\lambda = 10$")
plt.plot(x_new, y_pred_1, c="g", label="$\lambda = 1$")
plt.plot(x_new, y_pred_2, c="y", label="$\lambda = 0.1$")
plt.legend()
plt.show()
