"""
Smoothing of 2D data using local polynomial regression
======================================================

Examples of smoothing of two-dimensional data using local polynomial
regression.
"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from FDApy.preprocessing.smoothing.local_polynomial import LocalPolynomial

# Set general parameters
rng = 42
rnorm = np.random.default_rng(rng).standard_normal
n_points = 101

# Simulate data
x = rnorm((n_points, 2))
y = -1 * np.sin(x[:, 0]) + 0.5 * np.cos(x[:, 1]) + 0.2 * rnorm(n_points)
x_new = np.mgrid[-1:1:.1, -1:1:.1]
x_new = np.column_stack((x_new[0].ravel(), x_new[1].ravel()))

###############################################################################
# Assess the influence of the degree of the polynomials.

# Fit local polynomial regression with degree 0
lp = LocalPolynomial(
    kernel_name='epanechnikov', bandwidth=0.5, degree=0
)
y_pred_0 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with degree 1
lp = LocalPolynomial(
    kernel_name='epanechnikov', bandwidth=0.5, degree=1
)
y_pred_1 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with degree 2
lp = LocalPolynomial(
    kernel_name='epanechnikov', bandwidth=0.5, degree=2
)
y_pred_2 = lp.predict(y=y, x=x, x_new=x_new)

###############################################################################
fig = plt.figure(figsize=(10, 10))
# True
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_title("True")
ax1.scatter(x[:, 0], x[:, 1], y, c='grey', alpha=0.2)
ax1.scatter(x_new[:, 0], x_new[:, 1], -1 * np.sin(x_new[:, 0]) + 0.5 * np.cos(x_new[:, 1]), c='k')
ax1.set_xlim((-2, 2))
ax1.set_ylim((-2, 2))
ax1.set_zlim((-2, 2))
# Degree 0
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.set_title("Degree 0")
ax2.scatter(x[:, 0], x[:, 1], y, c='grey', alpha=0.2)
ax2.scatter(x_new[:, 0], x_new[:, 1], y_pred_0, c='r')
ax2.set_xlim((-2, 2))
ax2.set_ylim((-2, 2))
ax2.set_zlim((-2, 2))
# Degree 1
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.set_title("Degree 1")
ax3.scatter(x[:, 0], x[:, 1], y, c='grey', alpha=0.2)
ax3.scatter(x_new[:, 0], x_new[:, 1], y_pred_1, c='g')
ax3.set_xlim((-2, 2))
ax3.set_ylim((-2, 2))
ax3.set_zlim((-2, 2))
# Degree 2
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.set_title("Degree 2")
ax4.scatter(x[:, 0], x[:, 1], y, c='grey', alpha=0.2)
ax4.scatter(x_new[:, 0], x_new[:, 1], y_pred_2, c='y')
ax4.set_xlim((-2, 2))
ax4.set_ylim((-2, 2))
ax4.set_zlim((-2, 2))
plt.show()


###############################################################################
# Assess the influence of the bandwith :math:`\lambda`.

# Fit local polynomial regression with bandwidth 0.2
lp = LocalPolynomial(
    kernel_name='epanechnikov', bandwidth=0.2, degree=1
)
y_pred_0 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with bandwidth 0.5
lp = LocalPolynomial(
    kernel_name='epanechnikov', bandwidth=0.5, degree=1
)
y_pred_1 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with bandwidth 0.8
lp = LocalPolynomial(
    kernel_name='epanechnikov', bandwidth=0.8, degree=1
)
y_pred_2 = lp.predict(y=y, x=x, x_new=x_new)


###############################################################################
fig = plt.figure(figsize=(10, 10))
# True
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_title("True")
ax1.scatter(x[:, 0], x[:, 1], y, c='grey', alpha=0.2)
ax1.scatter(x_new[:, 0], x_new[:, 1], -1 * np.sin(x_new[:, 0]) + 0.5 * np.cos(x_new[:, 1]), c='k')
ax1.set_xlim((-2, 2))
ax1.set_ylim((-2, 2))
ax1.set_zlim((-2, 2))
# Bandwidth = 0.2
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.set_title("$\lambda = 0.2$")
ax2.scatter(x[:, 0], x[:, 1], y, c='grey', alpha=0.2)
ax2.scatter(x_new[:, 0], x_new[:, 1], y_pred_0, c='r')
ax2.set_xlim((-2, 2))
ax2.set_ylim((-2, 2))
ax2.set_zlim((-2, 2))
# Bandwidth = 0.5
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.set_title("$\lambda = 0.5$")
ax3.scatter(x[:, 0], x[:, 1], y, c='grey', alpha=0.2)
ax3.scatter(x_new[:, 0], x_new[:, 1], y_pred_1, c='g')
ax3.set_xlim((-2, 2))
ax3.set_ylim((-2, 2))
ax3.set_zlim((-2, 2))
# Bandwidth = 0.8
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.set_title("$\lambda = 0.8$")
ax4.scatter(x[:, 0], x[:, 1], y, c='grey', alpha=0.2)
ax4.scatter(x_new[:, 0], x_new[:, 1], y_pred_2, c='y')
ax4.set_xlim((-2, 2))
ax4.set_ylim((-2, 2))
ax4.set_zlim((-2, 2))
plt.show()