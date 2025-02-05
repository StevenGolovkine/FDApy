"""
Smoothing of 2D data using P-Splines
====================================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.preprocessing.smoothing.psplines import PSplines

###############################################################################
# The package includes a class to perform P-Splines smoothing. The class :class:`~FDApy.preprocessing.PSplines` allows to fit a P-Splines regression to a functional data object. P-Splines regression is a non-parametric method that fits a spline to the data. The spline is defined by a basis of B-Splines. The B-Splines basis is defined by a set of knots. The P-Splines regression is a penalized regression that adds a discrete constraint to the fit. The influence of the penalty is controlled by the parameter `penalty`.
#

###############################################################################
# We will show how to use the class :class:`~FDApy.preprocessing.PSplines` to smooth a two-dimensional dataset. We will simulate a dataset from a function and add some noise. The goal is to recover the function by fitting a P-Splines regression. The :class:`~FDApy.preprocessing.PSplines` class requires the specification of the number of segments and the degree of the B-Splines basis. The number of segments is used to define the number of knots. The degree of the B-Splines basis is used to define the order of the B-Splines basis. The smoothing of 2-dimensional data is similar to the smoothing of 1-dimensional data. The only difference is that the data is now a list with two entries containing the sampling points for each dimension. Note that it is theoretically possible to smooth data with more than two dimensions, however, it may not be feasible due to computational constraints.

# Set general parameters
rng = 42
runif = np.random.default_rng(rng).uniform
n_points = 21
n_points_new = 51

# Simulate data
x = np.sort(runif(-1, 1, n_points))
y = np.sort(runif(-1, 1, n_points))
X, Y = np.meshgrid(x, y)
Z = (
    -1 * np.sin(X)
    + 0.5 * np.cos(Y)
    + 0.2 * np.random.normal(loc=0, scale=1, size=X.shape)
)

argvals = [x, y]
Z_vec = Z.ravel()

x_new = np.linspace(-1, 1, n_points_new)
y_new = np.linspace(-1, 1, n_points_new)
X_new, Y_new = np.meshgrid(x_new, y_new)
argvals_new = [x_new, y_new]

###############################################################################
# Here, we are interested in the influence of the degree of the B-Splines basis on the P-Splines regression. We will fit a P-Splines regression with degree :math:`0`, :math:`1` and :math:`2`. The number of segments is set to :math:`20` and the penalty is set to :math:`1`. We remark that the P-Splines regression with degree :math:`0` is not a good fit to data, while the P-Splines regression with degree :math:`1` or :math:`2` are roughly similar.
#

# Fit P-Splines regression with degree 0
ps = PSplines(n_segments=20, degree=0)
ps.fit(Z, argvals, penalty=(1, 1))
y_pred_0 = ps.predict(argvals_new)

# Fit P-Splines regression with degree 1
ps = PSplines(n_segments=20, degree=1)
ps.fit(Z, argvals, penalty=(1, 1))
y_pred_1 = ps.predict(argvals_new)

# Fit P-Splines regression with degree 2
ps = PSplines(n_segments=20, degree=2)
ps.fit(Z, argvals, penalty=(1, 1))
y_pred_2 = ps.predict(argvals_new)

# Plot results
fig = plt.figure(figsize=(10, 10))
# True
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.set_title("True")
ax1.scatter(X, Y, Z, c="grey", alpha=0.2)
ax1.scatter(X_new, Y_new, -1 * np.sin(X_new) + 0.5 * np.cos(Y_new), c="k")
ax1.set_xlim((-2, 2))
ax1.set_ylim((-2, 2))
ax1.set_zlim((-2, 2))
# Degree 0
ax2 = fig.add_subplot(2, 2, 2, projection="3d")
ax2.set_title("Degree 0")
ax2.scatter(X, Y, Z, c="grey", alpha=0.2)
ax2.scatter(X_new, Y_new, y_pred_0, c="r")
ax2.set_xlim((-2, 2))
ax2.set_ylim((-2, 2))
ax2.set_zlim((-2, 2))
# Degree 1
ax3 = fig.add_subplot(2, 2, 3, projection="3d")
ax3.set_title("Degree 1")
ax3.scatter(X, Y, Z, c="grey", alpha=0.2)
ax3.scatter(X_new, Y_new, y_pred_1, c="g")
ax3.set_xlim((-2, 2))
ax3.set_ylim((-2, 2))
ax3.set_zlim((-2, 2))
# Degree 2
ax4 = fig.add_subplot(2, 2, 4, projection="3d")
ax4.set_title("Degree 2")
ax4.scatter(X, Y, Z, c="grey", alpha=0.2)
ax4.scatter(X_new, Y_new, y_pred_2, c="y")
ax4.set_xlim((-2, 2))
ax4.set_ylim((-2, 2))
ax4.set_zlim((-2, 2))
plt.show()

###############################################################################
# Here, we are interested in the influence of the penalty on the P-Splines regression. We will fit a P-Splines regression with penalty :math:`(10, 10)`, :math:`(1, 1)` and :math:`(0.1, 0.1)`. Note that for multidimensional P-Spline smoothing, it is possible to have different penalty parameters for the different dimensions. The number of segments is set to :math:`20` and the degree is set to :math:`2`.

# Fit P-Splines regression with penalty=10
ps = PSplines(n_segments=20, degree=2)
ps.fit(Z, argvals, penalty=(10, 10))
y_pred_0 = ps.predict(argvals_new)

# Fit P-Splines regression with penalty=1
ps = PSplines(n_segments=20, degree=2)
ps.fit(Z, argvals, penalty=(1, 1))
y_pred_1 = ps.predict(argvals_new)

# Fit P-Splines regression with penalty=0.1
ps = PSplines(n_segments=20, degree=2)
ps.fit(Z, argvals, penalty=(0.1, 0.1))
y_pred_2 = ps.predict(argvals_new)

# Plot results
fig = plt.figure(figsize=(10, 10))
# True
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.set_title("True")
ax1.scatter(X, Y, Z, c="grey", alpha=0.2)
ax1.scatter(X_new, Y_new, -1 * np.sin(X_new) + 0.5 * np.cos(Y_new), c="k")
ax1.set_xlim((-2, 2))
ax1.set_ylim((-2, 2))
ax1.set_zlim((-2, 2))
# Degree 0
ax2 = fig.add_subplot(2, 2, 2, projection="3d")
ax2.set_title("$\lambda = 10$")
ax2.scatter(X, Y, Z, c="grey", alpha=0.2)
ax2.scatter(X_new, Y_new, y_pred_0, c="r")
ax2.set_xlim((-2, 2))
ax2.set_ylim((-2, 2))
ax2.set_zlim((-2, 2))
# Degree 1
ax3 = fig.add_subplot(2, 2, 3, projection="3d")
ax3.set_title("$\lambda = 1$")
ax3.scatter(X, Y, Z, c="grey", alpha=0.2)
ax3.scatter(X_new, Y_new, y_pred_1, c="g")
ax3.set_xlim((-2, 2))
ax3.set_ylim((-2, 2))
ax3.set_zlim((-2, 2))
# Degree 2
ax4 = fig.add_subplot(2, 2, 4, projection="3d")
ax4.set_title("$\lambda = 0.1$")
ax4.scatter(X, Y, Z, c="grey", alpha=0.2)
ax4.scatter(X_new, Y_new, y_pred_2, c="y")
ax4.set_xlim((-2, 2))
ax4.set_ylim((-2, 2))
ax4.set_zlim((-2, 2))
plt.show()
