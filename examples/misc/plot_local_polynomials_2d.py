"""
Smoothing of 2D data using local polynomial regression
======================================================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.preprocessing import LocalPolynomial

###############################################################################
# The package includes a class to perform local polynomial regression. The class :class:`~FDApy.preprocessing.LocalPolynomial` allows to fit a local polynomial regression to a functional data object. Local polynomial regression is a non-parametric method that fits a polynomial to the data in a local neighborhood of each point.

###############################################################################
# We will show how to use the class :class:`~FDApy.preprocessing.LocalPolynomial` to smooth a two-dimensional dataset. We will simulate a dataset from a function and add some noise. The goal is to recover the function by fitting a local polynomial regression. The :class:`~FDApy.preprocessing.LocalPolynomial` class requires the specification of the kernel, the bandwidth and the degree of the polynomial. The kernel is used to define the weights of the local regression. Four kernels are implemented: `gaussian`, `epanechnikov`, `tricube` and `bisquare`. The bandwidth is used to define the size of the local neighborhood. The degree of the polynomial is used to define the order of the polynomial to fit. If the degree is set to :math:`0`, the local regression is a local constant regression. If the degree is set to :math:`1`, the local regression is a local linear regression. If the degree is set to :math:`2`, the local regression is a local quadratic regression. The smoothing of 2-dimensional data is similar to the smoothing of 1-dimensional data. The only difference is that the data is now a matrix with two columns. Note that it is theoretically possible to smooth data with more than two dimensions, however, it may not be feasible due to computational constraints.


# Set general parameters
rng = 42
rnorm = np.random.default_rng(rng).standard_normal
n_points = 101

# Simulate data
x = rnorm((n_points, 2))
y = -1 * np.sin(x[:, 0]) + 0.5 * np.cos(x[:, 1]) + 0.2 * rnorm(n_points)
x_new = np.mgrid[-1:1:0.1, -1:1:0.1]
x_new = np.column_stack((x_new[0].ravel(), x_new[1].ravel()))

###############################################################################
# Here, we are interested in the influence of the degree of the polynomial on the local polynomial regression. We will fit a local polynomial regression with degree :math:`0`, :math:`1` and :math:`2`. The bandwidth is set to :math:`0.5` and the kernel is set to `epanechnikov`. We remark that the local polynomial regression with degree :math:`2` overfits the data, while the local polynomial regression with degree :math:`0` or :math:`1` roughly recover the function.

# Fit local polynomial regression with degree 0
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.5, degree=0)
y_pred_0 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with degree 1
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.5, degree=1)
y_pred_1 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with degree 2
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.5, degree=2)
y_pred_2 = lp.predict(y=y, x=x, x_new=x_new)

# Plot results
fig = plt.figure(figsize=(10, 10))
# True
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.set_title("True")
ax1.scatter(x[:, 0], x[:, 1], y, c="grey", alpha=0.2)
ax1.scatter(
    x_new[:, 0],
    x_new[:, 1],
    -1 * np.sin(x_new[:, 0]) + 0.5 * np.cos(x_new[:, 1]),
    c="k",
)
ax1.set_xlim((-2, 2))
ax1.set_ylim((-2, 2))
ax1.set_zlim((-2, 2))
# Degree 0
ax2 = fig.add_subplot(2, 2, 2, projection="3d")
ax2.set_title("Degree 0")
ax2.scatter(x[:, 0], x[:, 1], y, c="grey", alpha=0.2)
ax2.scatter(x_new[:, 0], x_new[:, 1], y_pred_0, c="r")
ax2.set_xlim((-2, 2))
ax2.set_ylim((-2, 2))
ax2.set_zlim((-2, 2))
# Degree 1
ax3 = fig.add_subplot(2, 2, 3, projection="3d")
ax3.set_title("Degree 1")
ax3.scatter(x[:, 0], x[:, 1], y, c="grey", alpha=0.2)
ax3.scatter(x_new[:, 0], x_new[:, 1], y_pred_1, c="g")
ax3.set_xlim((-2, 2))
ax3.set_ylim((-2, 2))
ax3.set_zlim((-2, 2))
# Degree 2
ax4 = fig.add_subplot(2, 2, 4, projection="3d")
ax4.set_title("Degree 2")
ax4.scatter(x[:, 0], x[:, 1], y, c="grey", alpha=0.2)
ax4.scatter(x_new[:, 0], x_new[:, 1], y_pred_2, c="y")
ax4.set_xlim((-2, 2))
ax4.set_ylim((-2, 2))
ax4.set_zlim((-2, 2))
plt.show()


###############################################################################
# Here, we are interested in the influence of the bandwidth on the local polynomial regression. We will fit a local polynomial regression with bandwidth :math:`0.2`, :math:`0.5` and :math:`0.8`. The degree is set to :math:`1` and the kernel is set to `epanechnikov`. We remark that the local polynomial regression with bandwidth :math:`0.2` overfits the data. The better fit is obtained with the local polynomial regression with bandwidth :math:`0.8`.

# Fit local polynomial regression with bandwidth 0.2
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.2, degree=1)
y_pred_0 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with bandwidth 0.5
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.5, degree=1)
y_pred_1 = lp.predict(y=y, x=x, x_new=x_new)

# Fit local polynomial regression with bandwidth 0.8
lp = LocalPolynomial(kernel_name="epanechnikov", bandwidth=0.8, degree=1)
y_pred_2 = lp.predict(y=y, x=x, x_new=x_new)

# Plot results
fig = plt.figure(figsize=(10, 10))
# True
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.set_title("True")
ax1.scatter(x[:, 0], x[:, 1], y, c="grey", alpha=0.2)
ax1.scatter(
    x_new[:, 0],
    x_new[:, 1],
    -1 * np.sin(x_new[:, 0]) + 0.5 * np.cos(x_new[:, 1]),
    c="k",
)
ax1.set_xlim((-2, 2))
ax1.set_ylim((-2, 2))
ax1.set_zlim((-2, 2))
# Bandwidth = 0.2
ax2 = fig.add_subplot(2, 2, 2, projection="3d")
ax2.set_title("$\lambda = 0.2$")
ax2.scatter(x[:, 0], x[:, 1], y, c="grey", alpha=0.2)
ax2.scatter(x_new[:, 0], x_new[:, 1], y_pred_0, c="r")
ax2.set_xlim((-2, 2))
ax2.set_ylim((-2, 2))
ax2.set_zlim((-2, 2))
# Bandwidth = 0.5
ax3 = fig.add_subplot(2, 2, 3, projection="3d")
ax3.set_title("$\lambda = 0.5$")
ax3.scatter(x[:, 0], x[:, 1], y, c="grey", alpha=0.2)
ax3.scatter(x_new[:, 0], x_new[:, 1], y_pred_1, c="g")
ax3.set_xlim((-2, 2))
ax3.set_ylim((-2, 2))
ax3.set_zlim((-2, 2))
# Bandwidth = 0.8
ax4 = fig.add_subplot(2, 2, 4, projection="3d")
ax4.set_title("$\lambda = 0.8$")
ax4.scatter(x[:, 0], x[:, 1], y, c="grey", alpha=0.2)
ax4.scatter(x_new[:, 0], x_new[:, 1], y_pred_2, c="y")
ax4.set_xlim((-2, 2))
ax4.set_ylim((-2, 2))
ax4.set_zlim((-2, 2))
plt.show()
