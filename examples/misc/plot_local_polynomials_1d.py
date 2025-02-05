"""
Smoothing of 1D data using local polynomial regression
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
# We will show how to use the class :class:`~FDApy.preprocessing.LocalPolynomial` to smooth a one-dimensional dataset. We will simulate a dataset from a cosine function and add some noise. The goal is to recover the cosine function by fitting a local polynomial regression. The :class:`~FDApy.preprocessing.LocalPolynomial` class requires the specification of the kernel, the bandwidth and the degree of the polynomial. The kernel is used to define the weights of the local regression. Four kernels are implemented: `gaussian`, `epanechnikov`, `tricube` and `bisquare`. The bandwidth is used to define the size of the local neighborhood. The degree of the polynomial is used to define the order of the polynomial to fit. If the degree is set to :math:`0`, the local regression is a local constant regression. If the degree is set to :math:`1`, the local regression is a local linear regression. If the degree is set to :math:`2`, the local regression is a local quadratic regression.

# Set general parameters
rng = 42
rnorm = np.random.default_rng(rng).standard_normal
n_points = 101

# Simulate data
x = rnorm(n_points)
y = np.cos(x) + 0.2 * rnorm(n_points)
x_new = np.linspace(-1, 1, 51)

###############################################################################
# Here, we are interested in the influence of the degree of the polynomial on the local polynomial regression. We will fit a local polynomial regression with degree :math:`0`, :math:`1` and :math:`2`. The bandwidth is set to :math:`0.5` and the kernel is set to `epanechnikov`. We remark that the local polynomial regression with degree :math:`2` overfits the data, while the local polynomial regression with degree :math:`0` or :math:`1` roughly recover the cosine function.

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
plt.scatter(x, y, c="grey", alpha=0.2)
plt.plot(np.sort(x), np.cos(np.sort(x)), c="k", label="True")
plt.plot(x_new, y_pred_0, c="r", label="Degree 0")
plt.plot(x_new, y_pred_1, c="g", label="Degree 1")
plt.plot(x_new, y_pred_2, c="y", label="Degree 2")
plt.legend()
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
plt.scatter(x, y, c="grey", alpha=0.2)
plt.plot(np.sort(x), np.cos(np.sort(x)), c="k", label="True")
plt.plot(x_new, y_pred_0, c="r", label="$\lambda = 0.2$")
plt.plot(x_new, y_pred_1, c="g", label="$\lambda = 0.5$")
plt.plot(x_new, y_pred_2, c="y", label="$\lambda = 0.8$")
plt.legend()
plt.show()
