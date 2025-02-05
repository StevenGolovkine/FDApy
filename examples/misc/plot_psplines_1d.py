"""
Smoothing of 1D data using P-Splines
====================================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.preprocessing import PSplines

###############################################################################
# The package includes a class to perform P-Splines smoothing. The class :class:`~FDApy.preprocessing.PSplines` allows to fit a P-Splines regression to a functional data object. P-Splines regression is a non-parametric method that fits a spline to the data. The spline is defined by a basis of B-Splines. The B-Splines basis is defined by a set of knots. The P-Splines regression is a penalized regression that adds a discrete constraint to the fit. The influence of the penalty is controlled by the parameter `penalty`.
#

###############################################################################
# We will show how to use the class :class:`~FDApy.preprocessing.PSplines` to smooth a one-dimensional dataset. We will simulate a dataset from a cosine function and add some noise. The goal is to recover the cosine function by fitting a P-Splines regression. The :class:`~FDApy.preprocessing.PSplines` class requires the specification of the number of segments and the degree of the B-Splines basis. The number of segments is used to define the number of knots. The degree of the B-Splines basis is used to define the order of the B-Splines basis. If the degree is set to :math:`0`, the B-Splines basis is a set of step functions. If the degree is set to :math:`1`, the B-Splines basis is a set of piecewise linear functions. If the degree is set to :math:`2`, the B-Splines basis is a set of piecewise quadratic functions. To fit the model, the method :meth:`~FDApy.preprocessing.PSplines.fit` requires the data and the penalty.
#

# Set general parameters
rng = 42
rnorm = np.random.default_rng(rng).standard_normal
n_points = 101

# Simulate data
x = np.sort(rnorm(n_points))
y = np.cos(x) + 0.2 * rnorm(n_points)
x_new = np.linspace(-2, 2, 51)

###############################################################################
# Here, we are interested in the influence of the degree of the B-Splines basis on the P-Splines regression. We will fit a P-Splines regression with degree :math:`0`, :math:`1` and :math:`2`. The number of segments is set to :math:`20` and the penalty is set to :math:`5`. We remark that the P-Splines regression with degree :math:`0` is not a good fit to data, while the P-Splines regression with degree :math:`1` or :math:`2` are roughly similar and recover the cosine function.
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

# Plot results
plt.scatter(x, y, c="grey", alpha=0.2)
plt.plot(np.sort(x), np.cos(np.sort(x)), c="k", label="True")
plt.plot(x_new, y_pred_0, c="r", label="Degree 0")
plt.plot(x_new, y_pred_1, c="g", label="Degree 1")
plt.plot(x_new, y_pred_2, c="y", label="Degree 2")
plt.legend()
plt.show()

###############################################################################
# Here, we are interested in the influence of the penalty on the P-Splines regression. We will fit a P-Splines regression with penalty :math:`10`, :math:`1` and :math:`0.1`. The number of segments is set to :math:`20` and the degree is set to :math:`3`. The better fit is obtained with the P-Splines regression with penalty :math:`10`.
#

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

# Plot results
plt.scatter(x, y, c="grey", alpha=0.2)
plt.plot(np.sort(x), np.cos(np.sort(x)), c="k", label="True")
plt.plot(x_new, y_pred_0, c="r", label="$\lambda = 10$")
plt.plot(x_new, y_pred_1, c="g", label="$\lambda = 1$")
plt.plot(x_new, y_pred_2, c="y", label="$\lambda = 0.1$")
plt.legend()
plt.show()
