"""
Simulation of Brownian motion
=============================

Examples of simulation of Brownian motion.
"""

###############################################################################
# A Brownian motion is a real valued continuous-time random process.
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.simulation import Brownian
from FDApy.visualization import plot

# Set general parameters
rng = 42
n_obs = 10
argvals = np.arange(0, 1.01, 0.01)

# Set Brownian parameters
init_point = 1.0
mu = 1.0
sigma = 0.5
hurst = 0.8

###############################################################################
# Standard Brownian motion
# ------------------------
#
# A standard Brownian motion is a stochastic process define as
# :math:`\{X_t\}_{t \geq 0}`. The process has the following properties:
#
# * :math:`\{X_t\}_{t \geq 0}` is a Gaussian process.
#
# * For :math:`s, t \geq 0`, :math:`\mathbb{E}(X_t) = 0` and :math:`\mathbb{E}(X_sX_t) = \min(s, t)`.
#
# * The function :math:`t \rightarrow X_t` is continuous with probablity :math:`1`.
#
br = Brownian(name="standard", random_state=rng)
br.new(n_obs=n_obs, argvals=argvals, init_point=init_point)

_ = plot(br.data)


###############################################################################
# Geometric Brownian motion
# -------------------------
#
# A geometric Brownian motion is a stochastic process :math:`\{X_t\}_{t \geq 0}`
# in which the logarithm of the randomly varying quantity is a Brownian motion
# with drift.
#
# The process :math:`\{X_t\}_{t \geq 0}` satisfies the following stochastic
# differential equation:
#
# .. math::
#       dX_t = \mu X_t dt + \sigma X_t dW_t
#
# where :math:`\{W_t\}_{t \geq 0}` is a Brownian motion, :math:`\mu` is the
# percentage drift and :math:`\sigma` is the percentage volatility.
#
br = Brownian(name="geometric", random_state=rng)
br.new(n_obs=n_obs, argvals=argvals, init_point=init_point, mu=mu, sigma=sigma)

_ = plot(br.data)


###############################################################################
# Fractional Brownian motion
# --------------------------
#
# A fractional Brownian motion is a stochastic process
# :math:`\{X_t\}_{t \geq 0}` that generalize Brownian motion. Let
# :math:`H \in (0, 1)` be the Hurst parameter. The process has the following
# properties:
#
# * :math:`\{X_t\}_{t \geq 0}` is a Gaussian process.
#
# * For :math:`s, t \geq 0`, :math:`\mathbb{E}(X_t) = 0` and :math:`\mathbb{E}(X_sX_t) = \frac{1}{2}\left(|s|^{2H} + |t|^{2H} - |s - t|^{2H}\right)`.
#
# * The function :math:`t \rightarrow X_t` is continuous with probablity :math:`1`.
#
# The value of :math:`H` defines the process. If :math:`H = 1/2`, :math:`\{X_t\}
# _{t \geq 0}` is a Brownian motion. If :math:`H > 1/2`, the increments of
# :math:`\{X_t\}_{t \geq 0}` are positively correlated. If :math:`H < 1/2`, the
# increments of :math:`\{X_t\}_{t \geq 0}` are negatively correlated.
br = Brownian(name="fractional", random_state=rng)
br.new(n_obs=n_obs, argvals=argvals, hurst=hurst)

_ = plot(br.data)
