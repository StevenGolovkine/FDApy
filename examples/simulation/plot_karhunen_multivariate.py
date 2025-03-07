"""
Simulation using multivariate Karhunen-Loève decomposition
==========================================================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation import DenseArgvals
from FDApy.simulation import KarhunenLoeve
from FDApy.visualization import plot_multivariate

###############################################################################
# Multivariate functional data consist of independent trajectories of a
# vector-valued stochastic process
# :math:`X = (X^{(1)}, \dots, X^{(P)})^\top, P \geq 1`. Each coordinate
# :math:`X^{(p)}: \mathcal{T}_p \rightarrow \mathbb{R}` is assumed to be a
# squared-integrable real-valued functions defined on :math:`\mathcal{T}_p`.
# The simulation of multivariate functional data
# :math:`X` is based on the truncated multivariate Karhunen-Loève
# representation of :math:`X`. For a particular realization :math:`i` of the
# process :math:`X`:
#
# .. math::
#       X_i(t) = \mu(t) + \sum_{k = 1}^{K} c_{i,k}\phi_k(t),
#       \quad t \in \mathcal{T}
#
# with a common mean function :math:`\mu(t)` and eigenfunctions
# :math:`\phi_k, k = 1, \cdots, K`. The scores :math:`c_{i, k}`
# are the projection of the curves :math:`X_i` onto the eigenfunctions
# :math:`\phi_k`. These scores are random variables with mean :math:`0`
# and variance :math:`\lambda_k`, which are the eigenvalues associated to each
# eigenfunctions and that decreases toward :math:`0` when :math:`k` goes to
# infinity. This representation is valid for domains of arbitrary dimension,
# such as images (:math:`\mathcal{T} = \mathbb{R}^2`).


# Set general parameters
rng = 42
n_obs = 10


# Parameters of the basis
name = ["fourier", "bsplines"]
n_functions = [5, 5]
argvals = [
    DenseArgvals({"input_dim_0": np.arange(0, 10.01, 0.01)}),
    DenseArgvals({"input_dim_0": np.arange(-0.5, 0.51, 0.01)}),
]

###############################################################################
# Simulation for one-dimensional curve
# ------------------------------------
#
# **First example**
# ---
# We simulate :math:`N = 10` curves of a 2-dimensional process. The first component of the process is defined on the one-dimensional observation grid :math:`\{0, 0.01, 0.02, \cdots, 10\}`, based on the first :math:`K = 5` Fourier basis functions on :math:`[0, 10]` and the variance of the scores random variables equal to :math:`1` (default). The second component of the process is defined on the one-dimensional observation grid :math:`\{-0.5, -0.49, -0.48, \cdots, 0.5\}`, based on the first :math:`K = 5` B-splines basis functions on :math:`[0, 10]` and the variance of the scores random variables equal to :math:`1` (default).
kl = KarhunenLoeve(
    basis_name=name, n_functions=n_functions, argvals=argvals, random_state=rng
)
kl.new(n_obs=n_obs)

_ = plot_multivariate(kl.data)

###############################################################################
# **Second example**
# --- 
# We simulate :math:`N = 10` curves of a 2-dimensional process. The first component of the process is defined on the one-dimensional observation grid :math:`\{0, 0.01, 0.02, \cdots, 10\}`, based on the first :math:`K = 5` Fourier basis functions on :math:`[0, 1]` and the decreasing of the variance of the scores is exponential. The second component of the process is defined on the one-dimensional observation grid :math:`\{-0.5, -0.49, -0.48, \cdots, 0.5\}`, based on the first :math:`K = 5` B-splines basis functions on :math:`[0, 1]` and the decreasing of the variance of the scores is exponential.
kl = KarhunenLoeve(
    basis_name=name, n_functions=n_functions, argvals=argvals, random_state=rng
)
kl.new(n_obs=n_obs, clusters_std="exponential")

_ = plot_multivariate(kl.data)

###############################################################################
# Simulation for two-dimensional curve (image)
# --------------------------------------------
#
# We simulation a 2-dimensional process where the first component is a surface and the second component is a curve. For the simulation on a two-dimensional domain, we construct an two-dimensional eigenbasis based on tensor products of univariate eigenbasis.

# Parameters of the basis
name = [("fourier", "fourier"), "bsplines"]
n_functions = [(5, 5), 25]
argvals = [
    DenseArgvals(
        {
            "input_dim_0": np.arange(0, 10.01, 0.01),
            "input_dim_1": np.arange(0, 10.01, 0.01),
        }
    ),
    DenseArgvals({"input_dim_0": np.arange(-0.5, 0.51, 0.01)}),
]

###############################################################################
# **First example**
# ---
# We simulate :math:`N = 1` curves of a 2-dimensional process. The first component of the process is defined on the two-dimensional observation grid :math:`\{0, 0.01, 0.02, \cdots, 10\} \times \{0, 0.01, 0.02, \cdots, 10\}`, based on the tensor product of the first :math:`K = 25` Fourier basis functions on :math:`[0, 1]` and the variance of the scores random variables equal to :math:`1` (default). The second component of the process is defined on the one-dimensional observation grid :math:`\{-0.5, -0.49, -0.48, \cdots, 0.5\}`, based on the first :math:`K = 25` B-splines basis functions on :math:`[0, 1]` and the variance of the scores random variables equal to :math:`1` (default).
kl = KarhunenLoeve(
    basis_name=name, n_functions=n_functions, argvals=argvals, random_state=rng
)
kl.new(n_obs=1)

_ = plot_multivariate(kl.data)

###############################################################################
# **Second example**
# ---
# We simulate :math:`N = 1` curves of a 2-dimensional process. The first component of the process is defined on the two-dimensional observation grid :math:`\{0, 0.01, 0.02, \cdots, 10\} \times \{0, 0.01, 0.02, \cdots, 10\}`, based on the tensor product of the first :math:`K = 25` Fourier basis functions on :math:`[0, 1]` and the decreasing of the variance of the scores is linear. The second component of the process is defined on the one-dimensional observation grid :math:`\{-0.5, -0.49, -0.48, \cdots, 0.5\}` , based on the first :math:`K = 25` B-splines basis functions on :math:`[0, 1]` and the decreasing of the variance of the scores is linear.
kl = KarhunenLoeve(
    basis_name=name, n_functions=n_functions, argvals=argvals, random_state=rng
)
kl.new(n_obs=1, clusters_std="linear")

_ = plot_multivariate(kl.data)
