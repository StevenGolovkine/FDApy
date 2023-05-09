"""
Multivariate Basis of two-dimensional data
==========================================

"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation.basis import MultivariateBasis
from FDApy.visualization.plot import plot_multivariate

# Parameters
n_components = 2
basis_name = 'fourier'
argvals = [
    np.linspace(0, 1, 11),
    np.linspace(0, 0.5, 11)
]
n_functions = 3
dimension = ['2D', '2D']
random_state = np.random.default_rng(42)

###############################################################################
# Using split
basis = MultivariateBasis(
    simulation_type='split',
    n_components=n_components,
    name=basis_name,
    n_functions=n_functions,
    dimension=dimension,
    argvals=argvals,
    norm=False,
    rchoice=random_state.choice
)

_ = plot_multivariate(basis)

###############################################################################
# Using weighted
basis = MultivariateBasis(
    simulation_type='weighted',
    n_components=n_components,
    name=['fourier', 'legendre'],
    n_functions=n_functions,
    dimension=dimension,
    argvals=argvals,
    norm=False,
    runif=random_state.uniform
)

_ = plot_multivariate(basis)
