"""
Multivariate Basis
==================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import numpy as np

from FDApy.representation import MultivariateBasis
from FDApy.representation import DenseArgvals
from FDApy.visualization import plot_multivariate

###############################################################################
# Similarly to the multivariate functional, it is possible to define multivariate basis functions using the :class:`~FDApy.representation.MultivariateBasis` object. Multivariate basis functions are defined as a list of univariate basis functions and are represented with a :class:`~FDApy.representation.MultivariateBasis` object. The univariate basis functions can be of whatever dimension (curves, surfaces, ...). There is no restriction on the number of elements in the list but each univariate element must have the same number of functions.

##############################################################################
# First example
# -------------
# First, we will define a multivariate basis where the first component is the Fourier basis and the second component is the Legendre basis. The number of functions in the basis is set to 3 and the sampling points are defined as a :class:`~FDApy.representation.DenseArgvals` object with eleven points between :math:`0` and :math:`1` for the first component and with eleven points between :math:`0` and :math:`0.5` for the second component.

basis_name = ["fourier", "legendre"]
argvals = [
    DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)}),
    DenseArgvals({"input_dim_0": np.linspace(0, 0.5, 11)}),
]
n_functions = [3, 3]

basis = MultivariateBasis(name=basis_name, n_functions=n_functions, argvals=argvals)

_ = plot_multivariate(basis)


###############################################################################
# Second example
# --------------
# The second example is a multivariate basis where the first component is a multidimensional basis built from the tensor product of the Fourier basis and the second component is a multidimensional basis built from the tensor product of the Legendre basis. The number of functions in the multivariate basis is set to 3.

# Parameters
basis_name = [("fourier", "fourier"), ("legendre", "legendre")]
argvals = [
    DenseArgvals(
        {"input_dim_0": np.linspace(0, 1, 11), "input_dim_1": np.linspace(0, 0.5, 11)}
    ),
    DenseArgvals(
        {"input_dim_0": np.linspace(0, 0.5, 11), "input_dim_1": np.linspace(0, 1, 11)}
    ),
]
n_functions = [(3, 3), (3, 3)]

basis = MultivariateBasis(name=basis_name, n_functions=n_functions, argvals=argvals)

_ = plot_multivariate(basis)

###############################################################################
# Third example
# -------------
# The third example is a multivariate basis where the first component is a Fourier basis (unidimensional) and the second component is a tensor product of Legendre basis (multidimensional). The number of functions in the multivariate basis is set to 9. To be coherent with the number of functions in the Fourier basis, the number of functions in the Legendre basis is set to 3 for each dimension (it results in :math:`3 \times 3 = 9` multidimensional functions).

# Parameters
basis_name = ["fourier", ("legendre", "legendre")]
argvals = [
    DenseArgvals({"input_dim_0": np.linspace(0, 1, 11)}),
    DenseArgvals(
        {"input_dim_0": np.linspace(0, 0.5, 11), "input_dim_1": np.linspace(0, 1, 11)}
    ),
]
n_functions = [9, (3, 3)]

basis = MultivariateBasis(name=basis_name, n_functions=n_functions, argvals=argvals)

_ = plot_multivariate(basis)
