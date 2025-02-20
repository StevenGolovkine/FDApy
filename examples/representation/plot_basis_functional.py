"""
Representation of functional data using a basis
===============================================

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.representation import Basis, BasisFunctionalData
from FDApy.representation import DenseArgvals
from FDApy.visualization import plot

###############################################################################
# In this section, we are showing the building blocks of the representation of functional data using a basis. To define a :class:`~FDApy.representation.BasisFunctionalData` object, we need a :class:`~FDApy.representation.Basis` object and a set of coefficients. The basis object contains the information about the basis functions and the argvals. The coefficients are the weights of the basis functions. The basis functions are evaluated at the argvals and multiplied by the coefficients to obtain the functional data.


###############################################################################
# For unidimensional functional data
# ----------------------------------
# First, we will consider unidimensional functional data. We represent a functional data using a Fourier basis. The coefficients are drawn from a normal distribution.

n_functions = 5
argvals = DenseArgvals({"input_dim_0": np.linspace(0, 1, 101)})

basis = Basis(name="fourier", n_functions=n_functions, argvals=argvals)
coefs = np.random.normal(size=(3, n_functions))

data = BasisFunctionalData(basis=basis, coefficients=coefs)

plot(data)
plt.show()


###############################################################################
# For two-dimensional functional data
# -----------------------------------
# Second, we will consider two-dimensional functional data. We represent a functional data using a tensor product of two Fourier basis. The coefficients are drawn from a normal distribution.

name = ("fourier", "fourier")
n_functions = (5, 5)
argvals = DenseArgvals(
    {"input_dim_0": np.linspace(0, 1, 101), "input_dim_1": np.linspace(0, 1, 51)}
)

basis = Basis(name=name, n_functions=n_functions, argvals=argvals)
coefs = np.random.normal(size=(3, np.prod(n_functions)))

data_2d = BasisFunctionalData(basis=basis, coefficients=coefs)

plot(data_2d)
plt.show()
