"""
CD4 cell count analysis
=======================

This notebook shows how to deal with irregular functional data by analyzing the
dataset CD4 cell count.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2


from FDApy.misc.loader import read_csv
from FDApy.visualization.plot import plot

###############################################################################
# Load the data into Pandas dataframe.
cd4 = read_csv('./data/cd4.csv', index_col=0)

###############################################################################
# Print out an Irregular Functional data object.

# Print irregular functional data
print(cd4)

###############################################################################
# The sampling points of the data can easily be accessed.

# Accessing the argvals of the object
print(cd4.argvals['input_dim_0'].get(5))

###############################################################################
# The values associated to the sampling points are retrieved in a same way
# than the sampling points.

# Accessing the values of the object
print(cd4.values.get(5))

###############################################################################
# The number of observations within the data are obtained using the function
# :func:`~FDApy.irregular_functional.IrregularFunctional.n_obs`.

# Get the number of observations for the object
print(cd4.n_obs)

###############################################################################
# The number of sampling points per observation is given by the function
# :func:`~FDApy.irregular_functional.IrregularFunctional.n_points`.

# Retrieve the mean number of sampling points for the object
print(cd4.n_points)

###############################################################################
# The dimension of the data is given by the function
# :func:`~FDApy.irregular_functional.IrregularFunctional.n_dim`.

# Get the dimension of the domain of the observations
print(cd4.n_dim)

###############################################################################
# The extraction of observations is also easily done.

# Extract observations from the object
print(cd4[5:8])

###############################################################################
# Finally, we can plot the data.

_ = plot(cd4)
