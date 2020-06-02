"""
CD4 cell count analysis
=======================

This notebook shows how to deal with irregular functional data by analyzing the
dataset CD4 cell count.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

import numpy as np
import pandas as pd

from FDApy.irregular_functional import IrregularFunctionalData
from FDApy.plot import plot

###############################################################################
# Load the data into Pandas dataframe.
cd4 = pd.read_csv('./data/cd4.csv', index_col=0)


###############################################################################
# Create IrregularFunctionalData for cd4 data.
all_argvals = cd4.columns.astype(np.int64)
argvals = [np.array(all_argvals[~np.isnan(row)]) for row in cd4.values]
values = [row[~np.isnan(row)] for row in cd4.values]
cd4counts = IrregularFunctionalData(argvals, values)

###############################################################################
# Print out an Irregular Functional data object.

# Print irregular functional data
print(cd4counts)

###############################################################################
# The sampling points of the data can easily be accessed.

# Accessing the argvals of the object
print(cd4counts.argvals[0:5])

###############################################################################
# The values associated to the sampling points are retrieved in a same way
# than the sampling points.

# Accessing the values of the object
print(cd4counts.values[0:5])

###############################################################################
# The number of observations within the data are obtained using the function
# :func:`~FDApy.irregular_functional.IrregularFunctional.nObs`.

# Get the number of observations for the object
print(cd4counts.nObs())

###############################################################################
# The number of sampling points per observation is given by the function
# :func:`~FDApy.irregular_functional.IrregularFunctional.nObsPoint`.

# Retrieve the number of sampling points for the object
print(cd4counts.nObsPoint()[0:5])

###############################################################################
# The dimension of the data is given by the function
# :func:`~FDApy.irregular_functional.IrregularFunctional.dimension`.

# Get the dimension of the domain of the observations
print(cd4counts.dimension())

###############################################################################
# The extraction of observations is also easily done.

# Extract observations from the object
print(cd4counts[5:8])

###############################################################################
# Finally, we can plot the data.

fig, ax = plot(cd4counts,
               main='CD4 cell counts',
               xlab='Month since seroconversion',
               ylab='CD4 cell count (log-scale)')
