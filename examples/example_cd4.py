"""
CD4 cell count analysis
==========================

This notebook shows how to deal with irregular functional data by analyzing the
dataset CD4 cell count.
"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# shinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from FDApy.irregular_functional import IrregularFunctionalData
from FDApy.plot import plot

###############################################################################
# Load the data into Pandas dataframe
cd4 = pd.read_csv('./data/cd4.csv', index_col=0)


###############################################################################
# Create IrregularFunctionalData for cd4 data

all_argvals = cd4.columns.astype(np.int64)
argvals = [np.array(all_argvals[~np.isnan(row)]) for row in cd4.values]
values = [row[~np.isnan(row)] for row in cd4.values]
cd4counts = IrregularFunctionalData(argvals, values)

###############################################################################
# Print out an Irregular Functional data object

# Print irregular functional data
cd4counts

###############################################################################
# The attributs of the classes can easily be accessed.

# Accessing the argvals of the object
cd4counts.argvals[0:5]

# Accessing the values of the object
cd4counts.values[0:5]

# Get the number of observations for the object
cd4counts.nObs()

# Retrieve the number of sampling points for the object
cd4counts.nObsPoint()[0:5]

# Get the dimension of the domain of the observations
cd4counts.dimension()

###############################################################################
# The extraction of observations is also easily done.

# Extract observations from the object
cd4counts[5:8]

###############################################################################
# Finally, we can plot the data.

fig, ax = plot(cd4counts,
               main='CD4 cell counts',
               xlab='Month since seroconversion',
               ylab='CD4 cell count (log-scale)')
plt.show()
