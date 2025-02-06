"""
CD4 dataset
-----------

"""

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy import read_csv
from FDApy.visualization import plot

##############################################################################
# In this section, we will use the CD4 dataset to illustrate the use of the package concerning irregularly sampled functional data. The dataset contains CD4 counts for :math:`366` patients. The dataset can be downloaded from the `here <https://github.com/StevenGolovkine/FDApy/blob/examples/examples/data/cd4.csv>`_. This is an example of a :class:`~FDApy.representation.IrregularFunctionalData` object. It also shows how the CSV file should be formatted and can be read using the :func:`~FDApy.read_csv` function.

# Load data
cd4_data = read_csv("../data/cd4.csv", index_col=0)

_ = plot(cd4_data)
plt.show()
