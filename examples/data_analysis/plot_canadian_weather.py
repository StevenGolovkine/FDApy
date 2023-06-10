"""
Canadian weather dataset
------------------------

Example of the Canadian weather dataset.
"""

###############################################################################
#

# Author: Steven Golovkine <steven_golovkine@icloud.com>
# License: MIT

# Load packages
import matplotlib.pyplot as plt
import numpy as np

from FDApy.misc.loader import read_csv
from FDApy.visualization.plot import plot


# Load data
temp_data = read_csv('../data/canadian_temperature_daily.csv', index_col=0)


_ = plot(temp_data)


# Smooth the data
points = np.linspace(1, 365, 365)
kernel_name = "epanechnikov"
bandwidth = 30
degree = 1

temp_smooth = temp_data.smooth(
    points=points, kernel_name=kernel_name, bandwidth=bandwidth, degree=degree
)


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for idx, ax in enumerate(axes.flat):
    plot(temp_data[idx], colors='k', alpha=0.2, ax=ax)
    plot(temp_smooth[idx], colors='r', ax=ax)
    ax.set_title(f"Observation {idx + 1}")
plt.show()
