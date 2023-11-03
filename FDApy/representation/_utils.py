#!/usr/bin/env python
# -*-coding:utf8 -*
"""
Utilities
---------

"""

import numpy as np

from .functional_data import DenseFunctionalData
from .argvals import DenseArgvals
from .values import DenseValues

from ..misc.utils import _outer


def _tensor_product(
    data1: DenseFunctionalData,
    data2: DenseFunctionalData
) -> DenseFunctionalData:
    """Compute the tensor product between functional data.

    Compute the tensor product between all the observation of data1 with all
    the observation of data2.

    Parameters
    ----------
    data1: DenseFunctionalData
        First functional data.
    data2: DenseFunctionalData
        Second functional data.

    Returns
    -------
    DenseFunctionalData
        The tensor product between data1 and data2. It contains data1.n_obs *
        data2.n_obs observations.

    Notes
    -----
    TODO:
    * Add tests.

    """
    arg = {
        'input_dim_0': data1.argvals['input_dim_0'],
        'input_dim_1': data2.argvals['input_dim_0']
    }
    val = [_outer(i, j) for i in data1.values for j in data2.values]
    return DenseFunctionalData(DenseArgvals(arg), DenseValues(np.array(val)))
