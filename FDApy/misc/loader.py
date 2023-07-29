#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Loaders
-------

"""
import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import Union

from ..representation.argvals import DenseArgvals, IrregularArgvals
from ..representation.values import DenseValues, IrregularValues
from ..representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData
)


###############################################################################
# Loader for csv
def read_csv(
    filepath: str,
    **kwargs
) -> Union[DenseFunctionalData, IrregularFunctionalData]:
    """Read a comma-separated values (csv) file into Functional Data.

    Build a DenseFunctionalData or IrregularFunctionalData object upon a csv
    file passed as parameter.

    Notes
    -----
    We assumed that the data are unidimensional and is not checked.

    Parameters
    ----------
    filepath: str
        Any valid string path is acceptable.
    **kwargs
        Keywords arguments to passed to the pd.read_csv function.

    Returns
    -------
    Union[DenseFunctionalData, IrregularFunctionalData]
        The loaded csv file.

    """
    data = pd.read_csv(filepath, **kwargs)

    try:
        all_argvals = data.columns.astype(np.int64).to_numpy()
    except ValueError:
        all_argvals = np.arange(0, len(data.columns))

    if not data.isna().values.any():
        return _read_csv_dense(data, all_argvals)
    else:
        return _read_csv_irregular(data, all_argvals)


def _read_csv_dense(
    data: pd.DataFrame,
    argvals: npt.NDArray[np.float64]
) -> DenseFunctionalData:
    """Load a csv file into a DenseFunctionalData object.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe.
    argvals: npt.NDArray[np.float64]
        An array of argvals.

    Returns
    -------
    DenseFunctionalData
        The loaded csv file

    """
    argvals_ = DenseArgvals({'input_dim_0': argvals})
    values = DenseValues(np.array(data))
    return DenseFunctionalData(argvals_, values)


def _read_csv_irregular(
    data: pd.DataFrame,
    argvals: npt.NDArray[np.float64]
) -> IrregularFunctionalData:
    """Load a csv file into an IrregularFunctionalData object.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe.
    argvals: npt.NDArray[np.float64]
        An array of argvals.

    Returns
    -------
    IrregularFunctionalData
        The loaded csv file.

    """
    argvals_ = {
        idx: DenseArgvals({'input_dim_0': argvals[~np.isnan(row)]})
        for idx, row in enumerate(data.values)
    }
    values = {idx: row[~np.isnan(row)] for idx, row in enumerate(data.values)}
    return IrregularFunctionalData(
        IrregularArgvals(argvals_),
        IrregularValues(values)
    )
