#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Module for the definition of different loaders for FunctionalData types.

This modules is used to defined different loaders to load common data files
(such as csv, ts, ...) into an object of the class DenseFunctionalData,
IrregularFunctionalData or MultivariateFunctionalData.
"""
import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import Any, Union

from FDApy.representation.functional_data import (DenseFunctionalData,
                                                  IrregularFunctionalData)


###############################################################################
# Loader for csv
def read_csv(
    filepath: str,
    **kwargs: Any
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
    **kwargs:
        Keywords arguments to passed to the pd.read_csv function.

    Returns
    -------
    obj: DenseFunctionalData or IrregularFunctionalData
        The loaded csv file.

    """
    data = pd.read_csv(filepath, **kwargs)

    try:
        all_argvals = data.columns.astype(np.int64).to_numpy()
    except TypeError:
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
    argvals: np.ndarray
        An array of argvals.

    Returns
    -------
    obj: DenseFunctionalData
        The loaded csv file

    """
    argvals_ = {'input_dim_0': argvals}
    values = np.array(data)
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
    argvals: np.ndarray
        An array of argvals.

    Returns
    -------
    obj: IrregularFunctionalData
        The loaded csv file.

    """
    tt = {idx: argvals[~np.isnan(row)] for idx, row in enumerate(data.values)}
    argvals_ = {'input_dim_0': tt}
    values = {idx: row[~np.isnan(row)] for idx, row in enumerate(data.values)}
    return IrregularFunctionalData(argvals_, values)
