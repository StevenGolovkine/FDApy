#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Loaders
-------

"""
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..representation.argvals import DenseArgvals, IrregularArgvals
from ..representation.values import DenseValues, IrregularValues
from ..representation.functional_data import (
    DenseFunctionalData,
    IrregularFunctionalData,
)


###############################################################################
# Loader for csv
def read_csv(filepath: str, **kwargs) -> DenseFunctionalData | IrregularFunctionalData:
    """Load CSV file into functional data object.

    Build a DenseFunctionalData or IrregularFunctionalData object upon a CSV
    file passed as parameter. If the CSV file does not contain any `NA` values, the
    data will be loaded as a DenseFunctionalData object. Otherwise, it will be loaded
    as an IrregularFunctionalData object. See the `Canadian Weather example
    <../../auto_examples/data_analysis/plot_canadian_weather.html>`_ and
    `CD4 example <../../auto_examples/data_analysis/plot_cd4.html>`_ for the formating
    of the CSV file.


    Notes
    -----
    We assumed that the data are unidimensional without check.

    Parameters
    ----------
    filepath
        Any valid string path is acceptable.
    kwargs
        Keywords arguments to passed to the pd.read_csv function.

    Returns
    -------
    DenseFunctionalData | IrregularFunctionalData
        The loaded CSV file.

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
    data: pd.DataFrame, argvals: npt.NDArray[np.float64]
) -> DenseFunctionalData:
    """Load a csv file into a DenseFunctionalData object.

    Parameters
    ----------
    data
        Input dataframe.
    argvals
        An array of argvals.

    Returns
    -------
    DenseFunctionalData
        The loaded csv file

    """
    argvals_ = DenseArgvals({"input_dim_0": argvals})
    values = DenseValues(np.array(data))
    return DenseFunctionalData(argvals_, values)


def _read_csv_irregular(
    data: pd.DataFrame, argvals: npt.NDArray[np.float64]
) -> IrregularFunctionalData:
    """Load a csv file into an IrregularFunctionalData object.

    Parameters
    ----------
    data
        Input dataframe.
    argvals
        An array of argvals.

    Returns
    -------
    IrregularFunctionalData
        The loaded csv file.

    """
    argvals_ = {
        idx: DenseArgvals({"input_dim_0": argvals[~np.isnan(row)]})
        for idx, row in enumerate(data.values)
    }
    values = {idx: row[~np.isnan(row)] for idx, row in enumerate(data.values)}
    return IrregularFunctionalData(IrregularArgvals(argvals_), IrregularValues(values))
