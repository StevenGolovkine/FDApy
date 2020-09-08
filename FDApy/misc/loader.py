#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""
Module for the definition of different loaders for FunctionalData types.

This modules is used to defined different loaders to load common data files
(such as csv, ts, ...) into an object of the class DenseFunctionalData,
IrregularFunctionalData or MultivariateFunctionalData.
"""
import numpy as np
import pandas as pd

from sktime.utils.load_data import load_from_tsfile_to_dataframe

from FDApy.representation.functional_data import (DenseFunctionalData,
                                                  IrregularFunctionalData)


###############################################################################
# Loader for csv
def read_csv(filepath, **kwargs):
    """Read a comma-separated values (csv) file into Functional Data.

    Build a DenseFunctionalData or IrregularFunctionalData object upon a csv
    file passed as parameter.

    Notes
    -----
    It is assumed that the data are unidimensional. And so, it will not be
    checked.

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
        all_argvals = data.columns.astype(np.int64)
    except TypeError:
        all_argvals = np.arange(0, len(data.columns))

    if not data.isna().values.any():
        obj = read_csv_dense(data, all_argvals)
    else:
        obj = read_csv_irregular(data, all_argvals)
    return obj


def read_csv_dense(data, argvals):
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
    argvals = {'input_dim_0': argvals}
    values = np.array(data)
    return DenseFunctionalData(argvals, values)


def read_csv_irregular(data, argvals):
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
    argvals = {'input_dim_0': {idx: np.array(argvals[~np.isnan(row)])
                               for idx, row in enumerate(data.values)}}
    values = {idx: row[~np.isnan(row)] for idx, row in enumerate(data.values)}
    return IrregularFunctionalData(argvals, values)


###############################################################################
# Loader for ts
def read_ts(filepath, **kwargs):
    """Read a ts file into Functional Data.

    Build a DenseFunctionalData or IrregularFunctionalData object upon a ts
    file passed as parameter.

    Notes
    -----
    It is assumed that the data are unidimensional. And so, it will not be
    checked.

    Parameters
    ----------
    filepath: str
        Any valid string path is acceptable.
    **kwargs:
        Keywords arguments to passed to the load_from_tsfile_to_dataframe
        function.

    Returns
    -------
    obj: DenseFunctionalData or IrregularFunctionalData
        The loaded csv file.
    labels: np.ndarray
        Labels

    """
    data, labels = load_from_tsfile_to_dataframe(filepath, **kwargs)

    len_argavals = data.applymap(len)['dim_0'].unique()

    if len(len_argavals) == 1:
        obj = read_ts_dense(data)
    else:
        obj = read_ts_irregular(data)
    return obj, labels


def read_ts_dense(data):
    """Load a ts file into a DenseFunctionalData object.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe.

    Returns
    -------
    obj: DenseFunctionalData
        The loaded ts file.

    """
    argvals = data.loc[0, 'dim_0'].index.values
    values = np.zeros((len(data), len(argvals)))
    for idx, row in data.iterrows():
        values[idx, :] = row['dim_0'].values
    return DenseFunctionalData({'input_dim_0': argvals}, values)


def read_ts_irregular(data):
    """Load a ts file into an IrregularFunctionalData object.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe

    Returns
    -------
    obj: IrregularFunctionalData
        The loaded ts file.

    """
    argvals, values = {}, {}
    for idx, row in data.iterrows():
        argvals[idx] = row['dim_0'].index.values
        values[idx] = row['dim_0'].values
    return IrregularFunctionalData({'input_dim_0': argvals}, values)
