#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import scipy
import sklearn

def rangeStandardization_(X):
    """Transform a vector [a, b] into a vector [0, 1].

    Parameters
    ----------
    X : array-like, shape = (n_features, )
        Data

    Return
    ------
    range_ : array_like, shape = (n_features)
    """
    range_ = (X - np.min(X)) / (np.max(X) - np.min(X))
    return range_

def rowMean_(X):
    """Compute the mean of an array with respect to the rows.

    Parameters
    ----------
    X : array-like, shape = (n_obs, n_features)
        Data

    Return
    ------
    mean_ : array-like, shape = (n_features,)
    """
    scaler = sklearn.preprocessing.StandardScaler()
    return scaler.fit(X).mean_

def rowVar_(X):
    """Compute the variance of an array with respect to the rows.
    
    Parameters
    ----------
    X : array-like, shape = (n_obs, n_features)
        Data

    Return
    ------
    var_ : array-like, shape = (n_features,)
    """
    scaler = sklearn.preprocessing.StandardScaler()
    return scaler.fit(X).var_

def colMean_(X):
    """Compute the mean of an array with respect to the columns.

    Parameters
    ----------
    X : array-like, shape = (n_obs, n_features)
        Data

    Return
    ------
    mean_ : array-like, shape = (n_obs,)
    """
    scaler = sklearn.preprocessing.StandardScaler()
    return scaler.fit(X.T).mean_

def colVar_(X):
    """Compute the variance of an array with respect to the columns.
    
    Parameters
    ----------
    X : array-like, shape = (n_obs, n_features)
        Data
        
    Return
    ------
    var_ : array-like, shape = (n_obs,)
    """
    scaler = sklearn.preprocessing.StandardScaler()
    return scaler.fit(X.T).var_

def tensorProduct_(X, Y):
    """Compute the tensor product of two vectors."""
    return np.outer(X, Y)

def integrate_(X, Y, method='simpson'):
    """Integrate Y over the domain X.

    Parameters
    ----------
    X : array-like, shape = (n_features,)
        Domain for the integration, it has to be ordered.
    Y : array-like, shape = (n_features,)
        Observations
    method : str, default = 'simpson'
        The method used to integrated. Currently, only the Simpsons method is implemented.

    Return
    ------
    res : int
        Estimation of the integration of Y over X. 
    """
    if method is not 'simpson':
        raise ValueError('Only the Simpsons method is implemented!')
    return scipy.integrate.simps(Y, X)
