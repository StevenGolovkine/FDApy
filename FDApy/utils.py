#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np
import scipy
import sklearn


#############################################################################
# Standardization functions
#############################################################################
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

############################################################################
# Array manipulation functions.
############################################################################
def shift_(X, num, fill_value=np.nan):
	"""Shift an array `X` by a number `num`.
	
	Parameters
	----------
	X : array-like ,shape = (n_obs, n_features)
		Input array
	num : int
		The number of columns to shift.
	fill_value : float or np.nan
		The value with one fill the array.
		
	Return
	------
	res : array-like, shape = (n_obs, n_features)
		The shift array.
		
	References
	----------
	* https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array/42642326
    
    TODO: PA LE BON SENS!
	"""
	res = np.empty_like(X)
	if num > 0:
		res[:num] = fill_value
		res[num:] = X[:-num]
	elif num < 0:
		res[num:] = fill_value
		res[:num] = X[-num:]
	else:
		res = X
	return res

##############################################################################
# Array computation
##############################################################################
def tensorProduct_(X, Y):
    """Compute the tensor product of two vectors.
	
	Parameters
	----------
	X : array-like, shape = (n_obs1,)
		First input vector
	Y : array-like, shape = (n_obs2,)
		Second input vector
		
	Return
	------
	res : ndarray, shape = (n_obs1, n_obs2)
	"""
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
        The method used to integrated. Currently, only the Simpsons method is
		implemented.

    Return
    ------
    res : int
        Estimation of the integration of Y over X. 
    """
    if method is not 'simpson':
        raise ValueError('Only the Simpsons method is implemented!')
    return scipy.integrate.simps(Y, X)

def integrationWeights_(X, method='trapz'):
	"""Compute weights for numerical integration over the domain `X` given 
	the method `method`.
	
	Parameters
	----------
	X : array-like, shape = (n_points,)
		Domain on which compute the weights.
	method : str or callable, default = 'trapz'
		The method to compute the weights.
		
	Return
	------
	W : array-like, shape = (n_points,)
		The weights
		
	Notes
	-----
	TODO :
	* Add other methods: Simpson, midpoints, ...
	* Add tests
	
	References
	----------
	* https://en.wikipedia.org/wiki/Trapezoidal_rule
	"""
	L = len(X)
	if method is 'trapz':
		W = 1/2 * np.concatenate([[X[1] - X[0]], 
							  X[2:] - X[:(L-2)], 
							  [X[L-1] - X[L-2]]])
	elif callable(method):
		W = method(X)
	else:
		raise ValueError('Method {} not implemented!'.format(method))
	
	return W
	