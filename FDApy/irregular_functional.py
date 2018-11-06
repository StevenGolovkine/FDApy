#!/usr/bin/python3.7
# -*-coding:utf8 -*

import itertools
import numpy as np 

############################################################################
# Checkers used by the IrregularFunctionalData class

def _check_argvals(argvals):
    """ Check the user provided `argvals`. 

    Parameters
    ----------
    argvals : list of tuples
        A list of numeric vectors (tuples) giving the sampling points for each realization of the process. 

    Return
    ------
    argvals : list of tuples
    """
    if type(argvals) not in (tuple, list):
        raise ValueError('argvals has to be a list of tuples or a tuple!')
    if isinstance(argvals, list) and \
            not all([isinstance(i, tuple) for i in argvals]):
        raise ValueError('argvals has to be a list of tuples or a tuple!')
    if isinstance(argvals, tuple):
        print('argvals is convert into one dimensional list.')
        argvals = [argvals]

    # Check if all entries of `argvals` are numeric. 
    argvals_ = list(itertools.chain.from_iterable(argvals))
    if not all([type(i) in (int, float) for i in argvals_]):
        raise ValueError('All argvals elements must be numeric!')
        
    return argvals

def _check_values(values):
    """Check the user provided `values`.
    
    Parameters
    ----------
    values : list of numpy.array
        A list of numpy array containing values.

    Return
    ------
    values : list of numpy array
    """

    # TODO: Modify the function to deal with other types of data.
    if isinstance(values, np.ndarray):
        print('values is convert into one dimensional list.')
        values = [values]
    if not all([isinstance(i, np.ndarray) for i in values]):
        raise ValueError('values has to be a list of numpy array!')

    return values 

############################################################################
# Class IrregularFunctionalData
class IrregularFunctionalData(object):
    """An object for defining Irregular Functional Data.Functional

    Parameters
    ----------
    argvals : list of tuples
        A list of numeric vectors (tuples) giving the sampling points for each realization of the process. 
    values : list of numpy.array
        A list of numeric arrays, representing the values of each realization of the process on the corresponding observation points. 

    Attributes
    ----------

    Notes
    -----

    References
    ----------

    """
    def __init__(self, argvals, values):

        argvals = _check_argvals(argvals)
        values = _check_values(values)

        if len(argvals) != len(values):
            raise ValueError('argvals and values elements have different support dimensions!')
        if [len(i) for i in argvals] != [len(i) for i in values]:
            raise ValueError('argvals and values have different number of sampling points!')

        self.argvals = argvals
        self.values = values

    @property
    def argvals(self):
        return self._argvals
    
    @argvals.setter
    def argvals(self, new_argvals):
        new_argvals = _check_argvals(new_argvals)
        self._argvals = new_argvals

    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, new_values):
        new_values = _check_values(new_values)
        self._values = new_values

    def nObs(self):
        """Number of observations of the object.

        Return
        ------
        n : int
            Number of observations of the object. 

        """
        n = len(self.values)
        return n