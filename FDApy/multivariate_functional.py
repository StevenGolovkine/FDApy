#!/usr/bin/python3.7
# -*-coding:utf8 -*

import itertools
import numpy as np 

import FDApy

############################################################################
# Checkers used by the MultivariateFunctionalData class. 

def _check_data(data):
    """Check the user provided `data`. 

    Parameters
    ----------
    data : list of UnivariateFunctionalData or IrregularFunctionalData
        A list of elements from the class UnivariateFunctionalData or IrregularFunctionalData giving individuals. 
        
    Return
    ------
    data : list of UnivariateFunctionalData ot IrregularFunctionalData
    """
    if type(data) not in (list, 
            FDApy.univariate_functional.UnivariateFunctionalData,
            FDApy.irregular_functional.IrregularFunctionalData):
        raise ValueError('data has to be a list or elements of FDApy.univariate_functional.UnivariateFunctionalData or FDApy.irregular_functional.IrregularFunctionalData!')
    if type(data) in (FDApy.univariate_functional.UnivariateFunctionalData,
            FDApy.irregular_functional.IrregularFunctionalData):
        print('Convert data into one dimensional list.')
        data = [data]
    if not all(
            [type(i) in (
                FDApy.univariate_functional.UnivariateFunctionalData, 
                FDApy.irregular_functional.IrregularFunctionalData
                ) for i in data]):
        raise ValueError('Elements of the list have to be objects from the class FDApy.univariate_functional.UnivariateFunctionalData or FDApy.irregular_functional.IrregularFunctionalData!')
    if any(np.diff([i.nObs() for i in data])):
        raise ValueError('Elements of the list must have the same number of observations!')
    return data

############################################################################
# Class MultivariateFunctionalData
class MultivariateFunctionalData(object):
    """An object for defining Multivariate Functional Data. 

    Parameters
    ----------
    data : list of UnivariateFunctionalData or IrregularFunctionalData
        A list of elements from the class UnivariateFunctionalData or IrregularFunctionalData giving individuals. 

    Attributes
    ----------

    Notes
    -----

    References
    ----------

    """
    def __init__(self, data):
        
        data = _check_data(data)
        self.data = data

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        new_data = _check_data(new_data)
        self._data = new_data

    def nFunctions(self):
        """Number of functions of the objects. 

        Return
        ------
        n : int
            Number of functions of the objects. 

        """
        n = len(self.data)
        return n

    def nObs(self):
        """Number of observations of the object.

        Return
        ------
        n : int
            Number of observations of the object. 

        """
        n = self.data[0].nObs()
        return n