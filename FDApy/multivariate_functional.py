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
        raise ValueError(
            'data has to be a list or elements of FDApy.univariate_functional.UnivariateFunctionalData or FDApy.irregular_functional.IrregularFunctionalData!')
    if type(data) in (FDApy.univariate_functional.UnivariateFunctionalData,
                      FDApy.irregular_functional.IrregularFunctionalData):
        print('Convert data into one dimensional list.')
        data = [data]
    if not all(
            [type(i) in (
                FDApy.univariate_functional.UnivariateFunctionalData,
                FDApy.irregular_functional.IrregularFunctionalData
            ) for i in data]):
        raise ValueError(
            'Elements of the list have to be objects from the class FDApy.univariate_functional.UnivariateFunctionalData or FDApy.irregular_functional.IrregularFunctionalData!')
    if any(np.diff([i.nObs() for i in data])):
        raise ValueError(
            'Elements of the list must have the same number of observations!')
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
        self.data = data

    def __repr__(self):
        res = "Multivariate Functional data objects with " +\
            str(self.nFunctions()) +\
            " funtions:\n"
        for i in self.data:
            res += "- " + repr(i) + "\n"
        return res

    def __getitem__(self, index):
        """Function called when self[index]

        Parameters
        ----------
        index : int
            The observation(s) of the object to retrieve. 

        Return
        ------
        res : MultivariateFunctionalData object
            The selected observation(s) as MultivariateFunctionalData object.

        """
        data = self.data[index]
        res = MultivariateFunctionalData(data)
        return res

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

    def rangeObs(self):
        """Range of the observations of the objects. 

        Return
        ------
        range_ : list of tuples
            List of tuple containing the range of the observations for each individual functions. 

        """
        range_ = [i.rangeObs() for i in self.data]
        return range_

    def nObsPoint(self):
        """Number of sampling points of the objects. 

        Return
        ------
        n : list of list of int
            List of the length of self.nFunctions() where the (i,j)-th entry correpond to the number of sampling points of the i-th functions of the j-th dimensions of the observations. 

        """
        n = [i.nObsPoint() for i in self.data]
        return n

    def rangeObsPoint(self):
        """ Range of the observations of the objects. 

        Return
        ------
        range_ : list of list of tuples of the length of self.nFunctions() containing the minimum and maximum number where the (i,j)-th entry contains the range of the i-th function of the j-th dimensions of the observations. 

        """
        range_ = [i.rangeObsPoint() for i in self.data]
        return range_

    def dimension(self):
        """ Common dimension of the observation of the object. 

        Return
        ------
        dim : list of int
            List of length self.nFunctions() where the i-th entry contains the number of dimension of the observations for the i-th function of the object. 

        """
        dim = [i.dimension() for i in self.data]
        return dim

    def add(self, new_function):
        """ Add a ne function to the MultivariateFunctionalData object.

        Parameters
        ----------
        new_function : FDApy.univariate_functional.UnivariateFunctionalData or FDApy.irregular_functional.IrregularFunctionalData
            an object of the class FDApy.univariate_functional.UnivariateFunctionalData or FDApy.irregular_functional.IrregularFunctionalData to add to the MultivariateFunctionalData object.

        Return
        ------
        obj : an object of the class FDApy.multivariate_functional.MultivariateFunctionalData

        """
        data = self.data
        data.append(new_function)
        return FDApy.multivariate_functional.MultivariateFunctionalData(data)