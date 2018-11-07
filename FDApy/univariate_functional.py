#!/usr/bin/python3.7
# -*-coding:utf8 -*

import itertools
import numpy as np 


#############################################################################
# Checkers used by the UnivariateFunctionalData class.

def _check_argvals(argvals):
    """Check the user provided `argvals`.
    
    Parameters
    ---------
    argvals : list of tuples
        A list of numeric vectors (tuples) or a single numeric vector (tuple) giving the sampling points in the domains. 

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
        print('argvals is converted into one dimensional list!')
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
    values : numpy.array
        A numpy array containing values.

    Return
    ------
    values : numpy array
    """

    # TODO: Modify the function to deal with other types of data.
    if not isinstance(values, np.ndarray):
        raise ValueError('values has to be a numpy array!')

    return values

#############################################################################
# Class UnivariateFunctionalData 
class UnivariateFunctionalData(object):
    """An object for defining Univariate Functional Data.

    Parameters
    ----------
    argvals : list of tuples
        A list of numeric vectors (tuples) or a single numeric vector (tuple) giving the sampling points in the domains.

    values : array-like
        An array, giving the observed values for N observations. Missing values should be included via `None` (or `np.nan`). The shape depends on `argvals`::

            (N, M) if `argvals` is a single numeric vector,
            (N, M_1, ..., M_d) if `argvals` is a list of numeric vectors.

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

        if len(argvals) != len(values.shape[1:]):
            raise ValueError('argvals and values elements have different support dimensions!')
        if tuple(len(i) for i in argvals) != values.shape[1:]:
            raise ValueError('argvals and values have different number of sampling points!')

        self.argvals = argvals
        self.values = values

    def __repr__(self):
        res = "Univariate Functional data objects with " +\
                str(self.nObs()) +\
                " observations of " +\
                str(self.dimension()) +\
                "-dimensional support\n" +\
                "argvals:\n"
        for i in range(len(self.argvals)):
            res += "\t" +\
                    str(self.argvals[i][0]) +\
                    ", " +\
                    str(self.argvals[i][1]) +\
                    ", ... ," +\
                    str(self.argvals[i][-1]) +\
                    "\t(" +\
                    str(len(self.argvals[i])) +\
                    " sampling points)\n"
        res += "values:\n\tarray of size " +\
                str(self.values.shape)
        return res
        
    def __getitem__(self, index):
        """Function call when self[index]

        Parameters
        ----------
        index : int
            The observation of the object to retrieve.  

        """
        # TODO: Modify the function
        argvals = self.argvals
        values = self.values[index]
        res = UnivariateFunctionalData(argvals, values)
        return res

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

    def rangeObs(self):
        """Range of the observations of the objects. 

        Return
        ------
        min(values_), max(values_) : tuple
            Tuple containing the minimum and maximum number of all the observations for an object.

        """
        if self.dimension() == 1:
            min_ = min(list(itertools.chain.from_iterable(self.values)))
            max_ = max(list(itertools.chain.from_iterable(self.values)))
        else:
            min_ = min([min(i) 
                for i in itertools.chain.from_iterable(self.values)])
            max_ = max([max(i) 
                for i in itertools.chain.from_iterable(self.values)])
        return min_, max_

    def nObsPoint(self):
        """Number of sampling points of the objects. 

        Return
        ------
        n : list of int
            List of the length self.dimension() where the i-th entry correspond to the number of sampling points of the i-th dimension of the observations.

        """
        n = [len(i) for i in self.argvals]
        return n

    def rangeObsPoint(self):
        """Range of the observations of the objects.

        Return
        ------
        range_ : list of tuples containing the minimum and maximum number where the i-th entry of the list contains the range of the i-th dimension of the object.
        """
        range_ = [(min(i), max(i)) for i in self.argvals]
        return range_

    def dimension(self):
        """Common dimension of the observations of the object.

        Return
        ------
        dim : int
            Number of dimension of the observations of the object. 

        """
        dim = len(self.argvals)
        return dim 