#!/usr/bin/python3.7
# -*-coding:utf8 -*

import itertools
import numpy as np

import FDApy

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
    # TODO: Modify the condition to accept multidimensional irregular functional data.
    if isinstance(argvals, list) and \
            not all([isinstance(i, tuple) for i in argvals]):
        raise ValueError('argvals has to be a list of tuples or a tuple!')
    if isinstance(argvals, tuple):
        print('argvals is convert into one dimensional list.')
        argvals = [argvals]

    # Check if all entries of `argvals` are numeric.
    argvals_ = list(itertools.chain.from_iterable(argvals))
    if not all([type(i) in (int, float, np.int_, np.float_) 
            for i in argvals_]):
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


def _check_argvals_values(argvals, values):
    """Check the compatibility of argvals and values. 

    Parameters
    ----------
    argvals : list of tuples
        List of tuples containing the sample points. 
    values : list of numpy.ndarray
        list of numpy array containing the values. 

    Return
    ------
    True, if the argvals and the values are ok.
    """

    if len(argvals) != len(values):
        raise ValueError(
            'argvals and values elements have different support dimensions!')
    if [len(i) for i in argvals] != [len(i) for i in values]:
        raise ValueError(
            'argvals and values have different numbers of sampling points!')

    return True

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

    standardize : boolean, default = True
        Do we standardize the argvals to be in [0, 1].    

    Attributes
    ----------

    Notes
    -----
    Currently, only one dimensional irregular functional data have been implemented.

    The standardization of the Irregular Functional Data is useful only if all are defined on different domains and we have the complete trajectories. And it may not be useful if all the functional data are defined on the same domain and we do not record some the signal for some times. 

    References
    ----------

    """

    def __init__(self, argvals, values, standardize=True):

        self.argvals = argvals
        self.values = values

        if standardize:
            argvals_stand = []
            for argval in self.argvals:
                if len(argval) > 1:
                    argvals_stand.append(tuple(
                        FDApy.utils.rangeStandardization_(argval)))
                else:
                    argvals_stand.append(tuple([0]))
            self.argvals_stand = argvals_stand

    def __repr__(self):
        res = "Irregular Functional data objects with " +\
            str(self.nObs()) +\
            " observations of " +\
            str(self.dimension()) +\
            "-dimensional support\n" +\
            "argvals:\n" +\
            "\tValues in " +\
            str(self.rangeObsPoint()[0]) +\
            " ... " +\
            str(self.rangeObsPoint()[1]) +\
            ".\nvalues:\n" +\
            "\tValues in " +\
            str(self.rangeObs()[0]) +\
            " ... " +\
            str(self.rangeObs()[1]) +\
            ".\nThere are " +\
            str(min(self.nObsPoint())) +\
            " - " +\
            str(max(self.nObsPoint())) +\
            " sampling points per observation."
        return res

    def __getitem__(self, index):
        """Function called when self[index]

        Parameters
        ----------
        index : int
            The observation(s) of the object to retrieve. 

        Return
        ------
        res : IrregularFunctionalData object
            The selected obsevation(s) as IrregularFunctionalData object.

        """
        argvals = self.argvals[index]
        values = self.values[index]
        res = IrregularFunctionalData(argvals, values)
        return res

    @property
    def argvals(self):
        return self._argvals

    @argvals.setter
    def argvals(self, new_argvals):
        new_argvals = _check_argvals(new_argvals)
        if hasattr(self, 'values'):
            _check_argvals_values(new_argvals, self.values)
        self._argvals = new_argvals

    @property
    def argvals_stand(self):
        return self._argvals_stand
    
    @argvals_stand.setter
    def argvals_stand(self, new_argvals_stand):
        self._argvals_stand = new_argvals_stand

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        new_values = _check_values(new_values)
        if hasattr(self, 'argvals'):
            _check_argvals_values(self.argvals, new_values)
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
        values_ = list(itertools.chain.from_iterable(self.values))
        return min(values_), max(values_)

    def nObsPoint(self):
        """Number of sampling points of the objects. 

        Return
        ------
        n : list of int
            List of the same length of self.nObs() where the i-th entry correspond to the number of sampling points of the i-th observed function. 

        """
        n = [len(i) for i in self.values]
        return n

    def rangeObsPoint(self):
        """Range of sampling points of the objects. 

        Return
        ------
        min(argvals_), max(argvals_) : tuple
            Tuple containing the minimum and maximum number of all the sampling points for an object.

        """
        argvals_ = list(itertools.chain.from_iterable(self.argvals))
        return min(argvals_), max(argvals_)

    def dimension(self):
        """Common dimension of the observations of the object.

        Return
        ------
        dim : int
            Number of dimension of the observations of the object. 

        Note
        ----
        Currently, this function must always return 1 as the multi-dimensional irregular functional data is not yet implemented.
        """
        dim = 1
        return dim

    