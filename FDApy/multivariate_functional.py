#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""Module for MultivariateFunctionalData classes.

This module is used define multivariate functional data.
"""
import itertools
import numpy as np

from .univariate_functional import UnivariateFunctionalData
from .irregular_functional import IrregularFunctionalData

############################################################################
# Checkers used by the MultivariateFunctionalData class.


def _check_data(data):
    """Check the user provided `data`.

    Parameters
    ----------
    data : list of UnivariateFunctionalData or IrregularFunctionalData
        A list of elements from the class UnivariateFunctionalData or
        IrregularFunctionalData giving individuals.

    Returns
    -------
    data : list of UnivariateFunctionalData ot IrregularFunctionalData

    """
    if not isinstance(data, (list,
                             UnivariateFunctionalData,
                             IrregularFunctionalData)):
        raise ValueError(
            """Data has to be a list or elements of UnivariateFunctionalData
            or IrregularFunctionalData!""")
    if isinstance(data, (UnivariateFunctionalData,
                         IrregularFunctionalData)):
        data = [data]
    if not all(
            [isinstance(i, (UnivariateFunctionalData, IrregularFunctionalData))
                for i in data]):
        raise ValueError(
            """Elements of the list have to be objects from the class
            UnivariateFunctionalData or IrregularFunctionalData!""")
    if any(np.diff([i.nObs() for i in data])):
        raise ValueError(
            'Elements of the list must have the same number of observations!')
    return data

############################################################################
# Class MultivariateFunctionalData


class MultivariateFunctionalData(object):
    """An object for defining Multivariate Functional Data."""

    def __init__(self, data):
        """Initialize MultivariateFunctionalData object.

        Parameters
        ----------
        data : list of UnivariateFunctionalData or IrregularFunctionalData
            A list of elements from the class UnivariateFunctionalData or
            IrregularFunctionalData giving individuals.
        """
        self.data = data

    def __repr__(self):
        """Override print function."""
        res = "Multivariate Functional data objects with " +\
            str(self.nFunctions()) +\
            " funtions:\n"
        for i in self.data:
            res += "- " + repr(i) + "\n"
        return res

    def __getitem__(self, index):
        """Function called when self[index].

        Parameters
        ----------
        index : int
            The function(s) of the object to retrieve.

        Returns
        -------
        res : UnivariateFunctionalData, IrregularFunctionalData or
        MultivariateFunctionalData object
            The selected function(s) as UnivariateFunctionalData,
            IrregularFunctionalData or MultivariateFunctionalData object.
        """
        data = self.data[index]

        if isinstance(data, UnivariateFunctionalData):
            res = UnivariateFunctionalData(data.argvals, data.values)
        elif isinstance(data, IrregularFunctionalData):
            res = IrregularFunctionalData(data.argvals, data.values)
        else:
            res = MultivariateFunctionalData(data)

        return res

    @property
    def data(self):
        """Getter for `data`."""
        return self._data

    @data.setter
    def data(self, new_data):
        new_data = _check_data(new_data)
        self._data = new_data

    @property
    def mean_(self):
        """Getter for `mean_`."""
        return self._mean_

    @mean_.setter
    def mean_(self, new_mean):
        self._mean_ = new_mean

    def nFunctions(self):
        """Number of functions of the objects.

        Returns
        -------
        n : int
            Number of functions of the objects.

        """
        n = len(self.data)
        return n

    def nObs(self):
        """Number of observations of the object.

        Returns
        -------
        n : int
            Number of observations of the object.

        """
        n = self.data[0].nObs()
        return n

    def rangeObs(self):
        """Range of the observations of the objects.

        Returns
        -------
        range_ : list of tuples
            List of tuple containing the range of the observations for each
            individual functions.

        """
        range_ = [i.rangeObs() for i in self.data]
        return range_

    def nObsPoint(self):
        """Number of sampling points of the objects.

        Returns
        -------
        n : list of list of int
            List of the length of self.nFunctions() where the (i,j)-th entry
            correpond to the number of sampling points of the i-th functions
            of the j-th dimensions of the observations.

        """
        n = [i.nObsPoint() for i in self.data]
        return n

    def rangeObsPoint(self):
        """Range of the observations of the objects.

        Returns
        -------
        range_ : list of list of tuples of the length of self.nFunctions()
        containing the minimum and maximum number where the (i,j)-th entry
        contains the range of the i-th function of the j-th dimensions of the
        observations.

        """
        range_ = [i.rangeObsPoint() for i in self.data]
        return range_

    def dimension(self):
        """Common dimension of the observation of the object.

        Returns
        -------
        dim : list of int
            List of length self.nFunctions() where the i-th entry contains the
            number of dimension of the observations for the i-th function of
            the object.

        """
        dim = [i.dimension() for i in self.data]
        return dim

    def asUnivariateFunctionalData(self):
        """Convert multivariate to univariate functional data.

        Convert a MultivariateFunctionalData object into a
        UnivariateFunctionalData object.

        Notes
        -----
        Be sure that all elements of the list came from the same stochastic
        process generation.
        Currently, only implemented for UnivariateFunctionalData in the list of
        the MultivariateFunctionaData.

        """
        if not all([isinstance(i, UnivariateFunctionalData)
                   for i in self.data]):
            raise ValueError(
                'The data must be a list of UnivariateFunctionalData!')
        if not all([self.data[i - 1].argvals == self.data[i].argvals
                    for i in np.arange(1, self.nFunctions())]):
            raise ValueError('All the argvals are not equals!')

        data_ = []
        for func in self.data:
            data_.append(list(itertools.chain.from_iterable(func.values)))
        return UnivariateFunctionalData(self.data[0].argvals, np.array(data_))

    def mean(self, smooth=False, **kwargs):
        """Compute the mean function.

        Compute the pointwise mean functions of each element of the
        multivariate functional data.

        Parameters
        ----------
        smooth: boolean, default=False
            Should we smooth the mean?
        **kwargs: dict
            The following parameters are taken into account
                - method: 'gaussian', 'epanechnikov', 'tricube', 'bisquare'
                    default='gaussian'
                - degree: int
                    default: 2
                - bandwith: float
                    default=1

        Returns
        -------
        obj : FDApy.multivariate_functional.MultivariateFunctionalData object
            Object of class MultivariateFunctionalData with containing the
            different mean function.

        """
        mean_ = []
        for function in self.data:
            if getattr(function, 'mean_', None) is None:
                function.mean(smooth, **kwargs)
            mean_.append(function.mean_)
        self.mean_ = MultivariateFunctionalData(mean_)

    def covariance(self, smooth=False, **kwargs):
        """Compute the covariance surface.

        Compute the pointwise covariance functions of each element of the
        multivariate functional data.

        Parameters
        ----------
        smooth: boolean, default=False
            Should we smooth the covariance?
        **kwargs: dict
            The following parameters are taken into account
                - method: 'gaussian', 'epanechnikov', 'tricube', 'bisquare'
                    default='gaussian'
                - degree: int
                    default: 2
                - bandwith: float
                    default=1

        Returns
        -------
        obj : FDApy.multivariate_functional.MultivariateFunctionalData object
            Object of class MultivariateFunctionalData with containing the
            different covariance function.

        """
        cov_ = []
        for function in self.data:
            if getattr(function, 'covariance_', None) is None:
                function.covariance(smooth, **kwargs)
            cov_.append(function.covariance_)
        self.covariance_ = MultivariateFunctionalData(cov_)

    def add(self, new_function):
        """Add a one function to the MultivariateFunctionalData object.

        Parameters
        ----------
        new_function : UnivariateFunctionalData or IrregularFunctionalData an
        object of class UnivariateFunctionalData or IrregularFunctionalData to
        add to the MultivariateFunctionalData object.

        Returns
        -------
        obj : object of class MultivariateFunctionalData

        """
        data = self.data
        data.append(new_function)
        return MultivariateFunctionalData(data)
