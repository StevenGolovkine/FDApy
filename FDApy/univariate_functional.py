#!/usr/bin/python3.7
# -*-coding:utf8 -*

import itertools
import numpy as np

import FDApy

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
        #print('argvals is converted into one dimensional list!')
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

def _check_argvals_values(argvals, values):
    """Check the compatibility of argvals and values. 

    Parameters
    ----------
    argvals : list of tuples
        List of tuples containing the sample points. 
    values : numpy.ndarray
        Numpy array containing the values. 

    Return
    ------
    True, if the argvals and the values are ok.
    """

    if len(argvals) != len(values.shape[1:]):
        raise ValueError(
            'argvals and values elements have different support dimensions!')
    if tuple(len(i) for i in argvals) != values.shape[1:]:
        raise ValueError(
            'argvals and values have different numbers of sampling points!')

    return True

def _check_argvals_equality(argvals1, argvals2):
    """Check if two provided `argvals` are equals.

    Parameters
    ----------
    argvals1 : list of tuples
    argvals2 : list of tuples

    Return
    ------
    True, if the two argvals are the same.
    """
    if argvals1 != argvals2:
        raise ValueError('The two UnivariateFunctionalData objects are defined on different argvals.')
    return True


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

    standardize : boolean, default = True
        Do we standardize the argvals to be in [0, 1].

    Attributes
    ----------

    Notes
    -----

    References
    ----------

    """

    def __init__(self, argvals, values, standardize=True):

        self.argvals = argvals
        self.values = values

        if standardize:
            argvals_stand = []
            for argval in self.argvals:
                argvals_stand.append(tuple(
                    FDApy.utils.rangeStandardization_(argval)))
            self.argvals_stand = argvals_stand

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
                ", ... , " +\
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
            The observation(s) of the object to retrieve.

        Return
        ------
        res : UnivariateFunctionalData object
            The selected observation(s) as UnivariateFunctionalData object.  

        """
        argvals = self.argvals
        values = self.values[index]

        if len(values.shape) == len(argvals):
            values = np.array(values, ndmin=(len(argvals)+1))

        res = UnivariateFunctionalData(argvals, values)
        return res

    def __add__(self, new):
        if not isinstance(new, 
            FDApy.univariate_functional.UnivariateFunctionalData):
            raise ValueError('The object to add must be an object of the class UnivariateFunctionalData.')
        _check_argvals_equality(self.argvals, new.argvals)
        values = self.values + new.values
        res = UnivariateFunctionalData(self.argvals, values)
        return res

    def __sub__(self, new):
        if not isinstance(new,
            FDApy.univariate_functional.UnivariateFunctionalData):
            raise ValueError('The object to substract must be an object of the class UnivariateFunctionalData.')
        _check_argvals_equality(self.argvals, new.argvals)
        values = self.values - new.values
        res = UnivariateFunctionalData(self.argvals, values)
        return res

    def __mul__(self, obj):
        values = np.empty(shape=self.values.shape)
        if type(obj) in (int, float, np.int_, np.float_):
            values = self.values * obj
        elif (isinstance(obj, list)) and (self.nObs() == len(obj)):
            for i in np.arange(0, len(obj)):
                values[i, :] = self.values[i] * obj[i]
        else:
            raise ValueError('The multiplcation can not be performed! Not the right type!')

        res = UnivariateFunctionalData(self.argvals, values)
        return res

    def __rmul__(self, nb):
        return self * nb

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

    def asIrregularFunctionalData(self):
        """Coerce univariate functional data of dimension 1 into irregular functional data.

        Return
        ------
        obj : FDApy.irregular_functional.IrregularFunctionalData
            An object of the class FDApy.irregular_functional.IrregularFunctionalData
        """

        if self.dimension() != 1:
            raise ValueError('It is not possible to coerce a UnivariateFunctionalData as IrregularFunctionalData other than the ones with dimension 1!')

        argvals = []
        values = []
        # TODO: Add the case of NA values in an observations
        for row in self.values:
            argvals.append(self.argvals[0])
            values.append(np.array(row))
        return FDApy.irregular_functional.IrregularFunctionalData(
            argvals, values)

    def asMultivariateFunctionalData(self):
        """Coerce univariate functional data into mulivariate functional data with one function.

        Return
        ------
        obj : FDApy.mulivariate_functional.MultivariateFunctionalData
            An object of the class FDApy.mulivariate_functional.MultivariateFunctionalData
        """
        return FDApy.multivariate_functional.MultivariateFunctionalData(
            [self])

    def mean(self):
        """Compute the pointwise mean function.

        Return
        ------
        obj : FDApy.univariate_functional.UnivariateFunctionalData object
            Object of the class FDApy.univariate_functional.UnivariateFunctionalData with the same argvals as self and one observation.

        """
        mean_ = FDApy.utils.rowMean_(self.values)
        return FDApy.univariate_functional.UnivariateFunctionalData(
            self.argvals, np.array(mean_, ndmin=2))

    def covariance(self):
        """Compute the pointwise covariance function.
        
        Return
        ------
        obj : FDApy.univariate_functional.UnivariateFunctionalData object
            Object of the class FDApy.univariate_functional.UnivariateFunctionalData with dimension 2 and one observation.
        """
        if self.dimension() != 1:
            raise ValueError(
                'Only one dimensional functional data are supported!')

        new_argvals = [self.argvals[0], self.argvals[0]]
        X = self - self.mean()
        cov = np.dot(X.values.T, X.values) / (self.nObs() - 1)
        return UnivariateFunctionalData(new_argvals, np.array(cov, ndmin=3))

    def integrate(self, method='simpson'):
        """Integrate all the observations over the argvals.

        Parameters
        ----------
        method : str, default = 'simpson'
            The method used to integrated. Currently, only the Simpsons method is implemented.

        Return
        ------
        obj : list of int
            List where entry i is the integration of the observation i over the argvals.

        Note
        ----
        Only work with 1-dimensional functional data.
        """
        if method is not 'simpson':
            raise ValueError('Only the Simpsons method is implemented!')
        return [FDApy.utils.integrate_(self.argvals[0], i) for i in self.values]

    def tensorProduct(self, data):
        """Compute the tensor product of two univariate functional data.

        Parameter
        ---------
        data : FDApy.univariate_functional.UnivariateFunctionalData object
            A one dimensional univariate functional data

        Return
        ------
        obj : FDApy.univariate_functional.UnivariateFunctionalData
            Object of the class FDApy.univariate_functional.UnivariateFunctionalData of dimension 2 (self.argvals x data.argvals)
        """
        if (self.dimension() != 1) or (data.dimension() != 1):
            raise ValueError(
                'Only one dimensional functional data are supported!')

        new_argvals = [self.argvals[0], data.argvals[0]]
        new_values = [FDApy.utils.tensorProduct_(i, j) 
            for i in self.values for j in data.values]
        return UnivariateFunctionalData(new_argvals, np.array(new_values))

    def smooth(self, points, kernel="gaussian", bandwith=0.05, degree=2):
        """Smooth the data.

        Currently, it uses local polynomial regression.
        Currently, only for one dimensional univariate functional data.
        TODO: Add other smoothing methods.

        Parameters
        ----------
        points : array-like, shape = [n_samples]
            Points where evaluate the function.
        kernel : string, default="gaussian"
            Kernel name used as weight (default = 'gaussian').
        bandwith : float, default=0.05
            Strictly positive. Control the size of the associated neighborhood. 
        degree: integer, default=2
            Degree of the local polynomial to fit.

        Return
        ------
        res : FDApy.univariate_functional.UnivariateFunctionalData
            An object of the class FDApy.univariate_functional.UnivariateFunctionalData which correpond to the data that have been smooth::
                argvals = `points` given as input

        """
        if self.dimension() != 1:
            raise ValueError('Only 1-dimensional univariate functional data can be smoothed!')

        # Define the smoother
        lp = FDApy.local_polynomial.LocalPolynomial(
                kernel=kernel,
                bandwith=bandwith,
                degree=degree)

        points_ = np.array(points)
        pred = np.empty([self.nObs(), len(points_)])
        count = 0
        for row in self:
            argvals = [i for i in itertools.chain.from_iterable(row.argvals)]
            values = [i for i in itertools.chain.from_iterable(row.values)]
            lp.fit(argvals, values)
            pred[count] = lp.predict(points_)
            count += 1

        res = UnivariateFunctionalData(points, pred)

        return res
