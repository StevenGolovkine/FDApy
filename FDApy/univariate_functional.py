#!/usr/bin/python3.7
# -*-coding:utf8 -*

"""
Module for UnivariateFunctionalData classes.

This module is used define univariate functional data.
"""
import itertools
import numpy as np
import pygam

from .irregular_functional import IrregularFunctionalData

from .local_polynomial import LocalPolynomial
from .utils import (integrate_, integrationWeights_,
                    rangeStandardization_, rowMean_,
                    tensorProduct_)


###############################################################################
# Checkers used by the UnivariateFunctionalData class.


def _check_argvals(argvals):
    """Check the user provided `argvals`.

    Parameters
    ----------
    argvals: list of numpy.ndarray
        A list of numeric vectors (numpy.ndarray) or a single numeric vector
        (numpy.ndarray) giving the sampling points in the domains.

    Returns
    -------
    list of numpy.ndarray

    """
    if not isinstance(argvals, (np.ndarray, list)):
        raise ValueError(
            'argvals has to be a list of numpy.ndarray or a numpy.ndarray!')
    if isinstance(argvals, list) and \
            not all([isinstance(i, np.ndarray) for i in argvals]):
        raise ValueError(
            'argvals has to be a list of numpy.ndarray or a numpy.ndarray!')
    if isinstance(argvals, np.ndarray):
        argvals = [argvals]

    # Check if all entries of `argvals` are numeric.
    argvals_ = list(itertools.chain.from_iterable(argvals))
    if not all([isinstance(i, (int, float, np.int_, np.float_))
                for i in argvals_]):
        raise ValueError(
            'All argvals elements must be numeric!')

    return argvals


def _check_values(values):
    """Check the user provided `values`.

    Parameters
    ----------
    values : numpy.array
        A numpy array containing values.

    Returns
    -------
    values : numpy array

    """
    # TODO: Modify the function to deal with other types of data.
    if not isinstance(values, np.ndarray):
        raise ValueError(
            'values has to be a numpy.ndarray!')

    return values


def _check_argvals_values(argvals, values):
    """Check the compatibility of argvals and values.

    Parameters
    ----------
    argvals : list of tuples
        List of tuples containing the sample points.
    values : numpy.ndarray
        Numpy array containing the values.

    Returns
    -------
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
        First argument
    argvals2 : list of tuples
        Second argument

    Returns
    -------
    True, if the two argvals are the same.

    """
    if argvals1 != argvals2:
        raise ValueError(
            """The two UnivariateFunctionalData objects are defined
                on different argvals.""")
    return True


#############################################################################
# Class UnivariateFunctionalData


class UnivariateFunctionalData(object):
    """An object for defining Univariate Functional Data."""

    def __init__(self, argvals, values, standardize=True):
        """Initialize UnivariateFunctionalData object.

        Parameters
        ----------
        argvals : list of numpy.ndarray
            A list of numeric vectors (numpy.ndarray) or a single numeric
            vector (numpy.ndarray) giving the sampling points in the domains.

        values : array-like
            An array, giving the observed values for N observations.
            Missing values should be included via `None` (or `np.nan`). The
            shape depends on `argvals`::

                (N, M) if `argvals` is a single numeric vector,
                (N, M_1, ..., M_d) if `argvals` is a list of numeric vectors.

        standardize : boolean, default = True
            Do we standardize the argvals to be in [0, 1].

        """
        self.argvals = argvals
        self.values = values

        if standardize:
            argvals_stand = []
            for argval in self.argvals:
                argvals_stand.append(rangeStandardization_(argval))
            self.argvals_stand = argvals_stand

    def __repr__(self):
        """Override print function."""
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
        """Function call when self[index].

        Parameters
        ----------
        index : int
            The observation(s) of the object to retrieve.

        Returns
        -------
        res : UnivariateFunctionalData object
            The selected observation(s) as UnivariateFunctionalData object.

        """
        argvals = self.argvals
        values = self.values[index]

        if len(values.shape) == len(argvals):
            values = np.array(values, ndmin=(len(argvals) + 1))

        res = UnivariateFunctionalData(argvals, values)
        return res

    def __add__(self, new):
        """Override add function."""
        if not isinstance(new, UnivariateFunctionalData):
            raise ValueError(
                """The object to add must be an object of the class
                    UnivariateFunctionalData.""")
        _check_argvals_equality(self.argvals, new.argvals)
        values = self.values + new.values
        res = UnivariateFunctionalData(self.argvals, values)
        return res

    def __sub__(self, new):
        """Override sub funcion."""
        if not isinstance(new, UnivariateFunctionalData):
            raise ValueError(
                """The object to substract must be an object of the class
                    UnivariateFunctionalData.""")
        _check_argvals_equality(self.argvals, new.argvals)
        values = self.values - new.values
        res = UnivariateFunctionalData(self.argvals, values)
        return res

    def __mul__(self, obj):
        """Overide mul function."""
        values = np.empty(shape=self.values.shape)
        if isinstance(obj, (int, float, np.int_, np.float_)):
            values = self.values * obj
        elif (isinstance(obj, list)) and (self.nObs() == len(obj)):
            for i in np.arange(0, len(obj)):
                values[i, :] = self.values[i] * obj[i]
        else:
            raise ValueError(
                """The multiplcation can not be performed!
                    Not the right type!""")

        res = UnivariateFunctionalData(self.argvals, values)
        return res

    def __rmul__(self, nb):
        """Override rmul function."""
        return self * nb

    @property
    def argvals(self):
        """Getter for argvals."""
        return self._argvals

    @argvals.setter
    def argvals(self, new_argvals):
        new_argvals = _check_argvals(new_argvals)
        if hasattr(self, 'values'):
            _check_argvals_values(new_argvals, self.values)
        self._argvals = new_argvals

    @property
    def argvals_stand(self):
        """Getter for argvals_stand."""
        return self._argvals_stand

    @argvals_stand.setter
    def argvals_stand(self, new_argvals_stand):
        self._argvals_stand = new_argvals_stand

    @property
    def values(self):
        """Getter for values."""
        return self._values

    @values.setter
    def values(self, new_values):
        new_values = _check_values(new_values)
        if hasattr(self, 'argvals'):
            _check_argvals_values(self.argvals, new_values)
        self._values = new_values

    @property
    def mean_(self):
        """Getter for mean_."""
        return self._mean_

    @mean_.setter
    def mean_(self, new_mean):
        self._mean_ = new_mean

    def nObs(self):
        """Number of observations of the object.

        Returns
        -------
        n : int
            Number of observations of the object.

        """
        n = len(self.values)
        return n

    def rangeObs(self):
        """Range of the observations of the objects.

        Returns
        -------
        min(values_), max(values_) : tuple
            Tuple containing the minimum and maximum number of all the
            observations for an object.

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

        Returns
        -------
        n : list of int
            List of the length self.dimension() where the i-th entry
            correspond to the number of sampling points of the i-th dimension
            of the observations.

        """
        n = [len(i) for i in self.argvals]
        return n

    def rangeObsPoint(self):
        """Range of the observations of the objects.

        Returns
        -------
        range_ : list of tuples containing the minimum and maximum number
        where the i-th entry of the list contains the range of the i-th
        dimension of the object.

        """
        range_ = [(min(i), max(i)) for i in self.argvals]
        return range_

    def dimension(self):
        """Common dimension of the observations of the object.

        Returns
        -------
        dim : int
            Number of dimension of the observations of the object.

        """
        dim = len(self.argvals)
        return dim

    def asIrregularFunctionalData(self):
        """Convert to univariate to irregular functional data.

        Coerce univariate functional data of dimension 1 into irregular
        functional data.

        Returns
        -------
        obj : IrregularFunctionalData
            An object of the class IrregularFunctionalData

        """
        if self.dimension() != 1:
            raise ValueError(
                """It is not possible to coerce a UnivariateFunctionalData as
                IrregularFunctionalData other than the ones with
                dimension 1!""")

        argvals = []
        values = []
        # TODO: Add the case of NA values in an observations
        for row in self.values:
            argvals.append(self.argvals[0])
            values.append(np.array(row))
        return IrregularFunctionalData(argvals, values)

    def mean(self, smooth=False, method='LocalLinear', **kwargs):
        """Compute the mean function.

        Parameters
        ----------
        smooth: boolean, default=False
            Should we smooth the mean?
        method: 'LocalLinear', 'GAM', default='LocalLinear'
            Which smoothing method do we use?
        **kwargs: dict
            The following parameters are taken into account:
            * For method='LocalLinear':
                - kernel: 'gaussian', 'epanechnikov', 'tricube', 'bisquare'
                    default='gaussian'
                - degree: int
                    default: 2
                - bandwidth: float
                    default=1
            * For method='GAM':
                - n_basis: int
                    default=10

        Returns
        -------
        obj : UnivariateFunctionalData object
            Object of the class UnivariateFunctionalData with the same
            argvals as self and one observation.

        """
        mean_ = rowMean_(self.values)
        if smooth:
            if method == 'LocalLinear':
                kernel = kwargs.get('kernel', 'gaussian')
                degree = kwargs.get('degree', 2)
                bandwidth = kwargs.get('bandwidth', 1)

                lp = LocalPolynomial(kernel=kernel,
                                     bandwidth=bandwidth,
                                     degree=degree)
                lp.fit(self.argvals, mean_)
                mean_ = lp.X_fit_
            elif method == 'GAM':
                n_basis = kwargs.get('n_basis', 10)

                X = np.array(self.argvals[0])

                mean_ = pygam.LinearGAM(pygam.s(0, n_splines=n_basis)).\
                    fit(X, mean_).\
                    predict(X)
            else:
                raise ValueError('Method not implemented!')

        self.mean_ = UnivariateFunctionalData(self.argvals,
                                              np.array(mean_, ndmin=2))

    def covariance(self, smooth=False, method='LocalLinear', **kwargs):
        """Compute the covariance function.

        Parameters
        ----------
        smooth: boolean, default=False
            Should we smooth the covariance?
        method: 'LocalLinear', 'GAM', default: 'LocalLinear'
            Which smoothing method do we use?
        **kwargs: dict
            The following parameters are taken into account:
            * For method='LocalLinear'
                - kernel: 'gaussian', 'epanechnikov', 'tricube', 'bisquare'
                    default='gaussian'
                - degree: int
                    default: 2
                - bandwidth: float
                    default=1
            * For method='GAM'
                - n_basis: int
                    default=10

        Returns
        -------
        obj : UnivariateFunctionalData object
            Object of the class UnivariateFunctionalData with dimension 2
            and one observation.

        References
        ----------
        Yao, Müller and Wang (2005), Functional Data Analysis for Sparse
        Longitudinal Data,
        Journal of the American Statistical Association, Vol. 100, No. 470

        Notes
        -----
        Currently, for smoothing, please consider to use the 'GAM' method as
        the 'LocalLinear' is not working.

        """
        if self.dimension() != 1:
            raise ValueError(
                'Only one dimensional functional data are supported!')

        new_argvals = [self.argvals[0], self.argvals[0]]
        if getattr(self, 'mean_', None) is None:
            self.mean(smooth, method, **kwargs)
        X = self - self.mean_
        cov = np.dot(X.values.T, X.values) / (self.nObs() - 1)
        diag = np.copy(np.diag(cov))

        if smooth:
            if method == 'LocalLinear':
                kernel = kwargs.get('kernel', 'gaussian')
                degree = kwargs.get('degree', 2)
                bandwidth = kwargs.get('bandwidth', 1)

                lp = LocalPolynomial(kernel=kernel,
                                     bandwidth=bandwidth,
                                     degree=degree)
                lp.fit(new_argvals, cov)
                cov = lp.X_fit_
            elif method == 'GAM':
                n_basis = kwargs.get('n_basis', 10)

                # Remove covariance diagonale because of measurement errors.
                np.fill_diagonal(cov, None)
                cov = cov[~np.isnan(cov)]

                # Define train and predict vector
                X = np.transpose(
                    np.vstack(
                        (np.repeat(new_argvals[0],
                                   repeats=len(new_argvals[0])),
                         np.tile(new_argvals[0],
                                 reps=len(new_argvals[0]))
                         )
                    )
                )
                X_train = X[X[:, 0] != X[:, 1], :]

                cov = pygam.LinearGAM(pygam.te(0, 1, n_splines=n_basis)).\
                    fit(X_train, cov).\
                    predict(X).\
                    reshape((len(new_argvals[0]), len(new_argvals[0])))
            else:
                raise ValueError('Method not implemented!')

        cov = (cov + cov.T) / 2

        # Estimation of sigma2 (Yao, Müller and Wang, 2005)
        D = np.asarray(self.argvals[0])
        # Local linear smoother focusing on diagonal values.
        lp = LocalPolynomial(kernel='gaussian',
                             degree=1,
                             bandwidth=kwargs.get('bandwidth', 1))
        lp.fit(D, diag)
        V_hat = lp.predict(D)

        # Staniswalis and Lee (1998)
        T_len = D[len(D) - 1] - D[0]
        T1_lower = np.sum(~(D >= (D[0] + 0.25 * T_len)))
        T1_upper = np.sum((D <= (D[len(D) - 1] - 0.25 * T_len)))
        W = integrationWeights_(D[T1_lower:T1_upper], method='trapz')

        nume = np.dot(W, (V_hat - np.diag(cov))[T1_lower:T1_upper])
        sigma2 = np.maximum(nume / (D[T1_upper] - D[T1_lower]), 0)

        self.covariance_ = UnivariateFunctionalData(new_argvals,
                                                    np.array(cov, ndmin=3))
        self.sigma2 = sigma2

    def estimate_noise(self):
        """Estimation of the noise.

        This method estimates the (heteroscedastic) noise for a univariate
        functional data object.
        Model: :math:`Z_i(t_k) = f_i(t_k) + sigma(f_i(t_k))epsilon_i`
        It is assume that all the curves have been sampled on the same design
        points. Let's t_1, ..., t_k be that points, the estimation of the
        noise at t_k is::
        :math:`sigma^2(t_k) = 1/2n sum_{i} [Z_i(t_{k+1}) - Z_i(t_{k})]^2`

        """
        if self.dimension() != 1:
            raise ValueError(
                'Only one dimensional functional data are supported!')

    def integrate(self, method='simpson'):
        """Integrate all the observations over the argvals.

        Parameters
        ----------
        method : str, default = 'simpson'
            The method used to integrated. Currently, only the Simpsons method
            is implemented.

        Returns
        -------
        obj : list of int
            List where entry i is the integration of the observation i over
            the argvals.

        Note
        ----
        Only work with 1-dimensional functional data.

        """
        if method != 'simpson':
            raise ValueError('Only the Simpsons method is implemented!')
        return [integrate_(self.argvals[0], i) for i in self.values]

    def tensorProduct(self, data):
        """Compute the tensor product of two univariate functional data.

        Parameter
        ---------
        data : FDApy.univariate_functional.UnivariateFunctionalData object
            A one dimensional univariate functional data

        Returns
        -------
        obj : FDApy.univariate_functional.UnivariateFunctionalData
            Object of the class UnivariateFunctionalData of dimension 2
            (self.argvals x data.argvals)

        """
        if (self.dimension() != 1) or (data.dimension() != 1):
            raise ValueError(
                'Only one dimensional functional data are supported!')

        new_argvals = [self.argvals[0], data.argvals[0]]
        new_values = [tensorProduct_(i, j) for i in self.values
                      for j in data.values]
        return UnivariateFunctionalData(new_argvals, np.array(new_values))

    def smooth(self, t0, k0,
               points=None, degree=0, kernel="epanechnikov", bandwidth=None):
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
        bandwidth : float, default=0.05
            Strictly positive. Control the size of the associated neighborhood.
        degree: integer, default=2
            Degree of the local polynomial to fit.

        Returns
        -------
        res : FDApy.univariate_functional.UnivariateFunctionalData
            Object of the class UnivariateFunctionalData which correpond to
            the data that have been smooth:: argvals = `points` given as input

        """
        if self.dimension() != 1:
            raise ValueError(
                """Only 1-dimensional univariate functional data can
                be smoothed!""")

        data = self.asIrregularFunctionalData()
        data_smooth = data.smooth(t0, k0,
                                  points=points, degree=degree,
                                  kernel=kernel, bandwidth=bandwidth)

        return data_smooth
