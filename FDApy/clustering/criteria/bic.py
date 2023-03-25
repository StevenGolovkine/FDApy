#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Bayesian Information Criterion
------------------------------

"""
import pandas as pd
import numpy as np
import numpy.typing as npt

from typing import Generator, Iterable, NamedTuple, Optional

from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from sklearn.mixture import GaussianMixture


###############################################################################
# Class BICResult

class _BICResult(NamedTuple):
    """Class that contains BIC results.

    Attributes
    ----------
    n_cluster: np.int64
        Number of clusters for which the BIC is computed.
    value: np.float64
        Value of the BIC.

    """

    n_clusters: np.int64
    value: np.float64

    def __repr__(self) -> str:
        """Override print function."""
        return f"Number of clusters: {self.n_clusters} - BIC: {self.value}"


###############################################################################
# Utility functions

def _compute_bic(
    data: npt.NDArray[np.float64],
    n_clusters: np.int64,
    random_state: Optional[np.int64] = None
) -> _BICResult:
    """Compute the BIC statistic.

    The BIC is computed after fitting a Gaussian mixture model of the dataset.
    It uses the implementation of ``sklearn.mixture.Gaussian`` to do so. The
    BIC is defined in [1]_.

    Parameters
    ----------
    data: npt.NDArray[np.float64]
        The data to be clustered.
    n_clusters: np.int64
        Number of clusters.
    random_state: Optional[np.int64], default=None
        Controls the random seed given to the ``GaussianMixture`` method chosen
        to initialize the parameters.

    Returns
    -------
    _BICResult
        An estimation of the BIC.

    References
    ----------
    .. [1] Schwarz G. (1978), Estimating the dimension of a model, Annals of
        Statistics, 6(2), 461--464.

    """
    model = GaussianMixture(n_clusters, random_state=random_state)
    model.fit(data)
    return _BICResult(n_clusters, model.bic(data))


###############################################################################
# Class BIC

class BIC():
    r"""Bayesian Information Criterion (BIC).

    This module computes the BIC [1]_ for a given dataset and number of
    clusters ``n_clusters``. The BIC is given by

    .. math::

        BIC = -2\log(L) + \log(N)d,

    where :math:`L` is an estimation of the likelihood of the model, :math:`d`
    is the number of parameters of the model and :math:`N` is the number of
    samples.

    Parameters
    ----------
    n_jobs: np.int64, default=multiprocessing.cpu_count()
        Number of cores to use in case of multiprocessing. The default is the
        number of available cores, determined by
        ``multiprocessing.cpu_count()``. If ``n_jobs < 1``, it will be set to 1
        and if ``n_jobs > cpu_count()``, it will be set to ``cpu_count()``.
    parallel_backend: np.str_, default='multiprocessing'
        Parallel backend used for the computation.

    Attributes
    ----------
    n_clusters: np.int64
        Best number of clusters found. It is defined as the number of clusters
        that minimize the value of the BIC (according to the definition).
    bic: pd.DataFrame
        BIC values for different values of n_clusters.

    Notes
    -----
    It uses the implementation of the BIC in the module
    ``sklearn.mixture.GaussianMixture``.

    References
    ----------
    .. [1] Schwarz G. (1978), Estimating the dimension of a model, Annals of
        Statistics, 6(2), 461--464.

    """

    def __init__(
        self,
        n_jobs: np.int64 = cpu_count(),
        parallel_backend: np.str_ = 'multiprocessing'
    ) -> None:
        """Initialize BIC object."""
        if parallel_backend is None:
            self.parallel_backend, self.n_jobs = parallel_backend, 1
        elif parallel_backend == 'multiprocessing':
            self.parallel_backend = parallel_backend
            self.n_jobs = min(max(1, n_jobs), cpu_count())
        else:
            raise ValueError(
                "The parallel backend has to be 'multiprocessing' or None."
            )

    def __str__(self) -> np.str_:
        """Override __str__ function."""
        return (
            f'BIC(n_jobs={self.n_jobs}, parallel_backend='
            f'{self.parallel_backend})'
        )

    def __repr__(self) -> np.str_:
        """Override __repr__ function."""
        return self.__str__()

    def __call__(
        self,
        data: npt.NDArray[np.float64],
        n_clusters: Iterable[np.int64]
    ) -> np.int64:
        """Compute the BIC statistic.

        Parameters
        ----------
        data: npt.NDArray[np.float64], shape=(n_obs, n_components)
            Data as an array of shape (n_obs, n_components).
        n_clusters: Iterable[np.int64]
            The different number of clusters to try.

        Returns
        -------
        np.int_
            Returns the number of clusters that minimizes the BIC.

        """
        if self.parallel_backend == 'multiprocessing':
            engine = self._process_with_multiprocessing
        else:
            engine = self._process_non_parallel

        # Compute BIC for each cluster count
        bic_df = pd.DataFrame.from_records(
            engine(data, n_clusters), columns=['n_clusters', 'value']
        )

        self.bic = bic_df.sort_values(by='n_clusters')
        self.n_clusters = bic_df.loc[bic_df['value'].idxmin(), 'n_clusters']
        return self.n_clusters

    def _process_with_multiprocessing(
        self,
        data: npt.NDArray[np.float64],
        cluster_array: Iterable[np.int64]
    ) -> Generator[_BICResult, None, None]:
        """Compute BIC with multiprocessing parallelization.

        Parameters
        ----------
        data: npt.NDArray[np.float64], shape=(n_obs, n_components)
            Data as an array of shape (n_obs, n_components).
        cluster_array: Iterable[np.int64]
            The different number of clusters to try.

        Returns
        -------
        Generator[_BICResult]
            Generator that contains the BIC for each number of clusters.

        """
        print('Coucou, I am running multiprocessing!')
        for result in Parallel(n_jobs=self.n_jobs)(delayed(_compute_bic)(
            data, n_clusters) for n_clusters in cluster_array
        ):
            yield result
        #with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        #    jobs = [
        #        executor.submit(_compute_bic, data, n_clusters)
        #        for n_clusters in cluster_array
        #    ]
        #    print(jobs)
        #print(jobs)
        #return (future.result() for future in as_completed(jobs))

    def _process_non_parallel(
        self,
        data: npt.NDArray[np.float64],
        cluster_array: Iterable[np.int64]
    ) -> Generator[_BICResult, None, None]:
        """Compute BIC without parallelization.

        Parameters
        ----------
        data: npt.NDArray[np.float64], shape=(n_obs, n_components)
            Data as an array of shape (n_obs, n_components).
        cluster_array: Iterable[np.int64]
            The different number of clusters to try.

        Returns
        -------
        Generator[_BICResult]
            Generator that contains the BIC for each number of clusters.

        """
        return (_compute_bic(data, n_clusters) for n_clusters in cluster_array)
