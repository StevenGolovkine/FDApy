#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Bayesian Information Criterion
------------------------------

"""
import pandas as pd
import numpy as np
import numpy.typing as npt

from typing import Iterable, NamedTuple, Optional

from concurrent.futures import ProcessPoolExecutor, as_completed
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

    n_cluster: np.int64
    value: np.float64

    def __repr__(self) -> str:
        """Override print function."""
        return f"Number of clusters: {self.n_cluster} - BIC: {self.value}"


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
    n_jobs: np.int64, default=-1
        Number of cores to use in case of multiprocessing. If ``n_jobs==-1``,
        all the available cores are used.
    parallel_backend: np.str_, default='multiprocessing'
        Parallel backend used for the computation.

    Attributes
    ----------
    n_clusters: np.int64
        Best number of clusters found
    bic_df: pd.DataFrame
        BIC value for different values of n_clusters.

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
        n_jobs: np.int64 = -1,
        parallel_backend: np.str_ = 'multiprocessing'
    ) -> None:
        """Initialize BIC object."""
        self.parallel_backend = (parallel_backend
                                 if parallel_backend == 'multiprocessing'
                                 else None)
        self.n_jobs = n_jobs if 1 <= n_jobs <= cpu_count() else cpu_count()
        self.n_jobs = 1 if self.parallel_backend is None else self.n_jobs

    def __str__(self) -> str:
        """Override __str__ function."""
        return (
            f'BIC(n_jobs={self.n_jobs}, parallel_backend='
            f'{self.parallel_backend})'
        )

    def __repr__(self) -> str:
        """Override __repr__ function."""
        return self.__str__()

    def __call__(
        self,
        data: np.array,
        cluster_array: Iterable[int] = ()
    ) -> int:
        """Compute the BIC statistic.

        Parameters
        ----------
        data: np.array, shape=(n, p)
            The data as an array of shape (n, p).
        cluster_array: Iterable[int]
            Represents the number of clusters to try on the data.

        Returns
        -------
        n_clusters: int
            Best number of clusters found in the data according to the BIC
            statistic.

        """
        if self.parallel_backend == 'multiprocessing':
            engine = self._process_with_multiprocessing
        else:
            engine = self._process_non_parallel

        # Compute BIC stat for each cluster count
        bic_df = pd.DataFrame({'n_clusters': [],
                               'bic_value': []})
        for bic_results in engine(data, cluster_array):
            bic_df = bic_df.append(
                {
                    'n_clusters': int(bic_results.k),
                    'bic_value': bic_results.value
                }, ignore_index=True)

        self.bic_df = bic_df.sort_values(by="n_clusters", ascending=True).\
            reset_index(drop=True)
        self.n_clusters = int(
            self.bic_df.loc[np.argmin(self.bic_df.bic_value.values)].n_clusters
        )
        return self.n_clusters

    def _process_with_multiprocessing(
        self,
        data: np.array,
        cluster_array: Iterable[int]
    ) -> _BICResult:
        """Compute BIC stat with multiprocessing parallelization."""
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            jobs = [executor.submit(
                    _compute_bic, data, n_clusters)
                    for n_clusters in cluster_array
                    ]
            for future in as_completed(jobs):
                yield future.result()

    def _process_non_parallel(
        self,
        data: np.array,
        cluster_array: Iterable[int]
    ) -> _BICResult:
        """Compute BIC stat without parallelization."""
        if self.parallel_backend is not None:
            raise ValueError('Parallel backend have to be None.')
        for gap_results in [
            _compute_bic(data, n_clusters)
            for n_clusters in cluster_array
        ]:
            yield gap_results
