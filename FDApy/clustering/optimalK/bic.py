#!/usr/bin/env python
# -*-coding:utf8 -*

"""Module for the representation of the BIC statistic.

This module is used to represented the BIC statistic as an object.

References
----------
    Gideon E. Schwarz, « Estimating the dimension of a model », Annals of
    Statistics, vol. 6, no 2,‎ 1978, p. 461-464

"""

import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from sklearn.mixture import GaussianMixture

from typing import Iterable, NamedTuple


###############################################################################
# Class BICResult

class BICResult(NamedTuple):
    """An object containing the BIC statistic for given dataset and k."""

    value: float
    k: int

    def __repr__(self) -> str:
        """Override print function."""
        return (f"For {self.k} clusters considered, the BIC statistic is"
                f" {self.value}.")


###############################################################################
# Utility functions
def _compute_bic(
    data: np.array,
    n_clusters: int
) -> BICResult:
    """Compute the BIC statistic.

    Parameters
    ----------
    data: np.array
        The data to cluster.
    n_clusters: int
        Number of clusters to test.

    Returns
    -------
    results: BICResult
        The results as a BICResult object.

    """
    gm = GaussianMixture(n_clusters)
    gm.fit(data)
    return BICResult(gm.bic(data), n_clusters)


###############################################################################
# Class BIC

class BIC():
    """A module for the computation of the BIC statistic.

    This module is used to compute the BIC statistic for a given dataset and
    a given number of cluster k.

    Attributes
    ----------
    n_clusters: int
        Best number of clusters found
    bic_df: pd.DataFrame
        BIC value for different values of n_clusters.

    """

    def __init__(
        self,
        n_jobs: int = -1,
        parallel_backend: str = "multiprocessing"
    ) -> None:
        """Initialize BIC object.

        Parameters
        ----------
        n_jobs: int, default=-1
            Number of cores to use in case of multiprocessing. If -1, it will
            use all the cores.
        parallel_backend: str, default='multiprocessing'
            Parallel backend used for the computation.

        """
        self.parallel_backend = (parallel_backend
                                 if parallel_backend == 'multiprocessing'
                                 else None)
        self.n_jobs = n_jobs if 1 <= n_jobs <= cpu_count() else cpu_count()
        self.n_jobs = 1 if self.parallel_backend is None else self.n_jobs

    def __str__(self):
        """Override __str__ function."""
        return (f"BIC(n_jobs={self.n_jobs}, parallel_backend="
                f"{self.parallel_backend})")

    def __repr__(self):
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
    ):
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
    ):
        """Compute BIC stat without parallelization."""
        for gap_results in [
            _compute_bic(data, n_clusters)
            for n_clusters in cluster_array
        ]:
            yield gap_results
