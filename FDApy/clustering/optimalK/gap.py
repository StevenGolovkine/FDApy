#!/usr/bin/env python
# -*-coding:utf8 -*

"""Module for the representation of the Gap statistic.

This module is used to represent the Gap statistic as an object.

References
----------
    Estimating the number of clusters in a data setp via the gap statistic,
    Tibshirani R., Walther G., and Hastie T., J. R. Statist. Soc. B (2001)
    63, Part 2, pp.411-423

    A comparison of Gap statistic definitions with and without logarithm
    function, Mohajer M., Englmeier K.-H., Schmid V. J., 2010

    Granger M. - https://github.com/milesgranger/gap_statistic

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from typing import Callable, Iterable, NamedTuple


###############################################################################
# Utility functions
def _compute_dispersion(
        data: np.ndarray,
        labels: np.ndarray,
        metric: str = 'euclidean'
) -> float:
    r"""Compute the dispersion of a given dataset.

    Notes
    -----
    ..math::

        W_k = \sum_{r=1}^k \frac{1}{2n_r}\sum_{i, i^\prime \in C_r}
        d_{ii^\prime}

    Parameters
    ----------
    data: np.ndarray, shape=(n, p)
        The data
    labels: np.ndarray, shape=(n,)
        The label of each observation
    metric: str, default='euclidean'
        The metric used for the computation of the distance between the
        observations. See `sklearn.metrics.pairwise_distance` documentation
        for the list of available metric function.

    Returns
    -------
    result: np.ndarray
        The dispersion of the points in their cluster.

    """
    distances = [pairwise_distances(data[labels == i], metric=metric)
                 for i in np.unique(labels)]
    return np.sum([np.sum(d) / (2 * d.shape[0]) for d in distances])


def _generate_uniform(
    data: np.ndarray,
    n_obs: int,
    a: float = 0.0,
    b: float = 1.0
) -> np.ndarray:
    """Generate `n_obs` data according to a uniform distribution."""
    return np.random.uniform(a, b, (n_obs, data.ndim))


def _generate_pca(
    data: np.ndarray,
    n_obs: int,
    a: float = 0.,
    b: float = 1.
) -> np.ndarray:
    """Generate data sccording to a uniform distribution after PCA."""
    data_centered = data - np.mean(data, axis=0)
    _, _, eigenvec = np.linalg.svd(data_centered)
    data_prime = np.matmul(data_centered, eigenvec)
    data_transfo_prime = _generate_uniform(data_prime, n_obs, a, b)
    data_transfo = np.matmul(data_transfo_prime, eigenvec.transpose())
    return data_transfo + np.mean(data, axis=0)


def _clustering(
    data: np.ndarray,
    n_clusters: int,
    **clusterer_kwargs
):
    """Cluster algorithm for Gap computation."""
    return KMeans(n_clusters=n_clusters, **clusterer_kwargs).fit_predict(data)


##############################################################################
# Class GapResult

class GapResult(NamedTuple):
    """An object containing the Gap statistic for given dataset and k."""

    value: float
    k: int
    sk: float
    value_star: float
    sk_star: float

    def __repr__(self) -> str:
        """Override print function."""
        return (f"For {self.k} clusters considered, the Gap statistic is"
                f" {self.value} ({self.sk}).")


##############################################################################
# Class Gap

class Gap():
    """A module for the computation of the Gap statistic.

    This module is used to compute the Gap statistic for a given dataset and
    a given number of cluster k. The Gap statistic is defined in the article
    of Tibshirani, Walther and Hastie - Estimating the number of clusters in a
    data set via the gap statistic.

    Attributes
    ----------
    n_clusters: int
        Best number of clusters found
    gap_df: pd.DataFrame
        Gap value for different values of n_clusters.

    """

    def __init__(
        self,
        n_jobs: int = -1,
        parallel_backend: str = "multiprocessing",
        clusterer: Callable = None,
        clusterer_kwargs: dict = None,
        generating_process: str = 'pca',
        metric: str = 'euclidean'
    ) -> None:
        """Initialize Gap object.

        Parameters
        ----------
        n_jobs: int, default=-1
            Number of cores to use in case of multiprocessing. If -1, it will
            use all the cores.
        parallel_backend: str, default='multiprocessing'
            Parallel backend used for the computation.
        clusterer: Callable, default=None
            An user-provided function for the clustering of the dataset. The
            function has to be compliant with sklearn clutering class and
            return only the labels such as the `predict` function.
        clusterer_kwargs: dict, default=None
            The parameters to be used by the clustering function.
        generating_process: str, default='pca'
            The generating process of the data for the reference datasets. One
            of `uniform` or `pca`.
        metric: str, default='euclidean'
            The metric used to compute distance between the observations.

        """
        self.parallel_backend = (parallel_backend
                                 if parallel_backend == 'multiprocessing'
                                 else None)
        self.n_jobs = n_jobs if 1 <= n_jobs <= cpu_count() else cpu_count()
        self.n_jobs = 1 if self.parallel_backend is None else self.n_jobs
        self.clusterer = clusterer if clusterer is not None else _clustering
        self.clusterer_kwargs = (
            clusterer_kwargs or dict()
            if clusterer is not None
            else dict(init='k-means++', n_init=10)
        )
        if generating_process == 'uniform':
            self.generate_process = _generate_uniform
        elif generating_process == 'pca':
            self.generate_process = _generate_pca
        else:
            raise ValueError("The generating process for the reference data"
                             " have to be 'uniform' or 'pca'.")
        self.metric = metric if metric is not None else 'euclidean'

    def __str__(self):
        """Override __str__ function."""
        return (f'Gap(n_jobs={self.n_jobs}, parallel'
                f' backend={self.parallel_backend})')

    def __repr__(self):
        """Override __repr__ function."""
        return self.__str__()

    def __call__(
        self,
        data: np.ndarray,
        cluster_array: Iterable[int] = (),
        n_refs: int = 3
    ) -> int:
        """Compute the Gap statistic.

        Parameters
        ----------
        data: np.ndarray, shape=(n, p)
            The data as an array of shape (n, p).
        cluster_array: Iterable[int]
            Represents the number of clusters to try on the data.
        n_refs: int, default=3
            Number of random reference datasets to consider.

        Returns
        -------
        n_clusters: int
            Best number of clusters found in the data according to the Gap
            statistic.

        """
        if self.parallel_backend == 'multiprocessing':
            engine = self._process_with_multiprocessing
        else:
            engine = self._process_non_parallel

        # Compute Gap stat for each cluster count.
        gap_df = pd.DataFrame({'n_clusters': [],
                               'gap_value': [],
                               'sk': [],
                               'gap_value_star': [],
                               'sk_star': []})
        for gap_results in engine(data, n_refs, cluster_array):
            gap_df = gap_df.append(
                {
                    'n_clusters': int(gap_results.k),
                    'gap_value': gap_results.value,
                    'sk': gap_results.sk,
                    'gap_value_star': gap_results.value_star,
                    'sk_star': gap_results.sk_star
                }, ignore_index=True)
            gap_df["gap_k+1"] = gap_df["gap_value"].shift(-1)
            gap_df["gap_star_k+1"] = gap_df["gap_value_star"].shift(-1)
            gap_df["sk+1"] = gap_df["sk"].shift(-1)
            gap_df["sk_star+1"] = gap_df["sk_star"].shift(-1)
            gap_df["diff"] = (
                gap_df["gap_value"] - gap_df["gap_k+1"] + gap_df["sk+1"])
            temp = gap_df["gap_value_star"] - gap_df["gap_star_k+1"]
            gap_df["diff_star"] = temp + gap_df["sk_star+1"]

        gap_df.drop(
            labels=["gap_k+1", "gap_star_k+1", "sk+1", "sk_star+1"],
            axis=1,
            inplace=True,
            errors="ignore",
        )

        self.gap_df = gap_df.sort_values(by="n_clusters", ascending=True).\
            reset_index(drop=True)
        self.n_clusters = int(
            self.gap_df.loc[np.argmax(self.gap_df.gap_value.values)].n_clusters
        )
        return self.n_clusters

    def plot(self, axes=None, scatter_args=None, **plt_kwargs):
        """Plot the results of the Gap computation.

        Parameters
        ----------
        axes: matplotlib.axes._subplots.AxesSubplot
            Axes object onto which the objects are plotted.
        scatter_args: dict
            Keywords scatter plot arguments
        **plt_kwargs:
            Keywords plotting arguments

        Returns
        -------
        axes: matplotlib.axes._subplots.AxesSubplot
            Axes object containing the graphs.

        """
        if axes is None:
            _, axes = plt.subplots(2, 2)

        gap_df = self.gap_df

        # Gap value plots
        axes[0, 0].errorbar(gap_df['n_clusters'],
                            gap_df['gap_value'],
                            gap_df['sk'], **plt_kwargs)
        axes[0, 0].scatter(gap_df[gap_df['n_clusters'] == 3].n_clusters,
                           gap_df[gap_df['n_clusters'] == 3].gap_value,
                           c='r', **scatter_args)
        axes[0, 0].set_title('Gap value per cluster count')
        axes[0, 0].set_xlabel('Cluster count')
        axes[0, 0].set_ylabel(r'Gap($k$)')

        axes[0, 1].plot(gap_df['n_clusters'],
                        gap_df['diff'], **plt_kwargs)
        axes[0, 1].set_title(r'Diff value per cluster count')
        axes[0, 1].set_xlabel(r'$k$')
        axes[0, 1].set_ylabel(r'Gap($k$) - Gap($k+1$) + $s_{k+1}$')

        # Gap_star value plots
        axes[1, 0].errorbar(gap_df['n_clusters'],
                            gap_df['gap_value_star'],
                            gap_df['sk_star'], **plt_kwargs)
        axes[1, 0].scatter(gap_df[gap_df['n_clusters'] == 3].n_clusters,
                           gap_df[gap_df['n_clusters'] == 3].gap_value_star,
                           c='r', **scatter_args)
        axes[1, 0].set_title(r'Gap$^\star$ value per cluster count')
        axes[1, 0].set_xlabel(r'$k$')
        axes[1, 0].set_ylabel(r'Gap$^\star$($k$)')

        axes[1, 1].plot(gap_df['n_clusters'],
                        gap_df['diff_star'], **plt_kwargs)
        axes[1, 1].set_title(r'Diff$^\star$ value per cluster count')
        axes[1, 1].set_xlabel(r'$k$')
        axes[1, 1].set_ylabel(
            r'Gap$^\star$($k$) - Gap$^\star$($k+1$) + $s_{k+1}$')

        return axes

    def _compute_gap(
        self,
        data: np.ndarray,
        n_clusters: int,
        n_refs: int,
        metric: str = 'euclidean'
    ) -> GapResult:
        """Compute the Gap statistic.

        Parameters
        ----------
        data: np.ndarray
            The data to cluster.
        n_clusters: int
            Number of clusters to test.
        n_refs: int
            Number of reference dataset to generate.
        metric: str, default='euclidean'
            The metric used to compute the Gap statistic.

        Returns
        -------
        results: GapResult
            The results as a GapResult object.

        """
        n_obs = np.ma.size(data, 0)
        a, b = data.min(axis=0), data.max(axis=0)

        # Generate the reference distributions and compute dispersions
        ref_dispersions = np.zeros(n_refs)
        for idx in range(n_refs):
            data_gen = self.generate_process(data, n_obs, a, b)
            labels = self.clusterer(
                data_gen, n_clusters, **self.clusterer_kwargs)
            ref_dispersions[idx] = _compute_dispersion(
                data_gen, labels, metric)

        # Compute dispersion the original dataset
        labels = self.clusterer(data, n_clusters, **self.clusterer_kwargs)
        dispersion = _compute_dispersion(data, labels, metric)

        # Compute Gap statistic
        log_dispersion = np.log(dispersion)
        log_ref_dispersion = np.mean(np.log(ref_dispersions))
        gap_value = log_ref_dispersion - log_dispersion
        sk = np.sqrt(1 + 1 / n_refs) * np.std(np.log(ref_dispersions))

        # Compute Gap star statistic
        gap_star_value = np.mean(ref_dispersions) - dispersion
        sk_star = np.sqrt(1 + 1 / n_refs) * np.std(ref_dispersions)

        return GapResult(gap_value, n_clusters, sk, gap_star_value, sk_star)

    def _process_with_multiprocessing(
        self,
        data: np.ndarray,
        n_refs,
        cluster_array: Iterable[int]
    ):
        """Compute Gap stat with multiprocessing parallelization."""
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            jobs = [executor.submit(
                    self._compute_gap, data, n_clusters, n_refs, self.metric)
                    for n_clusters in cluster_array
                    ]
            for future in as_completed(jobs):
                yield future.result()

    def _process_non_parallel(
        self,
        data: np.ndarray,
        n_refs: int,
        cluster_array: Iterable[int]
    ):
        """Compute Gap stat without parallelization."""
        for gap_results in [
            self._compute_gap(data, n_clusters, n_refs, self.metric)
            for n_clusters in cluster_array
        ]:
            yield gap_results
