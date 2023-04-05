#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Gap Statistic
-------------

"""
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from matplotlib.axes import Axes

from typing import (
    Callable, Dict, Generator, Optional, Iterable, NamedTuple, Union
)
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from sklearn.cluster import KMeans


##############################################################################
# Class GapResult

class _GapResult(NamedTuple):
    """Class that contains Gap statistics.

    Attributes
    ----------
    n_clusters: np.int64
        Number of clusters for which the Gap is computed.
    log_value: np.float64
        Value of the Gap statistic computed as in [2]_.
    log_error: np.float64
        Simulation error computed as in [2]_.
    value: np.float64
        Value of the Gap statistic computed as in [1]_.
    error: np.float64
        Simulation error computed as in [1]_.

    References
    ----------
    .. [1] Mohajer M., Englmeier K.-H., and Schmid V. J., (2010), A comparison
        of Gap statistic definitions with and without logarithm function,
        Technical Report Number 096, Department of Statistics, University of
        Munich.
    .. [2] Tibshirani R., Walther G., and Hastie T. (2001), Estimating the
        number of clusters in a data setp via the gap statistic, Journal of the
        Royal Statistical Society, Series B, 63(2), 411--423.

    """

    n_clusters: np.int64
    log_value: np.float64
    log_error: np.float64
    value: np.float64
    error: np.float64

    def __repr__(self) -> str:
        """Override print function."""
        return f"Number of clusters: {self.n_clusters} - Gap: {self.log_value}"


###############################################################################
# Utility functions
def _compute_dispersion(
        data: npt.NDArray[np.float64],
        labels: npt.NDArray[np.float64],
        centroids: npt.NDArray[np.float64],
        metric: Optional[Union[np.str_, np.int64]] = None
) -> np.float64:
    r"""Compute the dispersion of a given dataset.

    ..math::

        W_k = \sum_{r=1}^k \frac{1}{2n_r}\sum_{i, i^\prime \in C_r}
        d_{ii^\prime}^2 = \sum_{r=1}^k \sum_{i \in C_r}
        d_{i\mu_r},

    where :math:`C_r` is the set of observation for the :math:`r`th cluster,
    :math:`d_{ii^\prime}` is some distance measure between the observations
    :math:`i` and :math:`i^\prime` and :math:`d_{i\mu_i}` is some distance
    between the observation :math:`i` and the the center of the cluster
    :math:`r`, :math:`\mu_r`.

    Parameters
    ----------
    data: npt.NDArray[np.float64], shape=(n_obs, n_features)
        Data.
    labels: npt.NDArray[np.float64], shape=(n_obs,)
        Label for each observation.
    centroids: npt.NDArray[np.float64]
        Center of the clusters.
    metric: Optional[Union[np.str_, np.int64]], default=None
        The metric used for the computation of the distance between the
        observations. See ``numpy.linalg.norm`` documentation
        for the list of available metric function.

    Returns
    -------
    np.float64
        The dispersion of the points in their cluster.

    """
    return np.sum([
        np.linalg.norm(obs - centroids[label], ord=metric)**2
        for obs, label in zip(data, labels)
    ])


def _generate_uniform(
    data: npt.NDArray[np.float64],
    runif: Callable = np.random.uniform
) -> npt.NDArray[np.float64]:
    """Generate data according to a uniform distribution.

    This reference distribution is given in P414 in [1]_.

    Parameters
    ----------
    data: npt.NDArray[np.float64], shape=(n_obs, n_features)
        Data.
    runif: Callable, default=np.random.uniform
        Random data generator.

    Returns
    -------
    npt.NDArray[np.float64]
        Generated samples.

    References
    ----------
    .. [1] Tibshirani R., Walther G., and Hastie T. (2001), Estimating the
        number of clusters in a data setp via the gap statistic, Journal of the
        Royal Statistical Society, Series B, 63(2), 411--423.

    """
    return runif(low=data.min(axis=0), high=data.max(axis=0), size=data.shape)


def _generate_pca(
    data: npt.NDArray[np.float64],
    runif: Callable = np.random.uniform
) -> npt.NDArray[np.float64]:
    """Generate data according to a uniform distribution after PCA.

    This reference distribution is given in P414 in [1]_.

    Parameters
    ----------
    data: npt.NDArray[np.float64], shape=(n_obs, n_features)
        Data.
    runif: Callable, default=np.random.uniform
        Random data generator.

    Returns
    -------
    npt.NDArray[np.float64]
        Generated samples.

    References
    ----------
    .. [1] Tibshirani R., Walther G., and Hastie T. (2001), Estimating the
        number of clusters in a data setp via the gap statistic, Journal of the
        Royal Statistical Society, Series B, 63(2), 411--423.

    """
    data_c = data - data.mean(axis=0)
    _, _, vectors_t = np.linalg.svd(data_c, full_matrices=False)
    data_prime = np.matmul(data_c, vectors_t.T)
    features_prime = _generate_uniform(data_prime, runif)
    features = np.matmul(features_prime, vectors_t)
    return features + data.mean(axis=0)


def _clustering(
    data: npt.NDArray[np.float64],
    n_clusters: np.int64,
    **clusterer_kwargs: Optional[Dict]
) -> Union[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Define default clustering algorithm for Gap computation.

    This function uses the ``KMeans`` function from the ``sklearn`` library.

    Parameters
    ----------
    data: npt.NDArray[np.float64]
        Data.
    n_clusters: np.int64
        The number of clusters to form.
    **clusterer_kwargs: Optional[Dict]
        See ``sklearn.cluster.KMeans`` documentation for the list of available
        parameters.

    Returns
    -------
    Union[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        A tuple containing the estimated labels of each observations and the
        center of the clusters.

    """
    kmeans = KMeans(n_clusters=n_clusters, **clusterer_kwargs).fit(data)
    return kmeans.labels_, kmeans.cluster_centers_


def _estimate_gap(dispersion, references):
    value = np.mean(references) - dispersion
    error = np.sqrt(1 + 1 / len(references)) * np.std(references)
    return value, error


##############################################################################
# Class Gap

class Gap():
    r"""Gap Statistic.

    This module computes the Gap statistic [2]_ for a given dataset and number
    of clusters ``n_clusters``. Assuming the Euclidean distance as a measure of
    proximity between two observations, we note :math:`W_k` the pooled
    within-cluster sum of squares around the cluster means of the cluster
    :math:`k`. The Gap statistic is then given by

    .. math::

        Gap(k) = \mathbb{E}(\log W_k) - \log W_k,

    where :math:`\mathbb{E}` denotes the expectation under a sample of size
    :math:`n` from the reference distribution. The estimation of the number of
    clusters in the dataset in then given as the smallest :math:`k` such that
    :math:`Gap(k) \leq Gap(k + 1) - s_{k + 1}` where :math:`s_{k + 1}` is an
    estimate of the simulation error. Consider looking at the paper [2]_ for
    detailled information. In [1]_, the Gap statistic is defined without the
    logarithm. The code is adapted from [3]_.

    Parameters
    ----------
    n_jobs: np.int64, default=multiprocessing.cpu_count()
        Number of cores to use in case of multiprocessing. The default is the
        number of available cores, determined by
        ``multiprocessing.cpu_count()``. If ``n_jobs < 1``, it will be set to 1
        and if ``n_jobs > cpu_count()``, it will be set to ``cpu_count()``.
    parallel_backend: np.str_, default='multiprocessing'
        Parallel backend used for the computation.
    clusterer: Optional[Callable], default=None
        An user-provided function for the clustering of the dataset. The
        function has to return a tuple containing the labels of each
        observation and the centers of the different clusters.
    clusterer_kwargs: Optional[Dict], default=None
        The parameters to be used by the clustering function.
    generating_process: np.str_, default='pca'
        The generating process of the data for the reference datasets. One
        of `uniform` or `pca`.
    metric: Optional[Union[np.str_, np.int64]], default=None
        The metric used for the computation of the distance between the
        observations. See ``numpy.linalg.norm`` documentation
        for the list of available metric function. The default use the
        Euclidean distance.

    Attributes
    ----------
    n_clusters: np.int64
        Best number of clusters found. The estimation of the number of clusters
        in the dataset in given as the smallest :math:`k` such that
        :math:`Gap(k) \leq Gap(k + 1) - s_{k + 1}` where :math:`s_{k + 1}` is
        an estimate of the simulation error.
    gap: pd.DataFrame
        Gap value for different values of ``n_clusters``.

    References
    ----------
    .. [1] Mohajer M., Englmeier K.-H., and Schmid V. J., (2010), A comparison
        of Gap statistic definitions with and without logarithm function,
        Technical Report Number 096, Department of Statistics, University of
        Munich.
    .. [2] Tibshirani R., Walther G., and Hastie T. (2001), Estimating the
        number of clusters in a data setp via the gap statistic, Journal of the
        Royal Statistical Society, Series B, 63(2), 411--423.
    .. [3] Granger M., https://github.com/milesgranger/gap_statistic

    """

    def __init__(
        self,
        n_jobs: np.int64 = cpu_count(),
        parallel_backend: np.str_ = "multiprocessing",
        clusterer: Optional[Callable] = None,
        clusterer_kwargs: Optional[Dict] = None,
        generating_process: np.str_ = 'pca',
        metric: Optional[Union[np.str_, np.int64]] = None
    ) -> None:
        """Initialize Gap object."""
        # Initialize backend
        if parallel_backend is None:
            self.parallel_backend, self.n_jobs = parallel_backend, 1
        elif parallel_backend == 'multiprocessing':
            self.parallel_backend = parallel_backend
            self.n_jobs = min(max(1, n_jobs), cpu_count())
        else:
            raise ValueError(
                "The parallel backend has to be 'multiprocessing' or None."
            )

        # Initialize clustering parameters
        self.metric = metric
        self.clusterer = clusterer if clusterer is not None else _clustering
        self.clusterer_kwargs = clusterer_kwargs
        if self.clusterer is _clustering and self.clusterer_kwargs is None:
            self.clusterer_kwargs = {'init': 'k-means++', 'n_init': 10}

        # Initialize reference datasets genreating process
        if generating_process == 'uniform':
            self.generate_process = _generate_uniform
        elif generating_process == 'pca':
            self.generate_process = _generate_pca
        else:
            raise ValueError(
                "The generating process for the reference datasets "
                "have to be 'uniform' or 'pca'."
            )

    def __str__(self) -> str:
        """Override __str__ function."""
        return (
            f'Gap(n_jobs={self.n_jobs}, parallel_backend='
            f'{self.parallel_backend})'
        )

    def __repr__(self) -> str:
        """Override __repr__ function."""
        return self.__str__()

    def __call__(
        self,
        data: npt.NDArray[np.float64],
        cluster_array: Iterable[np.int64],
        n_refs: np.int64 = 3,
        runif: Callable = np.random.uniform
    ) -> np.int64:
        """Compute the Gap statistic.

        Parameters
        ----------
        data: npt.NDArray[np.float64], shape=(n_obs, n_components)
            Data as an array of shape (n_obs, n_components).
        cluster_array: Iterable[np.int64]
            The different number of clusters to try.
        n_refs: np.int64, default=3
            Number of random reference data sets used as inertia reference to
            actual data.
        runif: Callable, default=np.random.uniform
            Random data generator.

        Returns
        -------
        np.int64
            Returns the best number of clusters

        """
        if self.parallel_backend == 'multiprocessing':
            engine = self._process_with_multiprocessing
        else:
            engine = self._process_non_parallel

        # Compute Gap stat for each cluster count.
        gap_df = pd.DataFrame.from_records(
            engine(data, cluster_array, n_refs, runif),
            columns=['n_clusters', 'log_value', 'log_error', 'value', 'error']
        )
        gap_df_shift = gap_df.shift(periods=-1)

        # Compute the '1-standard-error' style rule
        gap_df['log_test'] = (
            gap_df['log_value']
            - gap_df_shift['log_value']
            + gap_df_shift['log_error']
        )
        gap_df['test'] = (
            gap_df['value']
            - gap_df_shift['value']
            + gap_df_shift['error']
        )

        self.gap = gap_df.sort_values(by="n_clusters", ascending=True).\
            reset_index(drop=True)
        self.n_clusters = gap_df.\
            loc[self.gap['log_test'].ge(0).argmax(), 'n_clusters']
        return self.n_clusters

    def plot(
        self,
        axes: Optional[Axes] = None,
        scatter_args: Optional[Dict] = None,
        **plt_kwargs
    ) -> Axes:
        """Plot the results of the Gap computation.

        Parameters
        ----------
        axes: matplotlib.axes._subplots.AxesSubplot
            Axes object onto which the objects are plotted.
        scatter_args: Optional[Dict]
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
        data: npt.NDArray[np.float64],
        n_clusters: np.int64,
        n_refs: np.int64,
        metric: Optional[Union[np.str_, np.int64]] = None,
        runif: Callable = np.random.uniform
    ) -> _GapResult:
        """Compute the Gap statistic.

        Parameters
        ----------
        data: npt.NDArray[np.float64]
            The data to cluster.
        n_clusters: np.int64
            Number of clusters to test.
        n_refs: np.int64
            Number of random reference data sets used as inertia reference to
            actual data.
        metric: Optional[Union[np.str_, np.int64]], default=None
            The metric used for the computation of the distance between the
            observations. The default uses euclidean distance. See
            ``numpy.linalg.norm`` documentation for the list of available
            metric function.
        runif: Callable, default=np.random.uniform
            Random data generator.

        Returns
        -------
        _GapResult
            Results as a GapResult object.

        """
        # Generate the reference distributions and compute dispersions
        ref_dispersions = np.zeros(n_refs)
        for idx in range(n_refs):
            samples = self.generate_process(data, runif)
            labels, centers = self.clusterer(
                samples, n_clusters, **self.clusterer_kwargs
            )
            ref_dispersions[idx] = _compute_dispersion(
                samples, labels, centers, metric
            )

        # Compute dispersion the original dataset
        labels, centers = self.clusterer(
            data, n_clusters, **self.clusterer_kwargs
        )
        dispersion = _compute_dispersion(data, labels, centers, metric)

        # Compute Gap statistic
        log_value, log_error = _estimate_gap(
            np.log(dispersion), np.log(ref_dispersions)
        )
        value, error = _estimate_gap(dispersion, ref_dispersions)
        return _GapResult(n_clusters, log_value, log_error, value, error)

    def _process_with_multiprocessing(
        self,
        data: npt.NDArray[np.float64],
        cluster_array: Iterable[np.int64],
        n_refs: np.int64 = 3,
        runif: Callable = np.random.uniform
    ) -> Generator[_GapResult, None, None]:
        """Compute Gap statistics with multiprocessing parallelization.

        Parameters
        ----------
        data: npt.NDArray[np.float64], shape=(n_obs, n_components)
            Data as an array of shape (n_obs, n_components).
        cluster_array: Iterable[np.int64]
            The different number of clusters to try.
        n_refs: np.int64, default=3
            Number of random reference data sets used as inertia reference to
            actual data.
        runif: Callable, default=np.random.uniform
            Random data generator.

        Returns
        -------
        Generator[_GapResult]
            Generator that contains the BIC for each number of clusters.

        """
        for result in Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_gap)(
                data, n_clusters, n_refs, self.metric, runif
            ) for n_clusters in cluster_array
        ):
            yield result

    def _process_non_parallel(
        self,
        data: npt.NDArray[np.float64],
        cluster_array: Iterable[np.int64],
        n_refs: np.int64 = 5,
        runif: Callable = np.random.uniform
    ) -> Generator[_GapResult, None, None]:
        """Compute Gap statistics without parallelization.

        Parameters
        ----------
        data: npt.NDArray[np.float64], shape=(n_obs, n_components)
            Data as an array of shape (n_obs, n_components).
        cluster_array: Iterable[np.int64]
            The different number of clusters to try.
        n_refs: np.int64, default=3
            Number of random reference datasets used as inertia reference to
            actual data.
        runif: Callable, default=np.random.uniform
            Random data generator.

        Returns
        -------
        Generator[_GapResult]
            Generator that contains the BIC for each number of clusters.

        """
        return (
            self._compute_gap(data, n_clusters, n_refs, self.metric, runif)
            for n_clusters in cluster_array
        )
