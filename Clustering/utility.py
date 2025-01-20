import traceback
import logging
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np


@ contextmanager
def random_seed(seed):
    random_state = np.random.get_state()
    if seed != None:
        np.random.seed(seed)
    try:
        yield
    finally:
        if seed != None:
            np.random.set_state(random_state)


def generate_data(n=100, p=2, K=3, seed=None):
    """Generates blob data (as a gaussian mixture)

    -- n the number of samples

    -- p the dimension

    -- K the number of blobs

    -- seed can be set (to any integer) for reproducibility (default is None, with no seed set)

    returns data, mu, clusters

    -- data is the nxp numpy array of data 

    -- mu is a Kxp numpy arrays of the blob centers

    -- clusters is a (N,) numpy array of integers between 0 and K-1 assigning the data to clusters

    """

    with random_seed(seed):
        mu_ = 1 * np.random.randn(K, p)
        std_ = 0.5 * np.sqrt(np.random.rayleigh(1, K))
        data_ = np.random.randn(n, p, K)
        clusters = np.random.randint(0, K, n)

    for k in range(K):
        data_[:, :, k] = data_[:, :, k] * std_[k] + mu_[k]
    data = np.take_along_axis(data_, clusters[:, None, None], -1)[:, :, 0]

    return data, mu_, clusters


def show_clusters(data, clusters, centroids=None, ax=None, sample_size=5, **kw):
    """Show the clusters as a colormap.

    -- data: a nxp numpy array,  the data to sho (only two first dims will be shown)

    -- clusters: a (n,) numpy array of integers, assigning the samples of data to clusters

    -- centroids: centroids of cluster. Either None (no centroids
    shown), 'compute' (centroids are computed) or a Kxp nump array. Default is None

    -- ax: a pyplot axis to use to show. If None (default) the last one created will we used.

    -- sample_size: the size of the dots representing the data.

    """

    if isinstance(centroids, str) and centroids == 'compute':
        raise NotImplementedError('Sorry, not yet implemented')

    if data.shape[1] > 2:
        logging.warning('Showing only 2 first dims of {}'.format(data.shape[1]))

    if not ax:
        ax = plt.gca()

    K = clusters.max() + 1

    if ax is None:
        ax = plt.gca()

    ax.scatter(data[:, 0], data[:, 1], c=clusters, s=sample_size, **kw)
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c=[*range(len(centroids))],
                   marker='x', s=2 * sample_size, **kw)


def compute_centroids(data, clusters, K=None):
    """Compute the centroids of data distributed by clusters.

    -- data is a nxp numpy array

    -- clusters is a (n,) numpy array

    -- K is a number of clusters (if K is None (default) K is guessed
    as being clusters.max() + 1

    -- returns: centroids a Kxp numpy array with the centroids

    """

    K_ = clusters.max() + 1
    if not K:
        K = K_
    assert K_ <= K
    assert len(data) == len(clusters)

    n, p = data.shape

    centroids = np.zeros((K, p))

    for k in range(K):
        i = clusters == k
        centroids[k] = data[i].mean(0)

    return centroids


def assign_clusters(data, centroids, norm_ord=2):
    """
    will assign clusters to each row of data as the nearest centroid

    -- data is a nxp numpy array

    -- centroids is a Kxp array

    -- returns: clusters (a (n,) numpy array) of the clusters. 

    """
    n, _ = data.shape

    clusters = np.zeros(n, dtype=int)

    for i in range(n):

        dist = np.linalg.norm(data[i] - centroids, axis=1, ord=norm_ord)
        clusters[i] = dist.argmin()

    return clusters


def inertia(data, clusters):
    """Compute intra cluster, inter cluster and total inertia of
    datapoints relatively to the clustering.

    -- data a n,p numpy array

    -- clusters a (n,) numpy array

    -- returns intra, inter, total

    """
    total = data.var(0).sum() * len(clusters)
    intra = 0

    for k, n in zip(*np.unique(clusters, return_counts=True)):

        intra += data[clusters == k].var(0).sum() * n

    inter = total - intra

    return intra, inter, total


if __name__ == '__main__':

    seed = 1
    n = 10000
    p = 2

    K = 3
    plt.close('all')

    data, centroids, clusters = generate_data(n, p, K, seed=1)

    show_clusters(data, clusters, centroids)

    plt.show(block=False)
