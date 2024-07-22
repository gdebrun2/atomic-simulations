import numpy as np
import numpy.linalg as la
import numba as nb


@nb.njit
def kmeans_init(data, k):

    idx = np.random.choice(data.shape[0], k, replace=False)

    if data.ndim == 1:

        init_centroids = data[idx]

    else:

        init_centroids = data[idx, :]

    return init_centroids


@nb.njit
def assign(data, centroids):

    N = data.shape[0]
    labels = np.empty(N, dtype=np.int8)

    for i, xi in enumerate(data):

        minimum = 1e10
        centroid_idx = 0

        for j, centroid in enumerate(centroids):

            d = la.norm(xi - centroid)

            if d < minimum:

                minimum = d
                centroid_idx = j

        labels[i] = centroid_idx

    return labels


@nb.njit
def update(data, k, labels):

    if data.ndim == 1:

        centroids = np.zeros(k)

    else:

        centroids = np.zeros((k, data.shape[1]))

    centroid_counts = np.zeros(k)

    for idx, pair in enumerate(zip(data, labels)):

        xi, label = pair
        centroids[label] += xi
        centroid_counts[label] += 1

    for idx, count in enumerate(centroid_counts):

        centroids[idx] /= count

    return centroids


@nb.njit
def run(data, k):

    np.random.seed(0)
    centroids = kmeans_init(data, k)
    while True:

        assignment = assign(data, centroids)
        prev_centroids = centroids.copy()
        centroids = update(data, k, assignment)

        if np.array_equal(prev_centroids, centroids):

            break

    return assignment
