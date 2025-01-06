import numpy as np


def find_clusters(X, tol):
    """
    Group vectors into clusters based on the given tolerance.

    Parameters:
    - X: A 2D numpy array where each column is a vector to be clustered.
    - tol: Tolerance to declare that two vectors are close enough to be in the same cluster.

    Returns:
    - cluster_id: A list indicating the cluster assignment for each vector.
    - num_cluster: Total number of clusters formed.
    """

    if X.size == 0:
        raise ValueError("Input Data must be nonempty.")

    m, n = X.shape
    index_temp = np.arange(n)
    cluster_id = np.ones(n, dtype=int)
    reference_id = 0

    while len(index_temp) > 0:
        reference_id += 1
        reference_point = X[:, index_temp[0]]
        cluster_id[index_temp[0]] = reference_id
        index_temp = index_temp[1:]

        Xdiff = X[:, index_temp] - reference_point[:, np.newaxis]
        normXdiff = np.linalg.norm(Xdiff, axis=0)
        idx = np.where(normXdiff < tol * m)[0]
        index_t = index_temp[idx]
        cluster_id[index_t] = reference_id

        index_temp = np.setdiff1d(index_temp, index_t)

    num_cluster = reference_id
    return cluster_id, num_cluster
