import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.linalg import sqrtm, inv


def get_initial_centers(val, centers):
    """
    Calculate initial centers for clustering.

    Parameters:
    val (np.array): An array of values to be clustered.
    centers (int): Number of clusters.

    Returns:
    list: A list of initial centers calculated as quantiles.
    """
    quantiles = []  # Initialize an empty list to store quantiles.
    for i in range(centers):
        # Calculate quantile index for each center and append to the list.
        quantiles.append(i * int(val.shape[0] / centers))
    return quantiles  # Return the list of quantiles.



def align_order(k, K):
    """
    Create an order for reordering vectors with noise.

    Parameters:
    k (int): Index to be set to zero in the order.
    K (int): Total number of elements.

    Returns:
    np.array: An array representing the order for reordering vectors with noise.
    """
    order = np.zeros(K, dtype=int)  # Initialize an array of zeros with length K.
    # Set all indices except k to a random permutation of 1 to K-1.
    order[np.where(np.arange(K) != k)[0]] = np.random.choice(np.arange(1, K), K - 1, replace=False)
    order[k] = 0  # Set the k-th index to 0.
    return order  # Return the order array.



def reorder_with_noise(v, order, K, r):
    """
    Reorder a vector with noise based on a given order.

    Parameters:
    v (np.array): The input vector to be reordered.
    order (np.array): The order in which to reorder the vector.
    K (int): Total number of elements.
    r (float): Probability of applying noise.

    Returns:
    np.array: The reordered vector, possibly with noise.
    """
    u = np.random.rand()  # Generate a random number between 0 and 1.
    if u < r:
        # If u is less than r, apply noise by randomly permuting the order array and using it to reorder v.
        return v[order[np.random.choice(range(K), K, replace=False)]]
    else:
        # Otherwise, sort v in descending order and reorder it according to the order array.
        sorted_row = np.sort(v)[::-1]
        return sorted_row[order]



def sample_MN(p, N):
    return np.random.multinomial(N, p, size=1)


def generate_graph_clustering(N, n, p, K, r):
    """
    Generate a graph with specified properties.

    Parameters:
    N (int): Total number of documents.
    n (int): Number of nodes.
    p (int): Number of words.
    K (int): Number of topics.
    r (float): Noise parameter.

    Returns:
    pd.DataFrame: A DataFrame containing coordinates and group information for each node.
    """
    coords = np.zeros((n, 2)) 
    coords[:, 0] = np.random.uniform(0, 1, n)  # Assign random x-coordinates between 0 and 1.
    coords[:, 1] = np.random.uniform(0, 1, n)  # Assign random y-coordinates between 0 and 1.

    # Perform k-means clustering with K clusters, initializing with specified coordinates.
    cluster_obj = KMeans(n_clusters=K, init=coords[get_initial_centers(coords, K), :], n_init=1)
    grps = cluster_obj.fit_predict(coords)  # Fit the model and predict cluster indices.

    # Create a DataFrame with coordinates and cluster assignments.
    coords_df = pd.DataFrame(coords, columns=["x", "y"])
    coords_df["grp"] = grps % K  # Assign each node to a group (topic) based on its cluster.
    coords_df["grp_blob"] = grps  # Store the original cluster assignments.
    return coords_df  # Return the DataFrame.

def generate_graph(N, n, p, K, r):
    """
    Generate a graph with specified properties.

    Parameters:
    N (int): Total number of documents.
    n (int): Number of nodes.
    p (int): Number of words.
    K (int): Number of topics.
    r (float): Noise parameter.

    Returns:
    pd.DataFrame: A DataFrame containing coordinates and group information for each node.
    """
    coords = np.zeros((n, 2)) 
    coords[:, 0] = np.random.uniform(0, 1, n)  # Assign random x-coordinates between 0 and 1.
    coords[:, 1] = np.random.uniform(0, 1, n)  # Assign random y-coordinates between 0 and 1.

    # Assign groups based on equally dividing the x-axis into K segments
    group_edges = np.linspace(0, 1, K+1)  # Create K+1 edges to define K segments
    grps = np.digitize(coords[:, 0], group_edges) - 1  # Assign groups based on x-coordinates

    # Create a DataFrame with coordinates and group assignments
    coords_df = pd.DataFrame(coords, columns=["x", "y"])
    coords_df["grp"] = grps  # Assign each node to a group based on its x-coordinate
    coords_df["grp_blob"] = grps  # Store the original group assignments
    
    return coords_df  # Return the DataFrame



def generate_W(coords_df, N, n, p, K, r):
    """
    Generate a weight matrix W.

    Parameters:
    coords_df (pd.DataFrame): DataFrame containing coordinates and group information for each node.
    N (int): Total number of documents.
    n (int): Number of nodes.
    p (int): Number of words.
    K (int): Number of topics.
    r (float): Noise parameter.

    Returns:
    np.array: A weight matrix W.
    """
    W = np.zeros((K, n))
    for k in range(K):
        alpha = np.random.uniform(0.1, 0.5, K)
        cluster_size = coords_df[coords_df["grp"] == k].shape[0]
        order = align_order(k, K)
        inds = coords_df["grp"] == k
        W[:, inds] = np.transpose(
            np.apply_along_axis(
                reorder_with_noise,
                1,
                np.random.dirichlet(alpha, size=cluster_size),
                order,
                K,
                r,
            )
        )

        # generate pure doc
        cano_ind = np.random.choice(np.where(coords_df["grp"] == k)[0])
        W[:, cano_ind] = 0
        W[k, cano_ind] = 1
    return W



def generate_W_strong(coords_df, N, n, p, K, r):
    """
    Generate a strong weight matrix W.

    Parameters:
    coords_df (pd.DataFrame): DataFrame containing coordinates and group information for each node.
    N (int): Total number of documents.
    n (int): Number of nodes.
    p (int): Number of words.
    K (int): Number of topics.
    r (float): Noise parameter.

    Returns:
    np.array: A strong weight matrix W.
    """
    W = np.zeros((K, n)) 
    
    for k in coords_df["grp"].unique():  # Loop over each unique group.
        for b in coords_df[coords_df["grp"] == k]["grp_blob"].unique():  # Loop over each unique blob within the group.
            alpha = np.random.normal(0.3, 0.5, K) 
            subset_df = coords_df[
                (coords_df["grp"] == k) & (coords_df["grp_blob"] == b)
            ]  # Subset the DataFrame to include only rows with the current group and blob.

            c = subset_df.shape[0]  # Get the number of rows in the subset.
            order = align_order(k, K)  # Generate an order array for reordering with noise.
            weight = reorder_with_noise(alpha, order, K, r)  # Reorder the weights with noise.
            inds = (coords_df["grp"] == k) & (coords_df["grp_blob"] == b)  # Get the indices of nodes in the current subset.
            # Assign the weights to the appropriate columns in W and add some noise.
            W[:, inds] = np.column_stack([weight] * c) + np.abs(
                np.random.normal(scale=0.05, size=c * K).reshape((K, c))
            )

    WTW = W.T.dot(W)
    eigenvalues, eigenvectors = np.linalg.eigh(WTW)
    eigenvalues_clipped = np.clip(eigenvalues, a_min=1e-8, a_max=None)
    WTW_stabilized = eigenvectors.dot(np.diag(np.sqrt(eigenvalues_clipped))).dot(eigenvectors.T)
    normalized_W = W.dot(np.linalg.inv(WTW_stabilized))

    return normalized_W # Return the normalized weight matrix W.



def generate_A(coords_df, N, n, p, K, r):
    A = np.random.uniform(0, 1, size=(p, K))

    # generate pure word
    cano_ind = np.random.choice(np.arange(p), K, replace=False)
    A[cano_ind, :] = np.eye(K)
    A = np.apply_along_axis(lambda x: x / np.sum(x), 0, A)
    return A


def generate_sparse_A(coords_df, N, n, p, K, r, sparse):
    #A = -np.log(np.random.uniform(0, 1, size=(p, K)))
    A = np.random.uniform(0, 0.01, size=(p, K))
    for k in range(K):
        supp = np.random.choice(np.arange(p), p // sparse, replace=False)
        A[supp,k] = np.random.uniform(0.1, 1, size=(len(supp)))
        
    # generate pure word
    #cano_ind = np.random.choice(supp, K, replace=False)
    #A[cano_ind, :] = np.eye(K)
    A = np.apply_along_axis(lambda x: x / np.sum(x), 0, A)
    return A


def generate_data(N, n, p, K, r, method="strong", sparse=5):
    coords_df = generate_graph(N, n, p, K, r)
    if method == "strong":
        W = generate_W_strong(coords_df, N, n, p, K, r)
    else:
        W = generate_W(coords_df, N, n, p, K, r)
    if sparse > 1:
        A = generate_sparse_A(coords_df, N, n, p, K, r, sparse)
    else:
        A = generate_A(coords_df, N, n, p, K, r)
    D0 = np.dot(A, W)
    D = np.apply_along_axis(sample_MN, 0, D0, N).reshape(p, n)
    assert np.sum(np.apply_along_axis(np.sum, 0, D) != N) == 0
    D = D / N

    return coords_df, W, A, D.T


def generate_weights_edge(coords_df, nearest_n, phi):
    """
    Generate weights and edges for a graph based on coordinates.

    Parameters:
    coords_df (pd.DataFrame): DataFrame containing coordinates and group information for each node.
    nearest_n (int): Number of nearest neighbors to consider.
    phi (float): Parameter for the RBF kernel.

    Returns:
    tuple: A tuple containing the sparse weight matrix and a DataFrame of edges with weights.
    """
    K = rbf_kernel(coords_df[["x", "y"]], gamma=phi)  # Calculate RBF kernel matrix for coordinates.
    np.fill_diagonal(K, 0)  # Set the diagonal to zero to remove self-loops.
    weights = np.zeros_like(K)  # Initialize an array of zeros with the same shape as K.

    for i in range(K.shape[0]):
        # Get indices of the nearest_n largest values in row i of K.
        top_indices = np.argpartition(K[i], -nearest_n)[-nearest_n:]
        weights[i, top_indices] = K[i, top_indices]  # Assign the corresponding weights.

    weights = (weights + weights.T) / 2  # Symmetrize the weight matrix.
    weights_csr = csr_matrix(weights)  # Convert the weight matrix to a sparse matrix format.

    rows, cols = weights_csr.nonzero()  # Get the non-zero indices of the sparse matrix.
    w = weights[weights.nonzero()]  # Get the corresponding weights.
    # Create a DataFrame containing source, target, and weight columns for the edges.
    edge_df = pd.DataFrame({"src": rows, "tgt": cols, "weight": w})
    return weights_csr, edge_df  # Return the sparse matrix and edge DataFrame.



def plot_scatter(coords_df):
    unique_groups = coords_df["grp"].unique()
    cmap = plt.get_cmap("Set3", len(unique_groups))
    colors = [cmap(i) for i in range(len(unique_groups))]

    for group, color in zip(unique_groups, colors):
        grp_data = coords_df[coords_df["grp"] == group]
        plt.scatter(grp_data["x"], grp_data["y"], label=group, color=color)


def get_colors(coords_df):
    grps = list(set(coords_df["grp"]))
    colors = []
    color_palette = ["cyan", "yellow", "greenyellow", "coral", "plum"]
    colormap = {value: color for value, color in zip(grps, color_palette[: len(grps)])}

    for value in coords_df["grp"]:
        colors.append(colormap[value])
    return colors


def plot_2d_tree(colors, G, mst):
    pos = nx.get_node_attributes(G, "pos")
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=10,
        node_color=colors,
        edge_color="gray",
        alpha=0.6,
    )
    nx.draw(
        mst,
        pos,
        with_labels=False,
        node_size=10,
        node_color=colors,
        edge_color="r",
        alpha=1,
    )
    plt.show()
