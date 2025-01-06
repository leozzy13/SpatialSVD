import time
import random
import numpy as np
from numpy.linalg import norm, svd, solve, qr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

import utils.spatial_lda.model
from utils.spatial_lda.featurization import make_merged_difference_matrices


def get_mst(edge_df):
    """
    Constructs a graph from the given edge DataFrame and computes its Minimum Spanning Tree (MST).

    Parameters:
    edge_df (pd.DataFrame): A DataFrame containing the edges of the graph with columns "src" and "tgt".

    Returns:
    tuple: The original graph `G` and the MST `mst`.
    """
    G = nx.from_pandas_edgelist(edge_df, "src", "tgt")
    connected_subgraphs = list(nx.connected_components(G))  # List connected components
    mst = nx.minimum_spanning_tree(G)  # Compute the Minimum Spanning Tree of the graph
    #if len(connected_subgraphs) > 1:
    #    raise ValueError("Graph is not connected.")
    return G, mst


def get_shortest_paths(mst, srn):
    """
    Computes the shortest paths from a source node to all other nodes in the MST.

    Parameters:
    mst (Graph): The Minimum Spanning Tree.
    srn (int): The source node.

    Returns:
    dict: A dictionary of shortest path lengths from the source node to each other node.
    """
    shortest_paths = dict(nx.shortest_path_length(mst, source=srn))
    return shortest_paths


def get_folds(mst):
    """
    Generates two folds of nodes based on the shortest path lengths from a randomly chosen source node in the MST.

    Parameters:
    mst (Graph): The Minimum Spanning Tree.

    Returns:
    tuple: The source node `srn`, and the two folds `fold1` and `fold2`.
    """
    srn = np.random.choice(mst.nodes)  # Randomly select a source node
    path = get_shortest_paths(mst, srn)
    fold1 = [key for key, value in path.items() if value % 2 == 0]  # Nodes at even distances
    fold2 = [key for key, value in path.items() if value % 2 == 1]  # Nodes at odd distances
    return srn, fold1, fold2


def get_folds_disconnected_G(edge_df):
    G = nx.from_pandas_edgelist(edge_df, "src", "tgt")
    connected_subgraphs = list(nx.connected_components(G))
    folds = {i: [] for i in range(5)}
    for graph in connected_subgraphs:
        G_sub = G.subgraph(graph)
        mst = nx.minimum_spanning_tree(G_sub)
        srn = np.random.choice(mst.nodes)
        path = get_shortest_paths(mst, srn)
        for node, length in path.items():
            folds[length % 5].append(node)
    return srn, folds, G, mst



def interpolate_X(X, G, folds, foldnum):
    """
    Interpolates the values of `X` for nodes in a fold by averaging the values of their neighbors.

    Parameters:
    X (np.array): The data matrix.
    G (Graph): The graph.
    folds (list): A list of folds.
    foldnum (int): The index of the fold to interpolate.

    Returns:
    np.array: The interpolated data matrix.
    """
    fold = folds[foldnum]

    X_tilde = X.copy()
    for node in fold:
        neighs = list(G.neighbors(node))  # Get the neighbors of the node
        neighs = list(set(neighs) - set(fold))  # Exclude other nodes in the fold
        X_tilde[node, :] = np.mean(X[neighs, :], axis=0)  # Average the values of the neighbors
    return X_tilde

def proj_simplex(v):
    """
    Projects a vector onto the simplex.

    Parameters:
    v (np.array): The input vector.

    Returns:
    np.array: The projected vector.
    """
    n = len(v)
    if np.sum(v) == 1 and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    rho = np.max(np.where(u * np.arange(1, n + 1) > (np.cumsum(u) - 1)))
    theta = (np.cumsum(u) - 1) / rho
    w = np.maximum(v - theta, 0)
    return w


def get_component_mapping(stats_1, stats_2):
    """
    Computes a mapping between components based on similarity.

    Parameters:
    stats_1 (np.array): First set of component statistics.
    stats_2 (np.array): Second set of component statistics.

    Returns:
    np.array: A binary matrix representing the component mapping.
    """
    similarity = stats_1 @ stats_2.T  # Compute the similarity matrix
    cost_matrix = 1 - np.abs(similarity)  # Compute the cost matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Solve the linear sum assignment problem
    P = np.zeros_like(cost_matrix)
    P[row_ind, col_ind] = 1  # Create the binary mapping matrix
    return P


def get_cosine_sim(A_1, A_2):
    """
    Computes the average cosine similarity between rows of two matrices.

    Parameters:
    A_1 (np.array): First matrix.
    A_2 (np.array): Second matrix.

    Returns:
    float: The average cosine similarity.
    """
    K = A_1.shape[0]
    A_1_norm = A_1 / norm(A_1, axis=1, keepdims=True)
    A_2_norm = A_2 / norm(A_2, axis=1, keepdims=True)
    s = np.sum(np.diag(A_1_norm @ A_2_norm.T))
    return s/K


def get_accuracy(coords_df, n, W_hat):
    """
    Computes the accuracy of group assignments.

    Parameters:
    coords_df (pd.DataFrame): DataFrame containing the true group assignments.
    n (int): Number of samples.
    W_hat (np.array): Predicted group assignment matrix.

    Returns:
    float: The accuracy of the group assignments.
    """
    assgn = np.argmax(W_hat, axis=1)  # Get the predicted group assignments
    accuracy = np.sum(assgn == coords_df["grp"].values) / n  # Compute the accuracy
    return accuracy


def get_F_err(W, W_hat):
    """
    Computes the Frobenius norm error between the true and predicted assignment matrices.

    Parameters:
    W (np.array): True assignment matrix.
    W_hat (np.array): Predicted assignment matrix.

    Returns:
    float: The Frobenius norm error.
    """
    err = norm(W.T - W_hat, ord="fro")  # Compute the Frobenius norm error
    return err


def get_l1_err(W, W_hat):
    """
    Computes the L1 norm error between the true and predicted assignment matrices.

    Parameters:
    W (np.array): True assignment matrix.
    W_hat (np.array): Predicted assignment matrix.

    Returns:
    float: The L1 norm error.
    """
    err = (abs(W.T - W_hat)).sum()  # Compute the L1 norm error
    return err



def get_l2_err(W, W_hat):
    """
    Computes the minimum L2 norm error between the transposed true assignment matrix 
    and the predicted assignment matrix using the Hungarian algorithm for optimal column alignment.

    Parameters:
    W (np.array): True assignment matrix.
    W_hat (np.array): Predicted assignment matrix.

    Returns:
    float: The minimum L2 norm error, summed over rows, with the optimal column alignment.
    """
    W_T = W.T  
    n_cols = W_hat.shape[1]
    
    cost_matrix = np.zeros((n_cols, n_cols))
    
    for i in range(n_cols):
        for j in range(n_cols):
            cost_matrix[i, j] = np.linalg.norm(W_T[:, i] - W_hat[:, j], ord=2)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    W_hat_permuted = W_hat[:, col_ind]

    row_errors = np.sqrt(((W_T - W_hat_permuted) ** 2).sum(axis=1))
    total_l2_error = row_errors.sum()

    return total_l2_error / W_hat.shape[0]





def inverse_L(L):
    """
    Computes the inverse of the diagonal elements of matrix L.

    Parameters:
    L (np.array): Input matrix.

    Returns:
    np.array: Matrix with the inverse of the diagonal elements.
    """
    d = np.diagonal(L)  # Extract the diagonal elements
    non_zero = d != 0  # Identify non-zero diagonal elements
    inv_d = np.zeros_like(d, dtype=float)
    inv_d[non_zero] = 1.0 / d[non_zero]  # Compute the inverse of the non-zero diagonal elements
    inv = np.diag(inv_d)  # Create a diagonal matrix with the inverted values
    return inv


def create_1NN_edge(coord_df):
    """
    Creates edges based on the 1-nearest neighbor for each point in the coordinate DataFrame.

    Parameters:
    coord_df (pd.DataFrame): DataFrame containing the coordinates with columns "x" and "y".

    Returns:
    pd.DataFrame: A DataFrame containing the edges with columns "src", "tgt", and "distance".
    """
    nn_model = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn_model.fit(coord_df[["x", "y"]])
    distances, indices = nn_model.kneighbors(coord_df[["x", "y"]])

    edges = []
    for i in range(len(coord_df)):
        for j in indices[i]:
            if i != j:
                edges.append({"src": i, "tgt": j, "distance": distances[i][1]})
    edge_df = pd.DataFrame(edges)
    return edge_df



def get_CHAOS(W, nodes, coord_df, n, K):
    """
    Computes the CHAOS score for spatial data clustering.

    Parameters:
    W (np.array): Weight matrix (n x K) representing the assignment probabilities of n nodes to K clusters.
    nodes (list): List of node indices.
    coord_df (pd.DataFrame): DataFrame containing the coordinates of the nodes with columns "x" and "y".
    n (int): Number of nodes.
    K (int): Number of clusters.

    Returns:
    tuple: The CHAOS score and a list of distances for each cluster.
    """
    # based on https://www.nature.com/articles/s41467-022-34879-1#citeas
    d_ij = 0  # Initialize the total distance within clusters
    d_all = []  # List to store distances for each cluster
    edge_df_1NN = create_1NN_edge(coord_df)  # Create a DataFrame of edges based on 1-nearest neighbor
    edge_df_1NN = edge_df_1NN.assign(
        normalized_distance=np.apply_along_axis(
            norm,
            1,
            coord_df.loc[edge_df_1NN["src"], ["x", "y"]].values
            - coord_df.loc[edge_df_1NN["tgt"], ["x", "y"]].values,
        )
    )
    # Assign source nodes, target nodes, and distances from the 1NN edge DataFrame
    src_nodes = edge_df_1NN["src"]
    tgt_nodes = edge_df_1NN["tgt"]
    distances = edge_df_1NN["normalized_distance"]
    nodes = np.asarray(nodes)
    for k in range(K):  # Iterate over each cluster
        K_nodes = nodes[np.argmax(W, axis=1) == k]  # Nodes belonging to cluster k
        src = np.isin(src_nodes, K_nodes)  # Check if source nodes belong to cluster k
        tgt = np.isin(tgt_nodes, K_nodes)  # Check if target nodes belong to cluster k
        d_ijk = np.sum(distances[src & tgt])  # Sum distances within cluster k
        d_all.append(distances[src & tgt])  # Store distances for cluster k
        d_ij += d_ijk  # Add to the total distance
    chaos = d_ij / n  # Compute the CHAOS score
    return 1 - chaos, d_all  # Return the CHAOS score and the list of distances for each cluster


def moran(W, edge_df):
    """
    Computes Moran's I statistic for spatial autocorrelation.

    Parameters:
    W (np.array): Weight matrix.
    edge_df (pd.DataFrame): DataFrame containing the edges with columns "src", "tgt", and "weight".

    Returns:
    tuple: Moran's I statistic and local Moran's I values.
    """
    # based on https://www.paulamoraga.com/book-spatial/spatial-autocorrelation.html
    weights = edge_df["weight"]  # Extract the edge weights
    tpc = np.argmax(W, axis=1)  # Determine the topic for each node
    tpc_avg = tpc - np.mean(tpc)  # Center the topics by subtracting the mean
    n = tpc_avg.shape[0]  # Number of nodes
    edge_df["cov"] = weights * tpc_avg[edge_df["src"]] * tpc_avg[edge_df["tgt"]]  # Calculate covariance for each edge
    src_grouped = edge_df.groupby("src")["cov"].sum().reset_index()  # Group by source and sum the covariances
    tgt_grouped = edge_df.groupby("tgt")["cov"].sum().reset_index()  # Group by target and sum the covariances
    result_df = pd.merge(
        src_grouped, tgt_grouped, left_on="src", right_on="tgt", how="outer"
    )
    val = result_df["cov_x"].fillna(0) + result_df["cov_y"].fillna(0)  # Combine source and target covariances
    val_by_node = val.values  # Get values as an array
    m2 = np.sum(tpc_avg**2)  # Sum of squared deviations from the mean
    I_local = n * (val_by_node / (m2 * 2))  # Local Moran's I
    I = np.sum(I_local) / np.sum(weights)  # Global Moran's I
    return I, I_local


def get_PAS(W, edge_df):
    """
    Computes the proportion of asymmetric spatial weights (PAS).

    Parameters:
    W (np.array): The spatial weights matrix.
    edge_df (pd.DataFrame): A DataFrame containing the edges of the graph with columns "src" and "tgt".

    Returns:
    float: The proportion of asymmetric spatial weights.
    """
    topics = np.argmax(np.array(W), axis=1)  # Determine the topic for each node
    edge_df["tpc_src"] = topics[edge_df["src"]]  # Assign topics to source nodes
    edge_df["tpc_tgt"] = topics[edge_df["tgt"]]  # Assign topics to target nodes
    src_grouped = (
        edge_df.groupby("src")
        .apply(lambda x: (x["tpc_src"] != x["tpc_tgt"]).mean())
        .rename("prop")
    )
    tgt_grouped = (
        edge_df.groupby("tgt")
        .apply(lambda x: (x["tpc_tgt"] != x["tpc_src"]).mean())
        .rename("prop")
    )
    result_df = pd.merge(
        src_grouped, tgt_grouped, left_on="src", right_on="tgt", how="outer"
    )
    val = result_df["prop_x"].fillna(0) + result_df["prop_y"].fillna(0)  # Combine proportions
    pas = (val >= 0.6).mean()  # Calculate proportion of asymmetric spatial weights
    return 1 - pas


def soft_threshold(x, thr):
    """
    Applies soft thresholding to the input array.

    Parameters:
    x (np.array): The input array.
    thr (float): The threshold value.

    Returns:
    np.array: The thresholded array.
    """
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0)


def hard_threshold(x, thr):
    """
    Applies hard thresholding to the input array.

    Parameters:
    x (np.array): The input array.
    thr (float): The threshold value.

    Returns:
    np.array: The thresholded array.
    """
    return np.where(np.abs(x) < thr, 0, x)


def get_Kfolds(n, nfolds):
    """
    Generates K folds for cross-validation.

    Parameters:
    n (int): The number of samples.
    nfolds (int): The number of folds.

    Returns:
    list: A list of K folds, each containing the indices of the samples.
    """
    indices = list(range(n))  # Create a list of sample indices
    random.shuffle(indices)  # Shuffle the indices randomly
    folds = []
    fold_size = [n // nfolds for _ in range(nfolds)]  # Calculate the size of each fold
    r = n % nfolds  # Calculate the remainder

    for i in range(nfolds):
        if i < r:
            fold_size[i] += 1  # Distribute the remainder

    start = 0
    for i in range(nfolds):
        end = start + fold_size[i]
        folds.append(indices[start:end])  # Create each fold with the calculated size
        start = end
    return folds


from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

def group_and_compare(U, coords_df):
    true_groups = coords_df['grp']
    num_clusters = len(true_groups.unique())

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    predicted_groups = kmeans.fit_predict(U)

    contingency_matrix = pd.crosstab(predicted_groups, true_groups)

    # Find the best permutation using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix.to_numpy())

    label_mapping = {row: col for row, col in zip(row_ind, col_ind)}
    predicted_groups_aligned = pd.Series(predicted_groups).map(label_mapping)

    accuracy = accuracy_score(true_groups, predicted_groups_aligned)

    return accuracy
    

def group_and_compare_spectral(U, coords_df):
    true_groups = coords_df['grp'] 
    num_clusters = len(true_groups.unique())  


    spectral = SpectralClustering(n_clusters=num_clusters, random_state=42, affinity='nearest_neighbors',n_neighbors=200)
    predicted_groups = spectral.fit_predict(U)


    contingency_matrix = pd.crosstab(predicted_groups, true_groups)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix.to_numpy())

    predicted_groups_aligned = predicted_groups.copy()
    for i, j in zip(row_ind, col_ind):
        predicted_groups_aligned[predicted_groups == i] = j

    accuracy = accuracy_score(true_groups, predicted_groups_aligned)

    return accuracy


def get_accuracy(predicted_clusters, coords_df):
    true_groups = coords_df['grp'] 
    num_clusters = len(true_groups.unique())  
    predicted_groups = predicted_clusters

    contingency_matrix = pd.crosstab(predicted_groups, true_groups)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix.to_numpy())

    predicted_groups_aligned = predicted_groups.copy()
    for i, j in zip(row_ind, col_ind):
        predicted_groups_aligned[predicted_groups == i] = j

    accuracy = accuracy_score(true_groups, predicted_groups_aligned)

    return accuracy


def euclidean_proj_simplex(V, s=1):
    """
    Projects each row of matrix V onto the simplex.

    Parameters:
    - V: 2D numpy array (matrix) where each row will be projected onto the simplex.
    - s: Sum of the target simplex, default is 1.

    Returns:
    - W: 2D numpy array with the same shape as V, with each row projected onto the simplex.
    """
    n_rows, n_cols = V.shape
    W = np.zeros_like(V) 
    
    for i in range(n_rows):
        v = V[i, :]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n_cols + 1) > (cssv - s))[0][-1]
        theta = (cssv[rho] - s) / (rho + 1.0)
        W[i, :] = (v - theta).clip(min=0)

    return W

def multinomial_from_rows(V, n=100):
    """
    Generates a multinomial sample for each row of V using the row as a probability vector.
    
    Parameters:
    - V: 2D numpy array (matrix) where each row represents probabilities for multinomial sampling.
    - n: Number of trials for each multinomial distribution (default is 100).
    
    Returns:
    - W: 2D numpy array with the same shape as V, where each row is a multinomial sample.
    """
    n_rows, n_cols = V.shape
    W = np.zeros_like(V, dtype=int)

    for i in range(n_rows):
        W[i, :] = np.random.multinomial(n, V[i, :])

    return W
