import sys
import os
import time
import pickle
import numpy as np
from numpy.linalg import norm, svd, solve, qr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import ast

from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

# !git clone https://github.com/dx-li/pycvxcluster.git
sys.path.append("./pycvxcluster/src/")
import pycvxcluster.pycvxcluster

from SpLSI.utils import *
from utils.spatial_lda.featurization import make_merged_difference_matrices


def tuple_converter(s):
    return ast.literal_eval(s)


def normaliza_coords(coords):
    """
    Input: pandas dataframe (n x 2) of x,y coordinates
    Output: pandas dataframe (n x 2) of normalizaed (0,1) x,y coordinates
    """
    minX = min(coords["x"])
    maxX = max(coords["x"])
    minY = min(coords["y"])
    maxY = max(coords["y"])
    diaglen = np.sqrt((minX - maxX) ** 2 + (minY - maxY) ** 2)
    coords["x"] = (coords["x"] - minX) / diaglen
    coords["y"] = (coords["y"] - minY) / diaglen

    return coords[["x", "y"]].values


def dist_to_exp_weight(df, coords, phi):
    """
    Input:
    - df: pandas dataframe (n x 2) of src, dst nodes
    - coords: pandas dataframe (n x 2) of normalizaed (0,1) x,y coordinates
    - phi: weight parameter
    Ouput: pandas dataframe (n x 3) of src, dst, squared exponential kernel distance
    """
    diff = (
        coords.loc[df["src"], ["x", "y"]].values
        - coords.loc[df["tgt"], ["x", "y"]].values
    )
    w = np.exp(-phi * np.apply_along_axis(norm, 1, diff) ** 2)
    return w


def dist_to_normalized_weight(distance):
    # dist_inv = 1/distance
    dist_inv = distance
    norm_dist_inv = (dist_inv - np.min(dist_inv)) / (
        np.max(dist_inv) - np.min(dist_inv)
    )
    return norm_dist_inv


def plot_topic(spatial_models, ntopics_list, fig_root, tumor, s):
    aligned_models = apply_order(spatial_models, ntopics_list)
    color_palette = sns.color_palette("husl", 10)
    colors = np.array(color_palette[:10])

    names = ["SPLSI", "PLSI", "SLDA"]
    for ntopic in ntopics_list:
        img_output = os.path.join(fig_root, tumor + "_" + str(ntopic))
        chaoss = spatial_models[ntopic][0]["chaoss"]
        morans = spatial_models[ntopic][0]["morans"]
        pas = spatial_models[ntopic][0]["pas"]
        times = spatial_models[ntopic][0]["times"]
        plt.clf()
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for j, ax in enumerate(axes):
            w = np.argmax(aligned_models[ntopic][0]["Whats"][j], axis=1)
            samp_coord_ = aligned_models[ntopic][0]["coord_df"].copy()
            samp_coord_["tpc"] = w
            ax.scatter(samp_coord_["x"], samp_coord_["y"], s=s, c=colors[w])
            name = names[j]
            ax.set_title(
                f"{name} (chaos:{np.round(chaoss[j],7)}, moran:{np.round(morans[j],2)}, pas:{np.round(pas[j],2)}, time:{np.round(times[j],2)})"
            )
        plt.tight_layout()
        plt.savefig(img_output, dpi=300, bbox_inches="tight")
        plt.close()
    return aligned_models


def plot_What(What, coord_df, ntopic):
    samp_coord_ = coord_df.copy()
    fig, axes = plt.subplots(1, ntopic, figsize=(18, 6))
    for j, ax in enumerate(axes):
        w = What[:, j]
        samp_coord_[f"w{j+1}"] = w
        sns.scatterplot(
            x="x",
            y="y",
            hue=f"w{j+1}",
            data=samp_coord_,
            palette="viridis",
            ax=ax,
            s=17,
        )
        ax.set_title(f"Original Plot {j+1}")
    plt.tight_layout()
    plt.show()
    plt.close()


def get_component_mapping(stats_1, stats_2):
    similarity = stats_1 @ stats_2.T
    cost_matrix = -similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P = np.zeros_like(cost_matrix)
    P[row_ind, col_ind] = 1
    return P


def get_component_mapping_(stats_1, stats_2):
    similarity = stats_1 @ stats_2.T
    assignment = linear_sum_assignment(-similarity)
    mapping = {k: v for k, v in zip(*assignment)}
    return mapping


def get_consistent_order(stats_1, stats_2, ntopic):
    ntopics_1 = stats_1.shape[1]
    ntopics_2 = stats_2.shape[1]
    mapping = get_component_mapping_(stats_1[:, :ntopic].T, stats_2.T)
    mapped = mapping.values()
    unmapped = set(range(ntopics_1)).difference(mapped)
    order = [mapping[k] for k in range(ntopics_2)] + list(unmapped)
    return order


def apply_order(spatial_models, ntopics_list):
    init_topic = ntopics_list[0]
    P_v = get_component_mapping(
        spatial_models[init_topic][0]["Whats"][1].T,
        spatial_models[init_topic][0]["Whats"][0].T,
    )
    P_slda = get_component_mapping(
        spatial_models[init_topic][0]["Whats"][2].T,
        spatial_models[init_topic][0]["Whats"][0].T,
    )
    W_hat_v = spatial_models[init_topic][0]["Whats"][1] @ P_v
    W_hat_slda = spatial_models[init_topic][0]["Whats"][2] @ P_slda
    spatial_models[init_topic][0]["Whats"][1] = W_hat_v
    spatial_models[init_topic][0]["Whats"][2] = W_hat_slda

    for ntopic1, ntopic2 in zip(ntopics_list[:-1], ntopics_list[1:]):
        # alignment within (K-1, K) initial topic
        src = spatial_models[ntopic1][0]["Whats"][0]
        tgt = spatial_models[ntopic2][0]["Whats"][0]
        P1 = get_component_mapping(tgt.T, src.T)
        P = np.zeros((ntopic2, ntopic2))
        P[:, :ntopic1] = P1
        col_ind = np.where(np.all(P == 0, axis=0))[0].tolist()
        row_ind = np.where(np.all(P == 0, axis=1))[0].tolist()
        for i, col in enumerate(col_ind):
            P[row_ind[i], col] = 1
        spatial_models[ntopic2][0]["Whats"][0] = (
            spatial_models[ntopic2][0]["Whats"][0] @ P
        )

        # alignment within each ntopic
        P_v = get_component_mapping(
            spatial_models[ntopic2][0]["Whats"][1].T,
            spatial_models[ntopic2][0]["Whats"][0].T,
        )
        P_slda = get_component_mapping(
            spatial_models[ntopic2][0]["Whats"][2].T,
            spatial_models[ntopic2][0]["Whats"][0].T,
        )
        W_hat_v = spatial_models[ntopic2][0]["Whats"][1] @ P_v
        W_hat_slda = spatial_models[ntopic2][0]["Whats"][2] @ P_slda
        spatial_models[ntopic2][0]["Whats"][1] = W_hat_v
        spatial_models[ntopic2][0]["Whats"][2] = W_hat_slda
    return spatial_models
