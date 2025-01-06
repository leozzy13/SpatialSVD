import os
import sys
import time
import random
import pickle
import numpy as np
from numpy.linalg import norm, svd, solve
from scipy.linalg import inv, sqrtm
import networkx as nx

from scipy.sparse.linalg import svds

from SpLSI.utils import *
from SpLSI import cfg

import pycvxcluster.pycvxcluster

# use pycvxcluster from "https://github.com/dx-li/pycvxcluster/tree/main"
from multiprocessing import Pool


def spatialSVDcs(
    X,
    K,
    edge_df,
    weights,
    lamb_start,
    step_size,
    grid_len,
    maxiter,
    eps,
    verbose,
    normalize,
    L_inv_,
    initialize,
    twostep
):
    n = X.shape[0]
    srn, fold1, fold2, G, mst = get_folds_disconnected_G(edge_df)
    folds = {0: fold1, 1: fold2}

    lambd_grid = (lamb_start * np.power(step_size, np.arange(grid_len))).tolist()
    lambd_grid.insert(0, 1e-06)

    lambd_grid_init = (0.001 * np.power(1.5, np.arange(10))).tolist()
    lambd_grid_init.insert(0, 1e-06)

    if initialize:
        #M_path = 'M_cook_initialize.pkl'
        #if os.path.exists(M_path):
        #    print("Loading matrix M from file...")
        #    with open(M_path, 'rb') as f:
        #        M = pickle.load(f)
        #    print("M loaded successfully.")
        #else:
        #    print('Initializing..')
        #    start_time = time.time()
        #    M, _, _ = initial_svd(X, G, weights, folds, lambd_grid_init)
        #    time_init = time.time() - start_time
        #    print(f'Initializing time: {time_init}')
        #    with open(M_path, 'wb') as f:
        #        pickle.dump(M, f)
        print('Initializing..')
        start_time = time.time()
        M, lambd, lambd_errs = initial_svd(X, G, weights, folds, lambd_grid_init)
        time_init = time.time() - start_time
        print(f'Initializing time: {time_init}')
        U, L, V = svds(M, k=K)
        V  = V.T
        L = np.diag(L)
        niter=0
    
    else:
        U, L, V = svds(X, k=K)
        V  = V.T
        L = np.diag(L)

    if twostep:
        score = 1
        niter = 0
        while score > eps and niter < maxiter:
            if n > 1000:
                idx = np.random.choice(range(n),1000,replace=False)
            else:
                idx = range(n)
            
            U_samp = U[idx,:]
            P_U_old = np.dot(U_samp, U_samp.T)
            P_V_old = np.dot(V, V.T)
            X_hat_old = (P_U_old @ X[idx,:]) @ P_V_old
            U, lambd, lambd_errs = update_U_tilde(X, V, L, G, weights, folds, lambd_grid, normalize, L_inv_)
            V, L = update_V_L_tilde(X, U, normalize)

            P_U = np.dot(U[idx,:], U[idx,:].T)
            P_V = np.dot(V, V.T)
            X_hat = (P_U @ X[idx,:]) @ P_V
            score = norm(X_hat-X_hat_old)/n
            niter += 1
            if verbose == 1:
                print(f"Error is {score}")
            

        print(f"SpatialSVD ran for {niter} steps.")

    return U, V, L, lambd, lambd_errs, niter


def lambda_search_init(j, folds, X, G, weights, lambd_grid):
    fold = folds[j]
    X_tilde = interpolate_X(X, G, folds, j)
    X_j = X[fold, :]
  
    errs = []
    best_err = float("inf")
    M_best = None
    lambd_best = 0

    ssnal = pycvxcluster.pycvxcluster.SSNAL(verbose=0)

    for fitn, lambd in enumerate(lambd_grid):
        ssnal.gamma = lambd
        ssnal.fit(
            X=X_tilde,
            weight_matrix=weights,
            save_centers=True,
            save_labels=False,
            recalculate_weights=(fitn == 0),
        )
        ssnal.kwargs["x0"] = ssnal.centers_
        ssnal.kwargs["y0"] = ssnal.y_
        ssnal.kwargs["z0"] = ssnal.z_
        M_hat = ssnal.centers_.T
        err = norm(X_j - M_hat[fold, :])
        errs.append(err)
        if err < best_err:
            lambd_best = lambd
            M_best = M_hat
            best_err = err
    return j, errs, M_best, lambd_best


def lambda_search(j, folds, X, V, L, G, weights, lambd_grid, normalize, L_inv_):
    fold = folds[j]
    X_tilde = interpolate_X(X, G, folds, j)
    L_inv = 1/np.diag(L)
    if L_inv_:
        print("Taking L_inv...")
        XVL_tinv = (X_tilde @ V) @ np.diag(L_inv)
    else:
        XVL_tinv = X_tilde @ V
    X_j = X[fold, :]
  
    errs = []
    best_err = float("inf")
    U_best = None
    lambd_best = 0

    ssnal = pycvxcluster.pycvxcluster.SSNAL(verbose=0)

    for fitn, lambd in enumerate(lambd_grid):
        ssnal.gamma = lambd
        ssnal.fit(
            X=XVL_tinv,
            weight_matrix=weights,
            save_centers=True,
            save_labels=False,
            recalculate_weights=(fitn == 0),
        )
        ssnal.kwargs["x0"] = ssnal.centers_
        ssnal.kwargs["y0"] = ssnal.y_
        ssnal.kwargs["z0"] = ssnal.z_
        U_tilde = ssnal.centers_.T
        if L_inv_:
            E = (U_tilde @ L) @ V.T
        else:
            E = U_tilde @ V.T
        err = norm(X_j - E[fold, :])
        errs.append(err)
        if err < best_err:
            lambd_best = lambd
            U_best = U_tilde
            best_err = err
    return j, errs, U_best, lambd_best


def initial_svd(X, G, weights, folds, lambd_grid):
    lambds_best = []
    lambd_errs = {"fold_errors": {}, "final_errors": []}
    
    with Pool(2) as p:
        results = p.starmap(
            lambda_search_init,
            [(j, folds, X, G, weights, lambd_grid) for j in folds.keys()],
        )
    for result in results:
        j, errs, _, lambd_best = result
        lambd_errs["fold_errors"][j] = errs
        lambds_best.append(lambd_best)

    cv_errs = np.add(lambd_errs["fold_errors"][0], lambd_errs["fold_errors"][1])
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
    ssnal.fit(X=X, weight_matrix=weights, save_centers=True)
    M_hat = ssnal.centers_.T

    print(f"Optimal lambda is {lambd_cv}...")
    return M_hat, lambd_cv, lambd_errs


def update_U_tilde(X, V, L, G, weights, folds, lambd_grid, normalize, L_inv_):
    lambds_best = []
    lambd_errs = {"fold_errors": {}, "final_errors": []}
    L_inv = 1/np.diag(L)
    if L_inv_:
        XVL_inv = (X @ V) @ np.diag(L_inv)
    else:
        XVL_inv = X @ V

    with Pool(2) as p:
        results = p.starmap(
            lambda_search,
            [(j, folds, X, V, L, G, weights, lambd_grid, normalize, L_inv_) for j in folds.keys()],
        )
    for result in results:
        j, errs, _, lambd_best = result
        lambd_errs["fold_errors"][j] = errs
        lambds_best.append(lambd_best)

    cv_errs = np.add(lambd_errs["fold_errors"][0], lambd_errs["fold_errors"][1])
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
    ssnal.fit(X=XVL_inv, weight_matrix=weights, save_centers=True)
    U_tilde = ssnal.centers_.T

    if L_inv_:
        print('Normalizing...')
        U_hat = U_tilde @ sqrtm((inv(U_tilde.T @ U_tilde)))
    else:
        print('Taking SVD of U...')
        U_hat, _, _ = svd(U_tilde, full_matrices=False)

    #if normalize:
    #    print('Normalizing...')
    #    U_hat = U_tilde @ sqrtm((inv(U_tilde.T @ U_tilde)))
    #else:
    #    print('Taking QR...')
    #    U_hat, R = qr(U_tilde)
    print(f"Optimal lambda is {lambd_cv}...")
    return U_hat, lambd_cv, lambd_errs


def update_V_L_tilde(X, U_tilde, normalize):
    V_mul = np.dot(X.T, U_tilde)
    if normalize:
        V_hat, L_hat, _ = svd(V_mul, full_matrices=False)
        L_hat = np.diag(L_hat)
    else:
        V_hat, L_hat = qr(V_mul)
        L_hat = np.diag(np.diag(L_hat))
    return V_hat, L_hat