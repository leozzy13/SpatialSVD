import sys
import random
import numpy as np
from numpy.linalg import norm, svd, solve
from scipy.linalg import inv, sqrtm
import networkx as nx
import cvxpy as cp
from scipy.sparse.linalg import svds

from SpLSI.utils import *

sys.path.append("/Users/zhangzeyu/Downloads/research/SpLSI/pycvxcluster")
import pycvxcluster.pycvxcluster

# use pycvxcluster from "https://github.com/dx-li/pycvxcluster/tree/main"
from multiprocessing import Pool


def spatialSVD(
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
    L_inv_,
    initialize,
    fast_option,
    sparsity
):
    """
    Performs Spatial Singular Value Decomposition (Spatial SVD) on the input data.

    Parameters:
    X (np.array): The data matrix (n x p).
    K (int): The number of components to retain.
    edge_df (pd.DataFrame): DataFrame containing the edges of the graph.
    weights (np.array): Weight matrix for the graph.
    lamb_start (float): Initial value for lambda in the grid search.
    step_size (float): Multiplicative step size for the lambda grid.
    grid_len (int): Number of values in the lambda grid.
    maxiter (int): Maximum number of iterations.
    eps (float): Convergence threshold.
    verbose (int): Verbosity level.
    normalize (bool): Whether to normalize the U matrix.
    L_inv_ (bool): Whether to use the inverse of the L matrix.
    initialize (bool): Whether to initialize using a separate SVD.

    Returns:
    tuple: U, V, L matrices from the decomposition, best lambda, lambda errors, and number of iterations.
    """
    n = X.shape[0]  # Number of samples
    srn, folds, G, mst = get_folds_disconnected_G(edge_df)  

    lambd_grid = (lamb_start * np.power(step_size, np.arange(grid_len))).tolist()  
    lambd_grid.insert(0, 1e-04)
    beta_grid = (lamb_start * np.power(step_size, np.arange(grid_len+15))).tolist()  
    beta_grid.insert(0, 1e-04)
    beta_grid.insert(-1, 1)

    lambd_grid_init = (0.0001 * np.power(1.5, np.arange(15))).tolist()
    lambd_grid_init.insert(0, 1e-06)

    if initialize:
        print('Initializing..')
        M, _, _ = initial_svd(X, G, weights, folds, lambd_grid_init)  
        U, L, V = svds(M, k=K, solver='propack', maxiter=300)
        V = V.T
        L = np.diag(L)
    else:
        U, L, V = svds(X, k=K, solver='propack')  
        V = V.T
        L = np.diag(L)

    lambd_list = [0,1,2]
    score = 1
    niter = 0
    while score > eps and niter < maxiter:
        if n > 1000:
            idx = np.random.choice(range(n), 1000, replace=False)
        else:
            idx = range(n)
        
        U_samp = U[idx, :]
        P_U_old = np.dot(U_samp, U_samp.T)
        P_V_old = np.dot(V, V.T)
        X_hat_old = (P_U_old @ X[idx, :]) @ P_V_old
        if fast_option:
            if lambd_list[-1]==lambd_list[-2]==lambd_list[-3]:
                U, lambd, lambd_errs = update_U_tilde(X, V, L, G, weights, folds, lambd_grid, L_inv_, fast=True, lambd_prev=lambd_list[-1])
            else:
                U, lambd, lambd_errs = update_U_tilde(X, V, L, G, weights, folds, lambd_grid, L_inv_, fast=False, lambd_prev=lambd_list[-1])
        else:
            U, lambd, lambd_errs = update_U_tilde(X, V, L, G, weights, folds, lambd_grid, L_inv_, fast=False, lambd_prev=lambd_list[-1])

        lambd_list.append(lambd)

        if sparsity:
            V, L, beta_errs = update_V_L_tilde(X, U, L, folds, beta_grid, sparsity=True)   
        else:
            V, L, beta_errs = update_V_L_tilde(X, U, L, folds, beta_grid, sparsity=False)

        P_U = np.dot(U[idx, :], U[idx, :].T)
        P_V = np.dot(V, V.T)
        X_hat = (P_U @ X[idx, :]) @ P_V
        score = norm(X_hat - X_hat_old) / n
        niter += 1
        if verbose == 1:
            print(f"Error is {score}")
        
    print(f"SpatialSVD ran for {niter} steps.")

    return U, V, L, lambd, lambd_errs, beta_errs, niter



def lambda_search_init(j, folds, X, G, weights, lambd_grid):
    """
    Performs initial lambda search for each fold.

    Parameters:
    j (int): Index of the fold.
    folds (dict): Dictionary containing the folds.
    X (np.array): The data matrix.
    G (Graph): The graph.
    weights (np.array): Weight matrix for the graph.
    lambd_grid (list): List of lambda values for the grid search.

    Returns:
    tuple: The index of the fold, list of errors for each lambda, best M matrix, and best lambda value.
    """
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



def lambda_search(j, folds, X, V, L, G, weights, lambd_grid, L_inv_):
    """
    Performs lambda search for updating U matrix.

    Parameters:
    j (int): Index of the fold.
    folds (dict): Dictionary containing the folds.
    X (np.array): The data matrix.
    V (np.array): The V matrix from the SVD.
    L (np.array): The L matrix from the SVD.
    G (Graph): The graph.
    weights (np.array): Weight matrix for the graph.
    lambd_grid (list): List of lambda values for the grid search.
    normalize (bool): Whether to normalize the U matrix.
    L_inv_ (bool): Whether to use the inverse of the L matrix.

    Returns:
    tuple: The index of the fold, list of errors for each lambda, best U matrix, and best lambda value.
    """
    fold = folds[j]
    X_tilde = interpolate_X(X, G, folds, j)
    L_inv = 1 / np.diag(L)
    if L_inv_:
        print("Taking L_inv...")
        XVL_tinv = (X_tilde @ V) @ np.diag(L_inv)
        X_j = X[fold, :]
    else:
        XVL_tinv = X_tilde @ V
        X_j = X[fold, :] @ V
  
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
            E = U_tilde 
        err = norm(X_j - E[fold, :])/len(fold)
        errs.append(err)
        if err < best_err:
            lambd_best = lambd
            U_best = U_tilde
            best_err = err
    return j, errs, U_best, lambd_best



def initial_svd(X, G, weights, folds, lambd_grid):
    """
    Performs initial SVD and lambda search.

    Parameters:
    X (np.array): The data matrix.
    G (Graph): The graph.
    weights (np.array): Weight matrix for the graph.
    folds (dict): Dictionary containing the folds.
    lambd_grid (list): List of lambda values for the grid search.

    Returns:
    tuple: The best M matrix, the best lambda value, and the lambda errors.
    """
    lambds_best = []
    lambd_errs = {"fold_errors": {}, "final_errors": []}
    
    with Pool(5) as p:
        results = p.starmap(
            lambda_search_init,
            [(j, folds, X, G, weights, lambd_grid) for j in folds.keys()],
        )
    for result in results:
        j, errs, _, lambd_best = result
        lambd_errs["fold_errors"][j] = errs
        lambds_best.append(lambd_best)

    cv_errs = np.sum([lambd_errs["fold_errors"][i] for i in range(5)], axis=0)
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
    ssnal.fit(X=X, weight_matrix=weights, save_centers=True)
    M_hat = ssnal.centers_.T

    print(f"Optimal lambda is {lambd_cv}...")
    return M_hat, lambd_cv, lambd_errs



def update_U_tilde(X, V, L, G, weights, folds, lambd_grid, L_inv_, fast, lambd_prev):
    """
    Updates the U matrix using the current V and L matrices.

    Parameters:
    X (np.array): The data matrix.
    V (np.array): The V matrix from the SVD.
    L (np.array): The L matrix from the SVD.
    G (Graph): The graph.
    weights (np.array): Weight matrix for the graph.
    folds (dict): Dictionary containing the folds.
    lambd_grid (list): List of lambda values for the grid search.
    normalize (bool): Whether to normalize the U matrix.
    L_inv_ (bool): Whether to use the inverse of the L matrix.

    Returns:
    tuple: The updated U matrix, the best lambda value, and the lambda errors.
    """
    lambds_best = []
    lambd_errs = {"fold_errors": {}, "final_errors": []}
    L_inv = 1 / np.diag(L)
    if L_inv_:
        XVL_inv = (X @ V) @ np.diag(L_inv)
    else:
        XVL_inv = X @ V

    if fast:
        lambd_cv = lambd_prev
    else:
        with Pool(5) as p:
            results = p.starmap(
                lambda_search,
                [(j, folds, X, V, L, G, weights, lambd_grid, L_inv_) for j in folds.keys()],
            )
        for result in results:
            j, errs, _, lambd_best = result
            lambd_errs["fold_errors"][j] = errs
            lambds_best.append(lambd_best)

        cv_errs = np.sum([lambd_errs["fold_errors"][i] for i in range(5)], axis=0)
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

    print(f"Optimal lambda is {lambd_cv}...")

    return U_hat, lambd_cv, lambd_errs



def update_V_L_tilde(X, U_tilde, L, folds, beta_grid, sparsity=False):
    """
    Updates the V and L matrices using the current U matrix.

    Parameters:
    X (np.array): The data matrix.
    U_tilde (np.array): The updated U matrix from the SVD.
    normalize (bool): Whether to normalize the V matrix.

    Returns:
    tuple: The updated V matrix and the updated L matrix.
    """
    beta_errs = {"fold_errors": {}, "final_errors": []}
    betas_best = []

    if sparsity:
        betas_best = []
        XTU = X.T @ U_tilde

        with Pool(5) as p:
            results = p.starmap(
                beta_search,
                [(j, folds, X, U_tilde, beta_grid, L) for j in folds.keys()],
            )
        for result in results:
            j, errs, _, beta_best = result
            beta_errs["fold_errors"][j] = errs
            betas_best.append(beta_best)

        cv_errs = np.sum([beta_errs["fold_errors"][i] for i in range(5)], axis=0)
        beta_cv = beta_grid[np.argmin(cv_errs)]

        V = cp.Variable(XTU.shape)
        row_norms = cp.norm(V, axis=1)
        objective = cp.Minimize(cp.norm(XTU - V, "fro")**2 + beta_cv * cp.sum(row_norms))
        problem = cp.Problem(objective)
        problem.solve()
        V_mul = V.value
        print(f"Optimal beta is {beta_cv}...")
    else:
        V_mul = np.dot(X.T, U_tilde)

    V_hat, L_hat, _ = svd(V_mul, full_matrices=False)
    L_hat = np.diag(L_hat)

    return V_hat, L_hat, beta_errs



def beta_search(j, folds, X, U, beta_grid, L):
    """
    Performs initial beta search for each fold with an L2,1 penalty.

    Parameters:
    j (int): Index of the fold.
    folds (dict): Dictionary containing the folds.
    X (np.array): The data matrix.
    G (Graph): The graph.
    beta_grid (list): List of beta values for the grid search.

    Returns:
    tuple: The index of the fold, list of errors for each beta, best V matrix, and best beta value.
    """
    fold = folds[j]
    XTU = X[fold, :].T @ U[fold, :]
    X_j = np.delete(X, fold, axis=0)

    errs = []
    best_err = float("inf")
    V_best = None
    beta_best = 0

    for beta in beta_grid:
        V = cp.Variable(XTU.shape)
        
        row_norms = cp.norm(V, axis=1)
        objective = cp.Minimize(cp.norm(XTU - V, "fro")**2 + beta * cp.sum(row_norms))
        problem = cp.Problem(objective)

        problem.solve()
        V_mul = V.value
        V_hat, L_hat, _ = svd(V_mul, full_matrices=False)
        L_hat = np.diag(L_hat)

        E = U @ L_hat @ V_hat.T
        err = np.linalg.norm(X_j - np.delete(E, fold, axis=0))/len(fold)
        errs.append(err)
        
        if err < best_err:
            beta_best = beta
            V_best = V_hat
            best_err = err

    return j, errs, V_best, beta_best
