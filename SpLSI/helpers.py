import numpy as np
from numpy.linalg import norm
import cvxpy as cp
from cvxpy import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem

def _euclidean_proj_simplex(v, s=1):
        (n,) = v.shape
        # check if we are already on the simplex
        if v.sum() == s and np.alltrue(v >= 0):
            return v
        
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
       
        theta = (cssv[rho] - s) / (rho + 1.0)
        w = (v - theta).clip(min=0)
        return w

def get_W_hat(U, H):
        projector = H.T.dot(np.linalg.inv(H.dot(H.T)))
        theta = U.dot(projector)
        theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj

def preprocess_U(U, K):
        for k in range(K):
            if U[0, k] < 0:
                U[:, k] = -1 * U[:, k]
        return U

def get_A_hat_(H_hat, L_hat, V_hat):
        #projector = (np.linalg.inv(W_hat.T.dot(W_hat))).dot(W_hat.T)
        #theta = projector.dot(M)
        theta = (H_hat @ L_hat) @ V_hat.T
        theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj

def get_A_hat(W_hat, M):
        projector = (np.linalg.inv(W_hat.T.dot(W_hat))).dot(W_hat.T)
        theta = projector.dot(M)
        #theta = (H_hat @ L_hat) @ V_hat.T
        theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
        return theta_simplex_proj

def get_A_hat_cvx(W, U, L, V, p, K):
        Theta = Variable((K, p))
        constraints = [cp.sum(Theta[i, :]) == 1 for i in range(K)]
        constraints += [Theta[i, j] >= 0 for i in range(K) for j in range(p)]
        L = np.diag(np.diag(L))
        M = (U @ L) @ V.T
        obj = Minimize(cp.norm(M - W @ Theta , "fro"))
        prob = Problem(obj, constraints)
        prob.solve()
        return np.array(Theta.value)

def spoc(U, V, X, K):
        J = []

        S = preprocess_U(U, K).T
        #S = U.T
        for t in range(K):
                maxind = np.argmax(norm(S, axis=0))
                s = np.reshape(S[:, maxind], (K, 1))
                S1 = (np.eye(K) - np.dot(s, s.T) / norm(s) ** 2).dot(S)
                S = S1
                J.append(maxind)
        H_hat = U[J, :]
        L_hat = np.diag(np.diag((U.T @ X) @ V))
        M = (U @ L_hat) @ V.T
        W_hat = get_W_hat(U, H_hat)
        A_hat = get_A_hat(W_hat, M)
        return W_hat, A_hat

def run_test(ntopics_list, X, A, W, K_true):
    W_spatials = []
    A_spatials = []
    V_spatials = []

    A_spatials.append(A.T)
    W_spatials.append(W)
    V_spatials.append(V_true.T)

    # Vanilla
    model_plsi = splsi.SpLSI(lamb_start=0.001, step_size=1.2, grid_len=26, verbose=0, method="non-spatial", eps=1e-04)
    model_plsi.fit(X, K_true, edge_df, weights)
    W_hat_v, A_hat_v = spoc(model_plsi.U, model_plsi.V, X, K_true)

    A_spatials.append(A_hat_v)
    W_spatials.append(W_hat_v)
    V_spatials.append(model_plsi.V)

    for i, K in enumerate(ntopics_list):
        print(f"Running SPLSI for {K}...")
        model_splsi = splsi.SpLSI(lamb_start=0.001, step_size=1.2, grid_len=26, verbose=0, eps=1e-04)
        model_splsi.fit(X, K, edge_df, weights)
        W_hat_spatial, A_hat_spatial = spoc(model_splsi.U, model_splsi.V, X, K)
        W_spatials.append(W_hat_spatial)
        A_spatials.append(A_hat_spatial)
        V_spatials.append(model_splsi.V)
    return W_spatials, A_spatials, V_spatials

def plot_test(matrices):
    size = len(matrices)
    fig, axes = plt.subplots(1, size, figsize=(3*size, 3)) 

    vmin = min(matrix.min() for matrix in matrices)
    vmax = max(matrix.max() for matrix in matrices)

    titles = ['True', 'VanillaSVD', 'Spatial(2)', 'Spatial(3)', 'Spatial(5)', 'Spatial(7)']

    k = 0
    for ax, matrix in zip(axes.flatten(), matrices):
        ntopics = matrix.shape[0]
        row_names = ["T"+str(i) for i in range(1, ntopics+1)]
        im = ax.imshow(matrix.T, cmap='Blues', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(row_names)))
        ax.set_xticklabels(row_names)
        ax.set_title(titles[k])
        ax.tick_params(axis='x', labeltop=True, labelbottom=False)
        k +=1

    fig.subplots_adjust(right=2)  # Adjust the right parameter to make space for the colorbar
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # Adjust the dimensions [left, bottom, width, height] as needed
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout()
    plt.show()