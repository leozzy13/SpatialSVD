from numpy.linalg import svd, norm
from scipy.optimize import linear_sum_assignment
import numpy as np
import networkx as nx
import pycvxcluster.pycvxcluster

def generate_graph_from_weights(df, weights, n):
    np.random.seed(127)
    G = nx.Graph()
    for node in range(n):
        x = df['x'].iloc[node]
        y = df['y'].iloc[node]
        G.add_node(node, pos=(x, y))
    
    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 < node2:
                #pos1 = G.nodes[node1]['pos']
                #pos2 = G.nodes[node2]['pos']
                w = weights[node1,node2]
                #dist = norm(np.array(pos1) - np.array(pos2))
                if w > 0:
                    G.add_edge(node1, node2, weight=w)
    return G

def get_mst_path(G):
    mst = nx.minimum_spanning_tree(G)
    path = dict(nx.all_pairs_shortest_path_length(mst))
    return mst, path

def generate_mst(df, weights, n):
    G = generate_graph_from_weights(df, weights, n)
    mst, path = get_mst_path(G)
    return G, mst, path

def get_parent_node(mst, path, srn, nodenum):
    neighs = list(mst.adj[nodenum].keys())
    length_to_srn = [path[neigh][srn] for neigh in neighs]
    parent = neighs[np.argmin(length_to_srn)]
    return parent

def interpolate_X(X, folds, foldnum, path, mst, srn):
    fold = folds[foldnum]
    
    for node in fold:
        parent = get_parent_node(path, mst, srn, node)
        X[node,:] = X[parent,:]
    return X

def get_folds(mst, path, n, plot_tree=False):
    np.random.seed(127)
    srn = np.random.choice(range(n),1)[0]
    print(f"Source node is {srn}")

    fold1 = []
    fold2 = []
    colors = []
    for key, value in path[srn].items():
        if (value%2)==0:
            fold1.append(key)
            colors.append("orange")
        elif (value%2)==1:
            fold2.append(key)
            colors.append("blue")
        else:
            colors.append("red")
    if plot_tree:
        nx.draw_kamada_kawai(mst, node_color = colors, node_size=10)
    return srn, fold1, fold2

def trunc_svd(X, K):
    U, L, VT = svd(X, full_matrices=False)
    U_k = U[:, :K]
    L_k = np.diag(L[:K])
    VT_k = VT[:K, :]
    return U_k, L_k, VT_k.T

def proj_simplex(v):
    n = len(v)
    if np.sum(v) == 1 and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    rho = np.max(np.where(u * np.arange(1, n + 1) > (np.cumsum(u) - 1)))
    theta = (np.cumsum(u) - 1) / rho
    w = np.maximum(v - theta, 0)
    return w

def get_component_mapping(stats_1, stats_2):
    similarity = np.dot(stats_1, stats_2.T)
    cost_matrix = -similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P = np.zeros_like(cost_matrix)
    P[row_ind, col_ind] = 1
    return P

def get_accuracy(df, n, W_hat):
    assgn = np.argmax(W_hat, axis=1)
    accuracy = np.sum(assgn == df['grp'].values) / n
    return accuracy

def get_F_err(W, W_hat):
    err = norm(W.T - W_hat, ord='fro')
    return err

def fit_SPOC(df, D, W, U, K, w, method="spatial"):
    if method != "spatial":
        print("Running vanilla SPOC...")
        X = D.T
        U, L, V = trunc_svd(X, K)

    J = []
    S = preprocess_U(U, K).T
    
    for t in range(K):
        maxind = np.argmax(norm(S, axis=0))
        s = np.reshape(S[:, maxind], (K, 1))
        S1 = (np.eye(K) - np.dot(s, s.T) / norm(s)**2).dot(S)
        S = S1
        J.append(maxind)
    
    H_hat = U[J, :]
    W_hat = get_W_hat(U, H_hat)

    P = get_component_mapping(W_hat.T, W)
    W_hat = np.dot(W_hat, P)
    
    assgn = np.argmax(W_hat, axis=1)
    accuracy = np.sum(assgn == df['grp'].values) / n
    err = norm(W.T - W_hat, ord='fro')
    print(err)
    return {'acc': accuracy, 'f.err': err, 'What': W_hat}

from scipy.sparse import csr_matrix
dense_array = w.toarray()
dense_array.shape
sparse_matrix = csr_matrix(weights)
sparse_matrix.nonzero()


def update_U_tilde2(X, V, weights, folds, path, mst, srn, lambd_grid, n, K):
    U_best_comb = np.zeros((n,K))
    lambds_best = []
    lambd_errs = {'fold_errors': {}, 'final_errors': []}

    for j in folds.keys():
        fold = folds[j]
        X_tilde = interpolate_X(X, folds, j, path, mst, srn)
        # print((X_tilde[fold[j],:]==X[fold[j],:]).sum()) # shouldn't be large
        #assert((X_tilde[fold[j],:]==X[fold[j],:]).sum()<=1)
        XV = np.dot(X, V)
        XV_tilde = np.dot(X_tilde, V)
        XV_j = XV[fold,:]
        row_sums = norm(XV_j, axis=1, keepdims=True)
        XV_j = XV_j / row_sums

        errs = []
        best_err = float("inf")
        U_best = None
        lambd_best = 0
        
        for lambd in lambd_grid:
            ssnal = pycvxcluster.pycvxclt.SSNAL(gamma=lambd, verbose=0)
            ssnal.fit(X=XV_tilde, weight_matrix=weights, save_centers=True)
            U_hat = ssnal.centers_.T
            row_sums = norm(U_hat, axis=1, keepdims=True)
            U_hat = U_hat / row_sums
            err = norm(XV_j-U_hat[fold,:])
            #E = np.dot(U_hat, V.T)
            #err = norm(X_j-E[fold,:])
            errs.append(err)
            if err < best_err:
                lambd_best = lambd
                U_best = U_hat
                best_err = err
        lambd_errs['fold_errors'][j] = errs
        U_best_comb[fold,:] = U_best[fold,:]
        lambds_best.append(lambd_best)

    final_errs = []
    best_final_err = float("inf")
    lambd_cv = 0
    for lambd in lambd_grid:
        ssnal = pycvxcluster.pycvxclt.SSNAL(gamma=lambd, verbose=0)
        ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
        U_hat_full = ssnal.centers_.T
        row_sums = norm(U_hat_full, axis=1, keepdims=True)
        U_hat_full = U_hat_full / row_sums
        final_err = norm(U_hat_full-U_best_comb)
        final_errs.append(final_err)
        if final_err < best_final_err:
            lambd_cv = lambd
            U_cv = U_hat_full
            best_final_err = final_err
    lambd_errs['final_errors'] = final_errs


def update_U_tilde_old(X, V, weights, folds, path, mst, srn, lambd_grid, n, K):
    U_best_comb = np.zeros((n,K))
    lambds_best = []
    lambd_errs = {'fold_errors': {}, 'final_errors': []}
    XV = np.dot(X, V)

    for j in folds.keys():
        fold = folds[j]
        X_tilde = interpolate_X(X, folds, j, path, mst, srn)
        # print((X_tilde[fold[j],:]==X[fold[j],:]).sum()) # shouldn't be large
        #assert((X_tilde[fold[j],:]==X[fold[j],:]).sum()<=1)
        XV_tilde = np.dot(X_tilde, V)
        X_j = X[fold,:]

        errs = []
        best_err = float("inf")
        U_best = None
        lambd_best = 0
        
        for lambd in lambd_grid:
            ssnal = pycvxcluster.pycvxclt.SSNAL(gamma=lambd, verbose=0)
            ssnal.fit(X=XV_tilde, weight_matrix=weights, save_centers=True)
            U_hat = ssnal.centers_.T
            row_sums = norm(U_hat, axis=1, keepdims=True)
            U_hat = U_hat / row_sums
            E = E = np.dot(U_hat, V.T)
            err = norm(X_j-E[fold,:])
            errs.append(err)
            if err < best_err:
                lambd_best = lambd
                U_best = U_hat
                best_err = err
        lambd_errs['fold_errors'][j] = errs
        U_best_comb[fold,:] = U_best[fold,:]
        lambds_best.append(lambd_best)

    final_errs = []
    best_final_err = float("inf")
    lambd_cv = 0
    for lambd in lambd_grid:
        ssnal = pycvxcluster.pycvxclt.SSNAL(gamma=lambd, verbose=0)
        ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
        U_hat_full = ssnal.centers_.T
        row_sums = norm(U_hat_full, axis=1, keepdims=True)
        U_hat_full = U_hat_full / row_sums
        E_best = np.dot(U_best, V.T)
        E_full = np.dot(U_hat_full, V.T)
        final_err = norm(E_full-E_best)
        final_errs.append(final_err)
        if final_err < best_final_err:
            lambd_cv = lambd
            U_cv = U_hat_full
            best_final_err = final_err
    lambd_errs['final_errors'] = final_errs

    Q, R = qr(U_cv)
    return Q, lambd_cv, lambd_errs

def interpolate_X(X, G, folds, foldnum, path, mst, srn):
    fold = folds[foldnum]
    
    X_tilde = X.copy()
    for node in fold:
        #parent = get_parent_node(mst, path, srn, node)
        neighs = list(G.neighbors(node))
        X_tilde[node,:] = np.mean(X[neighs,:], axis=0)
    return X_tilde

def update_L_tilde(X, U_tilde, V_tilde):
    L_tilde = np.dot(np.dot(U_tilde.T, X), V_tilde)
    return L_tilde

def generate_graph_from_weights_(df, weights, n):
    G = nx.Graph()
    pos = {node: (df['x'].iloc[node], df['y'].iloc[node]) for node in range(n)}
    G.add_nodes_from(pos)

    rows, cols = np.triu_indices(n, k=1)
    valid_edges = weights[rows, cols] > 0
    G.add_weighted_edges_from(zip(rows[valid_edges], cols[valid_edges], weights[rows, cols][valid_edges]))

    return G

def generate_graph_from_weights(df, weights, n):
    G = nx.Graph()
    for node in range(n):
        x = df['x'].iloc[node]
        y = df['y'].iloc[node]
        G.add_node(node, pos=(x, y))
    
    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 < node2:
                w = weights[node1,node2]
                if w > 0:
                    G.add_edge(node1, node2, weight=w)
    return G

def get_mst_path(G):
    mst = nx.minimum_spanning_tree(G)
    path = dict(nx.all_pairs_shortest_path_length(mst))
    return mst, path

def generate_mst(edge_df):
    G, mst = get_mst(edge_df)
    mst, path = get_mst_path(G)
    return G, mst, path

def get_parent_node(mst, path, srn, nodenum):
    neighs = list(mst[nodenum].keys())
    length_to_srn = [path[neigh][srn] for neigh in neighs]
    parent = neighs[np.argmin(length_to_srn)]
    return parent

def get_folds(mst, path, n, plot_tree=False):
    srn = np.random.choice(range(n),1)[0]
    #print(f"Source node is {srn}")

    fold1 = []
    fold2 = []
    colors = []
    for key, value in path[srn].items():
        if (value%2)==0:
            fold1.append(key)
            colors.append("orange")
        elif (value%2)==1:
            fold2.append(key)
            colors.append("blue")
        else:
            colors.append("red")
    if plot_tree:
        nx.draw_kamada_kawai(mst, node_color = colors, node_size=10)
    return srn, fold1, fold2


def generate_W_strong_ver2(df, N, n, p, K, r):
    W = np.zeros((K, n))
    for k in df['grp'].unique():
        alpha = np.random.uniform(0.1, 0.5, K)
        alpha = np.random.dirichlet(alpha)
        for b in df[df['grp'] == k]['grp_blob'].unique():
            alpha_blob = alpha + np.abs(np.random.normal(scale=0.03))
            subset_df = df[(df['grp'] == k) & (df['grp_blob'] == b)]

            c = subset_df.shape[0]
            order = align_order(k, K)
            weight = reorder_with_noise(alpha_blob, order, K, r)
            inds = (df['grp'] == k) & (df['grp_blob'] == b)
            W[:, inds] = np.column_stack([weight]*c)+np.abs(np.random.normal(scale=0.01, size = c*K).reshape((K,c)))

        # generate pure doc 
        cano_ind = np.random.choice(np.where(inds)[0], 1)
        W[:, cano_ind] = np.eye(K)[0, :].reshape(K,1)

    col_sums = np.sum(W, axis=0)
    W = W / col_sums
    return W


while True:
            try:
                coords_df, W, A, X = gen_model.generate_data(N, n, p, K, r)
                weights, edge_df = gen_model.generate_weights_edge(coords_df, m, phi)
                # Spatial SVD (two-step)
                start_time = time.time()
                model_splsi = splsi.SpLSI(lamb_start=lamb_start, step_size=step_size, grid_len=grid_len, step="two-step", verbose=0)
                model_splsi.fit(X, K, edge_df, weights)
                time_splsi = time.time() - start_time
                print(f"CV Lambda is {model_splsi.lambd}")
                # SLDA
                start_time = time.time()
                model_slda = utils.spatial_lda.model.run_simulation(X, K, coords_df)
                time_slda = time.time() - start_time
                break
            except Exception as e:
                print(f"Regenerating dataset due to error: {e}")


# Weight dist
# w = spleen_data['distance']
# np.histogram(np.sqrt(-0.1*np.log(w[w>0])))
# np.histogram(w)  # should we apply an extra sigmoid funciton?

# Neighbors dist
# neighs = np.concatenate([edge_df['src'].values, edge_df['dst'].values])
# uniq_elem, freq = np.unique(neighs, return_counts=True) 
# plt.hist(freq, bins=15)


def train_no_xi(sample_features, n_topics,
          max_lda_iter=5, n_iters=3, n_parallel_processes=1):
    """Train a Spatial-LDA model.
    
    Args:
        sample_features: Dataframe that contains neighborhood features of index cells indexed by (sample ID, cell ID).
                         (See featurization.featurize_samples).
        difference_matrices: Difference matrix corresponding to the spatial regularization structure imposed on the
                             samples. (I.e., which cells should be regularized to have similar priors on topics).
                             (See featurization.make_merged_difference_matrices).
        n_topics: Number of topics to fit.
        difference_penalty: Penalty on topic priors of "adjacent" index cells.
        n_iters: Number of outer-loop iterations (LDA + ADMM) to run.
        max_lda_iter: Maximum number of LDA iterations to run.
        max_admm_iter: Maximum number of ADMM iterations to run.
        max_primal_dual_iter: Maximum number of primal-dual iterations to run.
        max_dirichlet_iter: Maximum number of newton steps to take in computing updates for tau (see 5.2.8 in the
                            appendix).
        max_dirichlet_ls_iter: Maximum number of line-search steps to take in computing updates for tau
                               (see 5.2.8 in the appendix).
        n_parallel_processes: Number of parallel processes to use.
        verbosity: Amount of debug / info updates to see.
        primal_dual_mu: mu used in primal-dual updates (see paper for more details).
        admm_rho: rho used in ADMM optimization (see paper for more details).
        primal_tol: tolerance level for primal-dual updates.  In general, this value should not be
                    greater than 0.05.
        threshold: Cutoff for the percent change in the admm objective function.  Must be
                    greater than 0 and less than 1.  Typical value is 0.01.

    Returns:
        A Spatial-LDA model.
    """
    start_time = time.time()
    xis = None
    for i in range(n_iters):
        logging.info(f'>>> Starting iteration {i}')
        m_step_start_time = time.time()
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0,
                                        n_jobs=n_parallel_processes, max_iter=max_lda_iter,
                                        doc_topic_prior=xis)
        lda.fit(sample_features.values)
        m_duration = time.time() - m_step_start_time
        logging.info(f'>>> Iteration {i}, M-step took {m_duration} seconds.')
        
    last_m_step_start = time.time()
    columns = ['Topic-%d' % i for i in range(n_topics)]
    lda.topic_weights = pd.DataFrame(lda.fit_transform(sample_features.values),
                                     index=sample_features.index,
                                     columns=columns)
    logging.info(f'>>> Final M-step took {time.time() - last_m_step_start} seconds.')
    logging.info(f'>>> Training took {time.time() - start_time} seconds.')
    return lda

# Subset small N
minx = 0.0
maxx = 0.75
miny = 0.0
maxy = 0.7
val = np.sum(spleen_D.loc['BALBc-1'].values, axis=1)
small_N_ind = np.where(val<12)[0]
colors = ['lightcoral' if value in small_N_ind else 'lightblue' for value in coord_df.index.tolist()]
plt.scatter(x=coord_df['x'], y=coord_df['y'], c=colors, s=0.5)
plt.plot([minx,minx,maxx,maxx,minx],[miny,maxy,maxy,miny,miny], color='red')

coord_df_ = coord_df.loc[small_N_ind]
samp_coord = coord_df_.loc[(coord_df_['x']<maxx) & (coord_df_['x']>minx) & (coord_df_['y']<maxy) & (coord_df_['y']>miny)]
samp_nodes = samp_coord.index.tolist()
nsamp = len(samp_nodes)
edge_df__ = edge_df.loc[small_N_ind]
samp_edge = edge_df__.loc[(edge_df__['src'].isin(samp_nodes)) & (edge_df__['tgt'].isin(samp_nodes))]
samp_dict = dict(zip(samp_nodes, range(nsamp)))
samp_edge_ = samp_edge.copy()
samp_edge_['src'] = samp_edge['src'].map(samp_dict).values
samp_edge_['tgt'] = samp_edge['tgt'].map(samp_dict).values
samp_weights = csr_matrix((samp_edge_['weight'].values, (samp_edge_['src'].values, samp_edge_['tgt'].values)),shape=(nsamp,nsamp))
sampX = X[samp_nodes,:]

spleen_spl = splsi.SpLSI(lamb_start=0.0001, step_size=1.17, grid_len=50, step="two-step", verbose=1)
spleen_spl.fit(sampX, 3, samp_edge_, samp_weights)

spleen_v = splsi.SpLSI(lamb_start=0.0001, step_size=1.17, grid_len=50, method="nonspatial", verbose=1)
spleen_v.fit(sampX, 3, samp_edge_, samp_weights)

features = spleen_D.loc['BALBc-1'].iloc[samp_nodes,:]
cell_org  = [x[1] for x in spleen_D.loc['BALBc-1'].index]
cell_dict = dict(zip(range(len(spleen_D.loc['BALBc-1'])), cell_org))
samp_coord_ = samp_coord.copy()
samp_coord_.index = samp_coord.index.map(cell_dict)
samp_coord__ = {'BALBc-1':samp_coord_}
difference_matrices = make_merged_difference_matrices(features, samp_coord__, 'x','y')
spatial_lda_model = utils.spatial_lda.model.train(sample_features=features,
                                                difference_matrices=difference_matrices,
                                                difference_penalty=0.25,
                                                n_topics=3,
                                                n_parallel_processes=2,
                                                verbosity=1,
                                                admm_rho=0.1,
                                                primal_dual_mu=1e+5)
W_splsi = spleen_spl.W_hat
W_plsi = spleen_v.W_hat
W_slda = spatial_lda_model.topic_weights.values
Whats = [W_splsi, W_plsi, W_slda]
names = ['SPLSI','PLSI','SLDA']
morans = [moran_splsi, moran_plsi, moran_slda]
chaoss = [chaos_splsi, chaos_plsi, chaos_slda]

fig, axes = plt.subplots(1,3, figsize=(18,6))
for j, ax in enumerate(axes):
    w = np.argmax(Whats[j], axis=1)
    samp_coord_ = samp_coord.copy()
    samp_coord_['tpc'] = w
    sns.scatterplot(x='x',y='y',hue='tpc', data=samp_coord_, palette='viridis', ax=ax, s=20)
    name = names[j]
    ax.set_title(f'{name} (chaos:{np.round(chaoss[j],5)}, moran:{np.round(morans[j],3)})')
plt.tight_layout()
plt.show()

G = nx.from_pandas_edgelist(samp_edge, 'src', 'tgt')
connected_subgraphs = list(nx.connected_components(G))

lens = [len(j) for j in connected_subgraphs]
max_con_G = np.argmax(lens)
samp_nodes_con_G = connected_subgraphs[max_con_G]

val = np.sum(spleen_D.loc['BALBc-1'].values, axis=1)
colors = ['lightcoral' if value in samp_nodes_con_G else 'lightblue' for value in coord_df.index.tolist()]
plt.scatter(x=coord_df['x'], y=coord_df['y'], c=colors, s=0.5)


def update_U_tilde_mpi_old(X, V, G, weights, folds, lambd_grid, n, K):

    UL_best_comb = np.zeros((n, K))
    lambds_best = []
    lambd_errs = {"fold_errors": {}, "final_errors": []}
    XV = np.dot(X, V)

    # Distribute folds to processes
    if cfg.rank == 0:
        tasks = [(j, folds, X, V, G, weights, lambd_grid) for j in folds.keys()]
    else:
        tasks = None

    # Scatter tasks across MPI processes
    tasks = cfg.comm.scatter(tasks, root=0)

    # Each process works on its assigned task
    j, errs, UL_best, lambd_best = lambda_search(*tasks)

    # Gather results from all processes
    results = cfg.comm.gather((j, errs, UL_best, lambd_best), root=0)

    if cfg.rank == 0:
        # Master process combines results
        results = [res for res in results if res is not None]
        for j, errs, UL_best, lambd_best in results:
            lambd_errs["fold_errors"][j] = errs
            UL_best_comb[folds[j], :] = UL_best[folds[j], :]
            lambds_best.append(lambd_best)

        cv_errs = np.add(lambd_errs["fold_errors"][0], lambd_errs["fold_errors"][1])
        lambd_cv = lambd_grid[np.argmin(cv_errs)]

        ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
        ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
        UL_hat_full = ssnal.centers_.T

        Q, R = qr(UL_hat_full)
        return Q, lambd_cv, lambd_errs
    
    def preprocess_crc_(minx, maxx, miny, maxy, coord_df, edge_df, D, phi, plot_sub, s):
    new_columns = [col.replace("X", "x").replace("Y", "y") for col in coord_df.columns]
    coord_df.columns = new_columns

    # normalize coordinate to (0,1)
    coord_df[["x", "y"]] = normaliza_coords(coord_df)
    if maxx == None:
        maxx = np.max(coord_df["x"])
    if maxy == None:
        maxy = np.max(coord_df["y"])
    if minx == None:
        minx = np.min(coord_df["x"])
    if miny == None:
        miny = np.min(coord_df["y"])

    # get weight
    cell_to_idx_dict = dict(zip(coord_df["CELL_ID"], range(coord_df.shape[0])))
    edge_df_ = edge_df.copy()
    edge_df_["src"] = edge_df["src"].map(cell_to_idx_dict)
    edge_df_["tgt"] = edge_df["tgt"].map(cell_to_idx_dict)
    edge_df_["weight"] = dist_to_exp_weight(edge_df_, coord_df, phi)

    # edge, coord, X, weights
    nodes = coord_df.index.tolist()
    row_sums = D.sum(axis=1)
    X = D.div(row_sums, axis=0)  # normalize
    n = X.shape[0]
    weights = csr_matrix(
        (edge_df_["weight"].values, (edge_df_["src"].values, edge_df_["tgt"].values)),
        shape=(n, n),
    )

    # plot subset nodes (optional)
    if plot_sub:
        plt.scatter(x=coord_df["x"], y=coord_df["y"], s=s)
        plt.plot(
            [minx, minx, maxx, maxx, minx], [miny, maxy, maxy, miny, miny], color="red"
        )

    # subset nodes (optional)
    if maxx * maxy < 1.0 or minx * miny > 0.0:
        print("Subsetting cells...")
        # subset nodes
        samp_coord = coord_df.loc[
            (coord_df["x"] < maxx)
            & (coord_df["x"] > minx)
            & (coord_df["y"] < maxy)
            & (coord_df["y"] > miny)
        ]
        nodes = samp_coord.index.tolist()
        n = len(nodes)
        edge_df_ = edge_df_.loc[
            (edge_df_["src"].isin(nodes)) & (edge_df_["tgt"].isin(nodes))
        ]
        cell_dict = dict(zip(nodes, range(n)))
        edge_df__ = edge_df_.copy()
        edge_df__["src"] = edge_df_["src"].map(cell_dict).values
        edge_df__["tgt"] = edge_df_["tgt"].map(cell_dict).values
        weights = csr_matrix(
            (
                edge_df__["weight"].values,
                (edge_df__["src"].values, edge_df__["tgt"].values),
            ),
            shape=(n, n),
        )
        X = X.iloc[nodes]
        edge_df = edge_df__
        coord_df = samp_coord

    return X, edge_df, coord_df, weights, n, nodes