from sklearn.base import BaseEstimator, ClusterMixin
from pycvxcluster.algos.compute_weights import compute_weights
from pycvxcluster.algos.compute_weights import compute_weight_matrix
from pycvxcluster.algos.compute_weights import get_nam_wv_from_wm
from pycvxcluster.algos.find_clusters import find_clusters
from pycvxcluster.algos.ssnal import ssnal_wrapper
from pycvxcluster.algos.admm import admm_l2
import time


class CVXClusterAlg(BaseEstimator, ClusterMixin):
    def __init__(self) -> None:
        super().__init__()
        self.node_arc_matrix_ = None
        self.weight_vec_ = None
        self._weight_matrix_ = None

    @property
    def weight_matrix_(self):
        return self._weight_matrix_

    @weight_matrix_.setter
    def weight_matrix_(self, value):
        self.node_arc_matrix_, self.weight_vec_ = get_nam_wv_from_wm(value)
        self._weight_matrix_ = value


class SSNAL(CVXClusterAlg):
    def __init__(
        self,
        k=10,
        phi=0.5,
        gamma=1,
        clustertol=1e-5,
        sigma=1,
        maxiter=1000,
        admm_iter=100,
        stoptol=1e-6,
        ncgtolconst=0.5,
        verbose=0,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbors to use in the graph construction.
        phi : float, optional
            Parameter for the weight matrix. The default is .5.
        gamma : float, optional
            Parameter for regularization. The default is 1.
        clustertol : float, optional
            Tolerance for deciding if data points are in the same cluster. The default is 1e-5.
        sigma : float, optional
            Parameter for the objective function. The default is 1.
        maxiter : int, optional
            Maximum number of iterations. The default is 1000.
        admm_iter : int, optional
            Number of ADMM iterations to warm-start. The default is 50.
        stoptol : float, optional
            Tolerance for the stopping criterion. The default is 1e-6.
        ncgtolconst : float, optional
            Constant for the stopping criterion in ssncg. The default is 0.5.
        verbose : int, optional
            Verbosity level (0, 1, or 2). The default is 1.
        **kwargs : dict
            Keyword arguments for the SSNAL algorithm.
        """
        self.k = k
        self.phi = phi
        self.gamma = gamma
        self.clustertol = clustertol
        self.sigma = sigma
        self.maxiter = maxiter
        self.admm_iter = admm_iter
        self.stoptol = stoptol
        self.ncgtolconst = ncgtolconst
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(
        self,
        X,
        y=None,
        save_labels=True,
        save_centers=False,
        weight_matrix=None,
        recalculate_weights=True,
    ):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.
        save_labels : bool, optional
            Whether to save the cluster labels. The default is True.
        save_centers : bool, optional
            Whether to save the cluster centers. The default is False.
        weight_matrix : array-like of shape (n_samples, n_samples), optional
            Weight matrix to use. The default is None.
        Returns
        -------
        self
        """
        if recalculate_weights:
            if weight_matrix is None:
                (
                    self.weight_matrix_,
                    t1,
                ) = compute_weight_matrix(X.T, self.k, self.phi, self.verbose)
            else:
                t1s = time.perf_counter()
                self.weight_matrix_ = weight_matrix
                t1 = time.perf_counter() - t1s
        else:
            t1 = 0
        (
            self.primobj_,
            self.dualobj_,
            y,
            xi,
            z,
            self.eta_,
            _,
            self.iter_,
            self.termination_,
            t2,
        ) = ssnal_wrapper(
            X.T,
            self.weight_vec_ * self.gamma,
            self.node_arc_matrix_,
            sigma=self.sigma,
            maxiter=self.maxiter,
            admm_iter=self.admm_iter,
            stoptol=self.stoptol,
            ncgtolconst=self.ncgtolconst,
            verbose=self.verbose,
            **self.kwargs,
        )
        if save_centers:
            self.centers_ = xi
            self.y_ = y
            self.z_ = z
        t3 = 0
        if save_labels:
            t3s = time.perf_counter()
            self.labels_, self.n_clusters_ = find_clusters(xi, self.clustertol)
            t3 = time.perf_counter() - t3s
        self.ssnal_runtime_ = t2
        self.total_time_ = t1 + t2 + t3
        if self.verbose > 0:
            print(f"Clustering completed in {self.total_time_} seconds.")
        return self

    def fit_predict(self, X, y=None, **kwargs):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X, y, **kwargs)
        return self.labels_
