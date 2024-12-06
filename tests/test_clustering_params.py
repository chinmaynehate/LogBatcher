import pytest
import numpy as np
import scipy.sparse as sp
from logbatcher.compare_clustering import optimize_eps_for_dbscan, estimate_n_clusters
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def test_optimize_eps_for_dbscan():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    X_sparse = sp.csr_matrix(X)
    eps = optimize_eps_for_dbscan(X_sparse, min_samples=5)
    assert 0.05 <= eps <= 0.5

def test_estimate_n_clusters_small():
    X = np.random.rand(50, 5)
    n_clusters = estimate_n_clusters(X, max_clusters=20)
    assert n_clusters <= 20
    assert n_clusters > 1

def test_estimate_n_clusters_larger():
    X = np.random.rand(500, 5)
    n_clusters = estimate_n_clusters(X, max_clusters=20)
    assert 2 <= n_clusters <= 20
