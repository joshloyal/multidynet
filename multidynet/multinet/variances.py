import numpy as np


def update_tau_sq(Y, X, X_sigma, a, b):
    n_nodes, n_features = X.shape

    a_tau_sq = a + n_nodes * n_features
    b_tau_sq = (b +
        np.trace(X_sigma, axis1=1, axis2=2).sum() +
        (X ** 2).sum())

    return a_tau_sq, b_tau_sq


def update_diag_tau_sq(Y, X, X_sigma, a, b):
    n_nodes, n_features = X.shape

    a_tau_sq = np.full(n_features, a + n_nodes)
    b_tau_sq = (b + np.diagonal(X_sigma, axis1=2, axis2=1).sum(axis=0) +
            (X ** 2).sum(axis=0))

    return a_tau_sq, b_tau_sq


def update_X0_precision(Y, X, X_sigma, df, scale):
    n_nodes, n_features = X.shape

    X0_cov_df = df + n_nodes
    X0_cov_scale = scale + (X_sigma.sum(axis=0) + X.T @ X)

    return X0_cov_df, np.linalg.pinv(X0_cov_scale)


def update_tau_sq_delta(delta, delta_sigma, a, b):
    n_layers, n_nodes = delta.shape
    a_tau_sq = a + n_nodes * n_layers
    b_tau_sq = (b + (delta_sigma + delta ** 2).sum())

    return a_tau_sq, b_tau_sq
