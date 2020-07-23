import numpy as np


def update_tau_sq(Y, X, X_sigma, a, b):
    n_nodes = Y.shape[2]
    n_features = X.shape[2]

    a_tau_sq = a + n_nodes * n_features
    b_tau_sq = (b +
        np.trace(X_sigma[0], axis1=1, axis2=2).sum() +
        (X[0] ** 2).sum())

    return a_tau_sq, b_tau_sq


def update_sigma_sq(Y, X, X_sigma, X_cross_cov, c, d):
    n_time_steps = Y.shape[1]
    n_nodes = Y.shape[2]
    n_features = X.shape[2]

    c_sigma_sq = (c +
        n_nodes * (n_time_steps - 1) * n_features)
    d_sigma_sq = d
    for t in range(1, n_time_steps):
        d_sigma_sq += np.trace(
            X_sigma[t], axis1=1, axis2=2).sum()
        d_sigma_sq += (X[t] ** 2).sum()

        d_sigma_sq += np.trace(
            X_sigma[t-1], axis1=1, axis2=2).sum()
        d_sigma_sq += (X[t-1] ** 2).sum()

        d_sigma_sq -= 2 * np.trace(
            X_cross_cov[t-1], axis1=1, axis2=2).sum()
        d_sigma_sq -= 2 * (X[t-1] * X[t]).sum()

    return c_sigma_sq, d_sigma_sq
