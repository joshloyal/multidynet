import numpy as np


def update_tau_sq(Y, X, X_sigma, a, b):
    _, n_nodes, n_features = X.shape

    a_tau_sq = a[0] + n_nodes * n_features
    b_tau_sq = (b[0] +
        np.trace(X_sigma[0], axis1=1, axis2=2).sum() +
        (X[0] ** 2).sum())

    return np.repeat(a_tau_sq, n_features), np.repeat(b_tau_sq, n_features)


def update_diag_tau_sq(Y, X, X_sigma, a, b):
    _, n_nodes, n_features = X.shape

    a_tau_sq = np.full(n_features, a + n_nodes)
    b_tau_sq = (b + np.diagonal(X_sigma[0], axis1=2, axis2=1).sum(axis=0) +
            (X[0] ** 2).sum(axis=0))

    return a_tau_sq, b_tau_sq


def update_X0_precision(Y, X, X_sigma, df, scale):
    _, n_nodes, n_features = X.shape

    X0_cov_df = df + n_nodes
    X0_cov_scale = scale + (X_sigma[0].sum(axis=0) + X[0].T @ X[0])

    return X0_cov_df, np.linalg.pinv(X0_cov_scale)


def update_tau_sq_delta(delta, delta_sigma, a, b):
    n_layers, _, n_nodes = delta.shape
    a_tau_sq = a + n_nodes * n_layers
    b_tau_sq = (b +
        (delta_sigma[:, 0, :] + delta[:, 0, :] ** 2).sum())

    return a_tau_sq, b_tau_sq


def update_sigma_sq(Y, X, X_sigma, X_cross_cov, c, d):
    n_time_steps, n_nodes, n_features = X.shape

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


def update_sigma_sq_delta(delta, delta_sigma, delta_cross_cov, c, d):
    n_layers, n_time_steps, n_nodes = delta.shape

    c_sigma_sq = (c +
        n_nodes * (n_time_steps - 1) * n_layers)
    d_sigma_sq = d
    for t in range(1, n_time_steps):
        d_sigma_sq += delta_sigma[:, t, :].sum()
        d_sigma_sq += (delta[:, t, :] ** 2).sum()

        d_sigma_sq += delta_sigma[:, t-1, :].sum()
        d_sigma_sq += (delta[:, t-1, :] ** 2).sum()

        d_sigma_sq -= 2 * delta_cross_cov[:, t-1, :].sum()
        d_sigma_sq -= 2 * (delta[:, t-1, :] * delta[:, t, :]).sum()

    return c_sigma_sq, d_sigma_sq
