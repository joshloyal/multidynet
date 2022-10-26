import itertools

import numpy as np

from scipy.stats import pearsonr
from scipy.special import logit
from sklearn.metrics import roc_auc_score


def calculate_metric_single(Y_true, Y_pred, metric, test_indices=None):
    n_time_steps, n_nodes, _ = Y_true.shape
    indices = np.tril_indices_from(Y_true[0], k=-1)

    y_true = []
    y_pred = []
    for t in range(n_time_steps):
        y_true_vec = Y_true[t][indices]
        y_pred_vec = Y_pred[t][indices]

        if test_indices is None:
            subset = y_true_vec != -1.0
        else:
            subset = test_indices[t]
        y_true.extend(y_true_vec[subset])
        y_pred.extend(y_pred_vec[subset])

    return metric(y_true,  y_pred)


def calculate_metric(Y_true, Y_pred, metric, test_indices=None):
    if Y_true.ndim == 3:
        return calculate_metric_single(
                Y_true, Y_pred, metric, test_indices=test_indices)

    n_layers, n_time_steps, n_nodes, _ = Y_true.shape
    indices = np.tril_indices_from(Y_true[0, 0], k=-1)

    y_true = []
    y_pred = []
    for k in range(n_layers):
        for t in range(n_time_steps):
            y_true_vec = Y_true[k, t][indices]
            y_pred_vec = Y_pred[k, t][indices]

            if test_indices is None:
                subset = y_true_vec != -1.0
            else:
                subset = test_indices[k][t]

            y_true.extend(y_true_vec[subset])
            y_pred.extend(y_pred_vec[subset])

    return metric(y_true, y_pred)


def calculate_auc_single(Y_true, Y_pred, test_indices=None):
    n_time_steps, n_nodes, _ = Y_true.shape
    indices = np.tril_indices_from(Y_true[0], k=-1)

    y_true = []
    y_pred = []
    for t in range(n_time_steps):
        y_true_vec = Y_true[t][indices]
        y_pred_vec = Y_pred[t][indices]

        if test_indices is None:
            subset = y_true_vec != -1.0
        else:
            subset = test_indices[t]
        y_true.extend(y_true_vec[subset])
        y_pred.extend(y_pred_vec[subset])

    return roc_auc_score(y_true,  y_pred)


def calculate_auc(Y_true, Y_pred, test_indices=None):
    if Y_true.ndim == 3:
        return calculate_auc_single(Y_true, Y_pred, test_indices=test_indices)

    n_layers, n_time_steps, n_nodes, _ = Y_true.shape
    indices = np.tril_indices_from(Y_true[0, 0], k=-1)

    y_true = []
    y_pred = []
    for k in range(n_layers):
        for t in range(n_time_steps):
            y_true_vec = Y_true[k, t][indices]
            y_pred_vec = Y_pred[k, t][indices]

            if test_indices is None:
                subset = y_true_vec != -1.0
            else:
                subset = test_indices[k][t]

            y_true.extend(y_true_vec[subset])
            y_pred.extend(y_pred_vec[subset])

    return roc_auc_score(y_true, y_pred)


def calculate_correlation_single(Y_true, Y_pred, test_indices=None):
    n_time_steps, n_nodes, _ = Y_true.shape
    indices = np.tril_indices_from(Y_true[0], k=-1)

    y_true = []
    y_pred = []
    for t in range(n_time_steps):
        y_true_vec = Y_true[t][indices]
        y_pred_vec = Y_pred[t][indices]

        if test_indices is None:
            subset = y_true_vec != -1.0
        else:
            subset = test_indices[t]
        y_true.extend(y_true_vec[subset])
        y_pred.extend(y_pred_vec[subset])

    return pearsonr(y_true, y_pred)[0]


def calculate_correlation(Y_true, Y_pred, test_indices=None):
    if Y_true.ndim == 3:
        return calculate_correlation_single(
                Y_true, Y_pred, test_indices=test_indices)

    n_layers, n_time_steps, n_nodes, _ = Y_true.shape
    indices = np.tril_indices_from(Y_true[0, 0], k=-1)

    y_true = []
    y_pred = []
    for k in range(n_layers):
        for t in range(n_time_steps):
            y_true_vec = Y_true[k, t][indices]
            y_pred_vec = Y_pred[k, t][indices]

            if test_indices is None:
                subset = y_true_vec != -1.0
            else:
                subset = test_indices[k][t]

            y_true.extend(y_true_vec[subset])
            y_pred.extend(y_pred_vec[subset])

    return pearsonr(y_true, y_pred)[0]



def calculate_eta(X, lmbda, delta):
    n_layers = delta.shape[0]
    n_time_steps = delta.shape[1]
    n_nodes = delta.shape[2]

    eta = np.zeros(
        (n_layers, n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    for k in range(n_layers):
        for t in range(n_time_steps):
            deltakt = delta[k, t].reshape(-1, 1)
            eta_kt = np.add(deltakt, deltakt.T)
            if X is not None:
                eta_kt += np.dot(X[t] * lmbda[k], X[t].T)
            eta[k, t] = eta_kt

    return eta


def calculate_lpp(Y, model, test_indices):
    n_layers, n_time_steps, _, _ = Y.shape
    eta = calculate_eta(model.X_, model.lambda_, model.delta_)

    lpp = 0.
    indices = np.tril_indices_from(Y[0, 0], k=-1)
    for k in range(n_layers):
        for t in range(n_time_steps):
           y_vec = Y[k, t][indices][test_indices[k, t]]
           eta_vec = eta[k, t][indices][test_indices[k, t]]
           lpp += (y_vec * eta_vec).sum()
           lpp -= np.logaddexp(np.ones(eta_vec.shape[0]), eta_vec).sum()

    return lpp


def score_latent_space_t(X_true, X_pred, perm):
    """The estimated latent space is still invariant to column permutations and
    sign flips. To fix these we do an exhaustive search over all permutations
    and sign flips and return the value with the lowest MSE."""
    n_features = X_true.shape[1]
    X = X_pred[..., perm]
    denom = np.sum(X_true ** 2)

    # no flip
    best_rel = np.sum((X_true - X) ** 2) / denom

    # loops through single feature flips
    for p in range(n_features):
        Xp = X.copy()
        Xp[..., p] = -X[..., p]
        rel = np.sum((X_true - Xp) ** 2) / denom
        if rel < best_rel:
            best_rel = rel

    # loop through all feature combinations
    for k in range(2, n_features + 1):
        for combo in itertools.combinations(range(n_features), k):
            Xp = X.copy()
            Xp[..., combo] = -X[..., combo]
            rel = np.sum((X_true - Xp) ** 2) / denom
            if rel < best_rel:
                best_rel = rel

    return best_rel


def score_latent_space_single_perml(X_true, X_pred):
    """The estimated latent space is still invariant to column permutations and
    sign flips. To fix these we do an exhaustive search over all permutations
    and sign flips and return the value with the lowest MSE.

    NOTE: This function allows the flips and perms to be different over all
    time-points
    """
    n_time_steps, _, n_features = X_true.shape
    best_rel = np.inf
    best_perm = None
    for perm in itertools.permutations(np.arange(n_features)):
        rel = 0
        for t in range(X_true.shape[0]):
            rel += score_latent_space_t(X_true[t], X_pred[t], perm)
        rel /= n_time_steps
        if rel < best_rel:
            best_rel = rel
            best_perm = perm

    return best_rel, best_perm


def score_latent_space(X_true, X_pred):
    n_time_steps, _, n_features = X_true.shape
    rel = 0
    for t in range(X_true.shape[0]):
        best_rel = np.inf
        for perm in itertools.permutations(np.arange(n_features)):
            rel_t = score_latent_space_t(X_true[t], X_pred[t], perm)
            if rel_t < best_rel:
                best_rel = rel_t
        rel += best_rel
    rel /= n_time_steps

    return rel


def score_homophily_matrix(lambda_true, lambda_pred):
    n_features = lambda_true.shape[1]

    best_rel = np.inf
    for perm in itertools.permutations(np.arange(n_features)):
        rel = np.sum((lambda_true - lambda_pred[:, perm]) ** 2)
        rel /= np.sum(lambda_true ** 2)
        if rel < best_rel:
            best_rel = rel

    return best_rel


def score_social_trajectories(delta_true, delta_pred):
    n_layers, n_time_steps, n_nodes = delta_true.shape
    num, dem = 0., 0.

    indices = np.tril_indices(n_nodes, k=-1)
    for k in range(n_layers):
        for t in range(n_time_steps):
            d_hat = delta_pred[k, t].reshape(-1, 1)
            D_hat = d_hat - d_hat.T

            d_true = delta_true[k, t].reshape(-1, 1)
            D_true = d_true - d_true.T

            num += np.sum((D_true[indices] - D_hat[indices]) ** 2)
            dem += np.sum(D_true[indices] ** 2)

    return num / dem
