import itertools

import numpy as np

from sklearn.metrics import roc_auc_score


def calculate_auc_layer(Y_true, Y_pred, test_indices=None):
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

    return roc_auc_score(y_true, y_pred)


def calculate_auc(Y_true, Y_pred, test_indices=None):
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
                subset = test_indices[k, t]
            y_true.extend(y_true_vec[subset])
            y_pred.extend(y_pred_vec[subset])

    return roc_auc_score(y_true, y_pred)


def score_latent_space(X_true, X_pred):
    """The estimated latent space is still invariant to column permutations and
    sign flips. To fix these we do an exhaustive search over all permutations
    and sign flips and return the value with the lowest MSE."""
    n_features = X_true.shape[2]
    best_mse = np.inf
    best_perm = None
    for perm in itertools.permutations(np.arange(n_features)):
        X = X_pred[..., perm]

        # no flip
        mse = np.mean((X_true - X_pred) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_perm = perm

        # loops through single feature flips
        for p in range(n_features):
            Xp = X.copy()
            Xp[..., p] = -X[..., p]
            mse = np.mean((X_true - Xp) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_perm = perm

        # loop through all feature combinations
        for k in range(2, n_features + 1):
            for combo in itertools.combinations(range(n_features), k):
                Xp = X.copy()
                Xp[..., combo] = -X[..., combo]
                mse = np.mean((X_true - Xp) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_perm = perm

    return best_mse, best_perm


def score_latent_space_t(X_true, X_pred, perm):
    """The estimated latent space is still invariant to column permutations and
    sign flips. To fix these we do an exhaustive search over all permutations
    and sign flips and return the value with the lowest MSE."""
    n_features = X_true.shape[1]
    X = X_pred[..., perm]

    # no flip
    best_mse = np.mean((X_true - X) ** 2)

    # loops through single feature flips
    for p in range(n_features):
        Xp = X.copy()
        Xp[..., p] = -X[..., p]
        mse = np.mean((X_true - Xp) ** 2)
        if mse < best_mse:
            best_mse = mse

    # loop through all feature combinations
    for k in range(2, n_features + 1):
        for combo in itertools.combinations(range(n_features), k):
            Xp = X.copy()
            Xp[..., combo] = -X[..., combo]
            mse = np.mean((X_true - Xp) ** 2)
            if mse < best_mse:
                best_mse = mse

    return best_mse


def score_latent_space_individual(X_true, X_pred):
    """The estimated latent space is still invariant to column permutations and
    sign flips. To fix these we do an exhaustive search over all permutations
    and sign flips and return the value with the lowest MSE.

    NOTE: This function allows the flips and perms to be different over all
    time-points
    """
    n_time_steps, _, n_features = X_true.shape
    best_mse = np.inf
    best_perm = None
    for perm in itertools.permutations(np.arange(n_features)):
        mse = 0
        for t in range(X_true.shape[0]):
            mse += score_latent_space_t(X_true[t], X_pred[t], perm)
        mse /= n_time_steps
        if mse < best_mse:
            best_mse = mse
            best_perm = perm

    return best_mse, best_perm
