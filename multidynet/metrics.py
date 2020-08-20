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
