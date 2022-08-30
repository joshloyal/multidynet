import numpy as np

from sklearn.model_selection import KFold, ShuffleSplit


def dynamic_multilayer_adjacency_to_vec(Y):
    n_layers, n_time_steps, n_nodes, _ = Y.shape
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    subdiag = np.tril_indices(n_nodes, k=-1)
    y = np.zeros((n_layers, n_time_steps, n_dyads), dtype=np.float64)
    for k in range(n_layers):
        for t in range(n_time_steps):
            y[k, t] = Y[k, t][subdiag]

    return y


def kfold(Y, n_splits=4, random_state=None):
    """Split dyads into k-folds. A dyad is removed from all time steps and layers.

    Parameters
    ----------
    Y : array-like, shape  (n_layers, n_time_steps, n_nodes, n_nodes)
    """
    n_layers, n_time_steps, n_nodes, _ = Y.shape
    y = dynamic_multilayer_adjacency_to_vec(Y)

    tril_indices = np.tril_indices(n_nodes, k=-1)
    kfolds = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train, test in kfolds.split(y[0, 0]):
        Y_new = np.zeros_like(Y)
        test_indices = []
        for k in range(n_layers):
            test_indices.append([])
            for t in range(n_time_steps):
                test_indices[k].append(test)

                y_vec = np.copy(y[k, t])
                y_vec[test] = -1.0
                Y_new[k, t][tril_indices] = y_vec
                Y_new[k, t] += Y_new[k, t].T

        yield Y_new, test_indices


def train_test_split(Y, test_size=0.2, random_state=None):
    n_layers, n_time_steps, n_nodes, _ = Y.shape
    y = dynamic_multilayer_adjacency_to_vec(Y)

    tril_indices = np.tril_indices(n_nodes, k=-1)
    Y_new = np.zeros_like(Y)
    test_indices = []

    rs = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    _, test = next(rs.split(y[0, 0]))
    for k in range(n_layers):
        test_indices.append([])
        for t in range(n_time_steps):
            test_indices[k].append(test)

            y_vec = np.copy(y[k, t])
            y_vec[test] = -1.0
            Y_new[k, t][tril_indices] = y_vec
            Y_new[k, t] += Y_new[k, t].T

    return Y_new, test_indices
