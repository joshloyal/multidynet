import numpy as np

from math import ceil

from sklearn.utils import check_random_state


def train_test_split(Y, test_size=0.1, random_state=None):
    """Split dyads into training and testing subsets.

    Parameters
    ----------
    Y : array-like, shape  (n_layers, n_time_steps, n_nodes, n_nodes)
    """
    n_layers, n_time_steps, n_nodes, _ = Y.shape

    random_state = check_random_state(random_state)

    # number of dyads in an undirected graph with n_nodes nodes
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    test_size_type = np.asarray(test_size).dtype.kind
    if test_size_type == 'f':
        n_test = ceil(test_size * n_dyads)
    else:
        n_test = int(test_size)

    Y_new = np.zeros_like(Y)
    test_indices = np.zeros((n_layers, n_time_steps, n_test), dtype=np.int64)
    for k in range(n_layers):
        for t in range(n_time_steps):
            tril_indices = np.tril_indices_from(Y[k, t], k=-1)

            perm = random_state.choice(np.arange(n_dyads), size=n_test)
            test_indices[k, t] = perm

            Y_vec = Y[k, t][tril_indices]
            Y_vec[perm] = -1.0
            Y_new[k, t][tril_indices] = Y_vec
            Y_new[k, t] += Y_new[k, t].T

    return Y_new, test_indices
