import numpy as np

from math import ceil

from sklearn.utils import check_random_state
from sklearn.model_selection import KFold


def train_test_split_dyads(Y, test_size=0.1, random_state=None):
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

            perm = random_state.choice(
                np.arange(n_dyads), size=n_test, replace=False)
            test_indices[k, t] = perm

            Y_vec = Y[k, t][tril_indices]
            Y_vec[perm] = -1.0
            Y_new[k, t][tril_indices] = Y_vec
            Y_new[k, t] += Y_new[k, t].T

    return Y_new, test_indices


def train_test_split_nodes(Y, test_size=0.1, random_state=None):
    n_layers, n_time_steps, n_nodes, _ = Y.shape

    random_state = check_random_state(random_state)

    test_size_type = np.asarray(test_size).dtype.kind
    if test_size_type == 'f':
        n_test = ceil(test_size * n_nodes)
    else:
        n_test = int(test_size)

    # for each layer randomly remove n_test nodes
    test_nodes = np.zeros((n_layers, n_test), dtype=np.int64)
    for k in range(n_layers):
        test_nodes[k] = random_state.choice(
            np.arange(n_nodes), size=n_test, replace=False)
    n_indices = int(n_test * n_nodes - 0.5 * n_test * (n_test + 1))
    test_indices = np.zeros((n_layers, n_time_steps, n_indices), dtype=np.int64)

    # apply the mask
    Y_new = np.zeros_like(Y)
    for t in range(n_time_steps):

        # re-sample nodes for second half of data
        #if t == int(n_time_steps / 2):
        #    for k in range(n_layers):
        #        test_nodes[k] = random_state.choice(
        #            np.arange(n_nodes), size=n_test, replace=False)

        for k in range(n_layers):
            tril_indices = np.tril_indices_from(Y[k, t], k=-1)
            Y_new[k, t] = Y[k, t].copy()
            for i in test_nodes[k]:
                Y_new[k, t, :, i] = -1
                Y_new[k, t, i, :] = -1
            Y_new[k, t][np.diag_indices_from(Y_new[k, t])] = 0.

            test_indices[k, t] = np.where(Y_new[k, t][tril_indices] == -1)[0]

    return Y_new, test_indices


def train_test_split(Y, test_size=0.1, test_type='dyads', random_state=None):
    if test_type == 'dyads':
        return train_test_split_dyads(
            Y, test_size=test_size, random_state=random_state)
    elif test_type == 'nodes':
        return train_test_split_nodes(
            Y, test_size=test_size, random_state=random_state)
    else:
        raise ValueError("test_type == {}, when it shoulde be 'dyads' or "
                         "'nodes'".format(test_type))


def mask_nodes(Y, node_list):
    n_time_steps, n_nodes, _ = Y.shape

    # initialize arrays holding masked dyads
    n_mask = node_list.shape[0]
    n_indices = int(n_mask * n_nodes - 0.5 * n_mask * (n_mask + 1))
    masked_indices = np.zeros((n_time_steps, n_indices), dtype=np.int64)

    # apply the mask
    Y_new = np.zeros_like(Y)
    for t in range(n_time_steps):
        tril_indices = np.tril_indices_from(Y[t], k=-1)
        Y_new[t] = Y[t].copy()
        for i in node_list:
            Y_new[t, :, i] = -1
            Y_new[t, i, :] = -1
        Y_new[t][np.diag_indices_from(Y_new[t])] = 0.

        masked_indices[t] = np.where(Y_new[t][tril_indices] == -1)[0]

    return Y_new, masked_indices
