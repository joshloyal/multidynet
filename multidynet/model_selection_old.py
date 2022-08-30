import numpy as np

from math import ceil

from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.model_selection import KFold


from .multidynet import DynamicMultilayerNetworkLSM


MAX_INT = np.iinfo(np.int32).max


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


def train_test_split_nodes(Y, test_nodes=None, test_size=0.1, random_state=None):
    """
    Single split of network cross-validation technique of Chen and Li (2018)
    """
    n_layers, n_time_steps, n_nodes, _ = Y.shape

    random_state = check_random_state(random_state)

    if test_nodes is None:
        test_size_type = np.asarray(test_size).dtype.kind
        if test_size_type == 'f':
            n_test = ceil(test_size * n_nodes)
        else:
            n_test = int(test_size)

        # randomly sample n_test nodes
        test_nodes = random_state.choice(
            np.arange(n_nodes), size=n_test, replace=False)
    else:
        n_test = test_nodes.shape[0]

    test_mask = (test_nodes[:, np.newaxis], test_nodes[np.newaxis, :])

    # store test dyads
    n_indices = int(0.5 * n_test * (n_test - 1))
    test_indices = np.zeros((n_layers, n_time_steps, n_indices), dtype=np.int64)

    # apply the mask
    Y_new = np.zeros_like(Y)
    tril_indices = np.tril_indices_from(Y_new[0, 0], k=-1)
    for t in range(n_time_steps):
        for k in range(n_layers):
            Y_new[k, t] = Y[k, t].copy()
            Y_new[k, t][test_mask] = -1
            Y_new[k, t][np.diag_indices(n_nodes)] = 0
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


def network_cross_validation(Y, n_folds=4, random_state=None):
    rng = check_random_state(random_state)

    kfolds = KFold(n_splits=n_folds, shuffle=True, random_state=rng)
    for _, test_nodes in kfolds.split(np.arange(Y.shape[2])):
        yield train_test_split_nodes(Y, test_nodes=test_nodes, random_state=rng)


def fit_single(Y, n_features, model_params, random_state=None):
    model_params['n_features'] = n_features
    model_params['random_state'] = random_state

    return DynamicMultilayerNetworkLSM(**model_params).fit(Y)



def select_dimension(Y, n_features=None, model_params=None,
                     n_jobs=1, random_state=None):
    random_state = check_random_state(random_state)

    n_features = np.arange(6) if n_features is None else n_features
    model_params = dict() if model_params is None else model_params

    seeds = random_state.randint(MAX_INT, size=len(n_features))

    models = Parallel(n_jobs=n_jobs)(delayed(fit_single)(
        Y, n_features=n_features[d], model_params=model_params.copy(),
        random_state=seed) for d, seed in enumerate(seeds))

    return models
