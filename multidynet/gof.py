import networkx as nx
import numpy as np

from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
from scipy.special import expit
from sklearn.utils import check_random_state

from .sample_lds import sample_gssm


__all__ = ['sample_connection_probabilities', 'simulate_network',
           'simulate_univariate_stats']


def max_degree(G):
    return max(dict(nx.degree(G)).values())


def avg_degree(G):
    return np.mean(list(dict(nx.degree(G)).values()))


def var_degree(G):
    return np.var(list(dict(nx.degree(G)).values()))


def branching_factor(G):
    degrees = np.asarray(list(dict(nx.degree(G)).values()))
    d_mean = np.mean(degrees)
    d_sq_mean = np.mean(degrees ** 2)
    return d_sq_mean / d_mean


def shortest_path(G):
    res = []
    for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
        res.append(nx.average_shortest_path_length(C))

    return np.mean(res)


def sample_connection_probabilities(model, random_state=None):
    n_layers, n_time_steps, n_nodes, _ = model.probas_.shape
    n_features = model.X_.shape[2]

    rng = check_random_state(random_state)

    probas = np.zeros_like(model.probas_)
    for t in range(n_time_steps):
        X = np.zeros((n_nodes, n_features))
        for i in range(n_nodes):
            X[i] = multivariate_normal.rvs(
                mean=model.X_[t, i], cov=model.X_sigma_[t, i],
                random_state=rng)

        for k in range(n_layers):
            delta = multivariate_normal.rvs(
                mean=model.delta_[k, t], cov=model.delta_sigma_[k, t],
                random_state=rng).reshape(-1, 1)

            if k == 0:
                lmbda = 2 * rng.binomial(1, p=model.lambda_proba_) - 1
            else:
                lmbda = multivariate_normal.rvs(
                    mean=model.lambda_[k], cov=model.lambda_sigma_[k],
                    random_state=rng)

            probas[k, t] = expit(delta + delta.T + np.dot(X * lmbda, X.T))

    return probas


def estimate_connection_probabilities(model, n_samples=200, n_jobs=-1,
                                      random_state=None):
    rng = check_random_state(random_state)

    # XXX: the result can take up a lot of memory....
    seeds = rng.randint(np.iinfo(np.int32).max, size=n_samples)
    results = Parallel(n_jobs=n_jobs)(
        delayed(sample_connection_probabilities)(model, seed) for
        seed in seeds)

    return np.asarray(results).mean(axis=0)


def simulate_network(model, random_state=None):
    n_layers, n_time_steps, n_nodes, _ = model.probas_.shape
    n_features = model.X_.shape[2]

    rng = check_random_state(random_state)

    Y = np.zeros_like(model.probas_)
    tril_indices = np.tril_indices_from(Y[0, 0], k=-1)
    for t in range(n_time_steps):
        X = np.zeros((n_nodes, n_features))
        for i in range(n_nodes):
            X[i] = multivariate_normal.rvs(
                mean=model.X_[t, i], cov=model.X_sigma_[t, i],
                random_state=rng)

        for k in range(n_layers):
            delta = multivariate_normal.rvs(
                mean=model.delta_[k, t], cov=model.delta_sigma_[k, t],
                random_state=rng).reshape(-1, 1)

            if k == 0:
                lmbda = 2 * rng.binomial(1, p=model.lambda_proba_) - 1
            else:
                lmbda = multivariate_normal.rvs(
                    mean=model.lambda_[k], cov=model.lambda_sigma_[k],
                    random_state=rng)

            prob = expit(delta + delta.T + np.dot(X * lmbda, X.T))
            y_vec = rng.binomial(1, p=prob[tril_indices])
            Y[k, t][tril_indices] = y_vec
            Y[k, t] = Y[k, t].T

    return Y


def simulate_network_gssm(model, random_state=None):
    n_layers, n_time_steps, n_nodes, _ = model.probas_.shape
    n_features = model.X_.shape[2]

    rng = check_random_state(random_state)

    Y = np.zeros_like(model.probas_)
    tril_indices = np.tril_indices_from(Y[0, 0], k=-1)

    X = np.zeros((n_time_steps, n_nodes, n_features))
    delta = np.zeros((n_layers, n_time_steps, n_nodes))
    for i in range(n_nodes):
        X[:, i] = sample_gssm(model.X_[:, i], model.X_sigma_[:, i],
                          model.X_cross_cov_[:, i], random_state=rng)

        for k in range(n_layers):
            delta[k, :, i] = sample_gssm(
                model.delta_[k, :, i], model.delta_sigma_[k, :, i],
                model.delta_cross_cov_[k, :, i], random_state=rng)

    lmbda = np.zeros((n_layers, n_features))
    for k in range(n_layers):
        if k == 0:
            lmbda[k] = 2 * rng.binomial(1, p=model.lambda_proba_) - 1
        else:
            lmbda[k] = multivariate_normal.rvs(
                mean=model.lambda_[k], cov=model.lambda_sigma_[k],
                random_state=rng)

    for t in range(n_time_steps):
        for k in range(n_layers):
            deltakt = delta[k, t].reshape(-1, 1)
            prob = expit(deltakt + deltakt.T + np.dot(X[t] * lmbda[k], X[t].T))
            y_vec = rng.binomial(1, p=prob[tril_indices])
            Y[k, t][tril_indices] = y_vec
            Y[k, t] = Y[k, t].T

    return Y


NET_STATS = {
    'density' : nx.density,
    'avg_degree': avg_degree,
    'max_degree': max_degree,
    'branching_factor': branching_factor,
    'avg_clustering': nx.average_clustering,
    'avg_shortest_path': shortest_path
}


def simulate_univariate_stats(model, Y_obs=None, n_sims=100, stats=None,
                              random_state=None):
    """Sample univariate statistics from the eigenmodel's approximate posterior.

    Parameters
    ----------
    model : DynamicMultilayerNetworkLSM
        Fitted estimator.

    Y_obs : array-like, shape (n_layers, n_time_steps, n_nodes, n_nodes)
        The training dynamic multilayer network. The networks should be
        represented as binary undirected adjacency matrices. For example,
        Y[0] is an array of shape (n_time_steps, n_nodes, n_nodes)
        corresponding to the adjacency matrices of the networks at in
        the first layer. The network should be stored as
        ``dtype=np.float64``.

    n_sims : int, (default=100)
        Number of sample networks to draw from the approximate posterior.

    stats : list of length n_stats, (default=None)
        List of univariate statistics to calculate. If a list of strings,
        they must take values {`density`, `avg_degree`, `max_degree`,
        `branching_factor`, `avg_clustring`, `avg_shortest_path`}. Can
        also be a list of functions that take in a networkx.Graph and
        return a scalar network statistic.

    random_state : int, RandomState instance or None (default=None)
        Controls the random seed used to generate random samples
        from the fitted posterior distribution. Pass an int for reproducible
        output across multiple function calls.

    Returns
    -------

    stat_sim : np.ndarray of shape (n_stats, n_layers, n_time_steps, n_sims)
        The values of each network statistic calculated on each simulated
        network.

    stat_obs : np.ndarray of shape (n_stats, n_layers, n_time_steps)
        The values of each network statistic calculated on the observed
        network Y_obs.
    """
    random_state = check_random_state(random_state)

    n_layers, n_time_steps, _, _ = model.probas_.shape

    if stats is None:
        stats = [nx.density, max_degree, avg_degree]
    elif stats == 'all':
        stats = list(NET_STATS.values())
    else:
        stats = [NET_STATS[stat] if stat in NET_STATS else stat for
                 stat in stats]

    n_stats = len(stats)

    # TODO: Run in parallel
    stat_sim = np.zeros((n_stats, n_layers, n_time_steps, n_sims))
    for i in range(n_sims):
        y_sim = simulate_network_gssm(model, random_state=random_state)
        for k in range(n_layers):
            for t in range(n_time_steps):
                G = nx.from_numpy_array(y_sim[k, t])
                for s, stat in enumerate(stats):
                    stat_sim[s, k, t, i] = stat(G)

    # calculate stats on observed network
    if Y_obs is not None:
        stat_obs = np.zeros((n_stats, n_layers, n_time_steps))
        for k in range(n_layers):
            for t in range(n_time_steps):
                G = nx.from_numpy_array(Y_obs[k, t])
                for s, stat in enumerate(stats):
                    stat_obs[s, k, t] = stat(G)

        return np.squeeze(stat_sim), np.squeeze(stat_obs)

    return np.squeeze(stat_sim)
