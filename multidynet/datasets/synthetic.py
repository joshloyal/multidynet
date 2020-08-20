"""
Generate synthetic networks
"""
import numpy as np

from scipy.special import expit
from sklearn.utils import check_random_state


__all__ = ['simple_dynamic_multilayer_network', 'simple_dynamic_network']


def multilayer_network_from_dynamic_latent_space(X, lmbda, delta,
                                                 random_state=None):
    rng = check_random_state(random_state)

    n_time_steps, n_nodes, _ = X.shape
    n_layers = lmbda.shape[0]

    if delta is None:
        delta = np.zeros((n_layers, n_nodes), dtype=np.float64)

    Y = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    probas = np.zeros(
        (n_layers, n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    for k in range(n_layers):
        for t in range(n_time_steps):
            # sample the adjacency matrix
            deltak = delta[k].reshape(-1, 1)
            eta = np.add(deltak, deltak.T) + np.dot(X[t] * lmbda[k], X[t].T)
            probas[k, t] = expit(eta)

            Y[k, t] = rng.binomial(1, probas[k, t]).astype(np.int)

            # make symmetric
            Y[k, t] = np.tril(Y[k, t], k=-1)
            Y[k, t] += Y[k, t].T

    return Y


def simple_dynamic_multilayer_network(n_nodes=100, n_time_steps=4,
                                      n_features=2, tau_sq=1.0, sigma_sq=0.05,
                                      lmbda_scale=1.0,
                                      lmbda=None,
                                      assortative_reference=True,
                                      random_state=42):
    rng = check_random_state(random_state)

    # construct latent features
    X = np.zeros((n_time_steps, n_nodes, n_features), dtype=np.float64)
    X[0] = np.sqrt(tau_sq) * rng.randn(n_nodes, n_features)
    for t in range(1, n_time_steps):
        X[t] = X[t-1] + np.sqrt(sigma_sq) * rng.randn(n_nodes, n_features)

    # assortative and dissassortative layers
    if lmbda is None:
        n_layers = 4
        lmbda = np.zeros((n_layers, n_features))

        if assortative_reference:
            lmbda[0] = np.array([1., 1.])
        else:
            lmbda[0] = -np.array([1., 1.])
        lmbda[1] = lmbda_scale * lmbda[0]
        lmbda[2] = -lmbda_scale * lmbda[0]
        lmbda[3] = -lmbda[0]
    else:
        n_layers = lmbda.shape[0]

    # degree effects
    delta = rng.randn(n_layers, n_nodes)

    # construct the network
    Y = multilayer_network_from_dynamic_latent_space(
        X, lmbda, delta, random_state=rng)

    return Y, X, lmbda, delta


def network_from_dynamic_latent_space(X, delta, random_state=None):
    rng = check_random_state(random_state)

    n_time_steps, n_nodes, _ = X.shape
    Y = np.zeros((n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    probas = np.zeros(
        (n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    deltat = delta.reshape(-1, 1)
    for t in range(n_time_steps):
        # sample the adjacency matrix
        eta = np.add(deltat, deltat.T) + np.dot(X[t], X[t].T)
        probas[t] = expit(eta)

        Y[t] = rng.binomial(1, probas[t]).astype(np.int)

        # make symmetric
        Y[t] = np.tril(Y[t], k=-1)
        Y[t] += Y[t].T

    return Y


def simple_dynamic_network(n_nodes=100, n_time_steps=4,
                           n_features=2, tau_sq=1.0, sigma_sq=0.05,
                           random_state=42):
    rng = check_random_state(random_state)

    # construct latent features
    X = np.zeros((n_time_steps, n_nodes, n_features), dtype=np.float64)
    X[0] = np.sqrt(tau_sq) * rng.randn(n_nodes, n_features)
    for t in range(1, n_time_steps):
        X[t] = X[t-1] + np.sqrt(sigma_sq) * rng.randn(n_nodes, n_features)

    # degree effects
    delta = rng.randn(n_nodes)

    # construct the network
    Y = network_from_dynamic_latent_space(
        X, delta, random_state=rng)

    return Y, X, delta
