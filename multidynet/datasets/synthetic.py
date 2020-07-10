"""
Generate synthetic networks
"""
import numpy as np

from scipy.special import expit
from sklearn.utils import check_random_state


__all__ = ['simple_dynamic_multilayer_network']


def network_from_dynamic_latent_space(X, lmbda, intercept=1, random_state=None):
    rng = check_random_state(random_state)

    n_time_steps, n_nodes, _ = X.shape
    n_layers = lmbda.shape[0]
    Y = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    probas = np.zeros(
        (n_layers, n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    for k in range(n_layers):
        for t in range(n_time_steps):
            # sample the adjacency matrix
            eta = intercept[k] + np.dot(X[t] * lmbda[k], X[t].T)
            probas[k, t] = expit(eta)

            Y[k, t] = rng.binomial(1, probas[k, t]).astype(np.int)

            # make symmetric
            Y[k, t] = np.triu(Y[k, t], 1)
            Y[k, t] += Y[k, t].T

    return Y, X


def simple_dynamic_multilayer_network(n_nodes=100, n_time_steps=4,
                                      n_features=2, tau_sq=5, sigma_sq=0.1,
                                      intercept=1.0, random_state=42):
    rng = check_random_state(random_state)

    # construct latent features
    X = np.zeros((n_time_steps, n_nodes, n_features), dtype=np.float64)
    X[0] = np.sqrt(tau_sq) * rng.randn(n_nodes, n_features)
    for t in range(1, n_time_steps):
        X[t] = X[t-1] + np.sqrt(sigma_sq) * rng.randn(n_nodes, n_features)

    # assortative and dissassortative layers
    n_layers = 2
    lmbda = np.zeros((n_layers, n_features))
    lmbda[0] = np.ones(n_features)
    lmbda[1] = -np.ones(n_features)

    if not isinstance(intercept, np.ndarray):
        intercept = np.repeat(intercept, n_layers)

    # construct the network
    return network_from_dynamic_latent_space(
        X, lmbda, intercept, random_state=rng)
