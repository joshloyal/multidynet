"""
Generate synthetic networks
"""
import numpy as np

from scipy.special import expit
from sklearn.utils import check_random_state


__all__ = ['simple_dynamic_multilayer_network', 'simple_dynamic_network',
           'dynamic_multilayer_network', 'correlated_dynamic_multilayer_network']


def multilayer_network_from_dynamic_latent_space(X, lmbda, delta,
                                                 random_state=None):
    rng = check_random_state(random_state)

    n_layers, n_time_steps, n_nodes = delta.shape

    if delta is None:
        delta = np.zeros((n_layers, n_time_steps, n_nodes), dtype=np.float64)

    Y = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    probas = np.zeros(
        (n_layers, n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    dists = np.zeros(
        (n_layers, n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    for k in range(n_layers):
        for t in range(n_time_steps):
            # sample the adjacency matrix
            deltak = delta[k, t].reshape(-1, 1)
            eta = np.add(deltak, deltak.T)
            if X is not None:
                dists[k, t] = np.dot(X[t] * lmbda[k], X[t].T)
                eta += dists[k, t]

            probas[k, t] = expit(eta)

            Y[k, t] = rng.binomial(1, probas[k, t]).astype(np.int)

            # make symmetric
            Y[k, t] = np.tril(Y[k, t], k=-1)
            Y[k, t] += Y[k, t].T

    return Y, probas, dists


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
    X -= np.mean(X, axis=(0, 1))

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
    delta = np.zeros((n_layers,  n_time_steps, n_nodes))
    for k in range(n_layers):
        delta[k, 0] = rng.randn(n_nodes)
        for t in range(1, n_time_steps):
            delta[k, t] = delta[k, t-1] + np.sqrt(0.1) * rng.randn(n_nodes)

    # construct the network
    Y, probas, dists = multilayer_network_from_dynamic_latent_space(
        X, lmbda, delta, random_state=rng)

    return Y, X, lmbda, delta, probas, dists


def dynamic_multilayer_network(n_nodes=100, n_layers=4, n_time_steps=10,
                               n_features=2, tau_sq=4.0, sigma_sq=0.05,
                               include_delta=True,
                               sigma_sq_delta=0.1, random_state=42):
    rng = check_random_state(random_state)

    # construct latent features
    n_features = n_features if n_features is not None else 0

    if n_features > 0:
        X = np.zeros((n_time_steps, n_nodes, n_features), dtype=np.float64)
        X[0] = np.sqrt(tau_sq) * rng.randn(n_nodes, n_features)
        X[0] -= np.mean(X[0], axis=0)
        for t in range(1, n_time_steps):
            X[t] = X[t-1] + np.sqrt(sigma_sq) * rng.randn(n_nodes, n_features)
            X[t] -= np.mean(X[t], axis=0)
        #X -= np.mean(X, axis=(0, 1))

        # sample assortativity parameters from a U(-2, 2)
        lmbda = np.zeros((n_layers, n_features))
        lmbda[0] = rng.choice([-1, 1], size=n_features)
        z = rng.choice([-1, 1], size=((n_layers - 1) * n_features)).reshape(n_layers - 1, n_features)
        lmbda[1:] = z + 0.25 * rng.randn(n_layers - 1, n_features)
        #lmbda[1:] = rng.uniform(
        #    #-2, 2, (n_layers - 1) * n_features).reshape(n_layers - 1, n_features)
        #    1, 1.5, (n_layers - 1) * n_features).reshape(n_layers - 1, n_features)
    else:
        X = None
        lmbda = None


    # sample degree effects from a U(-4, 4)
    delta = np.zeros((n_layers, n_time_steps, n_nodes))
    if include_delta:
        for k in range(n_layers):
            delta[k, 0] = rng.uniform(-2, -1, size=n_nodes)
            for t in range(1, n_time_steps):
                delta[k, t] = (
                    delta[k, t-1] + np.sqrt(sigma_sq_delta) * rng.randn(n_nodes))

    # construct the network
    Y, probas, dists = multilayer_network_from_dynamic_latent_space(
        X, lmbda, delta, random_state=rng)

    return Y, X, lmbda, delta, probas, dists


def correlated_dynamic_multilayer_network(n_nodes=100, n_layers=4, n_time_steps=10,
                               center=1, n_features=2, tau=2, rho=0., rho_t=0.5, sigma=0.01,
                               include_delta=True,
                               random_state=42):
    rng = check_random_state(random_state)

    # construct latent features
    n_features = n_features if n_features is not None else 0

    # error correlation structure
    #cov = (sigma ** 2) * ((1 - rho_t) * np.eye(n_time_steps-1) + rho_t * np.ones(
    #    (n_time_steps-1, n_time_steps-1)))
    ts = np.arange(n_time_steps - 1).reshape(-1, 1)
    cov = (sigma ** 2) * (rho_t ** np.abs(ts - ts.T))
    if n_features > 0:
        X = np.zeros((n_time_steps, n_nodes, n_features), dtype=np.float64)

        if isinstance(tau, np.ndarray):
            tau = np.diag(tau)
        else:
            tau = tau * np.eye(n_features)

        if n_features == 2:
            cor = np.array([[1.0, rho],
                            [rho, 1.0]])
            Sigma = tau @ cor @ tau
        else:
            Sigma = tau ** 2

        z = rng.choice([0, 1], size=n_nodes)
        c = np.zeros(n_features)
        c[0] = center
        mu = np.array([c, -c])
        X[0] = rng.multivariate_normal(mean=np.zeros(n_features), cov=Sigma, size=n_nodes)
        X[0] += mu[z]
        X[0] -= np.mean(X[0], axis=0)

        errors = rng.multivariate_normal(mean=np.zeros(n_time_steps-1), cov=cov, size=
                (n_nodes, n_features))
        for t in range(1, n_time_steps):
            X[t] = X[t-1] + errors[..., t-1]
            X[t] -= np.mean(X[t], axis=0)

        # sample assortativity parameters from a U(-1, 2) (average is 0.5)
        lmbda = np.zeros((n_layers, n_features))
        lmbda[0] = rng.choice([-1, 1], size=n_features)
        lmbda[1:] = rng.uniform(
            -1, 2, (n_layers - 1) * n_features).reshape(n_layers - 1, n_features)
    else:
        X = None
        lmbda = None


    # sample degree effects from a U(-2, 0)  # average is -1
    delta = np.zeros((n_layers, n_time_steps, n_nodes))
    if include_delta:
        for k in range(n_layers):
            delta[k, 0] = rng.uniform(-2, 0, size=n_nodes)
            errors = rng.multivariate_normal(mean=np.zeros(n_time_steps-1), cov=cov, size=n_nodes)
            for t in range(1, n_time_steps):
                delta[k, t] = (
                    delta[k, t-1] + errors[..., t-1])
    else:
        delta += -2

    # construct the network
    Y, probas, dists = multilayer_network_from_dynamic_latent_space(
        X, lmbda, delta, random_state=rng)

    return Y, X, lmbda, delta, probas, dists, z


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

    return Y, probas


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
    Y, probas = network_from_dynamic_latent_space(
        X, delta, random_state=rng)

    return Y, X, delta, probas
