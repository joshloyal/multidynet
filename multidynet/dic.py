import numpy as np

from scipy.stats import multivariate_normal
from sklearn.utils import check_random_state


def sample_log_eta(model, random_state=None):
    """Single sample of log(1 + exp(eta)) from the posterior"""
    rng = check_random_state(random_state)

    n_layers, n_time_steps, n_nodes = model.delta_.shape
    n_features = model.n_features_

    res = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))
    for t in range(n_time_steps):
        if n_features > 0:
            X = np.zeros((n_nodes, n_features))
            for i in range(n_nodes):
                X[i] = multivariate_normal.rvs(
                    mean=model.X_[t, i], cov=model.X_sigma_[t, i],
                    random_state=rng)

        for k in range(n_layers):
            delta = multivariate_normal.rvs(
                mean=model.delta_[k, t], cov=model.delta_sigma_[k, t],
                random_state=rng).reshape(-1, 1)

            eta = delta + delta.T
            if n_features > 0:
                if k == 0:
                    lmbda = 2 * rng.binomial(1, p=model.lambda_proba_) - 1
                else:
                    lmbda = multivariate_normal.rvs(
                        mean=model.lambda_[k], cov=model.lambda_sigma_[k],
                        random_state=rng)

                eta += np.dot(X * lmbda, X.T)

            res[k, t] = np.log(1 + np.exp(eta))

    return res


def estimate_log_eta(model, n_samples=100, random_state=None):
    """Estimate log(1 + exp(eta)) using posterior samples"""
    rng = check_random_state(random_state)

    n_layers, n_time_steps, n_nodes = model.delta_.shape
    logeta = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))
    for it in range(n_samples):
        logeta += sample_log_eta(model, random_state=rng)

    return logeta / n_samples


def log_eta_hat(model):
    """log(1 + exp(eta)) with posterior means plugged in."""
    n_layers, n_time_steps, n_nodes = model.delta_.shape

    res = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))
    for k in range(n_layers):
        for t in range(n_time_steps):
            deltakt = model.delta_[k, t].reshape(-1, 1)
            eta = np.add(deltakt, deltakt.T)
            if model.X_ is not None:
                eta += np.dot(model.X_[t] * model.lambda_[k], model.X_[t].T)
            res[k, t] = np.log(1 + np.exp(eta))

    return res


def calculate_dic(model, n_samples=2000, random_state=None):
    logeta = log_eta_hat(model)
    diff = (log_eta_hat(model) -
        estimate_log_eta(model, n_samples=n_samples, random_state=random_state))

    p_dic = 0
    for k in range(diff.shape[0]):
        for t in range(diff.shape[1]):
            indices = np.tril_indices_from(diff[k, t], k=-1)
            p_dic += diff[k, t][indices].sum()
    p_dic *= -2

    return -2 * model.loglik_ + 2 * p_dic, p_dic
