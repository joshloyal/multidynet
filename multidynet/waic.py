import numpy as np

from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.utils import check_random_state


def calculate_waic(model, Y, n_samples=100, random_state=None):
    if np.any(Y == -1):
        raise ValueError('Missing values unsupported')

    n_layers, n_time_steps, n_nodes, _ = Y.shape
    n_features = model.n_features_

    rng = check_random_state(random_state)

    n_elems = int(0.5 * n_nodes * (n_nodes - 1))
    lppd = np.zeros((n_samples, n_layers, n_time_steps, n_elems))
    tril = np.tril_indices_from(Y[0, 0], k=-1)
    for s in range(n_samples):
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

                y_vec = Y[k, t][tril]
                eta_vec = eta[tril]
                lppd[s, k, t] = (y_vec * eta_vec - np.log(1 + np.exp(eta_vec)))

    lppd = lppd.reshape(n_samples, np.prod(lppd.shape[1:]))
    vars_lpd = np.var(lppd, ddof=1, axis=0)
    lpd_i = logsumexp(lppd, axis=0) - np.log(lppd.shape[0])
    waic_i = -2 * lpd_i + 2 * vars_lpd

    return np.sum(waic_i), np.sum(vars_lpd)
