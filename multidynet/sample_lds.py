import numpy as np
import scipy

from sklearn.utils import check_random_state


def sample_gssm(mu, sigma, cross_cov, size=1, random_state=None):
    """Sample from the GSSM approximate posteriors"""
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
        sigma = np.expand_dims(sigma.reshape(-1, 1), axis=-1)
        cross_cov = np.expand_dims(cross_cov.reshape(-1, 1), axis=-1)

    n_time_steps, n_features = mu.shape
    rng = check_random_state(random_state)

    x = np.zeros((size, n_time_steps, n_features))

    x[:, 0] = rng.multivariate_normal(mu[0], sigma[0], size=size)
    for t in range(1, n_time_steps):
        # calculate conditional means and covariances
        sigma_inv = np.linalg.pinv(sigma[t-1])
        mu_cond = (mu[t].reshape(-1, 1) +
            cross_cov[t-1].T @ sigma_inv @ (x[:, t-1] - mu[t-1]).T).T
        cov_cond = sigma[t] - cross_cov[t-1].T @ sigma_inv @ cross_cov[t-1]

        # sample from conditional
        cov_sqrt = scipy.linalg.sqrtm(cov_cond)
        x[:, t] = (mu_cond + (
            cov_sqrt @ rng.randn(n_features * size).reshape(n_features, size)).T)

    return np.squeeze(x)
