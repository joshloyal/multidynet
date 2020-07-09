import numpy as np
import scipy.sparse as sp

from sklearn.utils import check_array, check_random_state

from .omega import update_omega
from .lds import update_latent_positions


__all__ = ['DynamicMultilayerNetworkLSM']


def _intialize_latent_positions(Y):
    pass


class DynamicMultilayerNetworkLSM(object):
    def __init__(self, n_init=1, max_iter=100, warm_start=False):
        pass

    def fit(self, Y):
        """
        Parameters
        ----------
        Y : array-like, shape (n_layers, n_time_steps, n_nodes, n_nodes)
        """
        Y = check_array(Y, order='C', copy=False)

        n_layers, n_time_steps, n_nodes, _ = Y.shape

        random_state = check_random_state(random_state)
        n_init = self.n_init if not self.warm_start else 1
        for init in range(n_init):
            if not self.warm_start:
                self._initialize_parameters(Y, random_state)

            for n_iter in range(1, self.max_iter + 1):
                self._estimate_omegas(Y)
                self._estimate_latent_positions(Y)

    def _initialize_parameters(self, Y, random_state):
        n_layers, n_time_steps, n_nodes, _ = Y.shape

        self.omega_ = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))
        self.X_ = np.zeros((n_time_steps, n_nodes, self.n_features))
        self.X_sigma_ = np.zeros((n_time_steps, n_nodes, self.n_features, self.n_features))
        self.X_cross_cov_ = np.zeros(
            (n_time_steps - 1, n_nodes, self.n_features, self.n_features))
        self.beta_ = np.zeros(n_layers)
        self.beta_sigma_ = np.zeros(n_layers)
        self.lambda_ = np.zeros((n_layers, self.n_features))
        self.lambda_sigma_ = np.zeros((n_layers, self.n_features, n_features))

    def _estimate_omegas(self, Y):
        update_omega(self.omegas_, self.X_, self.X_sigma_, self.beta_,
                     self.beta_sigma_, self.lambda_, self.lambda_sigma_)

    def _estimate_latent_positions(self, Y):
        update_latent_positions(
            Y, self.X_, self.X_sigma_, self.X_cross_cov_,
            self.lambda_, self.lambda_sigma_, self.beta_, self.omegas_,
            self.a / self.b, self.c / self.d)

    def _estimate_lambdas(self):
        pass

    def _estimate_intercepts(self):
        pass

    def _estimate_tau_sq(self):
        pass

    def _estimate_sigma_sq(self):
        pass
