import numpy as np
import scipy.sparse as sp

from sklearn.utils import check_array, check_random_state
from tqdm import tqdm

from .latent_space import generalized_mds
from .omega import update_omega
from .lds import update_latent_positions
from .lmbdas import update_lambdas
from .intercepts import update_intercepts


__all__ = ['DynamicMultilayerNetworkLSM']


def _intialize_latent_positions(Y):
    pass


class DynamicMultilayerNetworkLSM(object):
    def __init__(self, n_features=2, lambda_var_prior=4, intercept_var_prior=4,
                 a=4, b=4, c=4, d=4, n_init=1, max_iter=100, warm_start=False,
                 random_state=42):
        self.n_features = n_features
        self.lambda_var_prior = lambda_var_prior
        self.intercept_var_prior = intercept_var_prior
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.n_init = n_init
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.random_state = random_state

    def fit(self, Y):
        """
        Parameters
        ----------
        Y : array-like, shape (n_layers, n_time_steps, n_nodes, n_nodes)
        """
        Y = check_array(Y, order='C', dtype=np.float64,
                        ensure_2d=False, allow_nd=True, copy=False)

        n_layers, n_time_steps, n_nodes, _ = Y.shape

        random_state = check_random_state(self.random_state)
        #n_init = self.n_init if not self.warm_start else 1
        #for init in range(n_init):
        #    if not self.warm_start:
        self._initialize_parameters(Y, random_state)
        for n_iter in tqdm(range(1, self.max_iter + 1)):
            # coordinate descent
            self._estimate_omegas(Y)
            self._estimate_latent_positions(Y)
            self._estimate_lambdas(Y)
            self._estimate_intercepts(Y)
            self._estimate_tau_sq(Y)
            self._estimate_sigma_sq(Y)

            # check convergence
            #if abs(change) < self.tol:
            #    self.converged_ = True
            #    break

        #if not self.converged_:
        #    pass

        return self

    def _initialize_parameters(self, Y, rng):
        n_layers, n_time_steps, n_nodes, _ = Y.shape

        # omega is initialized by drawing from the prior?
        self.omega_ = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))
        #self.X_ = np.zeros((n_time_steps, n_nodes, self.n_features))

        # initialize using MDS
        #self.X_ = generalized_mds(
        #    Y, n_features=self.n_features, random_state=rng)
        self.X_ = 10 * rng.randn(n_time_steps, n_nodes, self.n_features)

        # intialize to prior values
        #self.X_sigma_ = np.zeros(
        #    (n_time_steps, n_nodes, self.n_features, self.n_features))
        sigma_init = 10 * np.eye(self.n_features)

        self.X_sigma_ = np.tile(sigma_init[None, None], reps=(n_time_steps, n_nodes, 1, 1))

        #self.X_cross_cov_ = np.zeros(
        #    (n_time_steps - 1, n_nodes, self.n_features, self.n_features))
        cross_init = 10 * np.eye(self.n_features)
        self.X_cross_cov_ = np.tile(cross_init[None, None], reps=(n_time_steps - 1, n_nodes, 1, 1))

        # initialize intercept based on edge densities?
        self.intercept_ = (
            Y.sum(axis=(1, 2, 3)) / (0.5 * n_nodes * (n_nodes - 1)))
        self.intercept_sigma_ = self.intercept_var_prior * np.ones(n_layers)

        # intialize to prior means
        #self.lambda_ = np.zeros((n_layers, self.n_features))
        self.lambda_ = np.sqrt(self.lambda_var_prior) * rng.randn(n_layers, self.n_features)
        self.lambda_sigma_ = self.lambda_var_prior * np.ones(
            (n_layers, self.n_features, self.n_features))

        # initialize based on prior information
        self.a_tau_sq_ = self.a + n_nodes
        self.b_tau_sq_ = self.b
        self.c_sigma_sq_ = self.c + n_nodes * (n_time_steps - 1)
        self.d_sigma_sq_ = self.d

    def _estimate_omegas(self, Y):
        update_omega(self.omega_, self.X_, self.X_sigma_, self.intercept_,
                     self.intercept_sigma_, self.lambda_, self.lambda_sigma_)

    def _estimate_latent_positions(self, Y):
        update_latent_positions(
            Y, self.X_, self.X_sigma_, self.X_cross_cov_,
            self.lambda_, self.lambda_sigma_, self.intercept_, self.omega_,
            self.a_tau_sq_ / self.b_tau_sq_,
            self.c_sigma_sq_ / self.d_sigma_sq_)

    def _estimate_lambdas(self, Y):
        update_lambdas(
            Y, self.X_, self.X_sigma_, self.intercept_, self.lambda_,
            self.lambda_sigma_, self.omega_, self.lambda_var_prior)

    def _estimate_intercepts(self, Y):
        update_intercepts(
            Y, self.X_, self.intercept_, self.intercept_sigma_,
            self.lambda_, self.omega_, self.intercept_var_prior)

    def _estimate_tau_sq(self, Y):
        self.b_tau_sq_ = (self.b +
            np.trace(self.X_sigma_[0], axis1=1, axis2=2).sum() +
            (self.X_[0] ** 2).sum())

    def _estimate_sigma_sq(self, Y):
        n_time_steps = Y.shape[1]

        self.d_sigma_sq_ = self.d
        for t in range(1, n_time_steps):
            self.d_sigma_sq_ += np.trace(
                self.X_sigma_[t], axis1=1, axis2=2).sum()
            self.d_sigma_sq_ += (self.X_[t] ** 2).sum()

            self.d_sigma_sq_ += np.trace(
                self.X_sigma_[t-1], axis1=1, axis2=2).sum()
            self.d_sigma_sq_ += (self.X_[t-1] ** 2).sum()

            self.d_sigma_sq_ -= 2 * np.trace(
                self.X_cross_cov_[t-1], axis1=1, axis2=2).sum()
            self.d_sigma_sq_ -= 2 * (self.X_[t-1] * self.X_[t]).sum()
