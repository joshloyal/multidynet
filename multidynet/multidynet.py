import numpy as np
import scipy.sparse as sp

from scipy.special import logit, gammainc
from sklearn.utils import check_array, check_random_state
from tqdm import tqdm

from .latent_space import generalized_mds
from .omega import update_omega
from .lds import update_latent_positions
from .lmbdas import update_lambdas
from .intercepts import update_intercepts
from .log_likelihood import log_likelihood


__all__ = ['DynamicMultilayerNetworkLSM']



class DynamicMultilayerNetworkLSM(object):
    def __init__(self, n_features=2,
                 lambda_odds_prior=2,
                 lambda_var_prior=4, intercept_var_prior=4,
                 a=0.1, b=0.1, c=0.1, d=0.1,
                 n_init=1, max_iter=100,
                 warm_start=False, random_state=42):
        self.n_features = n_features
        self.lambda_odds_prior = lambda_odds_prior
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
        self.X_ref_ = None

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

            self.logp_[n_iter] = self.logp(Y)

            #if self.X_ref_ is not None:
            #    self._scale_space()

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

        # initialize using MDS
        #self.X_ = generalized_mds(
        #    Y, n_features=self.n_features, random_state=rng)
        #self.X_ = np.zeros((n_time_steps, n_nodes, self.n_features))
        #evals, evecs = np.linalg.eigh(Y[0, 0])
        #self.X_[0] = evecs[:, ::-1][:, :self.n_features] * np.sqrt(evals[::-1][:self.n_features])
        ###self.X_[0] = rng.randn(n_nodes, self.n_features)
        #for t in range(1, n_time_steps):
        #    #self.X_[t] = self.X_[t-1] + np.sqrt(self.k) * rng.randn(n_nodes, self.n_features)
        #    self.X_[t] = self.X_[0].copy()
        self.X_ = rng.randn(n_time_steps, n_nodes, self.n_features)

        # intialize to prior values
        #self.X_sigma_ = np.zeros(
        #    (n_time_steps, n_nodes, self.n_features, self.n_features))
        sigma_init = 5 * np.ones((self.n_features, self.n_features))
        #sigma_init = np.eye(self.n_features)
        self.X_sigma_ = np.tile(
            sigma_init[None, None], reps=(n_time_steps, n_nodes, 1, 1))

        #self.X_cross_cov_ = np.zeros(
        #    (n_time_steps - 1, n_nodes, self.n_features, self.n_features))
        cross_init = 5 * np.ones((self.n_features, self.n_features))
        #cross_init = np.eye(self.n_features)
        self.X_cross_cov_ = np.tile(
            cross_init[None, None], reps=(n_time_steps - 1, n_nodes, 1, 1))

        # initialize intercept based on edge densities?
        self.intercept_ = logit(Y.mean(axis=(1, 2, 3)))
        self.intercept_sigma_ = self.intercept_var_prior * np.ones(n_layers)

        # intialize to prior means
        self.lambda_ = np.sqrt(2) * rng.randn(n_layers, self.n_features)
        self.lambda_[0] = (
            2 * (self.lambda_odds_prior / (1. + self.lambda_odds_prior)) - 1)
        self.lambda_sigma_ = self.lambda_var_prior * np.ones(
            (n_layers, self.n_features, self.n_features))
        self.lambda_sigma_[0] = (
            (1 - self.lambda_[0, 0] ** 2) * np.eye(self.n_features))
        self.lambda_logit_prior_ = np.log(self.lambda_odds_prior)

        # initialize based on prior information
        self.a_tau_sq_ = self.a
        self.b_tau_sq_ = self.b
        self.c_sigma_sq_ = self.c
        self.d_sigma_sq_ = self.d

        self.logp_ = np.zeros(self.max_iter + 1)
        self.logp_[0] = self.logp(Y)

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
            self.lambda_sigma_, self.omega_, self.lambda_var_prior,
            self.lambda_logit_prior_)

        # set first component of lambda to all ones
        #self.lambda_ /= self.lambda_[0]
        #D = np.diag(1./self.lambda_[0])
        #for k in range(Y.shape[0]):
        #    self.lambda_sigma_[k] = D @ self.lambda_sigma_[k] @ D

    def _estimate_intercepts(self, Y):
        update_intercepts(
            Y, self.X_, self.intercept_, self.intercept_sigma_,
            self.lambda_, self.omega_, self.intercept_var_prior)

    def _estimate_tau_sq(self, Y):
        n_nodes = Y.shape[2]

        self.a_tau_sq_ = self.a + n_nodes * self.n_features
        self.b_tau_sq_ = (self.b +
            np.trace(self.X_sigma_[0], axis1=1, axis2=2).sum() +
            (self.X_[0] ** 2).sum())

    def _estimate_sigma_sq(self, Y):
        n_time_steps = Y.shape[1]
        n_nodes = Y.shape[2]

        self.c_sigma_sq_ = (self.c +
            n_nodes * (n_time_steps - 1) * self.n_features)
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

        #self.k_mean_ = self.c_sigma_sq_ / self.d_sigma_sq_
        #self.k_mean_ *= ((1 - gammainc(self.c_sigma_sq_ + 1, self.d_sigma_sq_)) /
        #                    (1 - gammainc(self.c_sigma_sq_, self.d_sigma_sq_)))

    def logp(self, Y):
        return log_likelihood(Y, self.X_, self.lambda_, self.intercept_)
