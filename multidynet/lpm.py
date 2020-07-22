import numpy as np
import scipy.sparse as sp

from scipy.special import logit, gammainc
from sklearn.utils import check_array, check_random_state
from tqdm import tqdm

from .omega_lpm import update_omega
from .lds_lpm import update_latent_positions
from .intercept_lpm import update_intercept
from .log_likelihood import log_likelihood


__all__ = ['DynamicNetworkLPM']



class DynamicNetworkLPM(object):
    def __init__(self, n_features=2, intercept_var_prior=4,
                 a=0.1, b=0.1, c=0.1, d=0.1,
                 n_init=1, max_iter=100,
                 warm_start=False, random_state=42):
        self.n_features = n_features
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
        Y : array-like, shape (n_time_steps, n_nodes, n_nodes)
        """
        Y = check_array(Y, order='C', dtype=np.float64,
                        ensure_2d=False, allow_nd=True, copy=False)

        n_time_steps, n_nodes, _ = Y.shape

        random_state = check_random_state(self.random_state)
        #n_init = self.n_init if not self.warm_start else 1
        #for init in range(n_init):
        #    if not self.warm_start:
        self._initialize_parameters(Y, random_state)
        for n_iter in tqdm(range(1, self.max_iter + 1)):
            # coordinate descent
            self._estimate_omegas(Y)
            self._estimate_latent_positions(Y)
            self._estimate_intercepts(Y)
            self._estimate_tau_sq(Y)
            self._estimate_sigma_sq(Y)

            #self.logp_[n_iter] = self.logp(Y)

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
        n_time_steps, n_nodes, _ = Y.shape

        # omega is initialized by drawing from the prior?
        self.omega_ = np.zeros((n_time_steps, n_nodes, n_nodes))

        # initialize using MDS
        self.X_ = np.zeros((n_time_steps, n_nodes, self.n_features))
        #self.X_[0] = rng.randn(n_nodes, self.n_features)
        evals, evecs = np.linalg.eigh(Y[0])
        self.X_[0] = evecs[:, ::-1][:, :2]
        for t in range(1, n_time_steps):
            self.X_[t] = self.X_[t-1] + np.sqrt(0.05) * rng.randn(n_nodes, self.n_features)

        # intialize to prior values
        sigma_init = 0.1 * np.ones((self.n_features, self.n_features))
        self.X_sigma_ = np.tile(
            sigma_init[None, None], reps=(n_time_steps, n_nodes, 1, 1))

        cross_init = 0.1 * np.ones((self.n_features, self.n_features))
        self.X_cross_cov_ = np.tile(
            cross_init[None, None], reps=(n_time_steps - 1, n_nodes, 1, 1))

        # initialize intercept based on edge densities?
        self.intercept_ = logit(Y.mean())
        self.intercept_sigma_ = self.intercept_var_prior

        # initialize based on prior information
        self.a_tau_sq_ = self.a
        self.b_tau_sq_ = self.b
        self.c_sigma_sq_ = self.c
        self.d_sigma_sq_ = self.d

        self.logp_ = np.zeros(self.max_iter + 1)
        #self.logp_[0] = self.logp(Y)

    def _estimate_omegas(self, Y):
        update_omega(self.omega_, self.X_, self.X_sigma_, self.intercept_,
                     self.intercept_sigma_)

    def _estimate_latent_positions(self, Y):
        update_latent_positions(
            Y, self.X_, self.X_sigma_, self.X_cross_cov_,
            self.intercept_, self.omega_,
            1/2., 1/0.5)
            #self.a_tau_sq_/self.b_tau_sq_,
            #self.c_sigma_sq_/self.d_sigma_sq_)

    def _estimate_intercepts(self, Y):
        self.intercept_, self.intercept_sigma_ = update_intercept(
            Y, self.X_, self.omega_, self.intercept_var_prior)

    def _estimate_tau_sq(self, Y):
        n_nodes = Y.shape[2]

        self.a_tau_sq_ = self.a + n_nodes * self.n_features
        self.b_tau_sq_ = (self.b +
            np.trace(self.X_sigma_[0], axis1=1, axis2=2).sum() +
            (self.X_[0] ** 2).sum())

    def _estimate_sigma_sq(self, Y):
        n_time_steps = Y.shape[0]
        n_nodes = Y.shape[1]

        self.c_sigma_sq_ = self.c + n_nodes * (n_time_steps - 1) * self.n_features
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
