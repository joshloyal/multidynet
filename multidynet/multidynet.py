import warnings

import numpy as np
import scipy.sparse as sp

from joblib import Parallel, delayed
from scipy.special import logit, gammainc
from sklearn.utils import check_array, check_random_state
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from .latent_space import generalized_mds
from .omega import update_omega
from .lds import update_latent_positions
from .lmbdas import update_lambdas
from .intercepts import update_intercepts
from .variances import update_tau_sq, update_sigma_sq
from .log_likelihood import log_likelihood


__all__ = ['DynamicMultilayerNetworkLSM']



class ModelParameters(object):
    def __init__(self, omega, X, X_sigma, X_cross_cov,
                 intercept, intercept_sigma, lmbda, lmbda_sigma,
                 lmbda_logit_prior, a_tau_sq, b_tau_sq, c_sigma_sq,
                 d_sigma_sq):
        self.omega_ = omega
        self.X_ = X
        self.X_sigma_ = X_sigma
        self.X_cross_cov_ = X_cross_cov
        self.intercept_ = intercept
        self.intercept_sigma_ = intercept_sigma
        self.lambda_ = lmbda
        self.lambda_sigma_ = lmbda_sigma
        self.lambda_logit_prior_ = lmbda_logit_prior
        self.a_tau_sq_ = a_tau_sq
        self.b_tau_sq_ = b_tau_sq
        self.c_sigma_sq_ = c_sigma_sq
        self.d_sigma_sq_ = d_sigma_sq
        self.converged_ = False
        self.logp_ = []


def initialize_parameters(Y, n_features, lambda_odds_prior, lambda_var_prior,
                          intercept_var_prior, a, b, c, d, random_state):
    rng = check_random_state(random_state)

    n_layers, n_time_steps, n_nodes, _ = Y.shape

    # omega is initialized by drawing from the prior?
    omega = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))

    # intialize latent space randomly
    X = rng.randn(n_time_steps, n_nodes, n_features)

    # intialize to marginal covariances
    sigma_init = np.eye(n_features)
    X_sigma = np.tile(
        sigma_init[None, None], reps=(n_time_steps, n_nodes, 1, 1))

    # initialize cross-covariances
    cross_init = np.eye(n_features)
    X_cross_cov = np.tile(
        cross_init[None, None], reps=(n_time_steps - 1, n_nodes, 1, 1))

    # initialize intercept based on edge density
    intercept = logit(Y.mean(axis=(1, 2, 3)))
    intercept_sigma = intercept_var_prior * np.ones(n_layers)

    # intialize to prior means
    lmbda = np.sqrt(2) * rng.randn(n_layers, n_features)
    lmbda[0] = (
        2 * (lambda_odds_prior / (1. + lambda_odds_prior)) - 1)
    lmbda_sigma = lambda_var_prior * np.ones(
        (n_layers, n_features, n_features))
    lmbda_sigma[0] = (
        (1 - lmbda[0, 0] ** 2) * np.eye(n_features))
    lmbda_logit_prior = np.log(lambda_odds_prior)

    # initialize based on prior information
    a_tau_sq = a
    b_tau_sq = b
    c_sigma_sq = c
    d_sigma_sq = d

    return ModelParameters(
        omega=omega, X=X, X_sigma=X_sigma, X_cross_cov=X_cross_cov,
        intercept=intercept, intercept_sigma=intercept_sigma, lmbda=lmbda,
        lmbda_sigma=lmbda_sigma, lmbda_logit_prior=lmbda_logit_prior,
        a_tau_sq=a_tau_sq, b_tau_sq=b_tau_sq, c_sigma_sq=c_sigma_sq,
        d_sigma_sq=d_sigma_sq)



def optimize_elbo(Y, n_features, lambda_odds_prior, lambda_var_prior,
                  intercept_var_prior, a, b, c, d,
                  max_iter, tol, random_state, verbose=True):

    # convergence criteria (Eq{L(Y | theta)})
    loglik = -np.infty

    # initialize parameters of the model
    model = initialize_parameters(
        Y, n_features, lambda_odds_prior, lambda_var_prior, intercept_var_prior,
        a, b, c, d, random_state)

    for n_iter in tqdm(range(max_iter), disable=not verbose):
        prev_loglik = loglik

        # coordinate descent

        # omega updates
        loglik = update_omega(
            Y, model.omega_, model.X_, model.X_sigma_, model.intercept_,
            model.intercept_sigma_, model.lambda_, model.lambda_sigma_)

        # latent trajectory updates
        update_latent_positions(
            Y, model.X_, model.X_sigma_, model.X_cross_cov_,
            model.lambda_, model.lambda_sigma_, model.intercept_, model.omega_,
            model.a_tau_sq_ / model.b_tau_sq_,
            model.c_sigma_sq_ / model.d_sigma_sq_)

        # update lambda values
        update_lambdas(
            Y, model.X_, model.X_sigma_, model.intercept_, model.lambda_,
            model.lambda_sigma_, model.omega_, lambda_var_prior,
            model.lambda_logit_prior_)

        # update intercept
        update_intercepts(
            Y, model.X_, model.intercept_, model.intercept_sigma_,
            model.lambda_, model.omega_, intercept_var_prior)

        # update intial variance of the latent space
        model.a_tau_sq_, model.b_tau_sq_ = update_tau_sq(
            Y, model.X_, model.X_sigma_, a, b)

        # update step sizes
        model.c_sigma_sq_, model.d_sigma_sq_ = update_sigma_sq(
            Y, model.X_, model.X_sigma_, model.X_cross_cov_, c, d)

        model.logp_.append(loglik)

        # check convergence
        change = loglik - prev_loglik
        if abs(change) < tol:
            model.converged_ = True
            model.logp_ = np.asarray(model.logp_)
            break

    return model


class DynamicMultilayerNetworkLSM(object):
    def __init__(self, n_features=2,
                 lambda_odds_prior=2,
                 lambda_var_prior=4, intercept_var_prior=4,
                 a=4.0, b=8.0, c=10, d=0.1,
                 n_init=1, max_iter=500, tol=1e-2,
                 n_jobs=-1, random_state=42):
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
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, Y):
        """
        Parameters
        ----------
        Y : array-like, shape (n_layers, n_time_steps, n_nodes, n_nodes)
        """
        Y = check_array(Y, order='C', dtype=np.float64,
                        ensure_2d=False, allow_nd=True, copy=False)

        random_state = check_random_state(self.random_state)
        #n_init = self.n_init if not self.warm_start else 1
        #for init in range(n_init):
        #    if not self.warm_start:

        #self.converged_ = False
        #self.logp_ = []
        #loglik = -np.infty

        #self._initialize_parameters(Y, random_state)
        #for n_iter in tqdm(range(self.max_iter)):
        #    prev_loglik = loglik

        #    # coordinate descent
        #    loglik = self._estimate_omegas(Y)
        #    self._estimate_latent_positions(Y)
        #    self._estimate_lambdas(Y)
        #    self._estimate_intercepts(Y)
        #    self._estimate_tau_sq(Y)
        #    self._estimate_sigma_sq(Y)

        #    #self.logp_[n_iter] = self.logp(Y)
        #    self.logp_.append(loglik)

        #    # check convergence
        #    change = loglik - prev_loglik
        #    if abs(change) < self.tol:
        #        self.converged_ = True
        #        self.logp_ = np.asarray(self.logp_)
        #        break

        # run the elbo optimization over different initializations
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        verbose = True if self.n_init == 1 else False
        models = Parallel(n_jobs=self.n_jobs)(delayed(optimize_elbo)(
                Y, self.n_features, self.lambda_odds_prior,
                self.lambda_var_prior, self.intercept_var_prior,
                self.a, self.b, self.c, self.d,
                self.max_iter, self.tol, seed, verbose=verbose)
            for seed in seeds)

        #model = optimize_elbo(
        #    Y, self.n_features, self.lambda_odds_prior, self.lambda_var_prior,
        #    self.intercept_var_prior, self.a, self.b, self.c, self.d,
        #    self.max_iter, self.tol, self.random_state, verbose=True)

        # choose model with the largest convergence criteria
        best_model = models[0]
        best_criteria = models[0].logp_[-1]
        for i in range(1, len(models)):
            if models[i].logp_[-1] > best_criteria:
                best_model = models[i]

        if not best_model.converged_:
            warnings.warn('Best model did not converge. '
                          'Try a different random initialization, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.', ConvergenceWarning)

        self._set_parameters(best_model)

        return self

    def _set_parameters(self, model):
        self.omega_ = model.omega_
        self.X_ = model.X_
        self.X_sigma_ = model.X_sigma_
        self.X_cross_cov_ = model.X_cross_cov_
        self.intercept_ = model.intercept_
        self.intercept_sigma_ = model.intercept_sigma_
        self.lambda_ = model.lambda_
        self.lambda_sigma_ = model.lambda_sigma_
        self.a_tau_sq_ = model.a_tau_sq_
        self.b_tau_sq_ = model.b_tau_sq_
        self.c_sigma_sq_ = model.c_sigma_sq_
        self.d_sigma_sq_ = model.d_sigma_sq_
        self.logp_ = model.logp_

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
        #sigma_init = 5 * np.ones((self.n_features, self.n_features))
        sigma_init = np.eye(self.n_features)
        self.X_sigma_ = np.tile(
            sigma_init[None, None], reps=(n_time_steps, n_nodes, 1, 1))

        #self.X_cross_cov_ = np.zeros(
        #    (n_time_steps - 1, n_nodes, self.n_features, self.n_features))
        #cross_init = 5 * np.ones((self.n_features, self.n_features))
        cross_init = np.eye(self.n_features)
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

        #self.logp_ = np.zeros(self.max_iter)
        #self.logp_[0] = self.logp(Y)

    def _estimate_omegas(self, Y):
        loglik = update_omega(
            Y, self.omega_, self.X_, self.X_sigma_, self.intercept_,
            self.intercept_sigma_, self.lambda_, self.lambda_sigma_)
        return loglik

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
