import warnings

import numpy as np
import scipy.sparse as sp

from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

#from .metrics import calculate_auc_layer
from .multidynet import initialize_node_effects_single
from .omega_lsm import update_omega
from .deltas_lsm import update_deltas
from .lds_lsm import update_latent_positions
from .variances import update_tau_sq, update_sigma_sq


__all__ = ['DynamicNetworkLSM']


class ModelParameters(object):
    def __init__(self, omega, X, X_sigma, X_cross_cov,
                 delta, delta_sigma,
                 a_tau_sq, b_tau_sq, c_sigma_sq, d_sigma_sq):
        self.omega_ = omega
        self.X_ = X
        self.X_sigma_ = X_sigma
        self.X_cross_cov_ = X_cross_cov
        self.delta_ = delta
        self.delta_sigma_ = delta_sigma
        self.a_tau_sq_ = a_tau_sq
        self.b_tau_sq_ = b_tau_sq
        self.c_sigma_sq_ = c_sigma_sq
        self.d_sigma_sq_ = d_sigma_sq
        self.converged_ = False
        self.logp_ = []


def initialize_parameters(Y, n_features, delta_var_prior,
                          a, b, c, d, random_state):
    rng = check_random_state(random_state)

    n_time_steps, n_nodes, _ = Y.shape

    # omega is initialized by drawing from the prior?
    omega = np.zeros((n_time_steps, n_nodes, n_nodes))

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

    # initialize node-effects based on a logistic regression with
    # no higher order structure
    delta = initialize_node_effects_single(Y)
    delta_sigma = delta_var_prior * np.ones(n_nodes)

    # initialize based on prior information
    a_tau_sq = a
    b_tau_sq = b
    c_sigma_sq = c
    d_sigma_sq = d

    return ModelParameters(
        omega=omega, X=X, X_sigma=X_sigma, X_cross_cov=X_cross_cov,
        delta=delta, delta_sigma=delta_sigma,
        a_tau_sq=a_tau_sq, b_tau_sq=b_tau_sq, c_sigma_sq=c_sigma_sq,
        d_sigma_sq=d_sigma_sq)


def optimize_elbo(Y, n_features, delta_var_prior, tau_sq, sigma_sq, a, b, c, d,
                  max_iter, tol, random_state, verbose=True):

    # convergence criteria (Eq{L(Y | theta)})
    loglik = -np.infty

    # initialize parameters of the model
    model = initialize_parameters(
        Y, n_features, delta_var_prior, a, b, c, d, random_state)

    for n_iter in tqdm(range(max_iter), disable=not verbose):
        prev_loglik = loglik

        # coordinate ascent

        # omega updates
        loglik = update_omega(
            Y, model.omega_, model.X_, model.X_sigma_,
            model.delta_, model.delta_sigma_)

        # latent trajectory updates
        tau_sq_prec = (
            model.a_tau_sq_ / model.b_tau_sq_ if tau_sq == 'auto' else
                1. / tau_sq)
        sigma_sq_prec = (
            model.c_sigma_sq_ / model.d_sigma_sq_ if sigma_sq == 'auto' else
                1. / sigma_sq)

        update_latent_positions(
            Y, model.X_, model.X_sigma_, model.X_cross_cov_,
            model.delta_, model.omega_, tau_sq_prec, sigma_sq_prec)

        # update node random effects
        update_deltas(
            Y, model.X_, model.delta_, model.delta_sigma_,
            model.omega_, delta_var_prior)

        # update intial variance of the latent space
        if tau_sq == 'auto':
            model.a_tau_sq_, model.b_tau_sq_ = update_tau_sq(
                Y, model.X_, model.X_sigma_, a, b)

        # update step sizes
        if sigma_sq == 'auto':
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


def calculate_probabilities(X, delta):
    n_time_steps = X.shape[0]
    n_nodes = X.shape[1]

    probas = np.zeros(
        (n_time_steps, n_nodes, n_nodes), dtype=np.float64)

    deltas = delta.reshape(-1, 1)
    for t in range(n_time_steps):
        probas[t] = expit(np.add(deltas, deltas.T) + np.dot(X[t], X[t].T))

    return probas


class DynamicNetworkLSM(object):
    def __init__(self, n_features=2, delta_var_prior=4,
                 tau_sq='auto', sigma_sq='auto',
                 a=4.0, b=20.0, c=10, d=0.1,
                 n_init=1, max_iter=500, tol=1e-2,
                 n_jobs=-1, random_state=42):
        self.n_features = n_features
        self.delta_var_prior = delta_var_prior
        self.tau_sq = tau_sq
        self.sigma_sq = sigma_sq
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
        Y : array-like, shape (n_time_steps, n_nodes, n_nodes)
        """
        Y = check_array(Y, order='C', dtype=np.float64,
                        ensure_2d=False, allow_nd=True, copy=False)

        random_state = check_random_state(self.random_state)

        # run the elbo optimization over different initializations
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        verbose = True if self.n_init == 1 else False
        models = Parallel(n_jobs=self.n_jobs)(delayed(optimize_elbo)(
                Y, self.n_features, self.delta_var_prior,
                self.tau_sq, self.sigma_sq, self.a, self.b, self.c, self.d,
                self.max_iter, self.tol, seed, verbose=verbose)
            for seed in seeds)

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

        # calculate dyad-probabilities
        self.probas_ = calculate_probabilities(
            self.X_, self.delta_)

        # calculate in-sample AUC
        #self.auc_ = calculate_auc_layer(Y, self.probas_)

        return self

    def _set_parameters(self, model):
        self.omega_ = model.omega_
        self.X_ = model.X_
        self.X_sigma_ = model.X_sigma_
        self.X_cross_cov_ = model.X_cross_cov_
        self.delta_ = model.delta_
        self.delta_sigma_ = model.delta_sigma_
        self.a_tau_sq_ = model.a_tau_sq_
        self.b_tau_sq_ = model.b_tau_sq_
        self.tau_sq_ = self.b_tau_sq_ / (self.a_tau_sq_ - 1)
        self.c_sigma_sq_ = model.c_sigma_sq_
        self.d_sigma_sq_ = model.d_sigma_sq_
        self.sigma_sq_ = self.d_sigma_sq_ / (self.c_sigma_sq_ - 1)
        self.logp_ = model.logp_

        return self
