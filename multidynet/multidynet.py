import warnings

import numpy as np
import scipy.sparse as sp

from joblib import Parallel, delayed
from scipy.special import logit, gammainc, expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_array, check_random_state
from tqdm import tqdm

from .omega import update_omega
from .lds import update_latent_positions
from .lmbdas import update_lambdas
from .intercepts import update_intercepts
from .variances import update_tau_sq, update_sigma_sq
from .log_likelihood import log_likelihood
from .metrics import calculate_auc


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
    intercept = np.zeros(n_layers)
    for k in range(n_layers):
        for t in range(n_time_steps):
            Y_vec = Y[k, t][np.tril_indices_from(Y[k, t], k=-1)]
            intercept[k] += Y_vec[Y_vec != -1.0].mean()
        intercept[k] /= n_time_steps

    intercept = logit(intercept)
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
                  intercept_var_prior, tau_sq, sigma_sq, a, b, c, d,
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
        tau_sq_prec = (
            model.a_tau_sq_ / model.b_tau_sq_ if tau_sq == 'auto' else 1. / tau_sq)
        sigma_sq_prec = (
            model.c_sigma_sq_ / model.d_sigma_sq_ if sigma_sq == 'auto' else 1. / sigma_sq)


        update_latent_positions(
            Y, model.X_, model.X_sigma_, model.X_cross_cov_,
            model.lambda_, model.lambda_sigma_, model.intercept_, model.omega_,
            tau_sq_prec, sigma_sq_prec)

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


def calculate_probabilities(Y, X, lmbda, intercept):
    n_layers, n_time_steps, n_nodes, _ = Y.shape

    probas = np.zeros_like(Y)
    for k in range(n_layers):
        for t in range(n_time_steps):
            eta = intercept[k] + np.dot(X[t] * lmbda[k], X[t].T)
            probas[k, t] = expit(eta)

    return probas


class DynamicMultilayerNetworkLSM(object):
    def __init__(self, n_features=2,
                 lambda_odds_prior=2,
                 lambda_var_prior=4, intercept_var_prior=4,
                 tau_sq='auto', sigma_sq='auto',
                 a=4.0, b=8.0, c=10, d=0.1,
                 n_init=1, max_iter=500, tol=1e-2,
                 n_jobs=-1, random_state=42):
        self.n_features = n_features
        self.lambda_odds_prior = lambda_odds_prior
        self.lambda_var_prior = lambda_var_prior
        self.intercept_var_prior = intercept_var_prior
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
        Y : array-like, shape (n_layers, n_time_steps, n_nodes, n_nodes)
        """
        Y = check_array(Y, order='C', dtype=np.float64,
                        ensure_2d=False, allow_nd=True, copy=False)

        random_state = check_random_state(self.random_state)

        # run the elbo optimization over different initializations
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        verbose = True if self.n_init == 1 else False
        models = Parallel(n_jobs=self.n_jobs)(delayed(optimize_elbo)(
                Y, self.n_features, self.lambda_odds_prior,
                self.lambda_var_prior, self.intercept_var_prior,
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
            Y, self.X_, self.lambda_, self.intercept_)

        # calculate in-sample AUC
        self.auc_ = calculate_auc(Y, self.probas_)

        return self

    def _set_parameters(self, model):
        self.omega_ = model.omega_
        self.X_ = model.X_
        self.X_sigma_ = model.X_sigma_
        self.X_cross_cov_ = model.X_cross_cov_
        self.intercept_ = model.intercept_
        self.intercept_sigma_ = model.intercept_sigma_
        self.lambda_ = model.lambda_
        self.lambda_[0] = np.sign(model.lambda_[0])
        self.lambda_proba_ = (model.lambda_[0] + 1) / 2.
        self.lambda_sigma_ = model.lambda_sigma_
        self.a_tau_sq_ = model.a_tau_sq_
        self.b_tau_sq_ = model.b_tau_sq_
        self.tau_sq_ = self.b_tau_sq_ / (self.a_tau_sq_ - 1)
        self.c_sigma_sq_ = model.c_sigma_sq_
        self.d_sigma_sq_ = model.d_sigma_sq_
        self.sigma_sq_ = self.d_sigma_sq_ / (self.c_sigma_sq_ - 1)
        self.logp_ = model.logp_

    def logp(self, Y):
        return log_likelihood(Y, self.X_, self.lambda_, self.intercept_)
