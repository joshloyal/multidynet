import warnings

import numpy as np
import scipy.sparse as sp

from joblib import Parallel, delayed
from scipy.special import logit, gammainc, expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss
from tqdm import tqdm

from .omega import update_omega
from .lds import update_latent_positions
from .deltas_lds import update_deltas
from .lmbdas import update_lambdas
from .variances import update_tau_sq, update_X0_precision, update_diag_tau_sq
from .variances import update_tau_sq_delta
from .log_likelihood import log_likelihood
from ..metrics import calculate_auc


__all__ = ['MultilayerNetworkLSM']


class ModelParameters(object):
    def __init__(self, omega, X, X_sigma,
                 lmbda, lmbda_sigma, lmbda_logit_prior,
                 delta, delta_sigma,
                 X0_cov_df, X0_cov_scale,
                 a_tau_sq, b_tau_sq,
                 a_tau_sq_delta, b_tau_sq_delta,
                 callback):
        self.omega_ = omega
        self.X_ = X
        self.X_sigma_ = X_sigma
        self.lambda_ = lmbda
        self.lambda_sigma_ = lmbda_sigma
        self.lambda_logit_prior_ = lmbda_logit_prior
        self.delta_ = delta
        self.delta_sigma_ = delta_sigma
        self.X0_cov_df_ = X0_cov_df
        self.X0_cov_scale_ = X0_cov_scale
        self.a_tau_sq_ = a_tau_sq
        self.b_tau_sq_ = b_tau_sq
        self.a_tau_sq_delta_ = a_tau_sq_delta
        self.b_tau_sq_delta_ = b_tau_sq_delta
        self.converged_ = False
        self.logp_ = []
        self.criteria_ = []
        self.callback_ = callback


def initialize_node_effects_single(Y):
    n_nodes = Y.shape[0]

    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    dyads = np.tril_indices_from(Y, k=-1)
    y_vec = Y[dyads]

    # construct dummy node indicators
    cols = np.r_[dyads[0], dyads[1]]
    rows = np.r_[np.arange(n_dyads), np.arange(n_dyads)]
    X = sp.coo_matrix((np.ones(2 * n_dyads), (rows, cols)),
                       shape=(n_dyads, n_nodes)).tocsr()

    # remove missing values
    non_missing = y_vec != -1.0

    logreg = LogisticRegression(fit_intercept=False, C=1e5)
    logreg.fit(X[non_missing], y_vec[non_missing])

    return logreg.coef_[0]


def initialize_node_effects(Y):
    n_layers, n_nodes, _ = Y.shape

    delta = np.zeros((n_layers, n_nodes))
    for k in range(n_layers):
        delta[k] = initialize_node_effects_single(Y[k])

    return delta


def initialize_node_effects_cont(Y):
    n_nodes = Y.shape[0]

    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    dyads = np.tril_indices_from(Y, k=-1)
    y_vec = Y[dyads]

    # construct dummy node indicators
    cols = np.r_[dyads[0], dyads[1]]
    rows = np.r_[np.arange(n_dyads), np.arange(n_dyads)]
    X = sp.coo_matrix((np.ones(2 * n_dyads), (rows, cols)),
                       shape=(n_dyads, n_nodes)).tocsr()

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y_vec)

    return reg.coef_


def initialize_lambda(Y, U):
    n_nodes, n_features = U.shape
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    dyads = np.tril_indices_from(Y, k=-1)
    
    y_vec = Y[dyads] 
    X = np.zeros((n_dyads, n_features))
    for p in range(n_features):
        u = U[:, p].reshape(-1, 1)
        X[:, p] = (u @ u.T)[dyads]
         
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y_vec)

    return reg.coef_ 


def initialize_svt(Y, n_features, eps=1e-3):
    n_layers, n_nodes, _ = Y.shape

    delta_init = np.zeros((n_layers, n_nodes))
    resid = np.zeros((n_layers, n_nodes, n_nodes))
    lmbda_init = np.zeros((n_layers, n_features))
    V = None
    for k in range(n_layers):
        A = Y[k].copy()
        A[A == -1] = 0
        dyads = np.tril_indices_from(A, k=-1)
        tau = np.sqrt(n_nodes * np.mean(A[dyads]))
        u,s,v = np.linalg.svd(A)
        ids = s >= tau
        P_tilde = np.clip(u[:, ids] @ np.diag(s[ids]) @ v[ids, :], eps, 1-eps)
        Theta = logit(0.5 * (P_tilde + P_tilde.T))

        delta_init[k] = initialize_node_effects_cont(Theta)

        if n_features > 0:
            d = delta_init[k].reshape(-1, 1)
            resid[k] = Theta - d - d.T
            eigvals, eigvecs = np.linalg.eigh(resid[k])

            ids = np.argsort(np.abs(eigvals))[::-1]
            eigvecs = eigvecs[:, ids][:, :n_features] 
            if V is None:
                V = eigvecs
            else:
                V = np.hstack((V, eigvecs)) 
    
    if n_features > 0:
        X = np.zeros((n_nodes, n_features))
        u, s, v = np.linalg.svd(V)
        X = u[:, :n_features]

        for k in range(n_layers):
            lmbda_init[k] = initialize_lambda(resid[k], X)
        
        lambda0 = np.abs(lmbda_init[0])
        lmbda_init = lmbda_init / lambda0 
        X = np.sqrt(lambda0) * X
    else:
        X = None
        lmbda_init = None
    
    return X, lmbda_init, delta_init


def initialize_parameters(Y, n_features, lambda_odds_prior, lambda_var_prior,
                          init_covariance_type, init_params_type,
                          callback, random_state):
    rng = check_random_state(random_state)

    n_layers, n_nodes, _ = Y.shape

    # omega is initialized by drawing from the prior?
    omega = np.zeros((n_layers, n_nodes, n_nodes))

    # initialize latent space randomly and center to remove effect
    # social trajectory initialization
    if n_features > 0:
        if init_params_type == 'svt':
            X, lmbda, delta = initialize_svt(Y, n_features)
        else:
            X = rng.randn(n_nodes, n_features)
            X -= np.mean(X, axis=0)

        # initialize to marginal covariances
        sigma_init = np.eye(n_features)
        X_sigma = np.tile(
            sigma_init[None], reps=(n_nodes, 1, 1))

        # initialize to prior means
        if init_params_type != 'svt':
            lmbda = np.sqrt(2) * rng.randn(n_layers, n_features)
            lmbda[0] = rng.choice([-1, 1], 1)

        lmbda_sigma = lambda_var_prior * np.ones(
            (n_layers, n_features, n_features))

        # reference layer lambda initialized to one
        lmbda_sigma[0] = (
            (1 - lmbda[0, 0] ** 2) * np.eye(n_features))
        lmbda_logit_prior = np.log(lambda_odds_prior)
    else:
        X, lmbda, delta = initialize_svt(Y, n_features)
        X = None
        X_sigma = None
        lmbda = None
        lmbda_sigma = None
        lmbda_logit_prior = np.log(lambda_odds_prior)

    # initialize node-effects based on degree
    if init_params_type != 'svt' and n_features > 0:
        delta = initialize_node_effects(Y)

    delta_sigma = np.ones((n_layers, n_nodes))

    # initialize based on prior information
    X0_cov_scale = np.eye(n_features)
    X0_cov_df = n_features + 2
    a_tau_sq = np.full(n_features, 1.5) # 0.5 * (n_features + 2)
    b_tau_sq = np.full(n_features, 0.5)

    a_tau_sq_delta = 4.1
    b_tau_sq_delta = 2.1 * 10

    return ModelParameters(
        omega=omega, X=X, X_sigma=X_sigma,
        lmbda=lmbda, lmbda_sigma=lmbda_sigma,
        lmbda_logit_prior=lmbda_logit_prior,
        delta=delta, delta_sigma=delta_sigma,
        X0_cov_scale=X0_cov_scale, X0_cov_df=X0_cov_df,
        a_tau_sq=a_tau_sq, b_tau_sq=b_tau_sq,
        a_tau_sq_delta=a_tau_sq_delta, b_tau_sq_delta=b_tau_sq_delta,
        callback=callback)



def optimize_elbo(Y, n_features, lambda_odds_prior, lambda_var_prior,
                  init_covariance_type, init_params_type,
                   max_iter, tol, random_state,
                  stopping_criteria='loglik',
                  callback=None, verbose=True):

    n_layers, n_nodes, _ = Y.shape

    # convergence criteria:
    #   loglik: Eq{L(Y | theta)})
    #   auc: training AUC
    criteria = -np.infty
    n_nochange = 0

    # initialize parameters of the model
    model = initialize_parameters(
        Y, n_features, lambda_odds_prior, lambda_var_prior,
        init_covariance_type, init_params_type,
        callback, random_state)

    a = np.full(n_features, 1.5)  # 0.5 * (n_features + 2)
    b = np.full(n_features, 0.5)
    X0_cov_prior_df = n_features + 2
    X0_cov_prior_scale = np.eye(n_features)

    a_delta = 4.1
    b_delta = 2.1 * 10
    for n_iter in tqdm(range(max_iter), disable=not verbose):
        prev_criteria = criteria

        # coordinate ascent

        # update auxiliary PG variables
        loglik = update_omega(
            Y, model.omega_, model.X_, model.X_sigma_,
            model.lambda_, model.lambda_sigma_,
            model.delta_, model.delta_sigma_,
            n_features)

        # update latent trajectories
        if init_covariance_type == 'full':
            X0_cov_prec = model.X0_cov_df_ * model.X0_cov_scale_
        else:
            X0_cov_prec = np.diag(model.a_tau_sq_ / model.b_tau_sq_)

        XLX = np.zeros((n_layers, n_nodes, n_nodes))
        if n_features > 0:
            update_latent_positions(
                Y, model.X_, model.X_sigma_,
                model.lambda_, model.lambda_sigma_, model.delta_,
                model.omega_, X0_cov_prec)

            # update homophily parameters
            update_lambdas(
                Y, model.X_, model.X_sigma_, model.lambda_,
                model.lambda_sigma_, model.delta_, model.omega_,
                lambda_var_prior, model.lambda_logit_prior_)

            # update social trajectories
            for k in range(n_layers):
                XLX[k] = np.dot(
                    model.X_ * model.lambda_[k], model.X_.T)

        tau_sq_prec = model.a_tau_sq_delta_ / model.b_tau_sq_delta_
        update_deltas(
            Y, model.delta_, model.delta_sigma_,
            XLX, model.omega_, tau_sq_prec)

        # update initial variance of the latent space
        if n_features > 0:
            if init_covariance_type == 'full':
                model.X0_cov_df_, model.X0_cov_scale_ = update_X0_precision(
                    Y, model.X_, model.X_sigma_, X0_cov_prior_df,
                    X0_cov_prior_scale)
            elif init_covariance_type == 'diag':
                model.a_tau_sq_, model.b_tau_sq_ = update_diag_tau_sq(
                    Y, model.X_, model.X_sigma_, a, b)
            else:
                model.a_tau_sq_, model.b_tau_sq_ = update_tau_sq(
                    Y, model.X_, model.X_sigma_, a, b)

        # update initial variance of the social trajectories
        model.a_tau_sq_delta_, model.b_tau_sq_delta_ = update_tau_sq_delta(
            model.delta_, model.delta_sigma_, a_delta, b_delta)

        model.logp_.append(loglik)

        # callback
        if model.callback_ is not None:
            model.callback_(model, Y)

        if stopping_criteria == 'auc':
            probas = calculate_probabilities(
                model.X_, model.lambda_, model.delta_)
            criteria = calculate_auc(Y, probas)
        else:
            criteria = loglik
        model.criteria_.append(criteria)

        # check convergence
        change = criteria - prev_criteria
        if stopping_criteria == 'auc':
            if abs(change) < tol and criteria > 0.55:
                n_nochange += 1
                if n_nochange > 4:
                    model.converged_ = True
                    model.logp_ = np.asarray(model.logp_)
                    break
            else:
                n_nochange = 0
        else:
            if abs(change) < tol:
                n_nochange += 1
                if n_nochange >= 2:
                    model.converged_ = True
                    model.logp_ = np.asarray(model.logp_)
                    break
            else:
                n_nochange = 0

    return model


def calculate_probabilities(X, lmbda, delta):
    n_layers = delta.shape[0]
    n_nodes = delta.shape[1]

    probas = np.zeros(
        (n_layers, n_nodes, n_nodes), dtype=np.float64)
    for k in range(n_layers):
        deltak = delta[k].reshape(-1, 1)
        eta = np.add(deltak, deltak.T)
        if X is not None:
            eta += np.dot(X * lmbda[k], X.T)
        probas[k] = expit(eta)

    return probas


class MultilayerNetworkLSM(object):
    """An Eigenmodel for Multilayer Networks

    Parameters
    ----------
    n_features : int (default=2)
        The number of latent features. This is the dimension of the
        latent space.

    n_init : int (default=1)
        The number of initializations to perform. The result with the highest
        expected log-likelihood is kept.

    max_iter : int (default=500)
        The number of coordinate ascent variational inference (CAVI) iterations
        to perform.

    tol : float (default=1e-2)
        The convergence threshold. CAVI iterations will stop when the expected
        log-likelihood gain is below this threshold.

    lambda_odds_prior : float (default=2)
        The prior odds of a component in the reference layering being positive.
        Our prior assumes an assortative reference layer is twice as likely
        as a disassortative reference layer.

    lambda_var_prior : float (default=4)
        The variance of the normal prior placed on the assortativity parameters.

    n_jobs : int (default=1)
        The number of jobs to run in parallel. The number of initializations are
        run in parallel. `-1` means using all processors.

    random_state : int, RandomState instance or None (default=None)
        Controls the random seed given to the method chosen to initialize
        the parameters. In addition, it controls generation of random samples
        from the fitted posterior distribution. Pass an int for reproducible
        output across multiple function calls.
    """
    def __init__(self, n_features=2,
                 lambda_odds_prior=1,
                 lambda_var_prior=10,
                 init_covariance_type='full',
                 init_type='svt', n_init=1, max_iter=1000, tol=1e-2,
                 stopping_criteria='loglik',
                 n_jobs=1, random_state=None):
        self.n_features = n_features
        self.lambda_odds_prior = lambda_odds_prior
        self.lambda_var_prior = lambda_var_prior
        self.init_covariance_type = init_covariance_type
        self.init_type = init_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.stopping_criteria = stopping_criteria
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, Y, callback=None):
        Y = check_array(Y, order='C', dtype=np.float64,
                        ensure_2d=False, allow_nd=True, copy=False)

        self.n_features_ = self.n_features if self.n_features is not None else 0

        random_state = check_random_state(self.random_state)
        
        if self.init_type == 'svt':
            self.n_init = 1

        # run the elbo optimization over different initializations
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        verbose = True if self.n_init == 1 else False
        models = Parallel(n_jobs=self.n_jobs)(delayed(optimize_elbo)(
                Y, self.n_features_, self.lambda_odds_prior,
                self.lambda_var_prior, self.init_covariance_type,
                self.init_type,
                self.max_iter, self.tol, seed,
                stopping_criteria=self.stopping_criteria,
                callback=callback, verbose=verbose)
            for seed in seeds)

        # choose model with the largest convergence criteria
        best_model = models[0]
        best_criteria = models[0].criteria_[-1]
        for i in range(1, len(models)):
            if models[i].criteria_[-1] > best_criteria:
                best_model = models[i]

        if not best_model.converged_:
            warnings.warn('Best model did not converge. '
                          'Try a different random initialization, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.', ConvergenceWarning)

        self._set_parameters(best_model)

        # calculate dyad-probabilities
        self.probas_ = calculate_probabilities(
            self.X_, self.lambda_, self.delta_)

        # identifiable distances
        n_layers, n_nodes, _ = Y.shape

        # calculate in-sample AUC
        self.auc_ = calculate_auc(Y, self.probas_)

        return self

    def _set_parameters(self, model):
        self.omega_ = model.omega_
        self.X_ = model.X_
        self.X_sigma_ = model.X_sigma_

        n_nodes, n_features = self.X_.shape

        if self.init_covariance_type == 'full':
            self.init_cov_df_ = model.X0_cov_df_
            self.init_cov_scale_ = model.X0_cov_scale_
            self.init_cov_ = (np.linalg.pinv(self.init_cov_scale_) /
                (self.init_cov_df_ - n_features - 1))
        else:
            self.a_tau_sq_ = model.a_tau_sq_
            self.b_tau_sq_ = model.b_tau_sq_
            self.init_cov_ = np.diag(self.b_tau_sq_ / (self.a_tau_sq_ - 1))

        self.lambda_ = model.lambda_
        if self.lambda_ is not None:
            self.lambda_[0] = np.sign(model.lambda_[0])
            self.lambda_proba_ = (model.lambda_[0] + 1) / 2.
        else:
            self.lambda_proba_ = None

        self.lambda_sigma_ = model.lambda_sigma_

        self.delta_ = model.delta_
        self.delta_sigma_ = model.delta_sigma_

        self.a_tau_sq_delta_ = model.a_tau_sq_delta_
        self.b_tau_sq_delta_ = model.b_tau_sq_delta_
        self.tau_sq_delta_ = self.b_tau_sq_delta_ / (self.a_tau_sq_delta_ - 1)

        self.logp_ = model.logp_
        self.criteria_ = model.criteria_
        self.callback_ = model.callback_
        self.converged_ = model.converged_
