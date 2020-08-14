import numbers
import warnings

import numpy as np
import scipy.sparse as sp

from joblib import Parallel, delayed
from scipy.special import logit, gammainc, expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_array, check_random_state
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from .node_svi import update_node_effects
from .lmbdas_svi import update_lambdas
from .variances import update_tau_sq, update_sigma_sq
from .log_likelihood import log_likelihood
from .metrics import calculate_auc


__all__ = ['DynamicMultilayerNetworkLSM']



class ModelParameters(object):
    def __init__(self, X, X_sigma, X_cross_cov, X_eta1,
                 X_eta2, lmbda, lmbda_sigma,
                 lmbda_logit_prior, lmbda0_eta, lmbda_eta1, lmbda_eta2,
                 delta, delta_sigma, delta_eta1, delta_eta2,
                 a_tau_sq, b_tau_sq, c_sigma_sq,
                 d_sigma_sq, logp=None):
        self.X_ = X
        self.X_sigma_ = X_sigma
        self.X_cross_cov_ = X_cross_cov
        self.X_eta1_ = X_eta1
        self.X_eta2_ = X_eta2

        self.lambda_ = lmbda
        self.lambda_sigma_ = lmbda_sigma
        self.lambda_logit_prior_ = lmbda_logit_prior
        self.lambda0_eta_ = lmbda0_eta
        self.lambda_eta1_ = lmbda_eta1
        self.lambda_eta2_ = lmbda_eta2

        self.delta_ = delta
        self.delta_sigma_ = delta_sigma
        self.delta_eta1_ = delta_eta1
        self.delta_eta2_ = delta_eta2

        self.a_tau_sq_ = a_tau_sq
        self.b_tau_sq_ = b_tau_sq
        self.c_sigma_sq_ = c_sigma_sq
        self.d_sigma_sq_ = d_sigma_sq
        self.converged_ = False

        self.logp_ = [] if logp is None else logp


def initialize_node_effects_single(Y):
    n_time_steps, n_nodes, _ = Y.shape

    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    dyads = np.tril_indices_from(Y[0], k=-1)
    y_vec = np.zeros(n_time_steps * n_dyads)

    # construct dummy node indicators
    cols = np.r_[dyads[0], dyads[1]]
    rows = np.r_[np.arange(n_dyads), np.arange(n_dyads)]
    x_dummy = sp.coo_matrix((np.ones(2 * n_dyads), (rows, cols)),
                            shape=(n_dyads, n_nodes))

    # dyad target
    for t in range(n_time_steps):
        yt_vec = Y[t][dyads]
        y_vec[(t * n_dyads):((t+1) * n_dyads)] = Y[t][dyads]
        if t > 0:
            X = sp.vstack((X, x_dummy))
        else:
            X = x_dummy.copy()
    X = X.tocsr()

    # remove missing values
    non_missing = y_vec != -1.0

    logreg = LogisticRegression(fit_intercept=False, C=1e5)
    logreg.fit(X[non_missing], y_vec[non_missing])

    return logreg.coef_[0]


def initialize_node_effects(Y):
    n_layers, n_time_steps, n_nodes, _ = Y.shape

    delta = np.zeros((n_layers, n_nodes))
    for k in range(n_layers):
        delta[k] = initialize_node_effects_single(Y[k])

    return delta


def initialize_parameters(Y, n_features, lambda_odds_prior, lambda_var_prior,
                          delta_var_prior,
                          a, b, c, d, random_state):
    rng = check_random_state(random_state)

    n_layers, n_time_steps, n_nodes, _ = Y.shape

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

    # initialize natural parameters
    X_eta1 = np.zeros((n_nodes, n_time_steps, n_features), dtype=np.float64)
    X_eta2 = np.zeros(
        (n_nodes, n_time_steps, n_features, n_features), dtype=np.float64)

    # initialize node-effects based on a logistic regression fit
    delta = initialize_node_effects(Y)
    delta_sigma = delta_var_prior * np.ones((n_layers, n_nodes))

    # initialize natural parameters
    delta_eta1 = np.zeros((n_layers, n_nodes), dtype=np.float64)
    delta_eta2 = np.zeros((n_layers, n_nodes), dtype=np.float64)

    # intialize to prior means
    lmbda = np.sqrt(2) * rng.randn(n_layers, n_features)
    lmbda[0] = (
        2 * (lambda_odds_prior / (1. + lambda_odds_prior)) - 1)
    lmbda_sigma = lambda_var_prior * np.ones(
        (n_layers, n_features, n_features))
    lmbda_sigma[0] = (
        (1 - lmbda[0, 0] ** 2) * np.eye(n_features))
    lmbda_logit_prior = np.log(lambda_odds_prior)

    # initialize natural parameters
    lmbda0_eta = np.zeros(n_features, dtype=np.float64)
    lmbda_eta1 = np.zeros((n_layers, n_features), dtype=np.float64)
    lmbda_eta2 = np.zeros((n_layers, n_features, n_features), dtype=np.float64)

    # initialize based on prior information
    a_tau_sq = a
    b_tau_sq = b
    c_sigma_sq = c
    d_sigma_sq = d

    return ModelParameters(
        X=X, X_sigma=X_sigma, X_cross_cov=X_cross_cov,
        X_eta1=X_eta1, X_eta2=X_eta2,
        lmbda=lmbda,
        lmbda_sigma=lmbda_sigma, lmbda_logit_prior=lmbda_logit_prior,
        lmbda0_eta=lmbda0_eta, lmbda_eta1=lmbda_eta1, lmbda_eta2=lmbda_eta2,
        delta=delta, delta_sigma=delta_sigma, delta_eta1=delta_eta1,
        delta_eta2=delta_eta2,
        a_tau_sq=a_tau_sq, b_tau_sq=b_tau_sq, c_sigma_sq=c_sigma_sq,
        d_sigma_sq=d_sigma_sq)



def optimize_elbo(Y, n_features, lambda_odds_prior, lambda_var_prior,
                  delta_var_prior, tau_sq, sigma_sq, a, b, c, d,
                  gamma, max_degree, max_edges, n_neg_multiplier,
                  max_iter, tol, random_state, verbose=True):

    # initialize parameters of the model
    random_state = check_random_state(random_state)

    model = initialize_parameters(
        Y, n_features, lambda_odds_prior, lambda_var_prior, delta_var_prior,
        a, b, c, d, random_state)

    for n_iter in tqdm(range(max_iter), disable=not verbose):
        # stochastic gradient descent

        # update step size
        rho = (1 + n_iter) ** (-gamma)

        # latent trajectory updates
        tau_sq_prec = (
            model.a_tau_sq_ / model.b_tau_sq_ if tau_sq == 'auto' else
                1. / tau_sq)
        sigma_sq_prec = (
            model.c_sigma_sq_ / model.d_sigma_sq_ if sigma_sq == 'auto' else
                1. / sigma_sq)

        # update latent positions and degree effects
        node_params = update_node_effects(
                Y, model.X_, model.X_sigma_, model.X_eta1_, model.X_eta2_,
                model.lambda_, model.lambda_sigma_, model.delta_,
                model.delta_sigma_, model.delta_eta1_, model.delta_eta2_,
                delta_var_prior, tau_sq_prec, sigma_sq_prec,
                max_degree, n_neg_multiplier, rho, random_state)

        # update lambda values
        lmbda_params = update_lambdas(
            Y, model.X_, model.X_sigma_,  model.lambda_,
            model.lambda_sigma_, model.lambda0_eta_, model.lambda_eta1_,
            model.lambda_eta2_, model.delta_, model.delta_sigma_,
            lambda_var_prior, model.lambda_logit_prior_,
            max_edges, n_neg_multiplier, rho, random_state)

        # update intial variance of the latent space
        if tau_sq == 'auto':
            a_tau_sq, b_tau_sq = update_tau_sq(
                Y, model.X_, model.X_sigma_, a, b)

            a_tau_sq = (1 - rho) * model.a_tau_sq_ + rho * a_tau_sq
            b_tau_sq = (1 - rho) * model.b_tau_sq_ + rho * b_tau_sq
        else:
            a_tau_sq = model.a_tau_sq_
            b_tau_sq = model.b_tau_sq_

        # update step sizes
        if sigma_sq == 'auto':
            c_sigma_sq, d_sigma_sq = update_sigma_sq(
                Y, model.X_, model.X_sigma_, model.X_cross_cov_, c, d)

            c_sigma_sq = (1 - rho) * model.c_sigma_sq_ + rho * c_sigma_sq
            d_sigma_sq = (1 - rho) * model.d_sigma_sq_ + rho * d_sigma_sq
        else:
            c_sigma_sq = model.c_sigma_sq_
            d_sigma_sq = model.d_sigma_sq_


        # XXX: calculate change in latent space parameters
        change = ((model.X_ - node_params.X) ** 2).sum()
        model.logp_.append(change)

        # store updated model parameters
        model = ModelParameters(
            X=node_params.X, X_sigma=node_params.X_sigma,
            X_cross_cov=node_params.X_cross_cov, X_eta1=node_params.X_eta1,
            X_eta2=node_params.X_eta2, lmbda=lmbda_params.lmbda,
            lmbda_logit_prior=model.lambda_logit_prior_,
            lmbda_sigma=lmbda_params.lmbda_sigma,
            lmbda0_eta=lmbda_params.lmbda0_eta,
            lmbda_eta1=lmbda_params.lmbda_eta1,
            lmbda_eta2=lmbda_params.lmbda_eta2,
            delta=node_params.delta,
            delta_sigma=node_params.delta_sigma,
            delta_eta1=node_params.delta_eta1,
            delta_eta2=node_params.delta_eta2, a_tau_sq=a_tau_sq,
            b_tau_sq=b_tau_sq, c_sigma_sq=c_sigma_sq, d_sigma_sq=d_sigma_sq,
            logp=model.logp_)

        # We could calculate the expected log-likelihood on dyads sampled
        # to update lmbda

        # check convergence
        #change = loglik - prev_loglik
        if change < tol:
            model.converged_ = True
            #model.logp_ = np.asarray(model.logp_)
            break

    return model


def calculate_probabilities(X, lmbda, delta):
    n_layers = lmbda.shape[0]
    n_time_steps = X.shape[0]
    n_nodes = X.shape[1]

    probas = np.zeros(
        (n_layers, n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    for k in range(n_layers):
        for t in range(n_time_steps):
            deltak = delta[k].reshape(-1, 1)
            eta = np.add(deltak, deltak.T) + np.dot(X[t] * lmbda[k], X[t].T)
            probas[k, t] = expit(eta)

    return probas


class DynamicMultilayerNetworkLSM(object):
    def __init__(self, n_features=2,
                 lambda_odds_prior=2,
                 lambda_var_prior=4,
                 delta_var_prior=4,
                 tau_sq='auto', sigma_sq='auto',
                 a=4.0, b=8.0, c=10, d=0.1,
                 gamma=0.75, n_neg_multiplier=2,
                 n_init=1, max_iter=500, tol=1e-2,
                 n_jobs=-1, random_state=42):
        self.n_features = n_features
        self.lambda_odds_prior = lambda_odds_prior
        self.lambda_var_prior = lambda_var_prior
        self.delta_var_prior = delta_var_prior
        self.tau_sq = tau_sq
        self.sigma_sq = sigma_sq
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.gamma = gamma
        self.n_neg_multiplier = n_neg_multiplier
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

        n_nodes = Y.shape[2]

        # maximum degree, i.e. the maximum number of edges connected to a node
        max_degree = int(np.count_nonzero(Y == 1, axis=3).max())

        # maximum edge count
        max_edges = int((np.count_nonzero(Y == 1, axis=(2, 3)) / 2).max())

        if self.gamma >= 1.0:
            raise ValueError('Decay rate `gamma` must be between 0 and 1')

        random_state = check_random_state(self.random_state)

        # run the elbo optimization over different initializations
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        verbose = True if self.n_init == 1 else False
        models = Parallel(n_jobs=self.n_jobs)(delayed(optimize_elbo)(
                Y, self.n_features, self.lambda_odds_prior,
                self.lambda_var_prior, self.delta_var_prior,
                self.tau_sq, self.sigma_sq, self.a, self.b, self.c, self.d,
                self.gamma, max_degree, max_edges, self.n_neg_multiplier,
                self.max_iter, self.tol, seed, verbose=verbose)
            for seed in seeds)

        # choose model with the largest convergence criteria
        best_model = models[0]
        #best_criteria = models[0].logp_[-1]
        #for i in range(1, len(models)):
        #    if models[i].logp_[-1] > best_criteria:
        #        best_model = models[i]

        if not best_model.converged_:
            warnings.warn('Best model did not converge. '
                          'Try a different random initialization, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.', ConvergenceWarning)

        self._set_parameters(best_model)

        # calculate dyad-probabilities
        self.probas_ = calculate_probabilities(
            self.X_, self.lambda_, self.delta_)

        # calculate in-sample AUC
        self.auc_ = calculate_auc(Y, self.probas_)

        return self

    def _set_parameters(self, model):
        self.X_ = model.X_
        self.X_sigma_ = model.X_sigma_
        self.X_cross_cov_ = model.X_cross_cov_
        self.delta_ = np.asarray(model.delta_)
        self.delta_sigma_ = np.asarray(model.delta_sigma_)
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
        self.logp_ = np.asarray(model.logp_)
