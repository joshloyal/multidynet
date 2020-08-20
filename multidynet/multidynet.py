import warnings

import numpy as np
import scipy.sparse as sp

from joblib import Parallel, delayed
from scipy.special import logit, gammainc, expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from .omega import update_omega
from .lds import update_latent_positions
from .lmbdas import update_lambdas
from .deltas import update_deltas
from .variances import update_tau_sq, update_sigma_sq
from .metrics import calculate_auc


__all__ = ['DynamicMultilayerNetworkLSM']



class ModelParameters(object):
    def __init__(self, omega, X, X_sigma, X_cross_cov,
                 lmbda, lmbda_sigma, lmbda_logit_prior, delta, delta_sigma,
                 a_tau_sq, b_tau_sq, c_sigma_sq, d_sigma_sq):
        self.omega_ = omega
        self.X_ = X
        self.X_sigma_ = X_sigma
        self.X_cross_cov_ = X_cross_cov
        self.lambda_ = lmbda
        self.lambda_sigma_ = lmbda_sigma
        self.lambda_logit_prior_ = lmbda_logit_prior
        self.delta_ = delta
        self.delta_sigma_ = delta_sigma
        self.a_tau_sq_ = a_tau_sq
        self.b_tau_sq_ = b_tau_sq
        self.c_sigma_sq_ = c_sigma_sq
        self.d_sigma_sq_ = d_sigma_sq
        self.converged_ = False
        self.logp_ = []


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
                          delta_var_prior, a, b, c, d, random_state):
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

    # intialize to prior means
    lmbda = np.sqrt(2) * rng.randn(n_layers, n_features)
    lmbda[0] = (
        2 * (lambda_odds_prior / (1. + lambda_odds_prior)) - 1)
    lmbda_sigma = lambda_var_prior * np.ones(
        (n_layers, n_features, n_features))
    lmbda_sigma[0] = (
        (1 - lmbda[0, 0] ** 2) * np.eye(n_features))
    lmbda_logit_prior = np.log(lambda_odds_prior)

    # initialize node-effects based on degree
    delta = initialize_node_effects(Y)
    delta_sigma = delta_var_prior * np.ones((n_layers, n_nodes))

    # initialize based on prior information
    a_tau_sq = a
    b_tau_sq = b
    c_sigma_sq = c
    d_sigma_sq = d

    return ModelParameters(
        omega=omega, X=X, X_sigma=X_sigma, X_cross_cov=X_cross_cov,
        lmbda=lmbda, lmbda_sigma=lmbda_sigma,
        lmbda_logit_prior=lmbda_logit_prior,
        delta=delta, delta_sigma=delta_sigma,
        a_tau_sq=a_tau_sq, b_tau_sq=b_tau_sq, c_sigma_sq=c_sigma_sq,
        d_sigma_sq=d_sigma_sq)



def optimize_elbo(Y, n_features, lambda_odds_prior, lambda_var_prior,
                  delta_var_prior, tau_sq, sigma_sq, a, b, c, d,
                  max_iter, tol, random_state, verbose=True):

    # convergence criteria (Eq{L(Y | theta)})
    loglik = -np.infty

    # initialize parameters of the model
    model = initialize_parameters(
        Y, n_features, lambda_odds_prior, lambda_var_prior, delta_var_prior,
        a, b, c, d, random_state)

    for n_iter in tqdm(range(max_iter), disable=not verbose):
        prev_loglik = loglik

        # coordinate ascent

        # omega updates
        loglik = update_omega(
            Y, model.omega_, model.X_, model.X_sigma_,
            model.lambda_, model.lambda_sigma_,
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
            model.lambda_, model.lambda_sigma_, model.delta_,
            model.omega_, tau_sq_prec, sigma_sq_prec)

        # update lambda values
        update_lambdas(
            Y, model.X_, model.X_sigma_, model.lambda_,
            model.lambda_sigma_, model.delta_, model.omega_, lambda_var_prior,
            model.lambda_logit_prior_)

        # update node random effects
        update_deltas(
            Y, model.X_, model.lambda_, model.delta_,
            model.delta_sigma_, model.omega_, delta_var_prior)

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
    """An Eigenmodel for Dynamic Multilayer Networks

    Parameters
    ----------
    n_features : int (default=2)
        The number of latent features. This is the dimension of the
        latent space.

    lambda_odds_prior : float (default=2)
        The prior odds of a component in the reference layering being positive.
        Our prior assumes an assortative reference layer is twice as likely
        as a disassortative reference layer.

    lambda_var_prior : float (default=4)
        The variance of the normal prior placed on the assortativity parameters.

    delta_var_prior : float (default=4)
        The variance of the normal prior placed on the degree random effects.

    tau_sq : float or str (default='auto')
        The variance of the normal prior placed on the initial distribution of
        the latent positions. If tau_sq == 'auto', then the value is inferred
        from the data.

    sigma_sq : float or str (default='auto')
        The random-walk variance of the latent positions. If sigma_sq == 'auto',
        then the value is inferred from the data.

    a : float (default=4.)
        Shape parameter of the InvGamma(a/2, b/2) prior placed on `tau_sq`.
        Only relevant if `tau_sq`=='auto'.

    b : float (default=8.)
        Scale parameter of the InvGamma(a/2, b/2) prior placed on `tau_sq`.
        Only relevant if `tau_sq`=='auto'.

    c : float (default=10.)
        Shape parameter of the InvGamma(c/2, d/2) prior placed on `sigma_sq`.
        Only relevant if `sigma_sq`=='auto.

    d : float (default=0.1)
        Scale parameter of the InvGamma(c/2, d/2) prior placed on `sigma_sq`.
        Only relevant if `sigma_sq`=='auto.

    n_init : int (default=1)
        The number of initializations to perform. The result with the highest
        expected log-likelihood is kept.

    max_iter : int (default=500)
        The number of coordinate ascent variational inference (CAVI) iterations
        to perform.

    tol : float (default=1e-2)
        The convergence threshold. CAVI iterations will stop when the expected
        log-likelihood gain is below this threshold.

    n_jobs : int (default=-1)
        The number of jobs to run in parallel. The number of initializations are
        run in parallel. `-1` means using all processors.

    random_state : int, RandomState instance or None (default=42)
        Controls the random seed given to the method chosen to initialize
        the parameters. In addition, it controls generation of random samples
        from the fitted posterior distribution. Pass an int for reproducible
        output across multiple function calls.
    """
    def __init__(self, n_features=2,
                 lambda_odds_prior=2,
                 lambda_var_prior=4,
                 delta_var_prior=4,
                 tau_sq='auto', sigma_sq='auto',
                 a=4.0, b=8.0, c=10., d=0.1,
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
                self.lambda_var_prior, self.delta_var_prior,
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
            self.X_, self.lambda_, self.delta_)

        # calculate in-sample AUC
        self.auc_ = calculate_auc(Y, self.probas_)

        return self

    def _set_parameters(self, model):
        self.omega_ = model.omega_
        self.X_ = model.X_
        self.X_sigma_ = model.X_sigma_
        self.X_cross_cov_ = model.X_cross_cov_
        self.lambda_ = model.lambda_
        self.lambda_[0] = np.sign(model.lambda_[0])
        self.lambda_proba_ = (model.lambda_[0] + 1) / 2.
        self.lambda_sigma_ = model.lambda_sigma_
        self.delta_ = model.delta_
        self.delta_sigma_ = model.delta_sigma_
        self.a_tau_sq_ = model.a_tau_sq_
        self.b_tau_sq_ = model.b_tau_sq_
        self.tau_sq_ = self.b_tau_sq_ / (self.a_tau_sq_ - 1)
        self.c_sigma_sq_ = model.c_sigma_sq_
        self.d_sigma_sq_ = model.d_sigma_sq_
        self.sigma_sq_ = self.d_sigma_sq_ / (self.c_sigma_sq_ - 1)
        self.logp_ = model.logp_
        self.converged_ = model.converged_
