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
from .deltas_lds import update_deltas
from .lmbdas import update_lambdas
from .variances import update_tau_sq, update_sigma_sq
from .variances import update_tau_sq_delta, update_sigma_sq_delta
from .log_likelihood import log_likelihood
from .metrics import calculate_auc
from .sample_lds import sample_gssm


__all__ = ['DynamicMultilayerNetworkLSM']


class ModelParameters(object):
    def __init__(self, omega, X, X_sigma, X_cross_cov,
                 lmbda, lmbda_sigma, lmbda_logit_prior,
                 delta, delta_sigma, delta_cross_cov,
                 a_tau_sq, b_tau_sq, c_sigma_sq, d_sigma_sq,
                 a_tau_sq_delta, b_tau_sq_delta, c_sigma_sq_delta,
                 d_sigma_sq_delta):
        self.omega_ = omega
        self.X_ = X
        self.X_sigma_ = X_sigma
        self.X_cross_cov_ = X_cross_cov
        self.lambda_ = lmbda
        self.lambda_sigma_ = lmbda_sigma
        self.lambda_logit_prior_ = lmbda_logit_prior
        self.delta_ = delta
        self.delta_sigma_ = delta_sigma
        self.delta_cross_cov_ = delta_cross_cov
        self.a_tau_sq_ = a_tau_sq
        self.b_tau_sq_ = b_tau_sq
        self.c_sigma_sq_ = c_sigma_sq
        self.d_sigma_sq_ = d_sigma_sq
        self.a_tau_sq_delta_ = a_tau_sq_delta
        self.b_tau_sq_delta_ = b_tau_sq_delta
        self.c_sigma_sq_delta_ = c_sigma_sq_delta
        self.d_sigma_sq_delta_ = d_sigma_sq_delta
        self.converged_ = False
        self.logp_ = []


def sample_socialities(model, size=500, random_state=None):
    deltas, Xs, lambdas = model.sample(size=size, random_state=random_state)

    X_bar = np.mean(Xs, axis=2)
    Xs -= np.expand_dims(X_bar, axis=2)

    _, n_layers, n_time_steps, n_nodes =  deltas.shape
    gammas = np.zeros((size, n_layers, n_time_steps, n_nodes))
    MLM = np.einsum('stp,skp->kst', X_bar ** 2, lambdas)
    for k in range(n_layers):
        for i in range(n_nodes):
            ZL = Xs[:, :, i] * np.expand_dims(lambdas[:, k, :], axis=1)
            ZLM = np.einsum('stp,stp->st', ZL, X_bar)
            gammas[:, k, :, i] = deltas[:, k, :, i] + ZLM + 0.5 * MLM[k]

    gamma_mean = np.mean(gammas, axis=0)
    gamma_ci = np.quantile(gammas, [0.025, 0.975], axis=0)
    return gammas, gamma_mean, gamma_ci


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
    n_layers, n_time_steps, n_nodes, _ = Y.shape

    delta = np.zeros((n_layers, n_time_steps, n_nodes))
    for k in range(n_layers):
        for t in range(n_time_steps):
            delta[k, t] = initialize_node_effects_single(Y[k, t])

    return delta


def initialize_parameters(Y, n_features, lambda_odds_prior, lambda_var_prior,
                          a, b, c, d, a_delta, b_delta, c_delta, d_delta,
                          random_state):
    rng = check_random_state(random_state)

    n_layers, n_time_steps, n_nodes, _ = Y.shape

    # omega is initialized by drawing from the prior?
    omega = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))

    # initialize latent space randomly and center to remove effect
    # social trajectory initialization
    if n_features > 0:
        X = rng.randn(n_time_steps, n_nodes, n_features)
        for t in range(n_time_steps):
            X[t] -= np.mean(X[t], axis=0)

        # initialize to marginal covariances
        sigma_init = np.eye(n_features)
        X_sigma = np.tile(
            sigma_init[None, None], reps=(n_time_steps, n_nodes, 1, 1))

        # initialize cross-covariances
        cross_init = np.eye(n_features)
        X_cross_cov = np.tile(
            cross_init[None, None], reps=(n_time_steps - 1, n_nodes, 1, 1))

        # initialize to prior means
        lmbda = np.sqrt(2) * rng.randn(n_layers, n_features)
        lmbda_sigma = lambda_var_prior * np.ones(
            (n_layers, n_features, n_features))

        # reference layer lambda initialized to one
        lmbda[0] = 1.
        lmbda_sigma[0] = (
            (1 - lmbda[0, 0] ** 2) * np.eye(n_features))
        lmbda_logit_prior = np.log(lambda_odds_prior)
    else:
        X = None
        X_sigma = None
        X_cross_cov = None
        lmbda = None
        lmbda_sigma = None
        lmbda_logit_prior = np.log(lambda_odds_prior)

    # initialize node-effects based on degree
    delta = initialize_node_effects(Y)
    delta_sigma = np.ones((n_layers, n_time_steps, n_nodes))
    delta_cross_cov = np.ones((n_layers, n_time_steps - 1, n_nodes))

    # initialize based on prior information
    a_tau_sq = a
    b_tau_sq = b
    c_sigma_sq = c
    d_sigma_sq = d

    a_tau_sq_delta = a_delta
    b_tau_sq_delta = b_delta
    c_sigma_sq_delta = c_delta
    d_sigma_sq_delta = d_delta

    return ModelParameters(
        omega=omega, X=X, X_sigma=X_sigma, X_cross_cov=X_cross_cov,
        lmbda=lmbda, lmbda_sigma=lmbda_sigma,
        lmbda_logit_prior=lmbda_logit_prior,
        delta=delta, delta_sigma=delta_sigma, delta_cross_cov=delta_cross_cov,
        a_tau_sq=a_tau_sq, b_tau_sq=b_tau_sq, c_sigma_sq=c_sigma_sq,
        d_sigma_sq=d_sigma_sq, a_tau_sq_delta=a_delta, b_tau_sq_delta=b_delta,
        c_sigma_sq_delta=c_sigma_sq_delta, d_sigma_sq_delta=d_sigma_sq_delta)



def optimize_elbo(Y, n_features, lambda_odds_prior, lambda_var_prior,
                  a, b, c, d, a_delta, b_delta, c_delta, d_delta,
                  max_iter, tol, random_state, verbose=True):

    n_layers, n_time_steps, n_nodes, _ = Y.shape

    # convergence criteria (Eq{L(Y | theta)})
    loglik = -np.infty

    # initialize parameters of the model
    model = initialize_parameters(
        Y, n_features, lambda_odds_prior, lambda_var_prior,
        a, b, c, d, a_delta, b_delta, c_delta, d_delta, random_state)

    for n_iter in tqdm(range(max_iter), disable=not verbose):
        prev_loglik = loglik

        # coordinate ascent

        # update auxiliary PG variables
        loglik = update_omega(
            Y, model.omega_, model.X_, model.X_sigma_,
            model.lambda_, model.lambda_sigma_,
            model.delta_, model.delta_sigma_,
            n_features)

        # update latent trajectories
        tau_sq_prec = model.a_tau_sq_ / model.b_tau_sq_
        sigma_sq_prec = model.c_sigma_sq_ / model.d_sigma_sq_


        XLX = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))
        if n_features > 0:
            update_latent_positions(
                Y, model.X_, model.X_sigma_, model.X_cross_cov_,
                model.lambda_, model.lambda_sigma_, model.delta_,
                model.omega_, tau_sq_prec, sigma_sq_prec)

            # update homophily parameters
            update_lambdas(
                Y, model.X_, model.X_sigma_, model.lambda_,
                model.lambda_sigma_, model.delta_, model.omega_,
                lambda_var_prior, model.lambda_logit_prior_)

            # update social trajectories
            for k in range(n_layers):
                for t in range(n_time_steps):
                    XLX[k, t] = np.dot(
                        model.X_[t] * model.lambda_[k], model.X_[t].T)

        tau_sq_prec = model.a_tau_sq_delta_ / model.b_tau_sq_delta_
        sigma_sq_prec = model.c_sigma_sq_delta_ / model.d_sigma_sq_delta_

        update_deltas(
            Y, model.delta_, model.delta_sigma_, model.delta_cross_cov_,
            XLX, model.omega_, tau_sq_prec, sigma_sq_prec)

        # update initial variance of the latent space
        if n_features > 0:
            model.a_tau_sq_, model.b_tau_sq_ = update_tau_sq(
                Y, model.X_, model.X_sigma_, a, b)

            # update step sizes of the latent space
            model.c_sigma_sq_, model.d_sigma_sq_ = update_sigma_sq(
                Y, model.X_, model.X_sigma_, model.X_cross_cov_, c, d)

        # update initial variance of the social trajectories
        model.a_tau_sq_delta_, model.b_tau_sq_delta_ = update_tau_sq_delta(
            model.delta_, model.delta_sigma_, a_delta, b_delta)

        # update step sizes of the social trajectories
        model.c_sigma_sq_delta_, model.d_sigma_sq_delta_ = update_sigma_sq_delta(
            model.delta_, model.delta_sigma_, model.delta_cross_cov_,
            c_delta, d_delta)

        model.logp_.append(loglik)

        # check convergence
        change = loglik - prev_loglik
        if abs(change) < tol:
            model.converged_ = True
            model.logp_ = np.asarray(model.logp_)
            break

    return model


def calculate_probabilities(X, lmbda, delta):
    n_layers = delta.shape[0]
    n_time_steps = delta.shape[1]
    n_nodes = delta.shape[2]

    probas = np.zeros(
        (n_layers, n_time_steps, n_nodes, n_nodes), dtype=np.float64)
    for k in range(n_layers):
        for t in range(n_time_steps):
            deltakt = delta[k, t].reshape(-1, 1)
            eta = np.add(deltakt, deltakt.T)
            if X is not None:
                eta += np.dot(X[t] * lmbda[k], X[t].T)
            probas[k, t] = expit(eta)

    return probas


class DynamicMultilayerNetworkLSM(object):
    """An Eigenmodel for Dynamic Multilayer Networks

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

    a : float (default=4.)
        Shape parameter of the InvGamma(a/2, b/2) prior placed on `tau_sq`.

    b : float (default=20.)
        Scale parameter of the InvGamma(a/2, b/2) prior placed on `tau_sq`.

    c : float (default=20.)
        Shape parameter of the InvGamma(c/2, d/2) prior placed on `sigma_sq`.

    d : float (default=2.)
        Scale parameter of the InvGamma(c/2, d/2) prior placed on `sigma_sq`.

    a_delta : float (default=4.)
        Shape parameter of the InvGamma(a_delta/2, b_delta/2) prior placed
        on `tau_sq_delta`.

    b_delta : float (default=20.)
        Scale parameter of the InvGamma(a_delta/2, b_delta/2) prior placed
        on `tau_sq_delta`.

    c_delta : float (default=20.)
        Shape parameter of the InvGamma(c_delta/2, d_delta/2) prior placed
        on `sigma_sq_delta`.

    d_delta : float (default=2.)
        Scale parameter of the InvGamma(c_delta/2, d_delta/2) prior placed
        on `sigma_sq_delta`.

    n_jobs : int (default=1)
        The number of jobs to run in parallel. The number of initializations are
        run in parallel. `-1` means using all processors.

    random_state : int, RandomState instance or None (default=None)
        Controls the random seed given to the method chosen to initialize
        the parameters. In addition, it controls generation of random samples
        from the fitted posterior distribution. Pass an int for reproducible
        output across multiple function calls.

    Examples
    --------

    >>> from multidynet import DynamicMultilayerNetworkLSM
    >>> from dynetlsm.datasets import load_households
    >>> Y =
    >>> Y.shape
    (,,,)
    >>> model = DynamicMultilayerNetworkLSM().fit(Y)

    References
    ----------
    """
    def __init__(self, n_features=2,
                 lambda_odds_prior=1,
                 lambda_var_prior=10,
                 a=4.1, b=2.1 * 10, c=2., d=2.,
                 a_delta=4.1, b_delta=2.1 * 10, c_delta=2., d_delta=2.,
                 n_init=1, max_iter=1000, tol=1e-2,
                 n_jobs=1, random_state=None):
        self.n_features = n_features
        self.lambda_odds_prior = lambda_odds_prior
        self.lambda_var_prior = lambda_var_prior
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.a_delta = a_delta
        self.b_delta = b_delta
        self.c_delta = c_delta
        self.d_delta = d_delta
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, Y):
        """Infer the approximate variational posterior of the eigenmodel
        for dynamic multilayer networks based on the observed network Y.

        Parameters
        ----------
        Y : array-like, shape (n_layers, n_time_steps, n_nodes, n_nodes)
            The training dynamic multilayer network. The networks should be
            represented as binary undirected adjacency matrices. For example,
            Y[0] is an array of shape (n_time_steps, n_nodes, n_nodes)
            corresponding to the adjacency matrices of the networks at in
            the first layer. The network should be stored as
            ``dtype=np.float64``.

        Returns
        -------
        self : DynamicMultilayerNetworkLSM
            Fitted estimator.
        """
        Y = check_array(Y, order='C', dtype=np.float64,
                        ensure_2d=False, allow_nd=True, copy=False)

        if Y.ndim == 3:
            raise ValueError(
                "Y.ndim == {}, when it should be 4. "
                "If there is only a single layer, then reshape Y with "
                "Y = np.expand_dims(Y, axis=0) and re-fit.".format(Y.ndim))

        self.n_features_ = self.n_features if self.n_features is not None else 0

        random_state = check_random_state(self.random_state)

        # run the elbo optimization over different initializations
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        verbose = True if self.n_init == 1 else False
        models = Parallel(n_jobs=self.n_jobs)(delayed(optimize_elbo)(
                Y, self.n_features_, self.lambda_odds_prior,
                self.lambda_var_prior,
                self.a, self.b, self.c, self.d,
                self.a_delta, self.b_delta, self.c_delta, self.d_delta,
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

        # identifiable distances
        n_layers, n_time_steps, n_nodes, _ = Y.shape
        self.dist_ = np.zeros(
            (n_layers, n_time_steps, n_nodes, n_nodes), dtype=np.float64)
        if self.n_features_ > 0:
            for k in range(n_layers):
                for t in range(n_time_steps):
                    self.dist_[k, t] = np.dot(
                        self.Z_[t] * self.lambda_[k], self.Z_[t].T)

        # calculate in-sample AUC
        self.auc_ = calculate_auc(Y, self.probas_)

        # calculate in-sample log-likelihood
        self.loglik_ = log_likelihood(
            Y, self.X_, self.lambda_, self.delta_, self.n_features_)

        # information criteria

        # AIC
        self.p_aic_ = np.prod(self.delta_.shape)
        if self.n_features_ > 0:
            self.p_aic_ += np.prod(self.X_.shape)
            self.p_aic_ += np.prod(self.lambda_.shape)
        self.aic_ = -2 * self.loglik_ + 2 * self.p_aic_

        # BIC
        n_layers, n_time_steps, n_nodes = self.delta_.shape
        logn = (np.log(0.5 * n_nodes) + np.log(n_nodes - 1) +
            np.log(n_layers) + np.log(n_time_steps))
        self.p_bic_ = logn * np.prod(self.delta_.shape)
        if self.n_features_ > 0:
            self.p_bic_ += logn * np.prod(self.X_.shape)
            self.p_bic_ += logn * np.prod(self.lambda_.shape)
        self.bic_ = -2 * self.loglik_ + self.p_bic_

        return self

    def _set_parameters(self, model):
        self.omega_ = model.omega_

        self.X_ = model.X_
        self.X_sigma_ = model.X_sigma_
        self.X_cross_cov_ = model.X_cross_cov_

        # transform to identifiable parameterization
        if self.X_ is not None:
            self.Z_ = (self.X_ -
                np.expand_dims(np.mean(self.X_, axis=1), axis=1))

            n_time_steps, n_nodes, n_features = self.X_.shape
            self.Z_sigma_ = np.zeros(
                (n_time_steps, n_nodes, n_features, n_features))
            for i in range(n_nodes):
                self.Z_sigma_[:, i] = (
                    ((1 - 1./n_nodes) ** 2) * self.X_sigma_[:, i])
                self.Z_sigma_[:, i] -= ((1./n_nodes) ** 2) * np.sum(
                    self.X_sigma_[:, [j for j in range(n_nodes) if j != i]],
                    axis=1)
        else:
            self.Z_ = None
            self.Z_sigma_ = None

        self.a_tau_sq_ = model.a_tau_sq_
        self.b_tau_sq_ = model.b_tau_sq_
        self.tau_sq_ = self.b_tau_sq_ / (self.a_tau_sq_ - 1)
        self.c_sigma_sq_ = model.c_sigma_sq_
        self.d_sigma_sq_ = model.d_sigma_sq_
        self.sigma_sq_ = self.d_sigma_sq_ / (self.c_sigma_sq_ - 1)

        self.lambda_ = model.lambda_
        if self.lambda_ is not None:
            self.lambda_[0] = np.sign(model.lambda_[0])
            self.lambda_proba_ = (model.lambda_[0] + 1) / 2.
        else:
            self.lambda_proba_ = None

        self.lambda_sigma_ = model.lambda_sigma_

        self.delta_ = model.delta_
        self.delta_sigma_ = model.delta_sigma_
        self.delta_cross_cov_ = model.delta_cross_cov_

        # transform to identifiable parameters (requires sampling)
        if self.X_ is not None:
            self.gammas_, self.gamma_, self.gamma_ci_ = sample_socialities(
                self, size=2500, random_state=self.random_state)
        else:
            self.gammas_ = None
            self.gamma_ = None
            self.gamma_ci_ = None

        self.a_tau_sq_delta_ = model.a_tau_sq_delta_
        self.b_tau_sq_delta_ = model.b_tau_sq_delta_
        self.tau_sq_delta_ = self.b_tau_sq_delta_ / (self.a_tau_sq_delta_ - 1)
        self.c_sigma_sq_delta_ = model.c_sigma_sq_delta_
        self.d_sigma_sq_delta_ = model.d_sigma_sq_delta_
        self.sigma_sq_delta_ = (
            self.d_sigma_sq_delta_ / (self.c_sigma_sq_delta_ - 1))

        self.logp_ = model.logp_
        self.converged_ = model.converged_

    def sample(self, size=1, random_state=None):
        """Sample parameters from the model's approximate posterior.

        Parameters
        ----------
        size : int, (default=1)
            Number of samples to draw from the approximate posterior.

        Returns
        -------
        deltas  : np.ndarray of shape (size, n_layers, n_time_steps, n_nodes)
            The social trajectories of each node.

        Xs : np.ndarray of shape (size, n_time_steps, n_nodes, n_features)
            The latent trajectories of each node

        lambdas : np.ndarray of shape (size, n_layers, n_features)
            The homophily coefficients.
        """
        rng = check_random_state(random_state)

        n_layers, n_time_steps, n_nodes = self.delta_.shape

        deltas = np.zeros((size, n_layers, n_time_steps, n_nodes))
        if self.X_ is not None:
            Xs = np.zeros((size, n_time_steps, n_nodes, self.n_features))
            lambdas = np.zeros((size, n_layers, self.n_features))
        else:
            Xs = None
            lambdas = None

        for i in range(n_nodes):
            if self.X_ is not None:
                X_sampled = sample_gssm(
                    self.X_[:, i], self.X_sigma_[:, i],
                    self.X_cross_cov_[:, i], size=size, random_state=rng)
                if self.n_features_ == 1:
                    X_sampled = np.expand_dims(X_sampled, axis=-1)
                Xs[:, :, i, :] = X_sampled

            for k in range(n_layers):
                deltas[:, k, :, i] = sample_gssm(
                    self.delta_[k, :, i], self.delta_sigma_[k, :, i],
                    self.delta_cross_cov_[k, :, i],
                    size=size, random_state=rng)

        if self.X_ is not None:
            lambdas[:, 0, :] = (
                2 * rng.binomial(1, p=self.lambda_proba_,
                                 size=(size, self.n_features)) - 1)
            for k in range(1, n_layers):
                lambdas[:, k, :] = rng.multivariate_normal(
                    mean=self.lambda_[k], cov=self.lambda_sigma_[k],
                    size=size)

        return deltas, Xs, lambdas
