import warnings

import numpy as np
import scipy.sparse as sp

from joblib import Parallel, delayed
from scipy.special import logit, gammainc, expit, logsumexp
from scipy.linalg import orthogonal_procrustes
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, f1_score
from tqdm import tqdm

from .omega import update_omega
from .lds import update_latent_positions, update_latent_positions_MF
from .deltas_lds import update_deltas, update_deltas_MF
from .lmbdas import update_lambdas
from .variances import update_tau_sq, update_sigma_sq, update_X0_precision, update_diag_tau_sq
from .variances import update_tau_sq_delta, update_sigma_sq_delta
from .log_likelihood import log_likelihood, pointwise_log_likelihood
from .metrics import calculate_auc, calculate_metric
from .sample_lds import sample_gssm
from .model_selection import dynamic_multilayer_adjacency_to_vec


__all__ = ['DynamicMultilayerNetworkLSM']


EPS = np.finfo('float64').epsneg


class ModelParameters(object):
    def __init__(self, omega, X, X_sigma, X_cross_cov,
                 lmbda, lmbda_sigma, lmbda_logit_prior,
                 delta, delta_sigma, delta_cross_cov,
                 X0_cov_df, X0_cov_scale,
                 a_tau_sq, b_tau_sq, c_sigma_sq, d_sigma_sq,
                 a_tau_sq_delta, b_tau_sq_delta, c_sigma_sq_delta,
                 d_sigma_sq_delta, callback):
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
        self.X0_cov_df_ = X0_cov_df
        self.X0_cov_scale_ = X0_cov_scale
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
        self.criteria_ = []
        self.callback_ = callback


def find_permutation(U, U_ref):
    _, n_features = U.shape
    C = U_ref.T @ U
    perm = linear_sum_assignment(np.maximum(C, -C), maximize=True)[1]
    sign = np.sign(C[np.arange(n_features), perm])
    return sign * U[:, perm]


def smooth_positions(U):
    n_time_steps, _, _ = U.shape
    for t in range(1, n_time_steps):
        U[t] = find_permutation(U[t], U[t-1])

    return U


def smooth_positions_procrustes(U):
    n_time_steps, _, _ = U.shape
    for t in range(1, n_time_steps):
        R, _ = orthogonal_procrustes(U[t], U[t-1])
        U[t] = U[t] @ R

    return U



def sample_socialities(model, size=500, random_state=None):
    samples = model.sample(size=size, random_state=random_state)
    deltas = samples['delta']
    Xs = samples['X']
    lambdas = samples['lambda']

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


def initialize_node_effects(Y):
    n_layers, n_time_steps, n_nodes, _ = Y.shape

    delta = np.zeros((n_layers, n_time_steps, n_nodes))
    for k in range(n_layers):
        for t in range(n_time_steps):
            delta[k, t] = initialize_node_effects_single(Y[k, t])

    return delta

def initialize_lambda(Y, U):
    n_time_steps, n_nodes, n_features = U.shape
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    dyads = np.tril_indices_from(Y[0], k=-1)
    
    X, y_vec = None, None
    for t in range(n_time_steps):
        y_vec_t = Y[t][dyads]
    
        X_t = np.zeros((n_dyads, n_features))
        for p in range(n_features):
            u = U[t, :, p].reshape(-1, 1)
            X_t[:, p] = (u @ u.T)[dyads]
        
        if X is None:
            X = X_t
            y_vec = y_vec_t
        else:
            X = np.vstack((X, X_t))
            y_vec = np.r_[y_vec, y_vec_t]
    
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y_vec)

    return reg.coef_ 


def initialize_svt(Y, n_features):
    n_layers, n_time_steps, n_nodes, _ = Y.shape

    delta_init = np.zeros((n_layers, n_time_steps, n_nodes))
    resid = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))
    lmbda_init = np.zeros((n_layers, n_features))
    V = [None] * n_time_steps
    for t in range(n_time_steps):
        for k in range(n_layers):
            A = Y[k, t].copy()
            A[A == -1] = 0
            #dyads = np.tril_indices_from(A, k=-1)
            #tau = np.sqrt(n_nodes * np.mean(A[dyads]))
            tau = np.sqrt(n_nodes * np.mean(A))
            u,s,v = np.linalg.svd(A, hermitian=True)
            ids = s >= tau
            P_tilde = np.clip(u[:, ids] @ np.diag(s[ids]) @ v[ids, :], EPS, 1-EPS)
            Theta = logit(0.5 * (P_tilde + P_tilde.T))

            # diagonal is undefined (e.g., missing)
            #Theta[np.diag_indices_from(Theta)] = 0.  

            delta_init[k, t] = initialize_node_effects_cont(Theta)
            if n_features > 0:
                d = delta_init[k, t].reshape(-1, 1)
                resid[k, t] = Theta - d - d.T
                
                # center columns for identifiability: J @ resid @ J
                J = np.eye(n_nodes) - (1/n_nodes) * np.ones((n_nodes, n_nodes))
                resid[k, t] = J @ resid[k, t] @ J 
                if V[t] is None:
                    V[t] = resid[k, t]
                else:
                    V[t] = np.hstack((V[t], resid[k, t])) 
                
                #eigvals, eigvecs = np.linalg.eigh(resid[k, t])
                #ids = np.argsort(np.abs(eigvals))[::-1]
                #eigvecs = eigvecs[:, ids][:, :n_features] 
                #if V[t] is None:
                #    V[t] = eigvecs
                #else:
                #    V[t] = np.hstack((V[t], eigvecs)) 
    
    if n_features > 0:
        X = np.zeros((n_time_steps, n_nodes, n_features))
        for t in range(n_time_steps):
            u, s, v = sp.linalg.svds(V[t], k=n_features)
            X[t] = u 
        X = smooth_positions_procrustes(X)
        
        for k in range(n_layers):
            lmbda_init[k] = initialize_lambda(resid[k], X)
        lambda0 = np.abs(lmbda_init[0])
        lmbda_init = lmbda_init / lambda0
        for t in range(n_time_steps):
            X[t] = np.sqrt(lambda0) * X[t]
    else:
        X = None
        lmbda_init = None
    
    return X, lmbda_init, delta_init


def initialize_parameters(Y, n_features, init_params_type,
                          lambda_odds_prior, lambda_var_prior,
                          #a, b, c, d, a_delta, b_delta, c_delta, d_delta,
                          init_covariance_type, c, d,
                          a_delta, b_delta, c_delta, d_delta,
                          approx_type, callback, random_state):
    rng = check_random_state(random_state)

    n_layers, n_time_steps, n_nodes, _ = Y.shape

    # omega is initialized by drawing from the prior?
    omega = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))

    # initialize latent space randomly and center to remove effect
    # social trajectory initialization
    if n_features > 0:
        # initialize latent space to something smooth over time.
        #X = rng.randn(n_time_steps, n_nodes, n_features)
        #for t in range(n_time_steps):
        #    X[t] -= np.mean(X[t], axis=0)
        if init_params_type == 'svt':
            X, lmbda, delta = initialize_svt(Y, n_features)
        else:
            X = np.zeros((n_time_steps, n_nodes, n_features))
            for t in range(n_time_steps):
                if t > 0:
                    X[t] = X[t-1] + 0.1 * rng.randn(n_nodes, n_features)
                else:
                    X[t] = rng.randn(n_nodes, n_features)
                X[t] -= np.mean(X[t], axis=0)

        # initialize to marginal covariances
        sigma_init = np.eye(n_features)
        X_sigma = np.tile(
            sigma_init[None, None], reps=(n_time_steps, n_nodes, 1, 1))

        # initialize cross-covariances
        if approx_type == 'structured':
            cross_init = np.eye(n_features)
        else:
            cross_init = np.zeros((n_features, n_features))
        X_cross_cov = np.tile(
            cross_init[None, None], reps=(n_time_steps - 1, n_nodes, 1, 1))
        
        if init_params_type != 'svt':
            # initialize to prior means
            lmbda = np.sqrt(2) * rng.randn(n_layers, n_features)
            lmbda[0] = rng.choice([-1, 1], n_features)

        lmbda_sigma = np.ones(
            (n_layers, n_features, n_features))
        lmbda_sigma[0] = (
            (1 - lmbda[0, 0] ** 2) * np.eye(n_features))
        lmbda_logit_prior = np.log(lambda_odds_prior)
    else:
        X, lmbda, delta = initialize_svt(Y, n_features)
        X_sigma = None
        X_cross_cov = None
        lmbda_sigma = None
        lmbda_logit_prior = np.log(lambda_odds_prior)

    
    if init_params_type != 'svt' and n_features > 0:
        #_, _, delta = initialize_svt(Y, 0)
        #delta = initialize_node_effects(Y)
        delta = rng.randn(n_layers, n_time_steps, n_nodes)

    delta_sigma = np.ones((n_layers, n_time_steps, n_nodes))

    if approx_type == 'structured':
        delta_cross_cov = np.ones((n_layers, n_time_steps - 1, n_nodes))
    else:
        delta_cross_cov = np.zeros((n_layers, n_time_steps - 1, n_nodes))

    # initialize based on prior information
    X0_cov_scale = np.eye(n_features)
    X0_cov_df = n_features + 2
    a_tau_sq = np.full(n_features, 1.5) # 0.5 * (n_features + 2)
    b_tau_sq = np.full(n_features, 0.5)
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
        X0_cov_scale=X0_cov_scale, X0_cov_df=X0_cov_df,
        a_tau_sq=a_tau_sq, b_tau_sq=b_tau_sq, c_sigma_sq=c_sigma_sq,
        d_sigma_sq=d_sigma_sq, a_tau_sq_delta=a_delta, b_tau_sq_delta=b_delta,
        c_sigma_sq_delta=c_sigma_sq_delta, d_sigma_sq_delta=d_sigma_sq_delta,
        callback=callback)



def optimize_elbo(Y, n_features, lambda_odds_prior, lambda_var_prior,
                  init_covariance_type, init_params_type, c, d,
                  a_delta, b_delta, c_delta, d_delta,
                  approx_type, max_iter,  tol, random_state,
                  stopping_criteria='loglik',
                  callback=None, verbose=True, idx=0):

    n_layers, n_time_steps, n_nodes, _ = Y.shape

    # convergence criteria:
    #   loglik: Eq{L(Y | theta)})
    #   auc: training AUC
    criteria = -np.infty
    n_nochange = 0

    if init_params_type == 'both' and idx == 0:
        init_params_type = 'svt'

    # initialize parameters of the model
    model = initialize_parameters(
        Y, n_features, init_params_type, lambda_odds_prior, lambda_var_prior,
        init_covariance_type, c, d, a_delta, b_delta, c_delta, d_delta,
        approx_type, callback, random_state)

    a = np.full(n_features, 1.5)  # 0.5 * (n_features + 2)
    b = np.full(n_features, 0.5)
    X0_cov_prior_df = n_features + 2
    X0_cov_prior_scale = np.eye(n_features)
    XLX = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))

    if model.callback_ is not None:
        model.callback_.tick()

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

        sigma_sq_prec = model.c_sigma_sq_ / model.d_sigma_sq_

        if n_features > 0:
            if approx_type == 'structured':
                update_latent_positions(
                    Y, model.X_, model.X_sigma_, model.X_cross_cov_,
                    model.lambda_, model.lambda_sigma_, model.delta_,
                    model.omega_, X0_cov_prec, sigma_sq_prec)
            else:
                update_latent_positions_MF(
                    Y, model.X_, model.X_sigma_, model.X_cross_cov_,
                    model.lambda_, model.lambda_sigma_, model.delta_,
                    model.omega_, X0_cov_prec, sigma_sq_prec)

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
        
        if approx_type == 'structured':
            update_deltas(
                Y, model.delta_, model.delta_sigma_, model.delta_cross_cov_,
                XLX, model.omega_, tau_sq_prec, sigma_sq_prec)
        else:
            update_deltas_MF(
                Y, model.delta_, model.delta_sigma_, model.delta_cross_cov_,
                XLX, model.omega_, tau_sq_prec, sigma_sq_prec)

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
        if stopping_criteria is not None:
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
    

    probas = calculate_probabilities(
        model.X_, model.lambda_, model.delta_)
    model.auc_ = calculate_auc(Y, probas)
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


def calculate_single_proba(X, lmbda, delta):
    n_layers, n_nodes = delta.shape
    probas = np.zeros((n_layers, n_nodes, n_nodes))
    for k in range(n_layers):
        deltak = delta[k].reshape(-1, 1)
        eta = np.add(deltak, deltak.T)
        if X is not None:
            eta += np.dot(X * lmbda[k], X.T)
        probas[k] = expit(eta)

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
                 init_covariance_type='full',
                 c=2., d=2.,
                 a_delta=4.1, b_delta=2.1 * 10, c_delta=2., d_delta=2.,
                 init_type='svt', n_init=1, max_iter=1000, tol=1e-2,
                 stopping_criteria='loglik',
                 approx_type='structured',
                 n_jobs=1, random_state=None):
        self.n_features = n_features
        self.lambda_odds_prior = lambda_odds_prior
        self.lambda_var_prior = lambda_var_prior
        self.init_covariance_type = init_covariance_type
        self.c = c
        self.d = d
        self.a_delta = a_delta
        self.b_delta = b_delta
        self.c_delta = c_delta
        self.d_delta = d_delta
        self.init_type = init_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.stopping_criteria = stopping_criteria
        self.approx_type = approx_type
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, Y, callback=None, n_samples=None, verbose=True):
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
                        ensure_2d=False, allow_nd=True, copy=True)

        if Y.ndim == 3:
            raise ValueError(
                "Y.ndim == {}, when it should be 4. "
                "If there is only a single layer, then reshape Y with "
                "Y = np.expand_dims(Y, axis=0) and re-fit.".format(Y.ndim))

        self.n_features_ = self.n_features if self.n_features is not None else 0

        random_state = check_random_state(self.random_state)

        if self.init_type == 'svt':
            self.n_init = 1

        # run the elbo optimization over different initializations
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        verbose = verbose if self.n_init == 1 else False
        models = Parallel(n_jobs=self.n_jobs)(delayed(optimize_elbo)(
                Y, self.n_features_, self.lambda_odds_prior,
                self.lambda_var_prior, self.init_covariance_type,
                self.init_type,
                self.c, self.d,
                self.a_delta, self.b_delta, self.c_delta, self.d_delta,
                self.approx_type, self.max_iter, self.tol, seed,
                stopping_criteria=self.stopping_criteria,
                callback=callback, verbose=verbose, idx=i)
            for i, seed in enumerate(seeds))

        # choose model with the largest in-sample AUC #convergence criteria
        #best_model = models[0]
        #best_criteria = models[0].criteria_[-1]
        #for i in range(1, len(models)):
        #    if models[i].criteria_[-1] > best_criteria:
        #        best_model = models[i]
        #        best_criteria = models[i].criteria_[-1]

        best_idx = np.argmax([model.auc_ for model in models])
        #best_idx = np.argmax([model.criteria_[-1] for model in models])
        best_model = models[best_idx]
        #best_model = models[0]
        #best_criteria = models[0].auc_
        #for i in range(1, len(models)):
        #    if models[i].auc_ > best_criteria:
        #        best_model = models[i]
        #        best_criteria = models[i].auc_

        if not best_model.converged_:
            warnings.warn('Best model did not converge. '
                          'Try a different random initialization, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.', ConvergenceWarning)

        self._set_parameters(best_model)

        # save all callbacks
        self._callbacks = [models[i].callback_ for i in range(len(models))]

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

        # BIC
        n_layers, n_time_steps, n_nodes = self.delta_.shape
        logn = (np.log(0.5 * n_nodes) + np.log(n_nodes - 1) +
            np.log(n_layers) + np.log(n_time_steps))
        self.p_bic_ = np.prod(self.delta_.shape)
        if self.n_features_ > 0:
            self.p_bic_ += np.prod(self.X_.shape)
            self.p_bic_ += np.prod(self.lambda_.shape)


        self.aic_ = -2 * self.loglik_ + 2 * self.p_aic_
        self.bic_ = -2 * self.loglik_ + logn * self.p_bic_

        if n_samples is not None:
            self.samples_ = self.sample(
                n_samples, random_state=self.random_state)

        return self

    def waic(self, Y):
        Y = check_array(Y, order='C', dtype=np.float64,
                        ensure_2d=False, allow_nd=True, copy=True)
        n_layers, n_time_steps, n_nodes, _ = Y.shape

        if hasattr(self, 'samples_'):
            n_samples = self.samples_['delta'].shape[0]
            samples = self.samples_
        else:
            n_samples = 500
            samples = self.sample(size=n_samples)

        n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
        loglik = pointwise_log_likelihood(
            Y, samples['X'], samples['lambda'], samples['delta'],
            self.n_features_)
        loglik = loglik.reshape(-1, np.prod(loglik.shape[1:]))

        p_waic = loglik.var(axis=0).sum()
        lppd = (logsumexp(loglik, axis=0) - np.log(n_samples)).sum()
        waic = -2 * (lppd - p_waic)

        return waic, p_waic

    def dic(self, Y, n_samples=500):
        Y = check_array(Y, order='C', dtype=np.float64,
                        ensure_2d=False, allow_nd=True, copy=True)
        n_layers, n_time_steps, n_nodes, _ = Y.shape

        if hasattr(self, 'samples_'):
            n_samples = self.samples_['delta'].shape[0]
            samples = self.samples_
        else:
            n_samples = 500
            samples = self.sample(size=n_samples)

        loglik_bar = 0.
        for i in range(n_samples):
            loglik_bar += log_likelihood(Y,
                samples['X'][i], samples['lambda'][i], samples['delta'][i],
                self.n_features_) / n_samples

        p_dic = -2 * (loglik_bar - self.loglik_)
        dic =  -2 * self.loglik_ + 2 * p_dic

        return dic, p_dic


    def _set_parameters(self, model):
        self.omega_ = model.omega_

        self.X_ = model.X_
        self.X_sigma_ = model.X_sigma_
        self.X_cross_cov_ = model.X_cross_cov_

        # match signed-permutations across time
        if self.n_features > 0:
            self.X_ = smooth_positions(self.X_)

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


        if self.init_covariance_type == 'full':
            self.init_cov_df_ = model.X0_cov_df_
            self.init_cov_scale_ = model.X0_cov_scale_
            self.init_cov_ = (np.linalg.pinv(self.init_cov_scale_) /
                (self.init_cov_df_ - self.n_features - 1))
        else:
            self.a_tau_sq_ = model.a_tau_sq_
            self.b_tau_sq_ = model.b_tau_sq_
            self.init_cov_ = np.diag(self.b_tau_sq_ / (self.a_tau_sq_ - 1))

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
        self.criteria_ = model.criteria_
        self.callback_ = model.callback_
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

        return {'delta': deltas, 'X': Xs, 'lambda': lambdas}


    def loglikelihood(self, Y, test_indices=None):
        y = dynamic_multilayer_adjacency_to_vec(Y)
        n_layers, n_time_steps, n_dyads = y.shape

        subdiag = np.tril_indices(Y.shape[2], k=-1)
        loglik = 0.
        for k in range(n_layers):
            for t in range(n_time_steps):
                eta_test = self.probas_[k, t][subdiag]
                y_test = y[k, t]
                if test_indices is not None:
                    eta_test = eta_test[test_indices[k][t]]
                    y_test = y_test[test_indices[k][t]]

                loglik += -log_loss(y_test, eta_test)

        return loglik

    def forecast_probas(self, n_samples=1000, random_state=None):
        n_time_steps, n_nodes, n_features = self.X_.shape
        n_layers = self.delta_.shape[0]

        rng = check_random_state(random_state)
        probas = np.zeros((n_layers, n_nodes, n_nodes))
        for i in range(n_samples):
            X_new = (self.X_[-1] +
                np.sqrt(self.sigma_sq_) * rng.randn(n_nodes, n_features))
            delta_new = (self.delta_[:, -1, :] +
                np.sqrt(self.sigma_sq_delta_) *
                    rng.randn(n_layers, n_nodes))

            probas += (calculate_single_proba(X_new, self.lambda_, delta_new)
                / n_samples)

        return probas
