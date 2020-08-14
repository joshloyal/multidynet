# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as np

from libc.math cimport pow

from .omega cimport update_omega_svi


ctypedef np.int64_t INT


class NodeParams(object):
    def __init__(self, X, X_sigma, X_cross_cov, X_eta1, X_eta2,
                 delta, delta_sigma, delta_eta1, delta_eta2):
        self.X = X
        self.X_sigma = X_sigma
        self.X_cross_cov = X_cross_cov
        self.X_eta1 = X_eta1
        self.X_eta2 = X_eta2
        self.delta = delta
        self.delta_sigma = delta_sigma
        self.delta_eta1 = delta_eta1
        self.delta_eta2 = delta_eta2


def sample_omegas(double[:, :, ::1] X,
                  double[:, :, :, ::1] X_sigma,
                  double[:, ::1] lmbda,
                  double[:, :, ::1] lmbda_sigma,
                  double[:, ::1] delta,
                  double[:, ::1] delta_sigma,
                  INT[:, :, ::1] pos_subsamples,
                  INT[:, :, ::1] neg_subsamples,
                  int i):
    cdef size_t t, k, l
    cdef int j
    cdef size_t n_time_steps = X.shape[0]
    cdef size_t n_layers = lmbda.shape[0]
    cdef size_t n_pos_subsamples = pos_subsamples.shape[2]
    cdef size_t n_neg_subsamples = neg_subsamples.shape[2]

    cdef np.ndarray[double, ndim=3, mode='c'] omega_pos = np.zeros(
        (n_layers, n_time_steps, n_pos_subsamples), np.float64)
    cdef np.ndarray[double, ndim=3, mode='c'] omega_neg = np.zeros(
        (n_layers, n_time_steps, n_neg_subsamples), np.float64)

    for k in range(n_layers):
        for t in range(n_time_steps):

            # loop over positive dyads
            for l in range(n_pos_subsamples):
                j = pos_subsamples[k, t, l]
                if j < 0:
                    break

                omega_pos[k, t, l] = update_omega_svi(
                    X[t, i], X_sigma[t, i], X[t, j], X_sigma[t, j],
                    lmbda[k], lmbda_sigma[k], delta[k, i], delta[k, j],
                    delta_sigma[k, i], delta_sigma[k, j])

            # loop over positive dyads
            for l in range(n_neg_subsamples):
                j = neg_subsamples[k, t, l]
                if j < 0:
                    break

                omega_neg[k, t, l] = update_omega_svi(
                    X[t, i], X_sigma[t, i], X[t, j], X_sigma[t, j],
                    lmbda[k], lmbda_sigma[k], delta[k, i], delta[k, j],
                    delta_sigma[k, i], delta_sigma[k, j])

    return omega_pos, omega_neg


def calculate_natural_parameters_lds(const double[:, :, :, ::1] Y,
                                     double[:, :, ::1] X,
                                     double[:, :, :, ::1] X_sigma,
                                     double[:, ::1] lmbda,
                                     double[:, :, ::1] lmbda_sigma,
                                     double[:, ::1] delta,
                                     double[:, :, ::1] omega_pos,
                                     double[:, :, ::1] omega_neg,
                                     INT[:, :, ::1] pos_subsamples,
                                     INT[:, :, ::1] neg_subsamples,
                                     double[:, ::1] pos_ratios,
                                     double[:, ::1] neg_ratios,
                                     int i):
    cdef size_t t, k, l, p, q
    cdef int j
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = Y.shape[2]
    cdef size_t n_features = lmbda.shape[1]
    cdef size_t n_pos_subsamples = pos_subsamples.shape[2]
    cdef size_t n_neg_subsamples = neg_subsamples.shape[2]
    cdef double ratio

    cdef double[:, ::1] eta1 = np.zeros(
        (n_time_steps, n_features), dtype=np.float64)
    cdef double[:, :, ::1] eta2 = np.zeros(
        (n_time_steps, n_features, n_features), dtype=np.float64)

    # sample nodes
    for t in range(n_time_steps):
        for k in range(n_layers):
            # loop over positive dyads
            ratio = pos_ratios[k, t]
            for l in range(n_pos_subsamples):
                # extract subample and calculate omega
                j = pos_subsamples[k, t, l]
                if j < 0:
                    break

                for p in range(n_features):
                    eta1[t, p] += ratio * (
                        lmbda[k, p] * X[t, j, p] * (
                            Y[k, t, i, j] - 0.5 -
                                omega_pos[k, t, l] * (delta[k, i] + delta[k, j])))

                    for q in range(p + 1):
                        eta2[t, p, q] += ratio * omega_pos[k, t, l] * (
                            (lmbda_sigma[k, p, q] +
                                lmbda[k, p] * lmbda[k, q]) *
                            (X_sigma[t, j, p, q] +
                                X[t, j, p] * X[t, j, q]))
                        eta2[t, q, p] = eta2[t, p, q]

            # loop over negative (i.e., zero) dyads
            ratio = neg_ratios[k, t]
            for l in range(n_neg_subsamples):
                # extract subample and calculate omega
                j = neg_subsamples[k, t, l]
                if j < 0:
                    break

                for p in range(n_features):
                    eta1[t, p] += ratio * (
                        lmbda[k, p] * X[t, j, p] * (
                            Y[k, t, i, j] - 0.5 -
                                omega_neg[k, t, l] * (delta[k, i] + delta[k, j])))

                    for q in range(p + 1):
                        eta2[t, p, q] += ratio * omega_neg[k, t, l] * (
                            (lmbda_sigma[k, p, q] +
                                lmbda[k, p] * lmbda[k, q]) *
                            (X_sigma[t, j, p, q] +
                                X[t, j, p] * X[t, j, q]))
                        eta2[t, q, p] = eta2[t, p, q]

    return np.asarray(eta1), np.asarray(eta2)


def kalman_filter(np.ndarray[double, ndim=2, mode='c'] A,
                  np.ndarray[double, ndim=3, mode='c'] B,
                  double tau_prec,
                  double sigma_prec):
    cdef size_t t
    cdef size_t n_time_steps = A.shape[0]
    cdef size_t n_features = A.shape[1]

    # allocate temporary arrays
    cdef np.ndarray[double, ndim=2, mode='c'] mu = np.zeros(
        (n_time_steps, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] sigma = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] sigma_inv = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] sigma_star = np.zeros(
        (n_time_steps - 1, n_features, n_features))
    cdef np.ndarray[double, ndim=2, mode='c'] F_init_inv = (
        tau_prec * np.eye(n_features))
    cdef np.ndarray[double, ndim=2, mode='c'] F_inv = (
        sigma_prec * np.eye(n_features))

    # t = 1
    sigma_inv[0] = F_init_inv + B[0]
    sigma[0] = np.linalg.pinv(sigma_inv[0])
    mu[0] = np.dot(sigma[0], A[0])

    for t in range(1, n_time_steps):
        sigma_star[t-1] = np.linalg.pinv(F_inv + sigma_inv[t-1])
        sigma_inv[t] = F_inv + B[t] - (sigma_prec ** 2) * sigma_star[t-1]
        sigma[t] = np.linalg.pinv(sigma_inv[t])
        mu[t] = np.dot(sigma[t], A[t] +
            sigma_prec * np.dot(
                sigma_star[t-1], np.dot(sigma_inv[t-1],  mu[t-1])))

    return mu, sigma, sigma_inv, sigma_star


def kalman_smoother(np.ndarray[double, ndim=2, mode='c'] A,
                    np.ndarray[double, ndim=3, mode='c'] B,
                    double tau_prec,
                    double sigma_prec):
    cdef size_t t
    cdef size_t n_time_steps = A.shape[0]
    cdef size_t n_features = A.shape[1]

    # Allocate temporary arrays
    cdef np.ndarray[double, ndim=2, mode='c'] mu
    cdef np.ndarray[double, ndim=3, mode='c'] sigma
    cdef np.ndarray[double, ndim=3, mode='c'] sigma_inv
    cdef np.ndarray[double, ndim=3, mode='c'] sigma_star
    cdef np.ndarray[double, ndim=2, mode='c'] eta = np.zeros(
        (n_time_steps, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] psi = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] psi_inv = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] psi_star = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=2, mode='c'] F_init_inv = (
        tau_prec * np.eye(n_features))
    cdef np.ndarray[double, ndim=2, mode='c'] F_inv = (
        sigma_prec * np.eye(n_features))
    cdef np.ndarray[double, ndim=2, mode='c'] mean = np.zeros(
        (n_time_steps, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] cov = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] cross_cov = np.zeros(
        (n_time_steps - 1, n_features, n_features))

    # run the filter for the forward message variables
    mu, sigma, sigma_inv, sigma_star = kalman_filter(A, B, tau_prec, sigma_prec)

    # run the smoother
    mean[n_time_steps - 1] = mu[n_time_steps - 1]
    cov[n_time_steps - 1] = sigma[n_time_steps - 1]
    for t in range(n_time_steps - 1, 0, -1):
        psi_star[t] = np.linalg.pinv(F_inv + B[t] + psi_inv[t])
        psi_inv[t-1] = F_inv - (sigma_prec ** 2) * psi_star[t]
        psi[t-1] = np.linalg.pinv(psi_inv[t-1])
        eta[t-1] = sigma_prec * np.dot(psi[t-1], np.dot(psi_star[t],
            A[t] + np.dot(psi_inv[t], eta[t])))

        # update marginals and cross-covariances
        cov[t-1] = np.linalg.pinv(sigma_inv[t-1] + psi_inv[t-1])
        mean[t-1] = np.dot(cov[t-1],
            np.dot(sigma_inv[t-1], mu[t-1]) +
                np.dot(psi_inv[t-1], eta[t-1]))
        cross_cov[t-1] = sigma_prec * np.dot(sigma_star[t-1],
            np.linalg.pinv(
                F_inv + B[t] + psi_inv[t] - # psi_star^{-1}
                    (sigma_prec ** 2) * sigma_star[t-1]))

    return mean, cov, cross_cov


def calculate_natural_parameters_delta(const double[:, :, :, ::1] Y,
                                       double[:, :, ::1] X,
                                       double[:, ::1] lmbda,
                                       double[:, ::1] delta,
                                       double[:, :, ::1] omega_pos,
                                       double[:, :, ::1] omega_neg,
                                       double delta_var_prior,
                                       INT[:, :, ::1] pos_subsamples,
                                       INT[:, :, ::1] neg_subsamples,
                                       double[:, ::1] pos_ratios,
                                       double[:, ::1] neg_ratios,
                                       int k,
                                       int i):
    cdef size_t t, p
    cdef int j
    cdef size_t n_time_steps = X.shape[0]
    cdef size_t n_nodes = X.shape[1]
    cdef size_t n_features = X.shape[2]
    cdef size_t n_pos_subsamples = pos_subsamples.shape[2]
    cdef size_t n_neg_subsamples = neg_subsamples.shape[2]

    cdef double tmp = 0.
    cdef double eta1 = 0.
    cdef double eta2 = 1. / delta_var_prior
    cdef double ratio

    for t in range(n_time_steps):
        # positive dyads
        for l in range(n_pos_subsamples):
            j = pos_subsamples[k, t, l]
            if j < 0:
                break

            ratio = pos_ratios[k, t]

            eta2 += ratio * omega_pos[k, t, l]

            tmp = delta[k, j]
            for p in range(n_features):
                tmp += lmbda[k, p] * X[t, i, p] * X[t, j, p]
            eta1 += ratio * (Y[k, t, i, j] - 0.5 - omega_pos[k, t, l] * tmp)

        # negative dyads
        for l in range(n_neg_subsamples):
            j = neg_subsamples[k, t, l]
            if j < 0:
                break

            ratio = neg_ratios[k, t]

            eta2 += ratio * omega_pos[k, t, l]

            tmp = delta[k, j]
            for p in range(n_features):
                tmp += lmbda[k, p] * X[t, i, p] * X[t, j, p]
            eta1 += ratio * (Y[k, t, i, j] - 0.5 - omega_neg[k, t, l] * tmp)

    return eta1, eta2


def subsample_nodes(np.ndarray[double, ndim=4, mode='c'] Y,
                    int max_degree,
                    int n_neg_multiplier,
                    int i,
                    object rng):
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = Y.shape[2]
    cdef size_t k, t
    cdef size_t n_pos_samples, n_neg_samples, n_neg_subsamples

    cdef np.ndarray[double, ndim=1, mode='c'] y_row
    cdef np.ndarray[INT, ndim=1, mode='c'] indices
    cdef np.ndarray[INT, ndim=3, mode='c'] pos_subsamples = np.full(
        (n_layers, n_time_steps, max_degree), -1, dtype=np.int64)
    cdef np.ndarray[double, ndim=2, mode='c'] pos_ratios = np.empty(
        (n_layers, n_time_steps), dtype=np.float64)
    cdef np.ndarray[INT, ndim=3, mode='c'] neg_subsamples = np.full(
        (n_layers, n_time_steps, n_neg_multiplier * max_degree), -1, dtype=np.int64)
    cdef np.ndarray[double, ndim=2, mode='c'] neg_ratios = np.empty(
        (n_layers, n_time_steps), dtype=np.float64)

    for k in range(n_layers):
        for t in range(n_time_steps):
            y_row = Y[k, t, i].copy()

            # XXX: positive / negative dyads could be cached
            # include all edges (positive dyads)
            indices = np.where(y_row == 1)[0]
            n_pos_samples = indices.shape[0]
            if n_pos_samples > 0:
                #indices = rng.choice(indices, size=n_pos_subsamples)
                pos_subsamples[k, t, :n_pos_samples] = indices
                pos_ratios[k, t] = n_pos_samples / (<double> indices.shape[0])

            # XXX: postive / negative dyads could be cached
            # sample zero dyads (negative connections)
            y_row[i] = 1  # so we do not sample the diagonal
            indices = np.where(y_row == 0)[0]
            n_neg_samples = indices.shape[0]
            if n_neg_samples:
                n_neg_subsamples = min(n_neg_samples, n_neg_multiplier * n_pos_samples)
                if n_neg_subsamples < n_neg_samples:
                    indices = rng.choice(indices, size=n_neg_subsamples)
                neg_subsamples[k, t, :indices.shape[0]] = indices
                neg_ratios[k, t] = n_neg_samples / (<double> indices.shape[0])

    return pos_subsamples, neg_subsamples, pos_ratios, neg_ratios


def update_node_effects(const double[:, :, :, ::1] Y,
                        np.ndarray[double, ndim=3, mode='c'] X,
                        np.ndarray[double, ndim=4, mode='c'] X_sigma,
                        np.ndarray[double, ndim=3, mode='c'] X_eta1,
                        np.ndarray[double, ndim=4, mode='c'] X_eta2,
                        double[:, ::1] lmbda,
                        double[:, :, ::1] lmbda_sigma,
                        double[:, ::1] delta,
                        double[:, ::1] delta_sigma,
                        double[:, ::1] delta_eta1,
                        double[:, ::1] delta_eta2,
                        double delta_var_prior,
                        double tau_prec,
                        double sigma_prec,
                        int max_degree,
                        int n_neg_multiplier,
                        double rho,
                        object rng):
    cdef size_t i, k
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_time_steps = X.shape[0]
    cdef size_t n_nodes = X.shape[1]
    cdef size_t n_features = X.shape[2]

    # local variables
    cdef np.ndarray[INT, ndim=3, mode='c'] pos_subsamples
    cdef np.ndarray[INT, ndim=3, mode='c'] neg_subsamples
    cdef np.ndarray[double, ndim=2, mode='c'] pos_ratios
    cdef np.ndarray[double, ndim=2, mode='c'] neg_ratios
    cdef np.ndarray[double, ndim=3, mode='c'] omega_pos
    cdef np.ndarray[double, ndim=3, mode='c'] omega_neg

    # updated natural parameters (lds)
    cdef np.ndarray[double, ndim=2, mode='c'] lds_eta1
    cdef np.ndarray[double, ndim=3, mode='c'] lds_eta2
    cdef np.ndarray[double, ndim=3, mode='c'] X_eta1_new = np.zeros(
        (n_nodes, n_time_steps, n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=4, mode='c'] X_eta2_new = np.zeros(
        (n_nodes, n_time_steps, n_features, n_features), dtype=np.float64)

    # updated natural parameters (sociability)
    cdef double eta1
    cdef double eta2
    cdef double[:, ::1] delta_eta1_new = np.zeros(
        (n_layers, n_nodes), dtype=np.float64)
    cdef double[:, ::1] delta_eta2_new = np.zeros(
        (n_layers, n_nodes), dtype=np.float64)

    # updated moments (lds)
    cdef np.ndarray[double, ndim=3, mode='c'] X_new = np.zeros(
        (n_time_steps, n_nodes, n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=4, mode='c'] X_sigma_new = np.zeros(
        (n_time_steps, n_nodes, n_features, n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=4, mode='c'] X_cross_cov_new = np.zeros(
        (n_time_steps - 1, n_nodes, n_features, n_features), dtype=np.float64)

    # updated moments (sociability)
    cdef double[:, ::1] delta_new = np.zeros(
        (n_layers, n_nodes), dtype=np.float64)
    cdef double[:, ::1] delta_sigma_new = np.zeros(
        (n_layers, n_nodes), dtype=np.float64)

    for i in range(n_nodes):
        # XXX: this may be slow?
        # determine subsamples
        pos_subsamples, neg_subsamples, pos_ratios, neg_ratios = (
            subsample_nodes(
                np.asarray(Y), max_degree, n_neg_multiplier, i, rng))

        # sample local variables
        omega_pos, omega_neg = sample_omegas(
            X, X_sigma, lmbda, lmbda_sigma, delta, delta_sigma,
            pos_subsamples, neg_subsamples, i)

        # calculate natural parameters on this subsample
        lds_eta1, lds_eta2 = calculate_natural_parameters_lds(
            Y, X, X_sigma, lmbda, lmbda_sigma, delta, omega_pos, omega_neg,
            pos_subsamples, neg_subsamples, pos_ratios, neg_ratios, i)

        # take a step along the gradient
        X_eta1_new[i] = (1 - rho) * X_eta1[i] + rho * lds_eta1
        X_eta2_new[i] = (1 - rho) * X_eta2[i] + rho * lds_eta2

        # update latent position expectations
        X_new[:, i], X_sigma_new[:, i], X_cross_cov_new[:, i] = kalman_smoother(
            X_eta1_new[i], X_eta2_new[i], tau_prec, sigma_prec)

        # update sociability parameters
        for k in range(n_layers):
            eta1, eta2 = calculate_natural_parameters_delta(
                Y, X, lmbda, delta, omega_pos, omega_neg,
                delta_var_prior, pos_subsamples, neg_subsamples,
                pos_ratios, neg_ratios, k, i)

            delta_eta1_new[k, i] = (1 - rho) * delta_eta1[k, i] + rho * eta1
            delta_eta2_new[k, i] = (1 - rho) * delta_eta2[k, i] + rho * eta2

            delta_sigma_new[k, i] = 1. / delta_eta2_new[k, i]
            delta_new[k, i] = delta_sigma_new[k, i] * delta_eta1_new[k, i]

    return NodeParams(X=X_new, X_sigma=X_sigma_new, X_cross_cov=X_cross_cov_new,
                      X_eta1=X_eta1_new, X_eta2=X_eta2_new,
                      delta=delta_new, delta_sigma=delta_sigma_new,
                      delta_eta1=delta_eta1_new, delta_eta2=delta_eta2_new)
