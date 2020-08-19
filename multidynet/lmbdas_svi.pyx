# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

from libc.math cimport sqrt, ceil
from scipy.special.cython_special import expit

import numpy as np
cimport numpy as np

from .omega cimport update_omega_svi


ctypedef np.int64_t INT


cdef struct Index:
    size_t i
    size_t j


cdef inline Index get_row_col(int idx):
    cdef Index index

    index.i =  <int> (ceil(sqrt(2 * (idx + 1) + 0.25) - 0.5))
    index.j = <int> (idx - 0.5 * index.i * (index.i - 1))

    return index


class LambdaParams(object):
    def __init__(self,
                 lmbda,
                 lmbda_sigma,
                 lmbda0_eta,
                 lmbda_eta1,
                 lmbda_eta2):
        self.lmbda = lmbda
        self.lmbda_sigma = lmbda_sigma
        self.lmbda0_eta = lmbda0_eta
        self.lmbda_eta1 = lmbda_eta1
        self.lmbda_eta2 = lmbda_eta2



def sample_omegas(double[:, :, ::1] X,
                  double[:, :, :, ::1] X_sigma,
                  double[:, ::1] lmbda,
                  double[:, :, ::1] lmbda_sigma,
                  double[:, ::1] delta,
                  double[:, ::1] delta_sigma,
                  INT[:, :, ::1] pos_subsamples,
                  INT[:, :, ::1] neg_subsamples):
    cdef size_t k, t, l, i, j
    cdef int idx
    cdef size_t n_time_steps = X.shape[0]
    cdef size_t n_layers = lmbda.shape[0]
    cdef size_t n_pos_subsamples = pos_subsamples.shape[2]
    cdef size_t n_neg_subsamples = neg_subsamples.shape[2]
    cdef Index index

    cdef np.ndarray[double, ndim=3, mode='c'] omega_pos = np.zeros(
        (n_layers, n_time_steps, n_pos_subsamples), np.float64)
    cdef np.ndarray[double, ndim=3, mode='c'] omega_neg = np.zeros(
        (n_layers, n_time_steps, n_neg_subsamples), np.float64)

    for k in range(n_layers):
        for t in range(n_time_steps):

            # loop over positive dyads
            for l in range(n_pos_subsamples):
                idx = pos_subsamples[k, t, l]
                if idx < 0:
                    break

                index = get_row_col(idx)
                i = index.i
                j = index.j

                omega_pos[k, t, l] = update_omega_svi(
                    X[t, i], X_sigma[t, i], X[t, j], X_sigma[t, j],
                    lmbda[k], lmbda_sigma[k], delta[k, i], delta[k, j],
                    delta_sigma[k, i], delta_sigma[k, j])

            # loop over positive dyads
            for l in range(n_neg_subsamples):
                idx = neg_subsamples[k, t, l]
                if idx < 0:
                    break

                index = get_row_col(idx)
                i = index.i
                j = index.j

                omega_neg[k, t, l] = update_omega_svi(
                    X[t, i], X_sigma[t, i], X[t, j], X_sigma[t, j],
                    lmbda[k], lmbda_sigma[k], delta[k, i], delta[k, j],
                    delta_sigma[k, i], delta_sigma[k, j])

    return omega_pos, omega_neg


cpdef double calculate_natural_parameters_reference(const double[:, :, :, ::1] Y,
                                                    double[:, :, ::1] X,
                                                    double[:, :, :, ::1] X_sigma,
                                                    double[:, :] delta,
                                                    double[:] lmbda,
                                                    double[:, :, ::1] omega_pos,
                                                    double[:, :, ::1] omega_neg,
                                                    INT[:, :, ::1] pos_subsamples,
                                                    INT[:, :, ::1] neg_subsamples,
                                                    double[:, ::1] pos_ratios,
                                                    double[:, ::1] neg_ratios,
                                                    int p):
    cdef size_t t, i, l, q
    cdef int idx
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = X.shape[1]
    cdef size_t n_features = X.shape[2]
    cdef size_t n_pos_subsamples = pos_subsamples.shape[2]
    cdef size_t n_neg_subsamples = neg_subsamples.shape[2]
    cdef double ratio
    cdef Index index
    cdef double eta = 0.

    for t in range(n_time_steps):
        # loop over positive dyads
        for l in range(n_pos_subsamples):
            idx = pos_subsamples[0, t, l]
            if idx < 0:
                break

            index = get_row_col(idx)
            i = index.i
            j = index.j
            ratio = pos_ratios[0, t]

            eta += ratio * (
                (Y[0, t, i, j] - 0.5 - omega_pos[0, t, l] * (
                    delta[0, i] + delta[0, j])) *
                    X[t, i, p] * X[t, j, p])

            for q in range(n_features):
                if q != p:
                    eta -= ratio * omega_pos[0, t, l] * lmbda[q] * (
                        (X_sigma[t, i, q, p] + X[t, i, q] * X[t, i, p]) *
                        (X_sigma[t, j, q, p] + X[t, j, q] * X[t, j, p]))

        # loop over negative dyads
        for l in range(n_neg_subsamples):
            idx = neg_subsamples[0, t, l]
            if idx < 0:
                break

            index = get_row_col(idx)
            i = index.i
            j = index.j
            ratio = neg_ratios[0, t]

            eta += ratio * (
                (Y[0, t, i, j] - 0.5 - omega_neg[0, t, l] * (
                    delta[0, i] + delta[0, j])) *
                    X[t, i, p] * X[t, j, p])

            for q in range(n_features):
                if q != p:
                    eta -= ratio * omega_neg[0, t, l] * lmbda[q] * (
                        (X_sigma[t, i, q, p] + X[t, i, q] * X[t, i, p]) *
                        (X_sigma[t, j, q, p] + X[t, j, q] * X[t, j, p]))

    return 2 * eta


def calculate_natural_parameters(const double[:, :, :, ::1] Y,
                                 double[:, :, ::1] X,
                                 double[:, :, :, ::1] X_sigma,
                                 double[:, :] delta,
                                 double lmbda_var_prior,
                                 double[:, :, ::1] omega_pos,
                                 double[:, :, ::1] omega_neg,
                                 INT[:, :, ::1] pos_subsamples,
                                 INT[:, :, ::1] neg_subsamples,
                                 double[:, ::1] pos_ratios,
                                 double[:, ::1] neg_ratios,
                                 int k):
    cdef size_t t, i, j, l, p, q
    cdef int idx
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = X.shape[1]
    cdef size_t n_features = X.shape[2]
    cdef size_t n_pos_subsamples = pos_subsamples.shape[2]
    cdef size_t n_neg_subsamples = neg_subsamples.shape[2]
    cdef double ratio
    cdef Index index

    cdef np.ndarray[double, ndim=1, mode='c'] eta1 = np.zeros(n_features)
    cdef np.ndarray[double, ndim=2, mode='c'] eta2 = np.zeros(
        (n_features, n_features))

    for t in range(n_time_steps):
        # loop over positive dyads
        for l in range(n_pos_subsamples):
            idx = pos_subsamples[k, t, l]
            if idx < 0:
                break

            index = get_row_col(idx)
            i = index.i
            j = index.j
            ratio = pos_ratios[k, t]

            for p in range(n_features):
                eta1[p] += ratio * (
                    (Y[k, t, i, j] - 0.5 -
                        omega_pos[k, t, l] * (
                            delta[k, i] + delta[k, j])) *
                    X[t, i, p] * X[t, j, p])

                for q in range(p + 1):
                    eta2[p, q] += ratio * omega_pos[k, t, l] * (
                        (X_sigma[t, i, p, q] + X[t, i, p] * X[t, i, q]) *
                        (X_sigma[t, j, p, q] + X[t, j, p] * X[t, j, q]))
                    eta2[q, p] = eta2[p, q]

        # loop over negative dyads
        for l in range(n_neg_subsamples):
            idx = neg_subsamples[k, t, l]
            if idx < 0:
                break

            index = get_row_col(idx)
            i = index.i
            j = index.j
            ratio = neg_ratios[k, t]

            for p in range(n_features):
                eta1[p] += ratio * (
                    (Y[k, t, i, j] - 0.5 -
                        omega_neg[k, t, l] * (
                            delta[k, i] + delta[k, j])) *
                    X[t, i, p] * X[t, j, p])

                for q in range(p + 1):
                    eta2[p, q] += ratio * omega_neg[k, t, l] * (
                        (X_sigma[t, i, p, q] + X[t, i, p] * X[t, i, q]) *
                        (X_sigma[t, j, p, q] + X[t, j, p] * X[t, j, q]))
                    eta2[q, p] = eta2[p, q]

    eta2[np.diag_indices_from(eta2)] += (1. / lmbda_var_prior)

    return eta1, eta2


def subsample_dyads(np.ndarray[double, ndim=4, mode='c'] Y,
                    int max_edges,
                    int n_neg_multiplier,
                    object rng):
    cdef size_t k, t
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = Y.shape[2]
    cdef size_t n_pos_samples, n_neg_samples, n_neg_subsamples

    cdef np.ndarray[double, ndim=1, mode='c'] y_vec
    cdef np.ndarray[np.int64_t, ndim=1, mode='c'] indices
    cdef np.ndarray[np.int64_t, ndim=3, mode='c'] pos_subsamples = np.full(
        (n_layers, n_time_steps, max_edges), -1, dtype=np.int64)
    cdef np.ndarray[double, ndim=2, mode='c'] pos_ratios = np.empty(
        (n_layers, n_time_steps), dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=3, mode='c'] neg_subsamples = np.full(
        (n_layers, n_time_steps, n_neg_multiplier * max_edges), -1, dtype=np.int64)
    cdef np.ndarray[double, ndim=2, mode='c'] neg_ratios = np.empty(
        (n_layers, n_time_steps), dtype=np.float64)

    for k in range(n_layers):
        for t in range(n_time_steps):
            y_vec = Y[k, t][np.tril_indices_from(Y[k, t], k=-1)]

            # XXX: positive dyads could be cached
            indices = np.where(y_vec == 1)[0]
            n_pos_samples = indices.shape[0]
            if n_pos_samples > 0:
                pos_subsamples[k, t, :indices.shape[0]] = indices
                pos_ratios[k, t] = n_pos_samples / (<double> indices.shape[0])

            # XXX: negative dyads could be cached
            indices = np.where(y_vec == 0)[0]
            n_neg_samples = indices.shape[0]
            if n_neg_samples > 0:
                n_neg_subsamples = min(n_neg_samples, n_neg_multiplier * n_pos_samples)
                if n_neg_subsamples < n_neg_samples:
                    indices = rng.choice(indices, size=n_neg_subsamples)
                neg_subsamples[k, t, :indices.shape[0]] = indices
                neg_ratios[k, t] = n_neg_samples / (<double> indices.shape[0])


    return pos_subsamples, neg_subsamples, pos_ratios, neg_ratios


def update_lambdas(const double[:, :, :, ::1] Y,
                   double[:, :, ::1] X,
                   double[:, :, :, ::1] X_sigma,
                   np.ndarray[double, ndim=2, mode='c'] lmbda,
                   np.ndarray[double, ndim=3, mode='c'] lmbda_sigma,
                   double[:] lmbda0_eta,
                   np.ndarray[double, ndim=2, mode='c'] lmbda_eta1,
                   np.ndarray[double, ndim=3, mode='c'] lmbda_eta2,
                   double[:, ::1] delta,
                   double[:, ::1] delta_sigma,
                   double lmbda_var_prior,
                   double lmbda_logit_prior,
                   int max_edges,
                   int n_neg_subsamples,
                   double rho,
                   object rng):
    cdef size_t k, p
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_features = lmbda.shape[1]

    cdef np.ndarray[np.int64_t, ndim=3, mode='c'] pos_subsamples
    cdef np.ndarray[np.int64_t, ndim=3, mode='c'] neg_subsamples
    cdef np.ndarray[double, ndim=2, mode='c'] pos_ratios
    cdef np.ndarray[double, ndim=2, mode='c'] neg_ratios
    cdef np.ndarray[double, ndim=3, mode='c'] omega_pos
    cdef np.ndarray[double, ndim=3, mode='c'] omega_neg

    cdef double eta, proba
    cdef np.ndarray[double, ndim=1, mode='c'] lmbda0_eta_new = np.zeros(
        n_features, dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode='c'] lmbda_new = np.zeros(
        (n_layers, n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode='c'] lmbda_sigma_new = np.zeros(
        (n_layers, n_features, n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode='c'] eta1
    cdef np.ndarray[double, ndim=2, mode='c'] eta2
    cdef np.ndarray[double, ndim=2, mode='c'] lmbda_eta1_new = np.zeros(
        (n_layers, n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode='c'] lmbda_eta2_new = np.zeros(
        (n_layers, n_features, n_features), dtype=np.float64)

    # sample dyads
    # XXX: this may be slow?
    pos_subsamples, neg_subsamples, pos_ratios, neg_ratios = (
        subsample_dyads(np.asarray(Y), max_edges, n_neg_subsamples, rng))

    # sample local variables
    omega_pos, omega_neg = sample_omegas(
        X, X_sigma, lmbda, lmbda_sigma, delta, delta_sigma,
        pos_subsamples, neg_subsamples)

    # start by updating signs of the reference layer
    for p in range(n_features):
        eta = calculate_natural_parameters_reference(
            Y, X, X_sigma, delta, lmbda[0], omega_pos, omega_neg,
            pos_subsamples, neg_subsamples, pos_ratios, neg_ratios, p)

        lmbda0_eta_new[p] = (1 - rho) * lmbda0_eta[p] + rho * eta

        proba = expit(lmbda0_eta_new[p] + lmbda_logit_prior)
        lmbda_new[0, p] = 2 * proba - 1
        lmbda_sigma_new[0, p, p] = 1 - lmbda[0, p] ** 2

    for k in range(1, n_layers):
        eta1, eta2 = calculate_natural_parameters(
            Y, X, X_sigma, delta, lmbda_var_prior, omega_pos, omega_neg,
            pos_subsamples, neg_subsamples, pos_ratios, neg_ratios, k)

        lmbda_eta1_new[k] = (1 - rho) * lmbda_eta1[k] + rho * eta1
        lmbda_eta2_new[k] = (1 - rho) * lmbda_eta2[k] + rho * eta2

        lmbda_sigma_new[k] = np.linalg.pinv(lmbda_eta2_new[k])
        lmbda_new[k] = np.dot(lmbda_sigma_new[k], lmbda_eta1_new[k])

    return LambdaParams(lmbda=lmbda_new, lmbda_sigma=lmbda_sigma_new,
                        lmbda0_eta=lmbda0_eta_new,
                        lmbda_eta1=lmbda_eta1_new, lmbda_eta2=lmbda_eta2_new)
