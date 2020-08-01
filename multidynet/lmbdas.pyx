# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

from libc.math cimport exp
from scipy.special.cython_special import expit

import numpy as np
cimport numpy as np


def calculate_natural_parameters_reference(const double[:, :, :, ::1] Y,
                                           double[:, :, ::1] X,
                                           double[:, :, :, ::1] X_sigma,
                                           double[:] intercept,
                                           double[:, :] delta,
                                           double[:, :, :, ::1] omega,
                                           double[:] lmbda,
                                           int p):
    cdef size_t t, i, j, q
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = X.shape[1]
    cdef size_t n_features = X.shape[2]
    cdef double eta = 0.

    for t in range(n_time_steps):
        for i in range(n_nodes):
            for j in range(i):
                if Y[0, t, i, j] != -1.0:
                    eta += (
                        (Y[0, t, i, j] - 0.5 - omega[0, t, i, j] * (
                            intercept[0] + delta[0, i] + delta[0, j])) *
                            X[t, i, p] * X[t, j, p])

                    for q in range(n_features):
                        if q != p:
                            eta -= omega[0, t, i, j] * lmbda[q] * (
                                (X_sigma[t, i, q, p] + X[t, i, q] * X[t, i, p]) *
                                (X_sigma[t, j, q, p] + X[t, j, q] * X[t, j, p]))


    return 2 * eta


def calculate_natural_parameters(const double[:, :, :, ::1] Y,
                                 double[:, :, ::1] X,
                                 double[:, :, :, ::1] X_sigma,
                                 double[:] intercept,
                                 double[:, :] delta,
                                 double[:, :, :, ::1] omega,
                                 double lmbda_var_prior,
                                 int k):
    cdef size_t t, i, j, p, q
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = X.shape[1]
    cdef size_t n_features = X.shape[2]

    cdef np.ndarray[double, ndim=1, mode='c'] eta1 = np.zeros(n_features)
    cdef np.ndarray[double, ndim=2, mode='c'] eta2 = np.zeros(
        (n_features, n_features))

    for t in range(n_time_steps):
        for i in range(n_nodes):
            for j in range(i):
                if Y[k, t, i, j] != -1.0:
                    for p in range(n_features):
                        eta1[p] += (
                            (Y[k, t, i, j] - 0.5 -
                                omega[k, t, i, j] * (
                                    intercept[k] + delta[k, i] + delta[k, j])) *
                            X[t, i, p] * X[t, j, p])

                        for q in range(p + 1):
                            eta2[p, q] += omega[k, t, i, j] * (
                                (X_sigma[t, i, p, q] + X[t, i, p] * X[t, i, q]) *
                                (X_sigma[t, j, p, q] + X[t, j, p] * X[t, j, q]))
                            eta2[q, p] = eta2[p, q]

    eta2[np.diag_indices_from(eta2)] += (1. / lmbda_var_prior)

    return eta1, eta2


def update_lambdas(const double[:, :, :, ::1] Y,
                   double[:, :, ::1] X,
                   double[:, :, :, ::1] X_sigma,
                   double[:] intercept,
                   np.ndarray[double, ndim=2, mode='c'] lmbda,
                   np.ndarray[double, ndim=3, mode='c'] lmbda_sigma,
                   double[:, ::1] delta,
                   double[:, :, :, ::1] omega,
                   double lmbda_var_prior,
                   double lmbda_logit_prior):
    cdef size_t k, p
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_features = lmbda.shape[1]
    cdef double eta, proba
    cdef np.ndarray[double, ndim=1, mode='c'] eta1
    cdef np.ndarray[double, ndim=2, mode='c'] eta2

    # start by updating signs of the reference layer
    for p in range(n_features):
        eta = calculate_natural_parameters_reference(
            Y, X, X_sigma, intercept, delta,  omega, lmbda[0], p)
        proba = expit(eta + lmbda_logit_prior)
        lmbda[0, p] = 2 * proba - 1
        lmbda_sigma[0, p, p] = 1 - lmbda[0, p] ** 2

    for k in range(1, n_layers):
        eta1, eta2 = calculate_natural_parameters(
            Y, X, X_sigma, intercept, delta, omega, lmbda_var_prior, k)

        lmbda_sigma[k] = np.linalg.pinv(eta2)
        lmbda[k] = np.dot(lmbda_sigma[k], eta1)
