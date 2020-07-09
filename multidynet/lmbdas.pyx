# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

from libc.math cimport exp, sqrt, tanh

import numpy as np
cimport numpy as np


def calculate_natural_parameters(double[:, :, :, ::1] Y,
                                 double[:, :, ::1] X,
                                 double[:, :, :, ::1] X_sigma,
                                 double[:] beta,
                                 double[:, :, :, ::1] omega,
                                 double lmbda_var_prior,
                                 int k):
    cdef size_t t, i, j, p, q
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = Y.shape[2]
    cdef size_t n_features = X.shape[2]

    cdef np.ndarray[double, ndim=1, mode='c'] eta1 = np.empty(n_features)
    cdef np.ndarray[double, ndim=2, mode='c'] eta2 = np.empty(
        (n_features, n_features))

    for t in range(n_time_steps):
        for i in range(n_nodes):
            for j in range(i):

                for p in range(n_features):
                    eta1[p] += (
                        (Y[k, t, i, j] - 0.5 - omega[k, t, i, j] * beta[k]) *
                            X[t, i, p] * X[t, j, p])

                    for q in range(n_features):
                        eta2[p, q] += omega[k, t, i, j] * (
                            (X_sigma[t, i, p, q] + X[t, i, p] * X[t, i, q]) *
                            (X_sigma[t, j, p, q] + X[t, j, p] * X[t, j, q]))

    eta2[np.diag_indices_from(eta2)] += (1. / lmbda_var_prior)

    return eta1, eta2


def update_lambdas(double[:, :, :, ::1] Y,
                   double[:, :, ::1] X,
                   double[:, :, :, ::1] X_sigma,
                   double[:] beta,
                   np.ndarray[double, ndim=2, mode='c'] lmbda,
                   np.ndarray[double, ndim=3, mode='c'] lmbda_sigma,
                   double[:, :, :, ::1] omega,
                   double lmbda_var_prior):
    cdef size_t k
    cdef size_t n_layers = Y.shape[0]

    for k in range(n_layers):
        eta1, eta2 = calculate_natural_parameters(
            Y, X, X_sigma, beta, omega, lmbda_var_prior, k)

        lmbda_sigma[k] = np.linalg.pinv(eta2)
        lmbda[k] = np.dot(lmbda_sigma[k], eta1)
