# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

from libc.math cimport exp, log

import numpy as np
cimport numpy as np


cpdef double log_likelihood(double[:, :, :, ::1] Y,
                            double[:, :, ::1] X,
                            double[:, ::1] lmbda,
                            double[:, :, ::1] delta,
                            size_t n_features) nogil:

    cdef size_t k, t, i, j, p, q
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = delta.shape[2]
    cdef double eta = 0.
    cdef double loglik = 0.

    for k in range(n_layers):
        for t in range(n_time_steps):
            for i in range(n_nodes):
                for j in range(i):
                    if Y[k, t, i, j] != -1.:
                        eta = delta[k, t, i] + delta[k, t, j]
                        for p in range(n_features):
                            eta += lmbda[k, p] * X[t, i, p] * X[t, j, p]
                        loglik += Y[k, t, i, j] * eta - log(1 + exp(eta))

    return loglik


def pointwise_log_likelihood(double[:, :, :, ::1] Y,
                             double[:, :, ::1] X,
                             double[:, ::1] lmbda,
                             double[:, :, ::1] delta,
                             size_t n_features):

    cdef size_t k, t, i, j, p, q
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = delta.shape[2]
    cdef size_t n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    cdef size_t idx = 0
    cdef double eta = 0.
    cdef np.ndarray[double, ndim=3, mode='c'] loglik = np.zeros(
        (n_layers, n_time_steps, n_dyads), dtype=np.float64)

    for k in range(n_layers):
        for t in range(n_time_steps):
            for i in range(n_nodes):
                for j in range(i):
                    if Y[k, t, i, j] != -1.:
                        eta = delta[k, t, i] + delta[k, t, j]
                        for p in range(n_features):
                            eta += lmbda[k, p] * X[t, i, p] * X[t, j, p]
                        loglik[k, t, idx] =  Y[k, t, i, j] * eta - log(1 + exp(eta))
                    else:
                        loglik[k, t, idx] = 0
                    idx += 1

    return loglik
