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
                            double[:, :, :] X,
                            double[:, :] lmbda,
                            double[:] intercept) nogil:

    cdef size_t k, t, i, j, p, q
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = X.shape[1]
    cdef size_t n_features = X.shape[2]
    cdef double eta = 0.
    cdef double loglik = 0.

    for k in range(n_layers):
        for t in range(n_time_steps):
            for i in range(n_nodes):
                for j in range(i):
                    eta = intercept[k]
                    for p in range(n_features):
                        eta += lmbda[k, p] * X[t, i, p] * X[t, j, p]
                    loglik += eta - log(1 + exp(eta))

    return loglik
