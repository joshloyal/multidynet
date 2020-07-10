# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as np


def calculate_natural_parameters(np.ndarray[double, ndim=4, mode='c'] Y,
                                 np.ndarray[double, ndim=3, mode='c'] X,
                                 np.ndarray[double, ndim=2, mode='c'] lmbda,
                                 np.ndarray[double, ndim=4, mode='c'] omega,
                                 double intercept_var_prior,
                                 int k):
    cdef size_t t
    cdef size_t n_time_steps = Y.shape[1]
    cdef np.ndarray[double, ndim=2, mode='c'] tmp
    cdef double eta1 = 0.
    cdef double eta2 = 0.

    eta1 = np.sum(omega[k]) + (1. / intercept_var_prior)
    for t in range(n_time_steps):
        tmp = Y[k, t] - 1/2. - omega[k, t] * np.dot(X[t] * lmbda[k], X[t].T)
        eta2 += tmp[np.tril_indices_from(tmp, k=-1)].sum()

    return eta1, eta2


def update_intercepts(np.ndarray[double, ndim=4, mode='c'] Y,
                      np.ndarray[double, ndim=3, mode='c'] X,
                      double[:] intercept,
                      double[:] intercept_sigma,
                      np.ndarray[double, ndim=2, mode='c'] lmbda,
                      np.ndarray[double, ndim=4, mode='c'] omega,
                      double intercept_var_prior):
    cdef size_t k
    cdef size_t n_layers = Y.shape[0]
    cdef double eta1
    cdef double eta2

    for k in range(n_layers):
        eta1, eta2 = calculate_natural_parameters(
            Y, X, lmbda, omega, intercept_var_prior, k)

        intercept_sigma[k] = 1. / eta1
        intercept[k] = intercept_sigma[k] * eta2
