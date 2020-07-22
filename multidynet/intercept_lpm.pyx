# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as np


def calculate_natural_parameters(np.ndarray[double, ndim=3, mode='c'] Y,
                                 np.ndarray[double, ndim=3, mode='c'] X,
                                 np.ndarray[double, ndim=3, mode='c'] omega,
                                 double intercept_var_prior):
    cdef size_t t
    cdef size_t n_time_steps = Y.shape[0]
    cdef np.ndarray[double, ndim=2, mode='c'] tmp
    cdef double eta1 = 0.
    cdef double eta2 = 0.

    # divide by 2 since we only need the lower half of the matrix
    eta2 = 0.5 * np.sum(omega) + (1. / intercept_var_prior)
    for t in range(n_time_steps):
        tmp = Y[t] - 0.5 - omega[t] * np.dot(X[t], X[t].T)
        eta1 += tmp[np.tril_indices_from(tmp, k=-1)].sum()

    return eta1, eta2


def update_intercept(np.ndarray[double, ndim=3, mode='c'] Y,
                     np.ndarray[double, ndim=3, mode='c'] X,
                     np.ndarray[double, ndim=3, mode='c'] omega,
                     double intercept_var_prior):
    cdef double eta1
    cdef double eta2
    cdef double intercept,
    cdef double intercept_sigma,

    eta1, eta2 = calculate_natural_parameters(
        Y, X, omega, intercept_var_prior)

    intercept_sigma = 1. / eta2
    intercept = intercept_sigma * eta1

    return intercept, intercept_sigma
