# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as np


def calculate_natural_parameters(const double[:, :, :, ::1] Y,
                                 double[:, :, ::1] X,
                                 double[:, ::1] lmbda,
                                 double[:] intercept,
                                 double[:, ::1] delta,
                                 double[:, :, :, ::1] omega,
                                 double delta_var_prior,
                                 int k,
                                 int i):
    cdef size_t t, j, p
    cdef size_t n_time_steps = X.shape[0]
    cdef size_t n_nodes = X.shape[1]
    cdef size_t n_features = X.shape[2]

    cdef double tmp = 0.
    cdef double eta1 = 0.
    cdef double eta2 = 1. / delta_var_prior

    for t in range(n_time_steps):
        for j in range(n_nodes):
            if j != i and Y[k, t, i, j] != -1.0:
                eta2 += omega[k, t, i, j]

                tmp = intercept[k] + delta[k, j]
                for p in range(n_features):
                    tmp += lmbda[k, p] * X[t, i, p] * X[t, j, p]
                eta1 += Y[k, t, i, j] - 0.5 - omega[k, t, i, j] * tmp

    return eta1, eta2


def update_deltas(const double[:, :, :, ::1] Y,
                  double[:, :, ::1] X,
                  double[:, ::1] lmbda,
                  double[:] intercept,
                  np.ndarray[double, ndim=2, mode='c'] delta,
                  np.ndarray[double, ndim=2, mode='c'] delta_sigma,
                  double[:, :, :, ::1] omega,
                  double delta_var_prior):
    cdef size_t k, i
    cdef size_t n_layers = delta.shape[0]
    cdef size_t n_nodes = delta.shape[1]
    cdef double eta1
    cdef double eta2

    # start by updating signs of the reference layer
    for k in range(n_layers):
        for i in range(n_nodes):
            eta1, eta2 = calculate_natural_parameters(
                Y, X, lmbda, intercept, delta, omega, delta_var_prior, k, i)

            delta_sigma[k, i] = 1. / eta2
            delta[k, i] = delta_sigma[k, i] * eta1
