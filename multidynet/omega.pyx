# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

from libc.math cimport exp, sqrt, tanh, pow

import numpy as np
cimport numpy as np


cdef double update_omega_single(double interceptk,
                                double interceptk_sigma,
                                double[::1] Xit,
                                double[:, ::1] Xit_sigma,
                                double[::1] Xjt,
                                double[:, ::1] Xjt_sigma,
                                double[::1] lmbdak,
                                double[:, ::1] lmbdak_sigma) nogil:
    cdef double c_omega = 0.
    cdef double omega = 0.
    cdef size_t n_features = Xit.shape[0]
    cdef int p, q = 0

    # calculate the natural parameter
    c_omega += interceptk_sigma + interceptk ** 2
    for p in range(n_features):
        c_omega += 2 * interceptk * lmbdak[p] * Xit[p] * Xjt[p]

        for q in range(n_features):
            c_omega += ((lmbdak_sigma[p, q] + lmbdak[p] * lmbdak[q]) *
                        (Xit_sigma[p, q] + Xit[p] * Xit[q]) *
                        (Xjt_sigma[p, q] + Xjt[p] * Xjt[q]))

    # calculate mean of a PG(1, sqrt(c_omega)) random variable
    c_omega = sqrt(c_omega)
    omega = tanh(0.5 * c_omega)
    omega /= (2. * c_omega)

    return omega


cpdef void update_omega(double[:, :, :, ::1] omega,
                        double[:, :, ::1] X,
                        double[:, :, :, ::1] X_sigma,
                        double[::1] intercept,
                        double[::1] intercept_sigma,
                        double[:, ::1] lmbda,
                        double[:, :, ::1] lmbda_sigma) nogil:
    cdef size_t n_layers = omega.shape[0]
    cdef size_t n_time_steps = omega.shape[1]
    cdef size_t n_nodes = omega.shape[2]
    cdef size_t k, t, i, j = 0

    for k in range(n_layers):
        for t in range(n_time_steps):
            for i in range(n_nodes):
                for j in range(i):
                    omega[k, t, i, j] = update_omega_single(
                        intercept[k], intercept_sigma[k], X[t, i],
                        X_sigma[t, i], X[t, j], X_sigma[t, j], lmbda[k],
                        lmbda_sigma[k])

                    omega[k, t, j, i] = omega[k, t, i, j]