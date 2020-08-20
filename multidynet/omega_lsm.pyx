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


def update_omega_single(double[::1] Xit,
                        double[:, ::1] Xit_sigma,
                        double[::1] Xjt,
                        double[:, ::1] Xjt_sigma,
                        double deltai, double deltai_sigma,
                        double deltaj, double deltaj_sigma):
    cdef double psi_sq = 0.
    cdef double c_omega = 0.
    cdef double omega = 0.
    cdef size_t n_features = Xit.shape[0]
    cdef int p, q = 0

    # calculate the natural parameter
    psi_sq += deltai_sigma + deltai ** 2
    psi_sq += deltaj_sigma + deltaj ** 2
    psi_sq += 2 * deltai * deltaj
    for p in range(n_features):
        psi_sq += 2 * (deltai + deltaj) * Xit[p] * Xjt[p]

        for q in range(n_features):
            psi_sq += ((Xit_sigma[p, q] + Xit[p] * Xit[q]) *
                       (Xjt_sigma[p, q] + Xjt[p] * Xjt[q]))

    # calculate mean of a PG(1, sqrt(c_omega)) random variable
    c_omega = sqrt(psi_sq)
    omega = tanh(0.5 * c_omega)
    omega /= (2. * c_omega)

    return omega, psi_sq


cpdef double update_omega(const double[:, :, ::1] Y,
                          double[:, :, ::1] omega,
                          double[:, :, ::1] X,
                          double[:, :, :, ::1] X_sigma,
                          double[:] delta,
                          double[:] delta_sigma):
    cdef size_t n_time_steps = omega.shape[0]
    cdef size_t n_nodes = omega.shape[1]
    cdef size_t n_features = X.shape[2]
    cdef size_t t, i, j
    cdef double loglik = 0.
    cdef double psi, psi_sq = 0.

    for t in range(n_time_steps):
        for i in range(n_nodes):
            for j in range(i):
                if Y[t, i, j] != -1.0:
                    omega[t, i, j], psi_sq = update_omega_single(
                         X[t, i], X_sigma[t, i], X[t, j], X_sigma[t, j],
                         delta[i], delta_sigma[i], delta[j], delta_sigma[j])

                    omega[t, j, i] = omega[t, i, j]

                    psi = delta[i] + delta[j]
                    for p in range(n_features):
                        psi += X[t, i, p] * X[t, j, p]

                    loglik += (Y[t, i, j] - 0.5) * psi
                    loglik -= 0.5 * omega[t, i, j] * psi_sq

    return loglik
