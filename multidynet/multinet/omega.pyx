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


def update_omega_single(double[::1] Xi,
                        double[:, ::1] Xi_sigma,
                        double[::1] Xj,
                        double[:, ::1] Xj_sigma,
                        double[::1] lmbdak,
                        double[:, ::1] lmbdak_sigma,
                        double deltaki,
                        double deltakj,
                        double deltaki_sigma,
                        double deltakj_sigma,
                        size_t n_features):
    cdef double psi_sq = 0.
    cdef double c_omega = 0.
    cdef double omega = 0.
    cdef int p, q = 0

    # calculate the natural parameter
    psi_sq += deltaki_sigma + deltaki ** 2
    psi_sq += deltakj_sigma + deltakj ** 2
    psi_sq += 2 * deltaki * deltakj
    for p in range(n_features):
        psi_sq += (2 * (deltaki + deltakj) *
                    lmbdak[p] * Xi[p] * Xj[p])

        for q in range(n_features):
            psi_sq += ((lmbdak_sigma[p, q] + lmbdak[p] * lmbdak[q]) *
                       (Xi_sigma[p, q] + Xi[p] * Xi[q]) *
                       (Xj_sigma[p, q] + Xj[p] * Xj[q]))

    # calculate mean of a PG(1, sqrt(c_omega)) random variable
    c_omega = sqrt(psi_sq)
    omega = tanh(0.5 * c_omega)
    omega /= (2. * c_omega)

    return omega, psi_sq


cpdef double update_omega(const double[:, :, ::1] Y,
                          double[:, :, ::1] omega,
                          double[:, ::1] X,
                          double[:, :, ::1] X_sigma,
                          double[:, ::1] lmbda,
                          double[:, :, ::1] lmbda_sigma,
                          double[:, ::1] delta,
                          double[:, ::1] delta_sigma,
                          size_t n_features):
    cdef size_t n_layers = omega.shape[0]
    cdef size_t n_nodes = omega.shape[1]
    cdef size_t k, i, j, p
    cdef double loglik = 0.
    cdef double psi, psi_sq = 0.

    for k in range(n_layers):
        for i in range(n_nodes):
            for j in range(i):
                if Y[k, i, j] != -1.0:
                    omega[k, i, j], psi_sq = update_omega_single(
                        X[i], X_sigma[i], X[j], X_sigma[j],
                        lmbda[k], lmbda_sigma[k],
                        delta[k, i], delta[k, j],
                        delta_sigma[k, i], delta_sigma[k, j],
                        n_features)

                    omega[k, j, i] = omega[k, i, j]

                    psi = delta[k, i] + delta[k, j]
                    for p in range(n_features):
                        psi += lmbda[k, p] * X[i, p] * X[j, p]

                    loglik += (Y[k, i, j] - 0.5) * psi
                    loglik -= 0.5 * omega[k, i, j] * psi_sq

    return loglik
