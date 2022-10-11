# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as np


def calculate_natural_parameters(const double[:, :, ::1] Y,
                                 double[:, ::1] X,
                                 double[:, :, ::1] X_sigma,
                                 double[:, ::1] lmbda,
                                 double[:, :, ::1] lmbda_sigma,
                                 double[:, ::1] delta,
                                 double[:, :, ::1] omega,
                                 int i):
    cdef size_t  k, j, p, q
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_nodes = Y.shape[1]
    cdef size_t n_features = lmbda.shape[1]

    cdef double[:] eta1 = np.zeros(n_features, dtype=np.float64)
    cdef double[:, ::1] eta2 = np.zeros(
        (n_features, n_features), dtype=np.float64)

    for k in range(n_layers):
        for j in range(n_nodes):
            if j != i and Y[k, i, j] != -1.0:
                for p in range(n_features):
                    eta1[p] += (
                        lmbda[k, p] * X[j, p] * (
                            Y[k, i, j] - 0.5 -
                                omega[k, i, j] * (
                                     delta[k, i] + delta[k, j])))

                    for q in range(p + 1):
                        eta2[p, q] += omega[k, i, j] * (
                            (lmbda_sigma[k, p, q] +
                                lmbda[k, p] * lmbda[k, q]) *
                            (X_sigma[j, p, q] +
                                X[j, p] * X[j, q]))
                        eta2[q, p] = eta2[p, q]

    return np.asarray(eta1), np.asarray(eta2)


def update_latent_positions(const double[:, :, ::1] Y,
                            np.ndarray[double, ndim=2, mode='c'] X,
                            np.ndarray[double, ndim=3, mode='c'] X_sigma,
                            double[:, ::1] lmbda,
                            double[:, :, ::1] lmbda_sigma,
                            double[:, ::1] delta,
                            double[:, :, ::1] omega,
                            np.ndarray[double, ndim=2, mode='c'] X0_prec):
    cdef size_t i
    cdef size_t n_nodes = X.shape[0]
    cdef size_t n_features = lmbda.shape[1]
    cdef np.ndarray[double, ndim=1, mode='c'] A
    cdef np.ndarray[double, ndim=2, mode='c'] B

    for i in range(n_nodes):
        A, B = calculate_natural_parameters(
            Y, X, X_sigma, lmbda, lmbda_sigma, delta, omega, i)

        X_sigma[i] = np.linalg.pinv(B + X0_prec)
        X[i] = np.dot(X_sigma[i], A)
