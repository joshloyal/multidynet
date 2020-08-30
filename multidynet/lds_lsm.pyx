# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as np


def calculate_natural_parameters(double[:, :, ::1] Y,
                                 double[:, :, ::1] X,
                                 double[:, :, :, ::1] X_sigma,
                                 double[::1] lmbda,
                                 double[:, ::1] lmbda_sigma,
                                 double[:] delta,
                                 double[:, :, ::1] omega,
                                 int i):
    cdef size_t t, j, p, q
    cdef size_t n_time_steps = Y.shape[0]
    cdef size_t n_nodes = Y.shape[1]
    cdef size_t n_features = X.shape[2]

    cdef np.ndarray[double, ndim=2, mode='c'] eta1 = np.zeros(
        (n_time_steps, n_features), dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode='c'] eta2 = np.zeros(
        (n_time_steps, n_features, n_features), dtype=np.float64)

    for t in range(n_time_steps):
        for j in range(n_nodes):
            if j != i and Y[t, i, j] != -1.0:
                for p in range(n_features):
                    eta1[t, p] += (
                        lmbda[p] * X[t, j, p] * (
                            Y[t, i, j] - 0.5 -
                                omega[t, i, j] * (delta[i] + delta[j])))

                    for q in range(p + 1):
                        eta2[t, p, q] += omega[t, i, j] * (
                            (lmbda_sigma[p, q] +
                                lmbda[p] * lmbda[q]) *
                            (X_sigma[t, j, p, q] +
                                X[t, j, p] * X[t, j, q]))
                        eta2[t, q, p] = eta2[t, p, q]

    return eta1, eta2


def kalman_filter(np.ndarray[double, ndim=2, mode='c'] A,
                  np.ndarray[double, ndim=3, mode='c'] B,
                  double tau_prec,
                  double sigma_prec):
    cdef size_t t
    cdef size_t n_time_steps = A.shape[0]
    cdef size_t n_features = A.shape[1]

    # allocate temporary arrays
    cdef np.ndarray[double, ndim=2, mode='c'] mu = np.zeros(
        (n_time_steps, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] sigma = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] sigma_inv = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] sigma_star = np.zeros(
        (n_time_steps - 1, n_features, n_features))
    cdef np.ndarray[double, ndim=2, mode='c'] F_init_inv = (
        tau_prec * np.eye(n_features))
    cdef np.ndarray[double, ndim=2, mode='c'] F_inv = (
        sigma_prec * np.eye(n_features))

    # t = 1
    sigma_inv[0] = F_init_inv + B[0]
    sigma[0] = np.linalg.pinv(sigma_inv[0])
    mu[0] = np.dot(sigma[0], A[0])

    for t in range(1, n_time_steps):
        sigma_star[t-1] = np.linalg.pinv(F_inv + sigma_inv[t-1])
        sigma_inv[t] = F_inv + B[t] - (sigma_prec ** 2) * sigma_star[t-1]
        sigma[t] = np.linalg.pinv(sigma_inv[t])
        mu[t] = np.dot(sigma[t], A[t] +
            sigma_prec * np.dot(
                sigma_star[t-1], np.dot(sigma_inv[t-1],  mu[t-1])))

    return mu, sigma, sigma_inv, sigma_star


def kalman_smoother(np.ndarray[double, ndim=2, mode='c'] A,
                    np.ndarray[double, ndim=3, mode='c'] B,
                    double tau_prec,
                    double sigma_prec):
    cdef size_t t
    cdef size_t n_time_steps = A.shape[0]
    cdef size_t n_features = A.shape[1]

    # Allocate temporary arrays
    cdef np.ndarray[double, ndim=2, mode='c'] mu
    cdef np.ndarray[double, ndim=3, mode='c'] sigma
    cdef np.ndarray[double, ndim=3, mode='c'] sigma_inv
    cdef np.ndarray[double, ndim=3, mode='c'] sigma_star
    cdef np.ndarray[double, ndim=2, mode='c'] eta = np.zeros(
        (n_time_steps, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] psi = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] psi_inv = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] psi_star = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=2, mode='c'] F_init_inv = (
        tau_prec * np.eye(n_features))
    cdef np.ndarray[double, ndim=2, mode='c'] F_inv = (
        sigma_prec * np.eye(n_features))
    cdef np.ndarray[double, ndim=2, mode='c'] mean = np.zeros(
        (n_time_steps, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] cov = np.zeros(
        (n_time_steps, n_features, n_features))
    cdef np.ndarray[double, ndim=3, mode='c'] cross_cov = np.zeros(
        (n_time_steps - 1, n_features, n_features))

    # run the filter for the forward message variables
    mu, sigma, sigma_inv, sigma_star = kalman_filter(A, B, tau_prec, sigma_prec)

    # run the smoother
    mean[n_time_steps - 1] = mu[n_time_steps - 1]
    cov[n_time_steps - 1] = sigma[n_time_steps - 1]
    for t in range(n_time_steps - 1, 0, -1):
        psi_star[t] = np.linalg.pinv(F_inv + B[t] + psi_inv[t])
        psi_inv[t-1] = F_inv - (sigma_prec ** 2) * psi_star[t]
        psi[t-1] = np.linalg.pinv(psi_inv[t-1])
        eta[t-1] = sigma_prec * np.dot(psi[t-1], np.dot(psi_star[t],
            A[t] + np.dot(psi_inv[t], eta[t])))

        # update marginals and cross-covariances
        cov[t-1] = np.linalg.pinv(sigma_inv[t-1] + psi_inv[t-1])
        mean[t-1] = np.dot(cov[t-1],
            np.dot(sigma_inv[t-1], mu[t-1]) +
                np.dot(psi_inv[t-1], eta[t-1]))
        cross_cov[t-1] = sigma_prec * np.dot(sigma_star[t-1],
            np.linalg.pinv(
                F_inv + B[t] + psi_inv[t] - # psi_star^{-1}
                    (sigma_prec ** 2) * sigma_star[t-1]))

    return mean, cov, cross_cov


def update_latent_positions(double[:, :, ::1] Y,
                            np.ndarray[double, ndim=3, mode='c'] X,
                            np.ndarray[double, ndim=4, mode='c'] X_sigma,
                            np.ndarray[double, ndim=4, mode='c'] X_cross_cov,
                            double[::1] lmbda,
                            double[:, ::1] lmbda_sigma,
                            double[:] delta,
                            double[:, :, ::1] omega,
                            double tau_prec,
                            double sigma_prec):
    cdef size_t i
    cdef size_t n_nodes = X.shape[1]
    cdef np.ndarray[double, ndim=2, mode='c'] A
    cdef np.ndarray[double, ndim=3, mode='c'] B

    for i in range(n_nodes):
        A, B = calculate_natural_parameters(
            Y, X, X_sigma, lmbda, lmbda_sigma, delta, omega, i)

        X[:, i], X_sigma[:, i], X_cross_cov[:, i] = kalman_smoother(
            A, B, tau_prec, sigma_prec)
