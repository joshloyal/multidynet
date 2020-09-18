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
                                 double[:, :, :, ::1] XLX,
                                 double[:, :, ::1] delta,
                                 double[:, :, :, ::1] omega,
                                 int k, int i):
    cdef size_t t, j
    cdef size_t n_time_steps = Y.shape[1]
    cdef size_t n_nodes = Y.shape[2]

    cdef double[::1] eta1 = np.zeros(n_time_steps, dtype=np.float64)
    cdef double[::1] eta2 = np.zeros(n_time_steps, dtype=np.float64)

    for t in range(n_time_steps):
        for j in range(n_nodes):
            if j != i and Y[k, t, i, j] != -1.0:
                eta1[t] += (Y[k, t, i, j] - 0.5 -
                            omega[k, t, i, j] * (
                                 delta[k, t, j] + XLX[k, t, i, j]))

                eta2[t] += omega[k, t, i, j]

    return np.asarray(eta1), np.asarray(eta2)


def kalman_filter(double[:] A,
                  double[:] B,
                  double tau_prec,
                  double sigma_prec):
    cdef size_t t
    cdef size_t n_time_steps = A.shape[0]

    # allocate temporary arrays
    cdef np.ndarray[double, ndim=1, mode='c'] mu = np.zeros(n_time_steps)
    cdef np.ndarray[double, ndim=1, mode='c'] sigma = np.zeros(n_time_steps)
    cdef np.ndarray[double, ndim=1, mode='c'] sigma_inv = np.zeros(n_time_steps)
    cdef np.ndarray[double, ndim=1, mode='c'] sigma_star = np.zeros(
        n_time_steps - 1)

    # t = 1
    sigma_inv[0] = tau_prec + B[0]
    sigma[0] = 1. / sigma_inv[0]
    mu[0] = sigma[0] * A[0]

    for t in range(1, n_time_steps):
        sigma_star[t-1] = 1. / (sigma_prec + sigma_inv[t-1])
        sigma_inv[t] = sigma_prec + B[t] - (sigma_prec ** 2) * sigma_star[t-1]
        sigma[t] = 1. / sigma_inv[t]
        mu[t] = sigma[t] * (A[t] +
            sigma_prec * sigma_star[t-1] * sigma_inv[t-1] * mu[t-1])

    return mu, sigma, sigma_inv, sigma_star


def kalman_smoother(double[:] A,
                    double[:] B,
                    double tau_prec,
                    double sigma_prec):
    cdef size_t t
    cdef size_t n_time_steps = A.shape[0]

    # Allocate temporary arrays
    cdef np.ndarray[double, ndim=1, mode='c'] mu
    cdef np.ndarray[double, ndim=1, mode='c'] sigma
    cdef np.ndarray[double, ndim=1, mode='c'] sigma_inv
    cdef np.ndarray[double, ndim=1, mode='c'] sigma_star
    cdef np.ndarray[double, ndim=1, mode='c'] eta = np.zeros(n_time_steps)
    cdef np.ndarray[double, ndim=1, mode='c'] psi = np.zeros(n_time_steps)
    cdef np.ndarray[double, ndim=1, mode='c'] psi_inv = np.zeros(n_time_steps)
    cdef np.ndarray[double, ndim=1, mode='c'] psi_star = np.zeros(n_time_steps)
    cdef np.ndarray[double, ndim=1, mode='c'] mean = np.zeros(n_time_steps)
    cdef np.ndarray[double, ndim=1, mode='c'] cov = np.zeros(n_time_steps)
    cdef np.ndarray[double, ndim=1, mode='c'] cross_cov = np.zeros(
        n_time_steps - 1)

    # run the filter for the forward message variables
    mu, sigma, sigma_inv, sigma_star = kalman_filter(A, B, tau_prec, sigma_prec)

    # run the smoother
    mean[n_time_steps - 1] = mu[n_time_steps - 1]
    cov[n_time_steps - 1] = sigma[n_time_steps - 1]
    for t in range(n_time_steps - 1, 0, -1):
        psi_star[t] = 1. / (sigma_prec + B[t] + psi_inv[t])
        psi_inv[t-1] = sigma_prec - (sigma_prec ** 2) * psi_star[t]
        psi[t-1] = 1. / psi_inv[t-1]
        eta[t-1] = sigma_prec * psi[t-1] * psi_star[t] * (
            A[t] + psi_inv[t] * eta[t])

        # update marginals and cross-covariances
        cov[t-1] = 1./ (sigma_inv[t-1] + psi_inv[t-1])
        mean[t-1] = cov[t-1] * (
            sigma_inv[t-1] * mu[t-1] + psi_inv[t-1] * eta[t-1])
        cross_cov[t-1] = sigma_prec * sigma_star[t-1] / (
                sigma_prec + B[t] + psi_inv[t] - # psi_star^{-1}
                    (sigma_prec ** 2) * sigma_star[t-1])

    return mean, cov, cross_cov


def update_deltas(const double[:, :, :, ::1] Y,
                  np.ndarray[double, ndim=3, mode='c'] delta,
                  np.ndarray[double, ndim=3, mode='c'] delta_sigma,
                  np.ndarray[double, ndim=3, mode='c'] delta_cross_cov,
                  double[:, :, :, ::1] XLX,
                  double[:, :, :, ::1] omega,
                  double tau_prec,
                  double sigma_prec):
    cdef size_t k, i
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_nodes = Y.shape[2]
    cdef np.ndarray[double, ndim=1, mode='c'] A
    cdef np.ndarray[double, ndim=1, mode='c'] B

    for k in range(n_layers):
        for i in range(n_nodes):
            A, B = calculate_natural_parameters(
                Y, XLX, delta, omega, k, i)

            delta[k, :, i], delta_sigma[k, :, i], delta_cross_cov[k, :, i] = (
                kalman_smoother(A, B, tau_prec, sigma_prec))
