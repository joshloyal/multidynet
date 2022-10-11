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
                                 double[:, :, ::1] XLX,
                                 double[:, ::1] delta,
                                 double[:,  :, ::1] omega,
                                 int k, int i):
    cdef size_t j
    cdef size_t n_nodes = Y.shape[1]

    cdef double eta1 = 0.
    cdef double eta2 = 0.

    for j in range(n_nodes):
        if j != i and Y[k, i, j] != -1.0:
            eta1 += (Y[k, i, j] - 0.5 -
                        omega[k, i, j] * (
                             delta[k, j] + XLX[k, i, j]))

            eta2 += omega[k, i, j]

    return eta1, eta2


def update_deltas(const double[:, :, ::1] Y,
                  np.ndarray[double, ndim=2, mode='c'] delta,
                  np.ndarray[double, ndim=2, mode='c'] delta_sigma,
                  double[:, :, ::1] XLX,
                  double[:, :, ::1] omega,
                  double tau_prec):
    """Mean Field VB update"""
    cdef size_t k, i
    cdef size_t n_layers = Y.shape[0]
    cdef size_t n_nodes = Y.shape[1]
    cdef double A
    cdef double B

    for k in range(n_layers):
        for i in range(n_nodes):
            A, B = calculate_natural_parameters(
                Y, XLX, delta, omega, k, i)

            delta_sigma[k, i] = 1. / (B + tau_prec)
            delta[k, i] = delta_sigma[k, i] * A
