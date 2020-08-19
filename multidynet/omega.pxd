# cython: language_level=3

# update a single auxilary variable
cdef double update_omega_svi(double[::1] Xit,
                             double[:, ::1] Xit_sigma,
                             double[::1] Xjt,
                             double[:, ::1] Xjt_sigma,
                             double[::1] lmbdak,
                             double[:, ::1] lmbdak_sigma,
                             double deltaki,
                             double deltakj,
                             double deltaki_sigma,
                             double deltakj_sigma)
