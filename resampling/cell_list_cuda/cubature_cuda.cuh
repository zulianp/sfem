#ifndef __CUBATURE_CUDA_CUH__
#define __CUBATURE_CUDA_CUH__

#define QUAD_N1D 2
#define QUAD_TOTAL 8

#define GAUSS_QUAD_3D
#if defined(GAUSS_QUAD_3D)

__device__ __constant__ double qx_dbl[8] = {2.11324865405187079e-01,
                                            2.11324865405187079e-01,
                                            2.11324865405187079e-01,
                                            2.11324865405187079e-01,
                                            7.88675134594812866e-01,
                                            7.88675134594812866e-01,
                                            7.88675134594812866e-01,
                                            7.88675134594812866e-01};
__device__ __constant__ double qy_dbl[8] = {2.11324865405187079e-01,
                                            2.11324865405187079e-01,
                                            7.88675134594812866e-01,
                                            7.88675134594812866e-01,
                                            2.11324865405187079e-01,
                                            2.11324865405187079e-01,
                                            7.88675134594812866e-01,
                                            7.88675134594812866e-01};
__device__ __constant__ double qz_dbl[8] = {2.11324865405187079e-01,
                                            7.88675134594812866e-01,
                                            2.11324865405187079e-01,
                                            7.88675134594812866e-01,
                                            2.11324865405187079e-01,
                                            7.88675134594812866e-01,
                                            2.11324865405187079e-01,
                                            7.88675134594812866e-01};
__device__ __constant__ double qw_dbl[8] = {1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01};

/* ---- single precision ---- */

__device__ __constant__ float qx_flt[8] = {2.11324865e-01f,
                                           2.11324865e-01f,
                                           2.11324865e-01f,
                                           2.11324865e-01f,
                                           7.88675135e-01f,
                                           7.88675135e-01f,
                                           7.88675135e-01f,
                                           7.88675135e-01f};
__device__ __constant__ float qy_flt[8] = {2.11324865e-01f,
                                           2.11324865e-01f,
                                           7.88675135e-01f,
                                           7.88675135e-01f,
                                           2.11324865e-01f,
                                           2.11324865e-01f,
                                           7.88675135e-01f,
                                           7.88675135e-01f};
__device__ __constant__ float qz_flt[8] = {2.11324865e-01f,
                                           7.88675135e-01f,
                                           2.11324865e-01f,
                                           7.88675135e-01f,
                                           2.11324865e-01f,
                                           7.88675135e-01f,
                                           2.11324865e-01f,
                                           7.88675135e-01f};
__device__ __constant__ float qw_flt[8] = {1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f};
#else

/* ---- double precision ---- */

__device__ __constant__ double qx_dbl[8] = {2.50000000000000000e-01,
                                            2.50000000000000000e-01,
                                            2.50000000000000000e-01,
                                            2.50000000000000000e-01,
                                            7.50000000000000000e-01,
                                            7.50000000000000000e-01,
                                            7.50000000000000000e-01,
                                            7.50000000000000000e-01};

__device__ __constant__ double qy_dbl[8] = {2.50000000000000000e-01,
                                            2.50000000000000000e-01,
                                            7.50000000000000000e-01,
                                            7.50000000000000000e-01,
                                            2.50000000000000000e-01,
                                            2.50000000000000000e-01,
                                            7.50000000000000000e-01,
                                            7.50000000000000000e-01};

__device__ __constant__ double qz_dbl[8] = {2.50000000000000000e-01,
                                            7.50000000000000000e-01,
                                            2.50000000000000000e-01,
                                            7.50000000000000000e-01,
                                            2.50000000000000000e-01,
                                            7.50000000000000000e-01,
                                            2.50000000000000000e-01,
                                            7.50000000000000000e-01};

__device__ __constant__ double qw_dbl[8] = {1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01,
                                            1.25000000000000000e-01};

/* ---- single precision ---- */

__device__ __constant__ float qx_flt[8] = {2.50000000e-01f,
                                           2.50000000e-01f,
                                           2.50000000e-01f,
                                           2.50000000e-01f,
                                           7.50000000e-01f,
                                           7.50000000e-01f,
                                           7.50000000e-01f,
                                           7.50000000e-01f};

__device__ __constant__ float qy_flt[8] = {2.50000000e-01f,
                                           2.50000000e-01f,
                                           7.50000000e-01f,
                                           7.50000000e-01f,
                                           2.50000000e-01f,
                                           2.50000000e-01f,
                                           7.50000000e-01f,
                                           7.50000000e-01f};

__device__ __constant__ float qz_flt[8] = {2.50000000e-01f,
                                           7.50000000e-01f,
                                           2.50000000e-01f,
                                           7.50000000e-01f,
                                           2.50000000e-01f,
                                           7.50000000e-01f,
                                           2.50000000e-01f,
                                           7.50000000e-01f};

__device__ __constant__ float qw_flt[8] = {1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f,
                                           1.25000000e-01f};
#endif

// Primary template – left undefined intentionally;
// only the explicit specialisations below are valid.
template <typename T>
struct QuadPoints;

template <>
struct QuadPoints<double> {
    __device__ static const double* x() { return qx_dbl; }
    __device__ static const double* y() { return qy_dbl; }
    __device__ static const double* z() { return qz_dbl; }
    __device__ static const double* w() { return qw_dbl; }
};

template <>
struct QuadPoints<float> {
    __device__ static const float* x() { return qx_flt; }
    __device__ static const float* y() { return qy_flt; }
    __device__ static const float* z() { return qz_flt; }
    __device__ static const float* w() { return qw_flt; }
};

#endif  // __CUBATURE_CUDA_CUH__