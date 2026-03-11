#ifndef __SFEM_GPU_MATH_CUH__
#define __SFEM_GPU_MATH_CUH__

// Fast floor for float/double -> int (round down)
__device__ __forceinline__ int fast_floorf(float x) { return __float2int_rd(x); }
__device__ __forceinline__ int fast_floord(double x) { return __double2int_rd(x); }
template <typename T>
__device__ __forceinline__ int fast_floor(T x);
template <>
__device__ __forceinline__ int fast_floor<float>(float x) {
    return fast_floorf(x);
}
template <>
__device__ __forceinline__ int fast_floor<double>(double x) {
    return fast_floord(x);
}

// Fast ceil for float/double -> int (round up)
__device__ __forceinline__ int fast_ceilf(float x) { return __float2int_ru(x); }
__device__ __forceinline__ int fast_ceild(double x) { return __double2int_ru(x); }

template <typename T>
__device__ __forceinline__ int fast_ceil(T x);
template <>
__device__ __forceinline__ int fast_ceil<float>(float x) {
    return fast_ceilf(x);
}
template <>
__device__ __forceinline__ int fast_ceil<double>(double x) {
    return fast_ceild(x);
}

// Fast min/max for float/double
__device__ __forceinline__ float  fast_minf(float a, float b) { return a < b ? a : b; }
__device__ __forceinline__ double fast_mind(double a, double b) { return a < b ? a : b; }

template <typename T>
__device__ __forceinline__ T fast_min(T a, T b);
template <>
__device__ __forceinline__ float fast_min<float>(float a, float b) {
    return fast_minf(a, b);
}
template <>
__device__ __forceinline__ double fast_min<double>(double a, double b) {
    return fast_mind(a, b);
}

// Fast max for float/double
__device__ __forceinline__ float  fast_maxf(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ double fast_maxd(double a, double b) { return a > b ? a : b; }

template <typename T>
__device__ __forceinline__ T fast_max(T a, T b);
template <>
__device__ __forceinline__ float fast_max<float>(float a, float b) {
    return fast_maxf(a, b);
}
template <>
__device__ __forceinline__ double fast_max<double>(double a, double b) {
    return fast_maxd(a, b);
}

// Fast fused multiply-add for float/double
__device__ __forceinline__ float  fast_fmaf(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __forceinline__ double fast_fmad(double a, double b, double c) { return fma(a, b, c); }
template <typename T>
__device__ __forceinline__ T fast_fma(T a, T b, T c);

template <>
__device__ __forceinline__ float fast_fma<float>(float a, float b, float c) {
    return fast_fmaf(a, b, c);
}
template <>
__device__ __forceinline__ double fast_fma<double>(double a, double b, double c) {
    return fast_fmad(a, b, c);
}

// Fast sqrt for float/double
__device__ __forceinline__ float  fast_sqrtf(float x) { return sqrtf(x); }
__device__ __forceinline__ double fast_sqrtd(double x) { return sqrt(x); }
template <typename T>
__device__ __forceinline__ T fast_sqrt(T x);
template <>
__device__ __forceinline__ float fast_sqrt<float>(float x) {
    return fast_sqrtf(x);
}
template <>
__device__ __forceinline__ double fast_sqrt<double>(double x) {
    return fast_sqrtd(x);
}

// Fast abs for float/double
__device__ __forceinline__ float  fast_absf(float x) { return fabsf(x); }
__device__ __forceinline__ double fast_absd(double x) { return fabs(x); }

template <typename T>
T __device__ __forceinline__ fast_abs(T x);
template <>
__device__ __forceinline__ float fast_abs<float>(float x) {
    return fast_absf(x);
}
template <>
__device__ __forceinline__ double fast_abs<double>(double x) {
    return fast_absd(x);
}

#endif  // END __SFEM_GPU_MATH_CUH__