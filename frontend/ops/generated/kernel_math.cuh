#ifndef SFEM_CODEGEN_KERNEL_MATH_CUH
#define SFEM_CODEGEN_KERNEL_MATH_CUH

namespace sfem {
namespace codegen {

template <typename T>
static __host__ __device__ __forceinline__ T pow_2(const T x) {
    return x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_3(const T x) {
    return x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_4(const T x) {
    return x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_5(const T x) {
    return x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_6(const T x) {
    return x * x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_7(const T x) {
    return x * x * x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_8(const T x) {
    return x * x * x * x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_9(const T x) {
    return x * x * x * x * x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_10(const T x) {
    return x * x * x * x * x * x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_11(const T x) {
    return x * x * x * x * x * x * x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_12(const T x) {
    return x * x * x * x * x * x * x * x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_13(const T x) {
    return x * x * x * x * x * x * x * x * x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_14(const T x) {
    return x * x * x * x * x * x * x * x * x * x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_15(const T x) {
    return x * x * x * x * x * x * x * x * x * x * x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_16(const T x) {
    return x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m1(const T x) {
    return T(1) / x;
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m2(const T x) {
    return T(1) / pow_2(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m3(const T x) {
    return T(1) / pow_3(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m4(const T x) {
    return T(1) / pow_4(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m5(const T x) {
    return T(1) / pow_5(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m6(const T x) {
    return T(1) / pow_6(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m7(const T x) {
    return T(1) / pow_7(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m8(const T x) {
    return T(1) / pow_8(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m9(const T x) {
    return T(1) / pow_9(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m10(const T x) {
    return T(1) / pow_10(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m11(const T x) {
    return T(1) / pow_11(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m12(const T x) {
    return T(1) / pow_12(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m13(const T x) {
    return T(1) / pow_13(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m14(const T x) {
    return T(1) / pow_14(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m15(const T x) {
    return T(1) / pow_15(x);
}

template <typename T>
static __host__ __device__ __forceinline__ T pow_m16(const T x) {
    return T(1) / pow_16(x);
}

} // namespace codegen
} // namespace sfem

#endif
