#ifndef SFEM_CODEGEN_KERNEL_MATH_CUH
#define SFEM_CODEGEN_KERNEL_MATH_CUH

#include <cmath>

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif

#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

namespace sfem {
namespace codegen {

// log(1 + x) without forming 1 + x and losing the small part of x.
//
// A hyperelastic density needs log(det F), and a deformation gradient is
// the identity plus something small, so the argument lands near 1 and a
// plain log discards everything below eps relative to 1.  Measured
// against exact arithmetic that is 5.5e-14 relative at x ~ 1e-3, and it
// is what puts a floor under an energy-based line search.
//
// libm log1p is accurate but is a call with no vector form, so a loop
// containing it is rejected under -Werror=pass-failed.  This is built
// from log and arithmetic instead, and holds 2.5e-16 or better over
// x in [-0.9, 9].
//
// (T(1) - u) + x is the part of x that the addition rounded away, and
// (T(2) - u) is one Newton step for 1/u about u = 1 -- exact to
// O((u-1)^2) precisely where the correction matters, which is why no
// division is needed.  Beyond |x| = 1/2 the correction is both
// unnecessary and badly scaled, so it is dropped: log(1 + x) has
// nothing to cancel there.
//
// The arithmetic depends on IEEE semantics that -ffast-math is allowed
// to optimise away -- it would fold (T(1) - u) + x to zero -- so it is
// fenced.  Without the fence the function silently degrades to plain
// log(1 + x); sfem_LogOnePlusTest is what catches that.
#if defined(__CUDACC__)
#define SFEM_PRECISE_FP_BEGIN
#define SFEM_PRECISE_FP_END
#elif defined(__clang__)
#define SFEM_PRECISE_FP_BEGIN _Pragma("float_control(push)") _Pragma("float_control(precise, on)")
#define SFEM_PRECISE_FP_END _Pragma("float_control(pop)")
#elif defined(__GNUC__)
#define SFEM_PRECISE_FP_BEGIN _Pragma("GCC push_options") _Pragma("GCC optimize (\"no-fast-math\")")
#define SFEM_PRECISE_FP_END _Pragma("GCC pop_options")
#else
#define SFEM_PRECISE_FP_BEGIN
#define SFEM_PRECISE_FP_END
#endif

SFEM_PRECISE_FP_BEGIN
template <typename T>
static __host__ __device__ __forceinline__ T sfem_log1p(const T x) {
#if defined(__CUDACC__)
  // Device code is SIMT, so the library function costs nothing here.
  return ::log1p(x);
#else
  const T u = T(1) + x;
  const T e = (T(1) - u) + x;
  return std::log(u) + (T(2) * std::fabs(x) < T(1) ? e * (T(2) - u) : T(0));
#endif
}
SFEM_PRECISE_FP_END

// The continuation of J^(-m) below Jc for a Mooney-Rivlin energy that stays
// finite through J = 0: J^(-m) for J >= Jc; for Jmin < J < Jc the quartic in
// z = (J - Jc) / (Jc - Jmin) that matches it in value, slope and curvature at
// Jc and has zero slope and curvature at Jmin; constant below Jmin.  The
// quartic's coefficients a_k follow from those conditions.
template <typename T>
static __host__ __device__ __forceinline__ T sfem_continued_power(const T J, const T m, const T Jc, const T Jmin) {
  const T delta = Jc - Jmin;
  const T hc = pow(Jc, -m);
  const T a1 = -m * delta * hc / Jc;
  const T a2 = T(0.5) * m * (m + T(1)) * delta * delta * hc / (Jc * Jc);
  const T a3 = T(4) / T(3) * a2 - a1;
  const T a4 = T(0.5) * (a2 - a1);
  const T zr = (J - Jc) / delta;
  const T z = zr > T(-1) ? zr : T(-1);
  const T Jp = J > Jc ? J : Jc;
  const T hp = pow(Jp, -m);
  const T quartic = hc + z * (a1 + z * (a2 + z * (a3 + z * a4)));
  return J >= Jc ? hp : quartic;
}

template <typename T>
static __host__ __device__ __forceinline__ T sfem_continued_power_d1(const T J, const T m, const T Jc, const T Jmin) {
  const T delta = Jc - Jmin;
  const T hc = pow(Jc, -m);
  const T a1 = -m * delta * hc / Jc;
  const T a2 = T(0.5) * m * (m + T(1)) * delta * delta * hc / (Jc * Jc);
  const T a3 = T(4) / T(3) * a2 - a1;
  const T a4 = T(0.5) * (a2 - a1);
  const T zr = (J - Jc) / delta;
  const T z = zr > T(-1) ? zr : T(-1);
  const T Jp = J > Jc ? J : Jc;
  const T hp = pow(Jp, -m);
  const T quartic = (a1 + z * (T(2) * a2 + z * (T(3) * a3 + z * T(4) * a4))) / delta;
  return J >= Jc ? -m * hp / Jp : (J > Jmin ? quartic : T(0));
}

template <typename T>
static __host__ __device__ __forceinline__ T sfem_continued_power_d2(const T J, const T m, const T Jc, const T Jmin) {
  const T delta = Jc - Jmin;
  const T hc = pow(Jc, -m);
  const T a1 = -m * delta * hc / Jc;
  const T a2 = T(0.5) * m * (m + T(1)) * delta * delta * hc / (Jc * Jc);
  const T a3 = T(4) / T(3) * a2 - a1;
  const T a4 = T(0.5) * (a2 - a1);
  const T zr = (J - Jc) / delta;
  const T z = zr > T(-1) ? zr : T(-1);
  const T Jp = J > Jc ? J : Jc;
  const T hp = pow(Jp, -m);
  const T quartic = (T(2) * a2 + z * (T(6) * a3 + z * T(12) * a4)) / (delta * delta);
  return J >= Jc ? m * (m + T(1)) * hp / (Jp * Jp) : (J > Jmin ? quartic : T(0));
}

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
