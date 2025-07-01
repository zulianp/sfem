/**
 * @file sfem_cuda_math.cuh
 * @brief CUDA device functions for mathematical operations with real_t precision handling
 * 
 * This header provides CUDA device functions that automatically select the appropriate
 * precision (float or double) based on the SFEM_REAL_T_IS_FLOAT64 configuration.
 * All functions are inlined for optimal performance on GPU devices.
 * 
 * @author SFEM Team
 * @date 2025
 */

#ifndef __SFEM_CUDA_MATH_CUH__
#define __SFEM_CUDA_MATH_CUH__

#include "sfem_base.h"
#include "sfem_config.h"


/**
 * @brief Fused multiply-add operation for real_t precision
 * 
 * Computes a * b + c with improved accuracy compared to separate multiply and add operations.
 * Automatically selects double or single precision based on real_t configuration.
 * 
 * @param a First multiplicand
 * @param b Second multiplicand  
 * @param c Addend value
 * @return real_t Result of the fused multiply-add operation (a * b + c)
 * 
 * @note Uses hardware FMA instruction when available for better performance and accuracy
 */
__device__ inline real_t  //
fma_real_t(const real_t a, const real_t b, const real_t c) {
#if SFEM_REAL_T_IS_FLOAT64
    return fma(a, b, c);
#else
    return fmaf(a, b, c);
#endif
}

/**
 * @brief Floor function for real_t precision
 *
 * Returns the largest integer value that is less than or equal to x.
 * Automatically selects double or single precision based on real_t configuration.
 *
 * @param x Input value to floor
 * @return real_t Largest integer ≤ x, represented as real_t
 * 
 * @note For x = 2.7, returns 2.0; for x = -2.7, returns -3.0
 */
__device__ inline real_t        //
floor_real_t(const real_t x) {  //

#if SFEM_REAL_T_IS_FLOAT64
    return floor(x);
#else
    return floorf(x);
#endif
}

/**
 * @brief Square root function for real_t precision
 *
 * Computes the positive square root of x.
 * Automatically selects double or single precision based on real_t configuration.
 *
 * @param x Input value (must be non-negative)
 * @return real_t Square root of x
 * 
 * @note Returns NaN for negative inputs. For x = 0, returns 0.
 * @warning Input values should be non-negative for meaningful results
 */
__device__ inline real_t       //
sqrt_real_t(const real_t x) {  //

#if SFEM_REAL_T_IS_FLOAT64
    return sqrt(x);
#else
    return sqrtf(x);
#endif
}

/**
 * @brief Absolute value function for real_t precision
 *
 * Computes the absolute (non-negative) value of x.
 * Automatically selects double or single precision based on real_t configuration.
 *
 * @param x Input value
 * @return real_t Absolute value of x (always non-negative)
 * 
 * @note For x ≥ 0, returns x; for x < 0, returns -x
 */
__device__ inline real_t      //
abs_real_t(const real_t x) {  //

#if SFEM_REAL_T_IS_FLOAT64
    return fabs(x);
#else
    return fabsf(x);
#endif
}

/**
 * @brief Power function for real_t precision
 *
 * Computes x raised to the power of y (x^y).
 * Automatically selects double or single precision based on real_t configuration.
 *
 * @param x Base value
 * @param y Exponent value
 * @return real_t Result of x^y
 * 
 * @note Special cases:
 *       - pow(x, 0) = 1 for any x (including x = 0)
 *       - pow(0, y) = 0 for y > 0
 *       - pow(x, 1) = x
 * @warning May return NaN or infinity for certain combinations (e.g., negative base with non-integer exponent)
 */
__device__ inline real_t                      //
pow_real_t(const real_t x, const real_t y) {  //

#if SFEM_REAL_T_IS_FLOAT64
    return pow(x, y);
#else
    return powf(x, y);
#endif
}

#endif  // __SFEM_CUDA_MATH_CUH__