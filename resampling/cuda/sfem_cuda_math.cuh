#ifndef __SFEM_CUDA_MATH_CUH__
#define __SFEM_CUDA_MATH_CUH__

#include "sfem_base.h"
#include "sfem_config.h"

/**
 * @brief floor function for real_t
 *
 * @param x
 * @return __device__
 */
__device__ inline real_t        //
floor_real_t(const real_t x) {  //

#if SFEM_REAL_T_IS_FLOAT64
    return floor(x);
#else
    return floorf(x);
#endif
}  // end of floor_real_t
////////////////////////////////////////

/**
 * @brief sqrt function for real_t
 *
 * @param x
 * @return __device__
 */
__device__ inline real_t       //
sqrt_real_t(const real_t x) {  //

#if SFEM_REAL_T_IS_FLOAT64
    return sqrt(x);
#else
    return sqrtf(x);
#endif
}  // end of sqrt_real_t
////////////////////////////////////////

/**
 * @brief abs function for real_t
 *
 * @param x
 * @return __device__
 */
__device__ inline real_t      //
abs_real_t(const real_t x) {  //

#if SFEM_REAL_T_IS_FLOAT64
    return fabs(x);
#else
    return fabsf(x);
#endif
}  // end of abs_real_t
////////////////////////////////////////

/**
 * @brief pow function for real_t
 *
 * @param x
 * @param y
 * @return __device__
 */
__device__ inline real_t                      //
pow_real_t(const real_t x, const real_t y) {  //

#if SFEM_REAL_T_IS_FLOAT64
    return pow(x, y);
#else
    return powf(x, y);
#endif
}  // end of pow_real_t
////////////////////////////////////////

#endif  // __SFEM_CUDA_MATH_CUH__