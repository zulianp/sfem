#ifndef __PRECISION_TYPES_H__
#define __PRECISION_TYPES_H__

#include "sfem_config.h"

#if SFEM_REAL_T_IS_FLOAT32 == 1
#define USE_SINGLE_PRECISION
#elif SFEM_REAL_T_IS_FLOAT64 == 1
#define USE_DOUBLE_PRECISION
#else
#error "Please define either USE_SINGLE_PRECISION or USE_DOUBLE_PRECISION"
#endif

#ifdef USE_SINGLE_PRECISION
// typedef float real_t;
#define REAL_T_LENGTH 4
#elif defined(USE_DOUBLE_PRECISION)
// typedef double real_t;
#define REAL_T_LENGTH 8
#else
// Default to double precision if no macro is defined
// typedef double real_t;
#define REAL_T_LENGTH 8
#endif

#endif  // __PRECISION_TYPES_H__
