#ifndef OPERATOR_INLINE_CPU_H
#define OPERATOR_INLINE_CPU_H

#include "sfem_config.h"

#ifdef SFEM_ENABLE_FP32_KERNELS
typedef float scalar_t;
#else
typedef real_t scalar_t;
#endif

typedef real_t accumulator_t;

#endif
