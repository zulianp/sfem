#ifndef OPERATOR_INLINE_CPU_H
#define OPERATOR_INLINE_CPU_H

#include "sfem_config.h"

#ifdef SFEM_ENABLE_FP32_KERNELS
#define VSCALAR_SIZE 8
typedef float scalar_t;
#else
#define VSCALAR_SIZE 4
typedef real_t scalar_t;
#endif

typedef real_t accumulator_t;

#endif
