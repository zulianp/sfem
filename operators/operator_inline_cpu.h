#ifndef OPERATOR_INLINE_CPU_H
#define OPERATOR_INLINE_CPU_H

#include "sfem_base.h"
#include "sfem_vec.h"

#ifdef SFEM_ENABLE_FP32_KERNELS
// Single precision indepent from real_t
#define SFEM_VEC_SIZE 8
typedef float scalar_t;
#elif SFEM_ENABLE_FP64_KERNELS
// Double precision indepent from real_t
#define SFEM_VEC_SIZE 4
typedef double scalar_t;
#else
// Same precision as for real_t
#define SFEM_VEC_SIZE SFEM_VREAL_SIZE
typedef real_t scalar_t;
#endif

// TODO play around with accumulator precision
typedef real_t accumulator_t;

typedef scalar_t vec_t
        __attribute__((vector_size(SFEM_VEC_SIZE * sizeof(scalar_t)), aligned(SFEM_VEC_SIZE * sizeof(scalar_t))));


#endif //OPERATOR_INLINE_CPU_H
