#ifndef __TET10_WENO_CUDA_H__
#define __TET10_WENO_CUDA_H__

#define real_type double

#if real_type == double
#define Abs(x) fabs(x)
#elif real_type == float
#define Abs(x) fabsf(x)
#endif

#define List2_cu(ARRAY, AA, BB) \
    {                           \
        ARRAY[0] = (AA);        \
        ARRAY[1] = (BB);        \
    }

#define List3_cu(ARRAY, AA, BB, CC) \
    {                               \
        ARRAY[0] = (AA);            \
        ARRAY[1] = (BB);            \
        ARRAY[2] = (CC);            \
    }

#endif  // __TET10_WENO_CUDA_H__