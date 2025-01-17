#ifndef SFEM_VEC_H
#define SFEM_VEC_H

#include "sfem_base.h"

#ifdef __ARM_NEON

#define SFEM_VREAL_SIZE 8
typedef real_t vreal_t 	__attribute__ ((vector_size (sizeof(real_t)	* SFEM_VREAL_SIZE)));

typedef int vint_t 		__attribute__ ((vector_size (sizeof(int)    * SFEM_VREAL_SIZE)));
typedef long vmask_t 	__attribute__ ((vector_size (sizeof(long)    * SFEM_VREAL_SIZE)));
typedef idx_t vidx_t 	__attribute__ ((vector_size (sizeof(idx_t)  * SFEM_VREAL_SIZE)));

typedef real_t real4_t 	__attribute__ ((vector_size (sizeof(real_t)	* 4)));
typedef long   mask4_t  __attribute__ ((vector_size (sizeof(long)	* 4)));

typedef real_t real8_t 	__attribute__ ((vector_size (sizeof(real_t)	* 8)));
typedef long   mask8_t  __attribute__ ((vector_size (sizeof(long)	* 8)));

typedef float float16_t 	__attribute__ ((vector_size (sizeof(float)	* 16)));
typedef double double8_t 	__attribute__ ((vector_size (sizeof(double)	* 8)));

#else
#ifdef _WIN32
#define SFEM_VREAL_SIZE 1

typedef real_t vreal_t;
typedef int vint_t ;
typedef long vmask_t;
typedef idx_t vidx_t;

#else

#define SFEM_VREAL_SIZE 4
typedef real_t vreal_t 	__attribute__ ((vector_size (sizeof(real_t)	* SFEM_VREAL_SIZE)));
typedef int vint_t 		__attribute__ ((vector_size (sizeof(int)    * SFEM_VREAL_SIZE)));
typedef long vmask_t 	__attribute__ ((vector_size (sizeof(long)    * SFEM_VREAL_SIZE)));
typedef idx_t vidx_t 	__attribute__ ((vector_size (sizeof(idx_t)  * SFEM_VREAL_SIZE)));


typedef real_t real4_t 	__attribute__ ((vector_size (sizeof(real_t)	* 4)));
typedef long   mask4_t  __attribute__ ((vector_size (sizeof(long)	* 4)));

typedef float float8_t 	__attribute__ ((vector_size (sizeof(float)	* 8)));
typedef double double4_t 	__attribute__ ((vector_size (sizeof(double)	* 4)));

#endif
#endif
#endif //SFEM_BASE_H
