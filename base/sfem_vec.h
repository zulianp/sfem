#ifndef SFEM_VEC_H
#define SFEM_VEC_H

#include "sfem_base.h"

#define SFEM_VECTOR_SIZE 4
typedef real_t vreal_t 	__attribute__ ((vector_size (sizeof(real_t)	* SFEM_VECTOR_SIZE)));
typedef int vint_t 		__attribute__ ((vector_size (sizeof(int)    * SFEM_VECTOR_SIZE)));
typedef idx_t vidx_t 	__attribute__ ((vector_size (sizeof(idx_t)  * SFEM_VECTOR_SIZE)));

#endif //SFEM_BASE_H
