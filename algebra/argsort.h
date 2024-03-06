#ifndef SFEM_ARGSORT_H
#define SFEM_ARGSORT_H

#include "sfem_base.h"

#include <stddef.h>
#include <stdint.h>

void argsort_f(const ptrdiff_t n, const geom_t *key, idx_t *idx);
void argsort_i(const ptrdiff_t n, const idx_t *key, idx_t *idx);
void argsort_u32(const ptrdiff_t n, const uint32_t *key, idx_t *idx);
void argsort_u64(const ptrdiff_t n, const uint64_t *key, idx_t *idx);

void argsort_u32_ptrdiff_t(const ptrdiff_t n, const uint32_t *key, ptrdiff_t *idx);




#endif //SFEM_ARGSORT_H
