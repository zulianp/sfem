#ifndef SFEM_ARGSORT_H
#define SFEM_ARGSORT_H

#include "sfem_base.h"

#include <stddef.h>

void argsort_f(const ptrdiff_t n, const geom_t *key, idx_t *idx);

#endif //SFEM_ARGSORT_H
