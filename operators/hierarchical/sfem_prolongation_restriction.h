#ifndef SFEM_PROLONGATION_RESTRICTION_H
#define SFEM_PROLONGATION_RESTRICTION_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int hierarchical_prolongation(const enum ElemType from_element,
                              const enum ElemType to_element,
                              const ptrdiff_t nelements,
                              idx_t **const SFEM_RESTRICT elements,
                              const real_t *const SFEM_RESTRICT from,
                              real_t *const SFEM_RESTRICT to);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_PROLONGATION_RESTRICTION_H
