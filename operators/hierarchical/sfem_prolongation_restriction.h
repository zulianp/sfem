#ifndef SFEM_PROLONGATION_RESTRICTION_H
#define SFEM_PROLONGATION_RESTRICTION_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif


ptrdiff_t max_node_id(const enum ElemType type,
                      const ptrdiff_t nelements,
                      idx_t **const SFEM_RESTRICT elements);

int hierarchical_create_coarse_indices(const idx_t max_coarse_idx,
                                       const ptrdiff_t n_indices,
                                       idx_t *const SFEM_RESTRICT fine_indices,
                                       ptrdiff_t *n_coarse_indices,
                                       idx_t **SFEM_RESTRICT coarse_indices);

int hierarchical_collect_coarse_values(const idx_t max_coarse_idx,
                                       const ptrdiff_t n_indices,
                                       idx_t *const SFEM_RESTRICT fine_indices,
                                       const real_t *const SFEM_RESTRICT fine_values,
                                       real_t *const SFEM_RESTRICT coarse_values);

int hierarchical_prolongation(const enum ElemType from_element,
                              const enum ElemType to_element,
                              const ptrdiff_t nelements,
                              idx_t **const SFEM_RESTRICT elements,
                              const real_t *const SFEM_RESTRICT from,
                              real_t *const SFEM_RESTRICT to);

int hierarchical_restriction(const enum ElemType from_element,
                             const enum ElemType to_element,
                             const ptrdiff_t nelements,
                             idx_t **const SFEM_RESTRICT elements,
                             const real_t *const SFEM_RESTRICT from,
                             real_t *const SFEM_RESTRICT to);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_PROLONGATION_RESTRICTION_H
