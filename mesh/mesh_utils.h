#ifndef SFEM_MESH_UTILS_H
#define SFEM_MESH_UTILS_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

idx_t** allocate_elements(const int nxe, const ptrdiff_t n_elements);
void free_elements(const int nxe, idx_t** elements);
void select_elements(const int nxe,
                     const ptrdiff_t nselected,
                     const element_idx_t* const idx,
                     idx_t** const SFEM_RESTRICT elements,
                     idx_t** const SFEM_RESTRICT selection);

geom_t** allocate_points(const int dim, const ptrdiff_t n_points);
void free_points(const int dim, geom_t** points);
void select_points(const int dim,
                   const ptrdiff_t n_points,
                   const idx_t* idx,
                   geom_t** const points,
                   geom_t** const selection);

void minmax(const ptrdiff_t n, const geom_t* const SFEM_RESTRICT x, geom_t* xmin, geom_t* xmax);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_MESH_UTILS_H
