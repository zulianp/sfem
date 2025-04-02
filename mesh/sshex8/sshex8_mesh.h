#ifndef SFEM_SSHEX8_MESH_H
#define SFEM_SSHEX8_MESH_H

#include <stddef.h>
#include <stdint.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int sshex8_fill_points(const int       level,
                       const ptrdiff_t nelements,
                       idx_t **const   elements,
                       geom_t **const  macro_mesh_points,
                       geom_t **const  points);

int sshex8_fill_points_1D_map(const int                           level,
                              const ptrdiff_t                     nelements,
                              idx_t **const SFEM_RESTRICT         elements,
                              geom_t **const SFEM_RESTRICT        macro_mesh_points,
                              const scalar_t *const SFEM_RESTRICT ref_points,
                              geom_t **const SFEM_RESTRICT        points);

int sshex8_to_standard_hex8_mesh(const int                   level,
                                 const ptrdiff_t             nelements,
                                 idx_t **const SFEM_RESTRICT elements,
                                 idx_t **const SFEM_RESTRICT hex8_elements);

// FIXME move to appropriate file
int ssquad4_to_standard_quad4_mesh(const int                   level,
                                   const ptrdiff_t             nelements,
                                   idx_t **const SFEM_RESTRICT elements,
                                   idx_t **const SFEM_RESTRICT quad4_elements);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_SSHEX8_MESH_H
