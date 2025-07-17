#ifndef CU_CONTACT_SURFACE_H
#define CU_CONTACT_SURFACE_H

#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_displace_points(const int          dim,
                       const ptrdiff_t    n_nodes,
                       const idx_t *const idx,
                       geom_t **const     bulk_points,
                       const real_t      *disp,
                       geom_t **const     surface_points);

int cu_displace_surface_points(const int          dim,
                               const ptrdiff_t    n_nodes,
                               const idx_t *const idx,
                               geom_t **const     surface_points_rest,
                               const real_t      *disp,
                               geom_t **const     surface_points);

#ifdef __cplusplus
}
#endif

#endif  // CU_CONTACT_SURFACE_H
