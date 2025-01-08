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

#ifdef __cplusplus
}
#endif

#endif  // SFEM_SSHEX8_MESH_H
