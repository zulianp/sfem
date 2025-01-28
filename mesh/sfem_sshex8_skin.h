#ifndef SFEM_SSHEX8_SKIN_H
#define SFEM_SSHEX8_SKIN_H

#include <stddef.h>
#include <stdint.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int sshex8_skin(const int       L,
                const ptrdiff_t nelements,
                idx_t         **SFEM_RESTRICT elements,
                ptrdiff_t      *SFEM_RESTRICT n_surf_elements,
                idx_t **const   SFEM_RESTRICT surf_elements,
                element_idx_t **SFEM_RESTRICT parent_element);

int sshex8_surface_from_sideset(const int                                L,
                                const ptrdiff_t                          nelements,
                                idx_t **const SFEM_RESTRICT              elements,
                                const ptrdiff_t                          n_surf_elements,
                                const element_idx_t *const SFEM_RESTRICT parents,
                                const int16_t *const SFEM_RESTRICT       side_idx,
                                idx_t **SFEM_RESTRICT                    sides);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_SSHEX8_SKIN_H
