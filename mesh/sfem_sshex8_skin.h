#ifndef SFEM_SSHEX8_SKIN_H
#define SFEM_SSHEX8_SKIN_H

#include "sfem_base.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int sshex8_skin(const int       L,
                const ptrdiff_t nelements,
                idx_t         **elements,
                ptrdiff_t      *n_surf_elements,
                idx_t **const   surf_elements,
                element_idx_t **parent_element);

// Sideset is (sshex8 element id, macro_face local index)
int sshex8_sideset_to_ssquad4_surface(
    const int       L,
    const ptrdiff_t nelements,
    idx_t         **elements,
    const element_idx_t *parent_element,
    const int8_t *face_idx,
    ptrdiff_t      *n_surf_elements,
    idx_t **const   surf_elements);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_SSHEX8_SKIN_H
