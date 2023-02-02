#ifndef SFEM_DIRICHLET_H
#define SFEM_DIRICHLET_H

#include <stddef.h>
#include "sfem_base.h"

void constraint_nodes_to_value(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const real_t value,
    real_t *values
    );

void crs_constraint_nodes_to_identity(
    const ptrdiff_t n_dirichlet_nodes,
    const idx_t * dirichlet_nodes,
    const real_t diag_value,
    const idx_t *rowptr,
    const idx_t *colidx,
    real_t *values
    );

#endif //SFEM_DIRICHLET_H
