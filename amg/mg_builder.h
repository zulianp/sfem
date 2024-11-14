#ifndef MG_BUILDER_H
#define MG_BUILDER_H

#include "interpolation.h"
#include "sparse.h"

typedef struct {
    idx_t levels;
    real_t coarsening_factor;
    PiecewiseConstantTransfer **transer_operators;
    SymmCOOMatrix **matrices;
} PWCHierarchy;

int builder(const real_t coarsening_factor,
            const idx_t *free_dofs,
            SymmCOOMatrix *fine_mat,
            PWCHierarchy *hierarchy);

#endif  // MG_BUILDER_H
