#ifndef BOUNDARY_CONDITION_IO_H
#define BOUNDARY_CONDITION_IO_H

#include <mpi.h>
#include <stddef.h>

#include "sfem_base.h"
#include "sfem_mesh.h"
#include "boundary_condition.h"

#ifdef __cplusplus
extern "C" {
#endif


void read_boundary_conditions(MPI_Comm comm,
                              const char *sets,
                              const char *values,
                              const char *components,
                              boundary_condition_t **bcs,
                              int *nbc);

void read_dirichlet_conditions(const mesh_t *const mesh,
                               const char *sets,
                               const char *values,
                               const char *components,
                               boundary_condition_t **bcs,
                               int *nbc);

void read_neumann_conditions(const mesh_t *const mesh,
                             const char *sets,
                             const char *values,
                             const char *components,
                             boundary_condition_t **bcs,
                             int *nbc);

#ifdef __cplusplus
}
#endif
#endif  // BOUNDARY_CONDITION_IO_H
