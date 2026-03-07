#ifndef BOUNDARY_CONDITION_IO_H
#define BOUNDARY_CONDITION_IO_H

#include <mpi.h>
#include <stddef.h>

#include "sfem_base.hpp"
#include "smesh_mesh.hpp"
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

// void read_dirichlet_conditions(const sfem::Mesh *const mesh,
//                                const char *sets,
//                                const char *values,
//                                const char *components,
//                                boundary_condition_t **bcs,
//                                int *nbc);

// void read_neumann_conditions(const sfem::Mesh *const mesh,
//                              const char *sets,
//                              const char *values,
//                              const char *components,
//                              boundary_condition_t **bcs,
//                              int *nbc);

#ifdef __cplusplus
}
#endif
#endif  // BOUNDARY_CONDITION_IO_H
