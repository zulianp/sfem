#include "boundary_condition.h"

#include "dirichlet.h"
#include "neumann.h"
#include "sfem_defs.h"

#include "matrixio_array.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

void read_boundary_conditions(MPI_Comm comm,
                              char *sets,
                              char *values,
                              char *components,
                              boundary_condition_t **bcs,
                              int *nbc) {
    if (!sets) {
        *bcs = NULL;
        *nbc = 0;
        return;
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    const char *splitter = ",";

    int count = 1;
    {
        int i = 0;
        while (sets[i]) {
            count += (sets[i++] == splitter[0]);
            assert(i <= strlen(sets));
        }
    }

    printf("conds = %d, splitter=%c\n", count, splitter[0]);

    boundary_condition_t *conds = malloc(count * sizeof(boundary_condition_t));

    // NODESET/SIDESET
    {
        char *pch = strtok(sets, splitter);
        int i = 0;
        while (pch != NULL) {
            printf("Reading file (%d/%d): %s\n", i + 1, count, pch);
            if (array_create_from_file(comm,
                                       pch,
                                       SFEM_MPI_IDX_T,
                                       (void **)&conds[i].idx,
                                       &conds[i].local_size,
                                       &conds[i].global_size)) {
                fprintf(stderr, "Failed to read file %s\n", pch);
                return;
            }

            // Some default values
            conds[i].value = 0;
            conds[i].component = 0;
            i++;
            pch = strtok(NULL, splitter);
        }
    }

    if (values) {
        static const char *path_key = "path:";
        const int path_key_len = strlen(path_key);

        char *pch = strtok(values, splitter);
        int i = 0;
        while (pch != NULL) {
            printf("Parsing  values (%d/%d): %s\n", i + 1, count, pch);

            if (strncmp(pch, path_key, path_key_len) == 0) {
                conds[i].value = 0;

                ptrdiff_t local_check_size, check_size;
                if (array_create_from_file(comm,
                                           pch + path_key_len,
                                           SFEM_MPI_REAL_T,
                                           (void **)&conds[i].values,
                                           &local_check_size,
                                           &check_size)) {
                    MPI_Abort(comm, -1);
                }

                assert(local_check_size == conds[i].local_size);
                assert(check_size == conds[i].global_size);
                if (local_check_size != conds[i].local_size) {
                    if (!rank) {
                        fprintf(stderr,
                                "Wrong size for boundary condition with values %s\n",
                                pch + path_key_len);
                    }
                }
            } else {
                conds[i].value = atof(pch);
                conds[i].values = 0;
            }
            i++;

            pch = strtok(NULL, splitter);
        }
    }

    if (components) {
        char *pch = strtok(components, splitter);
        int i = 0;
        while (pch != NULL) {
            printf("Parsing comps (%d/%d): %s\n", i + 1, count, pch);
            conds[i].component = atoi(pch);
            i++;

            pch = strtok(NULL, splitter);
        }
    }

    *bcs = conds;
    *nbc = count;
}

void read_dirichlet_conditions(const mesh_t *const mesh,
                               char *sets,
                               char *values,
                               char *components,
                               boundary_condition_t **bcs,
                               int *nbc) {
    read_boundary_conditions(mesh->comm, sets, values, components, bcs, nbc);
}

void read_neumann_conditions(const mesh_t *const mesh,
                             char *sets,
                             char *values,
                             char *components,
                             boundary_condition_t **bcs,
                             int *nbc) {
    read_boundary_conditions(mesh->comm, sets, values, components, bcs, nbc);

    enum ElemType stype = side_type(mesh->element_type);
    int nns = elem_num_sides(stype);

    for (int i = 0; i < *nbc; i++) {
        (*bcs)[i].global_size /= nns;
        (*bcs)[i].local_size /= nns;
    }
}

void add_neumann_condition_to_gradient_vec(const int n_conditions,
                                           const boundary_condition_t *const cond,
                                           const mesh_t *const mesh,
                                           const int block_size,
                                           real_t *g) {
    for (int i = 0; i < n_conditions; i++) {
        surface_forcing_function_vec(side_type(mesh->element_type),
                                     cond[i].local_size,
                                     cond[i].idx,
                                     mesh->points,
                                     -  // Use negative sign since we are on LHS
                                     cond[i].value,
                                     block_size,
                                     cond[i].component,
                                     g);
    }
}

void apply_dirichlet_condition_vec(const int n_conditions,
                                   const boundary_condition_t *const cond,
                                   const mesh_t *const mesh,
                                   const int block_size,
                                   real_t *const x) {
    for (int i = 0; i < n_conditions; i++) {
        if (cond[i].values) {
            constraint_nodes_to_values_vec(
                cond[i].local_size, cond[i].idx, block_size, cond[i].component, cond[i].values, x);
        } else {
            constraint_nodes_to_value_vec(
                cond[i].local_size, cond[i].idx, block_size, cond[i].component, cond[i].value, x);
        }
    }
}

void apply_zero_dirichlet_condition_vec(const int n_conditions,
                                        const boundary_condition_t *const cond,
                                        const mesh_t *const mesh,
                                        const int block_size,
                                        real_t *const x) {
    for (int i = 0; i < n_conditions; i++) {
        constraint_nodes_to_value_vec(
            cond[i].local_size, cond[i].idx, block_size, cond[i].component, 0, x);
    }
}

void copy_at_dirichlet_nodes_vec(const int n_conditions,
                                 const boundary_condition_t *const cond,
                                 const mesh_t *const mesh,
                                 const int block_size,
                                 const real_t *const in,
                                 real_t *const out) {
    for (int i = 0; i < n_conditions; i++) {
        constraint_nodes_copy_vec(
            cond[i].local_size, cond[i].idx, block_size, cond[i].component, in, out);
    }
}

void apply_dirichlet_condition_to_hessian_crs_vec(const int n_conditions,
                                                  const boundary_condition_t *const cond,
                                                  const mesh_t *const mesh,
                                                  const int block_size,
                                                  const count_t *const rowptr,
                                                  const idx_t *const colidx,
                                                  real_t *const values) {
    SFEM_UNUSED(mesh);

    for (int i = 0; i < n_conditions; i++) {
        crs_constraint_nodes_to_identity_vec(cond[i].local_size,
                                             cond[i].idx,
                                             block_size,
                                             cond[i].component,
                                             1,
                                             rowptr,
                                             colidx,
                                             values);
    }
}

void destroy_conditions(const int n_conditions, boundary_condition_t *cond) {
    for (int i = 0; i < n_conditions; i++) {
        free(cond[i].idx);
    }

    free(cond);
}
