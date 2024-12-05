#include "boundary_condition_io.h"
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
                              const char *sets,
                              const char *values,
                              const char *components,
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
        const char *pch = strtok(sets, splitter);
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

        const char *pch = strtok(values, splitter);
        int i = 0;
        while (pch != NULL) {
            printf("Parsing  values (%d/%d): %s\n", i + 1, count, pch);
            assert(i < count);

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

                if (local_check_size != conds[i].local_size) {
                    if (!rank) {
                        fprintf(stderr,
                                "read_boundary_conditions: len(idx) != len(values) (%ld != "
                                "%ld)\nfile:%s\n",
                                (long)conds[i].local_size,
                                (long)local_check_size,
                                pch + path_key_len);
                    }
                }

                assert(local_check_size == conds[i].local_size);
                assert(check_size == conds[i].global_size);

            } else {
                conds[i].value = atof(pch);
                conds[i].values = 0;
            }
            i++;

            pch = strtok(NULL, splitter);
        }
    }

    if (components) {
        const char *pch = strtok(components, splitter);
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
                               const char *sets,
                               const char *values,
                               const char *components,
                               boundary_condition_t **bcs,
                               int *nbc) {
    read_boundary_conditions(mesh->comm, sets, values, components, bcs, nbc);
}

void read_neumann_conditions(const mesh_t *const mesh,
                             const char *sets,
                             const char *values,
                             const char *components,
                             boundary_condition_t **bcs,
                             int *nbc) {
    read_boundary_conditions(mesh->comm, sets, values, components, bcs, nbc);

    enum ElemType stype = side_type(mesh->element_type);
    int nns = elem_num_nodes(stype);

    for (int i = 0; i < *nbc; i++) {
        (*bcs)[i].global_size /= nns;
        (*bcs)[i].local_size /= nns;
    }
}
