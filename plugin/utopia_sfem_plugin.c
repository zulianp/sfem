#include "utopia_plugin_Function.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../matrix.io/array_dtof.h"
#include "../../matrix.io/matrixio_array.h"
#include "../../matrix.io/matrixio_crs.h"
#include "../../matrix.io/utils.h"

#include "crs_graph.h"
#include "sfem_base.h"

#include "laplacian.h"

#include "read_mesh.h"
#include "sfem_mesh.h"

typedef struct {
    mesh_t *mesh;
    int block_size;
} sfem_problem_t;

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_destroy_array(const plugin_Function_t *info, void *ptr) {
    free(ptr);
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_create_array(const plugin_Function_t *info, size_t size, void **ptr) {
    *ptr = malloc(size);
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_init(plugin_Function_t *info) {
    const char *SFEM_MESH_DIR = "[error] undefined";
    const char *SFEM_MATERIAL_MODEL = "[error] undefined";

    SFEM_READ_ENV(SFEM_MESH_DIR, );
    SFEM_READ_ENV(SFEM_MATERIAL_MODEL, );

    printf("sfem:\n - SFEM_MESH_DIR=%s\n - SFEM_MATERIAL_MODEL%s\n", SFEM_MESH_DIR, SFEM_MATERIAL_MODEL);

    // if (!SFEM_MESH_DIR || !SFEM_MATERIAL_MODEL) {
    //     return UTOPIA_PLUGIN_FAILURE;
    // }

    if (!SFEM_MESH_DIR) {
        return UTOPIA_PLUGIN_FAILURE;
    }

    mesh_t *mesh = (mesh_t *)malloc(sizeof(mesh_t));

    if (read_mesh(info->comm, SFEM_MESH_DIR, mesh)) {
        return UTOPIA_PLUGIN_FAILURE;
    }

    sfem_problem_t *problem = (sfem_problem_t *)malloc(sizeof(sfem_problem_t));
    problem->mesh = mesh;
    problem->block_size = 1;

    // Store problem
    info->user_data = (void *)problem;

    // Eventually initialize info->user_data here
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_create_crs_graph(const plugin_Function_t *info,
                                                                 ptrdiff_t *nlocal,
                                                                 ptrdiff_t *nglobal,
                                                                 ptrdiff_t *nnz,
                                                                 plugin_idx_t **rowptr,
                                                                 plugin_idx_t **colidx) {
    sfem_problem_t *problem = (sfem_problem_t *)info->user_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    build_crs_graph(mesh->nelements, mesh->nnodes, mesh->elements, rowptr, colidx);

    *nlocal = mesh->nnodes;
    *nglobal = mesh->nnodes;
    *nnz = (*rowptr)[*nlocal] * (problem->block_size * problem->block_size);
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_create_vector(const plugin_Function_t *info,
                                                              ptrdiff_t *nlocal,
                                                              ptrdiff_t *nglobal,
                                                              plugin_scalar_t **values) {
    sfem_problem_t *problem = (sfem_problem_t *)info->user_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    const size_t nbytes = mesh->nnodes * sizeof(plugin_scalar_t);

    *values = (plugin_scalar_t *)malloc(nbytes);
    memset(*values, 0, nbytes);
    *nlocal = mesh->nnodes;
    *nglobal = mesh->nnodes;
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_destroy_vector(const plugin_Function_t *info, plugin_scalar_t *values) {
    free(values);
    return UTOPIA_PLUGIN_SUCCESS;
}

// Optimization function
int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_value(const plugin_Function_t *info,
                                                      const plugin_scalar_t *x,
                                                      plugin_scalar_t *const out) {
    assert(false && "TODO");
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_gradient(const plugin_Function_t *info,
                                                         const plugin_scalar_t *const x,
                                                         plugin_scalar_t *const out) {
    assert(false && "TODO");
    return UTOPIA_PLUGIN_SUCCESS;
}

// We might want to have also other formats here
int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_hessian_crs(const plugin_Function_t *info,
                                                            const plugin_scalar_t *const x,
                                                            const plugin_idx_t *const rowptr,
                                                            const plugin_idx_t *const colidx,
                                                            plugin_scalar_t *const values) {
    sfem_problem_t *problem = (sfem_problem_t *)info->user_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    assemble_laplacian(mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, rowptr, colidx, values);
    return UTOPIA_PLUGIN_SUCCESS;
}

// Operator
int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_apply(const plugin_Function_t *info,
                                                      const plugin_scalar_t *const x,
                                                      plugin_scalar_t *const out) {
    assert(false && "TODO");
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_apply_constraints(const plugin_Function_t *info,
                                                                  plugin_scalar_t *const x) {
    assert(false && "TODO");
    // No constraints for this example
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_apply_zero_constraints(const plugin_Function_t *info,
                                                                       plugin_scalar_t *const x) {
    assert(false && "TODO");
    // No constraints for this example
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_copy_constrained_dofs(const plugin_Function_t *info,
                                                                      const plugin_scalar_t *const src,
                                                                      plugin_scalar_t *const dest) {
    assert(false && "TODO");
    // No constraints for this example
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_report_solution(const plugin_Function_t *info,
                                                                const plugin_scalar_t *const x) {
    assert(false && "TODO");
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_destroy(plugin_Function_t *info) {
    // No user-data for this example
    assert(false && "TODO");
    return UTOPIA_PLUGIN_SUCCESS;
}
