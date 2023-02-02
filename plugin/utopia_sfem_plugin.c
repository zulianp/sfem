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

#include "dirichlet.h"
#include "neumann.h"

typedef struct {
    mesh_t *mesh;
    idx_t *dirichlet_nodes;
    ptrdiff_t nlocal_dirchlet;
    ptrdiff_t nglobal_dirchlet;

    ptrdiff_t nlocal_neumann;
    ptrdiff_t nglobal_neumann;
    idx_t *faces_neumann;

    int block_size;

    const char *output_dir;
} sfem_problem_t;

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_destroy_array(const plugin_Function_t *info, void *ptr) {
    free(ptr);
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_create_array(const plugin_Function_t *info, size_t size, void **ptr) {
    *ptr = malloc(size);
    memset(*ptr, 0, size);
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_init(plugin_Function_t *info) {
    const char *SFEM_MESH_DIR = "[error] undefined";
    const char *SFEM_MATERIAL_MODEL = "[error] undefined";
    const char *SFEM_DIRICHLET_NODES = "[error] undefined";
    const char *SFEM_OUTPUT_DIR = "./sfem_output";
    const char *SFEM_NEUMAN_FACES = "[error] undefined";

    SFEM_READ_ENV(SFEM_MESH_DIR, );
    SFEM_READ_ENV(SFEM_MATERIAL_MODEL, );
    SFEM_READ_ENV(SFEM_DIRICHLET_NODES, );
    SFEM_READ_ENV(SFEM_OUTPUT_DIR, );
    SFEM_READ_ENV(SFEM_NEUMAN_FACES, );

    printf(
        "sfem:\n"
        "- SFEM_MESH_DIR=%s\n"
        "- SFEM_MATERIAL_MODEL%s\n"
        "- SFEM_DIRICHLET_NODES=%s\n"
        "- SFEM_OUTPUT_DIR=%s\n"
        "- SFEM_NEUMAN_FACES=%s\n",
        SFEM_MESH_DIR,
        SFEM_MATERIAL_MODEL,
        SFEM_DIRICHLET_NODES,
        SFEM_OUTPUT_DIR,
        SFEM_NEUMAN_FACES);

    if (!SFEM_MESH_DIR || !SFEM_DIRICHLET_NODES || !SFEM_NEUMAN_FACES) {
        return UTOPIA_PLUGIN_FAILURE;
    }

    mesh_t *mesh = (mesh_t *)malloc(sizeof(mesh_t));

    if (read_mesh(info->comm, SFEM_MESH_DIR, mesh)) {
        return UTOPIA_PLUGIN_FAILURE;
    }

    sfem_problem_t *problem = (sfem_problem_t *)malloc(sizeof(sfem_problem_t));

    if (array_read(info->comm,
                   SFEM_DIRICHLET_NODES,
                   SFEM_MPI_IDX_T,
                   (void **)&problem->dirichlet_nodes,
                   &problem->nlocal_dirchlet,
                   &problem->nglobal_dirchlet)) {
        return UTOPIA_PLUGIN_FAILURE;
    }

    if (array_read(info->comm,
                   SFEM_NEUMAN_FACES,
                   SFEM_MPI_IDX_T,
                   (void **)&problem->faces_neumann,
                   &problem->nlocal_neumann,
                   &problem->nglobal_neumann)) {
        return UTOPIA_PLUGIN_FAILURE;
    }

    problem->nlocal_neumann /= 3;
    problem->nglobal_neumann /= 3;

    // Redistribute Dirichlet nodes
    problem->mesh = mesh;
    problem->block_size = 1;
    problem->output_dir = SFEM_OUTPUT_DIR;

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
    sfem_problem_t *problem = (sfem_problem_t *)info->user_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    laplacian_assemble_value(mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, x, out);
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_gradient(const plugin_Function_t *info,
                                                         const plugin_scalar_t *const x,
                                                         plugin_scalar_t *const out) {
    sfem_problem_t *problem = (sfem_problem_t *)info->user_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    laplacian_assemble_gradient(mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, x, out);
    surface_forcing_function(problem->nlocal_neumann, problem->faces_neumann, mesh->points, -1, out);

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

    laplacian_assemble_hessian(mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, rowptr, colidx, values);

    crs_constraint_nodes_to_identity(problem->nlocal_dirchlet, problem->dirichlet_nodes, 1.0, rowptr, colidx, values);
    return UTOPIA_PLUGIN_SUCCESS;
}

// Operator
int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_apply(const plugin_Function_t *info,
                                                      const plugin_scalar_t *const x,
                                                      const plugin_scalar_t *const h,
                                                      plugin_scalar_t *const out) {
    sfem_problem_t *problem = (sfem_problem_t *)info->user_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    // Equivalent to operator application due to linearity of the problem
    laplacian_assemble_gradient(mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, h, out);
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_apply_constraints(const plugin_Function_t *info,
                                                                  plugin_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->user_data;
    assert(problem);

    constraint_nodes_to_value(problem->nlocal_dirchlet, problem->dirichlet_nodes, 0, x);
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_apply_zero_constraints(const plugin_Function_t *info,
                                                                       plugin_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->user_data;
    assert(problem);

    constraint_nodes_to_value(problem->nlocal_dirchlet, problem->dirichlet_nodes, 0, x);
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_copy_constrained_dofs(const plugin_Function_t *info,
                                                                      const plugin_scalar_t *const src,
                                                                      plugin_scalar_t *const dest) {
    sfem_problem_t *problem = (sfem_problem_t *)info->user_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    constraint_nodes_copy(problem->nlocal_dirchlet, problem->dirichlet_nodes, src, dest);

    // No constraints for this example
    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_report_solution(const plugin_Function_t *info,
                                                                const plugin_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->user_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    char path[2048];
    sprintf(path, "%s/out.raw", problem->output_dir);

    if (array_write(info->comm,
                    path,
                    SFEM_MPI_REAL_T,
                    x,
                    mesh->nnodes * problem->block_size,
                    mesh->nnodes * problem->block_size)) {
        return UTOPIA_PLUGIN_FAILURE;
    }

    return UTOPIA_PLUGIN_SUCCESS;
}

int UTOPIA_PLUGIN_EXPORT utopia_plugin_Function_destroy(plugin_Function_t *info) {
    sfem_problem_t *problem = (sfem_problem_t *)info->user_data;
    free(problem->mesh);
    free(problem->dirichlet_nodes);
    free(problem->faces_neumann);
    return UTOPIA_PLUGIN_SUCCESS;
}
