#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <sys/stat.h>
#include <sys/types.h>
// #include <unistd.h>

#include "utils.h"


#include "sfem_base.hpp"
#include "sfem_defs.hpp"
#include "sfem_vec.hpp"
#include "sortreduce.hpp"

#include "mass.hpp"

#include "dirichlet.hpp"
#include "neumann.hpp"


#include "stokes_mini.hpp"

#include "smesh_glob.hpp"
#include "sfem_API.hpp"

////////////////////////////////////////////////////////////////////////////////////
// The MINI mixed finite element for the Stokes problem: An experimental
// investigation
////////////////////////////////////////////////////////////////////////////////////
static SFEM_INLINE real_t rhs1_x(const real_t mu, const real_t x, const real_t y) {
    return -mu * (4 * y * (1 - y) * (2 * y - 1) * ((1 - 2 * x) * (1 - 2 * x) - 2 * x * (1 - x)) +
                  12 * x * x * (1 - x) * (1 - x) * (1 - 2 * y)) +
           (1 - 2 * x) * (1 - y);
}
static SFEM_INLINE real_t rhs1_y(const real_t mu, const real_t x, const real_t y) {
    return -mu * (4 * x * (1 - x) * (1 - 2 * x) * ((1 - 2 * y) * (1 - 2 * y) - 2 * y * (1 - y)) +
                  12 * y * y * (1 - y) * (1 - y) * (2 * x - 1)) -
           x * (1 - x);
}

static SFEM_INLINE real_t rhs2_x(const real_t mu, const real_t x, const real_t y) {
    return -mu * ((2 - 12 * x + 12 * x * x) * (2 * y - 6 * y * y + 4 * y * y * y) +
                  (x * x - 2 * x * x * x + x * x * x * x) * (-12 + 24 * y)) +
           1. / 24;
}

static SFEM_INLINE real_t rhs2_y(const real_t mu, const real_t x, const real_t y) {
    return mu * ((2 - 12 * y + 12 * y * y) * (2 * x - 6 * x * x + 4 * x * x * x) +
                 (y * y - 2 * y * y * y + y * y * y * y) * (-12 + 24 * x)) +
           1. / 24;
}

static SFEM_INLINE real_t rhs3_x(const real_t mu, const real_t x, const real_t y) {
    const real_t pis4 = 4 * M_PI * M_PI;
    const real_t pi2 = 2 * M_PI;
    return -pis4 * mu * sin(pi2 * y) * (2 * cos(pi2 * x) - 1) + pis4 * sin(pi2 * x);
}

static SFEM_INLINE real_t rhs3_y(const real_t mu, const real_t x, const real_t y) {
    const real_t pis4 = 4 * M_PI * M_PI;
    const real_t pi2 = 2 * M_PI;
    return pis4 * mu * sin(pi2 * x) * (2 * cos(pi2 * y) - 1) - pis4 * sin(pi2 * y);
}

static SFEM_INLINE real_t rhs4_x(const real_t mu, const real_t x, const real_t y, const real_t z) {
    const real_t pis4 = 4 * M_PI * M_PI;
    const real_t pi2 = 2 * M_PI;
    return -pis4 * mu * sin(pi2 * y) * (2 * cos(pi2 * x) - 1) + pis4 * sin(pi2 * x);
}

static SFEM_INLINE real_t rhs4_y(const real_t mu, const real_t x, const real_t y, const real_t z) {
    const real_t pis4 = 4 * M_PI * M_PI;
    const real_t pi2 = 2 * M_PI;
    return pis4 * mu * sin(pi2 * x) * (2 * cos(pi2 * y) - 1) - pis4 * sin(pi2 * y);
}

static SFEM_INLINE real_t rhs4_z(const real_t mu, const real_t x, const real_t y, const real_t z) {
    const real_t pis4 = 4 * M_PI * M_PI;
    const real_t pi2 = 2 * M_PI;
    return pis4 * mu * sin(pi2 * z) * (2 * cos(pi2 * x) - 1) - pis4 * sin(pi2 * y);
}

static SFEM_INLINE real_t rhs5_x(const real_t mu, const real_t x, const real_t y, const real_t z) {
    return 10000 * (x * x + y * y + z * z);
}

////////////////////////////////////////////////////////////////////////////////////

static void node_eval_f2D(const ptrdiff_t nnodes,
                          geom_t **const points,
                          const real_t mu,
                          real_t (*f)(const real_t, const real_t, const real_t),
                          real_t *values) {
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        values[i] = f(mu, points[0][i], points[1][i]);
    }
}

static void node_eval_f3D(const ptrdiff_t nnodes,
                          geom_t **const points,
                          const real_t mu,
                          real_t (*f)(const real_t, const real_t, const real_t, const real_t),
                          real_t *values) {
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        values[i] = f(mu, points[0][i], points[1][i], points[2][i]);
    }
}

//////////////////////////////////////////////

int solve_stokes(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 3) {
        fprintf(stderr, "usage: %s <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];
    smesh::create_directory(output_folder);

    const double tick = smesh::time_seconds();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    auto mesh = sfem::Mesh::create_from_file(comm, smesh::Path(folder));

    // Optional params
    real_t SFEM_MU = 1;
    real_t SFEM_RHO = 1;
    int SFEM_PROBLEM_TYPE = 1;
    int SFEM_AOS = 0;
    const char *SFEM_DIRICHLET_NODES = 0;

    SFEM_READ_ENV(SFEM_PROBLEM_TYPE, atoi);
    SFEM_READ_ENV(SFEM_MU, atof);
    SFEM_READ_ENV(SFEM_RHO, atof);
    SFEM_READ_ENV(SFEM_AOS, atoi);
    SFEM_READ_ENV(SFEM_DIRICHLET_NODES, );

    if (!comm->rank()) {
        printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_PROBLEM_TYPE=%d\n"
            "- SFEM_MU=%g\n"
            "- SFEM_RHO=%g\n"
            "- SFEM_DIRICHLET_NODES=%s\n"
            "----------------------------------------\n",
            SFEM_PROBLEM_TYPE,
            SFEM_MU,
            SFEM_RHO,
            SFEM_DIRICHLET_NODES);
    }

    double tack = smesh::time_seconds();
    printf("stokes.c: read\t\t%g seconds\n", tack - tick);

    ptrdiff_t nnz = 0;
    count_t *rowptr = 0;
    idx_t *colidx = 0;
    smesh::create_crs_graph_for_elem_type(
        mesh->element_type(0), mesh->n_elements(), mesh->n_nodes(), mesh->elements(0)->data(), &rowptr, &colidx);
    nnz = rowptr[mesh->n_nodes()];

    double tock = smesh::time_seconds();
    printf("stokes.c: build crs graph\t\t%g seconds\n", tock - tack);
    tack = tock;

    const int sdim = elem_manifold_dim(mesh->element_type(0));
    const int n_vars = sdim + 1;

    real_t *rhs_values[4] = {0, 0, 0, 0};
    switch (SFEM_PROBLEM_TYPE) {
        case 1: {
            rhs_values[0] = (real_t*)calloc(mesh->n_nodes(), sizeof(real_t));
            rhs_values[1] = (real_t*)calloc(mesh->n_nodes(), sizeof(real_t));
            node_eval_f2D(mesh->n_nodes(), mesh->points()->data(), SFEM_MU, &rhs1_x, rhs_values[0]);
            node_eval_f2D(mesh->n_nodes(), mesh->points()->data(), SFEM_MU, &rhs1_y, rhs_values[1]);
            break;
        }
        case 2: {
            rhs_values[0] = (real_t*)calloc(mesh->n_nodes(), sizeof(real_t));
            rhs_values[1] = (real_t*)calloc(mesh->n_nodes(), sizeof(real_t));
            node_eval_f2D(mesh->n_nodes(), mesh->points()->data(), SFEM_MU, &rhs2_x, rhs_values[0]);
            node_eval_f2D(mesh->n_nodes(), mesh->points()->data(), SFEM_MU, &rhs2_y, rhs_values[1]);
            break;
        }
        case 3: {
            rhs_values[0] = (real_t*)calloc(mesh->n_nodes(), sizeof(real_t));
            rhs_values[1] = (real_t*)calloc(mesh->n_nodes(), sizeof(real_t));
            node_eval_f2D(mesh->n_nodes(), mesh->points()->data(), SFEM_MU, &rhs3_x, rhs_values[0]);
            node_eval_f2D(mesh->n_nodes(), mesh->points()->data(), SFEM_MU, &rhs3_y, rhs_values[1]);
            break;
        }
        case 4: {
            rhs_values[0] = (real_t*)calloc(mesh->n_nodes(), sizeof(real_t));
            rhs_values[1] = (real_t*)calloc(mesh->n_nodes(), sizeof(real_t));
            rhs_values[2] = (real_t*)calloc(mesh->n_nodes(), sizeof(real_t));
            node_eval_f3D(mesh->n_nodes(), mesh->points()->data(), SFEM_MU, &rhs4_x, rhs_values[0]);
            node_eval_f3D(mesh->n_nodes(), mesh->points()->data(), SFEM_MU, &rhs4_y, rhs_values[1]);
            node_eval_f3D(mesh->n_nodes(), mesh->points()->data(), SFEM_MU, &rhs4_z, rhs_values[2]);
            break;
        }
        case 5: {
            rhs_values[0] = (real_t*)calloc(mesh->n_nodes(), sizeof(real_t));
            node_eval_f3D(mesh->n_nodes(), mesh->points()->data(), SFEM_MU, &rhs5_x, rhs_values[0]);
            break;
        }
        default: {
            break;
        }
    }

    if (SFEM_AOS) {
        real_t *values = (real_t*)calloc(n_vars * n_vars * nnz, sizeof(real_t));
        real_t *rhs = (real_t*)calloc(n_vars * mesh->n_nodes(), sizeof(real_t));

        ///////////////////////////////////////////////////////////////////////////////
        // Operator assembly
        ///////////////////////////////////////////////////////////////////////////////

        stokes_mini_assemble_hessian_aos(mesh->element_type(0),
                                         mesh->n_elements(),
                                         mesh->n_nodes(),
                                         mesh->elements(0)->data(),
                                         mesh->points()->data(),
                                         SFEM_MU,
                                         rowptr,
                                         colidx,
                                         values);

        stokes_mini_assemble_rhs_aos(mesh->element_type(0),
                                     mesh->n_elements(),
                                     mesh->n_nodes(),
                                     mesh->elements(0)->data(),
                                     mesh->points()->data(),
                                     SFEM_MU,
                                     SFEM_RHO,
                                     rhs_values,
                                     rhs);

        count_t *b_rowptr = (count_t *)malloc((mesh->n_nodes() + 1) * n_vars * sizeof(count_t));
        idx_t *b_colidx = (idx_t *)malloc((ptrdiff_t)rowptr[mesh->n_nodes()] * n_vars * n_vars * sizeof(idx_t));
        smesh::crs_graph_block_to_scalar(mesh->n_nodes(), n_vars, rowptr, colidx, b_rowptr, b_colidx);

        if (SFEM_DIRICHLET_NODES) {
            auto dirichlet_buf = sfem::Buffer<idx_t>::from_file(smesh::Path(SFEM_DIRICHLET_NODES));
            if (!dirichlet_buf) {
                SFEM_ERROR("Failed to read dirichlet nodes from %s\n", SFEM_DIRICHLET_NODES);
            }
            idx_t    *dirichlet_nodes = dirichlet_buf->data();
            ptrdiff_t nn              = dirichlet_buf->size();

            for (int d = 0; d < sdim; d++) {
                constraint_nodes_to_value_vec(nn, dirichlet_nodes, n_vars, d, 0, rhs);
            }

            for (int d1 = 0; d1 < sdim; d1++) {
                crs_constraint_nodes_to_identity_vec(
                    nn, dirichlet_nodes, n_vars, d1, 1, b_rowptr, b_colidx, values);
            }

            if (0) {
                // One point to 0 to fix pressure degree of freedom
                // ptrdiff_t node = nn - 1;
                ptrdiff_t node = 0;
                crs_constraint_nodes_to_identity_vec(
                    1, &dirichlet_nodes[node], n_vars, (n_vars - 1), 1, b_rowptr, b_colidx, values);

                constraint_nodes_to_value_vec(
                    1, &dirichlet_nodes[node], n_vars, n_vars - 1, 0, rhs);
            }

        } else {
            assert(0);
        }

        {
            const ptrdiff_t scalar_rows = mesh->n_nodes() * n_vars;
            const count_t   scalar_nnz  = b_rowptr[scalar_rows];
            auto rowptr_buf = sfem::Buffer<count_t>::wrap(scalar_rows + 1, b_rowptr);
            auto colidx_buf = sfem::Buffer<idx_t>::wrap(scalar_nnz, b_colidx);
            auto values_buf = sfem::Buffer<real_t>::wrap(scalar_nnz, values);
            auto crs        = sfem::h_crs_spmv<count_t, idx_t, real_t>(
                    scalar_rows, scalar_rows, rowptr_buf, colidx_buf, values_buf, (real_t)1);
            crs->to_file(smesh::Path(output_folder));
        }

        {
            char path[1024 * 10];
            snprintf(path, sizeof(path), "%s/rhs.raw", output_folder);
            sfem::Buffer<real_t>::wrap(mesh->n_nodes() * n_vars, rhs)->to_file(smesh::Path(path));
        }

        free(b_rowptr);
        free(b_colidx);
        free(values);
        free(rhs);
    } else {
        real_t **values = 0;
        values = (real_t **)malloc((n_vars * n_vars) * sizeof(real_t *));
        for (int d = 0; d < (n_vars * n_vars); d++) {
            values[d] = (real_t*)calloc(nnz, sizeof(real_t));
        }

        real_t **rhs = 0;
        rhs = (real_t **)malloc((n_vars) * sizeof(real_t *));
        for (int d = 0; d < n_vars; d++) {
            rhs[d] = (real_t*)calloc(mesh->n_nodes(), sizeof(real_t));
        }

        ///////////////////////////////////////////////////////////////////////////////
        // Operator assembly
        ///////////////////////////////////////////////////////////////////////////////

        stokes_mini_assemble_hessian_soa(mesh->element_type(0),
                                         mesh->n_elements(),
                                         mesh->n_nodes(),
                                         mesh->elements(0)->data(),
                                         mesh->points()->data(),
                                         SFEM_MU,
                                         rowptr,
                                         colidx,
                                         values);

        if (0) {
            // No static condensation contribution on RHS
            for (int i = 0; i < n_vars; i++) {
                if (rhs_values[i]) {
                    apply_mass(mesh->element_type(0),
                               mesh->n_elements(),
                               mesh->n_nodes(),
                               mesh->elements(0)->data(),
                               mesh->points()->data(),
                               1,
                               rhs_values[i],
                               1,
                               rhs[i]);
                }
            }
        } else {
            stokes_mini_assemble_rhs_soa(mesh->element_type(0),
                                         mesh->n_elements(),
                                         mesh->n_nodes(),
                                         mesh->elements(0)->data(),
                                         mesh->points()->data(),
                                         SFEM_MU,
                                         SFEM_RHO,
                                         rhs_values,
                                         rhs);
        }

        tock = smesh::time_seconds();
        printf("stokes.c: assembly\t\t%g seconds\n", tock - tack);
        tack = tock;

        ///////////////////////////////////////////////////////////////////////////////
        // Boundary conditions
        ///////////////////////////////////////////////////////////////////////////////

        if (SFEM_DIRICHLET_NODES) {
            auto dirichlet_buf = sfem::Buffer<idx_t>::from_file(smesh::Path(SFEM_DIRICHLET_NODES));
            if (!dirichlet_buf) {
                SFEM_ERROR("Failed to read dirichlet nodes from %s\n", SFEM_DIRICHLET_NODES);
            }
            idx_t    *dirichlet_nodes = dirichlet_buf->data();
            ptrdiff_t nn              = dirichlet_buf->size();

            for (int d = 0; d < sdim; d++) {
                constraint_nodes_to_value(nn, dirichlet_nodes, 0, rhs[d]);
            }

            for (int d1 = 0; d1 < sdim; d1++) {
                for (int d2 = 0; d2 < n_vars; d2++) {
                    crs_constraint_nodes_to_identity(
                        nn, dirichlet_nodes, d1 == d2, rowptr, colidx, values[d1 * n_vars + d2]);
                }
            }

            if (0) {
                // One point to 0 to fix pressure degree of freedom
                // ptrdiff_t node = nn - 1;
                ptrdiff_t node = 0;
                for (int d2 = 0; d2 < n_vars; d2++) {
                    crs_constraint_nodes_to_identity(1,
                                                     &dirichlet_nodes[node],
                                                     (n_vars - 1) == d2,
                                                     rowptr,
                                                     colidx,
                                                     values[(n_vars - 1) * n_vars + d2]);
                }

                constraint_nodes_to_value(1, &dirichlet_nodes[node], 0, rhs[n_vars - 1]);
            }

        } else {
            assert(0);
        }

        tock = smesh::time_seconds();
        printf("stokes.c: boundary\t\t%g seconds\n", tock - tack);
        tack = tock;

        ///////////////////////////////////////////////////////////////////////////////
        // Write to disk
        ///////////////////////////////////////////////////////////////////////////////

        {
            char path_rowptr[1024 * 10];
            snprintf(path_rowptr, sizeof(path_rowptr), "%s/rowptr.raw", output_folder);

            char path_colidx[1024 * 10];
            snprintf(path_colidx, sizeof(path_colidx), "%s/colidx.raw", output_folder);

            sfem::Buffer<count_t>::wrap(mesh->n_nodes() + 1, rowptr)->to_file(smesh::Path(path_rowptr));
            sfem::Buffer<idx_t>::wrap(nnz, colidx)->to_file(smesh::Path(path_colidx));

            char path[1024 * 10];
            for (int d = 0; d < n_vars * n_vars; d++) {
                snprintf(path, sizeof(path), "%s/values.%d.raw", output_folder, d);
                sfem::Buffer<real_t>::wrap(nnz, values[d])->to_file(smesh::Path(path));
            }
        }

        {
            char path[1024 * 10];
            for (int d = 0; d < n_vars; d++) {
                snprintf(path, sizeof(path), "%s/rhs.%d.raw", output_folder, d);
                sfem::Buffer<real_t>::wrap(mesh->n_nodes(), rhs[d])->to_file(smesh::Path(path));
            }
        }

        tock = smesh::time_seconds();
        printf("stokes.c: write\t\t%g seconds\n", tock - tack);
        tack = tock;

        ///////////////////////////////////////////////////////////////////////////////
        // Free resources
        ///////////////////////////////////////////////////////////////////////////////

        for (int d = 0; d < (n_vars * n_vars); d++) {
            free(values[d]);
        }

        free(values);

        for (int d = 0; d < n_vars; d++) {
            free(rhs[d]);
        }

        free(rhs);

        for (int d = 0; d < n_vars; d++) {
            if (rhs_values[d]) {
                free(rhs_values[d]);
            }
        }
    }

    // Mesh n2n graph
    free(rowptr);
    free(colidx);

    ptrdiff_t nelements = mesh->n_elements();
    ptrdiff_t nnodes = mesh->n_nodes();



    tock = smesh::time_seconds();
    if (!comm->rank()) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld\n", (long)nelements, (long)nnodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_stokes(ctx->communicator(), argc, argv);
}
