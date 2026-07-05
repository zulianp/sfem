#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array_dtof.h"
#include "utils.h"

#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#include "laplacian.hpp"
#include "mass.hpp"

#include "dirichlet.hpp"
#include "neumann.hpp"

#include "cvfem_tri3_diffusion.hpp"
#include "tet4_laplacian.hpp"

#include "sfem_API.hpp"

int solve_cvfem_assemble(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (comm->size() != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 3) {
        fprintf(stderr, "usage: %s <folder> <output_folder>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    printf("%s %s %s\n", argv[0], argv[1], output_folder);

    int SFEM_LAPLACIAN          = 1;
    int SFEM_HANDLE_DIRICHLET   = 1;
    int SFEM_HANDLE_NEUMANN     = 0;
    int SFEM_HANDLE_RHS         = 0;
    int SFEM_EXPORT_FP32        = 0;

    SFEM_READ_ENV(SFEM_LAPLACIAN, atoi);
    SFEM_READ_ENV(SFEM_HANDLE_DIRICHLET, atoi);
    SFEM_READ_ENV(SFEM_EXPORT_FP32, atoi);
    SFEM_READ_ENV(SFEM_HANDLE_NEUMANN, atoi);
    SFEM_READ_ENV(SFEM_HANDLE_RHS, atoi);

    printf("----------------------------------------\n");
    printf("Environment variables:\n- SFEM_LAPLACIAN=%d\n- SFEM_HANDLE_DIRICHLET=%d\n- "
           "SFEM_HANDLE_NEUMANN=%d\n- SFEM_HANDLE_RHS=%d\n- SFEM_EXPORT_FP32=%d\n",
           SFEM_LAPLACIAN,
           SFEM_HANDLE_DIRICHLET,
           SFEM_HANDLE_NEUMANN,
           SFEM_HANDLE_RHS,
           SFEM_EXPORT_FP32);
    printf("----------------------------------------\n");

    const double tick = smesh::time_seconds();

    const char *folder = argv[1];

    auto            mesh       = sfem::Mesh::create_from_file(comm, smesh::Path(folder));
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    double tack = smesh::time_seconds();
    printf("assemble.c: read\t\t%g seconds\n", tack - tick);

    ptrdiff_t nnz    = 0;
    count_t  *rowptr = 0;
    idx_t    *colidx = 0;
    real_t   *values = 0;

    smesh::create_crs_graph_for_elem_type(
            mesh->element_type(0), n_elements, n_nodes, mesh->elements(0)->data(), &rowptr, &colidx);

    nnz    = rowptr[n_nodes];
    values = (real_t *)malloc((size_t)nnz * sizeof(real_t));
    memset(values, 0, (size_t)nnz * sizeof(real_t));

    double tock = smesh::time_seconds();
    printf("assemble.c: build crs\t\t%g seconds\n", tock - tack);
    tack = tock;

    if (SFEM_LAPLACIAN) {
        switch (mesh->element_type(0)) {
            case smesh::TRI3: {
                cvfem_tri3_diffusion_assemble_hessian(n_elements,
                                                      n_nodes,
                                                      mesh->elements(0)->data(),
                                                      mesh->points()->data(),
                                                      rowptr,
                                                      colidx,
                                                      values);
                break;
            }
            case smesh::TET4: {
                tet4_laplacian_crs(n_elements,
                                   n_nodes,
                                   mesh->elements(0)->data(),
                                   mesh->points()->data(),
                                   rowptr,
                                   colidx,
                                   values);
                break;
            }
            default:
                return EXIT_FAILURE;
        }
    }

    tock = smesh::time_seconds();
    printf("assemble.c: assembly\t\t%g seconds\n", tock - tack);
    tack = tock;

    real_t *rhs = (real_t *)malloc((size_t)n_nodes * sizeof(real_t));
    memset(rhs, 0, (size_t)n_nodes * sizeof(real_t));

    if (SFEM_HANDLE_NEUMANN) {
        SFEM_ERROR("FIXME!");
    }

    if (SFEM_HANDLE_DIRICHLET) {
        char path[1024 * 10];
        snprintf(path, sizeof(path), "%s/zd.raw", folder);

        const char *SFEM_DIRICHLET_NODES = 0;
        SFEM_READ_ENV(SFEM_DIRICHLET_NODES, );

        if (SFEM_DIRICHLET_NODES) {
            strcpy(path, SFEM_DIRICHLET_NODES);
            printf("SFEM_DIRICHLET_NODES=%s\n", path);
        }

        auto dirichlet_buf = sfem::Buffer<idx_t>::from_file(smesh::Path(path));
        if (!dirichlet_buf) {
            SFEM_ERROR("Failed to read dirichlet nodes from %s\n", path);
        }

        const ptrdiff_t nn = dirichlet_buf->size();
        idx_t            *dirichlet_nodes = dirichlet_buf->data();

        constraint_nodes_to_value(nn, dirichlet_nodes, 0, rhs);
        crs_constraint_nodes_to_identity(nn, dirichlet_nodes, 1, rowptr, colidx, values);
    }

    smesh::create_directory(output_folder);

    if (SFEM_HANDLE_RHS) {
        auto rhs_buf = sfem::Buffer<real_t>::wrap(n_nodes, rhs);
        char path[1024 * 10];
        snprintf(path, sizeof(path), "%s/rhs.raw", output_folder);
        if (SFEM_EXPORT_FP32) {
            sfem::astype<float>(rhs_buf)->to_file(smesh::Path(path));
        } else {
            rhs_buf->to_file(smesh::Path(path));
        }
    }

    free(rhs);

    tock = smesh::time_seconds();
    printf("assemble.c: boundary\t\t%g seconds\n", tock - tack);
    tack = tock;

    auto rowptr_buf = sfem::Buffer<count_t>::wrap(n_nodes + 1, rowptr);
    auto colidx_buf = sfem::Buffer<idx_t>::wrap(nnz, colidx);
    auto values_buf = sfem::Buffer<real_t>::wrap(nnz, values);

    if (SFEM_EXPORT_FP32) {
        auto crs = sfem::h_crs_spmv<count_t, idx_t, float>(
                n_nodes, n_nodes, rowptr_buf, colidx_buf, sfem::astype<float>(values_buf), (float)1);
        crs->to_file(smesh::Path(output_folder));
    } else {
        auto crs = sfem::h_crs_spmv<count_t, idx_t, real_t>(n_nodes, n_nodes, rowptr_buf, colidx_buf, values_buf, (real_t)1);
        crs->to_file(smesh::Path(output_folder));
    }

    tock = smesh::time_seconds();
    printf("assemble.c: write\t\t%g seconds\n", tock - tack);
    tack = tock;

    free(rowptr);
    free(colidx);
    free(values);

    tock = smesh::time_seconds();

    if (!comm->rank()) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld\n", (long)n_elements, (long)n_nodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return solve_cvfem_assemble(ctx->communicator(), argc, argv);
}
