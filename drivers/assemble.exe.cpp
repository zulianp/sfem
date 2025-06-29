#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array_dtof.h"

#include "sfem_API.hpp"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double tick = MPI_Wtime();

    if (size != 1) {
        SFEM_ERROR("Parallel execution not supported!\n");
    }

    if (argc != 3) {
        SFEM_ERROR("usage: %s <folder> <output>\n", argv[0]);
    }

    const char *output_folder    = argv[2];
    const char *SFEM_OPERATOR    = "Laplacian";
    int         SFEM_BLOCK_SIZE  = 1;
    int         SFEM_EXPORT_FP32 = 0;

    SFEM_READ_ENV(SFEM_OPERATOR, );
    SFEM_READ_ENV(SFEM_BLOCK_SIZE, atoi);
    SFEM_READ_ENV(SFEM_EXPORT_FP32, atoi);

    MPI_Datatype value_type = SFEM_EXPORT_FP32 ? MPI_FLOAT : MPI_DOUBLE;
    auto         es         = sfem::EXECUTION_SPACE_HOST;

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    auto        m      = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);
    auto        fs     = sfem::FunctionSpace::create(m, SFEM_BLOCK_SIZE);
    auto        f      = sfem::Function::create(fs);

    auto op = sfem::create_op(fs, SFEM_OPERATOR, es);
    f->add_operator(op);

    auto dbc = sfem::DirichletConditions::create_from_env(fs);
    f->add_constraint(dbc);

    auto nbc = sfem::NeumannConditions::create_from_env(fs);
    f->add_operator(nbc);

    ///////////////////////////////////////////////////////////////////////////////
    // Zero solution vector
    ///////////////////////////////////////////////////////////////////////////////

    auto x = sfem::create_buffer<real_t>(m->n_nodes(), es);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS matrix
    ///////////////////////////////////////////////////////////////////////////////

    auto      crs_graph = f->crs_graph();
    ptrdiff_t nnz       = crs_graph->nnz();
    auto      values    = sfem::create_buffer<real_t>(nnz, es);
    f->hessian_crs(x->data(), crs_graph->rowptr()->data(), crs_graph->colidx()->data(), values->data());

    ///////////////////////////////////////////////////////////////////////////////
    // RHS
    ///////////////////////////////////////////////////////////////////////////////

    auto rhs = sfem::create_buffer<real_t>(m->n_nodes(), es);
    f->gradient(x->data(), rhs->data());

    auto blas = sfem::blas<real_t>(es);

    // Move to RHS
    blas->scal(rhs->size(), -1, rhs->data());

    ///////////////////////////////////////////////////////////////////////////////
    // Write CRS matrix and rhs vector
    ///////////////////////////////////////////////////////////////////////////////

    sfem::create_directory(output_folder);

    if (SFEM_EXPORT_FP32) {
        array_dtof(nnz, (const real_t *)values->data(), (float *)values->data());
    }

    {
        crs_t crs_out;
        crs_out.rowptr      = (char *)crs_graph->rowptr()->data();
        crs_out.colidx      = (char *)crs_graph->colidx()->data();
        crs_out.values      = (char *)values->data();
        crs_out.grows       = m->n_nodes();
        crs_out.lrows       = m->n_nodes();
        crs_out.lnnz        = nnz;
        crs_out.gnnz        = nnz;
        crs_out.start       = 0;
        crs_out.rowoffset   = 0;
        crs_out.rowptr_type = SFEM_MPI_COUNT_T;
        crs_out.colidx_type = SFEM_MPI_IDX_T;
        crs_out.values_type = value_type;
        crs_write_folder(comm, output_folder, &crs_out);
    }

    {
        if (SFEM_EXPORT_FP32) {
            array_dtof(rhs->size(), (const real_t *)rhs->data(), (float *)rhs->data());
        }

        {
            char path[1024 * 10];
            snprintf(path, sizeof(path), "%s/rhs.raw", output_folder);
            array_write(comm, path, value_type, rhs->data(), rhs->size(), rhs->size());
        }
    }

    ptrdiff_t nelements = m->n_elements();
    ptrdiff_t nnodes    = m->n_nodes();

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld\n", (long)nelements, (long)nnodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
