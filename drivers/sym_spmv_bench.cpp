#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <cassert>
#include <cstdlib>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "sfem_Buffer.hpp"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_crs_SpMV.hpp"
#include "sfem_tpl_blas.hpp"
#include "utils.h"

#include "dirichlet.h"

#include "sfem_base.h"
#include "spmv.h"

int SFEM_REPEAT = 10;
void time_operator(const std::shared_ptr<sfem::Operator<real_t>> op, const char* const name, const real_t* const x, real_t* const y);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 5) {
        fprintf(
            stderr, "usage: %s <alpha> <crs_folder> <x.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    SFEM_READ_ENV(SFEM_REPEAT, atoi);

    const real_t alpha = atof(argv[1]);
    const char *crs_folder = argv[2];
    const char *x_path = argv[3];
    const char *output_path = argv[4];

    double tick = MPI_Wtime();

    char rowptr_path[1024 * 10];
    char colidx_path[1024 * 10];
    char values_path[1024 * 10];

    sprintf(rowptr_path, "%s/indptr.raw", crs_folder);
    sprintf(colidx_path, "%s/indices.raw", crs_folder);
    sprintf(values_path, "%s/values.raw", crs_folder);

    crs_t crs;
    crs_read(comm,
             rowptr_path,
             colidx_path,
             values_path,
             SFEM_MPI_COUNT_T,
             SFEM_MPI_IDX_T,
             SFEM_MPI_REAL_T,
             &crs);

    real_t *x_host = 0;
    if (strcmp("gen:ones", x_path) == 0) {
        ptrdiff_t ndofs = crs.lrows;
        x_host = (real_t*) malloc(ndofs * sizeof(real_t));
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            x_host[i] = 1;
        }

    } else {
        ptrdiff_t _nope_, x_n;
        array_create_from_file(comm, x_path, SFEM_MPI_REAL_T, (void **)&x_host, &_nope_, &x_n);
    }
    printf("Loaded matrix...\n");

    ptrdiff_t ndofs = crs.grows;
    count_t nnz = crs.gnnz;
    auto x_host_buff = sfem::Buffer<real_t>::own(ndofs, x_host, free);

    // CSR buffers
    auto colidx_host =  sfem::Buffer<idx_t>::wrap(nnz, (idx_t*) crs.colidx);
    auto rowidx_host = sfem::h_buffer<idx_t>(nnz);
    auto rowptr_host =  sfem::Buffer<count_t>::wrap(ndofs + 1, (count_t*) crs.rowptr);
    auto values_host = sfem::Buffer<real_t>::wrap(nnz, (real_t*) crs.values);

    // Diagonal sparse matrix buffers
    count_t offdiag_nnz = (crs.gnnz - ndofs) / 2;
    auto diag_values_host = sfem::h_buffer<real_t>(ndofs);
    auto offdiag_values_host = sfem::h_buffer<real_t>(offdiag_nnz);
    auto offdiag_colidx_host = sfem::h_buffer<idx_t>(offdiag_nnz);
    auto offdiag_rowidx_host = sfem::h_buffer<idx_t>(offdiag_nnz);
    auto offdiag_rowptr_host = sfem::h_buffer<count_t>(ndofs + 1);

    count_t write_pos = 0;
    for (ptrdiff_t i = 0; i < ndofs; i++) {
        for (idx_t idx = crs.rowptr[i]; idx < crs.rowptr[i+1]; idx++) {
            ptrdiff_t j = crs.colidx[idx];
            real_t val = crs.values[idx];
            rowidx_host->data()[idx] = i;

            if (j > i) {
                offdiag_rowidx_host->data()[write_pos] = i;
                offdiag_colidx_host->data()[write_pos] = j;
                offdiag_values_host->data()[write_pos] = val;
                write_pos++;
            } else if (i == j) {
                diag_values_host->data()[i] = val;
            }
        }
        offdiag_rowptr_host->data()[i + 1] = write_pos;
    }
    // Matrix should be standard crs input 
    assert(write_pos == offdiag_nnz);

    real_t* y1_device = d_allocate(ndofs); 
    real_t* y2_device = d_allocate(ndofs); 

    auto y1 = sfem::Buffer(ndofs, y1_device, d_destroy, sfem::MEMORY_SPACE_DEVICE);
    auto y2 = sfem::Buffer(ndofs, y2_device, d_destroy, sfem::MEMORY_SPACE_DEVICE);
    auto x = sfem::to_device(x_host_buff);

    // CSR buffers device
    auto colidx = sfem::to_device(colidx_host);
    auto rowidx = sfem::to_device(rowidx_host);
    auto rowptr = sfem::to_device(rowptr_host);
    auto values = sfem::to_device(values_host);

    // Diagonal sparse matrix buffers device
    auto diag_values = sfem::to_device(diag_values_host);
    auto offdiag_values = sfem::to_device(offdiag_values_host);
    auto offdiag_colidx = sfem::to_device(offdiag_colidx_host);
    auto offdiag_rowidx = sfem::to_device(offdiag_rowidx_host);
    auto offdiag_rowptr = sfem::to_device(offdiag_rowptr_host);

    auto coo_sym_gpu = sfem::d_sym_coo_spmv(
        ndofs, 
        offdiag_nnz, 
        offdiag_rowidx, 
        offdiag_colidx, 
        offdiag_values,
        diag_values, 
        alpha
    );

    auto crs_gpu = sfem::d_crs_spmv(
            ndofs, ndofs,
            rowptr,
            colidx,
            values, alpha);

    // TODO:
    // GPU: coo not sym, csr sym, bsr not sym, bsr sym
    // CPU: (all) coo, csr, bsr / sym, not sym

    printf(" operator      avg time      giga dofs\n");
    time_operator(crs_gpu, "csr", x->data(), y2.data());
    time_operator(coo_sym_gpu, "coo sym", x->data(), y1.data());

    time_operator(crs_gpu, "csr", x->data(), y2.data());
    time_operator(coo_sym_gpu, "coo sym", x->data(), y1.data());

    time_operator(crs_gpu, "csr", x->data(), y2.data());
    time_operator(coo_sym_gpu, "coo sym", x->data(), y1.data());

    // TODO: verify y1 == y2

    crs_free(&crs);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("sym_spmv_bench.cpp\n");
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

void time_operator(const std::shared_ptr<sfem::Operator<real_t>> op, const char* const name, const real_t* const x, real_t* const y) {
    double spmv_tick = MPI_Wtime();

    for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
        op->apply(x, y);
    }

    double spmv_tock = MPI_Wtime();
    double avg_time = (spmv_tock - spmv_tick) / SFEM_REPEAT;
    double avg_throughput = (op->rows() / avg_time) * (sizeof(real_t) * 1e-9);
    printf("|%-13s|%-13f|%-13f|\n", name, avg_time, avg_throughput);
}
