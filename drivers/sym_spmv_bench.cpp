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
#include "sfem_cuda_blas.h"
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_crs_SpMV.hpp"
#include "sfem_tpl_blas.hpp"
#include "utils.h"

#include "dirichlet.h"

#include "sfem_base.h"
#include "spmv.h"

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

    int SFEM_REPEAT = 1;
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
    ptrdiff_t offdiag_nnz = (crs.gnnz - ndofs) / 2;
    auto x_host_buff = sfem::Buffer<real_t>::own(ndofs, x_host, free);

    idx_t* colidx_host = (idx_t*) malloc(offdiag_nnz * sizeof(idx_t));
    idx_t* rowidx_host= (idx_t*) malloc(offdiag_nnz * sizeof(idx_t));
    real_t* values_host= (real_t*) malloc(offdiag_nnz * sizeof(real_t));
    real_t* diag_values_host= (real_t*) malloc(ndofs * sizeof(real_t));

    auto colidx_host_buff = sfem::Buffer<idx_t>::own(offdiag_nnz, colidx_host, free);
    auto rowidx_host_buff = sfem::Buffer<idx_t>::own(offdiag_nnz, rowidx_host, free);
    auto values_host_buff = sfem::Buffer<real_t>::own(offdiag_nnz, values_host, free);
    auto diag_values_host_buff =  sfem::Buffer<real_t>::own(ndofs, diag_values_host, free);

    count_t write_pos = 0;
    for (ptrdiff_t i = 0; i < ndofs; i++) {
        for (idx_t idx = crs.rowptr[i]; idx < crs.rowptr[i+1]; idx++) {
            ptrdiff_t j = crs.colidx[idx];
            real_t val = crs.values[idx];

            if (j > i) {
                rowidx_host_buff->data()[write_pos] = i;
                colidx_host_buff->data()[write_pos] = j;
                values_host_buff->data()[write_pos] = val;
                write_pos++;
            } else if (i == j) {
                diag_values_host_buff->data()[i] = val;
            }
        }
    }
    // Matrix should be standard crs input 
    assert(write_pos == offdiag_nnz);

    real_t* y_device = d_allocate(ndofs); 
    auto y = sfem::Buffer(ndofs, y_device, d_destroy, sfem::MEMORY_SPACE_DEVICE);
    auto x = sfem::to_device(x_host_buff);
    auto rowidx =sfem::to_device(rowidx_host_buff); 
    auto colidx =sfem::to_device(colidx_host_buff); 
    auto values =sfem::to_device(values_host_buff); 
    auto diag_values =sfem::to_device(diag_values_host_buff); 

    auto coo_sym_gpu = sfem::d_sym_coo_spmv(
        ndofs, 
        offdiag_nnz, 
        rowidx, 
        colidx, 
        values,
        diag_values, 
        alpha
    );

    double spmv_tick = MPI_Wtime();

    for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
        coo_sym_gpu->apply(x->data(), y.data());
    }

    double spmv_tock = MPI_Wtime();
    double avg_time = (spmv_tock - spmv_tick) / SFEM_REPEAT;
    double avg_throughput = (crs.grows / avg_time) * (sizeof(real_t) * 1e-9);
    printf("spmv:  %g %g %ld %ld %ld\n", avg_time, avg_throughput, 0l, crs.lrows, crs.lnnz);

    //array_write(comm, output_path, SFEM_MPI_REAL_T, y, crs.grows, crs.grows);
    crs_free(&crs);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("sym_spmv_bench.cpp\n");
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
