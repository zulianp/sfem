#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <cassert>
#include <cstdlib>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "sfem_API.hpp"
#include "sfem_Buffer.hpp"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_MatrixFreeLinearSolver.hpp"
#include "sfem_crs_sym_SpMV.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_crs_SpMV.hpp"

#include "sfem_base.h"

int SFEM_REPEAT = 10;
void time_operator_cpu(const std::shared_ptr<sfem::Operator<real_t>> op, const char* const name,
                       const real_t* const x, real_t* const y);
void time_operator_gpu(const std::shared_ptr<sfem::Operator<real_t>> op, const char* const name,
                       const real_t* const x, real_t* const y);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    {
        MPI_Comm comm = MPI_COMM_WORLD;

        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        if (size != 1) {
            fprintf(stderr, "Parallel execution not supported!\n");
            return EXIT_FAILURE;
        }

        if (argc != 5) {
            fprintf(stderr, "usage: %s <alpha> <crs_folder> <x.raw> <output.raw>\n", argv[0]);
            return EXIT_FAILURE;
        }

        SFEM_READ_ENV(SFEM_REPEAT, atoi);

        const real_t alpha = atof(argv[1]);
        const char* crs_folder = argv[2];
        const char* x_path = argv[3];
        const char* output_path = argv[4];

        double tick = MPI_Wtime();

        char rowptr_path[1024 * 10];
        char colidx_path[1024 * 10];
        char values_path[1024 * 10];

        sprintf(rowptr_path, "%s/indptr.raw", crs_folder);
        sprintf(colidx_path, "%s/indices.raw", crs_folder);
        sprintf(values_path, "%s/values.raw", crs_folder);

        static const auto host = sfem::MEMORY_SPACE_HOST;
        static const auto dev = sfem::MEMORY_SPACE_DEVICE;

        crs_t crs;
        crs_read(comm,
                 rowptr_path,
                 colidx_path,
                 values_path,
                 SFEM_MPI_COUNT_T,
                 SFEM_MPI_IDX_T,
                 SFEM_MPI_REAL_T,
                 &crs);

        printf("Loaded matrix...\n");

        const count_t nnz = crs.gnnz;
        const ptrdiff_t ndofs = crs.grows;

        auto x_host = sfem::create_buffer<real_t>(ndofs, host);

        if (strcmp("gen:ones", x_path) == 0) {
            auto x_host_data = x_host->data();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < ndofs; ++i) {
                x_host_data[i] = 1;
            }

        } else {
            ptrdiff_t _nope_, x_n;
            if (array_read(comm, x_path, SFEM_MPI_REAL_T, (void*)x_host->data(), ndofs, ndofs)) {
                return SFEM_FAILURE;
            }
        }

        auto y1_host = sfem::create_buffer<real_t>(ndofs, host);
        auto y2_host = sfem::create_buffer<real_t>(ndofs, host);
        auto y3_host = sfem::create_buffer<real_t>(ndofs, host);

        // CSR buffers
        auto colidx_host = sfem::Buffer<idx_t>::wrap(nnz, (idx_t*)crs.colidx);
        auto rowptr_host = sfem::Buffer<count_t>::wrap(ndofs + 1, (count_t*)crs.rowptr);
        auto values_host = sfem::Buffer<real_t>::wrap(nnz, (real_t*)crs.values);

        auto rowidx_host = sfem::create_buffer<idx_t>(nnz, host);

        // Triangle sparse matrix buffers
        count_t offdiag_nnz = (crs.gnnz - ndofs) / 2;
        auto diag_values_host = sfem::create_buffer<real_t>(ndofs, host);
        auto offdiag_values_host = sfem::create_buffer<real_t>(offdiag_nnz, host);
        auto offdiag_colidx_host = sfem::create_buffer<idx_t>(offdiag_nnz, host);
        auto offdiag_rowidx_host = sfem::create_buffer<idx_t>(offdiag_nnz, host);
        auto offdiag_rowptr_host = sfem::create_buffer<count_t>(ndofs + 1, host);

        printf("-----------\n");
        printf("|matrix info|\n");
        printf("-----------\n");
        printf("ndofs: %d\n", (int)ndofs);
        printf("nnz: %d\n", nnz);
        printf("offdiag_nnz: %d\n", offdiag_nnz);
        fflush(stdout);

        count_t write_pos = 0;
        for (ptrdiff_t i = 0; i < ndofs; i++) {
            for (idx_t idx = rowptr_host->data()[i]; idx < rowptr_host->data()[i + 1]; idx++) {
                ptrdiff_t j = colidx_host->data()[idx];
                real_t val = values_host->data()[idx];
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

        auto y1 = sfem::create_buffer<real_t>(ndofs, sfem::MEMORY_SPACE_DEVICE);
        auto y2 = sfem::create_buffer<real_t>(ndofs, sfem::MEMORY_SPACE_DEVICE);
        auto x = sfem::to_device(x_host);

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

        // TODO
        // GPU: coo, csr sym, bsr, bsr sym

        auto coo_sym_gpu = sfem::d_sym_coo_spmv(
                ndofs, offdiag_rowidx, offdiag_colidx, offdiag_values, diag_values, alpha);

        auto crs_gpu = sfem::d_crs_spmv(ndofs, ndofs, rowptr, colidx, values, alpha);

        // TODO
        // CPU: coo, bsr, bsr sym

        // also this api should probably have alpha to match others...
        auto coo_sym_host = sfem::h_coosym(nullptr,
                                           offdiag_rowidx_host,
                                           offdiag_colidx_host,
                                           offdiag_values_host,
                                           diag_values_host);

        auto csr_sym_host = sfem::h_crs_sym_spmv(ndofs,
                                                 ndofs,
                                                 offdiag_rowptr_host,
                                                 offdiag_colidx_host,
                                                 diag_values_host,
                                                 offdiag_values_host,
                                                 alpha);

        auto csr_host =
                sfem::h_crs_spmv(ndofs, ndofs, rowptr_host, colidx_host, values_host, alpha);

        printf("\n-----------\n");
        printf("|GPU tests|\n");
        printf("-----------\n\n");
        printf(" operator      avg time      giga dofs\n");

        auto gpu_blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_DEVICE);

        gpu_blas->values(ndofs, 0, y2->data());
        gpu_blas->values(ndofs, 0, y1->data());
        time_operator_gpu(crs_gpu, "csr", x->data(), y2->data());
        time_operator_gpu(coo_sym_gpu, "coo sym", x->data(), y1->data());

        gpu_blas->values(ndofs, 0, y2->data());
        gpu_blas->values(ndofs, 0, y1->data());
        time_operator_gpu(crs_gpu, "csr", x->data(), y2->data());
        time_operator_gpu(coo_sym_gpu, "coo sym", x->data(), y1->data());

        auto y1_tocpu = sfem::to_host(y1);
        auto y2_tocpu = sfem::to_host(y2);

        gpu_blas->axpy(ndofs, -1, y2->data(), y1->data());
        auto norm_y1 = gpu_blas->norm2(ndofs, y1->data());
        printf("norm (gpu) crs out - (gpu) coo sym out: %g\n", norm_y1);
        fflush(stdout);

        /*
        time_operator_gpu(crs_gpu, "csr", x2->data(), y2->data());
        time_operator_gpu(coo_sym_gpu, "coo sym", 1->data(), y1->data());
        */

        // TODO: verify y1 == y2

        auto cpu_blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
        printf("\n-----------\n");
        printf("|CPU tests|\n");
        printf("-----------\n\n");
        printf(" operator      avg time      giga dofs\n");

        cpu_blas->values(ndofs, 0, y1_host->data());
        cpu_blas->values(ndofs, 0, y2_host->data());
        cpu_blas->values(ndofs, 0, y3_host->data());
        time_operator_cpu(coo_sym_host, "coo sym", x_host->data(), y1_host->data());
        time_operator_cpu(csr_sym_host, "csr sym", x_host->data(), y2_host->data());
        time_operator_cpu(csr_host, "csr", x_host->data(), y3_host->data());

        cpu_blas->axpy(ndofs, -1, y2_host->data(), y1_host->data());
        auto norm_y1_host = cpu_blas->norm2(ndofs, y1_host->data());
        printf("norm (cpu) coo sym out - (cpu) csr sym out: %g\n", norm_y1_host);

        cpu_blas->axpy(ndofs, -1, y3_host->data(), y2_host->data());
        auto norm_y2_host = cpu_blas->norm2(ndofs, y2_host->data());
        printf("norm (cpu) csr sym out - (cpu) csr out: %g\n", norm_y2_host);
        fflush(stdout);

        cpu_blas->axpy(ndofs, -1, y3_host->data(), y1_tocpu->data());
        auto norm_y1_tocpu = cpu_blas->norm2(ndofs, y1_tocpu->data());
        printf("norm (cpu) csr out - (gpu) coo sym out: %g\n", norm_y1_tocpu);
        fflush(stdout);

        cpu_blas->axpy(ndofs, -1, y3_host->data(), y2_tocpu->data());
        auto norm_y2_tocpu = cpu_blas->norm2(ndofs, y2_tocpu->data());
        printf("norm (cpu) csr out - (gpu) csr out: %g\n", norm_y2_tocpu);

        fflush(stdout);
        crs_free(&crs);

        double tock = MPI_Wtime();
        if (!rank) {
            printf("sym_spmv_bench.cpp\n");
            printf("TTS: %g seconds\n", tock - tick);
        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

void time_operator_gpu(const std::shared_ptr<sfem::Operator<real_t>> op, const char* const name,
                       const real_t* const x, real_t* const y) {
    sfem::device_synchronize();
    double spmv_tick = MPI_Wtime();

    for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
        op->apply(x, y);
    }

    double spmv_tock = MPI_Wtime();
    sfem::device_synchronize();

    double avg_time = (spmv_tock - spmv_tick) / SFEM_REPEAT;
    double avg_throughput = (op->rows() / avg_time) * 1e-9;
    printf("|%-13s|%-13f|%-13f|\n", name, avg_time, avg_throughput);
}

void time_operator_cpu(const std::shared_ptr<sfem::Operator<real_t>> op, const char* const name,
                       const real_t* const x, real_t* const y) {
    double spmv_tick = MPI_Wtime();

    for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
        op->apply(x, y);
    }

    double spmv_tock = MPI_Wtime();

    double avg_time = (spmv_tock - spmv_tick) / SFEM_REPEAT;
    double avg_throughput = (op->rows() / avg_time) * 1e-9;
    printf("|%-13s|%-13f|%-13f|\n", name, avg_time, avg_throughput);
}
