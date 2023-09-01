#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "dirichlet.h"

#include "inverse.c"
#include "sfem_base.h"

#define MAX_BLOCK_SIZE 4

static int bgs_init(const ptrdiff_t nnodes,
                    const int block_rows,
                    const count_t *const SFEM_RESTRICT rowptr,
                    const idx_t *const SFEM_RESTRICT colidx,
                    real_t **const SFEM_RESTRICT values,
                    real_t **const inv_bdiag) {
    //
    switch (block_rows) {
        case 1: {
            dinvert1(nnodes, rowptr, colidx, values, inv_bdiag);
            break;
        }

        case 2: {
            dinvert2(nnodes, rowptr, colidx, values, inv_bdiag);
            break;
        }

        case 3: {
            dinvert3(nnodes, rowptr, colidx, values, inv_bdiag);
            break;
        }
        case 4: {
            dinvert4(nnodes, rowptr, colidx, values, inv_bdiag);
            break;
        }
        default: {
            assert(0);
            return -1;
        }
    }

    return 0;
}

static int residual(const ptrdiff_t nnodes,
                    const int block_rows,
                    const count_t *const SFEM_RESTRICT rowptr,
                    const idx_t *const SFEM_RESTRICT colidx,
                    real_t **const SFEM_RESTRICT values,
                    real_t **const SFEM_RESTRICT rhs,
                    real_t **const SFEM_RESTRICT x,
                    real_t *res) {
#pragma omp parallel
    {
#pragma omp for
        for (int d1 = 0; d1 < block_rows; d1++) {
            res[d1] = 0;
        }

#pragma omp for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            const count_t r_begin = rowptr[i];
            const count_t r_end = rowptr[i + 1];
            const count_t r_extent = r_end - r_begin;
            const idx_t *const r_colidx = &colidx[r_begin];

            real_t r[MAX_BLOCK_SIZE];
            for (int d1 = 0; d1 < block_rows; d1++) {
                r[d1] = rhs[d1][i];
            }

            for (count_t k = 0; k < r_extent; k++) {
                const idx_t col = r_colidx[k];
                for (int d1 = 0; d1 < block_rows; d1++) {
                    for (int d2 = 0; d2 < block_rows; d2++) {
                        const int bb = d1 * block_rows + d2;
                        r[d1] -= values[bb][r_begin + k] * x[d2][col];
                    }
                }
            }

            for (int d1 = 0; d1 < block_rows; d1++) {
#pragma omp atomic update
                res[d1] += r[d1] * r[d1];
            }
        }
    }

    for (int d1 = 0; d1 < block_rows; d1++) {
        res[d1] = sqrt(res[d1]);
    }

    return 0;
}

static SFEM_INLINE real_t atomic_read(const real_t *p) {
    real_t value;
#pragma omp atomic read
    value = *p;
    return value;
}

static SFEM_INLINE void atomic_write(real_t *p, const real_t value) {
#pragma omp atomic write
    *p = value;
}

static int bgs_forward(const ptrdiff_t nnodes,
                       const int block_rows,
                       const count_t *const SFEM_RESTRICT rowptr,
                       const idx_t *const SFEM_RESTRICT colidx,
                       real_t **const SFEM_RESTRICT values,
                       real_t **const SFEM_RESTRICT inv_bdiag,
                       real_t **const SFEM_RESTRICT rhs,
                       real_t **const SFEM_RESTRICT x) {
#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            const count_t r_begin = rowptr[i];
            const count_t r_end = rowptr[i + 1];
            const count_t r_extent = r_end - r_begin;
            const idx_t *const r_colidx = &colidx[r_begin];

            real_t r[MAX_BLOCK_SIZE];
            for (int d1 = 0; d1 < block_rows; d1++) {
                r[d1] = rhs[d1][i];
            }

            for (count_t k = 0; k < r_extent; k++) {
                const idx_t col = r_colidx[k];
                if (col == i) continue;

                for (int d1 = 0; d1 < block_rows; d1++) {
                    for (int d2 = 0; d2 < block_rows; d2++) {
                        const int bb = d1 * block_rows + d2;
                        r[d1] -= values[bb][r_begin + k] * atomic_read(&x[d2][col]);
                    }
                }
            }

            for (int d1 = 0; d1 < block_rows; d1++) {
                real_t acc = 0;
                for (int d2 = 0; d2 < block_rows; d2++) {
                    const int bb = d1 * block_rows + d2;
                    assert(inv_bdiag[bb][i] == inv_bdiag[bb][i]);
                    const real_t val = inv_bdiag[bb][i] * r[d2];
                    acc += val;
                }

                atomic_write(&x[d1][i], acc);
            }
        }
    }

    return 0;
}

static int bgs_backward(const ptrdiff_t nnodes,
                        const int block_rows,
                        const count_t *const SFEM_RESTRICT rowptr,
                        const idx_t *const SFEM_RESTRICT colidx,
                        real_t **const SFEM_RESTRICT values,
                        real_t **const SFEM_RESTRICT inv_bdiag,
                        real_t **const SFEM_RESTRICT rhs,
                        real_t **const SFEM_RESTRICT x) {
#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t i = nnodes - 1; i >= 0; i--) {
            const count_t r_begin = rowptr[i];
            const count_t r_end = rowptr[i + 1];
            const count_t r_extent = r_end - r_begin;
            const idx_t *const r_colidx = &colidx[r_begin];

            real_t r[MAX_BLOCK_SIZE];
            for (int d1 = 0; d1 < block_rows; d1++) {
                r[d1] = rhs[d1][i];
            }

            for (count_t k = 0; k < r_extent; k++) {
                const idx_t col = r_colidx[k];
                if (col == i) continue;

                for (int d1 = 0; d1 < block_rows; d1++) {
                    for (int d2 = 0; d2 < block_rows; d2++) {
                        const int bb = d1 * block_rows + d2;
                        r[d1] -= values[bb][r_begin + k] * atomic_read(&x[d2][col]);
                    }
                }
            }

            for (int d1 = 0; d1 < block_rows; d1++) {
                real_t acc = 0;
                for (int d2 = 0; d2 < block_rows; d2++) {
                    const int bb = d1 * block_rows + d2;
                    assert(inv_bdiag[bb][i] == inv_bdiag[bb][i]);
                    const real_t val = inv_bdiag[bb][i] * r[d2];
                    acc += val;
                }

                atomic_write(&x[d1][i], acc);
            }
        }
    }

    return 0;
}

static int bgs(const ptrdiff_t nnodes,
               const int block_rows,
               const count_t *const SFEM_RESTRICT rowptr,
               const idx_t *const SFEM_RESTRICT colidx,
               real_t **const SFEM_RESTRICT values,
               real_t **const SFEM_RESTRICT inv_bdiag,
               real_t **const SFEM_RESTRICT rhs,
               real_t **const SFEM_RESTRICT x,
               const int num_sweeps,
               int symmetric_variant) {
    for (int s = 0; s < num_sweeps; s++) {
        bgs_forward(nnodes, block_rows, rowptr, colidx, values, inv_bdiag, rhs, x);
        if (symmetric_variant)
            bgs_backward(nnodes, block_rows, rowptr, colidx, values, inv_bdiag, rhs, x);
    }

    return 0;
}

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
            stderr, "usage: %s <block_rows> <crs_folder> <rhs_prefix> <output_prefix>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int block_rows = atoi(argv[1]);
    const char *crs_folder = argv[2];
    const char *rhs_prefix = argv[3];
    const char *output_prefix = argv[4];

    int SFEM_LS_MAX_IT = 1;
    SFEM_READ_ENV(SFEM_LS_MAX_IT, atoi);

    int SFEM_BGS_SWEEPS = 2000;
    SFEM_READ_ENV(SFEM_BGS_SWEEPS, atoi);

    int SFEM_BGS_SYMMETRIC = 0;
    SFEM_READ_ENV(SFEM_BGS_SYMMETRIC, atoi);

    // struct stat st = {0};
    // if (stat(output_path, &st) == -1) {
    //     mkdir(output_path, 0700);
    // }

    const double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read
    ///////////////////////////////////////////////////////////////////////////////

    char rowptr_path[10240];
    char colidx_path[10240];
    char values_path[10240];
    char vec_path[10240];

    sprintf(rowptr_path, "%s/rowptr.raw", crs_folder);
    sprintf(colidx_path, "%s/colidx.raw", crs_folder);
    sprintf(values_path, "%s/values.*.raw", crs_folder);

    ptrdiff_t local_ndofs, ndofs;
    real_t **rhs = (real_t **)malloc(block_rows * sizeof(real_t *));
    for (int d = 0; d < block_rows; d++) {
        sprintf(vec_path, "%s.%d.raw", rhs_prefix, d);
        array_create_from_file(
            comm, vec_path, SFEM_MPI_REAL_T, (void **)&rhs[d], &local_ndofs, &ndofs);
    }

    block_crs_t crs;
    block_crs_read(comm,
                   rowptr_path,
                   colidx_path,
                   values_path,
                   SFEM_MPI_COUNT_T,
                   SFEM_MPI_IDX_T,
                   SFEM_MPI_REAL_T,
                   &crs);

    assert(block_rows * block_rows == crs.block_size);
    if (block_rows * block_rows != crs.block_size) {
        fprintf(stderr, "Wrong number of blocks for block_crs\n");
        return EXIT_FAILURE;
    }

    real_t **x = (real_t **)malloc(block_rows * sizeof(real_t *));
    for (int d = 0; d < block_rows; d++) {
        x[d] = (real_t *)calloc(local_ndofs, sizeof(real_t));
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Solve
    ///////////////////////////////////////////////////////////////////////////////

    real_t res[MAX_BLOCK_SIZE];

    {
        real_t **inv_bdiag = (real_t **)malloc(block_rows * block_rows * sizeof(real_t *));
        for (int d = 0; d < block_rows * block_rows; d++) {
            inv_bdiag[d] = (real_t *)calloc(local_ndofs, sizeof(real_t));
        }

        count_t *rowptr = (count_t *)crs.rowptr;
        idx_t *colidx = (idx_t *)crs.colidx;
        real_t **values = (real_t **)crs.values;

        bgs_init(local_ndofs, block_rows, rowptr, colidx, values, inv_bdiag);

        for (int i = 0; i < SFEM_LS_MAX_IT; i += SFEM_BGS_SWEEPS) {
            bgs(local_ndofs,
                block_rows,
                rowptr,
                colidx,
                values,
                inv_bdiag,
                rhs,
                x,
                SFEM_BGS_SWEEPS,
                SFEM_BGS_SYMMETRIC);

            residual(local_ndofs, block_rows, rowptr, colidx, values, rhs, x, res);

            int stop = 0;

            printf("%d) ", i + SFEM_BGS_SWEEPS);
            for (int d = 0; d < block_rows; d++) {
                printf("%g ", res[d]);
                stop += res[d] < 1e-8;
            }
            printf("\n");

            if (stop == block_rows) break;
        }

        for (int d = 0; d < block_rows * block_rows; d++) {
            free(inv_bdiag[d]);
        }

        free(inv_bdiag);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Write
    ///////////////////////////////////////////////////////////////////////////////

    for (int d = 0; d < block_rows; d++) {
        sprintf(vec_path, "%s.%d.raw", output_prefix, d);
        array_write(comm, vec_path, SFEM_MPI_REAL_T, (void *)x[d], local_ndofs, ndofs);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    block_crs_free(&crs);

    for (int d = 0; d < block_rows; d++) {
        free(rhs[d]);
        free(x[d]);
    }

    free(rhs);
    free(x);

    const double tock = MPI_Wtime();

    if (!rank) {
        printf("bgs.c TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
