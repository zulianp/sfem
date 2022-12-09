#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

typedef float geom_t;
typedef int idx_t;
typedef double real_t;

#ifdef NDEBUG
#define INLINE inline
#else
#define INLINE
#endif

ptrdiff_t read_file(MPI_Comm comm, const char *path, void **data) {
    MPI_Status status;
    MPI_Offset nbytes;
    MPI_File file;
    CATCH_MPI_ERROR(MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
    CATCH_MPI_ERROR(MPI_File_get_size(file, &nbytes));
    *data = malloc(nbytes);

    CATCH_MPI_ERROR(MPI_File_read_at_all(file, 0, *data, nbytes, MPI_CHAR, &status));
    return nbytes;
}

INLINE real_t det3(const real_t *mat) {
    return mat[0] * mat[4] * mat[8] + mat[1] * mat[5] * mat[6] + mat[2] * mat[3] * mat[7] - mat[0] * mat[5] * mat[7] -
           mat[1] * mat[3] * mat[8] - mat[2] * mat[4] * mat[6];
}

INLINE void adjugate3(const real_t *mat, real_t *mat_adj) {
    mat_adj[0] = (mat[4] * mat[8] - mat[5] * mat[7]);
    mat_adj[1] = (mat[2] * mat[7] - mat[1] * mat[8]);
    mat_adj[2] = (mat[1] * mat[5] - mat[2] * mat[4]);
    mat_adj[3] = (mat[5] * mat[6] - mat[3] * mat[8]);
    mat_adj[4] = (mat[0] * mat[8] - mat[2] * mat[6]);
    mat_adj[5] = (mat[2] * mat[3] - mat[0] * mat[5]);
    mat_adj[6] = (mat[3] * mat[7] - mat[4] * mat[6]);
    mat_adj[7] = (mat[1] * mat[6] - mat[0] * mat[7]);
    mat_adj[8] = (mat[0] * mat[4] - mat[1] * mat[3]);
}

INLINE void invert3(const real_t *mat, real_t *mat_inv, const real_t det) {
    assert(det != 0.);
    adjugate3(mat, mat_inv);

    for (int i = 0; i < 9; ++i) {
        mat_inv[i] /= det;
    }
}

INLINE void mv3(const real_t A[3 * 3], const real_t v[3], real_t *out) {
    for (int i = 0; i < 3; ++i) {
        out[i] = 0;
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i] += A[i * 3 + j] * v[j];
        }
    }
}

INLINE real_t dot3(const real_t v1[3], const real_t v2[3]) {
    real_t ret = 0;
    for (int i = 0; i < 3; ++i) {
        ret += v1[i] * v2[i];
    }

    return ret;
}

INLINE real_t area3(const real_t left[3], const real_t right[3]) {
    real_t a = (left[1] * right[2]) - (right[1] * left[2]);
    real_t b = (left[2] * right[0]) - (right[2] * left[0]);
    real_t c = (left[0] * right[1]) - (right[0] * left[1]);
    return sqrt(a * a + b * b + c * c);
}

INLINE void integrate_neumann(real_t value, real_t area, real_t *element_vector) {
    element_vector[0] = value * area;
    element_vector[1] = value * area;
    element_vector[2] = value * area;
}

void integrate(const real_t *inverse_jacobian, const real_t volume, real_t *element_matrix) {
    const real_t grad_ref[4][3] = {{-1, -1, -1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    real_t grad_test[3];
    real_t grad_trial[3];

    for (int edof_i = 0; edof_i < 4; ++edof_i) {
        mv3(inverse_jacobian, grad_ref[edof_i], grad_test);

        const real_t aii = dot3(grad_test, grad_test) * volume;

        element_matrix[edof_i * 4 + edof_i] = aii;

        for (int edof_j = edof_i + 1; edof_j < 4; ++edof_j) {
            mv3(inverse_jacobian, grad_ref[edof_j], grad_trial);

            const real_t aij = dot3(grad_test, grad_trial) * volume;

            element_matrix[edof_i * 4 + edof_j] = aij;
            element_matrix[edof_i + edof_j * 4] = aij;
        }
    }
}

int cmpfunc(const void *a, const void *b) { return (*(idx_t *)a - *(idx_t *)b); }
INLINE void quicksort(idx_t *arr, idx_t size) { qsort(arr, size, sizeof(idx_t), cmpfunc); }

idx_t binarysearch(const idx_t key, const idx_t *arr, idx_t size) {
    idx_t *ptr = bsearch(&key, arr, size, sizeof(idx_t), cmpfunc);
    if (!ptr) return -1;
    return (idx_t)(ptr - arr);
}

idx_t unique(idx_t *arr, idx_t size) {
    idx_t *first = arr;
    idx_t *last = arr + size;

    if (first == last) return 0;

    idx_t *result = first;
    while (++first != last)
        if (*result != *first && ++result != first) *result = *first;

    return (++result) - arr;
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

    if (argc < 2) {
        fprintf(stderr, "usage: %s <folder> [output_folder=./]", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = "./";
    if (argc > 2) {
        output_folder = argv[2];
    }

    int pure_neumann = 0;

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    char path[1024 * 10];
    ptrdiff_t nnodes = 0;
    geom_t *xyz[3];

    {
        sprintf(path, "%s/x.raw", folder);
        ptrdiff_t x_nnodes = read_file(comm, path, (void **)&xyz[0]);

        sprintf(path, "%s/y.raw", folder);
        ptrdiff_t y_nnodes = read_file(comm, path, (void **)&xyz[1]);

        sprintf(path, "%s/z.raw", folder);
        ptrdiff_t z_nnodes = read_file(comm, path, (void **)&xyz[2]);

        assert(x_nnodes == y_nnodes);
        assert(x_nnodes == z_nnodes);

        x_nnodes /= sizeof(geom_t);
        assert(x_nnodes * sizeof(geom_t) == y_nnodes);
        nnodes = x_nnodes;
    }

    ptrdiff_t nelements = 0;
    idx_t *elems[4];

    {
        sprintf(path, "%s/i0.raw", folder);
        ptrdiff_t nindex0 = read_file(comm, path, (void **)&elems[0]);

        sprintf(path, "%s/i1.raw", folder);
        ptrdiff_t nindex1 = read_file(comm, path, (void **)&elems[1]);

        sprintf(path, "%s/i2.raw", folder);
        ptrdiff_t nindex2 = read_file(comm, path, (void **)&elems[2]);

        sprintf(path, "%s/i3.raw", folder);
        ptrdiff_t nindex3 = read_file(comm, path, (void **)&elems[3]);

        assert(nindex0 == nindex1);
        assert(nindex3 == nindex2);

        nindex0 /= sizeof(idx_t);
        assert(nindex0 * sizeof(idx_t) == nindex1);
        nelements = nindex0;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nnz = 0;
    idx_t *rowptr = (idx_t *)malloc((nnodes + 1) * sizeof(idx_t));
    idx_t *colidx = 0;
    real_t *values = 0;

    {
        idx_t *e2nptr = malloc((nnodes + 1) * sizeof(idx_t));
        memset(e2nptr, 0, nnodes * sizeof(idx_t));

        int *bookkepping = malloc((nnodes) * sizeof(int));
        memset(bookkepping, 0, (nnodes) * sizeof(int));

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            for (idx_t i = 0; i < nelements; ++i) {
                assert(elems[edof_i][i] < nnodes);
                assert(elems[edof_i][i] >= 0);

                ++e2nptr[elems[edof_i][i] + 1];
            }
        }

        for (idx_t i = 0; i < nnodes; ++i) {
            e2nptr[i + 1] += e2nptr[i];
        }

        idx_t *elindex = (idx_t *)malloc(e2nptr[nnodes] * sizeof(idx_t));

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            for (idx_t i = 0; i < nelements; ++i) {
                idx_t node = elems[edof_i][i];

                assert(e2nptr[node] + bookkepping[node] < e2nptr[node + 1]);

                elindex[e2nptr[node] + bookkepping[node]++] = i;
            }
        }

        free(bookkepping);

        rowptr[0] = 0;

        idx_t n2nbuff[2048];
        for (idx_t node = 0; node < nnodes; ++node) {
            idx_t ebegin = e2nptr[node];
            idx_t eend = e2nptr[node + 1];

            idx_t nneighs = 0;

            for (idx_t e = ebegin; e < eend; ++e) {
                idx_t eidx = elindex[e];
                assert(eidx < nelements);

                for (int edof_i = 0; edof_i < 4; ++edof_i) {
                    idx_t neighnode = elems[edof_i][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            quicksort(n2nbuff, nneighs);
            nneighs = unique(n2nbuff, nneighs);

            nnz += nneighs;
            rowptr[node + 1] = nnz;
        }

        colidx = (idx_t *)malloc(nnz * sizeof(idx_t));

        ptrdiff_t coloffset = 0;
        for (idx_t node = 0; node < nnodes; ++node) {
            idx_t ebegin = e2nptr[node];
            idx_t eend = e2nptr[node + 1];

            idx_t nneighs = 0;

            for (idx_t e = ebegin; e < eend; ++e) {
                idx_t eidx = elindex[e];
                assert(eidx < nelements);

                for (int edof_i = 0; edof_i < 4; ++edof_i) {
                    idx_t neighnode = elems[edof_i][eidx];
                    assert(nneighs < 2048);
                    n2nbuff[nneighs++] = neighnode;
                }
            }

            quicksort(n2nbuff, nneighs);
            nneighs = unique(n2nbuff, nneighs);

            for (idx_t i = 0; i < nneighs; ++i) {
                colidx[coloffset + i] = n2nbuff[i];
            }

            coloffset += nneighs;
        }

        free(e2nptr);
        free(elindex);

        values = (real_t *)malloc(nnz * sizeof(real_t));
        memset(values, 0, nnz * sizeof(real_t));
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Operator assembly
    ///////////////////////////////////////////////////////////////////////////////

    {
        real_t jacobian[3 * 3];
        real_t inverse_jacobian[3 * 3];
        real_t element_matrix[4 * 4];

        real_t grad_ref[4][3] = {{-1, -1, -1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        real_t grad_test[3];
        real_t grad_trial[3];

        for (ptrdiff_t i = 0; i < nelements; ++i) {
            // Collect element coordinates
            for (int d1 = 0; d1 < 3; ++d1) {
                real_t x0 = (real_t)xyz[d1][elems[0][i]];

                for (int d2 = 0; d2 < 3; ++d2) {
                    real_t x1 = (real_t)xyz[d1][elems[d2 + 1][i]];
                    jacobian[d1 * 3 + d2] = x1 - x0;
                }
            }

            if (0) {
                real_t jacobian_determinant = det3(jacobian);
                invert3(jacobian, inverse_jacobian, jacobian_determinant);

                assert(jacobian_determinant > 0.);

                const real_t dx = jacobian_determinant / 6.;
                integrate(inverse_jacobian, dx, element_matrix);
            } else {
                // 9 less operations
                real_t jacobian_determinant = det3(jacobian);
                adjugate3(jacobian, inverse_jacobian);

                assert(jacobian_determinant > 0.);

                const real_t dx = 1./(jacobian_determinant * 6.);
                integrate(inverse_jacobian, dx, element_matrix);
            }

#ifndef NDEBUG
            real_t sum_matrix = 0.0;

            for (int k = 0; k < 16; ++k) {
                sum_matrix += element_matrix[k];
            }

            assert(sum_matrix < 1e-10);
#endif

            // Local to global
            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                idx_t dof_i = elems[edof_i][i];
                idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                idx_t *row = &colidx[rowptr[dof_i]];
                real_t *rowvalues = &values[rowptr[dof_i]];

                for (int edof_j = 0; edof_j < 4; ++edof_j) {
                    idx_t dof_j = elems[edof_j][i];
                    int k = binarysearch(dof_j, row, lenrow);

                    rowvalues[k] += element_matrix[edof_i * 4 + edof_j];
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Boundary conditions
    ///////////////////////////////////////////////////////////////////////////////

    real_t *rhs = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(rhs, 0, nnodes * sizeof(real_t));

    {  // Neumann
        sprintf(path, "%s/on.raw", folder);
        idx_t *faces_neumann = 0;
        ptrdiff_t nfacesx3 = read_file(comm, path, (void **)&faces_neumann);
        idx_t nfaces = (nfacesx3 / 3) / sizeof(idx_t);
        assert(nfaces * 3 * sizeof(idx_t) == nfacesx3);

        real_t u[3], v[3];
        real_t element_vector[3];

        real_t jacobian[3 * 3] = {0, 0, 0, 0, 0, 0, 0, 0, 1};

        real_t value = 1.0;
        for (idx_t f = 0; f < nfaces; ++f) {
            idx_t i0 = faces_neumann[f * 3];
            idx_t i1 = faces_neumann[f * 3 + 1];
            idx_t i2 = faces_neumann[f * 3 + 2];

            real_t dx = 0;

            if (0) {
                for (int d = 0; d < 3; ++d) {
                    real_t x0 = (real_t)xyz[d][i0];
                    real_t x1 = (real_t)xyz[d][i1];
                    real_t x2 = (real_t)xyz[d][i2];

                    u[d] = x1 - x0;
                    v[d] = x2 - x0;
                }

                dx = area3(u, v) / 2;
            } else {
                // No square roots in this version
                for (int d = 0; d < 3; ++d) {
                    real_t x0 = (real_t)xyz[d][i0];
                    real_t x1 = (real_t)xyz[d][i1];
                    real_t x2 = (real_t)xyz[d][i2];

                    jacobian[d * 3] = x1 - x0;
                    jacobian[d * 3 + 1] = x2 - x0;
                }

                // Orientation of face is not proper
                dx = fabs(det3(jacobian)) / 2;
            }

            assert(dx > 0.);
            integrate_neumann(value, dx, element_vector);

            rhs[i0] += element_vector[0];
            rhs[i1] += element_vector[1];
            rhs[i2] += element_vector[2];
        }

        free(faces_neumann);
    }

    if (!pure_neumann) {
        // Dirichlet
        sprintf(path, "%s/zd.raw", folder);
        idx_t *dirichlet_nodes = 0;
        ptrdiff_t nn = read_file(comm, path, (void **)&dirichlet_nodes);
        assert((nn / sizeof(idx_t)) * sizeof(idx_t) == nn);
        nn /= sizeof(idx_t);

        // Set rhs should not be necessary (but let us do it anyway)
        for (idx_t node = 0; node < nn; ++node) {
            idx_t i = dirichlet_nodes[node];
            rhs[i] = 0;
        }

        for (idx_t node = 0; node < nn; ++node) {
            idx_t i = dirichlet_nodes[node];

            idx_t begin = rowptr[i];
            idx_t end = rowptr[i + 1];
            idx_t lenrow = end - begin;
            idx_t *cols = &colidx[begin];
            real_t *row = &values[begin];

            memset(row, 0, sizeof(real_t) * lenrow);

            int k = binarysearch(i, cols, lenrow);
            assert(k >= 0);
            row[k] = 1;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Write CRS matrix and rhs vector
    ///////////////////////////////////////////////////////////////////////////////

    {
        crs_t crs_out;
        crs_out.rowptr = (char *)rowptr;
        crs_out.colidx = (char *)colidx;
        crs_out.values = (char *)values;
        crs_out.grows = nnodes;
        crs_out.lrows = nnodes;
        crs_out.lnnz = nnz;
        crs_out.gnnz = nnz;
        crs_out.start = 0;
        crs_out.rowoffset = 0;

        crs_write_folder(comm, output_folder, MPI_INT, MPI_INT, MPI_DOUBLE, &crs_out);
    }

    {
        sprintf(path, "%s/rhs.raw", output_folder);
        array_write(comm, path, MPI_DOUBLE, rhs, nnodes, nnodes);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(rowptr);
    free(colidx);
    free(values);
    free(rhs);

    for (int d = 0; d < 3; ++d) {
        free(xyz[d]);
    }

    for (int i = 0; i < 4; ++i) {
        free(elems[i]);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    return EXIT_SUCCESS;
}
