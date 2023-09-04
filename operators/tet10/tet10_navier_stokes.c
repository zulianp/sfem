#include "tet10_navier_stokes.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <mpi.h>

#include "sfem_vec.h"

static SFEM_INLINE void tet10_momentum_lhs_scalar_kernel(const real_t px0,
                                                         const real_t px1,
                                                         const real_t px2,
                                                         const real_t px3,
                                                         const real_t py0,
                                                         const real_t py1,
                                                         const real_t py2,
                                                         const real_t py3,
                                                         const real_t pz0,
                                                         const real_t pz1,
                                                         const real_t pz2,
                                                         const real_t pz3,
                                                         const real_t dt,
                                                         const real_t nu,
                                                         real_t *const SFEM_RESTRICT
                                                             element_matrix) {
	//TODO
	assert(0);
}

static SFEM_INLINE void tet10_momentum_rhs_kernel(const real_t px0,
                                                  const real_t px1,
                                                  const real_t px2,
                                                  const real_t px3,
                                                  const real_t py0,
                                                  const real_t py1,
                                                  const real_t py2,
                                                  const real_t py3,
                                                  const real_t pz0,
                                                  const real_t pz1,
                                                  const real_t pz2,
                                                  const real_t pz3,
                                                  const real_t dt,
                                                  const real_t nu,
                                                  real_t *const SFEM_RESTRICT u,
                                                  real_t *const SFEM_RESTRICT element_vector) {
	//TODO
	assert(0);
}

static SFEM_INLINE void tet4_tet10_divergence_rhs_kernel(const real_t px0,
                                                         const real_t px1,
                                                         const real_t px2,
                                                         const real_t px3,
                                                         const real_t py0,
                                                         const real_t py1,
                                                         const real_t py2,
                                                         const real_t py3,
                                                         const real_t pz0,
                                                         const real_t pz1,
                                                         const real_t pz2,
                                                         const real_t pz3,
                                                         const real_t dt,
                                                         const real_t rho,
                                                         const real_t *const SFEM_RESTRICT u,
                                                         real_t *const SFEM_RESTRICT
                                                             element_vector) {
	//TODO
	assert(0);
}

static SFEM_INLINE void tet10_tet4_rhs_correction_kernel(const real_t px0,
                                                         const real_t px1,
                                                         const real_t px2,
                                                         const real_t px3,
                                                         const real_t py0,
                                                         const real_t py1,
                                                         const real_t py2,
                                                         const real_t py3,
                                                         const real_t pz0,
                                                         const real_t pz1,
                                                         const real_t pz2,
                                                         const real_t pz3,
                                                         const real_t dt,
                                                         const real_t rho,
                                                         const real_t *const SFEM_RESTRICT p,
                                                         real_t *const SFEM_RESTRICT
                                                             element_vector) {
	//TODO
	assert(0);
}

static SFEM_INLINE void tet10_add_diffusion_rhs_kernel(const real_t px0,
                                                       const real_t px1,
                                                       const real_t px2,
                                                       const real_t px3,
                                                       const real_t py0,
                                                       const real_t py1,
                                                       const real_t py2,
                                                       const real_t py3,
                                                       const real_t pz0,
                                                       const real_t pz1,
                                                       const real_t pz2,
                                                       const real_t pz3,
                                                       const real_t dt,
                                                       const real_t nu,
                                                       real_t *const SFEM_RESTRICT u,
                                                       real_t *const SFEM_RESTRICT element_vector) {
}

static SFEM_INLINE void tet10_add_convection_rhs_kernel(const real_t px0,
                                                        const real_t px1,
                                                        const real_t px2,
                                                        const real_t px3,
                                                        const real_t py0,
                                                        const real_t py1,
                                                        const real_t py2,
                                                        const real_t py3,
                                                        const real_t pz0,
                                                        const real_t pz1,
                                                        const real_t pz2,
                                                        const real_t pz3,
                                                        const real_t dt,
                                                        const real_t nu,
                                                        real_t *const SFEM_RESTRICT u,
                                                        real_t *const SFEM_RESTRICT
                                                            element_vector) {
	//TODO
	assert(0);
}

// static SFEM_INLINE void tet10_explict_momentum_rhs_kernel(const real_t px0,
//                                                          const real_t px1,
//                                                          const real_t px2,
//                                                          const real_t py0,
//                                                          const real_t py1,
//                                                          const real_t py2,
//                                                          const real_t dt,
//                                                          const real_t nu,
//                                                          const real_t convonoff,
//                                                          real_t *const SFEM_RESTRICT u,
//                                                          real_t *const SFEM_RESTRICT
//                                                              element_vector) {}

// static SFEM_INLINE void tet10_add_momentum_rhs_kernel(const real_t px0,
//                                                      const real_t px1,
//                                                      const real_t px2,
//                                                      const real_t py0,
//                                                      const real_t py1,
//                                                      const real_t py2,
//                                                      real_t *const SFEM_RESTRICT u,
//                                                      real_t *const SFEM_RESTRICT element_vector)
//                                                      {}

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

static SFEM_INLINE int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return linear_search(key, row, lenrow);

        // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
        // while (key > row[++k]) {
        //     // Hi
        // }
        // assert(k < lenrow);
        // assert(key == row[k]);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void find_cols10(const idx_t *targets,
                                    const idx_t *const row,
                                    const int lenrow,
                                    int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 10; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(10)
        for (int d = 0; d < 10; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(10)
            for (int d = 0; d < 10; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

void tet4_tet10_divergence(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const elems,
                           geom_t **const points,
                           const real_t dt,
                           const real_t rho,
                           const real_t nu,
                           real_t **const SFEM_RESTRICT vel,
                           real_t *const SFEM_RESTRICT f) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 2;
    static const int element_nnodes = 6;

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            real_t element_vector[3];
            real_t element_vel[6 * 2];

#pragma unroll(6)
            for (int v = 0; v < element_nnodes; ++v) {
                ev[v] = elems[v][i];
            }

            for (int enode = 0; enode < element_nnodes; ++enode) {
                idx_t dof = ev[enode];

                for (int b = 0; b < n_vars; ++b) {
                    element_vel[b * element_nnodes + enode] = vel[b][dof];
                }
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[2];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];
            const real_t x3 = points[0][i3];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];
            const real_t y3 = points[1][i3];

            const real_t z0 = points[2][i0];
            const real_t z1 = points[2][i1];
            const real_t z2 = points[2][i2];
            const real_t z3 = points[2][i3];

            tet4_tet10_divergence_rhs_kernel(
                // X-coordinates
                x0,
                x1,
                x2,
                x3,
                // Y-coordinates
                y0,
                y1,
                y2,
                y3,
                // Z-coordinates
                z0,
                z1,
                z2,
                z3,
                dt,
                rho,
                //  buffers
                element_vel,
                element_vector);

            for (int edof_i = 0; edof_i < 3; ++edof_i) {
                const idx_t dof_i = ev[edof_i];
#pragma omp atomic update
                f[dof_i] += element_vector[edof_i];
            }
        }
    }

    double tock = MPI_Wtime();
    // printf("tet10_naviers_stokes.c: tet10_explict_momentum_tentative\t%g seconds\n", tock -
    // tick);
}

void tet10_tet4_correction(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const elems,
                           geom_t **const points,
                           const real_t dt,
                           const real_t rho,
                           real_t *const SFEM_RESTRICT p,
                           real_t **const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 2;
    static const int element_nnodes = 6;

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            real_t element_vector[6 * 2];
            real_t element_pressure[3];

#pragma unroll(6)
            for (int v = 0; v < element_nnodes; ++v) {
                ev[v] = elems[v][i];
            }

#pragma unroll(3)
            for (int enode = 0; enode < 3; ++enode) {
                idx_t dof = ev[enode];
                element_pressure[enode] = p[dof];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[2];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];
            const real_t x3 = points[0][i3];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];
            const real_t y3 = points[1][i3];

            const real_t z0 = points[2][i0];
            const real_t z1 = points[2][i1];
            const real_t z2 = points[2][i2];
            const real_t z3 = points[2][i3];

            tet10_tet4_rhs_correction_kernel(
                // X-coordinates
                x0,
                x1,
                x2,
                x3,
                // Y-coordinates
                y0,
                y1,
                y2,
                y3,
                // Z-coordinates
                z0,
                z1,
                z2,
                z3,
                dt,
                rho,
                //  buffers
                element_pressure,
                element_vector);

            for (int b = 0; b < n_vars; ++b) {
#pragma unroll(6)
                for (int edof_i = 0; edof_i < element_nnodes; ++edof_i) {
                    const idx_t dof_i = ev[edof_i];
#pragma omp atomic update
                    values[b][dof_i] += element_vector[b * element_nnodes + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    // printf("tet10_naviers_stokes.c: tet10_explict_momentum_tentative\t%g seconds\n", tock -
    // tick);
}

void tet10_momentum_lhs_scalar_crs(const ptrdiff_t nelements,
                                   const ptrdiff_t nnodes,
                                   idx_t **const elems,
                                   geom_t **const points,
                                   const real_t dt,
                                   const real_t nu,
                                   const count_t *const SFEM_RESTRICT rowptr,
                                   const idx_t *const SFEM_RESTRICT colidx,
                                   real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];
            idx_t ks[10];
            real_t element_matrix[10 * 10];

#pragma unroll(10)
            for (int v = 0; v < 10; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[2];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];
            const real_t x3 = points[0][i3];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];
            const real_t y3 = points[1][i3];

            const real_t z0 = points[2][i0];
            const real_t z1 = points[2][i1];
            const real_t z2 = points[2][i2];
            const real_t z3 = points[2][i3];

            tet10_momentum_lhs_scalar_kernel(
                // X-coordinates
                x0,
                x1,
                x2,
                x3,
                // Y-coordinates
                y0,
                y1,
                y2,
                y3,
                // Z-coordinates
                z0,
                z1,
                z2,
                z3,
                dt,
                nu,
                element_matrix);

            for (int edof_i = 0; edof_i < 10; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                const idx_t *row = &colidx[rowptr[dof_i]];

                find_cols10(ev, row, lenrow, ks);

                real_t *rowvalues = &values[rowptr[dof_i]];
                const real_t *element_row = &element_matrix[edof_i * 10];

#pragma unroll(10)
                for (int edof_j = 0; edof_j < 10; ++edof_j) {
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tet4_laplacian.c: tet4_laplacian_assemble_hessian\t%g seconds\n", tock - tick);
}

void tet10_explict_momentum_tentative(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const elems,
                                      geom_t **const points,
                                      const real_t dt,
                                      const real_t nu,
                                      const real_t convonoff,
                                      real_t **const SFEM_RESTRICT vel,
                                      real_t **const SFEM_RESTRICT f) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int element_nnodes = 10;

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];
            real_t element_vector[10 * 3];
            real_t element_vel[10 * 3];

#pragma unroll(10)
            for (int v = 0; v < element_nnodes; ++v) {
                ev[v] = elems[v][i];
            }

            for (int enode = 0; enode < element_nnodes; ++enode) {
                idx_t dof = ev[enode];

                for (int b = 0; b < n_vars; ++b) {
                    element_vel[b * element_nnodes + enode] = vel[b][dof];
                }
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[2];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];
            const real_t x3 = points[0][i3];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];
            const real_t y3 = points[1][i3];

            const real_t z0 = points[2][i0];
            const real_t z1 = points[2][i1];
            const real_t z2 = points[2][i2];
            const real_t z3 = points[2][i3];

            memset(element_vector, 0, 6 * 2 * sizeof(real_t));

            tet10_add_diffusion_rhs_kernel(x0,
                                           x1,
                                           x2,
                                           x3,
                                           // Y coords
                                           y0,
                                           y1,
                                           y2,
                                           y3,
                                           // Z coords
                                           z0,
                                           z1,
                                           z2,
                                           z3,
                                           dt,
                                           nu,
                                           //  buffers
                                           element_vel,
                                           element_vector);

            if (convonoff != 0) {
                tet10_add_convection_rhs_kernel(x0,
                                                x1,
                                                x2,
                                                x3,
                                                // Y coords
                                                y0,
                                                y1,
                                                y2,
                                                y3,
                                                // Z coords
                                                z0,
                                                z1,
                                                z2,
                                                z3,
                                                dt,
                                                nu,
                                                //  buffers
                                                element_vel,
                                                element_vector);
            }

            for (int d1 = 0; d1 < n_vars; d1++) {
                for (int edof_i = 0; edof_i < element_nnodes; ++edof_i) {
                    const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                    f[d1][dof_i] += element_vector[d1 * element_nnodes + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    // printf("tet10_naviers_stokes.c: tet10_explict_momentum_tentative\t%g seconds\n", tock -
    // tick);
}