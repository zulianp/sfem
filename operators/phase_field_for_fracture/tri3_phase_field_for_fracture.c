#include "tri3_phase_field_for_fracture.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#define SFEM_CUDA_INLINE SFEM_INLINE
#include "FE2D_phase_field_for_fracture_kernels.h"
#include "Tri3_kernels.cu"

#define var_disp 1
#define var_phase 0
#define sdim 2
#define tdim 4
#define block_size 3
#define n_funs 3
#define n_qp 6

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - 4; i += 4) {
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

static SFEM_INLINE void find_cols_x(const idx_t *targets,
                                    const idx_t *const row,
                                    const int lenrow,
                                    int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < n_funs; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(n_funs)
        for (int d = 0; d < n_funs; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(n_funs)
            for (int d = 0; d < n_funs; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

static const real_t qx[n_qp] = {0.5, 0.5, 0.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};
static const real_t qy[n_qp] = {0.5, 0.0, 0.5, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
static const real_t qw[n_qp] = {1.0 / 30.0, 1.0 / 30.0, 1.0 / 30.0, 0.3, 0.3, 0.3};
static const real_t ref_vol = 1. / 2;

SFEM_INLINE static void element_init(const geom_t x0,
                                     const geom_t x1,
                                     const geom_t x2,
                                     const geom_t y0,
                                     const geom_t y1,
                                     const geom_t y2,
                                     real_t *jacobian_determinant,
                                     real_t *jacobian_inverse,
                                     real_t *fun,
                                     real_t *grad,
                                     const real_t *element_solution,
                                     real_t *phase,
                                     real_t *grad_phase,
                                     real_t *grad_disp) {
    Tri3_mk_jacobian_determinant_and_inverse(
        x0, x1, x2, y0, y1, y2, 1, jacobian_determinant, 1, jacobian_inverse);
    // Compute physical gradients
    for (int k = 0; k < n_qp; k++) {
        const int stride = k * n_funs * sdim;
        Tri3_mk_partial_x(qx[k], qy[k], 1, jacobian_inverse, sdim, &grad[stride]);
        Tri3_mk_partial_y(qx[k], qy[k], 1, jacobian_inverse, sdim, &grad[stride + 1]);
    }

    // Compute phase-field
    for (int k = 0; k < n_qp; k++) {
        real_t val = 0;
        for (int ii = 0; ii < n_funs; ii++) {
            const real_t c = element_solution[ii * block_size + var_phase];
            // assert(fabs(c) < 1e-6);
            assert(c == c);
            assert(fun[k * n_funs + ii] ==  fun[k * n_funs + ii]);

            val += fun[k * n_funs + ii] * c;
        }

        phase[k] = val;
    }

    // Compute phase-field gradient
    for (int k = 0; k < n_qp; k++) {
        real_t grad2[sdim] = {0, 0};

        for (int ii = 0; ii < n_funs; ii++) {
            const real_t c = element_solution[ii * block_size + var_phase];
            // assert(fabs(c) < 1e-6);
            assert(c == c);

            for (int d = 0; d < sdim; d++) {
                assert(grad[k * n_funs * sdim + ii * sdim + d]  == grad[k * n_funs * sdim + ii * sdim + d] );
                grad2[d] += grad[k * n_funs * sdim + ii * sdim + d] * c;
            }
        }

        for (int d = 0; d < sdim; d++) {
            grad_phase[k * sdim + d] = grad2[d];
        }
    }

    // Compute displacement-gradient
    for (int k = 0; k < n_qp; k++) {
        real_t grad2[tdim] = {0, 0, 0, 0};

        for (int ii = 0; ii < n_funs; ii++) {
            for (int d1 = 0; d1 < sdim; d1++) {
                const real_t c = element_solution[ii * block_size + var_disp + d1];
                assert(c == c);

                for (int d2 = 0; d2 < sdim; d2++) {
                    grad2[d1 * sdim + d2] += grad[k * n_funs * sdim + ii * sdim + d2] * c;
                }
            }
        }

        for (int d1 = 0; d1 < sdim; d1++) {
            for (int d2 = 0; d2 < sdim; d2++) {
                grad_disp[k * sdim * sdim + d1 * sdim + d2] = grad2[d1 * sdim + d2];
            }
        }
    }
}

void tri3_phase_field_for_fracture_assemble_hessian_aos(const ptrdiff_t nelements,
                                                        const ptrdiff_t nnodes,
                                                        idx_t **const SFEM_RESTRICT elems,
                                                        geom_t **const SFEM_RESTRICT xyz,
                                                        const real_t mu,
                                                        const real_t lambda,
                                                        const real_t Gc,
                                                        const real_t ls,
                                                        const real_t *const SFEM_RESTRICT solution,
                                                        const count_t *const SFEM_RESTRICT rowptr,
                                                        const idx_t *const SFEM_RESTRICT colidx,
                                                        real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[n_funs];
    idx_t ks[n_funs];

    real_t element_matrix[block_size * block_size];
    real_t element_solution[n_funs * block_size];

    real_t fun[n_qp * sdim];
    real_t grad[n_qp * n_funs * sdim];

    real_t jacobian_inverse[sdim * sdim];
    real_t jacobian_determinant;

    real_t phase[n_qp];
    real_t grad_phase[n_qp * sdim];
    real_t grad_disp[n_qp * sdim * sdim];

    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];

    for (int k = 0; k < n_qp; k++) {
        Tri3_mk_fun(qx[k], qy[k], 1, &fun[k * n_funs]);
    }

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(n_funs)
        for (int v = 0; v < n_funs; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        for (int enode = 0; enode < n_funs; ++enode) {
            idx_t edof = enode * block_size;
            idx_t dof = ev[enode] * block_size;

            for (int b = 0; b < block_size; ++b) {
                element_solution[edof + b] = solution[dof + b];
                assert(element_solution[edof + b] == element_solution[edof + b]);
            }
        }

        element_init(x[i0],
                     x[i1],
                     x[i2],
                     y[i0],
                     y[i1],
                     y[i2],
                     &jacobian_determinant,
                     jacobian_inverse,
                     fun,
                     grad,
                     element_solution,
                     phase,
                     grad_phase,
                     grad_disp);

        for (int edof_i = 0; edof_i < n_funs; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];
            const idx_t *row = &colidx[rowptr[dof_i]];
            find_cols_x(ev, row, lenrow, ks);

            // Blocks for row
            real_t *block_start = &values[rowptr[dof_i] * block_size * block_size];

            for (int edof_j = 0; edof_j < n_funs; ++edof_j) {
                memset(element_matrix, 0, block_size * block_size * sizeof(real_t));

                for (int k = 0; k < n_qp; k++) {
                    FE2D_phase_field_for_fracture_hessian(mu,
                                                          lambda,
                                                          Gc,
                                                          ls,
                                                          ref_vol,
                                                          jacobian_determinant,
                                                          fun[k * n_funs + edof_i],
                                                          &grad[k * n_funs * sdim + edof_i * sdim],
                                                          fun[k * n_funs + edof_j],
                                                          &grad[k * n_funs * sdim + edof_j * sdim],
                                                          phase[k],
                                                          &grad_phase[k * sdim],
                                                          &grad_disp[k * sdim * sdim],
                                                          element_matrix);
                }

                if (0) {
                    printf("(%d, %d)\n", dof_i, ks[edof_j]);

                    for (int d1 = 0; d1 < block_size; d1++) {
                        for (int d2 = 0; d2 < block_size; d2++) {
                            printf("%g ", element_matrix[d1 * block_size + d2]);
                        }
                        printf("\n");
                    }
                }

                const idx_t offset_j = ks[edof_j] * block_size;

                for (int bi = 0; bi < block_size; ++bi) {
                    // Jump rows (including the block-size for the columns)
                    real_t *row = &block_start[bi * lenrow * block_size];

                    for (int bj = 0; bj < block_size; ++bj) {
                        const real_t val = element_matrix[bi * block_size + bj];
                        assert(val == val);
                        row[offset_j + bj] += val;
                    }
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf(
        "tri3_phase_field_for_fracture.c: phase_field_for_fracture_assemble_hessian\t%g seconds\n",
        tock - tick);
}

void tri3_phase_field_for_fracture_assemble_gradient_aos(const ptrdiff_t nelements,
                                                         const ptrdiff_t nnodes,
                                                         idx_t **const SFEM_RESTRICT elems,
                                                         geom_t **const SFEM_RESTRICT xyz,
                                                         const real_t mu,
                                                         const real_t lambda,
                                                         const real_t Gc,
                                                         const real_t ls,
                                                         const real_t *const SFEM_RESTRICT solution,
                                                         real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[n_funs];
    idx_t ks[n_funs];

    real_t element_vector[block_size];
    real_t element_solution[n_funs * block_size];

    real_t fun[n_qp * n_funs];
    real_t grad[n_qp * n_funs * sdim];

    real_t jacobian_inverse[sdim * sdim];
    real_t jacobian_determinant;

    real_t phase[n_qp];
    real_t grad_phase[n_qp * sdim];
    real_t grad_disp[n_qp * sdim * sdim];

    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];

    for (int k = 0; k < n_qp; k++) {
        Tri3_mk_fun(qx[k], qy[k], 1, &fun[k * n_funs]);
    }

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(n_funs)
        for (int v = 0; v < n_funs; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        for (int enode = 0; enode < n_funs; ++enode) {
            idx_t edof = enode * block_size;
            idx_t dof = ev[enode] * block_size;

            for (int b = 0; b < block_size; ++b) {
                element_solution[edof + b] = solution[dof + b];
                assert(element_solution[edof + b] == element_solution[edof + b]);
            }
        }

        element_init(x[i0],
                     x[i1],
                     x[i2],
                     y[i0],
                     y[i1],
                     y[i2],
                     &jacobian_determinant,
                     jacobian_inverse,
                     fun,
                     grad,
                     element_solution,
                     phase,
                     grad_phase,
                     grad_disp);

        for (int edof_i = 0; edof_i < n_funs; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];
            memset(element_vector, 0, block_size * sizeof(real_t));

            for (int k = 0; k < n_qp; k++) {
                FE2D_phase_field_for_fracture_gradient(mu,
                                                       lambda,
                                                       Gc,
                                                       ls,
                                                       ref_vol,
                                                       jacobian_determinant,
                                                       fun[k * n_funs + edof_i],
                                                       &grad[k * n_funs * sdim + edof_i * sdim],
                                                       phase[k],
                                                       &grad_phase[k * sdim],
                                                       &grad_disp[k * sdim * sdim],
                                                       element_vector);
            }

            if (0) {
                printf("%d)\n", dof_i);
                for (int d1 = 0; d1 < block_size; d1++) {
                    printf("%g\n", element_vector[d1]);
                }
            }

            for (int b = 0; b < block_size; b++) {
                assert(element_vector[b] == element_vector[b]);
                values[dof_i * block_size + b] += element_vector[b];
            }
        }
    }

    double tock = MPI_Wtime();
    printf(
        "tri3_phase_field_for_fracture.c: phase_field_for_fracture_assemble_gradient\t%g seconds\n",
        tock - tick);
}

void tri3_phase_field_for_fracture_assemble_value_aos(const ptrdiff_t nelements,
                                                      const ptrdiff_t nnodes,
                                                      idx_t **const SFEM_RESTRICT elems,
                                                      geom_t **const SFEM_RESTRICT xyz,
                                                      const real_t mu,
                                                      const real_t lambda,
                                                      const real_t Gc,
                                                      const real_t ls,
                                                      const real_t *const SFEM_RESTRICT solution,
                                                      real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[n_funs];
    idx_t ks[n_funs];

    real_t element_vector[block_size];
    real_t element_solution[n_funs * block_size];

    real_t fun[n_qp * n_funs];
    real_t grad[n_qp * n_funs * sdim];

    real_t jacobian_inverse[sdim * sdim];
    real_t jacobian_determinant;

    real_t phase[n_qp];
    real_t grad_phase[n_qp * sdim];
    real_t grad_disp[n_qp * sdim * sdim];

    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];

    for (int k = 0; k < n_qp; k++) {
        Tri3_mk_fun(qx[k], qy[k], 1, &fun[k * n_funs]);
    }

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(n_funs)
        for (int v = 0; v < n_funs; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        for (int enode = 0; enode < n_funs; ++enode) {
            idx_t edof = enode * block_size;
            idx_t dof = ev[enode] * block_size;

            for (int b = 0; b < block_size; ++b) {
                element_solution[edof + b] = solution[dof + b];
                assert(element_solution[edof + b] == element_solution[edof + b]);
            }
        }

        element_init(x[i0],
                     x[i1],
                     x[i2],
                     y[i0],
                     y[i1],
                     y[i2],
                     &jacobian_determinant,
                     jacobian_inverse,
                     fun,
                     grad,
                     element_solution,
                     phase,
                     grad_phase,
                     grad_disp);

        real_t element_scalar = 0;

        for (int k = 0; k < n_qp; k++) {
            FE2D_phase_field_for_fracture_value(mu,
                                                lambda,
                                                Gc,
                                                ls,
                                                ref_vol,
                                                jacobian_determinant,
                                                phase[k],
                                                &grad_phase[k * sdim],
                                                &grad_disp[k * sdim * sdim],
                                                &element_scalar);
        }

        assert(element_scalar == element_scalar);
        values[0] += element_scalar;
    }

    double tock = MPI_Wtime();
    printf("tri3_phase_field_for_fracture.c: phase_field_for_fracture_assemble_value\t%g seconds\n",
           tock - tick);
}

// Clean-up
#undef var_disp
#undef var_phase
#undef sdim
#undef tdim
#undef block_size
#undef n_funs
#undef n_qp
