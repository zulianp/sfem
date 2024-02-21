#include "macro_tet4_laplacian.h"
#include "sfem_vec.h"

#include <mpi.h>
#include <stdio.h>

#define VEC_SIZE SFEM_VECTOR_SIZE
typedef real_t scalar_t;
typedef vreal_t vec_t;

// typedef float scalar_t;
// typedef float8_t vec_t;
// #define VEC_SIZE 8

// typedef double scalar_t;
// typedef double4_t vec_t;
// #define VEC_SIZE 4

#define POW2(a) ((a) * (a))
#define MIN(a, b) ((a) < (b) ? a : b)



static void fff_micro_kernel(const vec_t px0,
                             const vec_t px1,
                             const vec_t px2,
                             const vec_t px3,
                             const vec_t py0,
                             const vec_t py1,
                             const vec_t py2,
                             const vec_t py3,
                             const vec_t pz0,
                             const vec_t pz1,
                             const vec_t pz2,
                             const vec_t pz3,
                             vec_t *fff) {
    const vec_t x0 = -px0 + px1;
    const vec_t x1 = -py0 + py2;
    const vec_t x2 = -pz0 + pz3;
    const vec_t x3 = x1 * x2;
    const vec_t x4 = -pz0 + pz1;
    const vec_t x5 = -px0 + px2;
    const vec_t x6 = -py0 + py3;
    const vec_t x7 = x5 * x6;
    const vec_t x8 = -py0 + py1;
    const vec_t x9 = -px0 + px3;
    const vec_t x10 = -pz0 + pz2;
    const vec_t x11 = x10 * x6;
    const vec_t x12 = x2 * x5;
    const vec_t x13 = x1 * x9;
    const vec_t x14 = -x0 * x11 + x0 * x3 + x10 * x8 * x9 - x12 * x8 - x13 * x4 + x4 * x7;
    const vec_t x15 = -x13 + x7;
    const vec_t x16 = 1. / POW2(x14);
    const vec_t x17 = x10 * x9 - x12;
    const vec_t x18 = -x11 + x3;
    const vec_t x19 = -x0 * x6 + x8 * x9;
    const vec_t x20 = x15 * x16;
    const vec_t x21 = x0 * x2 - x4 * x9;
    const vec_t x22 = x16 * x17;
    const vec_t x23 = -x2 * x8 + x4 * x6;
    const vec_t x24 = x16 * x18;
    const vec_t x25 = x0 * x1 - x5 * x8;
    const vec_t x26 = -x0 * x10 + x4 * x5;
    const vec_t x27 = -x1 * x4 + x10 * x8;
    fff[0] = x14 * (POW2(x15) * x16 + x16 * POW2(x17) + x16 * POW2(x18));
    fff[1] = x14 * (x19 * x20 + x21 * x22 + x23 * x24);
    fff[2] = x14 * (x20 * x25 + x22 * x26 + x24 * x27);
    fff[3] = x14 * (x16 * POW2(x19) + x16 * POW2(x21) + x16 * POW2(x23));
    fff[4] = x14 * (x16 * x19 * x25 + x16 * x21 * x26 + x16 * x23 * x27);
    fff[5] = x14 * (x16 * POW2(x25) + x16 * POW2(x26) + x16 * POW2(x27));
}

static SFEM_INLINE void sub_fff_0(const vec_t *const SFEM_RESTRICT fff,
                                  vec_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = (scalar_t)(1.0 / 2.0) * fff[0];
    sub_fff[1] = (scalar_t)(1.0 / 2.0) * fff[1];
    sub_fff[2] = (scalar_t)(1.0 / 2.0) * fff[2];
    sub_fff[3] = (scalar_t)(1.0 / 2.0) * fff[3];
    sub_fff[4] = (scalar_t)(1.0 / 2.0) * fff[4];
    sub_fff[5] = (scalar_t)(1.0 / 2.0) * fff[5];
}

static SFEM_INLINE void sub_fff_4(const vec_t *const SFEM_RESTRICT fff,
                                  vec_t *const SFEM_RESTRICT sub_fff) {
    const vec_t x0 = (scalar_t)(1.0 / 2.0) * fff[0];
    const vec_t x1 = (scalar_t)(1.0 / 2.0) * fff[2];
    sub_fff[0] = fff[1] + (scalar_t)(1.0 / 2.0) * fff[3] + x0;
    sub_fff[1] = (scalar_t)(-1.0 / 2.0) * fff[1] - x0;
    sub_fff[2] = (scalar_t)(1.0 / 2.0) * fff[4] + x1;
    sub_fff[3] = x0;
    sub_fff[4] = -x1;
    sub_fff[5] = (scalar_t)(1.0 / 2.0) * fff[5];
}

static SFEM_INLINE void sub_fff_5(const vec_t *const SFEM_RESTRICT fff,
                                  vec_t *const SFEM_RESTRICT sub_fff) {
    const vec_t x0 = (scalar_t)(1.0 / 2.0) * fff[3];
    const vec_t x1 = fff[4] + (scalar_t)(1.0 / 2.0) * fff[5] + x0;
    const vec_t x2 = (scalar_t)(1.0 / 2.0) * fff[4] + x0;
    const vec_t x3 = (scalar_t)(1.0 / 2.0) * fff[1];
    sub_fff[0] = x1;
    sub_fff[1] = -x2;
    sub_fff[2] = (scalar_t)(-1.0 / 2.0) * fff[2] - x1 - x3;
    sub_fff[3] = x0;
    sub_fff[4] = x2 + x3;
    sub_fff[5] = (scalar_t)(1.0 / 2.0) * fff[0] + fff[1] + fff[2] + x1;
}

static SFEM_INLINE void sub_fff_6(const vec_t *const SFEM_RESTRICT fff,
                                  vec_t *const SFEM_RESTRICT sub_fff) {
    const vec_t x0 = (scalar_t)(1.0 / 2.0) * fff[3];
    const vec_t x1 = (scalar_t)(1.0 / 2.0) * fff[4];
    const vec_t x2 = (scalar_t)(1.0 / 2.0) * fff[1] + x0;
    sub_fff[0] = (scalar_t)(1.0 / 2.0) * fff[0] + fff[1] + x0;
    sub_fff[1] = (scalar_t)(1.0 / 2.0) * fff[2] + x1 + x2;
    sub_fff[2] = -x2;
    sub_fff[3] = fff[4] + (scalar_t)(1.0 / 2.0) * fff[5] + x0;
    sub_fff[4] = -x0 - x1;
    sub_fff[5] = x0;
}

static SFEM_INLINE void sub_fff_7(const vec_t *const SFEM_RESTRICT fff,
                                  vec_t *const SFEM_RESTRICT sub_fff) {
    const vec_t x0 = (scalar_t)(1.0 / 2.0) * fff[5];
    const vec_t x1 = (scalar_t)(1.0 / 2.0) * fff[2];
    sub_fff[0] = x0;
    sub_fff[1] = (scalar_t)(-1.0 / 2.0) * fff[4] - x0;
    sub_fff[2] = -x1;
    sub_fff[3] = (scalar_t)(1.0 / 2.0) * fff[3] + fff[4] + x0;
    sub_fff[4] = (scalar_t)(1.0 / 2.0) * fff[1] + x1;
    sub_fff[5] = (scalar_t)(1.0 / 2.0) * fff[0];
}

static void lapl_apply_micro_kernel(const vec_t *const SFEM_RESTRICT fff,
                                    const vec_t u0,
                                    const vec_t u1,
                                    const vec_t u2,
                                    const vec_t u3,
                                    vec_t *const SFEM_RESTRICT e0,
                                    vec_t *const SFEM_RESTRICT e1,
                                    vec_t *const SFEM_RESTRICT e2,
                                    vec_t *const SFEM_RESTRICT e3) {
    const vec_t x0 = (scalar_t)(1.0 / 6.0) * u0;
    const vec_t x1 = fff[0] * x0;
    const vec_t x2 = (scalar_t)(1.0 / 6.0) * u1;
    const vec_t x3 = fff[0] * x2;
    const vec_t x4 = fff[1] * x2;
    const vec_t x5 = (scalar_t)(1.0 / 6.0) * u2;
    const vec_t x6 = fff[1] * x5;
    const vec_t x7 = fff[2] * x2;
    const vec_t x8 = (scalar_t)(1.0 / 6.0) * u3;
    const vec_t x9 = fff[2] * x8;
    const vec_t x10 = fff[3] * x0;
    const vec_t x11 = fff[3] * x5;
    const vec_t x12 = fff[4] * x5;
    const vec_t x13 = fff[4] * x8;
    const vec_t x14 = fff[5] * x0;
    const vec_t x15 = fff[5] * x8;
    const vec_t x16 = fff[1] * x0;
    const vec_t x17 = fff[2] * x0;
    const vec_t x18 = fff[4] * x0;
    *e0 += (scalar_t)(1.0 / 3.0) * fff[1] * u0 + (scalar_t)(1.0 / 3.0) * fff[2] * u0 + (scalar_t)(1.0 / 3.0) * fff[4] * u0 + x1 +
           x10 - x11 - x12 - x13 + x14 - x15 - x3 - x4 - x6 - x7 - x9;
    *e1 += -x1 - x16 - x17 + x3 + x6 + x9;
    *e2 += -x10 + x11 + x13 - x16 - x18 + x4;
    *e3 += x12 - x14 + x15 - x17 - x18 + x7;
}

void macro_tet4_laplacian_apply(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT values) {
    // double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; i += VEC_SIZE) {
            const int nvec = MIN(nelements - (i + VEC_SIZE), VEC_SIZE);

            idx_t ev[10];
            vec_t x[4], y[4], z[4];
            vec_t fff[6];
            vec_t sub_fff[6];
            vec_t element_u[10];
            vec_t element_vector[10] = {0};

            for (int vi = 0; vi < nvec; ++vi) {
                const ptrdiff_t offset = i + vi;
                #pragma omp unroll full
                for (int d = 0; d < 4; ++d) {
                    const idx_t vidx = elems[d][offset];
                    x[d][vi] = xyz[0][vidx];
                    y[d][vi] = xyz[1][vidx];
                    z[d][vi] = xyz[2][vidx];
                }
                
                for (int d = 0; d < 10; ++d) {
                    const idx_t vidx = elems[d][offset];
                    element_u[d][vi] = u[vidx];
                }
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            fff_micro_kernel(
                // X
                x[0],
                x[1],
                x[2],
                x[3],
                // Y
                y[0],
                y[1],
                y[2],
                y[3],
                // Z
                z[0],
                z[1],
                z[2],
                z[3],
                //
                fff);

            {  // Corner tests
                sub_fff_0(fff, sub_fff);

                // [0, 4, 6, 7],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[0],
                                        element_u[4],
                                        element_u[6],
                                        element_u[7],
                                        &element_vector[0],
                                        &element_vector[4],
                                        &element_vector[6],
                                        &element_vector[7]);

                // [4, 1, 5, 8],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[4],
                                        element_u[1],
                                        element_u[5],
                                        element_u[8],
                                        &element_vector[4],
                                        &element_vector[1],
                                        &element_vector[5],
                                        &element_vector[8]);

                // [6, 5, 2, 9],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[6],
                                        element_u[5],
                                        element_u[2],
                                        element_u[9],
                                        &element_vector[6],
                                        &element_vector[5],
                                        &element_vector[2],
                                        &element_vector[9]);

                // [7, 8, 9, 3],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[7],
                                        element_u[8],
                                        element_u[9],
                                        element_u[3],
                                        &element_vector[7],
                                        &element_vector[8],
                                        &element_vector[9],
                                        &element_vector[3]);
            }

            {  // Octahedron tets

                // [4, 5, 6, 8],
                sub_fff_4(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[4],
                                        element_u[5],
                                        element_u[6],
                                        element_u[8],
                                        &element_vector[4],
                                        &element_vector[5],
                                        &element_vector[6],
                                        &element_vector[8]);

                // [7, 4, 6, 8],
                sub_fff_5(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[7],
                                        element_u[4],
                                        element_u[6],
                                        element_u[8],
                                        &element_vector[7],
                                        &element_vector[4],
                                        &element_vector[6],
                                        &element_vector[8]);

                // [6, 5, 9, 8],
                sub_fff_6(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[6],
                                        element_u[5],
                                        element_u[9],
                                        element_u[8],
                                        &element_vector[6],
                                        &element_vector[5],
                                        &element_vector[9],
                                        &element_vector[8]);

                // [7, 6, 9, 8]]
                sub_fff_7(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[7],
                                        element_u[6],
                                        element_u[9],
                                        element_u[8],
                                        &element_vector[7],
                                        &element_vector[6],
                                        &element_vector[9],
                                        &element_vector[8]);
            }

            for (int vi = 0; vi < nvec; ++vi) {
                const idx_t offset = i + vi;
                for (int v = 0; v < 10; ++v) {
                    const idx_t dof_i = elems[v][offset];
#pragma omp atomic update
                    values[dof_i] += element_vector[v][vi];
                }
            }
        }
    }

    // double tock = MPI_Wtime();
    // printf("macro_tet4_laplacian_apply[simd] %g (seconds)\n", tock - tick);
}
